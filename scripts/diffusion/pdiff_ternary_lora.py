#!/usr/bin/env python3
"""pdiff_ternary_lora.py — Ternary LoRA (NOT merged) for BitNet-b1.58 base.

MECHANISM: LoTA-QAF-style (arXiv 2505.18724)
    For each target AutoBitLinear MLP layer L:
        y = L(x) + (alpha/r) * WeightQuant_ste(B) @ (WeightQuant_ste(A) @ x)

    - L(x): base AutoBitLinear forward (FROZEN, UNTOUCHED)
    - A [r, in_features], B [out_features, r]: bf16 masters, TRAINABLE
    - WeightQuant_ste: BitNet absmean ternary with STE (gradients flow to bf16 masters)
    - Init A,B from gen_adapter lora_A/lora_B (warm start); cold start uses zeros/kaiming
    - Scale = alpha / r

WHY NOT MERGE:
    Merging LoRA into a ternary base weight (W_base + LoRA_delta) then re-ternarizing
    is catastrophic (PPL 55.6 vs base 11.8) because the delta is too small to survive
    projection onto the ternary grid.  Keeping the base FROZEN ensures the worst case
    is exactly the base — the ternary low-rank term simply adds 0 when A=0 or B=0.

LAMBDA-RAMP (--lambda-ramp-steps N):
    lam=0:   pure bf16 LoRA (A,B as-is floats, standard LoRA quality ~12.4 PPL)
    lam=1:   full ternary LoRA (WeightQuant_ste on A and B)
    blended: A_eff = (1-lam)*A + lam*ternarize_ste(A), same for B
    Schedule: lam = min(step / N, 1.0)

EVAL:
    PPL with ternary LoRA (lam=1 inference) and base-only PPL (LoRA disabled),
    both computed in the same run on the same eval split for direct comparison.

CLI:
    # CPU validation (3 correctness checks, no model download, <1s):
    python scripts/diffusion/pdiff_ternary_lora.py --validate-cpu

    # Full train + eval (GPU) — gen_adapter warm start:
    python scripts/diffusion/pdiff_ternary_lora.py \\
        --adapter .docs/diffusion/pdiff_boot_fixedinit/pdiff_func_out38/gen_adapter \\
        --last-n-layers 12 --steps 1500 --lambda-ramp-steps 500 \\
        --optim adafactor --device cuda \\
        --out E:/tlora_out/gen

    # Cold start (no adapter warm init):
    python scripts/diffusion/pdiff_ternary_lora.py \\
        --adapter none --r 16 --alpha 32 \\
        --last-n-layers 12 --steps 1500 --lambda-ramp-steps 500 \\
        --optim adafactor --device cuda \\
        --out E:/tlora_out/cold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure UTF-8 on Windows (avoids cp1252 errors from Greek letters, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent

BASE_MODEL_ID  = "microsoft/bitnet-b1.58-2B-4T-bf16"
EVAL_DATASET   = "HuggingFaceH4/no_robots"
MLP_PROJ_NAMES = frozenset({"gate_proj", "up_proj", "down_proj"})


# ---------------------------------------------------------------------------
# Global lambda-ramp, LoRA-enabled, and inference-cache state
# ---------------------------------------------------------------------------

_lora_lambda:  list[float] = [1.0]   # current ternarization strength; 0=pure bf16, 1=full ternary
_lora_enabled: list[bool]  = [True]  # False -> wrapper returns base(x) only (for base PPL eval)
_infer_cache:  list[bool]  = [False] # True -> cache frozen ternary weights; reuse across tokens


def _set_lora_lambda(val: float) -> None:
    """Set the ternary ramp lambda (clamped to [0, 1])."""
    _lora_lambda[0] = float(max(0.0, min(1.0, val)))


def _set_lora_enabled(val: bool) -> None:
    """Enable or disable the LoRA term globally (False = base-only output)."""
    _lora_enabled[0] = bool(val)


# ---------------------------------------------------------------------------
# TernaryLoRAWrapper
# ---------------------------------------------------------------------------

class TernaryLoRAWrapper(nn.Module):
    """Wraps a frozen AutoBitLinear with a trainable ternary low-rank additive term.

    y = base(x) + scale * WeightQuant_ste(B) @ (WeightQuant_ste(A) @ x)
    where scale = alpha / r.

    Ramp schedule (via global _lora_lambda):
        lam=0: pure bf16 LoRA — A,B used as plain floats
        lam=1: full ternary  — WeightQuant_ste applied to A and B
        0<lam<1: blended     — A_eff = (1-lam)*A + lam*WeightQuant_ste(A), same B

    Global _lora_enabled:
        False -> forward returns base(x) only (no LoRA term); used for base PPL eval.
    """

    def __init__(
        self,
        base: nn.Module,
        r: int,
        alpha: float,
        in_features: int,
        out_features: int,
        A_init: Optional[torch.Tensor] = None,
        B_init: Optional[torch.Tensor] = None,
        group_size: int = 0,
    ) -> None:
        super().__init__()
        self.base        = base
        self.r           = r
        self.alpha       = float(alpha)
        self.scale       = self.alpha / r
        self.in_features  = in_features
        self.out_features = out_features
        # 0 = per-tensor absmean (default). >0 = group-wise absmean along the last
        # dim in blocks of group_size (LoTA-QAF-style finer quantization).
        self.group_size  = int(group_size)

        # Trainable bf16 master weights for the ternary LoRA term.
        self.lora_A = nn.Parameter(torch.zeros(r, in_features,   dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r,  dtype=torch.bfloat16))

        if A_init is not None:
            with torch.no_grad():
                self.lora_A.copy_(A_init.to(torch.bfloat16))
        if B_init is not None:
            with torch.no_grad():
                self.lora_B.copy_(B_init.to(torch.bfloat16))

        # Freeze the base (base ternary structure never touched).
        for p in self.base.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _lora_term(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scale * B_q @ A_q @ x with STE on A and B per current lam."""
        from transformers.integrations.bitnet import WeightQuant

        lam = _lora_lambda[0]

        # Quantizer: group-wise STE when group_size>0, else BitNet per-tensor WeightQuant.
        def q(W):
            if self.group_size > 0:
                return _group_ternarize_ste(W, self.group_size)
            return WeightQuant.apply(W)

        if lam <= 0.0:
            # Pure bf16 path: A,B as-is (standard LoRA math, no ternarization).
            return self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)

        if lam >= 1.0:
            # Full ternary path: STE on both A and B.
            # Inference cache: compute q(A) and q(B) once; reuse across all token steps.
            if _infer_cache[0]:
                if not hasattr(self, '_cached_qA') or self._cached_qA is None:
                    with torch.no_grad():
                        self._cached_qA = q(self.lora_A).detach()
                        self._cached_qB = q(self.lora_B).detach()
                A_q = self._cached_qA
                B_q = self._cached_qB
            else:
                A_q = q(self.lora_A)
                B_q = q(self.lora_B)
            return self.scale * F.linear(F.linear(x, A_q), B_q)

        # Blended (0 < lam < 1): smooth warm-up for QAT convergence.
        # Gradient to lora_A: (1-lam)*grad_A + lam*STE*grad_A = grad_A  (STE: dQ/dW=1)
        A_q   = q(self.lora_A)
        B_q   = q(self.lora_B)
        A_eff = (1.0 - lam) * self.lora_A + lam * A_q
        B_eff = (1.0 - lam) * self.lora_B + lam * B_q
        return self.scale * F.linear(F.linear(x, A_eff), B_eff)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not _lora_enabled[0]:
            return base_out
        return base_out + self._lora_term(x)


# ---------------------------------------------------------------------------
# absmean ternary helper (for validation, no STE needed)
# ---------------------------------------------------------------------------

def _absmean_ternarize(W: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean ternary quantisation (matches BitNet WeightQuant)."""
    gamma = W.abs().mean().clamp(min=1e-5)
    return torch.round(W / gamma).clamp(-1.0, 1.0) * gamma


def _group_ternarize_ste(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """Group-wise absmean ternary quantisation with a straight-through estimator.

    W is [out, in]; grouping is along the last dim (in) in contiguous blocks of
    group_size, each block getting its own absmean scale (finer than per-tensor).
    Falls back to per-tensor when group_size<=0 or >= in, or when in is not
    divisible by group_size (keeps the fast even path; remainder handled by a
    trailing partial group). STE: forward is dequant(round(W/gamma)), gradient
    passes straight through to W (dQ/dW = 1).
    """
    in_ = W.shape[-1]
    if group_size <= 0 or group_size >= in_:
        gamma = W.abs().mean().clamp(min=1e-5)
        deq = torch.round(W / gamma).clamp(-1.0, 1.0) * gamma
        return (deq - W).detach() + W

    out = W.shape[0]
    ng_full = in_ // group_size
    used = ng_full * group_size
    parts = []
    head = W[:, :used].reshape(out, ng_full, group_size)
    gamma = head.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)          # [out, ng, 1]
    deq_head = (torch.round(head / gamma).clamp(-1.0, 1.0) * gamma).reshape(out, used)
    parts.append(deq_head)
    if used < in_:  # trailing partial group
        tail = W[:, used:]
        gt = tail.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        parts.append(torch.round(tail / gt).clamp(-1.0, 1.0) * gt)
    deq = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    return (deq - W).detach() + W


# ---------------------------------------------------------------------------
# CPU validation (--validate-cpu)
# ---------------------------------------------------------------------------

def _run_validate_cpu() -> tuple[bool, bool, bool, bool]:
    """Run 3 correctness checks on a tiny TernaryLoRAWrapper.

    No model download required — uses a tiny AutoBitLinear constructed directly.

    Returns (all_pass, check1, check2, check3).
    """
    from transformers.integrations.bitnet import AutoBitLinear

    print()
    print("[validate-cpu] ============================================================")
    print("[validate-cpu] TernaryLoRA CPU correctness checks (no model download)")
    print("[validate-cpu]   CHECK 1: LoRA term == (alpha/r)*ternarize(B)@ternarize(A)@x")
    print("[validate-cpu]   CHECK 2: A=0 -> output == base output (LoRA term = 0)")
    print("[validate-cpu]   CHECK 3: backward reaches lora_A and lora_B (grad != 0)")
    print("[validate-cpu] ============================================================")

    torch.manual_seed(42)
    IN, OUT, R, ALPHA = 8, 16, 4, 8.0

    # Tiny base AutoBitLinear (random bf16 weights, no real model).
    base = AutoBitLinear(in_features=IN, out_features=OUT, bias=False, online_quant=True)
    base.weight.data = torch.randn(OUT, IN).bfloat16()

    # Known A and B for warm-start init.
    A_known = torch.randn(R, IN).bfloat16()
    B_known = torch.randn(OUT, R).bfloat16()

    # ----- CHECK 1: LoRA term at lam=1 matches manual ternary computation -----
    # NOTE: We call wrapper1._lora_term(x) directly rather than (wrapper(x) - base(x))
    # to avoid bfloat16 subtraction cancellation error — adding two bfloat16 values
    # and then subtracting one does not exactly recover the other due to rounding.
    wrapper1 = TernaryLoRAWrapper(
        base=base, r=R, alpha=ALPHA,
        in_features=IN, out_features=OUT,
        A_init=A_known, B_init=B_known,
    )
    x = torch.randn(2, IN, dtype=torch.bfloat16)

    _set_lora_lambda(1.0)
    _set_lora_enabled(True)

    with torch.no_grad():
        lora_actual = wrapper1._lora_term(x)   # direct call, no subtraction cancellation

    # Manual reference: (alpha/r) * ternarize(B) @ ternarize(A) @ x
    scale = ALPHA / R
    A_q_ref = _absmean_ternarize(wrapper1.lora_A.detach())
    B_q_ref = _absmean_ternarize(wrapper1.lora_B.detach())
    lora_ref = scale * F.linear(F.linear(x, A_q_ref), B_q_ref)

    diff1   = (lora_actual - lora_ref).abs().max().item()
    # Both paths use identical tensor objects and same ops; diff should be exactly 0.
    check1  = diff1 < 1e-5
    status1 = "PASS" if check1 else "FAIL"
    print(f"\n[validate-cpu] CHECK 1 (LoRA term == manual ternary): {status1}")
    print(f"  max_abs_diff = {diff1:.4e}  (should be 0.0 — identical computation paths)")
    if not check1:
        print(f"  lora_actual[:4]: {lora_actual.flatten()[:4].tolist()}")
        print(f"  lora_ref[:4]:    {lora_ref.flatten()[:4].tolist()}")

    # ----- CHECK 2: A=0 -> ternarize(A)=0 -> LoRA term=0 -> output==base -----
    wrapper2 = TernaryLoRAWrapper(
        base=base, r=R, alpha=ALPHA,
        in_features=IN, out_features=OUT,
        A_init=torch.zeros(R, IN, dtype=torch.bfloat16),
        B_init=B_known,
    )
    _set_lora_lambda(1.0)
    _set_lora_enabled(True)

    with torch.no_grad():
        out_zero_A = wrapper2(x)
        base_ref   = base(x)

    diff2   = (out_zero_A - base_ref).abs().max().item()
    check2  = diff2 < 1e-5   # ternarize(zeros)=zeros => exact match
    status2 = "PASS" if check2 else "FAIL"
    print(f"\n[validate-cpu] CHECK 2 (A=0 -> output == base): {status2}")
    print(f"  max_abs_diff = {diff2:.4e}  (should be 0.0 — ternarize(zeros)=zeros)")
    if not check2:
        print(f"  out_zero_A[:4]: {out_zero_A.flatten()[:4].tolist()}")
        print(f"  base_ref[:4]:   {base_ref.flatten()[:4].tolist()}")

    # ----- CHECK 3: backward at lam=0.5 (blended) reaches lora_A and lora_B -----
    wrapper3 = TernaryLoRAWrapper(
        base=base, r=R, alpha=ALPHA,
        in_features=IN, out_features=OUT,
        A_init=A_known, B_init=B_known,
    )
    _set_lora_lambda(0.5)
    _set_lora_enabled(True)

    # Zero any pre-existing grads.
    for p in wrapper3.parameters():
        if p.grad is not None:
            p.grad.zero_()

    x3   = torch.randn(4, IN, dtype=torch.bfloat16)
    out3 = wrapper3(x3)
    loss3 = out3.float().sum()
    loss3.backward()

    grad_A = wrapper3.lora_A.grad.abs().sum().item() if wrapper3.lora_A.grad is not None else 0.0
    grad_B = wrapper3.lora_B.grad.abs().sum().item() if wrapper3.lora_B.grad is not None else 0.0
    check3 = (grad_A > 0.0) and (grad_B > 0.0)
    status3 = "PASS" if check3 else "FAIL"
    print(f"\n[validate-cpu] CHECK 3 (grad reaches lora_A and lora_B at lam=0.5): {status3}")
    print(f"  lora_A.grad.abs().sum() = {grad_A:.4f}  (should be > 0)")
    print(f"  lora_B.grad.abs().sum() = {grad_B:.4f}  (should be > 0)")

    # Verify base is truly frozen (no grad to base.weight).
    base_grad = base.weight.grad.abs().sum().item() if base.weight.grad is not None else 0.0
    print(f"  base.weight.grad        = {base_grad:.4f}  (should be 0 — base is frozen)")

    all_pass = check1 and check2 and check3
    print()
    print("[validate-cpu] ============================================================")
    print(f"[validate-cpu] RESULT: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("[validate-cpu] ============================================================")
    print()
    return all_pass, check1, check2, check3


# ---------------------------------------------------------------------------
# Ternary-weight inference cache
# ---------------------------------------------------------------------------

def enable_inference_cache(model: nn.Module) -> int:
    """Pre-compute all frozen ternary weights once; reuse across every forward.

    Eliminates the per-token ``WeightQuant.apply`` / ``unpack_weights`` calls that
    cause ~1 tok/s generation on BitNet models.  Safe to call multiple times
    (idempotent — already-patched modules are skipped).

    Call AFTER ``model.eval()`` and after LoRA weights are loaded at ``lam=1``.
    Do NOT call during training — ``_infer_cache`` defaults to ``False`` and is
    never set by the training loop.

    Returns the count of BitNet linear modules whose forward was patched.
    """
    from transformers.integrations.bitnet import AutoBitLinear, BitLinear, WeightQuant, unpack_weights

    _infer_cache[0] = True
    n_patched = 0

    for module in model.modules():

        # ---- AutoBitLinear (online_quant=True) — WeightQuant.apply every forward ----
        if isinstance(module, AutoBitLinear) and getattr(module, 'online_quant', False):
            if hasattr(module, '_orig_forward'):
                continue  # already patched — skip

            def _make_autobit_fwd(m: AutoBitLinear):
                from transformers.integrations.bitnet import ActQuant as _ActQuant
                from transformers.integrations.bitnet import WeightQuant as _WeightQuant

                def _cached_fwd(input_):
                    if m.rms_norm is not None:
                        input_ = m.rms_norm(input_)
                    # Compute qweight once on the first call; reuse every subsequent token.
                    if not hasattr(m, '_cached_qweight') or m._cached_qweight is None:
                        with torch.no_grad():
                            m._cached_qweight = _WeightQuant.apply(m.weight).detach()
                    inp = _ActQuant.apply(input_)
                    return F.linear(inp, m._cached_qweight, m.bias)
                    # NOTE: online_quant=True path does NOT multiply by weight_scale.

                return _cached_fwd

            module._orig_forward = module.forward   # marker + restore handle
            module.forward = _make_autobit_fwd(module)
            n_patched += 1

        # ---- BitLinear (packed uint8) — unpack_weights every forward ----
        elif isinstance(module, BitLinear):
            if hasattr(module, '_orig_forward'):
                continue  # already patched — skip

            def _make_bitlinear_fwd(m: BitLinear):
                from transformers.integrations.bitnet import unpack_weights as _unpack

                def _cached_fwd(input_):
                    if m.rms_norm is not None:
                        input_ = m.rms_norm(input_)
                    # Unpack once; reuse.
                    if not hasattr(m, '_cached_qweight') or m._cached_qweight is None:
                        with torch.no_grad():
                            m._cached_qweight = _unpack(m.weight, dtype=m.dtype).detach()
                    input_quant, input_scale = m.activation_quant(input_)
                    y = F.linear(input_quant.to(m.dtype), m._cached_qweight)
                    y = m.post_quant_process(y, m.weight_scale, input_scale)
                    if m.bias is not None:
                        y += m.bias.view(1, -1).expand_as(y)
                    return y

                return _cached_fwd

            module._orig_forward = module.forward
            module.forward = _make_bitlinear_fwd(module)
            n_patched += 1

    return n_patched


def disable_inference_cache(model: nn.Module) -> int:
    """Restore per-forward weight recompute; drop all cached tensors.

    Returns the count of BitNet linear modules restored.
    """
    from transformers.integrations.bitnet import AutoBitLinear, BitLinear

    _infer_cache[0] = False
    n_restored = 0

    for module in model.modules():
        # Restore patched BitNet linears.
        if isinstance(module, (AutoBitLinear, BitLinear)) and hasattr(module, '_orig_forward'):
            # Delete the instance-level 'forward' attribute we injected; falls back to class method.
            del module._orig_forward
            if 'forward' in module.__dict__:
                del module.forward
            if hasattr(module, '_cached_qweight'):
                del module._cached_qweight
            n_restored += 1

        # Clear TernaryLoRAWrapper LoRA caches.
        if isinstance(module, TernaryLoRAWrapper):
            for _attr in ('_cached_qA', '_cached_qB'):
                if hasattr(module, _attr):
                    delattr(module, _attr)

    return n_restored


# ---------------------------------------------------------------------------
# Inference-cache CPU validation (--validate-infer-cache)
# ---------------------------------------------------------------------------

def _run_validate_cpu_infer_cache() -> tuple[bool, bool, bool, bool, bool]:
    """Run 4 CPU checks for the ternary-weight inference cache (no model download).

    CHECK A: output with cache ON == output with cache OFF (exact match — pure memoization).
    CHECK B: cached qweight is the SAME tensor object after 3 consecutive forwards
             (WeightQuant called once, not per-token).
    CHECK C: disable_inference_cache restores per-forward recompute; _cached_* attrs gone.
    CHECK D: training path unchanged with cache OFF (grad still flows through lora_A, lora_B).

    Returns (all_pass, A, B, C, D).
    """
    from transformers.integrations.bitnet import AutoBitLinear

    print()
    print("[validate-infer-cache] ================================================")
    print("[validate-infer-cache] Ternary inference cache CPU checks (no model DL)")
    print("[validate-infer-cache]   CHECK A: output cache-ON == output cache-OFF")
    print("[validate-infer-cache]   CHECK B: qweight is same object after 3 fwds")
    print("[validate-infer-cache]   CHECK C: disable cleans up attrs; output unchanged")
    print("[validate-infer-cache]   CHECK D: training grad flows with cache OFF (default)")
    print("[validate-infer-cache] ================================================")

    torch.manual_seed(0)
    IN, MID, OUT, R, ALPHA = 8, 16, 8, 4, 8.0

    # ---- Build tiny test graph: TernaryLoRAWrapper(AutoBitLinear) -> AutoBitLinear ----
    base_lin = AutoBitLinear(in_features=IN, out_features=MID, bias=False, online_quant=True)
    base_lin.weight.data = torch.randn(MID, IN).bfloat16()

    wrapper = TernaryLoRAWrapper(
        base=base_lin, r=R, alpha=ALPHA,
        in_features=IN, out_features=MID,
        A_init=torch.randn(R, IN).bfloat16(),
        B_init=torch.randn(MID, R).bfloat16(),
    )

    bare_lin = AutoBitLinear(in_features=MID, out_features=OUT, bias=False, online_quant=True)
    bare_lin.weight.data = torch.randn(OUT, MID).bfloat16()

    class _TinyModel(nn.Module):
        def __init__(self_):
            super().__init__()
            self_.wrap = wrapper
            self_.lin  = bare_lin
        def forward(self_, x):
            return self_.lin(self_.wrap(x))

    model = _TinyModel()
    model.eval()
    _set_lora_lambda(1.0)
    _set_lora_enabled(True)
    # Guarantee cache starts disabled.
    _infer_cache[0] = False

    x = torch.randn(3, IN, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # CHECK A: output cache-ON == output cache-OFF  (exact, diff == 0.0)
    # ------------------------------------------------------------------
    with torch.no_grad():
        out_off = model(x).clone()

    n_patched = enable_inference_cache(model)

    with torch.no_grad():
        out_on = model(x).clone()

    diff_A  = (out_on - out_off).abs().max().item()
    check_A = diff_A == 0.0
    print(f"\n[validate-infer-cache] CHECK A (output cache-ON == cache-OFF): {'PASS' if check_A else 'FAIL'}")
    print(f"  n_patched={n_patched}  max_abs_diff={diff_A:.4e}  (should be 0.0 — pure memoization)")
    if not check_A:
        print(f"  out_off[:4]: {out_off.flatten()[:4].tolist()}")
        print(f"  out_on[:4]:  {out_on.flatten()[:4].tolist()}")

    # ------------------------------------------------------------------
    # CHECK B: cached tensors are the SAME object across multiple forwards
    #          (proves WeightQuant called ONCE, not per token)
    # ------------------------------------------------------------------
    # The first enable+forward above already set _cached_qweight on bare_lin.
    check_B_setup = hasattr(bare_lin, '_cached_qweight') and hasattr(wrapper, '_cached_qA')
    id_bare_1 = id(bare_lin._cached_qweight) if check_B_setup else None
    id_qA_1   = id(wrapper._cached_qA)       if check_B_setup else None

    with torch.no_grad():
        _ = model(x)  # 2nd forward
        _ = model(x)  # 3rd forward

    id_bare_3 = id(bare_lin._cached_qweight) if check_B_setup else None
    id_qA_3   = id(wrapper._cached_qA)       if check_B_setup else None

    check_B = (check_B_setup
               and id_bare_1 == id_bare_3
               and id_qA_1   == id_qA_3
               and hasattr(wrapper, '_cached_qB'))

    print(f"\n[validate-infer-cache] CHECK B (same tensor object after 3 fwds): {'PASS' if check_B else 'FAIL'}")
    print(f"  bare_lin._cached_qweight present = {check_B_setup}")
    print(f"  wrapper._cached_qA present       = {hasattr(wrapper, '_cached_qA')}")
    print(f"  wrapper._cached_qB present       = {hasattr(wrapper, '_cached_qB')}")
    if check_B_setup:
        same_bare = id_bare_1 == id_bare_3
        same_qA   = id_qA_1   == id_qA_3
        print(f"  bare_lin qweight id unchanged  = {same_bare}  (should be True)")
        print(f"  wrapper  qA     id unchanged   = {same_qA}   (should be True)")

    # ------------------------------------------------------------------
    # CHECK C: disable_inference_cache cleans up all attrs; output unchanged
    # ------------------------------------------------------------------
    n_restored = disable_inference_cache(model)

    no_qw_bare = not hasattr(bare_lin, '_cached_qweight')
    no_qA_wrap = not hasattr(wrapper,  '_cached_qA')
    no_qB_wrap = not hasattr(wrapper,  '_cached_qB')
    no_orig_bare = not hasattr(bare_lin, '_orig_forward')
    fw_attr_gone = 'forward' not in bare_lin.__dict__   # instance attr deleted

    with torch.no_grad():
        out_after_disable = model(x).clone()

    diff_C  = (out_after_disable - out_off).abs().max().item()
    check_C = (no_qw_bare and no_qA_wrap and no_qB_wrap
               and no_orig_bare and fw_attr_gone
               and diff_C == 0.0)
    print(f"\n[validate-infer-cache] CHECK C (disable cleans up; output correct): {'PASS' if check_C else 'FAIL'}")
    print(f"  n_restored={n_restored}")
    print(f"  _cached_qweight gone = {no_qw_bare}  _cached_qA gone = {no_qA_wrap}  _cached_qB gone = {no_qB_wrap}")
    print(f"  _orig_forward gone   = {no_orig_bare}  forward inst-attr gone = {fw_attr_gone}")
    print(f"  max_abs_diff         = {diff_C:.4e}  (should be 0.0)")

    # ------------------------------------------------------------------
    # CHECK D: training path unchanged with cache OFF (default False)
    # ------------------------------------------------------------------
    # After disable_inference_cache, _infer_cache[0] == False.
    cache_off_confirmed = not _infer_cache[0]

    base_tr = AutoBitLinear(in_features=IN, out_features=MID, bias=False, online_quant=True)
    base_tr.weight.data = torch.randn(MID, IN).bfloat16()

    wrapper_tr = TernaryLoRAWrapper(
        base=base_tr, r=R, alpha=ALPHA,
        in_features=IN, out_features=MID,
        A_init=torch.randn(R, IN).bfloat16(),
        B_init=torch.randn(MID, R).bfloat16(),
    )
    wrapper_tr.train()
    _set_lora_lambda(0.5)   # blended path exercises STE gradients
    _set_lora_enabled(True)

    x_tr  = torch.randn(2, IN, dtype=torch.bfloat16)
    out_tr = wrapper_tr(x_tr)
    loss_tr = out_tr.float().sum()
    loss_tr.backward()

    grad_A = wrapper_tr.lora_A.grad.abs().sum().item() if wrapper_tr.lora_A.grad is not None else 0.0
    grad_B = wrapper_tr.lora_B.grad.abs().sum().item() if wrapper_tr.lora_B.grad is not None else 0.0
    check_D = cache_off_confirmed and (grad_A > 0.0) and (grad_B > 0.0)
    print(f"\n[validate-infer-cache] CHECK D (training grad with cache OFF): {'PASS' if check_D else 'FAIL'}")
    print(f"  _infer_cache[0] = {_infer_cache[0]}  (should be False)")
    print(f"  lora_A.grad sum = {grad_A:.4f}  lora_B.grad sum = {grad_B:.4f}  (both should be > 0)")

    # Restore sane defaults for the rest of the session.
    _set_lora_lambda(1.0)
    _set_lora_enabled(True)

    all_pass = check_A and check_B and check_C and check_D
    print()
    print("[validate-infer-cache] ================================================")
    print(f"[validate-infer-cache] RESULT: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("[validate-infer-cache] ================================================")
    print()
    return all_pass, check_A, check_B, check_C, check_D


# ---------------------------------------------------------------------------
# Text extractor (handles multiple dataset schemas)
# ---------------------------------------------------------------------------

def _extract_text(row: dict) -> str:
    """Extract a plain text string from a dataset row.

    Handles:
    - messages-style (no_robots, ShareGPT): {'messages': [{role, content}]}
    - conversations-style: {'conversations': [{from, content/value}]}
    - alpaca instruction style: {'instruction', optional 'input', 'output'}
    - plain text field: {'text': '...'}
    Falls back to joining all non-empty string values.
    """
    if "messages" in row:
        parts = [m.get("content", m.get("value", "")) for m in row["messages"]]
        return " ".join(p for p in parts if p)
    if "conversations" in row:
        parts = [t.get("content", t.get("value", "")) for t in row["conversations"]]
        return " ".join(str(p) for p in parts if p)
    if "instruction" in row:
        parts = [str(row["instruction"])]
        if row.get("input"):
            parts.append(str(row["input"]))
        if row.get("output"):
            parts.append(str(row["output"]))
        return "\n\n".join(parts)
    if "text" in row:
        return str(row["text"])
    if "question" in row and "answer" in row:  # GSM8K-style CoT
        return str(row["question"]) + "\n" + str(row["answer"])
    return " ".join(str(v) for v in row.values() if isinstance(v, str) and v)


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def _build_corpus(
    tokenizer,
    n_seqs: int,
    seq_len: int,
    dataset_id: str,
    split: str,
    dataset_config: Optional[str] = None,
) -> list[torch.Tensor]:
    """Load and tokenize dataset into fixed-length token sequences."""
    import datasets as hf_datasets

    ds = (hf_datasets.load_dataset(dataset_id, dataset_config, split=split)
          if dataset_config else hf_datasets.load_dataset(dataset_id, split=split))
    all_ids: list[int] = []
    for row in ds:
        text = _extract_text(row)
        if not text:
            continue
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= n_seqs * seq_len + seq_len:
            break

    seqs: list[torch.Tensor] = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break
    return seqs


# ---------------------------------------------------------------------------
# PPL computation (self-contained — no import from pdiff_func_eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_ppl(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 512,
) -> dict:
    """Compute token-level perplexity over a list of text strings.

    Returns dict with keys: ppl, mean_nll, n_tokens, n_sequences.
    """
    model.eval()
    total_nll   = 0.0
    total_toks  = 0
    n_ok        = 0

    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        )
        ids = enc["input_ids"].to(device)
        if ids.size(1) < 2:
            continue
        logits = model(input_ids=ids).logits   # [1, T, V]
        V = logits.size(-1)
        nll = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, V),
            ids[:, 1:].contiguous().view(-1),
            reduction="sum",
        )
        total_nll  += nll.item()
        total_toks += ids.size(1) - 1
        n_ok       += 1

    mean_nll = total_nll / max(total_toks, 1)
    ppl      = math.exp(min(mean_nll, 100.0))
    return {"ppl": ppl, "mean_nll": mean_nll, "n_tokens": total_toks, "n_sequences": n_ok}


# ---------------------------------------------------------------------------
# Layer-wrapping helpers
# ---------------------------------------------------------------------------

def _get_adapter_weights(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load lora_A / lora_B tensors from adapter_model.safetensors.

    Keys are the full PEFT weight names, e.g.:
        base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight
    """
    from safetensors import safe_open

    st_path = Path(adapter_path) / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_path!r}")

    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(st_path), framework="pt") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    return weights


def _wrap_mlp_layers(
    model,
    target_layer_indices: Optional[list[int]],
    r: int,
    alpha: float,
    adapter_weights: Optional[dict[str, torch.Tensor]],
    proj_names: frozenset = MLP_PROJ_NAMES,
    group_size: int = 0,
) -> list[str]:
    """Replace target AutoBitLinear MLP projections with TernaryLoRAWrapper.

    Args:
        model:               BitNetForCausalLM.
        target_layer_indices: Layers to wrap (None = all).
        r:                   LoRA rank.
        alpha:               LoRA alpha.
        adapter_weights:     Pre-loaded safetensors weights dict (may be None for cold start).
        proj_names:          Which projection names to wrap.

    Returns:
        List of wrapped layer keys (for logging).
    """
    layers  = model.model.layers
    wrapped = []

    for layer_idx, layer in enumerate(layers):
        if target_layer_indices is not None and layer_idx not in target_layer_indices:
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        for proj_name in proj_names:
            submod = getattr(mlp, proj_name, None)
            if submod is None:
                continue

            in_f  = submod.in_features
            out_f = submod.out_features

            # Look up A_init and B_init from adapter safetensors.
            A_init: Optional[torch.Tensor] = None
            B_init: Optional[torch.Tensor] = None

            if adapter_weights is not None:
                prefix  = f"base_model.model.model.layers.{layer_idx}.mlp.{proj_name}"
                key_A   = f"{prefix}.lora_A.weight"
                key_B   = f"{prefix}.lora_B.weight"
                A_init  = adapter_weights.get(key_A)
                B_init  = adapter_weights.get(key_B)
                if A_init is None or B_init is None:
                    # Adapter may not cover this proj; fall back to cold-start init.
                    A_init = None
                    B_init = None
            else:
                # Cold start: Kaiming-uniform A, zeros B (standard LoRA init).
                A_init = torch.nn.init.kaiming_uniform_(
                    torch.empty(r, in_f, dtype=torch.bfloat16), a=math.sqrt(5)
                )
                B_init = torch.zeros(out_f, r, dtype=torch.bfloat16)

            wrapper = TernaryLoRAWrapper(
                base=submod,
                r=r,
                alpha=alpha,
                in_features=in_f,
                out_features=out_f,
                A_init=A_init,
                B_init=B_init,
                group_size=group_size,
            )

            setattr(mlp, proj_name, wrapper)
            key = f"layers.{layer_idx}.mlp.{proj_name}"
            wrapped.append(key)

    return wrapped


def _collect_lora_params(model) -> list[nn.Parameter]:
    """Return all lora_A / lora_B parameters from TernaryLoRAWrapper modules."""
    params = []
    for m in model.modules():
        if isinstance(m, TernaryLoRAWrapper):
            params.append(m.lora_A)
            params.append(m.lora_B)
    return params


def _save_lora_weights(model, out_dir: str, meta: dict) -> None:
    """Save all TernaryLoRAWrapper lora_A/lora_B masters to out_dir.

    Saves:
        tlora_weights.pt  — dict of {layer_key: {lora_A: Tensor, lora_B: Tensor}}
        tlora_meta.json   — training configuration and results
    """
    os.makedirs(out_dir, exist_ok=True)
    state: dict = {}
    for name, m in model.named_modules():
        if isinstance(m, TernaryLoRAWrapper):
            state[name] = {
                "lora_A": m.lora_A.data.cpu(),
                "lora_B": m.lora_B.data.cpu(),
                "r":      m.r,
                "alpha":  m.alpha,
            }

    torch.save(state, os.path.join(out_dir, "tlora_weights.pt"))
    with open(os.path.join(out_dir, "tlora_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[pdiff_ternary_lora] Saved: tlora_weights.pt + tlora_meta.json -> {out_dir!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Ternary LoRA (NOT merged): keep base BitNet frozen, train a separate "
            "ternary low-rank term with STE.  "
            "y = base(x) + (alpha/r)*ternarize_ste(B)@ternarize_ste(A)@x.  "
            "Warm-start A,B from a PEFT adapter's lora_A/lora_B."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--adapter", default="none",
        help=(
            "PEFT LoRA adapter directory for warm-start init of A,B, or 'none' for "
            "cold start (Kaiming A, zeros B).  The adapter is NOT merged into the base; "
            "it is only used to initialise the ternary LoRA A,B masters."
        ),
    )
    ap.add_argument(
        "--r", type=int, default=16,
        help="LoRA rank (used when adapter='none'; overridden by adapter_config.json when adapter given).",
    )
    ap.add_argument(
        "--alpha", type=float, default=32.0,
        help="LoRA alpha (used when adapter='none'; overridden by adapter_config.json when adapter given).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output directory for trained lora_A/lora_B masters + tlora_meta.json.",
    )
    ap.add_argument(
        "--steps", type=int, default=1500,
        help="Number of QAT optimizer steps.",
    )
    ap.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate.",
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Token sequence length for training sequences.",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="Training device: cpu or cuda.",
    )
    ap.add_argument(
        "--last-n-layers", type=int, default=None,
        help="Wrap only the last N transformer layers' MLP projections.",
    )
    ap.add_argument(
        "--ternary-group-size", type=int, default=0, metavar="G",
        help="Group-wise absmean ternary scale for A/B in blocks of G along the "
             "in-dim (LoTA-QAF-style). 0 = per-tensor (default).",
    )
    ap.add_argument(
        "--lambda-ramp-steps", type=int, default=0, metavar="N",
        help=(
            "Ramp ternary strength from lam=0 (pure bf16 LoRA) to lam=1 (full ternary) "
            "over the first N steps.  lam = min(step/N, 1.0).  "
            "0 = disabled (start at full ternary lam=1 immediately)."
        ),
    )
    ap.add_argument(
        "--optim", choices=["adamw", "adafactor"], default="adamw",
        help="Optimizer: adamw or adafactor (lower memory).",
    )
    ap.add_argument(
        "--dataset", default=EVAL_DATASET, metavar="DATASET_ID",
        help="HuggingFace dataset for training and eval.",
    )
    ap.add_argument(
        "--dataset-config", default=None, metavar="CONFIG",
        help="Dataset config name (e.g. 'main' for openai/gsm8k).",
    )
    ap.add_argument(
        "--train-split", default="train[500:2500]", metavar="SPLIT",
        help="Dataset split for the training corpus.",
    )
    ap.add_argument(
        "--eval-split", default="train[:500]", metavar="SPLIT",
        help="Dataset split for PPL evaluation (should not overlap with training).",
    )
    ap.add_argument(
        "--n-eval", type=int, default=100,
        help="Number of sequences to use for PPL evaluation.",
    )
    ap.add_argument(
        "--eval-steps", type=int, default=50,
        help="Log a progress line every N steps.",
    )
    ap.add_argument(
        "--validate-cpu", action="store_true",
        help=(
            "Run 3 CPU correctness checks on a tiny TernaryLoRAWrapper "
            "(no model download, <1s) and exit."
        ),
    )
    ap.add_argument(
        "--validate-infer-cache", action="store_true",
        help=(
            "Run 4 CPU checks for the ternary-weight inference cache "
            "(CHECK A-D: correctness, single-compute, cleanup, training-path). "
            "No model download, <1s."
        ),
    )
    ap.add_argument(
        "--no-train", action="store_true",
        help=(
            "Skip the training loop; evaluate the warm-started masters as-is. "
            "With --adapter <gen>, the lam=0 PPL is the pure bf16 ceiling."
        ),
    )
    ap.add_argument(
        "--base-model", default=BASE_MODEL_ID, dest="base_model",
        metavar="MODEL_ID",
        help=(
            "HuggingFace model ID or local path for the BitNet base model to fine-tune "
            "(default: %(default)s). "
            "Use 'HF1BitLLM/Llama3-8B-1.58-100B-tokens' for the 8B Llama-arch variant."
        ),
    )

    args = ap.parse_args(argv)

    # ------------------------------------------------------------------
    # --validate-cpu: correctness checks only
    # ------------------------------------------------------------------
    if args.validate_cpu:
        all_pass, c1, c2, c3 = _run_validate_cpu()
        if not all_pass:
            sys.exit(1)
        return

    if args.validate_infer_cache:
        all_pass, cA, cB, cC, cD = _run_validate_cpu_infer_cache()
        if not all_pass:
            sys.exit(1)
        return

    if args.out is None:
        ap.error("--out is required unless --validate-cpu is specified")

    os.makedirs(args.out, exist_ok=True)
    device       = torch.device(args.device)
    adapter_path = args.adapter.strip()
    use_adapter  = adapter_path.lower() != "none"
    ramp_steps   = max(0, args.lambda_ramp_steps)

    print("[pdiff_ternary_lora] ==================================================")
    print("[pdiff_ternary_lora] APPROACH: ternary LoRA NOT merged (LoTA-QAF style)")
    print("[pdiff_ternary_lora] base AutoBitLinear: FROZEN throughout training")
    print("[pdiff_ternary_lora] y = base(x) + (alpha/r)*ternarize_ste(B)@ternarize_ste(A)@x")
    if ramp_steps > 0:
        print(f"[pdiff_ternary_lora] LAMBDA-RAMP: 0->1 over first {ramp_steps} steps")
    print("[pdiff_ternary_lora] ==================================================")

    # ------------------------------------------------------------------
    # 1. Load base model (frozen — never trains)
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[pdiff_ternary_lora] Loading base model {args.base_model!r} ...")
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            local_files_only=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    n_layers     = len(model.model.layers)
    print(f"[pdiff_ternary_lora] Loaded: {total_params:,} params, {n_layers} layers")

    # ------------------------------------------------------------------
    # 2. Determine r, alpha, and target layer indices
    # ------------------------------------------------------------------
    r     = args.r
    alpha = args.alpha

    if use_adapter:
        adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            ap.error(f"adapter_config.json not found in {adapter_path!r}")
        with open(adapter_cfg_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        r     = int(adapter_cfg.get("r", r))
        alpha = float(adapter_cfg.get("lora_alpha", alpha))
        print(
            f"[pdiff_ternary_lora] Adapter: r={r}  alpha={alpha}  "
            f"target_modules={adapter_cfg.get('target_modules')}"
        )

    # Target layers: last-n-layers override, else adapter layers_to_transform, else all.
    target_layer_indices: Optional[list[int]] = None
    if args.last_n_layers is not None:
        target_layer_indices = list(range(max(0, n_layers - args.last_n_layers), n_layers))
        print(
            f"[pdiff_ternary_lora] --last-n-layers {args.last_n_layers}: "
            f"wrapping layers {target_layer_indices[0]}..{target_layer_indices[-1]}"
        )
    elif use_adapter:
        ltt = adapter_cfg.get("layers_to_transform")
        if ltt is not None:
            target_layer_indices = sorted(set(ltt))
            print(f"[pdiff_ternary_lora] Adapter layers_to_transform: {len(target_layer_indices)} layers")
        else:
            print(f"[pdiff_ternary_lora] layers_to_transform=null: wrapping ALL {n_layers} layers")

    # ------------------------------------------------------------------
    # 3. Load adapter weights for warm start (if given)
    # ------------------------------------------------------------------
    adapter_weights: Optional[dict[str, torch.Tensor]] = None
    if use_adapter:
        print(f"[pdiff_ternary_lora] Loading adapter weights from {adapter_path!r} ...")
        adapter_weights = _get_adapter_weights(adapter_path)
        print(f"[pdiff_ternary_lora] Loaded {len(adapter_weights)} tensors from adapter")

    # ------------------------------------------------------------------
    # 4. Wrap target MLP projections with TernaryLoRAWrapper
    # ------------------------------------------------------------------
    print("[pdiff_ternary_lora] Wrapping target MLP projections ...")
    wrapped_keys = _wrap_mlp_layers(
        model=model,
        target_layer_indices=target_layer_indices,
        r=r,
        alpha=alpha,
        adapter_weights=adapter_weights,
        group_size=args.ternary_group_size,
    )
    _gs = args.ternary_group_size
    print(f"[pdiff_ternary_lora] Wrapped {len(wrapped_keys)} projections  (r={r}, alpha={alpha}, "
          f"ternary={'group-'+str(_gs) if _gs > 0 else 'per-tensor'})")
    if not wrapped_keys:
        raise RuntimeError("No projections wrapped. Check --last-n-layers or model structure.")

    # Freeze all base model parameters; only lora_A/lora_B are trainable.
    for p in model.parameters():
        p.requires_grad_(False)
    # Re-enable lora_A and lora_B.
    lora_params = _collect_lora_params(model)
    for p in lora_params:
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in lora_params)
    print(
        f"[pdiff_ternary_lora] Trainable: {len(lora_params)} tensors  "
        f"{trainable:,} params  ({100 * trainable / max(total_params, 1):.3f}% of base)"
    )

    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters found after wrapping.")

    # The base loaded via device_map={"": device}; the freshly-created LoRA masters
    # default to CPU. Move them onto the base device so F.linear matches (mat2 device).
    for p in lora_params:
        if p.device != device:
            p.data = p.data.to(device)
    print(f"[pdiff_ternary_lora] Moved {len(lora_params)} LoRA masters to {device}")

    # ------------------------------------------------------------------
    # 5. Training corpus
    # ------------------------------------------------------------------
    n_corpus = max(20, args.steps + 10)
    print(
        f"[pdiff_ternary_lora] Building train corpus from {args.dataset!r} "
        f"split={args.train_split!r} ..."
    )
    corpus = _build_corpus(
        tokenizer=tokenizer,
        n_seqs=n_corpus,
        seq_len=args.max_seq_len,
        dataset_id=args.dataset,
        split=args.train_split,
        dataset_config=args.dataset_config,
    )
    if not corpus:
        raise RuntimeError(f"Empty corpus from dataset={args.dataset!r} split={args.train_split!r}.")
    print(f"[pdiff_ternary_lora] Corpus: {len(corpus)} seqs x {args.max_seq_len} tokens")

    # ------------------------------------------------------------------
    # 6. Optimizer
    # ------------------------------------------------------------------
    if args.optim == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(
            lora_params,
            lr=args.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        print(f"[pdiff_ternary_lora] Optimizer: Adafactor (lr={args.lr})")
    else:
        opt = torch.optim.AdamW(
            lora_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
        )
        print(f"[pdiff_ternary_lora] Optimizer: AdamW (lr={args.lr})")

    # ------------------------------------------------------------------
    # 7. Set initial lambda
    # ------------------------------------------------------------------
    initial_lam = 0.0 if ramp_steps > 0 else 1.0
    _set_lora_lambda(initial_lam)
    _set_lora_enabled(True)

    if ramp_steps > 0:
        print(
            f"[pdiff_ternary_lora] Lambda-ramp enabled: "
            f"lam 0.0 -> 1.0 over first {ramp_steps} steps  "
            f"({max(0, args.steps - ramp_steps)} steps at full ternary)"
        )
    else:
        print(f"[pdiff_ternary_lora] Ternary LoRA: lam=1.0 from step 0 (no ramp)")

    # ------------------------------------------------------------------
    # 8. Training loop (QAT-STE)
    # ------------------------------------------------------------------
    model.train()
    t0     = time.perf_counter()
    losses: list[float] = []
    ACCUM  = 8   # gradient accumulation for cleaner STE gradients

    if args.no_train:
        args.steps = 0
        print("[pdiff_ternary_lora] --no-train: skipping training loop "
              "(eval warm-started masters as-is; bf16 ceiling from gen adapter)")

    print(f"\n[pdiff_ternary_lora] Starting training: {args.steps} steps")
    print(f"  Grad accum: {ACCUM}  Effective batch = {ACCUM} seqs")
    print(f"  Base is frozen — only lora_A,lora_B receive gradients\n")

    for step in range(args.steps):
        # Update ramp lambda.
        if ramp_steps > 0:
            lam = min(step / ramp_steps, 1.0)
            _set_lora_lambda(lam)

        seq    = corpus[step % len(corpus)].unsqueeze(0).to(device)  # [1, T]
        logits = model(input_ids=seq).logits                          # [1, T, V]
        V      = logits.size(-1)
        loss   = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, V),
            seq[:, 1:].contiguous().view(-1),
        ) / ACCUM

        loss.backward()

        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            opt.zero_grad()

        lv = loss.item() * ACCUM
        losses.append(lv)

        if step < 5 or (step + 1) % args.eval_steps == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - t0
            sps     = (step + 1) / elapsed if elapsed > 0 else 0.0
            lam_str = f"  lam={_lora_lambda[0]:.3f}" if ramp_steps > 0 else ""
            print(
                f"  step {step+1:5d}/{args.steps}  loss={lv:.4f}  "
                f"{sps:.2f} steps/s{lam_str}",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    init_loss     = losses[0]  if losses else float("nan")
    final_loss    = losses[-1] if losses else float("nan")
    avg_loss      = sum(losses) / len(losses) if losses else float("nan")

    print(
        f"\n[pdiff_ternary_lora] Training done: {args.steps} steps in {elapsed_total:.1f}s "
        f"({args.steps / elapsed_total:.2f} steps/s)"
    )
    print(
        f"  init_loss={init_loss:.4f}  final_loss={final_loss:.4f}  avg_loss={avg_loss:.4f}"
    )

    # ------------------------------------------------------------------
    # 9. Evaluation: ternary LoRA PPL + base PPL
    # ------------------------------------------------------------------
    print(f"\n[pdiff_ternary_lora] Building eval corpus from {args.dataset!r} "
          f"split={args.eval_split!r} ...")

    import datasets as hf_datasets
    ds_eval  = (hf_datasets.load_dataset(args.dataset, args.dataset_config, split=args.eval_split)
                if args.dataset_config else hf_datasets.load_dataset(args.dataset, split=args.eval_split))
    n_take   = min(args.n_eval, len(ds_eval))
    eval_texts = [_extract_text(ex) for ex in ds_eval.select(range(n_take))]
    eval_texts = [t for t in eval_texts if t]
    print(f"[pdiff_ternary_lora] Eval: {len(eval_texts)} examples")

    # Ensure full ternary for inference (lam=1).
    _set_lora_lambda(1.0)

    # --- PPL with ternary LoRA (lam=1) ---
    _set_lora_enabled(True)
    print("[pdiff_ternary_lora] Computing PPL with ternary LoRA (lam=1) ...")
    metrics_lora = _compute_ppl(
        model=model, tokenizer=tokenizer,
        texts=eval_texts, device=device, max_seq_len=args.max_seq_len,
    )
    ppl_lora = metrics_lora["ppl"]

    # --- PPL bf16 ceiling (lam=0, LoRA on but NOT ternarized) ---
    _set_lora_lambda(0.0)
    print("[pdiff_ternary_lora] Computing bf16-ceiling PPL (lam=0, LoRA not ternarized) ...")
    metrics_bf16 = _compute_ppl(
        model=model, tokenizer=tokenizer,
        texts=eval_texts, device=device, max_seq_len=args.max_seq_len,
    )
    ppl_bf16 = metrics_bf16["ppl"]
    _set_lora_lambda(1.0)

    # --- PPL base only (LoRA disabled) ---
    _set_lora_enabled(False)
    print("[pdiff_ternary_lora] Computing base PPL (LoRA disabled) ...")
    metrics_base = _compute_ppl(
        model=model, tokenizer=tokenizer,
        texts=eval_texts, device=device, max_seq_len=args.max_seq_len,
    )
    ppl_base = metrics_base["ppl"]
    _set_lora_enabled(True)

    delta = ppl_lora - ppl_base
    arrow = "better" if delta < -0.1 else ("worse" if delta > 0.1 else "~same")
    # Fraction of the achievable bf16 gain that the ternary term captures.
    bf16_gain = ppl_base - ppl_bf16
    tern_gain = ppl_base - ppl_lora
    capture   = (tern_gain / bf16_gain) if abs(bf16_gain) > 1e-6 else float("nan")

    print()
    print("[pdiff_ternary_lora] ==============================")
    print(f"  BASE  PPL   = {ppl_base:.3f}  (base ternary, no LoRA)")
    print(f"  BF16  PPL   = {ppl_bf16:.3f}  (base + bf16 LoRA, lam=0 = ceiling)")
    print(f"  TLORA PPL   = {ppl_lora:.3f}  (base + ternary LoRA, lam=1)")
    print(f"  Delta       = {delta:+.3f}  ({arrow})")
    print(f"  bf16_gain   = {bf16_gain:+.3f}   tern_gain = {tern_gain:+.3f}   capture = {capture*100:.1f}%")
    print(f"  eval_split  = {args.eval_split!r}  n={len(eval_texts)}  no_train={args.no_train}")
    print("[pdiff_ternary_lora] ==============================")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    meta = {
        "base_model":          args.base_model,
        "adapter_init":        adapter_path,
        "r":                   r,
        "alpha":               alpha,
        "steps":               args.steps,
        "lr":                  args.lr,
        "optim":               args.optim,
        "device":              args.device,
        "lambda_ramp_steps":   ramp_steps,
        "dataset":             args.dataset,
        "train_split":         args.train_split,
        "eval_split":          args.eval_split,
        "n_eval":              args.n_eval,
        "last_n_layers":       args.last_n_layers,
        "ternary_group_size":  args.ternary_group_size,
        "target_layer_indices": target_layer_indices,
        "n_wrapped":           len(wrapped_keys),
        "trainable_params":    trainable,
        "init_loss":           float(init_loss),
        "final_loss":          float(final_loss),
        "avg_loss":            float(avg_loss),
        "elapsed_s":           round(elapsed_total, 2),
        "ppl_base":            round(ppl_base, 4),
        "ppl_tlora":           round(ppl_lora, 4),
        "ppl_delta":           round(delta, 4),
    }
    _save_lora_weights(model, args.out, meta)

    # ------------------------------------------------------------------
    # 11. Smoke gate
    # ------------------------------------------------------------------
    loss_finite   = not (math.isnan(final_loss) or math.isinf(final_loss))
    loss_stable   = final_loss < init_loss * 5.0 if not math.isnan(init_loss) else True
    ppl_sane      = ppl_base < 100.0 and ppl_lora < 200.0

    print(
        f"\n[pdiff_ternary_lora] GATE: "
        f"loss_finite={loss_finite}  loss_stable={loss_stable}  "
        f"n_wrapped={len(wrapped_keys)}>0={len(wrapped_keys) > 0}  ppl_sane={ppl_sane}"
    )
    if not (loss_finite and len(wrapped_keys) > 0 and ppl_sane):
        raise RuntimeError("[pdiff_ternary_lora] GATE FAILED — see GATE line above.")
    print("[pdiff_ternary_lora] GATE PASSED")


if __name__ == "__main__":
    main()
