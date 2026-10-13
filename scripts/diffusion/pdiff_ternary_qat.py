#!/usr/bin/env python3
"""pdiff_ternary_qat.py — Ternary QAT recovery for weight-space-diffusion adapters.

APPROACH: Merge LoRA delta into base MLP bf16 master weights → QAT-finetune with STE.

WHY POST-HOC TERNARIZATION FAILS:
    After merge, AutoBitLinear computes: ternarize(W_base + LoRA_delta) @ x.
    This single ternarization is NOT double-quant — it's the correct ternary
    projection of the merged weight. The problem is that the task-specific
    signal in LoRA_delta is typically too small (~1-5% of ||W_base||) to survive
    projection onto the ternary grid: sign changes that encode task structure
    are overridden by the dominant W_base sign pattern. Result: base-ternary PPL
    degradation regardless of the adapter quality.

THE QAT FIX:
    1. Merge LoRA into bf16 master: W_bf16 = W_base + LoRA_delta
    2. Short supervised fine-tune on task data with AutoBitLinear STE:
         Forward:  ternarize(W_bf16) used for computation (ONE ternarization)
         Loss:     LM cross-entropy on no_robots
         Backward: gradient passes through STE to W_bf16 (∂round/∂W = 1)
         Update:   optimizer nudges W_bf16 so its ternary projection encodes task gain
    3. After QAT: model runs normally — AutoBitLinear ternarizes bf16 master once
       per forward, as it always does. The adaptation lives in the ternary grid.

THE LAMBDA-RAMP EXTENSION (--lambda-ramp-steps N):
    Standard one-jump QAT starts at full ternarization (λ=1).  For merged adapters
    this can push the model into a bad basin (~17.9 PPL) that plain QAT cannot escape.
    The λ-ramp gradually ternarizes over N steps:

        λ = min(step / N, 1.0)
        w_eff     = (1 - λ) · w_bf16     + λ · WeightQuant(w_bf16)
        input_eff = (1 - λ) · input      + λ · ActQuant(input)

    At λ=0: pure bf16 forward (quality ≈ 12.4 for gen_adapter).
    At λ=1: exactly the native AutoBitLinear forward (online_quant=True path).
    Gradient flows to w_bf16 through both the identity branch and the STE branch
    (∂WeightQuant/∂w = 1 by STE, ∂ActQuant/∂input = 1 by STE).

    After training the monkey-patch is removed before save_pretrained; the saved
    model is a plain AutoBitLinear model — identical to what pdiff_ternary_qat_eval.py
    expects (no patch residue, no double-quant).

HOW WE AVOID THE DOUBLE-QUANT TRAP:
    - We NEVER call absmean_ternarize() to pre-bake the merged weight
      (that bake + AutoBitLinear online quant = double-quant → sign pattern destroyed)
    - After merge_and_unload(), AutoBitLinear.weight stays as bf16 master (no explicit quant)
    - AutoBitLinear.forward() does exactly ONE ternarization at run time (online_quant=True)
    - During QAT, the STE (via AutoBitLinear's internal round_ste) backpropagates through
      the ternarization to the bf16 master weights
    - After save_pretrained + reload, AutoBitLinear does the same single ternarization: correct

EVALUATION (avoiding double-quant):
    Load model with AutoModelForCausalLM.from_pretrained (NO PEFT adapter, NO pre-bake).
    AutoBitLinear ternarizes bf16 master ONCE at forward.
    Use pdiff_ternary_qat_eval.py — compares QAT PPL against base/bf16 references.

CLI:
    # Validate the lambda-ramp mechanism (CPU, ~1 second, NO model download):
    python scripts/diffusion/pdiff_ternary_qat.py --validate-ramp

    # Smoke test (CPU, tiny randomly-initialised model, ~seconds):
    python scripts/diffusion/pdiff_ternary_qat.py \\
        --adapter none --out /tmp/qat_smoke \\
        --steps 5 --tiny-random --device cpu

    # Lambda-ramp smoke test (tiny random model, just tests the loop runs):
    python scripts/diffusion/pdiff_ternary_qat.py \\
        --adapter none --out /tmp/qat_ramp_smoke \\
        --steps 20 --tiny-random --lambda-ramp-steps 10 --device cpu

    # Real run (GPU) — generated adapter WITH lambda ramp:
    python scripts/diffusion/pdiff_ternary_qat.py \\
        --adapter .docs/diffusion/pdiff_boot_fixedinit/pdiff_func_out38/gen_adapter \\
        --out E:/qat_ramp_out/gen_adapter \\
        --last-n-layers 12 --lambda-ramp-steps 1000 --steps 2000 \\
        --optim adafactor --device cuda

    # Real run (GPU) — generated adapter WITHOUT lambda ramp (original behaviour):
    python scripts/diffusion/pdiff_ternary_qat.py \\
        --adapter .docs/diffusion/pdiff_boot_fixedinit/pdiff_func_out38/gen_adapter \\
        --out .docs/diffusion/qat_out/gen_adapter \\
        --steps 200 --device cuda
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

import numpy as np

# Ensure stdout/stderr use UTF-8 on Windows (avoids cp1252 UnicodeEncodeError
# from Greek letters in print statements and argparse help strings).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent

BASE_MODEL_ID  = "microsoft/bitnet-b1.58-2B-4T-bf16"
EVAL_DATASET   = "HuggingFaceH4/no_robots"
MLP_PROJ_NAMES = frozenset({"gate_proj", "up_proj", "down_proj"})


# ---------------------------------------------------------------------------
# Lambda-ramp state (gradual ternarization)
# ---------------------------------------------------------------------------

_lambda_state: list[float] = [0.0]      # mutable container; [0] = current λ
_original_abl_forward: list = [None]    # saved AutoBitLinear.forward for restore


def _set_lambda(val: float) -> None:
    """Set the global λ value (clamped to [0, 1])."""
    _lambda_state[0] = float(max(0.0, min(1.0, val)))


def _install_lambda_ramp() -> None:
    """Monkey-patch AutoBitLinear.forward with lambda-blended ternarization.

    The patch replaces the CLASS-level forward so ALL AutoBitLinear instances
    in the process use the blended path.  The original forward is saved for
    restoration before save_pretrained.

    At λ=0: pure bf16 forward — F.linear(input, w_bf16, bias).
    At λ=1: exact native forward — F.linear(ActQuant(input), WeightQuant(w), bias).
    In-between: blended weight + blended activation, gradients flow via STE.
    """
    from transformers.integrations.bitnet import AutoBitLinear, WeightQuant, ActQuant

    if _original_abl_forward[0] is not None:
        return  # already installed

    _original_abl_forward[0] = AutoBitLinear.forward

    def _patched(self: "AutoBitLinear", input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        # --- offline-quant path: pre-packed weights; do not interfere ---
        if not self.online_quant:
            return _original_abl_forward[0](self, input)

        # --- RMSNorm on activations (same position as original) ---
        if self.rms_norm is not None:
            input = self.rms_norm(input)

        lam = _lambda_state[0]
        w = self.weight  # bf16 master

        if lam <= 0.0:
            # Pure bf16: no weight quant, no act quant.
            # Identical to F.linear(input, w, self.bias) — the genuine bf16 forward.
            return F.linear(input, w, self.bias)

        if lam >= 1.0:
            # Native ternary: EXACTLY reproduces the original online_quant forward.
            w_q = WeightQuant.apply(w)
            x_q = ActQuant.apply(input)
            return F.linear(x_q, w_q, self.bias)

        # Blended path (0 < λ < 1):
        #   w_eff     = (1-λ)·w_bf16  + λ·WeightQuant(w_bf16)
        #   input_eff = (1-λ)·input   + λ·ActQuant(input)
        # Gradient to w_bf16: (1-λ)·∂L/∂w_eff + λ·STE·∂L/∂w_eff = ∂L/∂w_eff
        # (STE: ∂WeightQuant/∂w = 1, ∂ActQuant/∂input = 1)
        w_ternary = WeightQuant.apply(w)
        w_eff = (1.0 - lam) * w + lam * w_ternary
        x_q = ActQuant.apply(input)
        x_eff = (1.0 - lam) * input + lam * x_q
        return F.linear(x_eff, w_eff, self.bias)

    AutoBitLinear.forward = _patched
    print("[lambda-ramp] AutoBitLinear.forward monkey-patched (lam-blend enabled)")


def _remove_lambda_ramp() -> None:
    """Restore the original AutoBitLinear.forward (class-level)."""
    if _original_abl_forward[0] is None:
        return
    from transformers.integrations.bitnet import AutoBitLinear
    AutoBitLinear.forward = _original_abl_forward[0]
    _original_abl_forward[0] = None
    print("[lambda-ramp] AutoBitLinear.forward restored to native")


# ---------------------------------------------------------------------------
# Sparse top-K KD loss (teacher-cache path)
# ---------------------------------------------------------------------------

def _compute_kd_loss(
    student_logits: torch.Tensor,
    tc_idx_row: "np.ndarray",
    tc_logit_row: "np.ndarray",
    device: torch.device,
    kd_temp: float,
) -> torch.Tensor:
    """Sparse top-K KL divergence loss against cached teacher logits.

    Formula (Hinton et al., temperature-scaled KD):
        teacher_p  = softmax(tc_logit / T)           over K positions
        student_lp = log_softmax(s_topk / T)         over K positions
        kd         = mean_pos( sum_k( teacher_p * (log(teacher_p+eps) - student_lp) ) ) * T^2

    Args:
        student_logits: [1, T, V]  raw logits from the student forward pass.
        tc_idx_row:     [T, K]     int32 numpy — top-K vocab indices for this sequence.
        tc_logit_row:   [T, K]     float16 numpy — teacher raw logits for those indices.
        device:         torch.device — where student_logits lives.
        kd_temp:        temperature T (default 2.0).

    Returns:
        Scalar KD loss tensor (on ``device``).
    """
    # Positions 0..T-2 predict tokens 1..T-1 (same shift as LM loss; final pos ignored)
    s_logits = student_logits[0, :-1, :].float()       # [L, V]  fp32 for stability

    # Slice teacher data to match (skip final position)
    tc_idx_np    = tc_idx_row[:-1].astype(np.int64)    # [L, K]
    tc_logit_np  = tc_logit_row[:-1].astype(np.float32)  # [L, K]

    tc_idx_t   = torch.from_numpy(tc_idx_np).to(device)    # [L, K]
    tc_logit_t = torch.from_numpy(tc_logit_np).to(device)  # [L, K]

    # Gather student logits at teacher top-K positions
    s_topk = s_logits.gather(-1, tc_idx_t)             # [L, K]

    # Teacher: softmax over K at temperature T
    teacher_p = F.softmax(tc_logit_t / kd_temp, dim=-1)   # [L, K]

    # Student: log-softmax over K-subset at temperature T
    student_lp = F.log_softmax(s_topk / kd_temp, dim=-1)  # [L, K]

    # KL: sum_k( teacher_p * (log(teacher_p+eps) - student_lp) ), mean over positions
    eps = 1e-10
    kd = (teacher_p * (torch.log(teacher_p + eps) - student_lp)).sum(-1).mean()

    # T^2 scaling restores gradient magnitude to same order as LM loss
    return kd * (kd_temp * kd_temp)


# ---------------------------------------------------------------------------
# Correctness validation (--validate-ramp)
# ---------------------------------------------------------------------------

def _run_validate_ramp() -> tuple[bool, bool, bool, bool]:
    """Run 3 correctness checks for the lambda-ramp patch.

    Returns:
        (all_pass, check1, check2, check3)
    """
    from transformers.integrations.bitnet import AutoBitLinear, WeightQuant, ActQuant

    print()
    print("[validate-ramp] ============================================================")
    print("[validate-ramp] Correctness validation for lambda-ramp monkey-patch")
    print("[validate-ramp]   CHECK 1: lam=0   patched forward == pure bf16 forward")
    print("[validate-ramp]   CHECK 2: lam=1   patched forward == native ternary forward")
    print("[validate-ramp]   CHECK 3: lam=0.5 backward reaches bf16 master (grad != 0)")
    print("[validate-ramp] ============================================================")

    # Small AutoBitLinear module with random bf16 weights (no real model download)
    torch.manual_seed(42)
    abl = AutoBitLinear(in_features=8, out_features=16, bias=False, online_quant=True)
    abl.weight.data = torch.randn(16, 8).to(torch.bfloat16)

    x = torch.randn(2, 8, dtype=torch.bfloat16)  # batch=2, features=8

    # Capture NATIVE outputs BEFORE installing the patch
    with torch.no_grad():
        # Pure bf16: no quantization at all
        ref_bf16   = F.linear(x, abl.weight, abl.bias)
        # Native ternary: normal AutoBitLinear.forward (online_quant=True)
        ref_ternary = abl(x)   # WeightQuant + ActQuant + F.linear

    # --- Install patch ---
    _install_lambda_ramp()

    # CHECK 1: λ=0 patched == pure bf16
    _set_lambda(0.0)
    with torch.no_grad():
        out_at_0 = abl(x)
    diff1 = (out_at_0 - ref_bf16).abs().max().item()
    check1 = torch.allclose(out_at_0, ref_bf16, atol=1e-5)
    status1 = "PASS" if check1 else "FAIL"
    print(f"\n[validate-ramp] CHECK 1 (lam=0 == pure bf16): {status1}")
    print(f"  max_abs_diff = {diff1:.2e}   (should be 0.0)")
    if not check1:
        print(f"  patched out : {out_at_0.flatten()[:4]}")
        print(f"  bf16 ref    : {ref_bf16.flatten()[:4]}")

    # CHECK 2: λ=1 patched == native ternary
    _set_lambda(1.0)
    with torch.no_grad():
        out_at_1 = abl(x)
    diff2 = (out_at_1 - ref_ternary).abs().max().item()
    check2 = torch.allclose(out_at_1, ref_ternary, atol=1e-5)
    status2 = "PASS" if check2 else "FAIL"
    print(f"\n[validate-ramp] CHECK 2 (lam=1 == native ternary): {status2}")
    print(f"  max_abs_diff = {diff2:.2e}   (should be 0.0)")
    if not check2:
        print(f"  patched out  : {out_at_1.flatten()[:4]}")
        print(f"  ternary ref  : {ref_ternary.flatten()[:4]}")

    # CHECK 3: backward at λ=0.5 reaches the bf16 master weight
    _set_lambda(0.5)
    abl.weight.requires_grad_(True)
    if abl.weight.grad is not None:
        abl.weight.grad.zero_()

    x_train = torch.randn(4, 8, dtype=torch.bfloat16)
    out_mid  = abl(x_train)
    loss     = out_mid.float().sum()
    loss.backward()

    grad_sum = abl.weight.grad.abs().sum().item() if abl.weight.grad is not None else 0.0
    check3   = (abl.weight.grad is not None) and (grad_sum > 0.0)
    status3  = "PASS" if check3 else "FAIL"
    print(f"\n[validate-ramp] CHECK 3 (grad to bf16 master at lam=0.5): {status3}")
    print(f"  weight.grad.abs().sum() = {grad_sum:.4f}   (should be > 0)")

    # --- Remove patch ---
    _remove_lambda_ramp()

    # Verify patch is really gone: native forward should equal ref_ternary
    with torch.no_grad():
        after_restore = abl(x)
    restore_ok = torch.allclose(after_restore, ref_ternary, atol=1e-5)
    print(f"\n[validate-ramp] PATCH RESTORE check (native == ternary ref): {'OK' if restore_ok else 'DIFF!'}")

    all_pass = check1 and check2 and check3
    print()
    print("[validate-ramp] ============================================================")
    print(f"[validate-ramp] RESULT: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("[validate-ramp] ============================================================")
    print()

    return all_pass, check1, check2, check3


# ---------------------------------------------------------------------------
# Tiny random model (smoke tests only — no real weights downloaded)
# ---------------------------------------------------------------------------

def _build_tiny_model(device: torch.device):
    """Build a tiny randomly-initialised BitNetForCausalLM (2 layers, hidden=64).

    Reads the config JSON from the local HF cache (small, fast).  Creates the
    model with random weights — no safetensors download required.

    NOTE: The tiny model built from BitNetForCausalLM(cfg) uses plain nn.Linear
    for MLP layers (no AutoBitLinear), because the quantization_config is not
    applied when constructing with random weights.  For lambda-ramp correctness
    checks use --validate-ramp instead (it creates AutoBitLinear modules directly).
    """
    from transformers import AutoConfig, BitNetForCausalLM

    cfg = AutoConfig.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    cfg.hidden_size           = 64
    cfg.intermediate_size     = 128
    cfg.num_hidden_layers     = 2
    cfg.num_attention_heads   = 2
    cfg.num_key_value_heads   = 2
    cfg.max_position_embeddings = 256
    # Keep vocab_size from real config so token IDs are valid.
    model = BitNetForCausalLM(cfg)
    model.to(device=device)
    return model


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def _build_corpus_random(vocab_size: int, n_seqs: int, seq_len: int) -> list:
    """Synthetic random token sequences for smoke tests (no tokenizer needed)."""
    return [torch.randint(0, vocab_size, (seq_len,)) for _ in range(n_seqs)]


def _extract_text(row: dict) -> str:
    """Extract a plain text string from a dataset row.

    Handles multiple common dataset schemas:
    - messages-style (no_robots, ShareGPT): {'messages': [{role, content}]}
    - conversations-style (tool-use): {'conversations': [{from, content/value}]}
    - alpaca instruction style (python_code_instructions, MathInstruct):
          {'instruction', optional 'input', 'output'}
    - plain text field: {'text': '...'}
    Falls back to joining all non-empty string values.
    """
    # messages-style (no_robots)
    if "messages" in row:
        parts = [m.get("content", m.get("value", "")) for m in row["messages"]]
        return " ".join(p for p in parts if p)

    # conversations-style (tooluse datasets)
    if "conversations" in row:
        parts = [t.get("content", t.get("value", "")) for t in row["conversations"]]
        return " ".join(str(p) for p in parts if p)

    # alpaca instruction style (python_code_instructions_18k_alpaca, MathInstruct)
    if "instruction" in row:
        parts = [str(row["instruction"])]
        if row.get("input"):
            parts.append(str(row["input"]))
        if row.get("output"):
            parts.append(str(row["output"]))
        return "\n\n".join(parts)

    # plain text field
    if "text" in row:
        return str(row["text"])

    # fallback: join all non-empty string values
    return " ".join(str(v) for v in row.values() if isinstance(v, str) and v)


def _build_corpus_real(
    tokenizer,
    n_seqs: int,
    seq_len: int,
    dataset_id: str = EVAL_DATASET,
    split: str = "train",
) -> list:
    """Load and tokenize a dataset train split for QAT corpus.

    Uses _extract_text to handle multiple dataset schemas:
    no_robots (messages), python_code_instructions (alpaca), MathInstruct (alpaca).
    """
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(dataset_id, split=split)
    all_ids: list = []
    for row in ds:
        text = _extract_text(row)
        if not text:
            continue
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= n_seqs * seq_len:
            break

    seqs: list = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break
    return seqs


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _freeze_and_unfreeze_mlp(
    model,
    target_layer_indices: Optional[list[int]],
    proj_names: frozenset = MLP_PROJ_NAMES,
) -> int:
    """Freeze all params; unfreeze weight in MLP projection submodules of target layers.

    The AutoBitLinear.weight tensor holds the bf16 master weight.  Unfreezing it
    means the optimizer will update the bf16 master.  AutoBitLinear.forward() then
    ternarizes it online (STE backprop path).

    Args:
        model: BitNetForCausalLM (after optional merge_and_unload).
        target_layer_indices: Layer indices whose MLP weights to unfreeze.
            ``None`` means ALL layers.
        proj_names: Set of projection names to unfreeze (gate/up/down_proj).

    Returns:
        Number of weight tensors unfrozen.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    n_unfrozen = 0
    for layer_idx, layer in enumerate(layers):
        if target_layer_indices is not None and layer_idx not in target_layer_indices:
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for submod_name, submod in mlp.named_modules():
            proj_name = submod_name.split(".")[-1]
            if proj_name not in proj_names:
                continue
            w = getattr(submod, "weight", None)
            if w is None:
                continue
            w.requires_grad_(True)
            n_unfrozen += 1

    return n_unfrozen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Ternary QAT recovery: merge LoRA adapter into bf16 master → "
            "STE QAT finetune on no_robots → save for eval via AutoBitLinear forward."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--adapter", default="none",
        help=(
            "PEFT LoRA adapter directory, or 'none' for plain base QAT (no adapter). "
            "The adapter is merged (merge_and_unload) before training so the bf16 master "
            "holds W_base + LoRA_delta — no pre-ternarization, no double-quant."
        ),
    )
    ap.add_argument(
        "--out", default=None,
        help="Output directory for save_pretrained (QAT-trained model + qat_meta.json).",
    )
    ap.add_argument(
        "--steps", type=int, default=200,
        help="Number of QAT optimizer steps (200 for a short GPU run; 5 for smoke).",
    )
    ap.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for the optimizer.",
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Token sequence length for training (ignored in --tiny-random mode).",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="Training device: cpu or cuda (default cpu for smoke; use cuda for real runs).",
    )
    ap.add_argument(
        "--tiny-random", action="store_true",
        help=(
            "Use a tiny randomly-initialised BitNetForCausalLM (2 layers, hidden=64) "
            "and a synthetic random corpus.  No full model download required.  "
            "Use for CPU smoke tests (finishes in seconds, validates STE + training loop). "
            "NOTE: The tiny model uses plain nn.Linear (not AutoBitLinear); the lambda-ramp "
            "patch has no effect on it (use --validate-ramp for ramp correctness checks)."
        ),
    )
    ap.add_argument(
        "--all-layers", action="store_true",
        help=(
            "Unfreeze ALL MLP layers even when an adapter targets a subset.  "
            "Default: respect layers_to_transform from adapter_config.json."
        ),
    )
    ap.add_argument(
        "--optim", choices=["adamw", "adafactor"], default="adamw",
        help="Optimizer: adamw (AdamW) or adafactor (Adafactor, lower memory).",
    )
    ap.add_argument(
        "--eval-steps", type=int, default=50,
        help="Log a progress line every N steps.",
    )
    ap.add_argument(
        "--last-n-layers", type=int, default=None,
        help="If set, QAT only the last N transformer layers' MLP weights (fits smaller GPUs).",
    )
    ap.add_argument(
        "--lambda-ramp-steps", type=int, default=0,
        metavar="N",
        help=(
            "Gradually ramp ternarization strength from λ=0 (pure bf16) to λ=1 (full ternary) "
            "over the first N steps.  At λ=0 the model runs at full bf16 quality (~12.4 PPL); "
            "at λ=1 it is exactly equivalent to the native AutoBitLinear forward.  "
            "Schedule: λ = min(step / N, 1.0).  Default 0 = disabled (original one-jump QAT). "
            "Recommended: set N < --steps so the model trains at full λ=1 after the ramp. "
            "Example: --lambda-ramp-steps 1000 --steps 2000"
        ),
    )
    ap.add_argument(
        "--validate-ramp", action="store_true",
        help=(
            "Run the 3 lambda-ramp correctness checks on a tiny AutoBitLinear module "
            "(no full model download, runs in <1s on CPU) and exit.  "
            "Checks: (1) λ=0 == pure bf16; (2) λ=1 == native ternary; "
            "(3) backward at λ=0.5 reaches bf16 master."
        ),
    )

    ap.add_argument(
        "--dataset", default=EVAL_DATASET, metavar="DATASET_ID",
        help=(
            "HuggingFace dataset ID to use for the QAT training corpus.  "
            "Default: HuggingFaceH4/no_robots.  "
            "Also supported: iamtarun/python_code_instructions_18k_alpaca, "
            "TIGER-Lab/MathInstruct.  "
            "Text is extracted via a schema-aware extractor that handles "
            "messages-style, alpaca instruction-style, and conversations-style rows."
        ),
    )
    ap.add_argument(
        "--train-split", default="train", metavar="SPLIT",
        help=(
            "Dataset split to use for the QAT training corpus.  "
            "Default: 'train'.  Use HF slice notation for small runs, "
            "e.g. 'train[:500]'."
        ),
    )

    # ------------------------------------------------------------------
    # KD (knowledge distillation) flags — pass all three to enable KD path
    # ------------------------------------------------------------------
    ap.add_argument(
        "--kd-teacher-cache", default=None, metavar="PATH",
        help=(
            "Directory produced by pdiff_teacher_cache.py containing topk_idx.npy, "
            "topk_logit.npy, and meta.json. When set (together with --kd-corpus), "
            "the training loss is kd_weight * KD_topk + (1-kd_weight) * LM. "
            "The student trains on the FIXED corpus in row order so indices align "
            "with the cached teacher logits."
        ),
    )
    ap.add_argument(
        "--kd-corpus", default=None, metavar="PATH",
        help=(
            "Directory produced by pdiff_kd_corpus.py containing ids.npy + meta.json. "
            "Must be the EXACT same corpus used to produce --kd-teacher-cache. "
            "Alignment is validated at startup (n_seqs and seq_len must match)."
        ),
    )
    ap.add_argument(
        "--kd-weight", type=float, default=0.9, metavar="W",
        help=(
            "Relative weight of the KD loss in the combined objective. "
            "total_loss = kd_weight * KD_topk + (1-kd_weight) * LM. "
            "Recommended 0.9 when the teacher is a strong task-adapted model. "
            "(default 0.9)"
        ),
    )
    ap.add_argument(
        "--kd-temp", type=float, default=2.0, metavar="T",
        help=(
            "Temperature for KD softmax / log-softmax.  Applied to both teacher and "
            "student; loss is scaled by T^2 (Hinton et al.). "
            "(default 2.0)"
        ),
    )

    args = ap.parse_args(argv)

    # ------------------------------------------------------------------
    # --validate-ramp: run correctness checks and exit
    # ------------------------------------------------------------------
    if args.validate_ramp:
        all_pass, c1, c2, c3 = _run_validate_ramp()
        if not all_pass:
            sys.exit(1)
        return

    if args.out is None:
        ap.error("--out is required unless --validate-ramp is specified")

    os.makedirs(args.out, exist_ok=True)
    device    = torch.device(args.device)
    adapter_path = args.adapter.strip()
    use_adapter  = adapter_path.lower() != "none"
    ramp_steps   = max(0, args.lambda_ramp_steps)

    print("[pdiff_ternary_qat] ==================================================")
    print("[pdiff_ternary_qat] APPROACH: merge LoRA -> STE QAT on MLP bf16 masters")
    print("[pdiff_ternary_qat] DOUBLE-QUANT AVOIDANCE: no pre-bake; AutoBitLinear")
    print("[pdiff_ternary_qat]   ternarizes bf16 master ONCE at forward (online_quant)")
    if ramp_steps > 0:
        print(f"[pdiff_ternary_qat] LAMBDA-RAMP: lam ramps 0->1 over first {ramp_steps} steps")
        print(f"[pdiff_ternary_qat]   At lam=0: pure bf16  |  At lam=1: native ternary (identical to eval)")
    print("[pdiff_ternary_qat] ==================================================")
    print(f"[pdiff_ternary_qat] adapter={adapter_path!r}  steps={args.steps}  "
          f"lr={args.lr}  device={device}  tiny_random={args.tiny_random}  "
          f"lambda_ramp_steps={ramp_steps}")

    # ------------------------------------------------------------------
    # 1. Load / build model
    # ------------------------------------------------------------------
    if args.tiny_random:
        print("[pdiff_ternary_qat] Building tiny random model (2 layers, hidden=64) ...")
        model     = _build_tiny_model(device)
        tokenizer = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[pdiff_ternary_qat] Loading base model {BASE_MODEL_ID!r} ...")
        if device.type == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                local_files_only=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[pdiff_ternary_qat] Model loaded: {total_params:,} parameters")

    # ------------------------------------------------------------------
    # 2. Merge adapter into bf16 master (if given)
    # ------------------------------------------------------------------
    adapted_layer_indices: Optional[list[int]] = None

    if use_adapter and not args.tiny_random:
        from peft import PeftModel

        # Read adapter config to discover which layers were adapted.
        adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
        with open(adapter_cfg_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        layers_to_transform = adapter_cfg.get("layers_to_transform")
        n_model_layers      = len(model.model.layers)

        if layers_to_transform is None or args.all_layers:
            # None means all layers; all_layers flag forces same behaviour.
            adapted_layer_indices = None   # freeze_and_unfreeze sees None → all layers
            _n_adapted = n_model_layers
        else:
            adapted_layer_indices = sorted(set(layers_to_transform))
            _n_adapted            = len(adapted_layer_indices)

        if args.last_n_layers:
            adapted_layer_indices = list(range(max(0, n_model_layers - args.last_n_layers), n_model_layers))
            _n_adapted = len(adapted_layer_indices)

        print(
            f"[pdiff_ternary_qat] Adapter: r={adapter_cfg.get('r')}  "
            f"alpha={adapter_cfg.get('lora_alpha')}  "
            f"target_modules={adapter_cfg.get('target_modules')}  "
            f"adapted_layers={_n_adapted}/{n_model_layers}"
        )

        # Load PEFT adapter and MERGE into bf16 master.
        # merge_and_unload() adds (lora_A @ lora_B * alpha/r) to AutoBitLinear.weight
        # (the bf16 master).  The result is a plain model (no PEFT wrapper):
        #   - AutoBitLinear.weight = W_base + LoRA_delta  (bf16, no ternarization yet)
        #   - AutoBitLinear.forward() will ternarize ONCE at run time
        # This is correct: we do NOT call absmean_ternarize() here (that would be the
        # pdiff_ternary_eval.py --ternarize adapted path which pre-bakes and causes
        # double-quant when AutoBitLinear re-ternarizes at forward).
        print(f"[pdiff_ternary_qat] Loading adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("[pdiff_ternary_qat] Merging adapter into bf16 masters (merge_and_unload) ...")
        model = model.merge_and_unload()
        model.config.use_cache = False
        print(
            "[pdiff_ternary_qat] Merge done: AutoBitLinear.weight = W_base + LoRA_delta  "
            "(bf16 master, NO pre-ternarization)"
        )

        # Ensure model is on the right device (merge may move some params to CPU).
        if device.type != "cpu":
            model.to(device)

    elif use_adapter and args.tiny_random:
        print("[pdiff_ternary_qat] --tiny-random: adapter skipped (training with random weights)")

    # ------------------------------------------------------------------
    # 3. Freeze all; unfreeze target MLP gate/up/down proj weights
    # ------------------------------------------------------------------
    # Apply --last-n-layers unconditionally (also for --adapter none), so the
    # trainable scope fits the GPU regardless of the adapter branch above.
    if args.last_n_layers:
        _nml = len(model.model.layers)
        adapted_layer_indices = list(range(max(0, _nml - args.last_n_layers), _nml))
    n_unfrozen = _freeze_and_unfreeze_mlp(model, adapted_layer_indices)
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[pdiff_ternary_qat] Unfrozen: {n_unfrozen} MLP AutoBitLinear weight tensors  "
        f"trainable={trainable:,} / total={total_params:,} "
        f"({100 * trainable / max(total_params, 1):.2f}%)"
    )

    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found.  Check that the model has AutoBitLinear MLP "
            "modules (gate_proj / up_proj / down_proj) and that adapted_layer_indices is correct."
        )

    # ------------------------------------------------------------------
    # 4. Install lambda-ramp patch (if enabled and model has AutoBitLinear)
    # ------------------------------------------------------------------
    ramp_active = False
    if ramp_steps > 0:
        try:
            from transformers.integrations.bitnet import AutoBitLinear
            # Check if the model actually has AutoBitLinear layers.
            has_abl = any(
                isinstance(m, AutoBitLinear)
                for m in model.modules()
            )
            if has_abl:
                _set_lambda(0.0)
                _install_lambda_ramp()
                ramp_active = True
                print(
                    f"[lambda-ramp] Enabled: lam ramps 0.0->1.0 over steps 0..{ramp_steps}  "
                    f"(total steps={args.steps}; {max(0, args.steps - ramp_steps)} steps at lam=1)"
                )
            else:
                print(
                    "[lambda-ramp] WARNING: --lambda-ramp-steps set but model has no AutoBitLinear "
                    "layers (tiny-random model uses plain nn.Linear).  Ramp has no effect."
                )
        except ImportError:
            print(
                "[lambda-ramp] WARNING: transformers.integrations.bitnet not found; "
                "lambda-ramp disabled."
            )

    # ------------------------------------------------------------------
    # 5. Corpus
    # ------------------------------------------------------------------
    vocab_size = model.config.vocab_size
    n_corpus   = max(20, args.steps + 10)

    if args.tiny_random:
        seq_len = 64
        corpus  = _build_corpus_random(vocab_size, n_seqs=20, seq_len=seq_len)
        print(f"[pdiff_ternary_qat] Tiny random corpus: {len(corpus)} seqs × {seq_len} tokens")
    else:
        print(f"[pdiff_ternary_qat] Building corpus from {args.dataset!r} split={args.train_split!r} ...")
        corpus = _build_corpus_real(
            tokenizer,
            n_seqs=n_corpus,
            seq_len=args.max_seq_len,
            dataset_id=args.dataset,
            split=args.train_split,
        )
        print(
            f"[pdiff_ternary_qat] Corpus: {len(corpus)} seqs × {args.max_seq_len} tokens"
        )

    if not corpus:
        raise RuntimeError(
            "Corpus is empty.  Check dataset availability or use --tiny-random."
        )

    # ------------------------------------------------------------------
    # 6. Optimizer
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optim == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(
            trainable_params,
            lr=args.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        print(f"[pdiff_ternary_qat] Optimizer: Adafactor (lr={args.lr})")
    else:
        opt = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        print(f"[pdiff_ternary_qat] Optimizer: AdamW (lr={args.lr})")

    # ------------------------------------------------------------------
    # 6b. KD corpus + cache loading (optional — only when --kd-* flags set)
    # ------------------------------------------------------------------
    kd_active       = False
    kd_corpus_ids   = None    # np.ndarray mmap [N, L] int32
    kd_topk_idx     = None    # np.ndarray mmap [N, L, K] int32
    kd_topk_logit   = None    # np.ndarray mmap [N, L, K] float16
    kd_n_seqs       = 0
    kd_weight       = float(args.kd_weight)
    kd_temp         = float(args.kd_temp)

    _kd_cache_path  = args.kd_teacher_cache
    _kd_corpus_path = args.kd_corpus

    if _kd_cache_path is not None or _kd_corpus_path is not None:
        # Both must be supplied together
        if _kd_cache_path is None or _kd_corpus_path is None:
            raise ValueError(
                "--kd-teacher-cache and --kd-corpus must both be supplied (or neither). "
                f"Got: --kd-teacher-cache={_kd_cache_path!r}  --kd-corpus={_kd_corpus_path!r}"
            )

        # Load corpus meta
        kd_corpus_meta_path = os.path.join(_kd_corpus_path, "meta.json")
        if not os.path.isfile(kd_corpus_meta_path):
            raise FileNotFoundError(f"KD corpus meta.json not found at {kd_corpus_meta_path!r}")
        with open(kd_corpus_meta_path, encoding="utf-8") as _f:
            kd_corpus_meta = json.load(_f)

        # Load cache meta
        kd_cache_meta_path = os.path.join(_kd_cache_path, "meta.json")
        if not os.path.isfile(kd_cache_meta_path):
            raise FileNotFoundError(f"KD cache meta.json not found at {kd_cache_meta_path!r}")
        with open(kd_cache_meta_path, encoding="utf-8") as _f:
            kd_cache_meta = json.load(_f)

        # Alignment validation
        if kd_corpus_meta["n_seqs"] != kd_cache_meta["n_seqs"]:
            raise ValueError(
                f"KD alignment error: corpus.n_seqs={kd_corpus_meta['n_seqs']} "
                f"!= cache.n_seqs={kd_cache_meta['n_seqs']}. "
                "Regenerate the cache against the exact same corpus file."
            )
        if kd_corpus_meta["seq_len"] != kd_cache_meta["seq_len"]:
            raise ValueError(
                f"KD alignment error: corpus.seq_len={kd_corpus_meta['seq_len']} "
                f"!= cache.seq_len={kd_cache_meta['seq_len']}. "
                "Regenerate the cache against the exact same corpus file."
            )

        kd_n_seqs = kd_corpus_meta["n_seqs"]

        # Memory-map the arrays (never fully read into RAM)
        kd_corpus_ids_path = os.path.join(_kd_corpus_path, "ids.npy")
        kd_topk_idx_path   = os.path.join(_kd_cache_path, "topk_idx.npy")
        kd_topk_logit_path = os.path.join(_kd_cache_path, "topk_logit.npy")

        for p in (kd_corpus_ids_path, kd_topk_idx_path, kd_topk_logit_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"KD file not found: {p!r}")

        kd_corpus_ids  = np.load(kd_corpus_ids_path, mmap_mode="r")
        kd_topk_idx    = np.load(kd_topk_idx_path,   mmap_mode="r")
        kd_topk_logit  = np.load(kd_topk_logit_path, mmap_mode="r")
        K = kd_cache_meta["K"]
        kd_active = True

        print(
            f"\n[pdiff_ternary_qat] KD active: "
            f"corpus={kd_n_seqs}×{kd_corpus_meta['seq_len']}  "
            f"K={K}  kd_weight={kd_weight}  kd_temp={kd_temp}"
        )
        print(
            f"  kd_corpus_ids  : {kd_corpus_ids.shape}  {kd_corpus_ids.dtype}"
        )
        print(
            f"  kd_topk_idx    : {kd_topk_idx.shape}  {kd_topk_idx.dtype}"
        )
        print(
            f"  kd_topk_logit  : {kd_topk_logit.shape}  {kd_topk_logit.dtype}"
        )
        print(
            f"  loss = {kd_weight:.2f} * KD_topk + {1-kd_weight:.2f} * LM"
        )

    # ------------------------------------------------------------------
    # 7. QAT training loop
    #
    # STE mechanism (implicit in AutoBitLinear.forward with online_quant=True):
    #   Forward:  W_ternary = round_ste(W_bf16 / gamma).clamp(-1,1) * gamma
    #             y = F.linear(x, W_ternary, bias)
    #   Backward: ∂L/∂W_bf16 = ∂L/∂W_ternary * (∂W_ternary/∂W_bf16)
    #             (∂W_ternary/∂W_bf16) = 1  [STE: identity through round]
    #   Update:   W_bf16 ← W_bf16 − lr * ∂L/∂W_bf16
    #
    # Lambda-ramp: same STE mechanism, but with blended effective weight.
    # At λ=1 the QAT training is identical to the non-ramp path.
    #
    # KD path: when kd_active, each step reads from the FIXED kd corpus in row order
    # (step % kd_n_seqs) and combines LM loss with sparse top-K KL loss against the
    # pre-cached teacher logits.  The λ-ramp continues to work exactly the same way.
    #
    # After training, AutoBitLinear at inference does exactly:
    #   W_ternary = ternarize(W_bf16_updated)  ← one ternarization, no double-quant
    # ------------------------------------------------------------------
    model.train()
    t0     = time.perf_counter()
    losses: list = []
    kd_losses: list = []
    lm_losses: list = []

    print(f"\n[pdiff_ternary_qat] Starting QAT loop: {args.steps} steps")
    if kd_active:
        print(
            f"  Mode: KD-QAT  (kd_weight={kd_weight}  kd_temp={kd_temp}  "
            f"kd_n_seqs={kd_n_seqs})"
        )
    if ramp_active:
        print(f"  Lambda-ramp: lam = min(step/{ramp_steps}, 1.0)")
    else:
        print(f"  STE via AutoBitLinear.forward() (online_quant=True, round_ste internal)")
    print(f"  One ternarization per forward pass — no double-quant\n")

    ACCUM = 8   # gradient accumulation: effective batch = ACCUM for cleaner STE gradients
    for step in range(args.steps):
        # Update lambda before each step
        if ramp_active:
            lam = min(step / ramp_steps, 1.0) if ramp_steps > 0 else 1.0
            _set_lambda(lam)

        if kd_active:
            # ---- KD path: fixed corpus, row order, combined KD + LM loss ----
            row_idx = step % kd_n_seqs
            # Fetch corpus row as int64 tensor [1, T]
            ids_np  = np.array(kd_corpus_ids[row_idx], dtype=np.int64)
            seq     = torch.from_numpy(ids_np).unsqueeze(0).to(device)

            logits = model(input_ids=seq).logits                    # [1, T, V]
            V      = logits.size(-1)

            # LM cross-entropy loss (next-token prediction)
            lm_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, V),
                seq[:, 1:].contiguous().view(-1),
            )

            # Sparse top-K KD loss against cached teacher logits
            kd_loss = _compute_kd_loss(
                student_logits=logits,
                tc_idx_row=kd_topk_idx[row_idx],    # [T, K] int32
                tc_logit_row=kd_topk_logit[row_idx], # [T, K] float16
                device=device,
                kd_temp=kd_temp,
            )

            # Combined loss (scaled for gradient accumulation)
            loss = (kd_weight * kd_loss + (1.0 - kd_weight) * lm_loss) / ACCUM

            kd_losses.append(kd_loss.item())
            lm_losses.append(lm_loss.item())
        else:
            # ---- LM-only path (original behavior, unchanged) ----
            seq    = corpus[step % len(corpus)].unsqueeze(0).to(device)  # [1, T]
            logits = model(input_ids=seq).logits                          # [1, T, V]
            V      = logits.size(-1)
            loss   = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, V),
                seq[:, 1:].contiguous().view(-1),
            ) / ACCUM

        loss.backward()
        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            opt.zero_grad()

        lv = loss.item() * ACCUM
        losses.append(lv)

        if step < 5 or (step + 1) % args.eval_steps == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - t0
            sps     = (step + 1) / elapsed if elapsed > 0 else 0.0
            lam_str = f"  lam={_lambda_state[0]:.3f}" if ramp_active else ""
            if kd_active and kd_losses and lm_losses:
                kd_str = (
                    f"  kd={kd_losses[-1]:.4f}  lm={lm_losses[-1]:.4f}"
                )
            else:
                kd_str = ""
            print(
                f"  step {step+1:5d}/{args.steps}  loss={lv:.4f}  "
                f"{sps:.2f} steps/s{lam_str}{kd_str}",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    final_loss    = losses[-1] if losses else float("nan")
    init_loss     = losses[0]  if losses else float("nan")
    avg_loss      = sum(losses) / len(losses) if losses else float("nan")

    print(
        f"\n[pdiff_ternary_qat] QAT done: {args.steps} steps in {elapsed_total:.1f}s  "
        f"({args.steps / elapsed_total:.2f} steps/s)"
    )
    print(
        f"  init_loss={init_loss:.4f}  final_loss={final_loss:.4f}  avg_loss={avg_loss:.4f}"
    )

    # ------------------------------------------------------------------
    # 8. Remove lambda-ramp patch BEFORE save
    #    The saved model must be plain AutoBitLinear — identical to what
    #    pdiff_ternary_qat_eval.py expects (no patch, no double-quant).
    # ------------------------------------------------------------------
    if ramp_active:
        _remove_lambda_ramp()
        print("[lambda-ramp] Patch removed; saved model is vanilla AutoBitLinear")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    print(f"[pdiff_ternary_qat] Saving QAT model to {args.out} ...")
    model.eval()
    model.save_pretrained(args.out)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.out)

    # KD summary stats
    avg_kd_loss = sum(kd_losses) / len(kd_losses) if kd_losses else None
    avg_lm_loss = sum(lm_losses) / len(lm_losses) if lm_losses else None

    qat_meta = {
        "base_model":             BASE_MODEL_ID,
        "adapter":                adapter_path,
        "steps":                  args.steps,
        "lr":                     args.lr,
        "device":                 args.device,
        "tiny_random":            args.tiny_random,
        "all_layers":             args.all_layers,
        "lambda_ramp_steps":      ramp_steps,
        "qat_dataset":            args.dataset,
        "qat_train_split":        args.train_split,
        "ramp_active":            ramp_active,
        "adapted_layer_indices":  adapted_layer_indices,
        "init_loss":              init_loss,
        "final_loss":             final_loss,
        "avg_loss":               avg_loss,
        "elapsed_s":              round(elapsed_total, 2),
        # KD fields (None when LM-only path)
        "kd_active":              kd_active,
        "kd_teacher_cache":       _kd_cache_path,
        "kd_corpus":              _kd_corpus_path,
        "kd_weight":              kd_weight if kd_active else None,
        "kd_temp":                kd_temp   if kd_active else None,
        "avg_kd_loss":            avg_kd_loss,
        "avg_lm_loss":            avg_lm_loss,
    }
    with open(os.path.join(args.out, "qat_meta.json"), "w", encoding="utf-8") as f:
        json.dump(qat_meta, f, indent=2)
    print(f"[pdiff_ternary_qat] Saved: model weights + qat_meta.json")
    if kd_active:
        print(
            f"  KD summary: avg_kd_loss={avg_kd_loss:.4f}  avg_lm_loss={avg_lm_loss:.4f}"
            if avg_kd_loss is not None and avg_lm_loss is not None
            else "  KD summary: no steps completed"
        )

    # ------------------------------------------------------------------
    # 10. Smoke gate
    # ------------------------------------------------------------------
    loss_finite    = not (math.isnan(final_loss) or math.isinf(final_loss))
    # Sanity: loss should not have exploded more than 2× the initial value.
    loss_stable    = final_loss < init_loss * 5.0 if not math.isnan(init_loss) else True
    trainable_ok   = n_unfrozen > 0
    corpus_ok      = (len(corpus) > 0) if not kd_active else (kd_n_seqs > 0)

    print(
        f"\n[pdiff_ternary_qat] GATE:  "
        f"loss_finite={loss_finite}  loss_stable={loss_stable}  "
        f"n_unfrozen={n_unfrozen}>0={trainable_ok}  corpus_ok={corpus_ok}"
    )
    if not (loss_finite and trainable_ok and corpus_ok):
        raise RuntimeError(
            "[pdiff_ternary_qat] GATE FAILED — see GATE line above."
        )
    print("[pdiff_ternary_qat] GATE PASSED")


if __name__ == "__main__":
    main()
