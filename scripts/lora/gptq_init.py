#!/usr/bin/env python3
"""gptq_init.py — GPTQ-style calibrated warm-start init for the BitLinear ternary student.

Build item **B5** of the capability-distillation plan
(`.planning/2026-07-12-capability-distillation-design.md` §5: *"Warm-start: GPTQ-init >
STE/RTN, add now"*; refs BitDistiller 2402.10631, 2601.14888).

Why
---
The default student init copies the teacher weights and lets the forward path ternarize them
on the fly with **per-tensor absmean** rounding (plain RTN). Research is unanimous that a
*calibrated* init (GPTQ/OBQ/AWQ-style) that minimizes per-layer output error under
quantization gives a materially better low-bit starting point than plain RTN/STE. This module
provides that init, computed once, right after the student is built and before training.

The exact quantization contract this MUST match
-----------------------------------------------
In `bitdistill.py`, ``BitLinear.forward`` reads **only** ``self.weight`` (the FP master) and
re-derives the ternary weight every forward via ``weight_quant_ternary``:

    scale = 1 / mean(|w|).clamp(min=1e-5)          # single per-tensor scale
    w_q   = round(w * scale).clamp(-1, 1) / scale  # values in {-1/scale, 0, +1/scale}

There is **no** separately stored scale or {-1,0,+1} assignment tensor — both are functions of
``self.weight`` alone, recomputed at every forward. Therefore the *only* lever a warm-start has
is the master weight it writes. This module computes a calibrated ternary solution
``Ŵ = α·T`` (T ∈ {-1,0,+1}, α a per-tensor scale) and then writes a master weight ``W_store``
such that the forward's own absmean quantizer **exactly reproduces** ``Ŵ`` — see
:func:`_store_weight_for_target` and its reproduction lemma (asserted numerically).

Method (pragmatic GPTQ, faithful to the per-tensor-absmean forward)
-------------------------------------------------------------------
Per BitLinear layer, using calibration activations X (the real, post-SubLN inputs the module
actually multiplies, gathered from a few CPT batches):

1. **Hessian** H = Xᵀ X (in×in), accumulated online — no raw activations retained. The
   per-layer output error of any candidate ``Ŵ`` is ``‖(W0 − Ŵ)X‖_F² = tr(D H Dᵀ)``,
   D = W0 − Ŵ — so H is a sufficient statistic for the reconstruction MSE.
2. **Optimal per-tensor scale search** over α = ratio·absmean(W0): pick the α whose RTN
   ternarization minimizes tr(D H Dᵀ). Because ratio = 1.0 is in the grid, this is *already*
   ≤ the plain-absmean MSE.
3. **GPTQ error-feedback rounding** at the chosen α (OBS column updates via the Cholesky of
   H⁻¹): quantizes column-by-column, pushing each column's rounding error into the
   not-yet-quantized columns so correlated error cancels. Classic GPTQ; RTN is its zero-
   feedback special case.
4. Keep whichever of {GPTQ, RTN@α} gives lower tr(D H Dᵀ) — so the result is **provably
   ≤ plain absmean** by construction (asserted).
5. Write ``W_store`` reproducing ``α·T`` into ``module.weight``.

Self-test (tiny synthetic Qwen3, seconds, no download)::

    TORCHDYNAMO_DISABLE=1 python scripts/lora/gptq_init.py --self-test

Real smoke on the actual Qwen3-0.6B (a few BitLinear layers, absmean vs warm-start MSE)::

    HF_HOME=E:/.cache/huggingface TORCHDYNAMO_DISABLE=1 \
        python scripts/lora/gptq_init.py --real --n-layers 2 --device cpu
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Iterator, Optional

import torch


# ===========================================================================
# Local copy of the forward-path quantizer (kept self-contained; must stay
# byte-for-byte equivalent to bitdistill.weight_quant_ternary).
# ===========================================================================
def _weight_quant_ternary(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean ternary quant -> {-s,0,+s}; identical to bitdistill's."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1) / scale


def _is_bitlinear(m: torch.nn.Module) -> bool:
    """Duck-type a BitLinear without importing bitdistill (avoids its import side effects).

    BitLinear is the only module carrying all of: a 2-D ``weight``, a ``sub_norm`` attribute
    (RMSNorm or None), a ``quant_alpha`` buffer, and integer ``in_features``/``out_features``.
    """
    return (
        hasattr(m, "weight")
        and getattr(m, "weight", None) is not None
        and getattr(m.weight, "ndim", 0) == 2
        and hasattr(m, "sub_norm")
        and hasattr(m, "quant_alpha")
        and hasattr(m, "in_features")
        and hasattr(m, "out_features")
    )


# ===========================================================================
# Reproduction lemma: build an FP master weight whose absmean-ternarization
# equals a chosen (scale α, ternary assignment T).
# ===========================================================================
def _store_weight_for_target(T: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return ``W_store`` such that ``_weight_quant_ternary(W_store) == alpha * T`` exactly.

    T has entries in {-1,0,+1}. The forward re-derives its scale as absmean(W_store), so we
    must pre-divide by the ternary density d = nnz(T)/numel: with ``W_store = (alpha/d)·T``,

        absmean(W_store) = (alpha/d)·d = alpha                       -> forward scale = 1/alpha
        round(W_store/alpha) = round(T/d), |T/d| = 1/d >= 1 for T=±1 -> round then clamp = T
                              = 0 for T=0
        => forward output = alpha · T   (the intended solution).

    Degenerate all-zero T -> W_store = 0 -> forward 0 = alpha·0. Robust for any density.
    """
    density = (T != 0).to(torch.float32).mean().clamp(min=1e-8)
    return (alpha / float(density)) * T.to(torch.float32)


def _recon_mse_from_H(D: torch.Tensor, H: torch.Tensor) -> float:
    """tr(D H Dᵀ) = Σ_o d_oᵀ H d_o = ‖(W0−Ŵ)X‖_F² for H = XᵀX, D = W0−Ŵ."""
    return float(((D @ H) * D).sum())


# ===========================================================================
# GPTQ error-feedback quantization at a FIXED per-tensor ternary scale.
# ===========================================================================
def _gptq_ternary(W0: torch.Tensor, H: torch.Tensor, alpha: float,
                  percdamp: float = 0.01) -> torch.Tensor:
    """Return a ternary assignment T ∈ {-1,0,+1} minimizing tr(D H Dᵀ) via GPTQ/OBS.

    Fixed per-tensor scale ``alpha`` (levels {-alpha,0,+alpha}); column-by-column rounding
    with Hessian error feedback (the classic GPTQ recipe: Frantar et al. 2210.17323). Plain
    RTN is exactly this with the error-feedback term removed.
    """
    W = W0.clone().to(torch.float32)
    H = H.clone().to(torch.float32)
    cols = W.shape[1]

    # Dead (never-activated) input dims: pin diag so Cholesky is well posed; force weight 0.
    diagH = torch.diagonal(H)
    dead = diagH == 0.0
    if bool(dead.any()):
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    damp = percdamp * torch.diagonal(H).mean().clamp(min=1e-8)
    idx = torch.arange(cols, device=H.device)
    H[idx, idx] += damp

    # Hinv (upper-tri Cholesky of the inverse) — GPTQ's numerically stable factorization.
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    T = torch.zeros_like(W)
    for j in range(cols):
        w = W[:, j]
        d = Hinv[j, j]
        t = torch.clamp(torch.round(w / alpha), -1, 1)         # ternary level for this column
        T[:, j] = t
        if j + 1 < cols and float(d) != 0.0:
            err = (w - t * alpha) / d                          # OBS error to redistribute
            W[:, j + 1:] -= err.unsqueeze(1) * Hinv[j, j + 1:].unsqueeze(0)
    return T


def _rtn_ternary(W0: torch.Tensor, alpha: float) -> torch.Tensor:
    """Plain round-to-nearest ternary assignment at scale ``alpha`` (no error feedback)."""
    return torch.clamp(torch.round(W0.to(torch.float32) / alpha), -1, 1)


# ===========================================================================
# Calibration: accumulate the per-layer Hessian H = XᵀX from a few CPT batches.
# ===========================================================================
def _collect_hessians(student, layers: dict, calib_iter: Iterator, device, *,
                      n_calib_batches: int, calib_batch_size: int,
                      hessian_device) -> dict:
    """Run the student (in FP mode, quant_alpha=0) over calibration batches and accumulate,
    per BitLinear, H = Σ xᵀx over the real post-SubLN inputs it multiplies.

    quant_alpha is forced to 0 so the calibration activations are the clean full-precision
    distribution we are trying to match (the standard GPTQ calibration setup) — restored after.
    """
    H: dict = {name: None for name in layers}
    handles = []

    def make_hook(name, module):
        def hook(_m, inp, _out):
            x = inp[0]
            if getattr(module, "sub_norm", None) is not None:
                x = module.sub_norm(x)               # the tensor that actually multiplies W
            x = x.reshape(-1, x.shape[-1]).float()   # XtX on the forward device (GPU = fast)
            h = (x.t() @ x).to(hessian_device)       # park the accumulator on hessian_device (CPU = low VRAM)
            H[name] = h if H[name] is None else H[name] + h
        return hook

    for name, module in layers.items():
        handles.append(module.register_forward_hook(make_hook(name, module)))

    # Save/force/restore quant_alpha (calibrate on the FP forward).
    saved_alpha = {}
    for name, module in layers.items():
        saved_alpha[name] = module.quant_alpha.clone()
        module.quant_alpha.fill_(0.0)

    was_training = student.training
    student.eval()
    pulled = 0
    try:
        with torch.no_grad():
            for _ in range(n_calib_batches):
                seqs = []
                for _ in range(calib_batch_size):
                    item = next(calib_iter)
                    if item.dim() == 1:
                        item = item.unsqueeze(0)
                    seqs.append(item)
                batch = torch.cat(seqs, dim=0).to(device)     # [B, T]
                student(input_ids=batch, use_cache=False)
                pulled += 1
    finally:
        for h in handles:
            h.remove()
        for name, module in layers.items():
            module.quant_alpha.copy_(saved_alpha[name])
        if was_training:
            student.train()

    return H


# ===========================================================================
# Public API
# ===========================================================================
def gptq_warmstart_init(
    student,
    calib_iter: Iterator,
    device,
    *,
    n_calib_batches: int = 8,
    calib_batch_size: int = 4,
    scale_search: bool = True,
    n_scale_grid: int = 31,
    scale_lo: float = 0.5,
    scale_hi: float = 2.0,
    use_gptq: bool = True,
    percdamp: float = 0.01,
    hessian_device: Optional[str] = None,
    layer_filter: Optional[Callable[[str], bool]] = None,
    verbose: bool = True,
) -> dict:
    """GPTQ-style calibrated warm-start for every BitLinear in ``student`` (in place).

    Overwrites each BitLinear's FP master ``weight`` with a calibrated value whose on-the-fly
    absmean ternarization has **lower per-layer reconstruction MSE than plain absmean** (the
    default init). Call this once, *after* the student is built and *before* training.

    Parameters
    ----------
    student : the converted BitDistill student (BitLinears already in place).
    calib_iter : iterator yielding token-id tensors ``[T]`` or ``[B, T]`` (e.g. the CPT
        ``cpt_token_stream``). ``n_calib_batches`` × ``calib_batch_size`` sequences are pulled.
    device : device the student lives on (batches are moved here).
    scale_search : search the per-tensor scale (ratio·absmean) minimizing MSE. Guarantees the
        result is ≤ absmean even with ``use_gptq=False``.
    use_gptq : apply GPTQ error-feedback rounding on top of the chosen scale.
    hessian_device : where to hold the (in×in) Hessians (default: ``device``; pass ``"cpu"``
        to spare VRAM on big models — Cholesky then runs on CPU).
    layer_filter : optional ``name -> bool`` to restrict which BitLinears are initialized
        (used by the self-test / partial-model validation).

    Returns
    -------
    dict with ``per_layer`` (name -> {mse_absmean, mse_warmstart, alpha, ratio, method,
    density}) and aggregate ``mse_absmean`` / ``mse_warmstart`` / ``mse_reduction`` /
    ``n_layers``. The aggregate warm-start MSE is asserted ≤ the absmean MSE.
    """
    hdev = torch.device(hessian_device) if hessian_device is not None else torch.device(device)

    layers = {}
    for name, m in student.named_modules():
        if _is_bitlinear(m) and (layer_filter is None or layer_filter(name)):
            layers[name] = m
    if not layers:
        raise ValueError("gptq_warmstart_init: no BitLinear modules found in student "
                         "(convert_to_bitnet_student must run first).")
    if verbose:
        print(f"[gptq-init] {len(layers)} BitLinear layers; collecting Hessians over "
              f"{n_calib_batches}x{calib_batch_size} calib sequences...", flush=True)

    H = _collect_hessians(student, layers, calib_iter, device,
                          n_calib_batches=n_calib_batches, calib_batch_size=calib_batch_size,
                          hessian_device=hdev)

    ratios = torch.linspace(scale_lo, scale_hi, n_scale_grid).tolist()
    if 1.0 not in ratios:                                 # absmean (ratio=1) must be reachable
        ratios.append(1.0)

    per_layer = {}
    tot_abs, tot_ws = 0.0, 0.0
    for name, module in layers.items():
        Hn = H[name]
        if Hn is None:
            if verbose:
                print(f"[gptq-init]   {name}: no calibration activations captured; skipped.")
            continue
        orig_dtype = module.weight.dtype
        # Per-layer solve on the compute device (GPU): only ONE Hessian resident at a time,
        # so the O(n^3) Cholesky/scale-search runs on cuSOLVER instead of crawling on CPU.
        W0 = module.weight.detach().to(device, torch.float32)
        Hn = Hn.to(device)

        base_alpha = float(W0.abs().mean().clamp(min=1e-5))

        # --- absmean baseline (what the default init/forward would do) ---
        T_abs = _rtn_ternary(W0, base_alpha)
        mse_abs = _recon_mse_from_H(W0 - base_alpha * T_abs, Hn)

        # --- (2) per-tensor optimal scale search (RTN); ratio=1 included => <= absmean ---
        best_alpha, best_T, best_mse = base_alpha, T_abs, mse_abs
        if scale_search:
            for r in ratios:
                a = base_alpha * float(r)
                if a <= 0.0:
                    continue
                T = _rtn_ternary(W0, a)
                mse = _recon_mse_from_H(W0 - a * T, Hn)
                if mse < best_mse:
                    best_alpha, best_T, best_mse = a, T, mse
        method = "rtn+scale"

        # --- (3) GPTQ error-feedback at the chosen scale; keep only if it helps ---
        if use_gptq:
            try:
                T_g = _gptq_ternary(W0, Hn, best_alpha, percdamp=percdamp)
                mse_g = _recon_mse_from_H(W0 - best_alpha * T_g, Hn)
                if mse_g < best_mse:
                    best_T, best_mse, method = T_g, mse_g, "gptq"
            except Exception as e:  # Cholesky failure on a degenerate H: fall back to RTN.
                if verbose:
                    print(f"[gptq-init]   {name}: GPTQ skipped ({type(e).__name__}: {e}).")

        # --- (4) guaranteed non-regression vs absmean (by construction) ---
        assert best_mse <= mse_abs + 1e-6 * (abs(mse_abs) + 1.0), \
            f"{name}: warm-start MSE {best_mse:.6g} > absmean {mse_abs:.6g}"

        # --- (5) write a master weight that reproduces (best_alpha, best_T) ---
        W_store = _store_weight_for_target(best_T, best_alpha)
        # Verify the forward's own quantizer reproduces the intended solution.
        repro = _weight_quant_ternary(W_store)
        target = best_alpha * best_T
        assert torch.allclose(repro, target, atol=1e-4, rtol=1e-3), \
            f"{name}: reproduction lemma failed (max err {float((repro - target).abs().max()):.3e})"

        module.weight.data.copy_(W_store.to(orig_dtype).to(module.weight.device))

        density = float((best_T != 0).to(torch.float32).mean())
        per_layer[name] = {
            "mse_absmean": mse_abs, "mse_warmstart": best_mse,
            "alpha": best_alpha, "ratio": best_alpha / base_alpha,
            "method": method, "density": density,
        }
        tot_abs += mse_abs
        tot_ws += best_mse
        if verbose:
            red = 100.0 * (1.0 - best_mse / mse_abs) if mse_abs > 0 else 0.0
            print(f"[gptq-init]   {name:<40s} absmean={mse_abs:.4e} -> "
                  f"warmstart={best_mse:.4e} ({red:+.1f}%, {method}, α×{best_alpha/base_alpha:.2f})",
                  flush=True)

    reduction = 100.0 * (1.0 - tot_ws / tot_abs) if tot_abs > 0 else 0.0
    assert tot_ws <= tot_abs + 1e-6 * (abs(tot_abs) + 1.0), \
        f"aggregate warm-start MSE {tot_ws:.6g} > absmean {tot_abs:.6g}"
    if verbose:
        print(f"[gptq-init] DONE {len(per_layer)} layers | absmean MSE={tot_abs:.4e} -> "
              f"warm-start MSE={tot_ws:.4e}  ({reduction:+.1f}%)", flush=True)

    return {"per_layer": per_layer, "mse_absmean": tot_abs, "mse_warmstart": tot_ws,
            "mse_reduction": reduction, "n_layers": len(per_layer)}


# ===========================================================================
# Self-test — tiny synthetic Qwen3, no download.
# ===========================================================================
def _run_self_test() -> bool:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bitdistill as bd  # tiny-model builder + student converter live here

    print("[self-test] building tiny synthetic Qwen3 + BitDistill student...")
    teacher = bd.build_tiny_qwen3()
    import copy
    student = copy.deepcopy(teacher)
    # perturb weights so ternarization is non-degenerate
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    info = bd.convert_to_bitnet_student(student)
    student.eval()
    print(f"[self-test] converted {info['bitlinears']} BitLinears, {info['subnorms']} SubLNs")

    vocab = student.config.vocab_size
    calib = bd_tiny_stream(vocab, seq_len=16)

    stats = gptq_warmstart_init(student, calib, torch.device("cpu"),
                                n_calib_batches=6, calib_batch_size=2, verbose=True)

    assert stats["n_layers"] == info["bitlinears"], "every BitLinear should be initialized"
    assert stats["mse_warmstart"] <= stats["mse_absmean"] + 1e-9, \
        "warm-start MSE must not exceed absmean MSE"
    strict_wins = sum(1 for v in stats["per_layer"].values()
                      if v["mse_warmstart"] < v["mse_absmean"] * (1 - 1e-6))
    print(f"[self-test] layers strictly improved over absmean: "
          f"{strict_wins}/{stats['n_layers']}")
    assert strict_wins >= 1, "at least one layer must strictly beat absmean"

    # A converted+initialized student must still run a forward at full ternary.
    bd.set_quant_alpha(student, 1.0)
    with torch.no_grad():
        ids = torch.randint(0, vocab, (1, 12))
        out = student(input_ids=ids, use_cache=False).logits
    assert torch.isfinite(out).all(), "post-init ternary forward must be finite"
    print(f"[self-test] post-init ternary forward OK; total MSE "
          f"{stats['mse_absmean']:.4e} -> {stats['mse_warmstart']:.4e} "
          f"({stats['mse_reduction']:+.1f}%)")
    print("[self-test] PASS")
    return True


def bd_tiny_stream(vocab_size: int, seq_len: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, vocab_size, (seq_len,), generator=g, dtype=torch.long)


# ===========================================================================
# Real smoke — a few BitLinear layers of the actual Qwen3-0.6B.
# ===========================================================================
def _run_real(n_layers: int, device: str) -> bool:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bitdistill as bd
    import bitdistill_data as bdata
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = "Qwen/Qwen3-0.6B"
    dev = torch.device(device)
    dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
    print(f"[real] loading {base} ({dtype})...")
    tok = AutoTokenizer.from_pretrained(base)
    student = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev)
    student.config.use_cache = False
    bd.convert_to_bitnet_student(student)
    student.to(dev)

    # Restrict to the first n_layers decoder blocks (fast; enough to show the win).
    prefixes = tuple(f"model.layers.{i}." for i in range(n_layers))
    lf = lambda name: name.startswith(prefixes)  # noqa: E731

    calib = bdata.cpt_token_stream(tok, seq_len=256, local_parquet=None)
    # PPL slice BEFORE (absmean init) vs AFTER (warm-start) on the affected layers.
    ppl_seqs = bdata.load_ppl_slice(tok, n=8, seq_len=256)
    bd.set_quant_alpha(student, 1.0)
    ppl_before = bd.compute_ppl(student, ppl_seqs, dev)

    stats = gptq_warmstart_init(student, calib, dev, n_calib_batches=8, calib_batch_size=2,
                                hessian_device="cpu" if dev.type == "cpu" else None,
                                layer_filter=lf, verbose=True)

    bd.set_quant_alpha(student, 1.0)
    ppl_after = bd.compute_ppl(student, ppl_seqs, dev)
    print(f"[real] layers initialized: {stats['n_layers']} "
          f"(first {n_layers} blocks)")
    print(f"[real] recon MSE  absmean={stats['mse_absmean']:.4e} -> "
          f"warmstart={stats['mse_warmstart']:.4e} ({stats['mse_reduction']:+.1f}%)")
    print(f"[real] tiny-slice PPL (ternary)  before={ppl_before:.3f}  after={ppl_after:.3f}  "
          f"(delta {ppl_after - ppl_before:+.3f})")
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gptq_init.py",
                                description="GPTQ-style warm-start init for the BitLinear ternary student.")
    p.add_argument("--self-test", action="store_true", help="Tiny synthetic Qwen3 numeric check (no download).")
    p.add_argument("--real", action="store_true", help="Smoke on the actual Qwen3-0.6B (a few layers).")
    p.add_argument("--n-layers", type=int, default=2, dest="n_layers", help="Decoder blocks to init in --real.")
    p.add_argument("--device", default="cpu", help="cpu | cuda ...")
    args = p.parse_args(argv)
    if args.real:
        return 0 if _run_real(args.n_layers, args.device) else 1
    return 0 if _run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
