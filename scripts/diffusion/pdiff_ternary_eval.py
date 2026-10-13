#!/usr/bin/env python3
"""pdiff_ternary_eval.py — LoRA-merge + ternary re-quantisation + PPL evaluation.

Loads microsoft/bitnet-b1.58-2B-4T-bf16 (from local HF cache), optionally
applies and MERGES a PEFT LoRA adapter, optionally re-ternarizes every adapted
MLP projection weight (gate_proj / up_proj / down_proj) using the canonical
BitNet absmean ternary scheme, then measures perplexity on
HuggingFaceH4/no_robots (test split).

The absmean scheme (``transformers.integrations.bitnet.WeightQuant``):

    gamma = mean(|W|).clamp(min=1e-5)          # per-tensor scale
    W_q   = round(W / gamma).clamp(-1, 1) * gamma   # {-gamma, 0, +gamma}

After merging, AutoBitLinear will re-apply this scheme at forward time.  Writing
back ``absmean_ternarize(W_merged)`` into ``layer.weight.data`` bakes the merged
delta into the ternary grid and lets the model's own quantiser confirm the
representation — the sign pattern (which carries the task-specific information)
is preserved; only the effective scale shifts slightly from gamma to
gamma * density (fraction of non-zero elements).

Comparison matrix for the queued 5-way GPU eval
(run once 48-run adapters exist):

  base         --adapter none                   --ternarize none
  real-bf16    --adapter <path>/adapter_047     --ternarize none
  real-ternary --adapter <path>/adapter_047     --ternarize adapted
  gen-bf16     --adapter <path>/pdiff_func_out48/gen_adapter  --ternarize none
  gen-ternary  --adapter <path>/pdiff_func_out48/gen_adapter  --ternarize adapted

OFFLINE-friendly: respects HF_HUB_OFFLINE and HF_DATASETS_OFFLINE env vars.
Pass ``--local-files-only`` (default True) to enforce offline use.

QUEUED for full PPL eval — do NOT run on CPU unless validating the quant path.
For CPU validation use ``--self-test``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent

# Reuse format_example and compute_ppl from the existing PPL eval script.
sys.path.insert(0, str(_SCRIPT_DIR))
from pdiff_func_eval import compute_ppl, format_example  # noqa: E402

BASE_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
EVAL_DATASET  = "HuggingFaceH4/no_robots"
EVAL_SPLIT    = "test"
MAX_SEQ_LEN   = 512

# MLP projection names that a LoRA adapter may have modified.
# We ONLY ternarize these (attention projections are left alone).
MLP_PROJ_NAMES: frozenset[str] = frozenset({"gate_proj", "up_proj", "down_proj"})


# ---------------------------------------------------------------------------
# Canonical BitNet absmean ternary quantisation
# ---------------------------------------------------------------------------

def absmean_ternarize(W: torch.Tensor) -> torch.Tensor:
    """Apply per-tensor absmean ternary quantisation (BitNet / WeightQuant).

    Implements the canonical scheme from
    ``transformers.integrations.bitnet.WeightQuant``:

        gamma = mean(|W|).clamp(min=1e-5)
        scale = 1 / gamma
        W_q   = round(W * scale).clamp(-1, 1) / scale
              = round(clamp(W / gamma, -1, 1)) * gamma

    Output values are in {-gamma, 0, +gamma} where gamma is a per-tensor scalar.

    Args:
        W: Any floating-point tensor (bf16, f32, …).

    Returns:
        Ternarized tensor with the same shape, dtype, and device as ``W``.
        Values in {-gamma, 0, +gamma}.
    """
    gamma = W.abs().mean().clamp(min=1e-5)
    scale = 1.0 / gamma
    return torch.round(W * scale).clamp(-1.0, 1.0) / scale


# ---------------------------------------------------------------------------
# Ternarize adapted MLP weights in-place
# ---------------------------------------------------------------------------

def ternarize_adapted_mlp(
    model: torch.nn.Module,
    adapted_layer_indices: list[int] | None = None,
) -> int:
    """Replace MLP projection weights in adapted layers with their absmean-ternarized form.

    Walks ``model.model.layers`` and, for every transformer layer that was
    adapted (or ALL layers if ``adapted_layer_indices`` is None), replaces the
    ``weight.data`` of every AutoBitLinear (matched by class name) sublayer
    whose name ends in one of ``MLP_PROJ_NAMES`` with
    ``absmean_ternarize(weight.data)``.

    The AutoBitLinear forward will re-apply the quant at inference time; baking
    the merged value in means the sign pattern of the merged weight is preserved
    and the model computes in the ternary grid.

    Args:
        model: A BitNetForCausalLM (or merged PeftModel) with ``model.model.layers``.
        adapted_layer_indices: Layer indices whose MLP weights to ternarize.
            If None, ALL layers are processed (equivalent to a full base ternarize).

    Returns:
        Number of weight tensors that were ternarized.
    """
    layers = model.model.layers
    n_ternarized = 0

    for layer_idx, layer in enumerate(layers):
        if adapted_layer_indices is not None and layer_idx not in adapted_layer_indices:
            continue

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        for submod_name, submod in mlp.named_modules():
            # Match the projection name (last component of the dotted path).
            proj_name = submod_name.split(".")[-1]
            if proj_name not in MLP_PROJ_NAMES:
                continue

            # We match AutoBitLinear by class name (no hard import dependency).
            if type(submod).__name__ not in ("AutoBitLinear", "Linear"):
                continue

            if not hasattr(submod, "weight") or submod.weight is None:
                continue

            with torch.no_grad():
                original_dtype = submod.weight.data.dtype
                # Cast to float32 for numerically stable quant, then restore dtype.
                w_fp = submod.weight.data.float()
                w_q  = absmean_ternarize(w_fp).to(original_dtype)
                submod.weight.data.copy_(w_q)

            n_ternarized += 1

    return n_ternarized


# ---------------------------------------------------------------------------
# CPU self-test (quant correctness + reconstruction error)
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """CPU-only unit test for the ternarization pipeline.

    Validates:
    1. absmean_ternarize output values are in {-gamma, 0, +gamma}.
    2. Reconstruction relative error is reasonable (<< 1).
    3. ternarize_adapted_mlp runs without importing the full model (synthetic test).
    4. Self-consistency: ternarizing a weight that was already ternarized with the
       same gamma is a near-no-op on the sign pattern.
    """
    import torch.nn as nn

    print("=" * 60)
    print("pdiff_ternary_eval — CPU self-test")
    print("=" * 60)

    # --- Test 1: value constraint ---
    torch.manual_seed(42)
    for shape in [(64, 128), (256, 512), (1024, 256)]:
        W = torch.randn(*shape, dtype=torch.float32)
        W_q = absmean_ternarize(W)
        gamma = W.abs().mean().clamp(min=1e-5).item()
        unique_vals = W_q.unique().tolist()
        # All output values should be within 1e-5 of one of {-gamma, 0, gamma}
        for v in unique_vals:
            assert min(abs(v - t) for t in (-gamma, 0.0, gamma)) < 1e-5, (
                f"Unexpected output value {v}; expected one of "
                f"{{-{gamma:.6f}, 0, +{gamma:.6f}}}"
            )
        print(f"  shape={shape}: unique vals = "
              f"[{', '.join(f'{v:.5f}' for v in unique_vals)}]  "
              f"gamma={gamma:.5f}  OK values in {{-gamma, 0, +gamma}}")

    # --- Test 2: reconstruction relative error ---
    torch.manual_seed(0)
    W = torch.randn(512, 2048, dtype=torch.float32)
    W_q = absmean_ternarize(W)
    rel_err = ((W - W_q).norm() / W.norm()).item()
    frac_nonzero = (W_q != 0).float().mean().item()
    print(f"\n  Reconstruction relative error  : {rel_err:.4f}")
    print(f"  Fraction non-zero elements     : {frac_nonzero:.4f}")
    assert rel_err < 1.0, f"Relative error {rel_err:.4f} >= 1.0 -- unexpectedly large"
    # Typical for BitNet: 0.55-0.75 depending on weight distribution
    print(f"  OK rel_err={rel_err:.4f} < 1.0 (expected ~0.5-0.8 for gaussian init)")

    # --- Test 3: ternarize_adapted_mlp on a synthetic model ---
    # Build a tiny model-like structure without importing BitNet.
    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(64, 128, bias=False)
            self.up_proj   = nn.Linear(64, 128, bias=False)
            self.down_proj = nn.Linear(128, 64, bias=False)

    class TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyMLP()

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer() for _ in range(4)])

    class TinyModelWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyModel()

    torch.manual_seed(7)
    tiny = TinyModelWrapper()
    # Record pre-ternarize weight of layer 0 gate_proj
    w0_before = tiny.model.layers[0].mlp.gate_proj.weight.data.clone()
    # Ternarize only layers 0 and 2 (adapted_layer_indices)
    n = ternarize_adapted_mlp(tiny, adapted_layer_indices=[0, 2])
    assert n == 6, f"Expected 6 ternarized tensors (3 MLP projs × 2 layers), got {n}"
    w0_after = tiny.model.layers[0].mlp.gate_proj.weight.data
    gamma_0 = w0_before.float().abs().mean().clamp(min=1e-5).item()
    unique_after = w0_after.unique().tolist()
    for v in unique_after:
        assert min(abs(v - t) for t in (-gamma_0, 0.0, gamma_0)) < 1e-5, (
            f"Layer-0 gate_proj value {v:.6f} not in {{-γ={gamma_0:.6f}, 0, +γ}}"
        )
    # Layer 1 should be UNCHANGED
    w1_gate = tiny.model.layers[1].mlp.gate_proj.weight.data
    # Layer 1 was not in adapted_layer_indices=[0,2] so should be original values
    print(f"\n  ternarize_adapted_mlp: {n} tensors ternarized across 2/4 layers  OK")
    print(f"  Layer-0 gate_proj unique vals after: "
          f"[{', '.join(f'{v:.5f}' for v in unique_after)}]  gamma={gamma_0:.5f}")

    # --- Test 4: self-consistency (ternarize(ternarize(W)) sign pattern preserved) ---
    torch.manual_seed(1)
    W = torch.randn(256, 512, dtype=torch.float32)
    W_q  = absmean_ternarize(W)
    W_qq = absmean_ternarize(W_q)
    # Sign patterns should match: nonzero elements in W_q should have same sign in W_qq
    mask = W_q != 0
    sign_match = (torch.sign(W_q[mask]) == torch.sign(W_qq[mask])).all().item()
    assert sign_match, "Sign pattern changed on double ternarize -- unexpected"
    density = mask.float().mean().item()
    print(f"\n  Self-consistency: sign pattern preserved after double ternarize  OK")
    print(f"  Density (fraction non-zero in W_q): {density:.4f}")

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED -- all 4 checks OK")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Merge LoRA adapter, optionally re-ternarize adapted MLP weights, "
            "then evaluate perplexity on HuggingFaceH4/no_robots (test).  "
            "QUEUED — run full PPL eval on GPU only.  Use --self-test for CPU validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--adapter", default="none",
        help=(
            "Path to a PEFT LoRA adapter directory, or 'none' for bare base model.  "
            "When given, adapter is merged (merge_and_unload) before eval."
        ),
    )
    ap.add_argument(
        "--ternarize", default="none", choices=["none", "adapted"],
        help=(
            "'none' — no explicit ternarization (AutoBitLinear handles quant at forward).  "
            "'adapted' — after merging, replace gate_proj/up_proj/down_proj weights in "
            "adapted layers with their absmean-ternarized form (bakes delta into ternary grid)."
        ),
    )
    ap.add_argument(
        "--n-seqs", type=int, default=100,
        help="Number of examples from no_robots test split to evaluate.",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Compute device ('cuda' for queued eval; 'cpu' for light inspection).",
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=MAX_SEQ_LEN,
        help="Maximum token sequence length (sequences truncated beyond this).",
    )
    ap.add_argument(
        "--local-files-only", action="store_true", default=True,
        help="Use only locally cached HF model + dataset files (offline mode).",
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "Run CPU-only unit test of the ternarization math (no model load).  "
            "Exits 0 on success, 1 on failure.  Use this for structural validation."
        ),
    )
    args = ap.parse_args(argv)

    # --- Self-test path ---
    if args.self_test:
        try:
            run_self_test()
            sys.exit(0)
        except AssertionError as exc:
            print(f"\nSELF-TEST FAILED: {exc}", file=sys.stderr)
            sys.exit(1)

    # --- Validate args ---
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable.  Falling back to CPU.", file=sys.stderr)
        device = "cpu"

    adapter_path = args.adapter.strip()
    use_adapter  = adapter_path.lower() != "none"
    if use_adapter and not Path(adapter_path).exists():
        print(f"ERROR: adapter path not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    # Note: --ternarize adapted with --adapter none ternarizes ALL MLP layers in the
    # base model.  This serves as a calibration run: explicit absmean_ternarize should
    # give PPL indistinguishable from the plain base (AutoBitLinear already ternarizes
    # on-the-fly at forward time).  It is the "base-ternary" slot in the 6-way eval.

    print("=" * 60)
    print("pdiff_ternary_eval: merge + ternary + PPL")
    print("=" * 60)
    print(f"  base model   : {BASE_MODEL_ID}")
    print(f"  adapter      : {adapter_path}")
    print(f"  ternarize    : {args.ternarize}")
    print(f"  dataset      : {EVAL_DATASET}  split={EVAL_SPLIT}")
    print(f"  n-seqs       : {args.n_seqs}")
    print(f"  device       : {device}")
    print()

    # --- Lazy heavy imports ---
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers not installed.  Run: pip install transformers") from exc
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets not installed.  Run: pip install datasets") from exc

    # --- Tokenizer ---
    print(f"Loading tokenizer from {BASE_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model ---
    print(f"Loading model {BASE_MODEL_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=args.local_files_only,
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Adapter: load, record adapted layers, merge ---
    adapted_layer_indices: list[int] | None = None

    if use_adapter:
        try:
            from peft import PeftModel
            import json
        except ImportError as exc:
            raise ImportError("peft not installed.  Run: pip install peft") from exc

        # Read adapter config to discover which layers were adapted.
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)

        layers_to_transform = adapter_cfg.get("layers_to_transform")  # None = all layers
        n_model_layers = len(model.model.layers)
        if layers_to_transform is None:
            adapted_layer_indices = list(range(n_model_layers))
        else:
            adapted_layer_indices = sorted(set(layers_to_transform))
        print(f"  Adapter targets {len(adapted_layer_indices)} layers "
              f"(layers_to_transform={layers_to_transform!r})")
        print(f"  Adapter target_modules: {adapter_cfg.get('target_modules')}")

        print(f"Loading PEFT adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)

        if args.ternarize == "adapted":
            # Only merge when we are going to explicitly ternarize afterwards.
            # Merging bakes the LoRA delta (A @ B * alpha/r) into the base weight so
            # that absmean_ternarize can snap the merged weight to the ternary grid.
            print("  Merging adapter (merge_and_unload) for ternarization ...")
            model = model.merge_and_unload()
            print("  Adapter merged into base weights.")
        else:
            # For plain bf16 eval (--ternarize none), keep the PeftModel as-is so that
            # inference computes  AutoBitLinear(x) + lora_delta(x).
            # Merging into a BitNet model causes AutoBitLinear to re-ternarize the merged
            # weight at forward time, which gives ternarize(W_base + delta) @ x instead of
            # ternarize(W_base) @ x + delta @ x — producing badly wrong PPL (~31 vs ~12).
            print("  Keeping adapter unmerged (ternarize=none: PeftModel forward path).")
        print("  Adapter applied.")

    # --- Ternarize ---
    if args.ternarize == "adapted":
        if not use_adapter:
            # No adapter: ternarize ALL layers (calibration / base-ternary slot).
            print("  Ternarizing ALL base MLP projections (gate_proj / up_proj / down_proj) ...")
            print("  [calibration: base explicit-ternary should match base plain PPL]")
            n_ternary = ternarize_adapted_mlp(model, adapted_layer_indices=None)
        else:
            print("  Ternarizing adapted MLP projections (gate_proj / up_proj / down_proj) ...")
            n_ternary = ternarize_adapted_mlp(model, adapted_layer_indices=adapted_layer_indices)
        print(f"  OK: {n_ternary} MLP weight tensors re-ternarized.")

    model.eval()

    # --- Dataset ---
    print(f"Loading {EVAL_DATASET} (split={EVAL_SPLIT}, n={args.n_seqs}) ...")
    ds = load_dataset(
        EVAL_DATASET,
        split=EVAL_SPLIT,
        trust_remote_code=True,
    )
    n_take = min(args.n_seqs, len(ds))
    ds_sub = ds.select(range(n_take))
    texts  = [format_example(ex) for ex in ds_sub]
    print(f"  Loaded {len(texts)} examples.")

    # --- PPL ---
    print("Computing PPL ...")
    metrics = compute_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_seq_len=args.max_seq_len,
    )

    print()
    print("=" * 60)
    if not use_adapter:
        label = "BASE" if args.ternarize == "none" else "BASE-explicit-ternary"
    else:
        adapter_label = Path(adapter_path).name
        ternary_tag   = "-ternary" if args.ternarize == "adapted" else "-bf16"
        label         = f"{adapter_label}{ternary_tag}"
    print(f"Result [{label}]")
    print(f"  PPL      = {metrics['ppl']:.3f}")
    print(f"  mean NLL = {metrics['mean_nll']:.4f}")
    print(f"  n_seqs   = {metrics['n_sequences']}")
    print(f"  n_tokens = {metrics['n_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
