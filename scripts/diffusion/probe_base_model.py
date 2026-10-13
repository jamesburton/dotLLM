#!/usr/bin/env python3
"""probe_base_model.py — Compatibility probe for the ternary-LoRA wrapper.

Loads a model, prints architecture info (n_layers, hidden_size, class of
layers[0].mlp.gate_proj), wraps the LAST 2 transformer layers with
TernaryLoRAWrapper (cold init, r=8), enables the inference cache, and runs
3-token greedy generation to confirm the full pipeline executes without error.

Designed for the parent process to run on a remote box before committing to a
full 8B training run, confirming that the 8B BitNet arch is compatible with
_wrap_mlp_layers.  Memory use is minimal — only the last 2 MLP layers are wrapped.

Usage:
    # Probe an 8B model on CUDA (run on a machine with enough VRAM):
    python scripts/diffusion/probe_base_model.py \\
        --base-model HF1BitLLM/Llama3-8B-1.58-100B-tokens --device cuda

    # Help only (no model load):
    python scripts/diffusion/probe_base_model.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 on Windows (avoids cp1252 errors from special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Ensure pdiff_ternary_lora is importable from the same directory.
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdiff_ternary_lora import (
    BASE_MODEL_ID,
    _wrap_mlp_layers,
    _set_lora_lambda,
    _set_lora_enabled,
    enable_inference_cache,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compatibility probe for the ternary-LoRA wrapper on any BitNet arch. "
            "Loads the model, prints arch info (n_layers, hidden_size, "
            "class of layers[0].mlp.gate_proj), wraps the LAST 2 transformer "
            "layers' MLP projections (cold init, r=8), enables the inference "
            "cache, and runs 3-token greedy generation to confirm the full "
            "pipeline runs cleanly. No training is performed. "
            "Memory is modest — only last-2-layers are wrapped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--base-model", default=BASE_MODEL_ID, dest="base_model",
        metavar="MODEL_ID",
        help="HuggingFace model ID or local path for the BitNet base model to probe.",
    )
    ap.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load and run the model on.",
    )

    args = ap.parse_args(argv)

    # ------------------------------------------------------------------
    # Deferred imports — only after --help is handled
    # ------------------------------------------------------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)

    print("[probe] ====================================================")
    print(f"[probe] base_model : {args.base_model!r}")
    print(f"[probe] device     : {args.device}")
    print("[probe] ====================================================")

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer
    # ------------------------------------------------------------------
    print(f"\n[probe] Loading model {args.base_model!r} ...")
    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if device.type != "cpu":
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if device.type == "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Print architecture info
    # ------------------------------------------------------------------
    n_layers     = len(model.model.layers)
    hidden       = model.config.hidden_size
    gate_proj    = getattr(model.model.layers[0].mlp, "gate_proj", None)
    gate_cls     = type(gate_proj).__name__ if gate_proj is not None else "N/A"
    in_feat      = getattr(gate_proj, "in_features",  "N/A")
    out_feat     = getattr(gate_proj, "out_features", "N/A")
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n[probe] Architecture:")
    print(f"  model class     = {type(model).__name__}")
    print(f"  n_layers        = {n_layers}")
    print(f"  hidden_size     = {hidden}")
    print(f"  total params    = {total_params:,}")
    print(f"  gate_proj class = {gate_cls}")
    print(f"  in_features     = {in_feat}")
    print(f"  out_features    = {out_feat}")

    if gate_proj is None:
        print("[probe] WARNING: layers[0].mlp.gate_proj not found — "
              "arch may differ from expected Llama/BitNet MLP structure")

    # ------------------------------------------------------------------
    # 3. Wrap last 2 layers (cold init, r=8)
    # ------------------------------------------------------------------
    last2 = list(range(max(0, n_layers - 2), n_layers))
    print(f"\n[probe] Wrapping last 2 layers (indices {last2}), "
          f"r=8, alpha=16.0 (cold init) ...")
    wrapped_keys = _wrap_mlp_layers(
        model=model,
        target_layer_indices=last2,
        r=8,
        alpha=16.0,
        adapter_weights=None,   # cold init: Kaiming A, zeros B
    )
    print(f"[probe] Wrapped {len(wrapped_keys)} projections:")
    for k in wrapped_keys:
        print(f"  {k}")

    if not wrapped_keys:
        print("[probe] FAIL: no projections were wrapped — "
              "check MLP structure (expected gate_proj/up_proj/down_proj)")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Enable inference cache + 3-token greedy generation
    # ------------------------------------------------------------------
    _set_lora_lambda(1.0)
    _set_lora_enabled(True)
    model.eval()
    model.config.use_cache = True

    n_cached = enable_inference_cache(model)
    print(f"\n[probe] Inference cache: {n_cached} BitNet linear modules patched")

    print("[probe] Running 3-token greedy generation ...")
    input_ids = tokenizer(
        "Hello", return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"[probe] generated (3 tokens): {generated_text!r}")

    # ------------------------------------------------------------------
    # 5. Final report
    # ------------------------------------------------------------------
    print()
    print("[probe] ====================================================")
    print("[probe] PROBE PASSED")
    print(f"[probe]   n_layers={n_layers}  hidden={hidden}")
    print(f"[probe]   gate_proj_class={gate_cls}")
    print(f"[probe]   in_features={in_feat}  out_features={out_feat}")
    print(f"[probe]   n_wrapped={len(wrapped_keys)}  n_cached={n_cached}")
    print(f"[probe]   generated={generated_text!r}")
    print("[probe] ====================================================")


if __name__ == "__main__":
    main()
