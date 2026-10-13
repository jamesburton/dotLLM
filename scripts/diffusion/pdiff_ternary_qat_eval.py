#!/usr/bin/env python3
"""pdiff_ternary_qat_eval.py — Evaluate a QAT-recovered ternary model on no_robots PPL.

Loads a model saved by pdiff_ternary_qat.py (via save_pretrained) and evaluates
perplexity on HuggingFaceH4/no_robots test split.

CORRECT EVAL PATH (no double-quant):
    - AutoModelForCausalLM.from_pretrained loads bf16 master weights (updated by QAT)
    - NO PEFT adapter loaded
    - NO explicit absmean_ternarize() pre-bake
    - AutoBitLinear.forward() ternarizes bf16 master ONCE per forward (online_quant=True)
    - This is the same single-ternarization path used in production inference
    - PPL reflects the true ternary performance of the QAT-adapted model

COMPARISON MATRIX (all on no_robots test, n=100):
    base              : PPL ~13.72   (base model, --model-path none)
    bf16 gen adapter  : PPL ~12.41   (PEFT forward, 93% of real adapter gain)
    bf16 real adapter : PPL ~12.3-12.4
    QAT-gen           : this script  --model-path .docs/diffusion/qat_out/gen_adapter
    QAT-real          : this script  --model-path .docs/diffusion/qat_out/adapter_037

CLI:
    # Baseline (base model via HF cache):
    python scripts/diffusion/pdiff_ternary_qat_eval.py --model-path none --device cuda

    # QAT-trained generated adapter:
    python scripts/diffusion/pdiff_ternary_qat_eval.py \\
        --model-path .docs/diffusion/qat_out/gen_adapter \\
        --device cuda

    # QAT-trained real adapter:
    python scripts/diffusion/pdiff_ternary_qat_eval.py \\
        --model-path .docs/diffusion/qat_out/adapter_037 \\
        --device cuda

QUEUED — run on GPU when free; do NOT run full PPL on CPU (minutes per sequence).
For CPU-only validation, use pdiff_ternary_qat.py --tiny-random smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent

# Reuse the PPL compute loop and no_robots formatter from the existing eval script.
sys.path.insert(0, str(_SCRIPT_DIR))
from pdiff_func_eval import compute_ppl, format_example  # noqa: E402

BASE_MODEL_ID      = "microsoft/bitnet-b1.58-2B-4T-bf16"
_DEFAULT_DATASET   = "HuggingFaceH4/no_robots"
_DEFAULT_EVAL_SPLIT = "test"

# Reference PPL values — measured on HuggingFaceH4/no_robots test split, n=100.
# Only printed when evaluating on no_robots; for other datasets the raw PPL is
# printed with a note that these references do not apply.
_NO_ROBOTS_PPL_REF = {
    "base":         13.72,
    "bf16_gen":     12.41,
    "bf16_real":    12.35,   # midpoint of reported 12.3-12.4 range
}


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a QAT-recovered ternary model (from pdiff_ternary_qat.py) on "
            "HuggingFaceH4/no_robots test PPL.  "
            "AutoBitLinear ternarizes bf16 master ONCE at forward (no double-quant).  "
            "QUEUED — run only when GPU is free."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model-path", default="none",
        help=(
            "Local save_pretrained directory from pdiff_ternary_qat.py, "
            "or 'none' to evaluate the plain base model (baseline from HF cache)."
        ),
    )
    ap.add_argument(
        "--dataset", default=_DEFAULT_DATASET, metavar="DATASET_ID",
        help=(
            "HuggingFace dataset ID for PPL evaluation.  "
            "Default: HuggingFaceH4/no_robots.  "
            "Also supported: iamtarun/python_code_instructions_18k_alpaca, "
            "TIGER-Lab/MathInstruct.  "
            "When evaluating on a non-no_robots dataset the printed reference "
            "PPL values (base/bf16) are labeled as no_robots-specific and "
            "should not be compared directly."
        ),
    )
    ap.add_argument(
        "--eval-split", default=_DEFAULT_EVAL_SPLIT, metavar="SPLIT",
        help=(
            "Dataset split to evaluate on.  Default: 'test'.  "
            "Use HF slice notation, e.g. 'test[:100]'."
        ),
    )
    ap.add_argument(
        "--n-seqs", type=int, default=100,
        help="Number of dataset examples to evaluate.",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Compute device (cuda required for real eval; cpu for structure checks).",
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Maximum token length (truncate longer sequences).",
    )
    args = ap.parse_args(argv)

    device     = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available.  Falling back to CPU.", file=sys.stderr)
        device = "cpu"

    model_path  = args.model_path.strip()
    is_base     = model_path.lower() == "none"
    load_path   = BASE_MODEL_ID if is_base else model_path
    label       = "BASE" if is_base else Path(model_path).name
    eval_dataset = args.dataset
    eval_split   = args.eval_split
    is_no_robots = (eval_dataset == _DEFAULT_DATASET)

    print("=" * 60)
    print("pdiff_ternary_qat_eval: QAT-native ternary PPL")
    print("=" * 60)
    print(f"  model path     : {load_path}")
    print(f"  eval dataset   : {eval_dataset}  split={eval_split}")
    print(f"  n-seqs         : {args.n_seqs}")
    print(f"  device         : {device}")
    print(f"  EVAL METHOD    : AutoBitLinear ternarizes bf16 master ONCE at forward")
    print(f"                   (no pre-bake, no double-quant)")
    if not is_no_robots:
        print(f"  NOTE           : reference PPL values (base/bf16) are no_robots-specific")
        print(f"                   and will NOT be printed for this dataset.")
    print()

    # ------------------------------------------------------------------
    # Lazy heavy imports
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tok_path  = BASE_MODEL_ID if is_base else load_path
    local_only = not is_base
    print(f"Loading tokenizer from {tok_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=local_only)
    # Fall back to base tokenizer if QAT save_pretrained didn't include one.
    if is_base is False:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        except Exception:
            print("  Tokenizer not found in model dir; loading from base model.")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model: load bf16 master weights; AutoBitLinear ternarizes at forward
    # ------------------------------------------------------------------
    print(f"Loading model from {load_path} ...")
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            local_files_only=local_only,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=local_only,
        )
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters loaded")

    # ------------------------------------------------------------------
    # Show QAT metadata if available
    # ------------------------------------------------------------------
    if not is_base:
        meta_path = Path(model_path) / "qat_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            print(
                f"  QAT info: steps={meta.get('steps')}  lr={meta.get('lr')}  "
                f"adapter={meta.get('adapter')}  "
                f"init_loss={meta.get('init_loss', float('nan')):.4f}  "
                f"final_loss={meta.get('final_loss', float('nan')):.4f}"
            )

    model.eval()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"Loading {eval_dataset} (split={eval_split}, n={args.n_seqs}) ...")
    ds    = load_dataset(eval_dataset, split=eval_split, trust_remote_code=True)
    n_take = min(args.n_seqs, len(ds))
    texts  = [_extract_text(ex) for ex in ds.select(range(n_take))]
    texts  = [t for t in texts if t]   # drop empty
    print(f"  {len(texts)} examples")

    # ------------------------------------------------------------------
    # PPL computation
    # ------------------------------------------------------------------
    print("Computing PPL  (AutoBitLinear: one ternarize per forward) ...")
    metrics = compute_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_seq_len=args.max_seq_len,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    ppl = metrics["ppl"]

    print()
    print("=" * 60)
    print(f"Result [{label}]  dataset={eval_dataset}  split={eval_split}")
    print(f"  PPL      = {ppl:.3f}")
    print(f"  mean NLL = {metrics['mean_nll']:.4f}")
    print(f"  n_seqs   = {metrics['n_sequences']}")
    print(f"  n_tokens = {metrics['n_tokens']}")
    print()
    if is_no_robots:
        print("Reference comparison (no_robots test, n=100):")
        for ref_label, ref_ppl in _NO_ROBOTS_PPL_REF.items():
            delta = ppl - ref_ppl
            arrow = "better" if delta < -0.1 else ("worse" if delta > 0.1 else "≈same")
            print(f"  {ref_label:20s}: PPL {ref_ppl:.2f}  → Δ={delta:+.2f} ({arrow})")
    else:
        print(f"  (Reference PPL values are no_robots-specific; not shown for {eval_dataset}.)")
        print(f"  Compute base/bf16 references for this task by running with --model-path none")
        print(f"  on the same --dataset and --eval-split.")
    print("=" * 60)


if __name__ == "__main__":
    main()
