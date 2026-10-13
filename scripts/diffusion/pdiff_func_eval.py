"""pdiff_func_eval.py — GPU PPL evaluation for base / real / recon / gen adapters.

QUEUED — do NOT run during CPU-only structural validation.
Run only when GPU is free, after assembling adapters with pdiff_func_assemble.py.

Loads microsoft/bitnet-b1.58-2B-4T-bf16 (from local HF cache) and optionally
a PEFT LoRA adapter, then computes mean token NLL and perplexity on a held-out
slice of HuggingFaceH4/no_robots.

Comparison matrix (run once per condition):
    base  : --adapter none
    real  : --adapter .docs/diffusion/pdiff_boot_fixedinit/adapter_015
    recon : --adapter .docs/diffusion/pdiff_func_out/recon_adapter
    gen   : --adapter .docs/diffusion/pdiff_func_out/gen_adapter

Usage (GPU):
    python pdiff_func_eval.py --adapter none           --device cuda
    python pdiff_func_eval.py --adapter <real_dir>     --device cuda
    python pdiff_func_eval.py --adapter <recon_dir>    --device cuda
    python pdiff_func_eval.py --adapter <gen_dir>      --device cuda
"""

import argparse
import math
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent

BASE_MODEL_ID  = "microsoft/bitnet-b1.58-2B-4T-bf16"
EVAL_DATASET   = "HuggingFaceH4/no_robots"
EVAL_SPLIT     = "test"
MAX_SEQ_LEN    = 512


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def format_example(example: dict) -> str:
    """Format a no_robots example as a single text string.

    Concatenates all message content fields in order (system, user, assistant).
    """
    parts = []
    messages = example.get("messages", [])
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ppl(
    model,
    tokenizer,
    texts:        list,
    device:       str,
    max_seq_len:  int = MAX_SEQ_LEN,
    batch_size:   int = 1,
) -> dict:
    """Compute mean token NLL and perplexity over a list of text strings.

    Returns:
        dict with 'ppl', 'mean_nll', 'n_sequences', 'n_tokens'.
    """
    model.eval()
    total_nll    = 0.0
    total_tokens = 0
    n_seqs       = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        n_tok     = input_ids.shape[1]
        if n_tok < 2:
            continue   # skip trivially short sequences

        out  = model(input_ids=input_ids, labels=input_ids)
        # out.loss is mean NLL per token (reduction='mean')
        nll  = float(out.loss.item())
        # accumulate weighted by (n_tok - 1) since the last token has no label
        total_nll    += nll * (n_tok - 1)
        total_tokens += n_tok - 1
        n_seqs       += 1

    if total_tokens == 0:
        return {"ppl": float("nan"), "mean_nll": float("nan"), "n_sequences": 0, "n_tokens": 0}

    mean_nll = total_nll / total_tokens
    ppl      = math.exp(mean_nll)
    return {
        "ppl":          ppl,
        "mean_nll":     mean_nll,
        "n_sequences":  n_seqs,
        "n_tokens":     total_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "GPU PPL evaluation: load base model + optional PEFT adapter, "
            "compute perplexity on HuggingFaceH4/no_robots test slice.  "
            "QUEUED — run only when GPU is free."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--adapter", default="none",
        help=(
            "Path to a PEFT adapter directory, or 'none' to evaluate the bare base model.  "
            "Compare: none (base), real adapter_015, recon_adapter/, gen_adapter/."
        ),
    )
    ap.add_argument(
        "--n-seqs", type=int, default=100,
        help="Number of examples from no_robots test split to evaluate.",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Compute device (cuda required; script will warn if GPU unavailable).",
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=MAX_SEQ_LEN,
        help="Maximum token sequence length (sequences truncated beyond this).",
    )
    ap.add_argument(
        "--local-files-only", action="store_true", default=True,
        help="Use only locally cached HF model files (default: True).",
    )
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available.  Falling back to CPU.", file=sys.stderr)
        device = "cpu"

    adapter_path = args.adapter.strip()
    if adapter_path.lower() != "none" and not Path(adapter_path).exists():
        print(f"ERROR: adapter path not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("pdiff_func_eval: PPL evaluation")
    print("=" * 60)
    print(f"  base model   : {BASE_MODEL_ID}")
    print(f"  adapter      : {adapter_path}")
    print(f"  dataset      : {EVAL_DATASET}  split={EVAL_SPLIT}")
    print(f"  n-seqs       : {args.n_seqs}")
    print(f"  device       : {device}")
    print()

    # --- Lazy imports (heavy; only at run time) ---
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers not installed.  Run: pip install transformers") from exc

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets not installed.  Run: pip install datasets") from exc

    # --- Load tokenizer ---
    print(f"Loading tokenizer from {BASE_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base model ---
    print(f"Loading model {BASE_MODEL_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=args.local_files_only,
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Apply adapter (if given) ---
    if adapter_path.lower() != "none":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft not installed.  Run: pip install peft") from exc

        print(f"Loading PEFT adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("  Adapter applied.")

    model.eval()

    # --- Load dataset ---
    print(f"Loading {EVAL_DATASET} (split={EVAL_SPLIT}, n={args.n_seqs}) ...")
    ds = load_dataset(
        EVAL_DATASET,
        split=EVAL_SPLIT,
        trust_remote_code=True,
    )
    # Take first n_seqs examples
    n_take  = min(args.n_seqs, len(ds))
    ds_sub  = ds.select(range(n_take))
    texts   = [format_example(ex) for ex in ds_sub]
    print(f"  Loaded {len(texts)} examples.")

    # --- Compute PPL ---
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
    label = "BASE" if adapter_path.lower() == "none" else Path(adapter_path).name
    print(f"Result [{label}]")
    print(f"  PPL      = {metrics['ppl']:.3f}")
    print(f"  mean NLL = {metrics['mean_nll']:.4f}")
    print(f"  n_seqs   = {metrics['n_sequences']}")
    print(f"  n_tokens = {metrics['n_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
