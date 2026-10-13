#!/usr/bin/env python3
"""pdiff_kd_corpus.py — Materialize a fixed, tokenized no_robots-train corpus for KD-QAT.

Produces two files consumed by pdiff_teacher_cache.py (teacher logit dump) and
pdiff_ternary_qat.py --kd-corpus (student KD-QAT):

  <out>/ids.npy   -- int32 [N_SEQS, SEQ_LEN]  tokenized, right-padded with pad_id
  <out>/meta.json -- {"tokenizer", "seq_len", "n_seqs", "pad_id", "mix", "seed"}

Row order is deterministic: HuggingFaceH4/no_robots train is loaded sequentially,
chunked into fixed-length sequences, then a Fisher-Yates shuffle (--seed, default 42)
is applied before writing. The seed is recorded in meta.json.

BOTH pdiff_teacher_cache.py and pdiff_ternary_qat.py --kd-corpus MUST receive the
same ids.npy directory so their row indices align.

CLI:
    # Tiny corpus for CPU smoke test (8 seqs x 64 tokens):
    python scripts/diffusion/pdiff_kd_corpus.py \\
        --n-seqs 8 --seq-len 64 --out E:/kd/corpus_smoke

    # Production corpus (2048 seqs x 512 tokens):
    python scripts/diffusion/pdiff_kd_corpus.py \\
        --n-seqs 2048 --seq-len 512 --out E:/kd/corpus_no_robots

"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ensure UTF-8 stdout/stderr on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_DEFAULT_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"
_DEFAULT_SEQ_LEN = 512
_DEFAULT_N_SEQS = 2048
_DEFAULT_SEED = 42
_DATASET = "HuggingFaceH4/no_robots"


def _get_pad_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _build_no_robots_sequences(tokenizer, n_seqs: int, seq_len: int) -> list[list[int]]:
    """Load HuggingFaceH4/no_robots train, chunk into fixed-length sequences."""
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(_DATASET, split="train")
    all_ids: list[int] = []
    for row in ds:
        msgs = row.get("messages", [])
        text = " ".join(m.get("content", "") for m in msgs if m.get("content"))
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= (n_seqs + 10) * seq_len:
            break

    seqs: list[list[int]] = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(all_ids[i : i + seq_len])
        if len(seqs) >= n_seqs + 10:  # gather a few extra for shuffle headroom
            break
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Materialize a fixed KD corpus from HuggingFaceH4/no_robots train. "
            "Writes ids.npy (int32 [N_SEQS, SEQ_LEN]) + meta.json. "
            "Both pdiff_teacher_cache.py and pdiff_ternary_qat.py --kd-corpus "
            "consume the same directory so row indices align."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output corpus directory.")
    ap.add_argument(
        "--n-seqs",
        type=int,
        default=_DEFAULT_N_SEQS,
        help="Number of fixed-length sequences to produce.",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=_DEFAULT_SEQ_LEN,
        help="Token sequence length; sequences are right-padded to this.",
    )
    ap.add_argument(
        "--base",
        default=_DEFAULT_BASE,
        help="HF model id (or local path) to load the tokenizer from.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Fisher-Yates shuffle seed. Recorded in meta.json.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    print(f"[pdiff_kd_corpus] loading tokenizer from {args.base!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = _get_pad_id(tokenizer)
    print(
        f"[pdiff_kd_corpus] tokenizer: vocab_size={tokenizer.vocab_size}  pad_id={pad_id}"
    )

    # ------------------------------------------------------------------
    # 2. Load no_robots train and chunk into fixed-length sequences
    # ------------------------------------------------------------------
    print(
        f"[pdiff_kd_corpus] loading {_DATASET} train  "
        f"target={args.n_seqs} seqs × {args.seq_len} tokens ..."
    )
    seqs = _build_no_robots_sequences(tokenizer, n_seqs=args.n_seqs, seq_len=args.seq_len)
    if not seqs:
        raise RuntimeError(
            f"No sequences returned from {_DATASET}. "
            "Check dataset availability (HF_HUB_OFFLINE may be set)."
        )
    print(f"[pdiff_kd_corpus] loaded {len(seqs)} sequences before shuffle/trim")

    # ------------------------------------------------------------------
    # 3. Deterministic Fisher-Yates shuffle + trim to n_seqs
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    seqs = [seqs[i] for i in indices][: args.n_seqs]
    actual_n = len(seqs)
    if actual_n < args.n_seqs:
        print(
            f"[pdiff_kd_corpus] WARNING: requested {args.n_seqs} seqs "
            f"but only {actual_n} available after loading."
        )

    # ------------------------------------------------------------------
    # 4. Build ids array [N, L] int32, right-padded with pad_id
    # ------------------------------------------------------------------
    ids = np.full((actual_n, args.seq_len), pad_id, dtype=np.int32)
    for i, seq in enumerate(seqs):
        arr = np.array(seq, dtype=np.int32)
        n = min(len(arr), args.seq_len)
        ids[i, :n] = arr[:n]

    # ------------------------------------------------------------------
    # 5. Write ids.npy
    # ------------------------------------------------------------------
    ids_path = os.path.join(args.out, "ids.npy")
    np.save(ids_path, ids)
    print(f"\n[pdiff_kd_corpus] ids.npy: shape={ids.shape}  dtype={ids.dtype}")
    print(f"  -> {ids_path}")

    # ------------------------------------------------------------------
    # 6. Write meta.json
    # ------------------------------------------------------------------
    meta = {
        "tokenizer": args.base,
        "seq_len": args.seq_len,
        "n_seqs": actual_n,
        "pad_id": pad_id,
        "mix": "no_robots_train",
        "seed": args.seed,
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[pdiff_kd_corpus] meta.json: {json.dumps(meta)}")
    print(f"  -> {meta_path}")

    # ------------------------------------------------------------------
    # 7. Sample decoded rows to confirm tokenization
    # ------------------------------------------------------------------
    print(f"\n[pdiff_kd_corpus] sample decoded rows (first 60 tokens each):")
    for i in range(min(4, actual_n)):
        row_ids = [t for t in ids[i].tolist() if t != pad_id][:60]
        decoded = tokenizer.decode(row_ids, skip_special_tokens=True)
        preview = repr(decoded[:100])
        try:
            print(f"  row[{i:3d}] {preview}")
        except UnicodeEncodeError:
            safe = preview.encode(sys.stdout.encoding or "ascii", errors="replace").decode(
                sys.stdout.encoding or "ascii"
            )
            print(f"  row[{i:3d}] {safe}")

    print(
        f"\n[pdiff_kd_corpus] DONE: "
        f"{actual_n} seqs × {args.seq_len} tokens  "
        f"pad_id={pad_id}  seed={args.seed}"
    )
    print(f"  ids.npy shape: {ids.shape}  dtype=int32")
    print(
        "  Alignment contract: pass this --out to both pdiff_teacher_cache.py "
        "and pdiff_ternary_qat.py --kd-corpus so rows stay aligned."
    )


if __name__ == "__main__":
    main()
