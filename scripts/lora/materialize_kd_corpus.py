"""Materialize a fixed, deterministic KD corpus for offline teacher-logit caching.

Produces two files consumed by strix_teacher_dump.py (teacher) and the MoTE
trainer's --kd-corpus flag (student):

  <out>/ids.npy   -- int32 [N_SEQS, SEQ_LEN]  tokenized, right-padded with pad_id
  <out>/meta.json -- {"tokenizer", "seq_len", "n_seqs", "pad_id", "mix"}

The corpus uses the SAME tokenizer and the SAME data sources as mote_train.py:
  * ``--mix ja_en`` -> load_mixed_ja_en_sequences from domain_data.py
    (~60% wiki_ja + ~40% no_robots train, interleaved 3:2)

Row order is deterministic: data loaders produce a fixed interleaved order;
an additional Fisher-Yates shuffle (``--seed``, default 42) is applied before
writing, and the seed is recorded in meta.json so the trainer can reproduce the
same row mapping independently.

Usage
-----
  # build a 4096-seq corpus (CPU-light, ~seconds)
  python scripts/lora/materialize_kd_corpus.py \\
      --mix ja_en --n-seqs 4096 --seq-len 512 \\
      --out .docs/kd/corpus_ja_en

  # smoke: 4 seqs, then eyeball decoded rows in stdout
  python scripts/lora/materialize_kd_corpus.py \\
      --mix ja_en --n-seqs 4 --seq-len 128 --out /tmp/kd_smoke
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from transformers import AutoTokenizer

# Allow importing domain_data from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_data import load_mixed_ja_en_sequences

_DEFAULT_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"
_DEFAULT_SEQ_LEN = 512
_DEFAULT_N_SEQS = 4096
_DEFAULT_SEED = 42


def _get_pad_id(tokenizer) -> int:
    """Return the pad token id, falling back to eos_token_id."""
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Materialize a fixed KD corpus from the same data sources as mote_train.py. "
            "Writes ids.npy (int32 [N_SEQS, SEQ_LEN]) + meta.json."
        )
    )
    ap.add_argument(
        "--mix",
        choices=["ja_en"],
        required=True,
        help=(
            "Corpus mix mode. "
            "'ja_en' = ~60%% wiki_ja + ~40%% no_robots train, "
            "interleaved 3:2 (same as mote_train.py --mix ja_en)."
        ),
    )
    ap.add_argument(
        "--n-seqs",
        type=int,
        default=_DEFAULT_N_SEQS,
        help=f"Number of fixed-length sequences to produce (default: {_DEFAULT_N_SEQS})",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=_DEFAULT_SEQ_LEN,
        help=f"Token sequence length; sequences are right-padded or truncated to this (default: {_DEFAULT_SEQ_LEN})",
    )
    ap.add_argument("--out", required=True, help="Output corpus directory")
    ap.add_argument(
        "--base",
        default=_DEFAULT_BASE,
        help=f"HF model id (or local path) to load the tokenizer from (default: {_DEFAULT_BASE})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=(
            f"Fisher-Yates shuffle seed applied after data loading to produce a "
            f"deterministic row order independent of data-loader streaming order. "
            f"Recorded in meta.json. (default: {_DEFAULT_SEED})"
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print(f"[materialize_kd_corpus] loading tokenizer from {args.base!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    pad_id = _get_pad_id(tokenizer)
    print(
        f"[materialize_kd_corpus] tokenizer ready: "
        f"vocab_size={tokenizer.vocab_size}  pad_id={pad_id}"
    )
    print(
        f"[materialize_kd_corpus] params: "
        f"mix={args.mix!r}  n_seqs={args.n_seqs}  "
        f"seq_len={args.seq_len}  seed={args.seed}"
    )

    # ------------------------------------------------------------------
    # 2. Load data via same loaders as mote_train.py
    # ------------------------------------------------------------------
    if args.mix == "ja_en":
        # Request slightly more than needed so the shuffle has room to trim to n_seqs.
        # load_mixed_ja_en_sequences returns exactly as many seqs as available from the
        # combined JA+EN token pools, up to the requested n_seqs.
        seqs, labels = load_mixed_ja_en_sequences(
            tokenizer=tokenizer,
            n_seqs=args.n_seqs,
            seq_len=args.seq_len,
            ja_frac=0.6,
        )
    else:
        raise ValueError(f"Unknown mix: {args.mix!r}")

    if not seqs:
        raise RuntimeError(
            f"Data loader returned 0 sequences for mix={args.mix!r}. "
            "Check HF dataset availability and --n-seqs."
        )

    print(f"[materialize_kd_corpus] loaded {len(seqs)} sequences before shuffle/trim")

    # ------------------------------------------------------------------
    # 3. Deterministic shuffle + trim to n_seqs
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    seqs = [seqs[i] for i in indices]
    seqs = seqs[: args.n_seqs]

    actual_n = len(seqs)
    if actual_n < args.n_seqs:
        print(
            f"[materialize_kd_corpus] WARNING: requested {args.n_seqs} seqs "
            f"but only {actual_n} available after data loading."
        )

    # ------------------------------------------------------------------
    # 4. Build ids array  [N, L] int32, right-padded with pad_id
    # ------------------------------------------------------------------
    ids = np.full((actual_n, args.seq_len), pad_id, dtype=np.int32)
    for i, seq in enumerate(seqs):
        # seq is a torch.Tensor [seq_len] from the data loader;
        # sequences from load_mixed_ja_en_sequences are exactly seq_len long
        # (chunked from concatenated text), so padding is only needed for edge cases.
        arr = np.asarray(seq, dtype=np.int32)
        n = min(len(arr), args.seq_len)
        ids[i, :n] = arr[:n]

    # ------------------------------------------------------------------
    # 5. Write ids.npy
    # ------------------------------------------------------------------
    ids_path = os.path.join(args.out, "ids.npy")
    np.save(ids_path, ids)
    print(f"\n[materialize_kd_corpus] ids.npy: shape={ids.shape}  dtype={ids.dtype}")
    print(f"  -> {ids_path}")

    # ------------------------------------------------------------------
    # 6. Write meta.json
    # ------------------------------------------------------------------
    meta = {
        "tokenizer": args.base,
        "seq_len": args.seq_len,
        "n_seqs": actual_n,
        "pad_id": pad_id,
        "mix": args.mix,
        "seed": args.seed,
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[materialize_kd_corpus] meta.json: {json.dumps(meta)}")
    print(f"  -> {meta_path}")

    # ------------------------------------------------------------------
    # 7. Human-readable sample rows to verify JA+EN presence
    # ------------------------------------------------------------------
    print(f"\n[materialize_kd_corpus] sample decoded rows (first 80 tokens each):")
    n_samples = min(6, actual_n)
    for i in range(n_samples):
        row_ids = ids[i].tolist()
        # Trim trailing pad tokens for display
        display_ids = [t for t in row_ids[:80] if t != pad_id] or row_ids[:8]
        decoded = tokenizer.decode(display_ids, skip_special_tokens=True)
        lang_hint = ""
        if i < len(labels):
            lang_hint = f"[{labels[indices[i]]}] "
        # repr() produces ASCII-safe output (escapes CJK as \uXXXX), safe on all consoles.
        preview = repr(decoded[:110])
        try:
            print(f"  row[{i:3d}] {lang_hint}{preview}")
        except UnicodeEncodeError:
            # Fallback for Windows consoles with narrow code pages (e.g. cp1252).
            safe = preview.encode(sys.stdout.encoding or "ascii", errors="replace").decode(
                sys.stdout.encoding or "ascii"
            )
            print(f"  row[{i:3d}] {lang_hint}{safe}")

    print(
        f"\n[materialize_kd_corpus] DONE: "
        f"{actual_n} seqs x {args.seq_len} tokens  "
        f"pad_id={pad_id}  seed={args.seed}"
    )
    print(f"  ids.npy shape  : {ids.shape}   dtype=int32")
    print(f"  contract check : ids.npy[i, :] = i-th sequence, right-padded with pad_id={pad_id}")


if __name__ == "__main__":
    main()
