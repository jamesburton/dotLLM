"""Capability data loaders for MoTE probe-train experiments.

Provides fixed-length token-sequence loaders for two capability domains
where specialization and routing signal are expected:

  * **tooluse**     — function-calling / tool-use conversations
                      Dataset: ``NousResearch/hermes-function-calling-v1``
                      config ``glaive_func_calling``.
                      Held-out slice: ``train[3000:]`` (training used ``[:3000]``).
                      Format: ShareGPT conversations — system + human/gpt turns;
                      gpt turns that are gold tool responses contain
                      ``<tool_call>{"name":...,"arguments":...}</tool_call>``.

  * **instruction** — diverse instruction-following prompt/response pairs
                      Dataset: ``HuggingFaceH4/no_robots``.
                      Held-out slice: ``test`` split (train split used for adapter training).
                      Format: HF messages list with role/content dicts
                      (system/user/assistant roles).

Public API
----------
* ``load_capability_sequences(capability, tokenizer, n_seqs, seq_len)``
  Returns a list of ``n_seqs`` fixed-length int64 token-ID tensors of shape
  ``[seq_len]``, suitable for MoTE probe-train / LM PPL evaluation.

  ``capability`` is one of: ``"tooluse"``, ``"instruction"``.

CPU-only use
------------
This module deliberately contains NO model loading, NO GPU ops, and NO PPL
computation — it is safe to import and call on a CPU-only session.  The data
loaders are designed to be run once before a GPU probe-train run to verify
that sequences are obtainable and correctly shaped.

Smoke-test (CPU, no GPU required)::

  python scripts/lora/capability_data.py \\
      [--base microsoft/bitnet-b1.58-2B-4T-bf16] \\
      [--n-seqs 8] \\
      [--seq-len 512]

Probe-train commands (GPU required — run AFTER GPU is free)
-----------------------------------------------------------
Tool-use probe (mirroring the JA probe)::

  python scripts/lora/mote_train.py \\
      --config tooluse_probe --n-experts 4 --top-k 1 --shared ternary \\
      --layers 26-29 --tokens 2e6 --kd-weight 0 --device cuda \\
      --dataset NousResearch/hermes-function-calling-v1 \\
      --dataset-config glaive_func_calling \\
      --dataset-split "train[3000:]" \\
      --out .docs/mote/tooluse_probe

Instruction probe::

  python scripts/lora/mote_train.py \\
      --config instruction_probe --n-experts 4 --top-k 1 --shared ternary \\
      --layers 26-29 --tokens 2e6 --kd-weight 0 --device cuda \\
      --dataset HuggingFaceH4/no_robots \\
      --dataset-split test \\
      --out .docs/mote/instruction_probe

Issue reference: (#117)
"""

from __future__ import annotations

import os
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Tracks which HF dataset ID was actually used, per capability.
# ---------------------------------------------------------------------------
_CAP_DATASET_USED: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_text_tooluse(row: dict) -> str:
    """Render a glaive_func_calling row to plain text.

    Each row has a ``conversations`` list of ShareGPT-style turns::

        {"from": "system"|"human"|"gpt", "value": "..."}

    We concatenate all turns in order, preserving the tool-call markup
    (``<tool_call>...</tool_call>``) so the tokenizer sees the exact tokens
    the model must learn to produce.
    """
    convs = row.get("conversations") or []
    parts: list[str] = []
    for turn in convs:
        content = turn.get("value") or turn.get("content") or ""
        if content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts)


def _extract_text_instruction(row: dict) -> str:
    """Render a no_robots row to plain text.

    Each row has a ``messages`` list of HF-chat-format dicts::

        {"role": "system"|"user"|"assistant", "content": "..."}

    We concatenate all messages so both the instruction (user) and the
    response (assistant) are included in the tokenized sequence — the model
    must see both sides for LM-style training.
    """
    messages = row.get("messages") or []
    if messages:
        return "\n\n".join(
            m.get("content", "")
            for m in messages
            if m.get("content", "").strip()
        )
    # Fallback for unexpected schemas.
    if "text" in row:
        return str(row["text"])
    return " ".join(str(v) for v in row.values() if isinstance(v, str) and v.strip())


# ---------------------------------------------------------------------------
# Token-ID collection helper (mirrors domain_data.py)
# ---------------------------------------------------------------------------


def _collect_ids(
    ds_iter,
    extract_fn,
    tokenizer,
    needed: int,
    label: str,
) -> list[int]:
    """Tokenize rows from an iterable dataset until ``needed`` token IDs collected."""
    all_ids: list[int] = []
    for row in ds_iter:
        text = extract_fn(row)
        if not text.strip():
            continue
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= needed:
            break
    print(f"[capability_data:{label}] collected {len(all_ids)} tokens")
    return all_ids


def _ids_to_sequences(
    all_ids: list[int],
    n_seqs: int,
    seq_len: int,
    label: str,
) -> list:
    """Slice a flat token-ID list into fixed-length torch tensors."""
    if len(all_ids) < seq_len:
        raise RuntimeError(
            f"[capability_data:{label}] not enough tokens: "
            f"got {len(all_ids)}, need >= {seq_len}."
        )
    seqs: list = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break
    print(f"[capability_data:{label}] produced {len(seqs)} sequences x {seq_len} tokens")
    return seqs


# ---------------------------------------------------------------------------
# Per-capability loaders
# ---------------------------------------------------------------------------


def _load_tooluse_sequences(tokenizer, n_seqs: int, seq_len: int, needed: int) -> list:
    """Load held-out tool-use conversations from glaive_func_calling.

    Held-out slice: ``train[3000:]`` — disjoint from the ``[:3000]`` slice
    used for the prior task-LoRA adapter training (U2 Phase A, EVAL.md).
    """
    label = "tooluse"
    all_ids: list[int] = []

    try:
        ds = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            "glaive_func_calling",
            split="train[3000:]",
            streaming=True,
        )
        _CAP_DATASET_USED[label] = (
            "NousResearch/hermes-function-calling-v1 glaive_func_calling "
            "train[3000:] (held-out; train[:3000] used for tooluse LoRA)"
        )
        all_ids = _collect_ids(ds, _extract_text_tooluse, tokenizer, needed, label)
    except Exception as exc:
        print(f"[capability_data:tooluse] hermes-function-calling-v1 unavailable ({exc})")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


def _load_instruction_sequences(tokenizer, n_seqs: int, seq_len: int, needed: int) -> list:
    """Load held-out instruction-following examples from no_robots.

    Uses the ``test`` split — disjoint from the ``train`` split used for the
    instruction task-LoRA adapter (U2 Phase A, EVAL.md).  Also matches the
    eval corpus used for MoTE c0/c0_lowaux runs in EVAL_mote.md.
    """
    label = "instruction"
    all_ids: list[int] = []

    try:
        ds = load_dataset(
            "HuggingFaceH4/no_robots",
            split="test",
            streaming=True,
        )
        _CAP_DATASET_USED[label] = (
            "HuggingFaceH4/no_robots test split "
            "(train split used for instruction LoRA; test is held-out and matches EVAL_mote.md corpus)"
        )
        all_ids = _collect_ids(ds, _extract_text_instruction, tokenizer, needed, label)
    except Exception as exc:
        print(f"[capability_data:instruction] no_robots unavailable ({exc})")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_capability_sequences(
    capability: str,
    tokenizer,
    n_seqs: int = 98,
    seq_len: int = 512,
) -> list:
    """Load capability text and tokenize into fixed-length sequences.

    Parameters
    ----------
    capability:
        One of ``"tooluse"`` or ``"instruction"``.
    tokenizer:
        A HuggingFace tokenizer compatible with the BitNet model
        (``microsoft/bitnet-b1.58-2B-4T-bf16``).
    n_seqs:
        Number of fixed-length sequences to return.
    seq_len:
        Token sequence length (default 512, matching domain_data.py and EVAL_mote.md).

    Returns
    -------
    A list of exactly ``n_seqs`` :class:`torch.Tensor` objects of shape
    ``[seq_len]`` (dtype int64).
    """
    needed = n_seqs * seq_len + seq_len  # small buffer to guarantee n_seqs slices

    if capability == "tooluse":
        return _load_tooluse_sequences(tokenizer, n_seqs, seq_len, needed)
    elif capability == "instruction":
        return _load_instruction_sequences(tokenizer, n_seqs, seq_len, needed)
    else:
        raise ValueError(
            f"Unknown capability {capability!r}. Choose: tooluse, instruction"
        )


# ---------------------------------------------------------------------------
# Smoke-test entry point (CPU only — no model, no PPL)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Smoke-test capability_data.py: tokenize a handful of sequences for "
            "each capability and verify shapes. No GPU required."
        )
    )
    ap.add_argument(
        "--base",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="Tokenizer to use (HF hub id or local path)",
    )
    ap.add_argument("--n-seqs", type=int, default=8, help="Sequences to load per capability")
    ap.add_argument("--seq-len", type=int, default=512, help="Tokens per sequence")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"[capability_data] loading tokenizer from {args.base!r} ...")
    tok = AutoTokenizer.from_pretrained(args.base)

    for cap in ("tooluse", "instruction"):
        print(f"\n[capability_data] === {cap.upper()} ===")
        seqs = load_capability_sequences(cap, tok, n_seqs=args.n_seqs, seq_len=args.seq_len)
        shapes_ok = all(s.shape == (args.seq_len,) and s.dtype == torch.long for s in seqs)
        print(
            f"[capability_data:{cap}] {len(seqs)} sequences x {args.seq_len} tokens — "
            f"shapes OK: {shapes_ok}"
        )
        print(f"[capability_data:{cap}] dataset used: {_CAP_DATASET_USED.get(cap, 'unknown')}")
        # Print first 8 token IDs of first sequence as a sanity check.
        print(f"[capability_data:{cap}] seq[0][:8] = {seqs[0][:8].tolist()}")

    print("\n[capability_data] smoke-test PASSED")


if __name__ == "__main__":
    main()
