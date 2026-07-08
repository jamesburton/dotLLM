"""multitask_routed_data.py — task-labeled multi-task corpus for supervised routing.

Campaign: trackM-mote — identity-expert routed-MoTE.

Emits ``(token_ids[seq_len], routing_label)`` pairs where ``routing_label`` is an
integer expert index used by the SUPERVISED routing loss in
``identity_mote_train.py``. Each capability's data is tagged with its own label so
the router is trained to send that requirement's data to ITS OWN expert
(math→expert 1, instruction→expert 2, …). Label 0 is RESERVED for the
identity/skip expert and is never emitted here (general/OOD tokens fall back to it
at inference).

Capabilities (label = position in the selected list, starting at 1)
-------------------------------------------------------------------
  * ``math``        — openai/gsm8k ``main`` train, chain-of-thought format
                      ``"Question: {q}\nAnswer: {answer_with_CoT ending in #### N}"``.
                      (Cached locally.)
  * ``instruction`` — HuggingFaceH4/no_robots train, via ``capability_data``.
                      (Cached locally.)
  * ``tooluse``     — NousResearch/hermes-function-calling-v1 glaive_func_calling,
                      via ``capability_data``. (NOT cached offline as of this
                      writing — stage it like orca-math, else this capability is
                      skipped with a warning.)
  * ``coding``      — iamtarun/python_code_instructions_18k_alpaca. (Cached; offered
                      as a cached stand-in axis when tooluse is unavailable.)

Design notes
------------
* Labels are assigned to the SELECTED capabilities in order, so the trainer sets
  ``n_capability_experts = len(selected)`` and ``n_experts = len(selected) + 1``.
* A capability whose dataset cannot be loaded is SKIPPED (warning), and the label
  map is compacted over the capabilities that did load — so smoke/dev runs work on
  whatever is cached without a hard failure.
* Per-sequence labels (the whole gsm8k sequence is "math") — broadcast to every
  token by the trainer. Simple and matches the data.
* CPU-only: no model, no GPU, no PPL. Safe to import/run in a CPU session.

Smoke-test (CPU, no GPU)::

    python scripts/lora/multitask_routed_data.py \
        [--capabilities math,instruction] [--n-seqs 8] [--seq-len 256]
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capability_data import (  # noqa: E402  (local import after sys.path tweak)
    load_capability_sequences,
    _ids_to_sequences,
    _collect_ids,
)

# Default capability order (labels 1..K assigned in this order among those selected).
DEFAULT_CAPABILITIES = ["math", "instruction", "tooluse"]

# Track which HF dataset id was actually used, per capability (for reporting).
DATASET_USED: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Per-capability sequence loaders → list[Tensor[seq_len]]
# ---------------------------------------------------------------------------


def _load_math_gsm8k(tokenizer, n_seqs: int, seq_len: int, needed: int) -> list:
    """GSM8K chain-of-thought sequences (math capability).

    Format matches the eval prompt family: ``Question: ...\nAnswer: <CoT ... #### N>``
    so the routed math expert sees exactly the tokens it must learn to produce.
    """
    from datasets import load_dataset

    label = "math"
    all_ids: list[int] = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        DATASET_USED[label] = "openai/gsm8k main train (CoT: 'Question: ..\\nAnswer: ..#### N')"

        def _extract(row: dict) -> str:
            return f"Question: {row['question']}\nAnswer: {row['answer']}"

        all_ids = _collect_ids(ds, _extract, tokenizer, needed, label)
    except Exception as exc:  # noqa: BLE001
        print(f"[multitask:math] openai/gsm8k unavailable ({exc})")
    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


def _load_coding(tokenizer, n_seqs: int, seq_len: int, needed: int) -> list:
    """Python code-instruction sequences (coding capability; cached stand-in)."""
    from datasets import load_dataset

    label = "coding"
    all_ids: list[int] = []
    try:
        ds = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True
        )
        DATASET_USED[label] = "iamtarun/python_code_instructions_18k_alpaca train"

        def _extract(row: dict) -> str:
            instr = row.get("instruction", "") or ""
            inp = row.get("input", "") or ""
            out = row.get("output", "") or ""
            parts = [p for p in (instr, inp, out) if p.strip()]
            return "\n\n".join(parts)

        all_ids = _collect_ids(ds, _extract, tokenizer, needed, label)
    except Exception as exc:  # noqa: BLE001
        print(f"[multitask:coding] python_code_instructions unavailable ({exc})")
    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


def _load_one_capability(cap: str, tokenizer, n_seqs: int, seq_len: int) -> list:
    """Dispatch to the right loader; returns list of Tensor[seq_len] (may be empty)."""
    needed = n_seqs * seq_len + seq_len
    try:
        if cap == "math":
            return _load_math_gsm8k(tokenizer, n_seqs, seq_len, needed)
        if cap == "coding":
            return _load_coding(tokenizer, n_seqs, seq_len, needed)
        if cap in ("tooluse", "instruction"):
            seqs = load_capability_sequences(cap, tokenizer, n_seqs=n_seqs, seq_len=seq_len)
            from capability_data import _CAP_DATASET_USED
            DATASET_USED[cap] = _CAP_DATASET_USED.get(cap, cap)
            return seqs
        raise ValueError(f"unknown capability {cap!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[multitask:{cap}] SKIPPED — could not load ({type(exc).__name__}: {exc})")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_routed_corpus(
    tokenizer,
    capabilities: list[str],
    n_seqs_per_cap: int,
    seq_len: int,
    tiny_random: bool = False,
    vocab_size: int = 32000,
    interleave: bool = True,
    seed: int = 0,
) -> tuple[list, list, dict]:
    """Build a task-labeled multi-task corpus.

    Args:
        tokenizer: HF tokenizer (ignored in ``tiny_random`` mode).
        capabilities: ordered capability names; labels assigned 1..K over the
            capabilities that actually load (0 reserved for identity/skip).
        n_seqs_per_cap: sequences to draw per capability.
        seq_len: tokens per sequence.
        tiny_random: synthetic random ids + labels, no downloads (smoke/CI).
        vocab_size: for ``tiny_random`` id sampling.
        interleave: round-robin capabilities so batches see mixed labels.
        seed: RNG seed for tiny_random / shuffling.

    Returns:
        ``(sequences, labels, label_map)`` where ``sequences[i]`` is a
        ``Tensor[seq_len]`` int64, ``labels[i]`` is its integer routing label
        (>=1), and ``label_map`` maps capability name → label int.
    """
    g = torch.Generator().manual_seed(seed)

    if tiny_random:
        label_map = {cap: i + 1 for i, cap in enumerate(capabilities)}
        per_cap: dict[str, list] = {}
        for cap in capabilities:
            per_cap[cap] = [
                torch.randint(0, vocab_size, (seq_len,), generator=g)
                for _ in range(n_seqs_per_cap)
            ]
        return _assemble(per_cap, label_map, interleave, g)

    # Real data: load each capability; compact label map over those that loaded.
    loaded: dict[str, list] = {}
    for cap in capabilities:
        seqs = _load_one_capability(cap, tokenizer, n_seqs_per_cap, seq_len)
        if seqs:
            loaded[cap] = seqs
        else:
            print(f"[multitask] capability {cap!r} produced 0 sequences — dropped")

    if not loaded:
        raise RuntimeError(
            "No capabilities loaded any data. Check the cache / --capabilities, "
            "or use --tiny-random for a smoke test."
        )

    # Labels are assigned in the ORIGINAL requested order among the loaded caps.
    label_map = {
        cap: i + 1 for i, cap in enumerate([c for c in capabilities if c in loaded])
    }
    return _assemble(loaded, label_map, interleave, g)


def _assemble(per_cap: dict, label_map: dict, interleave: bool, g) -> tuple:
    """Flatten per-capability sequences into parallel (sequences, labels) lists."""
    sequences: list = []
    labels: list = []
    if interleave:
        # Round-robin so consecutive training steps rotate through capabilities.
        iters = {cap: iter(seqs) for cap, seqs in per_cap.items()}
        exhausted: set = set()
        while len(exhausted) < len(iters):
            for cap, it in iters.items():
                if cap in exhausted:
                    continue
                try:
                    s = next(it)
                except StopIteration:
                    exhausted.add(cap)
                    continue
                sequences.append(s)
                labels.append(label_map[cap])
    else:
        for cap, seqs in per_cap.items():
            for s in seqs:
                sequences.append(s)
                labels.append(label_map[cap])
        # shuffle
        perm = torch.randperm(len(sequences), generator=g).tolist()
        sequences = [sequences[i] for i in perm]
        labels = [labels[i] for i in perm]

    return sequences, labels, label_map


# ---------------------------------------------------------------------------
# Smoke-test entry point (CPU only)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Smoke-test multitask_routed_data.py: load a few labeled "
        "sequences per capability and verify shapes/labels. No GPU."
    )
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
                    help="Tokenizer id or path.")
    ap.add_argument("--capabilities", default="math,instruction",
                    help="Comma-separated capability names in label order.")
    ap.add_argument("--n-seqs", type=int, default=6, help="Sequences per capability.")
    ap.add_argument("--seq-len", type=int, default=256, help="Tokens per sequence.")
    ap.add_argument("--tiny-random", action="store_true",
                    help="Synthetic ids + labels, no downloads.")
    args = ap.parse_args()

    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]

    tok = None
    if not args.tiny_random:
        from transformers import AutoTokenizer
        print(f"[multitask] loading tokenizer {args.base!r} ...")
        tok = AutoTokenizer.from_pretrained(args.base)

    seqs, labels, label_map = build_routed_corpus(
        tokenizer=tok,
        capabilities=caps,
        n_seqs_per_cap=args.n_seqs,
        seq_len=args.seq_len,
        tiny_random=args.tiny_random,
    )

    print(f"\n[multitask] label_map = {label_map}")
    print(f"[multitask] total sequences = {len(seqs)}  (labels present: {sorted(set(labels))})")
    shapes_ok = all(s.shape == (args.seq_len,) and s.dtype == torch.long for s in seqs)
    print(f"[multitask] all shapes == ({args.seq_len},) int64: {shapes_ok}")
    # Per-label counts
    from collections import Counter
    print(f"[multitask] per-label counts: {dict(Counter(labels))}")
    for cap, lbl in label_map.items():
        print(f"[multitask]   {cap:12s} -> label {lbl}   dataset: {DATASET_USED.get(cap, '(tiny_random)')}")
    assert shapes_ok, "shape check failed"
    assert all(l >= 1 for l in labels), "labels must be >= 1 (0 reserved for skip)"
    print("\n[multitask] smoke-test PASSED")


if __name__ == "__main__":
    main()
