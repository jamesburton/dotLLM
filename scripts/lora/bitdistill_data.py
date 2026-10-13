"""bitdistill_data.py — corpus loaders for the BitNet-Distillation harness.

Two corpora, per the BitDistill recipe (arXiv 2510.13998):

1. **CPT (continual-pretrain) slice** — a general-web stream used for the
   continual-pretraining warm-up that the paper runs for ~10B tokens on the
   FALCON corpus. We default to a *streaming* FineWeb-Edu slice
   (``HuggingFaceFW/fineweb-edu``) so nothing large is materialised on disk;
   ``--cpt-dataset falcon-refinedweb`` reproduces the paper's corpus family.
   Streaming means the budget-curve driver can pull an arbitrary token count
   without a full download.

2. **Task slice** — a small supervised set for the go/no-go eval. Default
   ``openai/gsm8k`` ("main"); any cached instruction set works.

``tiny_random`` mode returns synthetic token tensors with no download, for the
CPU self-test / smoke.

All loaders yield fixed-length ``torch.long`` sequences of ``seq_len`` tokens.
The CPT loader is an *iterator* (streaming, effectively unbounded); the task
loader returns a materialised list (small, for eval).
"""

from __future__ import annotations

import glob
import itertools
import os
from typing import Iterator, Optional

import torch


# ---------------------------------------------------------------------------
# Local-shard resolver — the robust calibration/CPT path.
# ---------------------------------------------------------------------------
# Streaming FineWeb-Edu over the network with a large shuffle-prefetch buffer
# balloons host RAM catastrophically on a flaky link: partial-download/retry
# buffers plus multi-hundred-MB parquet row-groups accumulate until the box
# swap-thrashes (this is what made warm-start appear to "hang at layer 4" —
# it was never the GPTQ solve). If a FineWeb-Edu shard is already in the HF
# cache we prefer it: memory stays bounded and no network can kill the run.
def resolve_local_fineweb(dataset_hint: str = "fineweb") -> Optional[str]:
    """Return a path to a locally-cached FineWeb(-Edu) parquet shard, or ``None``.

    Scans the HF datasets hub cache (``$HF_HOME/hub`` or ``~/.cache/huggingface/hub``)
    for a ``datasets--*<hint>*`` snapshot and returns the largest ``.parquet`` blob
    found (largest ⇒ a complete shard, not a truncated partial download).
    """
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))

    best, best_sz = None, 0
    for root in roots:
        pat = os.path.join(root, f"datasets--*{dataset_hint}*", "snapshots", "*", "**", "*.parquet")
        for p in glob.glob(pat, recursive=True):
            try:
                sz = os.path.getsize(os.path.realpath(p))  # follow symlink to blob
            except OSError:
                continue
            if sz > best_sz:
                best, best_sz = p, sz
    return best


# ---------------------------------------------------------------------------
# Tiny synthetic (no download) — used by --self-test / --tiny-random
# ---------------------------------------------------------------------------
def tiny_random_stream(vocab_size: int, seq_len: int, seed: int = 0) -> Iterator[torch.Tensor]:
    """Infinite stream of random token-ID sequences (no tokenizer, no download)."""
    g = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, vocab_size, (seq_len,), generator=g, dtype=torch.long)


def tiny_random_list(vocab_size: int, seq_len: int, n: int, seed: int = 1) -> list:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (seq_len,), generator=g, dtype=torch.long) for _ in range(n)]


# ---------------------------------------------------------------------------
# CPT streaming corpus (general web)
# ---------------------------------------------------------------------------
def cpt_token_stream(
    tokenizer,
    seq_len: int,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: Optional[str] = "sample-10BT",
    dataset_split: str = "train",
    text_field: str = "text",
    seed: int = 0,
    local_parquet: Optional[str] = None,
    prefer_local_cache: bool = True,
    shuffle_buffer: int = 10_000,
) -> Iterator[torch.Tensor]:
    """Yield fixed-length token sequences from a streaming HF dataset.

    Documents are tokenized on the fly and packed into contiguous ``seq_len``
    windows (GPT-style packing). Streaming avoids materialising the corpus; the
    budget-curve driver simply pulls as many windows as the token budget needs.

    ``local_parquet`` (path or glob): stream from a LOCAL parquet shard instead of
    the hub — no network dependency, so a flaky link can't kill a long run. This
    is the robust path for unattended runs (pre-download one fineweb-edu shard).
    Otherwise falls back gracefully: if ``dataset_config`` is invalid for the
    dataset the caller should catch and retry with ``dataset_config=None``.
    """
    import datasets as hf_datasets

    # Prefer a locally-cached shard when none was named explicitly: streaming over a
    # flaky network balloons host RAM (see resolve_local_fineweb). Callers wanting the
    # full network stream can pass prefer_local_cache=False.
    if not local_parquet and prefer_local_cache and "fineweb" in dataset_name.lower():
        cached = resolve_local_fineweb()
        if cached:
            local_parquet = cached
            print(f"[cpt] calibration/CPT from LOCAL shard (network-safe): {cached}", flush=True)

    if local_parquet:
        ds = hf_datasets.load_dataset(
            "parquet", data_files=local_parquet, split=dataset_split, streaming=True
        )
    else:
        print(f"[cpt] STREAMING {dataset_name} over network — a flaky link can balloon host "
              f"RAM; pre-cache a shard to avoid this.", flush=True)
        ds = hf_datasets.load_dataset(
            dataset_name, dataset_config, split=dataset_split, streaming=True
        )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    buf: list[int] = []
    for row in ds:
        text = row.get(text_field) or row.get("content") or ""
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len:
            window = buf[:seq_len]
            buf = buf[seq_len:]
            yield torch.tensor(window, dtype=torch.long)


def take(stream: Iterator[torch.Tensor], n: int) -> list:
    """Materialise the first ``n`` elements of a stream into a list."""
    return list(itertools.islice(stream, n))


# ---------------------------------------------------------------------------
# Task slice (eval) — gsm8k by default
# ---------------------------------------------------------------------------
def load_gsm8k(
    tokenizer,
    n: int = 100,
    split: str = "test",
    seq_len: int = 1024,
    few_shot: int = 4,
) -> list:
    """Return a small list of GSM8K examples for the accuracy eval.

    Each item is ``{"prompt_ids": LongTensor, "answer": str, "question": str}``.
    The gold numeric answer is parsed from the ``#### N`` marker GSM8K uses.

    ``few_shot`` prepends this many train examples (with their chain-of-thought
    solutions) to every prompt. Zero-shot GSM8K on a base/hybrid-reasoning model
    massively undercounts capability — measured FP Qwen3-1.7B: 15% zero-shot vs
    70% with 4-shot CoT — so few-shot is the default. Long prompts are kept from
    the tail (``[-seq_len:]``) so the actual question survives truncation.
    """
    import datasets as hf_datasets

    prefix = ""
    if few_shot > 0:
        train = hf_datasets.load_dataset("openai/gsm8k", "main", split="train")
        exs = list(itertools.islice(iter(train), few_shot))
        prefix = "".join(
            f"Question: {r['question'].strip()}\nAnswer: {r['answer'].strip()}\n\n" for r in exs)

    ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for row in ds:
        q = row["question"].strip()
        a_full = row["answer"].strip()
        gold = a_full.split("####")[-1].strip().replace(",", "") if "####" in a_full else ""
        prompt = prefix + f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-seq_len:]
        out.append(
            {
                "prompt_ids": torch.tensor(ids, dtype=torch.long),
                "answer": gold,
                "question": q,
            }
        )
        if len(out) >= n:
            break
    return out


def load_ppl_slice(
    tokenizer,
    n: int = 20,
    seq_len: int = 512,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: Optional[str] = "sample-10BT",
    local_parquet: Optional[str] = None,
) -> list:
    """Return a small held-out list of packed sequences for PPL tracking."""
    stream = cpt_token_stream(
        tokenizer, seq_len, dataset_name=dataset_name,
        dataset_config=dataset_config, seed=12345, local_parquet=local_parquet,
    )
    return take(stream, n)
