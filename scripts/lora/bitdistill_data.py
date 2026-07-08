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

import itertools
from typing import Iterator, Optional

import torch


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
) -> Iterator[torch.Tensor]:
    """Yield fixed-length token sequences from a *streaming* HF dataset.

    Documents are tokenized on the fly and packed into contiguous ``seq_len``
    windows (GPT-style packing). Streaming avoids materialising the corpus; the
    budget-curve driver simply pulls as many windows as the token budget needs.

    Falls back gracefully: if ``dataset_config`` is invalid for the dataset the
    caller should catch and retry with ``dataset_config=None``.
    """
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(
        dataset_name, dataset_config, split=dataset_split, streaming=True
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

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
    seq_len: int = 512,
) -> list:
    """Return a small list of GSM8K examples for the go/no-go accuracy eval.

    Each item is ``{"prompt_ids": LongTensor, "answer": str, "question": str}``.
    The gold numeric answer is parsed from the ``#### N`` marker GSM8K uses.
    """
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for row in ds:
        q = row["question"].strip()
        a_full = row["answer"].strip()
        gold = a_full.split("####")[-1].strip().replace(",", "") if "####" in a_full else ""
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:seq_len]
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
) -> list:
    """Return a small held-out list of packed sequences for PPL tracking."""
    stream = cpt_token_stream(
        tokenizer, seq_len, dataset_name=dataset_name,
        dataset_config=dataset_config, seed=12345,
    )
    return take(stream, n)
