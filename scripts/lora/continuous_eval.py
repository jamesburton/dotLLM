"""Continuous capability proxies for bitdistill — gold-continuation NLL.

Exact-match (GSM8K) and pass@1 (C#/tool-use) sit at 0 until a task-specific competence
threshold (FLP / emergent-point, arXiv 2410.08527); being harsh/discontinuous metrics they
hide sub-threshold progress (Schaeffer, "emergence is a metric artifact", arXiv 2304.15004).
These metrics instead score the mean per-token NLL the model assigns to the GOLD output given
the prompt — a smooth signal that moves *before* the harsh metric fires. Lower NLL = the
correct answer is more probable = capability improving underneath a still-zero pass@1.

Teacher-forced: ONE forward per example (no generation), so cheap enough for every milestone
and runnable on CPU alongside a GPU training job.
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def completion_nll(model, tokenizer, prompt_text: str, gold_text: str, device, max_len: int = 1024):
    """Mean per-token NLL of ``gold_text`` conditioned on ``prompt_text`` (teacher-forced).
    Tail-truncates the concatenation to ``max_len`` so the gold span always survives."""
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    g_ids = tokenizer(gold_text, add_special_tokens=False)["input_ids"]
    if not g_ids:
        return float("nan"), 0
    ids = (p_ids + g_ids)[-max_len:]
    n_gold = min(len(g_ids), len(ids) - 1)
    if n_gold <= 0:
        return float("nan"), 0
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=x, use_cache=False).logits[0].float()   # [L, V]
    logp = F.log_softmax(logits, dim=-1)
    tgt = x[0, -n_gold:]                                              # gold tokens
    pred = logp[-n_gold - 1:-1]                                       # logits predicting them
    return -pred.gather(1, tgt.unsqueeze(1)).squeeze(1).mean().item(), n_gold


def eval_completion_nll(model, tokenizer, examples: list, device) -> float:
    """Mean gold-continuation NLL over ``examples`` (each ``{'prompt_text','gold_text'}``).
    Lower is better; ``nan`` for an empty list."""
    if not examples:
        return float("nan")
    was = model.training
    model.eval()
    prev = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    tot, cnt = 0.0, 0
    for ex in examples:
        nll, n = completion_nll(model, tokenizer, ex["prompt_text"], ex["gold_text"], device)
        if n > 0 and not math.isnan(nll):
            tot += nll
            cnt += 1
    model.config.use_cache = prev
    if was:
        model.train()
    return tot / cnt if cnt else float("nan")


def load_gsm8k_nll(tokenizer, n: int = 40, few_shot: int = 4) -> list:
    """Held-out GSM8K test items as (few-shot CoT prompt, gold full-solution) pairs."""
    import datasets as hf
    train = list(itertools.islice(iter(hf.load_dataset("openai/gsm8k", "main", split="train")), few_shot))
    prefix = "".join(f"Question: {r['question'].strip()}\nAnswer: {r['answer'].strip()}\n\n" for r in train)
    ds = hf.load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for row in itertools.islice(iter(ds), n):
        q, a = row["question"].strip(), row["answer"].strip()
        out.append({"prompt_text": prefix + f"Question: {q}\nAnswer:", "gold_text": " " + a})
    return out


def eval_gsm8k_nll(model, tokenizer, examples: list, device) -> float:
    return eval_completion_nll(model, tokenizer, examples, device)
