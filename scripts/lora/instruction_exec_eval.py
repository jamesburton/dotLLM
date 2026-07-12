"""Instruction-following eval (auto-checkable) for bitdistill checkpoint eval — design build item B1.

``eval_instruction.py`` serves base-vs-adapter on open-ended probes and defers scoring to an
out-of-band pairwise LLM judge. That's too slow / non-deterministic for frequent in-training
checkpoints. This module instead evaluates an **auto-checkable constraint subset** (IFEval-style
length / format / keyword rules) so scoring is fully deterministic — no judge — and runs in-process
against the live PyTorch student, mirroring ``csharp_exec_eval.py`` (B2) and ``tooluse_exec_eval.py``:
batched, left-padded ``model.generate`` with ``use_cache`` toggled on for decode, graceful ``nan``
on an empty set.

Each probe pairs an instruction with a deterministic verifier over the generated text
(``verify(text) -> bool``). ``eval_instruction_inproc`` returns the fraction of probes whose output
satisfies its constraint, or ``nan`` for an empty set — same shape as ``eval_gsm8k``.

The chat-templating / generation flow reuses ``eval_instruction.py``'s pattern
(``apply_chat_template([{"role":"user",...}], add_generation_prompt=True)``); only the scoring is
new (deterministic constraint checks instead of a judge). The full open-ended judge eval
(``eval_instruction.py``) is still the end-of-run quality signal.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Verifier helpers (deterministic, operate on the generated text only)
# ---------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?]+", text.strip()) if s.strip()]


def _numbered_list_items(text: str) -> int:
    return sum(1 for ln in text.splitlines() if re.match(r"\s*\d+[.)]\s+\S", ln))


# ---------------------------------------------------------------------------
# Auto-checkable constraint probe set (IFEval-lite).
# Each entry: (instruction, verifier). Verifiers are intentionally lenient about
# incidental whitespace / punctuation but strict about the constraint itself.
# ---------------------------------------------------------------------------
_PROBES: list[tuple[str, "callable"]] = [
    ("Respond with exactly one word and nothing else.",
     lambda t: len(_words(t)) == 1),
    ("Answer using exactly three sentences.",
     lambda t: len(_sentences(t)) == 3),
    ("Reply using only lowercase letters (no uppercase letters at all).",
     lambda t: any(c.isalpha() for c in t) and t == t.lower()),
    ("Respond entirely in uppercase letters.",
     lambda t: any(c.isalpha() for c in t) and t == t.upper()),
    ("List exactly five items as a numbered list (1. 2. 3. 4. 5.).",
     lambda t: _numbered_list_items(t) == 5),
    ("Include the word 'banana' somewhere in your response.",
     lambda t: "banana" in t.lower()),
    ("Write a response of no more than ten words.",
     lambda t: 1 <= len(_words(t)) <= 10),
    ("End your entire response with the exact word DONE.",
     lambda t: t.strip().rstrip(".!?").endswith("DONE")),
    ("Give your answer as a comma-separated list of exactly four colors.",
     lambda t: len([p for p in t.split(",") if p.strip()]) == 4),
    ("Do not use the letter e anywhere in your response.",
     lambda t: any(c.isalpha() for c in t) and "e" not in t.lower()),
    ("Answer with exactly two sentences, and make the first word 'Yes'.",
     lambda t: len(_sentences(t)) == 2 and t.strip().lower().startswith("yes")),
    ("Respond with a single number and no other characters.",
     lambda t: t.strip().isdigit()),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_instruction_eval(tokenizer=None, n: Optional[int] = None) -> list:
    """Return the auto-checkable instruction constraint probes (optionally the first ``n``).

    Each element is a dict ``{"instruction", "verify"}`` where ``verify(text) -> bool`` deterministically
    checks the constraint. ``tokenizer`` is accepted for signature parity with the other B1 loaders
    (prompts are chat-templated at eval time, so it is unused here). Always returns a non-empty list
    unless ``n <= 0``.
    """
    probes = _PROBES if n is None else _PROBES[:max(0, n)]
    out = [{"instruction": ins, "verify": fn} for ins, fn in probes]
    print(f"[instruction-eval] loaded {len(out)} auto-checkable constraint probes.", flush=True)
    return out


# ---------------------------------------------------------------------------
# In-process eval
# ---------------------------------------------------------------------------
def eval_instruction_inproc(
    model,
    tokenizer,
    examples: list,
    device,
    max_new_tokens: int = 64,
    chunk: int = 8,
) -> float:
    """Chat-templated in-process generation → deterministic constraint check → fraction satisfied.

    Batched + left-padded; ``use_cache`` is forced on for decode (the student trains with it off).
    Returns the fraction of probes whose generated text satisfies its verifier, or ``nan`` for an
    empty example list — same shape as ``eval_gsm8k``/``eval_csharp_exec``/``eval_tooluse_inproc``.
    """
    if not examples:
        return float("nan")

    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    has_template = getattr(tokenizer, "chat_template", None) is not None

    # Pre-tokenize prompts (chat-templated for train==serve parity; raw fallback if no template).
    prompt_ids = []
    for e in examples:
        if has_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": e["instruction"]}],
                add_generation_prompt=True, tokenize=False)
        else:
            text = e["instruction"]
        prompt_ids.append(tokenizer(text, add_special_tokens=not has_template)["input_ids"])

    satisfied = 0
    with torch.no_grad():
        for i in range(0, len(examples), chunk):
            batch = list(range(i, min(i + chunk, len(examples))))
            maxlen = max(len(prompt_ids[j]) for j in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for k, j in enumerate(batch):  # left-pad so all sequences end aligned
                ids = prompt_ids[j]
                input_ids[k, maxlen - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attn[k, maxlen - len(ids):] = 1
            out = model.generate(
                input_ids=input_ids.to(device), attention_mask=attn.to(device),
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
            for k, j in enumerate(batch):
                gen = tokenizer.decode(out[k, maxlen:], skip_special_tokens=True)
                try:
                    ok = bool(examples[j]["verify"](gen))
                except Exception:  # noqa: BLE001 — a verifier crash scores 0, never aborts the eval
                    ok = False
                if ok:
                    satisfied += 1

    model.config.use_cache = prev_cache
    if was_training:
        model.train()
    return satisfied / len(examples)


# ---------------------------------------------------------------------------
# Standalone smoke (loads a tiny model, prints the float).
# ---------------------------------------------------------------------------
def _smoke(base: str = "Qwen/Qwen3-0.6B", n: Optional[int] = 4, device: Optional[str] = None) -> None:
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[instruction-eval:smoke] base={base} n={n} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    examples = load_instruction_eval(tok, n=n)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    t0 = time.time()
    score = eval_instruction_inproc(model, tok, examples, device)
    dt = time.time() - t0
    print(f"[instruction-eval:smoke] instruction_acc={score:.4f}  ({len(examples)} probes, {dt:.1f}s)", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test instruction_exec_eval (tiny model, few probes).")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    _smoke(a.base, a.n, a.device)
