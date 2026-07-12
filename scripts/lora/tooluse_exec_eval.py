"""Tool-use eval (name+args accuracy) for bitdistill checkpoint eval — design build item B1.

Unlike ``eval_tooluse.py`` (which shells to dotLLM / ``eval_serve`` and reports base-vs-adapter),
this runs **in-process** against the live PyTorch student during distillation: chat-templated
``model.generate`` → parse ``<tool_call>`` → score name-match + args-JSON-equal. It mirrors the
structure of ``csharp_exec_eval.py`` (B2): batched, left-padded generate with ``use_cache`` toggled
on for decode, graceful no-op (``nan``) when the eval set is empty.

Scoring logic is reused from ``eval_tooluse.py``:
  * ``parse_tool_call``          — first parseable ``<tool_call>{...}</tool_call>`` JSON (or bare obj)
  * ``build_prompt_and_gold``    — render system+user(+prior tool exchange) via the canonical
                                    ``tooluse_render`` path (template-generated ``<tools>`` block, so
                                    train==serve parity) and extract the gold ``(name, args)``.

An example is **correct** iff a ``<tool_call>`` was emitted, its function name matches the gold
name, AND its arguments JSON equals the gold arguments. ``eval_tooluse_inproc`` returns the fraction
of correct examples (strict name+args), or ``nan`` for an empty set — same shape as ``eval_gsm8k``.

Held-out data: ``NousResearch/hermes-function-calling-v1`` / ``glaive_func_calling`` ``train[3000:]``
(the ``train[:3000]`` slice was used for the tool-use LoRA / distillation train mix, so ``[3000:]``
is disjoint — same held-out convention as ``capability_data.py``).
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Canonical tool renderer (template-generated <tools> block == dotLLM serving).
from tooluse_render import extract_tools, _SHAREGPT_ROLE, _strip_tools_block  # noqa: E402

# ---------------------------------------------------------------------------
# Scoring helpers (mirrors eval_tooluse.py — copied rather than imported to
# avoid pulling in eval_serve, which shells out to dotLLM).
# ---------------------------------------------------------------------------
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str):
    """Return ``(name, args_dict)`` from the first parseable ``<tool_call>`` JSON, or ``None``.

    Falls back to a bare ``{... "name": ...}`` object when no ``<tool_call>`` tags are present.
    """
    m = TOOLCALL_RE.search(text)
    if not m:
        m2 = re.search(r"\{[^{}]*\"name\"\s*:.*?\}", text, re.DOTALL)
        if not m2:
            return None
        blob = m2.group(0)
    else:
        blob = m.group(1)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return (name, args)


def build_prompt_and_gold(tok, row):
    """Render the eval prompt (system+user(+prior tool exchange), ``add_generation_prompt``) and the
    gold ``(name, args)``. Returns ``None`` if the row has no gold ``<tool_call>`` turn. Mirrors
    ``eval_tooluse.build_prompt_and_gold``."""
    tools = extract_tools(row)
    conv = row["conversations"]
    target = None
    for i, turn in enumerate(conv):
        if turn["from"] == "gpt":
            c = turn.get("value", turn.get("content", ""))
            if c.lstrip().startswith("<tool_call>"):
                target = i
                break
    if target is None:
        return None
    prompt_msgs = []
    for turn in conv[:target]:
        role = _SHAREGPT_ROLE.get(turn["from"], turn["from"])
        content = turn.get("value", turn.get("content", ""))
        if role == "system":
            content = _strip_tools_block(content)
        prompt_msgs.append({"role": role, "content": content})
    prompt = tok.apply_chat_template(
        prompt_msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    gold = parse_tool_call(conv[target].get("value", conv[target].get("content", "")))
    if gold is None or gold[0] is None:
        return None
    return prompt, gold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tooluse_eval(
    tokenizer,
    n: int = 20,
    start: int = 3000,
    dataset_id: str = "NousResearch/hermes-function-calling-v1",
    dataset_config: str = "glaive_func_calling",
) -> list:
    """Load up to ``n`` held-out glaive tool-use eval examples.

    Renders each row's prompt (chat-templated with the model tokenizer for train==serve parity)
    and extracts the gold ``(name, args)``. Returns a list of dicts:
    ``{"prompt", "gold_name", "gold_args"}``.

    Returns ``[]`` (with a warning) if the dataset/tokenizer is unavailable, so a training run
    without the dataset simply skips the tool-use eval — same contract as ``load_gsm8k``/
    ``load_csharp_tasks`` returning ``[]`` when disabled.
    """
    try:
        import datasets  # lazy: training must not hard-depend on datasets being importable early
    except Exception as exc:  # noqa: BLE001
        print(f"[tooluse-eval] datasets import failed ({exc}); tool-use eval disabled.", flush=True)
        return []
    try:
        # Pull a generous slice (rows without a gold <tool_call> are skipped) and keep first n.
        ds = datasets.load_dataset(
            dataset_id, dataset_config, split=f"train[{start}:{start + n * 4}]")
    except Exception as exc:  # noqa: BLE001
        print(f"[tooluse-eval] dataset load failed ({type(exc).__name__}: {exc}); disabled.", flush=True)
        return []

    out = []
    for row in ds:
        if len(out) >= n:
            break
        try:
            pg = build_prompt_and_gold(tokenizer, row)
        except Exception:  # noqa: BLE001 — a malformed row is skipped, never aborts loading
            pg = None
        if pg is None:
            continue
        prompt, (gname, gargs) = pg
        out.append({"prompt": prompt, "gold_name": gname, "gold_args": gargs})
    print(f"[tooluse-eval] loaded {len(out)} held-out tool-use examples (start={start}, of {n} requested).", flush=True)
    return out


# ---------------------------------------------------------------------------
# In-process eval
# ---------------------------------------------------------------------------
def eval_tooluse_inproc(
    model,
    tokenizer,
    examples: list,
    device,
    max_new_tokens: int = 160,
    chunk: int = 8,
) -> float:
    """Chat-templated in-process generation → parse ``<tool_call>`` → name+args accuracy.

    Batched + left-padded (all sequences end aligned); ``use_cache`` is forced on for decode (the
    student trains with it off). An example counts as correct iff a tool call is emitted whose name
    matches the gold name AND whose arguments JSON equals the gold arguments. Returns the fraction
    correct, or ``nan`` for an empty example list — same shape as ``eval_gsm8k``/``eval_csharp_exec``.
    """
    if not examples:
        return float("nan")

    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    # Prompts are already chat-templated (add_generation_prompt) at load time → no special tokens.
    prompt_ids = [tokenizer(e["prompt"], add_special_tokens=False)["input_ids"] for e in examples]

    correct = 0
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
                call = parse_tool_call(gen)
                if call is not None and call[0] is not None:
                    name_ok = call[0] == examples[j]["gold_name"]
                    if name_ok and call[1] == examples[j]["gold_args"]:
                        correct += 1

    model.config.use_cache = prev_cache
    if was_training:
        model.train()
    return correct / len(examples)


# ---------------------------------------------------------------------------
# Standalone smoke (loads a tiny model + n examples, prints the float).
# ---------------------------------------------------------------------------
def _smoke(base: str = "Qwen/Qwen3-0.6B", n: int = 4, device: Optional[str] = None) -> None:
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tooluse-eval:smoke] base={base} n={n} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    examples = load_tooluse_eval(tok, n=n)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    t0 = time.time()
    score = eval_tooluse_inproc(model, tok, examples, device)
    dt = time.time() - t0
    print(f"[tooluse-eval:smoke] tooluse_acc={score:.4f}  ({len(examples)} ex, {dt:.1f}s)", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test tooluse_exec_eval (tiny model, few examples).")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    _smoke(a.base, a.n, a.device)
