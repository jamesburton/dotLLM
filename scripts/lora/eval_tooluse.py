"""Tool-use eval for the task-LoRA harness (U2 Phase C, Task 8).

Held-out glaive_func_calling rows (disjoint from the train[:3000] used for training),
rendered with the canonical tooluse_render path (template-generated <tools> block),
served base vs +lora, scoring per example:
  (a) a <tool_call> with parseable JSON was emitted,
  (b) the function name matches the gold call,
  (c) the arguments JSON is equal to the gold arguments.

Reports base-vs-adapted accuracy. Writes results JSON + a markdown fragment.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface python eval_tooluse.py \
     --gguf <q4_k_m.gguf> --lora <tooluse_adapter_dir> [--n 20] [--start 3000]
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tooluse_render import extract_tools, _SHAREGPT_ROLE, _strip_tools_block
import eval_serve

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str):
    """Return (name, args_dict) from the first parseable <tool_call> JSON, or None."""
    m = TOOLCALL_RE.search(text)
    if not m:
        # Some outputs may emit a bare JSON object without tags; try that too.
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
    """Render the eval prompt (system+user+tools, add_generation_prompt) and the gold call."""
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
    prompt = tok.apply_chat_template(prompt_msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    gold = parse_tool_call(conv[target].get("value", conv[target].get("content", "")))
    if gold is None or gold[0] is None:
        return None
    return prompt, gold


def score(gguf, prompt, lora, **kw):
    text = eval_serve.generate_text(gguf, prompt, lora=lora, **kw)
    call = parse_tool_call(text)
    return text, call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora", required=True, help="tool-use adapter dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=3000, help="held-out slice start (train[:3000] was used for training)")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / ".docs" / "eval" / "tooluse.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import datasets
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    # Pull a generous slice and keep the first N renderable rows.
    ds = datasets.load_dataset("NousResearch/hermes-function-calling-v1", "glaive_func_calling",
                               split=f"train[{args.start}:{args.start + args.n * 3}]")
    examples = []
    for row in ds:
        try:
            pg = build_prompt_and_gold(tok, row)
        except Exception:
            pg = None
        if pg is not None:
            examples.append(pg)
        if len(examples) >= args.n:
            break

    print(f"[tooluse] evaluating {len(examples)} held-out examples (start={args.start})", flush=True)

    agg = {"base": {"emitted": 0, "name": 0, "args": 0}, "adapter": {"emitted": 0, "name": 0, "args": 0}}
    per = []
    for i, (prompt, (gname, gargs)) in enumerate(examples):
        rec = {"i": i, "gold_name": gname, "gold_args": gargs}
        for cfg, lora in (("base", None), ("adapter", args.lora)):
            _, call = score(args.gguf, prompt, lora, max_tokens=args.max_tokens)
            emitted = call is not None and call[0] is not None
            name_ok = emitted and call[0] == gname
            args_ok = name_ok and call[1] == gargs
            if emitted: agg[cfg]["emitted"] += 1
            if name_ok: agg[cfg]["name"] += 1
            if args_ok: agg[cfg]["args"] += 1
            rec[cfg] = {"name": call[0] if call else None, "args": call[1] if call else None,
                        "emitted": emitted, "name_ok": name_ok, "args_ok": args_ok}
        per.append(rec)
        print(f"  [{i+1}/{len(examples)}] gold={gname} base={rec['base']['name']} adapter={rec['adapter']['name']}", flush=True)

    n = len(examples)
    summary = {cfg: {k: round(100 * v / n, 1) for k, v in agg[cfg].items()} for cfg in agg}
    out = {"n": n, "start": args.start, "summary_pct": summary, "per_example": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== TOOL-USE RESULTS (n=%d) ===" % n)
    print(f"{'metric':<14}{'base %':>10}{'adapter %':>12}")
    for k in ("emitted", "name", "args"):
        print(f"{k:<14}{summary['base'][k]:>10}{summary['adapter'][k]:>12}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
