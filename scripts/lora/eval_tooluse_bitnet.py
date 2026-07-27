"""BitNet tool-use eval — constrained-decoding rescue demo (U5 Phase C, issue #112).

The capability-bound BitNet b1.58 2B base "cannot reliably emit valid <tool_call> JSON"
(EVAL_bitnet.md). dotLLM constrained decoding (`--tool-choice required`, #106/#104)
guarantees a valid tool call regardless of the base. This measures the 2x2:

    base / +adapter   x   unconstrained / --tool-choice required

on held-out glaive_func_calling rows, scoring: a valid tool call was produced,
function-name correct, arguments JSON-equal to the gold call. The headline is that
the *constrained* configs produce valid tool calls where the unconstrained ones fail.

Each row's tools are passed via `--tools` (so dotLLM renders the BitNet tool template,
#101, the same form the tool-use adapter was trained on) and the raw user query is the
prompt. BitNet needs `--repeat-penalty 1.3 --repeat-last-n 256` for coherent decode.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface python eval_tooluse_bitnet.py \
     --gguf <i2_s.gguf> --lora <bitnet_tooluse_dir> [--n 10] [--start 3000]
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_serve
from eval_tooluse import parse_tool_call  # gold/text tool-call parser


def first_user_query(conv):
    for turn in conv:
        if turn["from"] in ("human", "user"):
            return turn.get("value", turn.get("content", ""))
    return None


def gold_call(conv):
    for turn in conv:
        if turn["from"] == "gpt":
            c = turn.get("value", turn.get("content", ""))
            if c.lstrip().startswith("<tool_call>"):
                return parse_tool_call(c)
    return None


import re as _re
# The forced tool-call skeleton always starts `{"name": "<fn>"` even when the weak
# base then rambles inside a free-form argument *value* and never terminates. This
# captures the function name from that leading skeleton.
_LEADING_NAME = _re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"')


def served(result):
    """Return (got_name, full_call) where got_name is the forced/parsed function name
    (or None) and full_call is a complete parsed (name, args) tuple (or None)."""
    full = None
    tcs = result.get("toolCalls") or []
    if tcs:
        tc = tcs[0]
        name = tc.get("functionName") or tc.get("name")
        args = tc.get("arguments")
        if isinstance(args, str):
            try: args = json.loads(args)
            except json.JSONDecodeError: pass
        if name:
            full = (name, args)
    text = result.get("text", "")
    if full is None:
        full = parse_tool_call(text)
    got_name = full[0] if full else None
    if got_name is None:  # incomplete (rambling value) — recover name from the forced skeleton
        m = _LEADING_NAME.match(text)
        if m:
            got_name = m.group(1)
    return got_name, full


CONFIGS = [
    ("base_unconstrained", {"lora": None, "tool_choice": None}),
    ("base_constrained",   {"lora": None, "tool_choice": "required"}),
    ("adapter_unconstrained", {"lora": "ADAPTER", "tool_choice": None}),
    ("adapter_constrained",   {"lora": "ADAPTER", "tool_choice": "required"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora", required=True, help="BitNet tool-use adapter dir")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / ".docs" / "eval" / "tooluse_bitnet.json"))
    args = ap.parse_args()

    import datasets
    ds = datasets.load_dataset("NousResearch/hermes-function-calling-v1", "glaive_func_calling",
                               split=f"train[{args.start}:{args.start + args.n * 3}]")
    examples = []
    for row in ds:
        conv = row.get("conversations") or []
        q = first_user_query(conv)
        g = gold_call(conv)
        toolspec = row.get("tools")
        if q and g and g[0] and toolspec:
            examples.append((q, g, toolspec))
        if len(examples) >= args.n:
            break

    print(f"[bitnet tooluse] {len(examples)} held-out examples (start={args.start})", flush=True)
    # name = correct function name (forced skeleton or parsed); complete = full valid
    # tool-call JSON parsed; args = complete AND arguments equal the gold call.
    agg = {c: {"name": 0, "complete": 0, "args": 0} for c, _ in CONFIGS}
    per = []
    tmpdir = tempfile.mkdtemp()
    for i, (query, (gname, gargs), toolspec) in enumerate(examples):
        tools_path = os.path.join(tmpdir, f"tools_{i}.json")
        Path(tools_path).write_text(toolspec if isinstance(toolspec, str) else json.dumps(toolspec), encoding="utf-8")
        rec = {"i": i, "gold_name": gname, "query": query[:120]}
        for cfg, opts in CONFIGS:
            lora = args.lora if opts["lora"] == "ADAPTER" else None
            got_name, full = None, None
            try:
                res = eval_serve.generate(
                    args.gguf, query, lora=lora, tools_file=tools_path, tool_choice=opts["tool_choice"],
                    device="gpu", max_tokens=args.max_tokens, repeat_penalty=1.3, repeat_last_n=256)
                got_name, full = served(res)
            except Exception as e:
                rec.setdefault("errors", {})[cfg] = str(e)[:200]
            name_ok = got_name == gname
            complete = full is not None and full[0] is not None
            args_ok = complete and full[0] == gname and full[1] == gargs
            if name_ok: agg[cfg]["name"] += 1
            if complete: agg[cfg]["complete"] += 1
            if args_ok: agg[cfg]["args"] += 1
            rec[cfg] = {"got_name": got_name, "name_ok": name_ok, "complete": complete, "args_ok": args_ok}
        per.append(rec)
        line = " ".join(f"{''.join(w[0] for w in c.split('_'))}={'N' if rec[c]['name_ok'] else '.'}" for c, _ in CONFIGS)
        print(f"  [{i+1}/{len(examples)}] gold={gname:<22} {line}", flush=True)

    n = len(examples)
    summary = {c: {k: round(100 * agg[c][k] / n, 1) for k in agg[c]} for c, _ in CONFIGS}
    out = {"n": n, "start": args.start, "summary_pct": summary, "per_example": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== BITNET TOOL-USE (n={n}) — name / complete / args (%) ===")
    print(f"{'config':<24}{'name':>8}{'complete':>10}{'args':>8}")
    for c, _ in CONFIGS:
        s = summary[c]
        print(f"{c:<24}{s['name']:>8}{s['complete']:>10}{s['args']:>8}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
