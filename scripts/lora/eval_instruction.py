"""Instruction eval for the task-LoRA harness (U2 Phase C, Task 10).

Serves base vs +lora on a fixed set of instruction-following probes and records the
output pairs for pairwise judging (base vs adapter). The judge is run out-of-band by
the harness operator (an LLM judge / Claude in this environment), per the U2 plan —
this script produces the reproducible (probe, base, adapter) pairs and a JSON file.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface python eval_instruction.py \
     --gguf <q4_k_m.gguf> --lora <instruction_adapter_dir> [--max-tokens 140]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_serve
from transformers import AutoTokenizer

PROBES = [
    "Explain what a hash map is in two sentences.",
    "Write a haiku about autumn leaves.",
    "List three practical tips for staying focused while working from home.",
    "Rewrite this sentence in a formal tone: 'hey can u send me that file asap'.",
    "Summarize the water cycle in one sentence.",
    "Give me a short, friendly reply declining a meeting invitation.",
    "What are two pros and two cons of remote work? Be concise.",
    "Convert this to a polite request: 'give me the report now'.",
    "Explain the difference between weather and climate in two sentences.",
    "Write a one-line motivational message for someone starting a new job.",
    "Describe the taste of coffee to someone who has never had it, in two sentences.",
    "Give three bullet points on how to reduce household energy use.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora", required=True, help="instruction adapter dir")
    ap.add_argument("--max-tokens", type=int, default=140)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / ".docs" / "eval" / "instruction.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    pairs = []
    for i, probe in enumerate(PROBES):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": probe}], add_generation_prompt=True, tokenize=False)
        base = eval_serve.generate_text(args.gguf, prompt, lora=None, max_tokens=args.max_tokens)
        adapt = eval_serve.generate_text(args.gguf, prompt, lora=args.lora, max_tokens=args.max_tokens)
        pairs.append({"i": i, "probe": probe, "base": base.strip(), "adapter": adapt.strip()})
        print(f"  [{i+1}/{len(PROBES)}] {probe[:50]}", flush=True)

    out = {"n": len(PROBES), "pairs": pairs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(PROBES)} probe pairs for pairwise judging)")


if __name__ == "__main__":
    main()
