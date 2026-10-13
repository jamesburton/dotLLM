"""MoTE **accuracy** eval: GSM8K exact-match + HumanEval pass@1 by real generation.

The existing ``mote_eval.py`` only measures PPL / expert-entropy / router-dependence.
The MoE-granularity research (2508.18672, 2604.07030) shows PPL-optimal != accuracy-optimal,
and our ternary low-rank-adapter thread already proved PPL gains need NOT translate to task
accuracy (2B GSM8K: 47.4% base == 47.4% expert). This tool closes that gap for the *real*
full-FFN-expert Track-M MoTE: does it add TASK ACCURACY where adapters did not?

It reuses the validated building blocks rather than re-implementing them:
  * ``mote_eval._load_student``            — base -> build_mote -> shim-wrap -> load adapter (device-verified)
  * ``pdiff_ternary_lora.enable_inference_cache`` — caches frozen ternary weights once (~11.6x gen speedup);
                                             applies to the AutoBitLinear experts too.
  * ``tlora_capability_hard._run_math_gsm8k`` / ``_run_coding_humaneval`` — the exact GSM8K/HumanEval
                                             scorers used across the capability thread (same harness).

By default it evaluates the MoTE **and** the frozen dense base with the *same* harness, so the
delta is confound-free (base GSM8K measured here, not quoted from a different eval).

CLI
---
  python scripts/lora/mote_accuracy_eval.py \
      --adapter .docs/mote/c0_local \
      --n-math 100 --n-code 40 \
      --out .docs/mote/c0_local/accuracy.json

  # fast first datapoint (GSM8K only, no base):
  python scripts/lora/mote_accuracy_eval.py --adapter .docs/mote/c0_local --n-math 50 --no-eval-base --skip-code

Reads from <adapter>/:
  mote_config.json   — n_experts, top_k, shared, layers, base
  adapter_weights.pt — trainable MoTE weights (router + routed experts)

Writes (with --out):
  <out> — JSON: {mote_config, mote: {gsm8k, humaneval}, base: {...}, delta: {...}}
"""

# Windows / mismatched-triton: transformers' BitNet forward may be torch.compile-wrapped;
# fall back to eager instead of crashing on an inductor/triton import error. (Mirrors mote_eval.py.)
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sibling scripts: scripts/lora (this dir) and scripts/diffusion.
_LORA_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFUSION_DIR = os.path.join(os.path.dirname(_LORA_DIR), "diffusion")
sys.path.insert(0, _LORA_DIR)
sys.path.insert(0, _DIFFUSION_DIR)

from mote_eval import _load_student  # noqa: E402
from pdiff_ternary_lora import enable_inference_cache  # noqa: E402
from tlora_capability_hard import (  # noqa: E402
    _run_math_gsm8k, _run_coding_humaneval,
    _extract_gsm8k_gold, _extract_gsm8k_predicted,
)

_GSM8K_PROMPT = (
    "Solve the following math problem. Show your reasoning, then on the last line write "
    "the final answer as: #### <number>\n\n"
)

BASE_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"


def _free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_tokenizer(base_name: str):
    tok = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _gsm8k_prompts(tokenizer, n_math: int):
    """Return (chat-templated prompts, gold answers, questions) for the first ``n_math`` test items."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)[:n_math]
    prompts, golds, questions = [], [], []
    for item in problems:
        golds.append(_extract_gsm8k_gold(item["answer"]))
        questions.append(item["question"])
        messages = [{"role": "user", "content": _GSM8K_PROMPT + item["question"]}]
        prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    return prompts, golds, questions


def _generate_batched(model, tokenizer, prompts, max_new_tokens, device):
    """Greedy-decode a batch of prompts (left-padded — required for decoder-only batched gen).

    Batched greedy is numerically equivalent to per-item greedy (padding is attention-masked),
    but amortizes the weight reads across the batch — ~batch× throughput on a bandwidth-bound
    BitNet decode (batch=32 measured ~32× on a 3060). Returns the new text per prompt.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id,
            )
        return [tokenizer.decode(out[i][ids.shape[1]:], skip_special_tokens=True) for i in range(len(prompts))]
    finally:
        tokenizer.padding_side = prev_side


def _run_gsm8k_batched(model, tokenizer, device, n_math: int, batch_size: int) -> dict:
    """Batched GSM8K exact-match accuracy (same scoring as tlora_capability_hard._run_math_gsm8k)."""
    model.eval()
    prompts, golds, questions = _gsm8k_prompts(tokenizer, n_math)
    print(f"[hard_eval] GSM8K batched: {len(prompts)} problems, batch={batch_size}, max_new_tokens=256", flush=True)
    correct = 0
    results = []
    for b in range(0, len(prompts), batch_size):
        comps = _generate_batched(model, tokenizer, prompts[b:b + batch_size], 256, device)
        for j, comp in enumerate(comps):
            gold = golds[b + j]
            pred = _extract_gsm8k_predicted(comp)
            ok = gold is not None and pred is not None and gold == pred
            correct += int(ok)
            print(f"  [{'ok' if ok else 'no'}] gold={gold!r} pred={pred!r} q={questions[b + j][:40]!r}", flush=True)
            results.append({"question": questions[b + j], "gold": gold, "predicted": pred, "correct": ok})
    n = len(prompts)
    return {"category": "math_gsm8k", "n": n, "correct": correct,
            "score": round(correct / n, 4) if n else 0.0, "items": results}


def _eval_model(model, tokenizer, device, n_math: int, n_code: int, skip_code: bool, batch_size: int = 1) -> dict:
    """Enable the inference cache, then run GSM8K (+ optionally HumanEval) on ``model``."""
    model.eval()
    # Generation wants the KV cache on (mote_eval disabled it for full-seq PPL).
    model.config.use_cache = True
    n_cached = enable_inference_cache(model)
    print(f"[mote_acc] inference cache enabled on {n_cached} BitNet linear modules", flush=True)

    out = {}
    t0 = time.time()
    if batch_size > 1:
        out["gsm8k"] = _run_gsm8k_batched(model, tokenizer, device, n_math, batch_size)
    else:
        out["gsm8k"] = _run_math_gsm8k(model, tokenizer, device, n_math)
    print(f"[mote_acc] GSM8K score={out['gsm8k']['score']:.4f} "
          f"({out['gsm8k']['correct']}/{out['gsm8k']['n']}) in {time.time()-t0:.0f}s", flush=True)

    if not skip_code:
        t0 = time.time()
        out["humaneval"] = _run_coding_humaneval(model, tokenizer, device, n_code)
        print(f"[mote_acc] HumanEval pass@1={out['humaneval']['score']:.4f} "
              f"({out['humaneval']['correct']}/{out['humaneval']['n']}) in {time.time()-t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MoTE accuracy eval: GSM8K + HumanEval by generation.")
    ap.add_argument("--adapter", required=True,
                    help="Trained MoTE dir (mote_config.json + adapter_weights.pt).")
    ap.add_argument("--base", default=None,
                    help="Override base model id (default: from mote_config.json, else BASE_MODEL_ID).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-math", type=int, default=100, help="Number of GSM8K problems.")
    ap.add_argument("--n-code", type=int, default=40, help="Number of HumanEval problems.")
    ap.add_argument("--skip-code", action="store_true", help="GSM8K only (fast first datapoint).")
    ap.add_argument("--no-eval-base", dest="eval_base", action="store_false",
                    help="Skip the same-harness dense-base comparison.")
    ap.set_defaults(eval_base=True)
    ap.add_argument("--out", default=None, help="Write results JSON here.")
    ap.add_argument("--batch", type=int, default=32,
                    help="GSM8K generation batch size (1 = original per-item path; 32 ~= 32x on 3060).")
    args = ap.parse_args()

    device = torch.device(args.device)

    cfg_path = os.path.join(args.adapter, "mote_config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        mote_cfg = json.load(fh)
    base_name = args.base or mote_cfg.get("base", BASE_MODEL_ID)
    adapter_path = os.path.join(args.adapter, "adapter_weights.pt")

    print(f"[mote_acc] adapter={args.adapter}  base={base_name}  device={device}", flush=True)
    print(f"[mote_acc] config: n_experts={mote_cfg['n_experts']} top_k={mote_cfg['top_k']} "
          f"shared={mote_cfg['shared']} layers={mote_cfg['layers']} "
          f"tokens_trained={mote_cfg.get('tokens_trained')}", flush=True)

    tokenizer = _load_tokenizer(base_name)

    # --- MoTE ---
    print("\n[mote_acc] === evaluating MoTE ===", flush=True)
    student = _load_student(
        base_name, False, None,
        mote_cfg["layers"], mote_cfg["n_experts"], mote_cfg["top_k"], mote_cfg["shared"],
        adapter_path, device,
    )
    mote_res = _eval_model(student, tokenizer, device, args.n_math, args.n_code, args.skip_code, args.batch)
    _free(student)

    # --- dense base (same harness) ---
    base_res = None
    if args.eval_base:
        print("\n[mote_acc] === evaluating dense base (same harness) ===", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        base_res = _eval_model(base_model, tokenizer, device, args.n_math, args.n_code, args.skip_code, args.batch)
        _free(base_model)

    # --- summary ---
    delta = {}
    print("\n[mote_acc] ================ SUMMARY ================", flush=True)
    print(f"  MoTE   GSM8K: {mote_res['gsm8k']['score']:.4f} "
          f"({mote_res['gsm8k']['correct']}/{mote_res['gsm8k']['n']})", flush=True)
    if base_res is not None:
        print(f"  base   GSM8K: {base_res['gsm8k']['score']:.4f} "
              f"({base_res['gsm8k']['correct']}/{base_res['gsm8k']['n']})", flush=True)
        delta["gsm8k"] = round(mote_res["gsm8k"]["score"] - base_res["gsm8k"]["score"], 4)
        print(f"  delta  GSM8K: {delta['gsm8k']:+.4f}", flush=True)
    if "humaneval" in mote_res:
        print(f"  MoTE   HumanEval: {mote_res['humaneval']['score']:.4f}", flush=True)
        if base_res is not None and "humaneval" in base_res:
            delta["humaneval"] = round(mote_res["humaneval"]["score"] - base_res["humaneval"]["score"], 4)
            print(f"  base   HumanEval: {base_res['humaneval']['score']:.4f}", flush=True)
            print(f"  delta  HumanEval: {delta['humaneval']:+.4f}", flush=True)
    print("[mote_acc] =========================================", flush=True)

    if args.out:
        payload = {
            "adapter": args.adapter,
            "base": base_name,
            "mote_config": mote_cfg,
            "n_math": args.n_math,
            "n_code": None if args.skip_code else args.n_code,
            "mote": mote_res,
            "dense_base": base_res,
            "delta": delta,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[mote_acc] results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
