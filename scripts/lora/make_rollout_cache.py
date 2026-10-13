#!/usr/bin/env python3
"""make_rollout_cache.py — teacher-scored KD cache over GENERATED sequences.

Where ``generate_teacher_cache`` scores ground-truth DATA sequences, this scores sequences the
model *generates itself*, to test two exposure-bias-fighting distillation variants that share
one pipeline (they differ only in *who generates*):

  * **Sequence-level KD** (``--gen-source teacher``): the frozen FP teacher generates the
    continuations — the student learns the teacher's own *generation* distribution rather than
    the data distribution (Kim & Rush 2016).
  * **On-policy / self-conditioning** (``--gen-source student``): a student checkpoint generates
    the continuations, then the teacher *scores* them — the student learns to recover from its
    own trajectory (the AR analog of diffusion self-conditioning).
  * **Combined** (``--gen-source combined``): half teacher-generated, half student-generated.

Generator and teacher are never co-resident (generate → free → score), so the FP teacher fits
the 12 GB 3060. Output is byte-compatible with ``train_from_cache`` (same shard dict + meta.json),
and is **completion-masked**: ``loss_mask=1`` on the generated continuation only (prompt +
padding = 0), so ``train_from_cache --completion-mask`` distills on the generation distribution.

  python scripts/lora/make_rollout_cache.py --gen-source teacher \
      --base Qwen/Qwen3-1.7B --out .docs/bitdistill/rollout_seqkd \
      --unique-tokens 0.5e6 --seq-len 512 --max-new 256
  # student / on-policy:
  #   --gen-source student --gen-from .docs/bitdistill/p1v2_1p7b_warmstart/ckpt_5000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitdistill as bd            # noqa: E402
import bitdistill_data as bdata    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt pool — GSM8K questions + capability-mix (instruction), chat-templated.
# ---------------------------------------------------------------------------
def build_prompt_pool(tok, n_gsm8k: int, n_instruction: int, seed: int = 0) -> list:
    """Return a list of prompt token-id lists (chat-templated, generation-prompt appended)."""
    has_tpl = getattr(tok, "chat_template", None) is not None

    def templ(user_text: str) -> list:
        if has_tpl:
            text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                           add_generation_prompt=True, tokenize=False)
            return tok(text, add_special_tokens=False)["input_ids"]
        return tok(user_text, add_special_tokens=True)["input_ids"]

    prompts, srcs = [], []
    # GSM8K train questions
    try:
        gs = bdata.load_gsm8k(tok, n=n_gsm8k, split="train", few_shot=0)
        for e in gs:
            q = e.get("question")
            if q:
                prompts.append(templ(q)); srcs.append("gsm8k")
    except Exception as exc:  # noqa: BLE001
        print(f"[rollout] gsm8k prompts unavailable ({exc}); skipping.", flush=True)
    # Instruction-following prompts (capability mix)
    if n_instruction > 0:
        try:
            import instruction_exec_eval as iue
            ins = iue.load_instruction_eval(tok, n=n_instruction)
            for e in ins:
                p = e.get("prompt") or e.get("instruction")
                if p:
                    prompts.append(templ(p)); srcs.append("instruction")
        except Exception as exc:  # noqa: BLE001
            print(f"[rollout] instruction prompts unavailable ({exc}); skipping.", flush=True)

    if not prompts:
        raise RuntimeError("no prompts assembled for rollout generation")
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(prompts), generator=g).tolist()
    prompts = [prompts[i] for i in order]
    from collections import Counter
    print(f"[rollout] prompt pool: {len(prompts)} prompts {dict(Counter(srcs))}", flush=True)
    return prompts


# ---------------------------------------------------------------------------
# Generation phase — batched, cycle prompts until `unique_tokens` completion tokens.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_rollouts(model, tok, prompts, *, unique_tokens: int, seq_len: int, max_new: int,
                      gen_batch: int, sample: bool, temperature: float, device, tag: str) -> list:
    """Return a list of (ids[seq_len] int64, mask[seq_len] int8): [prompt+continuation] right-padded,
    mask=1 on continuation tokens only (prompt + pad = 0)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    seqs, produced, pi, t0 = [], 0, 0, time.time()
    while produced < unique_tokens:
        batch_prompts = [prompts[(pi + k) % len(prompts)] for k in range(gen_batch)]
        pi += gen_batch
        # cap prompt length so there's room for a continuation inside seq_len
        batch_prompts = [p[-(seq_len - max_new):] if len(p) > seq_len - max_new else p
                         for p in batch_prompts]
        maxlp = max(len(p) for p in batch_prompts)
        input_ids = torch.full((gen_batch, maxlp), pad_id, dtype=torch.long)
        attn = torch.zeros((gen_batch, maxlp), dtype=torch.long)
        for k, p in enumerate(batch_prompts):  # LEFT-pad for generation
            input_ids[k, maxlp - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[k, maxlp - len(p):] = 1
        out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                             max_new_tokens=max_new, do_sample=sample,
                             temperature=temperature if sample else None,
                             pad_token_id=pad_id)
        gen_part = out[:, maxlp:].cpu()  # [B, <=max_new] continuation only
        for k, p in enumerate(batch_prompts):
            cont = gen_part[k].tolist()
            if tok.eos_token_id is not None and tok.eos_token_id in cont:  # trim at first EOS (inclusive)
                cont = cont[:cont.index(tok.eos_token_id) + 1]
            full = list(p) + cont
            full = full[:seq_len]
            lp = min(len(p), len(full))
            ncont = len(full) - lp
            if ncont <= 0:
                continue
            ids = torch.full((seq_len,), pad_id, dtype=torch.long)
            ids[:len(full)] = torch.tensor(full, dtype=torch.long)
            mask = torch.zeros(seq_len, dtype=torch.int8)
            mask[lp:lp + ncont] = 1  # completion tokens only
            seqs.append((ids, mask))
            produced += ncont
        if len(seqs) % (gen_batch * 8) < gen_batch:
            rate = produced / max(time.time() - t0, 1e-6)
            print(f"[rollout:{tag}] {produced}/{unique_tokens} completion tok, {len(seqs)} seqs, "
                  f"{rate:.0f} tok/s", flush=True)
    model.config.use_cache = prev_cache
    return seqs


# ---------------------------------------------------------------------------
# Scoring phase — teacher forward over generated seqs → same shard format as generate_teacher_cache.
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_and_cache(teacher, seqs, out_dir, *, bs: int, seq_len: int, K: int, tau: float,
                    base: str, device) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    n_batches = len(seqs) // bs
    mass_sum, mass_cnt = 0.0, 0
    for i in range(n_batches):
        chunk = seqs[i * bs:(i + 1) * bs]
        batch = torch.stack([c[0] for c in chunk]).to(device)          # [B, T]
        lmask = torch.stack([c[1] for c in chunk]).to(torch.int8)      # [B, T]
        logits = teacher(input_ids=batch, use_cache=False).logits.float()
        lse = torch.logsumexp(logits / tau, dim=-1)
        topv, topi = logits.topk(K, dim=-1)
        mass = (topv / tau - lse.unsqueeze(-1)).exp().sum(-1)
        mass_sum += float(mass.mean()); mass_cnt += 1
        torch.save({"input_ids": batch.to(torch.int32).cpu(),
                    "topk_logits": topv.to(torch.float16).cpu(),
                    "topk_idx": topi.to(torch.int32).cpu(),
                    "lse": lse.to(torch.float32).cpu(),
                    "loss_mask": lmask.cpu()},
                   os.path.join(out_dir, f"batch_{i:06d}.pt"))
        if i % 50 == 0 or i == n_batches - 1:
            print(f"[score] {i + 1}/{n_batches}  mean top-{K} mass={mass_sum / max(mass_cnt,1):.4f}",
                  flush=True)
    tokens = sum(int(c[1].sum()) for c in seqs[:n_batches * bs])
    meta = {"n_batches": n_batches, "batch_size": bs, "seq_len": seq_len, "top_k": K, "tau": tau,
            "base": base, "tokens": tokens, "mean_topk_mass": mass_sum / max(mass_cnt, 1),
            "rollout_cache": True}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _load_student(base, ckpt, device, dtype):
    m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    m.config.use_cache = False
    bd.convert_to_bitnet_student(m)
    sd = torch.load(os.path.join(ckpt, "student_state.pt"), map_location="cpu")
    m.load_state_dict(sd); del sd
    bd.set_quant_alpha(m, 1.0)          # full ternary forward (serve parity)
    return m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-source", choices=["teacher", "student", "combined"], required=True)
    ap.add_argument("--gen-from", default=None, help="student ckpt dir (for student/combined)")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unique-tokens", type=float, default=0.5e6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--cache-bs", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true", default=True)
    ap.add_argument("--greedy", dest="sample", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--n-gsm8k", type=int, default=4000)
    ap.add_argument("--n-instruction", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.base)
    prompts = build_prompt_pool(tok, args.n_gsm8k, args.n_instruction, seed=args.seed)
    uniq = int(args.unique_tokens)

    seqs = []
    # --- generation phase (generator freed before the teacher loads) ---
    if args.gen_source in ("teacher", "combined"):
        need = uniq // 2 if args.gen_source == "combined" else uniq
        print(f"[rollout] teacher generating {need} completion tokens ...", flush=True)
        gen = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
        gen.config.use_cache = True
        seqs += generate_rollouts(gen, tok, prompts, unique_tokens=need, seq_len=args.seq_len,
                                  max_new=args.max_new, gen_batch=args.gen_batch, sample=args.sample,
                                  temperature=args.temperature, device=device, tag="teacher")
        del gen; torch.cuda.empty_cache() if device.type == "cuda" else None
    if args.gen_source in ("student", "combined"):
        if not args.gen_from:
            raise SystemExit("--gen-from <ckpt> required for student/combined")
        need = uniq // 2 if args.gen_source == "combined" else uniq
        print(f"[rollout] student ({args.gen_from}) generating {need} completion tokens ...", flush=True)
        gen = _load_student(args.base, args.gen_from, device, dtype)
        seqs += generate_rollouts(gen, tok, prompts, unique_tokens=need, seq_len=args.seq_len,
                                  max_new=args.max_new, gen_batch=args.gen_batch, sample=args.sample,
                                  temperature=args.temperature, device=device, tag="student")
        del gen; torch.cuda.empty_cache() if device.type == "cuda" else None

    g = torch.Generator().manual_seed(args.seed + 1)
    seqs = [seqs[i] for i in torch.randperm(len(seqs), generator=g).tolist()]
    print(f"[rollout] generated {len(seqs)} sequences; loading teacher to score ...", flush=True)

    # --- scoring phase ---
    teacher = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)
    meta = score_and_cache(teacher, seqs, args.out, bs=args.cache_bs, seq_len=args.seq_len,
                           K=args.top_k, tau=args.tau, base=args.base, device=device)
    print(f"[rollout] DONE -> {args.out}  ({meta['n_batches']} batches, {meta['tokens']} "
          f"completion tokens, mean top-{args.top_k} mass={meta['mean_topk_mass']:.4f})", flush=True)


if __name__ == "__main__":
    main()
