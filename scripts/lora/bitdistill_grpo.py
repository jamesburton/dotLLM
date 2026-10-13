#!/usr/bin/env python3
"""bitdistill_grpo.py — GRPO smoke test: RL on GSM8K exact-match reward for the ternary student.

Every KD variant in the P3 campaign (control / SeqKD-blend / on-policy) moved the smooth
gold-NLL proxy substantially but left GSM8K exact-match pinned at noise (0.0-0.075) — KD never
rewards the model's own generated correctness, only a teacher-forced proxy for it. This script
tests whether RL (reward = does the model's OWN rollout get the right answer) can move the hard
metric where offline KD couldn't, as a bounded, wall-clock-governed smoke test.

Per-step loop (mirrors train_from_cache's grad-accum pattern, but every step has a generation
phase, so the memory-flush discipline that KD only needed at eval milestones happens every step
here):
  1. no_grad, model.eval(): generate K rollouts per prompt for P prompts (group-relative GRPO).
  2. gc.collect() + torch.cuda.empty_cache() (KV-cache from step 1 must not linger).
  3. score each rollout's exact-match correctness (regex extraction vs GSM8K gold) -> reward.
  4. per-prompt group-relative advantage: (r_i - mean)/(std+eps), zeroed if the group is degenerate
     (all-same reward -> no signal).
  5. model.train(): grad-accum micro-batches of a teacher-forced re-forward over the *fixed*
     generated token ids, REINFORCE-with-baseline loss weighted by advantage (no PPO clipping --
     the policy that generated the rollouts IS the policy being re-scored, so the importance
     ratio is 1 by construction; no old/new-policy drift to clip).
  6. every --kd-anchor-every steps: one micro-batch from the existing combined_seqkd_v2 KD cache,
     to anchor against catastrophic forgetting (the same fix the SeqKD-blend campaign needed for
     pure-rollout training) -- accumulated into the SAME optimizer step as the RL gradient.
  7. clip_grad_norm_ + opt.step().

No frozen-reference KL term (a second full model copy doesn't fit the ~12GB budget alongside the
resident policy) -- see (5) for why the single-step-per-batch design makes this unnecessary.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitdistill as bd            # noqa: E402
import bitdistill_data as bdata    # noqa: E402
import continuous_eval as cve      # noqa: E402
import make_rollout_cache as mrc   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def build_prompt_pool(tok, n: int, seq_len: int, few_shot: int, seed: int) -> list:
    """GSM8K train questions, few-shot CoT prefixed (same regime as the eval_gsm8k milestone,
    so what RL optimizes is measured in the same prompting convention as the go/no-go metric)."""
    ex = bdata.load_gsm8k(tok, n=n, split="train", seq_len=seq_len, few_shot=few_shot)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ex), generator=g).tolist()
    return [ex[i] for i in order]


# ---------------------------------------------------------------------------
# Generation — P prompts x K rollouts each, exact group cardinality (not a token-budget loop).
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_grpo_groups(model, tok, prompt_batch: list, *, K: int, gen_batch: int, seq_len: int,
                         max_new: int, temperature: float, device) -> list:
    """Returns P groups: [{"gold": str, "rollouts": [(ids[seq_len] int64, mask[seq_len] int8), xK]}].
    mask=1 on the generated completion span only (prompt + pad = 0), matching train_from_cache's
    cache format so the same loss-mask convention applies."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    flat_prompts, flat_group = [], []
    for gi, ex in enumerate(prompt_batch):
        p = ex["prompt_ids"].tolist()
        for _ in range(K):
            flat_prompts.append(p)
            flat_group.append(gi)

    rollouts = [None] * len(flat_prompts)
    n_chunks = (len(flat_prompts) + gen_batch - 1) // gen_batch
    for ci, i in enumerate(range(0, len(flat_prompts), gen_batch)):
        t_chunk = time.time()
        chunk = flat_prompts[i:i + gen_batch]
        maxlp = max(len(p) for p in chunk)
        input_ids = torch.full((len(chunk), maxlp), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlp), dtype=torch.long)
        for k, p in enumerate(chunk):          # LEFT-pad for generation
            input_ids[k, maxlp - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[k, maxlp - len(p):] = 1
        out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                             max_new_tokens=max_new, do_sample=True, temperature=temperature,
                             pad_token_id=pad_id)
        print(f"    [gen] chunk {ci + 1}/{n_chunks} ({len(chunk)} seqs, prompt_len={maxlp}) "
              f"took {time.time() - t_chunk:.1f}s", flush=True)
        gen_part = out[:, maxlp:].cpu()
        for k, p in enumerate(chunk):
            cont = gen_part[k].tolist()
            if tok.eos_token_id is not None and tok.eos_token_id in cont:  # trim at first EOS
                cont = cont[:cont.index(tok.eos_token_id) + 1]
            full = (p + cont)[:seq_len]
            lp = min(len(p), len(full))
            ncont = len(full) - lp
            ids = torch.full((seq_len,), pad_id, dtype=torch.long)
            ids[:len(full)] = torch.tensor(full, dtype=torch.long)
            mask = torch.zeros(seq_len, dtype=torch.int8)
            if ncont > 0:
                mask[lp:lp + ncont] = 1        # completion tokens only (RIGHT-packed for training)
            rollouts[i + k] = (ids, mask)

    model.config.use_cache = prev_cache
    if was_training:
        model.train()

    groups = [{"gold": ex["answer"], "rollouts": []} for ex in prompt_batch]
    for row, gi in enumerate(flat_group):
        groups[gi]["rollouts"].append(rollouts[row])
    return groups


# ---------------------------------------------------------------------------
# Reward — shaped binary exact-match, reusing bd._extract_number + the GSM8K gold parse.
# ---------------------------------------------------------------------------
def score_rollout(tok, ids: torch.Tensor, mask: torch.Tensor, gold: str, partial_credit: float) -> float:
    idx = mask.bool()
    if not idx.any():
        return 0.0
    text = tok.decode(ids[idx].tolist(), skip_special_tokens=True).split("Question:")[0]
    pred = bd._extract_number(text)
    if pred == gold and gold != "":
        return 1.0
    if pred != "":
        return partial_credit
    return 0.0


def compute_group_advantages(rewards: list, eps: float = 1e-6) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32)
    std = r.std()
    if std < eps or torch.isnan(std):
        return torch.zeros_like(r)          # degenerate group (all-same reward): no signal
    return (r - r.mean()) / (std + eps)


# ---------------------------------------------------------------------------
# RL loss — REINFORCE-with-baseline over a teacher-forced re-forward of the FIXED rollout ids.
# ---------------------------------------------------------------------------
def rl_microbatch_loss(model, ids: torch.Tensor, mask: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids, use_cache=False).logits
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    cmask = mask[:, 1:].float()
    per_seq_logp = (tok_logp * cmask).sum(-1) / cmask.sum(-1).clamp(min=1.0)   # token-avg, not summed
    return -(advantage * per_seq_logp).mean()


# ---------------------------------------------------------------------------
# KD anchor — anti-forgetting, same fix the SeqKD-blend campaign needed for pure-rollout training.
# ---------------------------------------------------------------------------
def load_kd_anchor_batch(cache_files: list, idx: int) -> dict:
    return torch.load(cache_files[idx % len(cache_files)])


def kd_anchor_loss(model, c: dict, device, tau: float, V: int) -> torch.Tensor:
    batch = c["input_ids"].to(device).long()
    topv = c["topk_logits"].to(device)
    topi = c["topk_idx"].to(device).long()
    lse = c["lse"].to(device)
    s_logits = model(input_ids=batch, use_cache=False).logits
    tgt = batch[:, 1:].contiguous()
    ce = F.cross_entropy(s_logits[:, :-1, :].contiguous().view(-1, V), tgt.reshape(-1))
    Kd = topv.size(-1)
    ld = bd.sparse_logit_kd_loss(s_logits[:, :-1, :].contiguous().view(-1, V),
                                 topv[:, :-1, :].contiguous().view(-1, Kd),
                                 topi[:, :-1, :].contiguous().view(-1, Kd),
                                 lse[:, :-1].contiguous().reshape(-1), tau)
    return ce + 10.0 * ld               # lambda_kd=10, the already-tuned campaign default


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def train_grpo(args) -> int:
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    student = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    student.config.use_cache = False
    info = bd.convert_to_bitnet_student(student)
    student.to(device)   # convert_to_bitnet_student's fresh BitLinear params default to CPU
    sd = torch.load(os.path.join(args.init_ckpt, "student_state.pt"), map_location="cpu")
    student.load_state_dict(sd); del sd
    bd.set_quant_alpha(student, 1.0)
    student.to(device)   # load_state_dict(map_location="cpu") sd doesn't move params off-device,
                         # but belt-and-suspenders given the bug this just came from
    if args.grad_checkpoint:
        try:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    student.train()
    print(f"[grpo] student: {info['bitlinears']} BitLinears + {info['subnorms']} SubLNs, "
          f"init={args.init_ckpt}", flush=True)

    trainable = [p for p in student.parameters() if p.requires_grad]
    from transformers.optimization import Adafactor
    opt = Adafactor(trainable, lr=args.lr, relative_step=False, scale_parameter=False,
                    warmup_init=False, weight_decay=0.0)

    prompts = build_prompt_pool(tokenizer, args.prompt_pool_n, args.gsm8k_seq_len,
                                args.gsm8k_few_shot, args.seed)
    print(f"[grpo] prompt pool: {len(prompts)} GSM8K train questions", flush=True)

    kd_files = sorted(glob.glob(os.path.join(args.kd_anchor_cache, "batch_*.pt")))
    with open(os.path.join(args.kd_anchor_cache, "meta.json")) as f:
        kd_meta = json.load(f)
    kd_tau = kd_meta["tau"]
    print(f"[grpo] KD anchor cache: {len(kd_files)} shards from {args.kd_anchor_cache}", flush=True)

    nll_ex = cve.load_gsm8k_nll(tokenizer, n=args.eval_n_nll)
    gsm8k_ex = bdata.load_gsm8k(tokenizer, n=args.eval_n, split="test", seq_len=args.gsm8k_seq_len,
                                few_shot=args.gsm8k_few_shot)
    seq_len = args.gsm8k_seq_len + args.rollout_max_new_tokens
    V = student.config.vocab_size

    os.makedirs(args.out, exist_ok=True)
    curve, rolling_rate, pi, kd_i, step = [], [], 0, 0, 0
    t0 = time.time()

    def flush_gpu():
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def milestone(tag: str):
        bd.set_quant_alpha(student, 1.0)
        acc = bd.eval_gsm8k(student, tokenizer, gsm8k_ex, device, max_new_tokens=256, chunk=16)
        nll = cve.eval_gsm8k_nll(student, tokenizer, nll_ex, device)
        rr = sum(rolling_rate[-5:]) / max(len(rolling_rate[-5:]), 1) if rolling_rate else float("nan")
        row = {"step": step, "elapsed_s": time.time() - t0, "gsm8k_acc": acc, "gsm8k_nll": nll,
              "rolling_inloop_reward": rr, "tag": tag}
        curve.append(row)
        print(f"[milestone {tag}] step={step} gsm8k_acc={acc:.4f} gsm8k_nll={nll:.4f} "
              f"rolling_inloop_reward={rr:.4f}", flush=True)
        bd.save_checkpoint(student, os.path.join(args.out, f"ckpt_step{step}"), step, 0,
                           extra={"gsm8k_acc": acc, "gsm8k_nll": nll})
        with open(os.path.join(args.out, "curve.json"), "w") as f:
            json.dump({"base": args.base, "init_ckpt": args.init_ckpt, "curve": curve}, f, indent=2)
        flush_gpu()

    milestone("pre-rl-baseline")   # step 0 sanity check (must reproduce the source checkpoint's numbers)

    while (time.time() - t0) < args.time_budget_min * 60 and (args.max_steps == 0 or step < args.max_steps):
        batch_prompts = [prompts[(pi + k) % len(prompts)] for k in range(args.prompts_per_step)]
        pi += args.prompts_per_step

        groups = generate_grpo_groups(student, tokenizer, batch_prompts, K=args.group_size,
                                      gen_batch=args.gen_batch, seq_len=seq_len,
                                      max_new=args.rollout_max_new_tokens,
                                      temperature=args.temperature, device=device)
        flush_gpu()

        flat_ids, flat_mask, flat_adv, step_rewards = [], [], [], []
        for g in groups:
            rewards = [score_rollout(tokenizer, ids, mask, g["gold"], args.reward_partial_credit)
                      for ids, mask in g["rollouts"]]
            step_rewards.extend(rewards)
            adv = compute_group_advantages(rewards)
            for (ids, mask), a in zip(g["rollouts"], adv):
                flat_ids.append(ids); flat_mask.append(mask); flat_adv.append(a)
        rolling_rate.append(sum(1.0 for r in step_rewards if r == 1.0) / len(step_rewards))

        ids_t = torch.stack(flat_ids)
        mask_t = torch.stack(flat_mask)
        adv_t = torch.stack(flat_adv)
        B = ids_t.size(0)
        mb = args.micro_batch if args.micro_batch > 0 else B

        student.train()
        student.config.use_cache = False
        opt.zero_grad()
        for j in range(0, B, mb):
            sl = slice(j, min(j + mb, B))
            frac = (sl.stop - sl.start) / B
            loss = rl_microbatch_loss(student, ids_t[sl].to(device), mask_t[sl].to(device),
                                      adv_t[sl].to(device))
            (loss * frac).backward()

        if step % args.kd_anchor_every == 0:
            c = load_kd_anchor_batch(kd_files, kd_i); kd_i += 1
            kd_loss = kd_anchor_loss(student, c, device, kd_tau, V)
            kd_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1

        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"  step {step:4d}  elapsed {time.time() - t0:6.0f}s  step_reward_rate="
              f"{rolling_rate[-1]:.3f}  peak_mem={peak:.2f}GB", flush=True)
        flush_gpu()

        if step % args.eval_every_steps == 0:
            milestone(f"step{step}")

    milestone("final")
    print(f"[grpo] done: {step} steps, {time.time() - t0:.0f}s. curve -> {args.out}/curve.json", flush=True)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--group-size", "-K", type=int, default=8)
    ap.add_argument("--prompts-per-step", "-P", type=int, default=3)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--rollout-max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--gsm8k-few-shot", type=int, default=4)
    ap.add_argument("--gsm8k-seq-len", type=int, default=512)
    ap.add_argument("--prompt-pool-n", type=int, default=300)
    ap.add_argument("--reward-partial-credit", type=float, default=0.1)
    ap.add_argument("--kd-anchor-cache", default=".docs/bitdistill/combined_seqkd_v2")
    ap.add_argument("--kd-anchor-every", type=int, default=4)
    ap.add_argument("--eval-every-steps", type=int, default=6)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--eval-n-nll", type=int, default=40)
    ap.add_argument("--time-budget-min", type=float, default=100.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    return train_grpo(args)


if __name__ == "__main__":
    raise SystemExit(main())
