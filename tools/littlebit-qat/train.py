"""LittleBit QAT with a frozen FP teacher (KL on logits + 10 * sum of per-layer hidden-state MSE).

Differences from the spec recipe are deliberate and listed in README.md (fp32 master weights + bf16
compute, chunked KL, token budget sized to the 3060).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from littlebit import (LittleBitConfig, LittleBitLinear, convert_model, export_checkpoint, quantised_modules)


# ----------------------------------------------------------------------------- hidden-state capture
class HiddenCapture:
    """Collects decoder-layer outputs 0..L-2 and the post-final-norm state (== HF hidden_states[1:]).
    Keyed by index so gradient-checkpoint recomputation overwrites instead of appending."""

    def __init__(self, model):
        self.d: dict[int, torch.Tensor] = {}
        layers = model.model.layers
        self.n = len(layers)
        for i, l in enumerate(layers[:-1]):
            l.register_forward_hook(self._mk(i))
        model.model.norm.register_forward_hook(self._mk(self.n - 1))

    def _mk(self, i):
        def hook(_m, _in, out):
            self.d[i] = out[0] if isinstance(out, tuple) else out
        return hook

    def get(self):
        return [self.d[i] for i in range(self.n)]


def _kl_chunk(hs, ht, w):
    ls = torch.log_softmax((hs @ w.t()).float(), -1)
    lt = torch.log_softmax((ht @ w.t()).float(), -1)
    return (lt.exp() * (lt - ls)).sum()


def chunked_kl(hs: torch.Tensor, ht: torch.Tensor, lm_w: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """sum over (batch, positions, vocab) of KL(teacher || student), lm_head applied per position chunk
    under activation checkpointing so full-vocab fp32 logits are never resident (vocab 152k x seq 2048 = 1.2 GB each)."""
    tot = hs.new_zeros((), dtype=torch.float32)
    T = hs.shape[1]
    for s in range(0, T, chunk):
        a, b = hs[:, s:s + chunk], ht[:, s:s + chunk]
        tot = tot + checkpoint(_kl_chunk, a, b, lm_w, use_reentrant=False)
    return tot


def distill_loss(student, teacher, cap_s: HiddenCapture, cap_t: HiddenCapture, ids, l2l_scale=10.0,
                 kd_reduction="spec", kl_chunk=256):
    with torch.no_grad():
        teacher.model(input_ids=ids, use_cache=False)
        ht = cap_t.get()
    student.model(input_ids=ids, use_cache=False)
    hs = cap_s.get()
    B, T = ids.shape
    kl = chunked_kl(hs[-1], ht[-1].detach(), student.lm_head.weight, kl_chunk)
    # spec: 'batchmean' on 3-D logits divides by batch only (sum over positions); 'token' = per-token mean.
    kl = kl / B if kd_reduction == "spec" else kl / (B * T)
    mse = sum(F.mse_loss(a.float(), b.detach().float()) for a, b in zip(hs, ht))
    return kl + l2l_scale * mse, kl.detach(), mse.detach()


# ----------------------------------------------------------------------------- optimiser with fp32 masters
class MasterAdamW:
    """AdamW over fp32 master copies of any bf16 trainable params (norms, biases); fp32 params are used directly.
    bf16 params cannot absorb lr~4e-5 updates (bf16 resolution ~ 0.4%)."""

    def __init__(self, params, lr, betas=(0.9, 0.999), wd=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.masters = [p if p.dtype == torch.float32 else p.detach().float().clone().requires_grad_(True)
                        for p in self.params]
        self.opt = torch.optim.AdamW(self.masters, lr=lr, betas=betas, weight_decay=wd)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
        for m in self.masters:
            m.grad = None

    def step(self, clip=1.0):
        for p, m in zip(self.params, self.masters):
            if m is not p:
                m.grad = None if p.grad is None else p.grad.float()
        gn = torch.nn.utils.clip_grad_norm_(self.masters, clip) if clip else None
        self.opt.step()
        for p, m in zip(self.params, self.masters):
            if m is not p:
                p.data.copy_(m.data)
        return gn

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr


def lr_at(step, total, base, warmup_ratio):
    w = max(1, int(total * warmup_ratio))
    if step < w:
        return base * (step + 1) / w
    return 0.5 * base * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def freeze_for_qat(student):
    for p in student.parameters():
        p.requires_grad = False
    for n, m in student.named_modules():
        if isinstance(m, LittleBitLinear):
            for p in m.parameters():
                p.requires_grad = True       # latent U,V, scales, bias
        elif "norm" in type(m).__name__.lower():
            for p in m.parameters():
                p.requires_grad = True       # RMSNorm / LayerNorm weights


def build_student(teacher, cfg: LittleBitConfig, device, log=print):
    student = copy.deepcopy(teacher)
    student.to(device)
    ranks = convert_model(student, cfg, log)
    freeze_for_qat(student)
    return student, ranks


# ----------------------------------------------------------------------------- data
class BlockData:
    def __init__(self, path, seq):
        self.a = np.memmap(path, dtype=np.uint32, mode="r")
        self.seq = seq
        self.n = len(self.a) // seq

    def batch(self, idx, device):
        rows = [np.asarray(self.a[i * self.seq:(i + 1) * self.seq], dtype=np.int64) for i in idx]
        return torch.from_numpy(np.stack(rows)).to(device)


def eval_ids(tok, seq):
    from datasets import load_dataset
    return tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids


@torch.no_grad()
def eval_ppl_live(model, ids, seq, windows, dev):
    was = model.training
    model.eval()
    n = min(ids.shape[1] // seq, windows)
    nll = 0.0
    for i in range(n):
        x = ids[:, i * seq:(i + 1) * seq].to(dev)
        lg = model(input_ids=x, use_cache=False).logits.float()
        nll += F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="sum").item()
        del lg
    model.train(was)
    return math.exp(nll / (n * (seq - 1)))


def save_ckpt(out, step, cur, opt, keep=2):
    d = os.path.join(out, "ckpt")
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"tmp_{step}.pt")
    torch.save(dict(step=step, cur=cur, params=[p.detach().cpu() for p in opt.params],
                    masters=[m.detach().cpu() for m in opt.masters], opt=opt.opt.state_dict()), tmp)
    os.replace(tmp, os.path.join(d, f"step_{step:06d}.pt"))
    old = sorted(f for f in os.listdir(d) if f.startswith("step_"))
    for f in old[:-keep]:
        os.remove(os.path.join(d, f))


def load_latest_ckpt(out, opt):
    d = os.path.join(out, "ckpt")
    fs = sorted(f for f in os.listdir(d) if f.startswith("step_")) if os.path.isdir(d) else []
    if not fs:
        return 0, 0
    s = torch.load(os.path.join(d, fs[-1]), map_location="cpu")
    with torch.no_grad():
        for p, v in zip(opt.params, s["params"]):
            p.copy_(v)
        for m, v in zip(opt.masters, s["masters"]):
            m.copy_(v)
    opt.opt.load_state_dict(s["opt"])
    print(f"resumed from {fs[-1]}", flush=True)
    return s["step"], s["cur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", required=True, help="uint32 token .bin from data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eff-bit", type=float, default=0.55)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-itq", action="store_true")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000, help="optimizer steps (batch*accum sequences each)")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--l2l", type=float, default=10.0)
    ap.add_argument("--kd-reduction", default="spec", choices=["spec", "token"])
    ap.add_argument("--ckpt-every", type=int, default=100, help="resumable checkpoint (keeps last 2)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--export-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop (after a checkpoint+export) past this wall time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    dev = a.device
    cfg = LittleBitConfig(eff_bit=a.eff_bit, residual=not a.no_residual, use_itq=not a.no_itq)
    print("loading teacher", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(a.model)
    ev_ids = eval_ids(tok, a.seq)
    t0 = time.time()
    student, ranks = build_student(teacher, cfg, dev)
    print(f"init done in {time.time() - t0:.0f}s", flush=True)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    cap_s, cap_t = HiddenCapture(student), HiddenCapture(teacher)
    opt = MasterAdamW([p for p in student.parameters()], a.lr)
    print(f"trainable params {sum(p.numel() for p in opt.params) / 1e6:.1f}M", flush=True)

    data = BlockData(a.data, a.seq)
    order = np.random.default_rng(a.seed).permutation(data.n)
    per_step = a.batch * a.accum
    tok_per_step = per_step * a.seq
    print(f"blocks {data.n}  steps {a.steps}  tokens {a.steps * tok_per_step / 1e6:.1f}M", flush=True)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    start, cur = (load_latest_ckpt(a.out, opt) if a.resume else (0, 0))

    def jlog(rec, echo=True):
        log.write(json.dumps(rec) + "\n"); log.flush()
        if echo:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

    def do_eval(step):
        t = time.time()
        jlog(dict(type="eval", step=step, tokens=step * tok_per_step,
                  wikitext2_ppl=eval_ppl_live(student, ev_ids, a.seq, a.eval_windows, dev), eval_s=time.time() - t))

    def do_export(step):
        rep = export_checkpoint(student, os.path.join(a.out, "export"), cfg, tok)
        print("exported", step, rep, flush=True)

    tstart = time.time()
    for step in range(start, a.steps):
        opt.set_lr(lr_at(step, a.steps, a.lr, a.warmup))
        opt.zero_grad()
        tl = tk = tm = 0.0
        for _ in range(a.accum):
            idx = [order[(cur + j) % data.n] for j in range(a.batch)]
            cur += a.batch
            ids = data.batch(idx, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                loss, kl, mse = distill_loss(student, teacher, cap_s, cap_t, ids, a.l2l, a.kd_reduction)
            (loss / a.accum).backward()
            tl += loss.item() / a.accum; tk += kl.item() / a.accum; tm += mse.item() / a.accum
        gn = opt.step()
        el = time.time() - tstart
        done = step + 1
        jlog(dict(type="train", step=done, tokens=done * tok_per_step, loss=tl, kl=tk, mse=tm, gnorm=float(gn),
                  lr=opt.opt.param_groups[0]["lr"], tok_s=(done - start) * tok_per_step / el, elapsed=el,
                  peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else 0),
             echo=(done <= 5 or done % 10 == 0))
        timeout = a.max_hours > 0 and el > a.max_hours * 3600
        last = done == a.steps or timeout
        if done % a.eval_every == 0 or last:
            do_eval(done)
        if done % a.ckpt_every == 0 or last:
            save_ckpt(a.out, done, cur, opt)
            print("checkpoint", done, flush=True)
        if done % a.export_every == 0 or last:
            do_export(done)
        if timeout:
            print("max-hours reached, stopping", flush=True)
            break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
