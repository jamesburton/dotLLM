"""bench_train.py — identity-MoTE TRAINING-throughput micro-benchmark.

Campaign: trackM-mote — pick the fastest Windows-native training config.

What this does
--------------
Builds an identity-MoTE model (depth-expanded BitNet + IdentityMoTEBlock experts,
exactly the artifact ``identity_mote_train.py`` trains) and runs its OWN minimal,
self-contained batched train-step loop::

    forward  ->  lm_loss + route_weight * CE(router_logits, task_label)
             ->  backward
             ->  optimizer.step()

It measures **steady-state throughput (tokens/sec)** — excluding ``--warmup-steps``
warm-up iterations (the ones that pay compile/allocator/cuDNN-autotune costs) — and
**peak VRAM** (``torch.cuda.max_memory_allocated`` measured over the *timed* region
only). Every optimization lever is a CLI flag so a matrix runner can sweep them.

Independence
------------
This file is deliberately independent of ``identity_mote_train.py``'s training loop
(a parallel agent is editing that). It imports ONLY:
  * ``bitnet_depth_expand``          (build the tiny base + depth expansion)
  * ``identity_mote.build_identity_mote``
It uses SYNTHETIC random token batches + round-robin routing labels (throughput is
independent of the data content), so it needs no dataset and no downloads.

CPU smoke (seconds, no GPU, no downloads)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 \
      python scripts/lora/bench_train.py --tiny-random --device cpu \
        --batch-size 2 --seq-len 32 --steps 3 --warmup-steps 1 \
        --n-capability-experts 2 --every 1

Real GPU run (launched separately by the orchestrator — do NOT run here)::

    python scripts/lora/bench_train.py --device cuda \
        --batch-size 4 --seq-len 256 --steps 30 --warmup-steps 5 \
        --every 4 --n-capability-experts 3 \
        --grad-checkpoint on --optim adamw-fused --amp bf16 --tf32 on

Every run appends one JSON line to ``--results-file`` and prints a human summary.
It also prints a single machine-readable line ``BENCH_JSON: {...}`` on stdout that
``bench_matrix.py`` parses.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
import traceback
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# scripts/lora on sys.path (identity_mote, ...) and scripts/ (bitnet_depth_expand).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bitnet_depth_expand as bde  # noqa: E402
from identity_mote import IdentityMoTEBlock, build_identity_mote  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers (inlined — intentionally NOT imported from identity_mote_train)
# ---------------------------------------------------------------------------


def _iter_mote_blocks(model: nn.Module):
    for layer in model.model.layers:
        if isinstance(layer.mlp, IdentityMoTEBlock):
            yield layer.mlp


def _freeze_for_training(model: nn.Module) -> None:
    """Freeze everything; unfreeze routers + capability experts (not the skip expert)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in _iter_mote_blocks(model):
        for p in block.router.parameters():
            p.requires_grad_(True)
        for e in range(1, block.n_experts):  # experts[0] is the frozen skip expert
            for p in block.experts[e].parameters():
                p.requires_grad_(True)


def _supervised_route_loss(model: nn.Module, targets_flat: torch.Tensor) -> torch.Tensor:
    """Mean over identity-MoTE layers of CE(router_logits[N,E], per-token target[N])."""
    total: Optional[torch.Tensor] = None
    n = 0
    for block in _iter_mote_blocks(model):
        logits = block.last_logits
        if logits is None:
            continue
        ce = F.cross_entropy(logits.float(), targets_flat.to(logits.device))
        total = ce if total is None else total + ce
        n += 1
    if total is None:
        return torch.zeros((), device=targets_flat.device)
    return total / n


# ---------------------------------------------------------------------------
# Optimizer factory (records what actually ran)
# ---------------------------------------------------------------------------


def _make_optimizer(choice: str, params, lr: float, device: torch.device):
    """Return (optimizer, actual_name, note). Falls back gracefully; never raises."""
    note = ""
    if choice == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95)), "adamw", note
    if choice == "adamw-fused":
        try:
            opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), fused=True)
            return opt, "adamw-fused", note
        except Exception as exc:  # noqa: BLE001  (fused needs CUDA + supported dtypes)
            note = f"fused unavailable ({type(exc).__name__}: {exc}); fell back to adamw"
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95)), "adamw", note
    if choice == "adafactor":
        try:
            from transformers.optimization import Adafactor
            opt = Adafactor(params, lr=lr, relative_step=False,
                            scale_parameter=False, warmup_init=False)
            return opt, "adafactor", note
        except Exception as exc:  # noqa: BLE001
            note = f"adafactor unavailable ({type(exc).__name__}: {exc}); fell back to adamw"
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95)), "adamw", note
    if choice == "adamw8bit":
        try:
            import bitsandbytes as bnb
            try:
                opt = bnb.optim.PagedAdamW8bit(params, lr=lr, betas=(0.9, 0.95))
            except AttributeError:
                opt = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95))
            return opt, "adamw8bit", note
        except Exception as exc:  # noqa: BLE001
            note = f"bitsandbytes unavailable ({type(exc).__name__}: {exc}); fell back to adamw"
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95)), "adamw", note
    raise ValueError(f"unknown optim {choice!r}")


# ---------------------------------------------------------------------------
# SDPA backend selection
# ---------------------------------------------------------------------------


def _sdpa_context(attn: str):
    """Return a context manager restricting the SDPA backend, plus a label.

    ``auto`` = no restriction (PyTorch chooses). Others force a single backend so a
    GPU run reveals whether flash / mem-efficient actually work on this box.
    """
    if attn == "auto":
        return contextlib.nullcontext(), "auto"
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        backend = mapping[attn]
        return sdpa_kernel([backend]), attn
    except Exception as exc:  # noqa: BLE001
        return contextlib.nullcontext(), f"{attn}(unavailable:{type(exc).__name__})"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_base(args, device: torch.device):
    if args.tiny_random:
        return bde.build_tiny_bitnet().to(device)
    from transformers import AutoModelForCausalLM
    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, device_map={"": device}, **kwargs)
    return model


def _build_identity_mote_model(args, device: torch.device):
    base = _build_base(args, device)
    base.eval()
    # For the tiny synthetic model, try to route attention through SDPA so --attn is meaningful.
    if args.tiny_random:
        try:
            base.config._attn_implementation = "sdpa"
        except Exception:  # noqa: BLE001
            pass

    n_layers_before = base.config.num_hidden_layers
    every = args.every if args.every is not None else 1
    positions = bde.plan_insertions(n_layers_before, every=every, at=None)
    model, info = bde.expand_model(base, positions)
    model = build_identity_mote(
        model,
        inserted_indices=info["inserted_indices"],
        n_capability_experts=args.n_capability_experts,
        capability_init="zero",
        router_identity_bias=0.0,
        top_k=1,
    )
    model.to(device)
    model.config.use_cache = False
    return model, info


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(args) -> dict:
    on = lambda v: v == "on"  # noqa: E731
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    # --- TF32 ---
    torch.backends.cuda.matmul.allow_tf32 = on(args.tf32)
    torch.backends.cudnn.allow_tf32 = on(args.tf32)

    # --- Build model ---
    model, info = _build_identity_mote_model(args, device)
    vocab_size = model.config.vocab_size

    # --- Freeze base; train routers + capability experts ---
    _freeze_for_training(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]

    # --- Optimizer ---
    opt, optim_actual, optim_note = _make_optimizer(args.optim, params, args.lr, device)

    # --- Gradient checkpointing ---
    ckpt = on(args.grad_checkpoint)
    if ckpt:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # --- torch.compile (Windows-support probe) ---
    compiled_ok = False
    compile_note = ""
    want_compile = on(args.compile)
    if want_compile:
        try:
            model = torch.compile(model)
            compiled_ok = True  # call succeeded (may be a no-op under TORCHDYNAMO_DISABLE=1)
            if os.environ.get("TORCHDYNAMO_DISABLE") == "1":
                compile_note = "torch.compile called but TORCHDYNAMO_DISABLE=1 (no-op)"
        except Exception as exc:  # noqa: BLE001
            compile_note = f"{type(exc).__name__}: {exc}"

    # --- AMP autocast ---
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None else contextlib.nullcontext()
    )

    # --- SDPA backend ---
    sdpa_ctx, attn_label = _sdpa_context(args.attn)

    # --- Synthetic batch generator (fixed content — throughput is data-independent) ---
    K = args.n_capability_experts
    g = torch.Generator().manual_seed(1234)

    def make_batch():
        ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), generator=g).to(device)
        # Per-sequence routing labels round-robin over 1..K (0 reserved for skip).
        labels = torch.tensor([(i % K) + 1 for i in range(args.batch_size)], dtype=torch.long)
        targets_flat = labels.view(-1, 1).expand(args.batch_size, args.seq_len).reshape(-1).to(device)
        return ids, targets_flat

    model.train()

    def train_step():
        ids, targets_flat = make_batch()
        opt.zero_grad(set_to_none=True)
        with sdpa_ctx, autocast_ctx:
            logits = model(input_ids=ids, use_cache=False).logits
            lm_loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size).float(),
                ids[:, 1:].reshape(-1),
            )
            route_loss = _supervised_route_loss(model, targets_flat)
            loss = lm_loss + args.route_weight * route_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        return float(loss.detach())

    error = None
    tokens_per_sec = None
    elapsed = None
    peak_vram_gb = None
    last_loss = float("nan")
    try:
        # --- Warmup (untimed; pays compile / allocator / autotune costs) ---
        for _ in range(max(args.warmup_steps, 0)):
            last_loss = train_step()
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)  # peak reflects steady state only

        # --- Timed steady-state ---
        t0 = time.perf_counter()
        for _ in range(args.steps):
            last_loss = train_step()
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        total_tokens = args.batch_size * args.seq_len * args.steps
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float("inf")
        if is_cuda:
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ok": error is None,
        "error": error,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb,          # None on CPU / failure
        "elapsed_s": elapsed,
        "last_loss": last_loss,
        # config (every lever) ---------------------------------------------------
        "device": args.device,
        "tiny_random": args.tiny_random,
        "base": None if args.tiny_random else args.base,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "grad_checkpoint": args.grad_checkpoint,
        "optim": args.optim,
        "optim_actual": optim_actual,
        "optim_note": optim_note,
        "compile": args.compile,
        "compiled_ok": compiled_ok,
        "compile_note": compile_note,
        "tf32": args.tf32,
        "attn": args.attn,
        "attn_ran": attn_label,
        "amp": args.amp,
        "every": args.every if args.every is not None else 1,
        "n_capability_experts": args.n_capability_experts,
        # model info -------------------------------------------------------------
        "layers_after": info["final_layers"],
        "inserted": info["inserted"],
        "trainable_params": trainable,
        "total_params": total,
        # env --------------------------------------------------------------------
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None),
        "platform": platform.platform(),
    }
    return row


def _print_summary(row: dict) -> None:
    tps = row["tokens_per_sec"]
    vram = row["peak_vram_gb"]
    print("\n" + "=" * 68)
    print("  identity-MoTE TRAINING throughput benchmark")
    print("=" * 68)
    print(f"  status         : {'OK' if row['ok'] else 'FAILED — ' + str(row['error'])}")
    print(f"  device         : {row['device']}  (gpu={row['gpu']})")
    print(f"  model          : tiny_random={row['tiny_random']}  "
          f"layers_after={row['layers_after']} (+{row['inserted']} inserted)  "
          f"K={row['n_capability_experts']}")
    print(f"  trainable/total: {row['trainable_params']:,} / {row['total_params']:,}")
    print(f"  batch x seq    : {row['batch_size']} x {row['seq_len']}   "
          f"steps={row['steps']} (warmup {row['warmup_steps']})")
    print(f"  grad_ckpt={row['grad_checkpoint']}  optim={row['optim']}->{row['optim_actual']}  "
          f"compile={row['compile']}(ok={row['compiled_ok']})  tf32={row['tf32']}  "
          f"attn={row['attn']}->{row['attn_ran']}  amp={row['amp']}")
    if row["optim_note"]:
        print(f"  optim_note     : {row['optim_note']}")
    if row["compile_note"]:
        print(f"  compile_note   : {row['compile_note']}")
    print("-" * 68)
    print(f"  TOKENS/SEC     : {tps:,.1f}" if tps is not None else "  TOKENS/SEC     : n/a")
    print(f"  peak VRAM      : {vram:.3f} GB" if vram is not None else "  peak VRAM      : n/a (CPU)")
    print(f"  elapsed        : {row['elapsed_s']:.3f} s" if row["elapsed_s"] is not None else "  elapsed        : n/a")
    print("=" * 68 + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="identity-MoTE training-throughput micro-benchmark (tok/s + peak VRAM)."
    )
    # Model / data
    ap.add_argument("--tiny-random", action="store_true",
                    help="Tiny synthetic BitNet + synthetic token batches (no downloads).")
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
                    help="Base BitNet checkpoint (ignored with --tiny-random).")
    ap.add_argument("--device", default="cuda", help="cpu or cuda (default cuda).")
    ap.add_argument("--every", type=int, default=None,
                    help="LLaMA-Pro interleave for depth expansion (default 1).")
    ap.add_argument("--n-capability-experts", type=int, default=3,
                    help="K capability experts per inserted block (model size lever).")
    # Loop shape
    ap.add_argument("--batch-size", type=int, default=1, help="Sequences per step.")
    ap.add_argument("--seq-len", type=int, default=256, help="Tokens per sequence.")
    ap.add_argument("--steps", type=int, default=20, help="Timed steady-state steps.")
    ap.add_argument("--warmup-steps", type=int, default=3,
                    help="Warm-up steps excluded from timing (pay compile/autotune costs).")
    ap.add_argument("--route-weight", type=float, default=1.0, help="Routing CE weight.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate (throughput-neutral).")
    # Optimization levers
    ap.add_argument("--grad-checkpoint", choices=["on", "off"], default="off",
                    help="Gradient checkpointing (activation recompute).")
    ap.add_argument("--optim", choices=["adamw", "adamw-fused", "adafactor", "adamw8bit"],
                    default="adamw", help="Optimizer (adamw-fused=fused=True; adamw8bit=bnb).")
    ap.add_argument("--compile", choices=["on", "off"], default="off",
                    help="torch.compile(model) — Windows-support probe (records compiled vs errored).")
    ap.add_argument("--tf32", choices=["on", "off"], default="off",
                    help="TF32 matmul + cuDNN.")
    ap.add_argument("--attn", choices=["auto", "flash", "mem_efficient", "math"], default="auto",
                    help="SDPA backend (records which actually ran).")
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16",
                    help="Autocast dtype (bf16) or none (fp32).")
    # Output
    ap.add_argument("--results-file", default=os.path.join(".docs", "mote", "bench_train.jsonl"),
                    help="JSONL results file (one line appended per run).")
    ap.add_argument("--tag", default="", help="Free-form label copied into the JSON row.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    row = run_benchmark(args)
    if args.tag:
        row["tag"] = args.tag

    # Append JSON line to the results file.
    os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
    with open(args.results_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")

    _print_summary(row)
    # Machine-readable line for bench_matrix.py.
    print("BENCH_JSON: " + json.dumps(row))
    print(f"[bench] appended -> {args.results_file}")
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
