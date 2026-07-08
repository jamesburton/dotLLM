"""bitdistill_bench.py — BitNet-Distillation THROUGHPUT micro-benchmark.

Campaign: cross-machine venue selection — measure which machine (3060 / Strix /
Kaggle) completes the BitNet-distillation workload fastest BEFORE committing the
real multi-billion-token budget-curve run to any one box.

What this does
--------------
Runs the **actual distillation step** of ``bitdistill.py`` — the exact hot loop the
real conversion run executes::

    teacher forward (frozen FP, no_grad)  ->  t_logits, t_qkv
    student forward (BitLinear STE + SubLN) ->  s_logits, s_qkv
    L = CE  +  lambda * logit_KD(tau)  +  gamma * attn_relation
    backward (through the ternary STE)  ->  clip_grad_norm  ->  optimizer.step()

for a fixed number of steps and reports **tokens/sec** (steady-state, EXCLUDING
``--warmup-steps`` warm-up iterations that pay compile/allocator/autotune costs) +
**peak VRAM** (``torch.cuda.max_memory_allocated`` over the timed region only) + the
full config. It imports the real building blocks from ``bitdistill.py``
(``convert_to_bitnet_student``, ``logit_kd_loss``, ``attn_relation_loss``,
``QKVCapture``, ``build_tiny_qwen3``) so the measured work is representative, not a
proxy.

Device-agnostic
---------------
Uses only ``torch.cuda.is_available()`` / a device string and the ``torch.cuda.*``
memory API (which ROCm also exposes) — no CUDA-only calls that break on ROCm. Runs
on CUDA (3060, ``2.9+cu128``), ROCm (Strix, ``2.9.1+rocm``), single-GPU (Kaggle T4/
P100/L4) and CPU. Records ``torch.__version__``, ``torch.version.cuda`` / ``.hip``,
GPU name and platform so rows from different machines are directly comparable.

Data is SYNTHETIC random token batches (throughput is data-independent) — no dataset
download, no tokenizer needed, even for the real ``Qwen/Qwen3-0.6B`` model.

CPU smoke (seconds, no GPU, no download)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 \
      python scripts/lora/bitdistill_bench.py --tiny-model --tiny-random \
        --device cpu --batch-size 2 --steps 4 --warmup-steps 1

Real GPU probe (launched separately per machine by the orchestrator)::

    CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 \
      python scripts/lora/bitdistill_bench.py --base Qwen/Qwen3-0.6B --device cuda \
        --batch-size 8 --seq-len 512 --steps 20 --warmup-steps 5 \
        --lambda-kd 10 --tau 5 --optim adamw --amp bf16 --grad-checkpoint off \
        --tag rtx3060 --results-file .docs/bitdistill/bench_venue.jsonl

Every run appends one JSON line to ``--results-file`` and prints a human summary plus
a single machine-readable ``BENCH_JSON: {...}`` line on stdout (mirrors
``bench_train.py`` so a matrix runner can parse it).
"""

from __future__ import annotations

# Windows: torch.compile's Triton/Inductor back-end needs cl.exe; suppress dynamo.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import contextlib
import copy
import json
import os
import platform
import sys
import time
import traceback

import torch
import torch.nn.functional as F

# scripts/lora on sys.path so we can import the real bitdistill building blocks.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bitdistill as bd  # noqa: E402


# ---------------------------------------------------------------------------
# Optimizer factory (records what actually ran; graceful fallback, never raises)
# ---------------------------------------------------------------------------
def _make_optimizer(choice: str, params, lr: float):
    """Return (optimizer, actual_name, note).

    Fallback chain: ``adamw-fused`` -> ``adamw`` if fused unavailable; ``adafactor``
    -> ``adamw`` if transformers missing; ``adamw8bit`` -> ``adafactor`` -> ``adamw``
    (bitsandbytes fails on some boxes — notably ROCm/Strix and older CUDA).
    """
    betas = (0.9, 0.95)
    if choice == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=0.0), "adamw", ""
    if choice == "adamw-fused":
        try:
            opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=0.0, fused=True)
            return opt, "adamw-fused", ""
        except Exception as exc:  # noqa: BLE001
            note = f"fused unavailable ({type(exc).__name__}: {exc}); fell back to adamw"
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=0.0), "adamw", note
    if choice == "adafactor":
        return _adafactor(params, lr)
    if choice == "adamw8bit":
        try:
            import bitsandbytes as bnb
            try:
                opt = bnb.optim.PagedAdamW8bit(params, lr=lr, betas=betas)
            except AttributeError:
                opt = bnb.optim.AdamW8bit(params, lr=lr, betas=betas)
            return opt, "adamw8bit", ""
        except Exception as exc:  # noqa: BLE001
            # per-request fallback: adamw8bit -> adafactor (bnb fails on some boxes)
            opt, actual, note2 = _adafactor(params, lr)
            note = (f"bitsandbytes unavailable ({type(exc).__name__}: {exc}); "
                    f"fell back to {actual}")
            if note2:
                note += f" [{note2}]"
            return opt, actual, note
    raise ValueError(f"unknown optim {choice!r}")


def _adafactor(params, lr: float):
    try:
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=lr, relative_step=False,
                        scale_parameter=False, warmup_init=False)
        return opt, "adafactor", ""
    except Exception as exc:  # noqa: BLE001
        note = f"adafactor unavailable ({type(exc).__name__}: {exc}); fell back to adamw"
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0), "adamw", note


# ---------------------------------------------------------------------------
# Model construction (teacher = frozen FP; student = BitLinear-converted)
# ---------------------------------------------------------------------------
def _build_teacher_student(args, device: torch.device):
    if args.tiny_model:
        teacher = bd.build_tiny_qwen3().to(device)
        student = copy.deepcopy(teacher).to(device)
    else:
        from transformers import AutoModelForCausalLM
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        print(f"[bitdistill_bench] loading base {args.base!r} (dtype={dtype}) ...", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
        student = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False

    info = bd.convert_to_bitnet_student(student)
    student.to(device)
    student.config.use_cache = False
    student.train()
    return teacher, student, info


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(args) -> dict:
    on = lambda v: v == "on"  # noqa: E731
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    teacher, student, info = _build_teacher_student(args, device)
    vocab_size = student.config.vocab_size

    # Attn-relation distill layer (negative indexes from the end, as in bitdistill).
    attn_layer = args.attn_distill_layer
    if attn_layer < 0:
        attn_layer = student.config.num_hidden_layers + attn_layer
    s_cap = bd.QKVCapture(student, attn_layer)
    t_cap = bd.QKVCapture(teacher, attn_layer)
    rel_heads = student.config.num_key_value_heads

    trainable = [p for p in student.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())

    opt, optim_actual, optim_note = _make_optimizer(args.optim, trainable, args.lr)

    # Gradient checkpointing (activation recompute — trades compute for VRAM).
    ckpt = on(args.grad_checkpoint)
    if ckpt:
        try:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    # AMP autocast (bf16) — device-agnostic (cpu/cuda both accept bf16 autocast).
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None else contextlib.nullcontext()
    )

    # Synthetic batch generator (fixed content — throughput is data-independent).
    g = torch.Generator().manual_seed(1234)

    def make_batch():
        return torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                             generator=g, dtype=torch.long).to(device)

    def train_step():
        batch = make_batch()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx:
            with torch.no_grad():
                t_logits = teacher(input_ids=batch, use_cache=False).logits
            t_qkv = t_cap.get()
            s_logits = student(input_ids=batch, use_cache=False).logits
            s_qkv = s_cap.get()
            ce = F.cross_entropy(
                s_logits[:, :-1, :].contiguous().view(-1, vocab_size),
                batch[:, 1:].contiguous().view(-1),
            )
            ld = bd.logit_kd_loss(s_logits[:, :-1, :], t_logits[:, :-1, :], tau=args.tau)
            ad = bd.attn_relation_loss(s_qkv, t_qkv, n_rel_heads=rel_heads)
            loss = ce + args.lambda_kd * ld + args.gamma * ad
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        return float(loss.detach())

    error = None
    tokens_per_sec = elapsed = peak_vram_gb = None
    last_loss = float("nan")
    try:
        for _ in range(max(args.warmup_steps, 0)):
            last_loss = train_step()
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)  # steady-state peak only

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

    s_cap.remove()
    t_cap.remove()

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
        "tiny_model": args.tiny_model,
        "tiny_random": args.tiny_random,
        "base": None if args.tiny_model else args.base,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "lambda_kd": args.lambda_kd,
        "gamma": args.gamma,
        "tau": args.tau,
        "attn_distill_layer": args.attn_distill_layer,
        "attn_layer_resolved": attn_layer,
        "relation_heads": rel_heads,
        "grad_checkpoint": args.grad_checkpoint,
        "optim": args.optim,
        "optim_actual": optim_actual,
        "optim_note": optim_note,
        "amp": args.amp,
        # model info -------------------------------------------------------------
        "n_layers": student.config.num_hidden_layers,
        "hidden_size": student.config.hidden_size,
        "bitlinears": info["bitlinears"],
        "subnorms": info["subnorms"],
        "trainable_params": trainable_params,
        "student_params": student_params,
        "teacher_params": teacher_params,
        # env --------------------------------------------------------------------
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    return row


def _print_summary(row: dict) -> None:
    tps = row["tokens_per_sec"]
    vram = row["peak_vram_gb"]
    print("\n" + "=" * 68)
    print("  BitNet-Distillation THROUGHPUT benchmark")
    print("=" * 68)
    print(f"  status         : {'OK' if row['ok'] else 'FAILED — ' + str(row['error'])}")
    print(f"  device         : {row['device']}  (gpu={row['gpu']})")
    print(f"  torch          : {row['torch']}  cuda={row['torch_cuda']}  hip={row['torch_hip']}")
    print(f"  platform       : {row['platform']}  py={row['python']}")
    print(f"  model          : {'tiny_qwen3' if row['tiny_model'] else row['base']}  "
          f"layers={row['n_layers']} hidden={row['hidden_size']}  "
          f"({row['bitlinears']} BitLinears / {row['subnorms']} SubLNs)")
    print(f"  teacher/student: {row['teacher_params']:,} / {row['student_params']:,} params  "
          f"(trainable {row['trainable_params']:,})")
    print(f"  batch x seq    : {row['batch_size']} x {row['seq_len']}   "
          f"steps={row['steps']} (warmup {row['warmup_steps']})")
    print(f"  loss           : lambda_kd={row['lambda_kd']} gamma={row['gamma']} tau={row['tau']}  "
          f"attn_layer={row['attn_layer_resolved']} rel_heads={row['relation_heads']}")
    print(f"  grad_ckpt={row['grad_checkpoint']}  optim={row['optim']}->{row['optim_actual']}  "
          f"amp={row['amp']}")
    if row["optim_note"]:
        print(f"  optim_note     : {row['optim_note']}")
    print("-" * 68)
    print(f"  TOKENS/SEC     : {tps:,.1f}" if tps is not None else "  TOKENS/SEC     : n/a")
    print(f"  peak VRAM      : {vram:.3f} GB" if vram is not None else "  peak VRAM      : n/a (CPU)")
    print(f"  elapsed        : {row['elapsed_s']:.3f} s" if row["elapsed_s"] is not None
          else "  elapsed        : n/a")
    print("=" * 68 + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BitNet-Distillation throughput micro-benchmark (tok/s + peak VRAM)."
    )
    # Model / data
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B",
                    help="Base FP model (HF id or path). Ignored with --tiny-model.")
    ap.add_argument("--tiny-model", action="store_true", dest="tiny_model",
                    help="Tiny synthetic Qwen3 teacher+student (fastest smoke, no download).")
    ap.add_argument("--tiny-random", action="store_true", dest="tiny_random",
                    help="Synthetic random token batches (default behaviour; no dataset). "
                         "Throughput is data-independent so this is always effectively on.")
    ap.add_argument("--device", default="cuda", help="cpu | cuda | cuda:0 ... (default cuda).")
    # Loop shape
    ap.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                    help="Sequences per step (paper: 32; probe default 8).")
    ap.add_argument("--seq-len", type=int, default=512, dest="seq_len",
                    help="Tokens per sequence (paper: 512).")
    ap.add_argument("--steps", type=int, default=20, help="Timed steady-state steps.")
    ap.add_argument("--warmup-steps", type=int, default=5, dest="warmup_steps",
                    help="Warm-up steps excluded from timing (pay compile/autotune costs).")
    ap.add_argument("--lr", type=float, default=1e-4, help="AdamW LR (throughput-neutral).")
    # Distillation loss levers (match bitdistill.py defaults)
    ap.add_argument("--lambda-kd", type=float, default=10.0, dest="lambda_kd",
                    help="Weight lambda on logit-KD loss (paper classification: 10).")
    ap.add_argument("--gamma", type=float, default=1e-5,
                    help="Weight gamma on attention-relation loss (recipe note: 1e-5).")
    ap.add_argument("--tau", type=float, default=5.0, help="Logit-KD temperature (paper: 5).")
    ap.add_argument("--attn-distill-layer", type=int, default=-1, dest="attn_distill_layer",
                    help="Single layer for MiniLM attn-relation distill. -1 = last.")
    # Optimization levers
    ap.add_argument("--grad-checkpoint", choices=["on", "off"], default="off",
                    help="Gradient checkpointing (activation recompute).")
    ap.add_argument("--optim", choices=["adamw", "adamw-fused", "adafactor", "adamw8bit"],
                    default="adamw",
                    help="Optimizer. adamw8bit=bnb (falls back adafactor->adamw if unavailable).")
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16",
                    help="Autocast dtype (bf16) or none (fp32).")
    # Output
    ap.add_argument("--results-file", default=os.path.join(".docs", "bitdistill", "bench_venue.jsonl"),
                    help="JSONL results file (one line appended per run).")
    ap.add_argument("--out", default="", help="Optional dir; a copy of the row is written to "
                                              "<out>/bench_row.json when set.")
    ap.add_argument("--tag", default="", help="Free-form label (e.g. machine name) into the row.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    row = run_benchmark(args)
    if args.tag:
        row["tag"] = args.tag

    os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
    with open(args.results_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "bench_row.json"), "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)

    _print_summary(row)
    print("BENCH_JSON: " + json.dumps(row))
    print(f"[bitdistill_bench] appended -> {args.results_file}")
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
