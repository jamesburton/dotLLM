"""Curriculum-N recursion heal-train: LM loss + optional KD distillation.

Trains the slab layers [P:Q), fusion adapter, and gate.g of a RecurModel
(see recur_model.py) using a LogNormal curriculum that ramps the mean
recurrence count from mu_min to mu_max over the training run.

Loss
----
  loss = lm_loss + kd_weight * KL(student || teacher)

KD is disabled by default (kd_weight=0); when disabled the teacher is never
loaded.  Adafactor is the default optimizer — bnb 8-bit crashes on Windows
(OSError 0xc000001d in its CUDA kernels).

Gradient checkpointing is enabled by default on the slab loop to keep peak
VRAM bounded at high N (N × full-slab activations without checkpointing).

CLI
---
  python scripts/lora/recur_train.py \\
      --out .docs/recursion/r1 --tokens 5e6

Smoke test (CPU, seconds):
  python scripts/lora/recur_train.py \\
      --tiny-random --tokens 1e4 --device cpu \\
      --out .docs/recursion/smoke_r3 \\
      --P 0 --Q 1 --mu-min 1 --mu-max 2 --n-max 2 --optim adafactor

Writes
------
  <out>/adapter_weights.pt  — trainable slab + fusion + gate weights
  <out>/recur_config.json   — hyperparameters for this run
  <out>/metrics.json        — final losses + recurrence histogram + timing
"""

# Windows: torch.compile's Triton/Inductor back-end requires cl.exe; suppress
# dynamo errors so the script runs on Windows without a compiler tool-chain.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
except Exception:
    pass  # older torch or dynamo not available

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as hf_datasets

# Allow importing recur_model from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recur_model import build_recur


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token mean KL(student || teacher).

    Uses log-softmax for numerical stability.  Teacher logits are detached and
    moved to the student's device (teacher may be on CPU when --teacher-device cpu).
    """
    V = student_logits.size(-1)
    # Move teacher logits to the student's device (no-op when both are same device).
    t_logits = teacher_logits.to(student_logits.device)
    s_lp = F.log_softmax(student_logits.float().view(-1, V), dim=-1)
    t_p = F.softmax(t_logits.float().view(-1, V).detach(), dim=-1)
    # NOTE: F.kl_div(log_p, q) computes KL(q || p) = KL(teacher || student).
    # This is the standard LM-distillation direction (student learns teacher's mass).
    return F.kl_div(s_lp, t_p, reduction="batchmean")


def _build_corpus(
    tokenizer,
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    max_seq_len: int,
    max_sequences: int,
    tiny_random: bool,
    vocab_size: int,
) -> list:
    """Return a list of fixed-length token-ID tensors for LM training.

    In ``tiny_random`` mode: synthetic random IDs, no download or tokenizer
    required.  Otherwise: loads ``dataset_name`` from the HF hub (cached) and
    tokenizes its text content.
    """
    if tiny_random:
        return [
            torch.randint(0, vocab_size, (max_seq_len,))
            for _ in range(max_sequences)
        ]

    ds = hf_datasets.load_dataset(dataset_name, dataset_config, split=dataset_split)

    all_ids: list = []
    for row in ds:
        # Support multiple dataset schemas gracefully.
        if "messages" in row:
            text = " ".join(
                m["content"] for m in row["messages"] if m.get("content")
            )
        elif "text" in row:
            text = row["text"]
        elif "prompt" in row:
            text = row.get("prompt", "")
        else:
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= max_sequences * max_seq_len:
            break

    seqs = []
    for i in range(0, len(all_ids) - max_seq_len, max_seq_len):
        seqs.append(torch.tensor(all_ids[i : i + max_seq_len], dtype=torch.long))
        if len(seqs) >= max_sequences:
            break
    return seqs


def sample_recurrence(
    step: int,
    total_steps: int,
    mu_min: float,
    mu_max: float,
    n_max: int,
) -> int:
    """Sample recurrence N from LogNormal with linearly ramped mean mu.

    mu is linearly ramped from mu_min at step=0 to mu_max at step=total_steps.
    Samples are rounded and clamped to [1, n_max] — the recurrence >= 1 guard
    is mandatory (recurrence=0 would produce a no-op slab loop).
    """
    progress = min(1.0, step / max(1, total_steps))
    mu = mu_min + (mu_max - mu_min) * progress
    sigma = 0.5
    log_mu = math.log(max(mu, 1e-6))
    raw = torch.distributions.LogNormal(log_mu, sigma).sample().item()
    recurrence = max(1, min(n_max, round(raw)))  # clamp to [1, n_max]
    return recurrence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Curriculum-N RecurModel heal-train: LM + optional KD distillation"
    )
    ap.add_argument("--P", type=int, default=7, help="First slab layer index (inclusive)")
    ap.add_argument("--Q", type=int, default=22, help="Last slab layer index (exclusive)")
    ap.add_argument(
        "--tokens", type=float, default=5e6,
        help="Approximate total training tokens (default 5e6 = 5M)",
    )
    ap.add_argument(
        "--mu-min", type=float, default=2.0,
        help="Starting curriculum recurrence mean (default 2)",
    )
    ap.add_argument(
        "--mu-max", type=float, default=8.0,
        help="Ending curriculum recurrence mean (default 8)",
    )
    ap.add_argument(
        "--n-max", type=int, default=8,
        help="Hard cap on sampled recurrence N (default 8)",
    )
    ap.add_argument(
        "--kd-weight", type=float, default=0.0,
        help="KD weight; 0 = disabled (no teacher loaded, default)",
    )
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default 1e-4)")
    ap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Token sequence length per training step (default 512)",
    )
    ap.add_argument(
        "--optim", choices=["adafactor", "adamw8bit", "adamw"], default="adafactor",
        help=(
            "Optimizer: adafactor (default, no state memory); adamw8bit (bnb, auto-falls "
            "back to adafactor on Windows); adamw (full AdamW, most memory)."
        ),
    )
    ap.add_argument("--device", default="cuda", help="Training device: cpu or cuda")
    ap.add_argument("--out", required=True, help="Output directory for adapter + metrics")
    ap.add_argument(
        "--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="Base model checkpoint (HF hub id or local path)",
    )
    ap.add_argument(
        "--dataset", default="HuggingFaceH4/no_robots",
        help="HF dataset for heal-training corpus (default: HuggingFaceH4/no_robots)",
    )
    ap.add_argument("--dataset-config", default=None, help="HF dataset config name")
    ap.add_argument("--dataset-split", default="train", help="HF dataset split")
    ap.add_argument(
        "--teacher-device", default="cpu", choices=["cpu", "cuda"],
        help="Device for the frozen teacher model (default: cpu to save GPU VRAM)",
    )
    ap.add_argument(
        "--tiny-random", action="store_true",
        help=(
            "Use a tiny randomly-initialised model (2 layers, hidden=64) and a "
            "synthetic random corpus.  No downloads.  Runs in seconds on CPU."
        ),
    )
    ap.add_argument(
        "--grad-checkpoint", action="store_true", default=True,
        help="Enable gradient checkpointing on slab forward (default: True)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)
    max_tokens = int(args.tokens)
    os.makedirs(args.out, exist_ok=True)

    print(
        f"[recur_train] P={args.P}  Q={args.Q}  "
        f"mu_min={args.mu_min}  mu_max={args.mu_max}  n_max={args.n_max}"
    )
    print(
        f"[recur_train] tokens={max_tokens:.2e}  kd_weight={args.kd_weight}  "
        f"lr={args.lr}  device={device}  optim={args.optim!r}  "
        f"grad_checkpoint={args.grad_checkpoint}  tiny_random={args.tiny_random}"
    )

    # ------------------------------------------------------------------
    # 1. Tiny config (shared by teacher and student in tiny mode)
    # ------------------------------------------------------------------
    tiny_cfg = None
    if args.tiny_random:
        from transformers import AutoConfig  # noqa: F401
        tiny_cfg = AutoConfig.from_pretrained(args.base, local_files_only=True)
        tiny_cfg.hidden_size = 64
        tiny_cfg.intermediate_size = 128
        tiny_cfg.num_hidden_layers = 2
        tiny_cfg.num_attention_heads = 2
        tiny_cfg.num_key_value_heads = 2
        tiny_cfg.max_position_embeddings = 256
        # Keep vocab_size from the real config so token IDs are valid.

    # ------------------------------------------------------------------
    # 2. Teacher — skip entirely when kd_weight == 0
    # ------------------------------------------------------------------
    teacher = None
    if args.kd_weight > 0.0:
        if args.tiny_random:
            from transformers import BitNetForCausalLM
            teacher = BitNetForCausalLM(tiny_cfg).to(device=teacher_device)
        else:
            # BitNet refuses device_map with a CPU device; load without device_map
            # (defaults to CPU) and move manually, or use device_map only for CUDA.
            if teacher_device.type == "cpu":
                teacher = AutoModelForCausalLM.from_pretrained(
                    args.base, dtype=torch.bfloat16
                )
            else:
                teacher = AutoModelForCausalLM.from_pretrained(
                    args.base, dtype=torch.bfloat16, device_map={"": teacher_device}
                )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(
            f"[recur_train] teacher loaded and frozen on {teacher_device} "
            "(dense BitNet, no grad)"
        )
    else:
        print("[recur_train] KD disabled — no teacher loaded")

    # ------------------------------------------------------------------
    # 3. Student — build RecurModel
    # ------------------------------------------------------------------
    if args.tiny_random:
        from transformers import BitNetForCausalLM
        base_model = BitNetForCausalLM(tiny_cfg)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map={"": device}
        )
    base_model.config.use_cache = False

    recur = build_recur(base_model, P=args.P, Q=args.Q)

    # Move student to training device (build_recur creates fusion + gate on CPU)
    recur.to(device)

    # Assert all parameters and buffers are on the training device
    for name, p in recur.named_parameters():
        assert p.device.type == device.type, (
            f"Parameter {name} is on {p.device} but training device is {device}"
        )
    for name, buf in recur.named_buffers():
        assert buf.device.type == device.type, (
            f"Buffer {name} is on {buf.device} but training device is {device}"
        )
    print(f"[recur_train] recur model on device {device}; all params/buffers verified")

    # ------------------------------------------------------------------
    # 4. Freeze/unfreeze partition
    # ------------------------------------------------------------------
    # 4a. Freeze everything
    for p in recur.parameters():
        p.requires_grad_(False)

    # 4b. Unfreeze slab layers [P:Q), fusion adapter, gate.g only.
    #     Slab layers are BitLinear — training them = ternary QAT in bf16 shadow.
    for layer in recur.base.model.layers[args.P:args.Q]:
        for p in layer.parameters():
            p.requires_grad_(True)
    recur.fusion.weight.requires_grad_(True)
    recur.fusion.bias.requires_grad_(True)
    recur.gate.g.requires_grad_(True)

    trainable_params = [p for p in recur.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in recur.parameters())
    print(
        f"[recur_train] trainable={trainable:,} / total={total_params:,} "
        f"({100 * trainable / max(total_params, 1):.2f}%)"
    )

    # ------------------------------------------------------------------
    # 5. Gradient checkpointing on slab
    # ------------------------------------------------------------------
    recur.use_grad_checkpoint = args.grad_checkpoint
    print(
        f"[recur_train] gradient checkpointing: "
        f"{'ENABLED' if args.grad_checkpoint else 'disabled'}"
    )

    # ------------------------------------------------------------------
    # 6. Optimizer
    # ------------------------------------------------------------------
    _lr = args.lr
    _optim_choice = args.optim
    opt: torch.optim.Optimizer

    if _optim_choice == "adamw":
        opt = torch.optim.AdamW(
            trainable_params, lr=_lr, weight_decay=0.0, betas=(0.9, 0.95)
        )
        print(f"[recur_train] optimizer: AdamW (full, lr={_lr})")
    elif _optim_choice == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(
            trainable_params,
            lr=_lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        print(f"[recur_train] optimizer: Adafactor (lr={_lr})")
    else:  # adamw8bit — try bnb, fall back to adafactor on Windows
        # bnb may import and construct fine but fail at first step() on Windows
        # (OSError 0xc000001d — illegal instruction in its CUDA kernels).
        # Verify with a synthetic parameter step before committing.
        _bnb_ok = False
        _bnb_err_msg = "not attempted"
        try:
            import bitsandbytes as bnb
            try:
                opt = bnb.optim.PagedAdamW8bit(
                    trainable_params, lr=_lr, weight_decay=0.0, betas=(0.9, 0.95)
                )
                _bnb_cls_name = "PagedAdamW8bit"
            except AttributeError:
                opt = bnb.optim.AdamW8bit(
                    trainable_params, lr=_lr, weight_decay=0.0, betas=(0.9, 0.95)
                )
                _bnb_cls_name = "AdamW8bit"
            # Verify the optimizer kernel actually works at runtime
            _vp = torch.zeros(32, device=device, requires_grad=True)
            _vo = type(opt)([_vp], lr=_lr)
            _vp.sum().backward()
            _vo.step()
            _vo.zero_grad()
            del _vp, _vo
            _bnb_ok = True
            print(
                f"[recur_train] optimizer: bitsandbytes {_bnb_cls_name} "
                f"(verified, lr={_lr})"
            )
        except (ImportError, OSError, RuntimeError, Exception) as _bnb_exc:
            _bnb_err_msg = repr(_bnb_exc)

        if not _bnb_ok:
            from transformers.optimization import Adafactor
            opt = Adafactor(
                trainable_params,
                lr=_lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            print(
                f"[recur_train] optimizer: Adafactor fallback "
                f"(bitsandbytes runtime error: {_bnb_err_msg}, lr={_lr})"
            )

    # ------------------------------------------------------------------
    # 7. Corpus
    # ------------------------------------------------------------------
    vocab_size = recur.base.config.vocab_size
    max_seqs = max(200, max_tokens // args.max_seq_len + 1)
    corpus = _build_corpus(
        tokenizer=(
            None if args.tiny_random
            else AutoTokenizer.from_pretrained(args.base)
        ),
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        max_seq_len=args.max_seq_len,
        max_sequences=int(max_seqs),
        tiny_random=args.tiny_random,
        vocab_size=vocab_size,
    )
    if not corpus:
        raise RuntimeError(
            "Corpus is empty.  "
            "Check --dataset / --dataset-split, or use --tiny-random."
        )
    print(
        f"[recur_train] corpus: {len(corpus)} sequences "
        f"× {args.max_seq_len} tokens each"
    )

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    total_steps = max(1, int(max_tokens / args.max_seq_len))
    recur.train()
    tokens_seen = 0
    step = 0
    recurrence_hist: Counter = Counter()

    final_lm = final_kd = float("nan")
    _t0 = time.perf_counter()
    _steps_per_sec: Optional[float] = None

    while tokens_seen < max_tokens:
        seq = corpus[step % len(corpus)].unsqueeze(0).to(device)  # [1, T]

        # Curriculum-N: sample recurrence for this step
        recurrence = sample_recurrence(step, total_steps, args.mu_min, args.mu_max, args.n_max)

        # Teacher forward (only if kd_weight > 0)
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(input_ids=seq.to(teacher_device)).logits

        # Student forward (with sampled N)
        student_logits = recur(seq, recurrence=recurrence)

        # LM loss (causal; shift by 1)
        lm_loss = F.cross_entropy(
            student_logits[:, :-1].contiguous().view(-1, vocab_size),
            seq[:, 1:].contiguous().view(-1),
        )

        # KD loss
        if args.kd_weight > 0.0 and teacher_logits is not None:
            kd_loss = _kl_loss(
                student_logits[:, :-1].contiguous(),
                teacher_logits[:, :-1].contiguous(),
            )
        else:
            kd_loss = torch.zeros(1, device=device).squeeze()

        # Log values BEFORE backward (tensors freed after loss.backward())
        lm_val = lm_loss.item()
        kd_val = kd_loss.item()

        # Total loss
        loss = lm_loss + args.kd_weight * kd_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        opt.zero_grad()

        tokens_seen += seq.size(1)
        step += 1
        recurrence_hist[recurrence] += 1

        # Steps/sec (warm-up excluded, computed from step 10 onward)
        if step == 10:
            _t0 = time.perf_counter()
        elif step > 10:
            elapsed = time.perf_counter() - _t0
            _steps_per_sec = (step - 10) / elapsed if elapsed > 0 else None

        # Progress logging
        if step <= 5 or step % 20 == 0:
            sps_str = f"  {_steps_per_sec:.2f} steps/s" if _steps_per_sec else ""
            print(
                f"  step {step:5d}  tokens {tokens_seen:.2e}  "
                f"lm {lm_val:.4f}  N={recurrence}{sps_str}",
                flush=True,
            )

        final_lm = lm_val
        final_kd = kd_val

    # Peak VRAM if on CUDA
    peak_vram_gb = None
    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / 1e9
        print(f"[recur_train] peak VRAM: {peak_vram_gb:.2f} GB")

    sps_report = f"  {_steps_per_sec:.3f} steps/s" if _steps_per_sec else ""
    print(
        f"[recur_train] training done — {step} steps / {tokens_seen} tokens{sps_report}\n"
        f"             final: lm={final_lm:.4f}  kd={final_kd:.4f}"
    )

    # ------------------------------------------------------------------
    # 9. Save adapter weights (trainable params only)
    # ------------------------------------------------------------------
    trainable_names = {
        name for name, p in recur.named_parameters() if p.requires_grad
    }
    adapter_state = {
        k: v.cpu() for k, v in recur.state_dict().items()
        if k in trainable_names
    }
    adapter_path = os.path.join(args.out, "adapter_weights.pt")
    torch.save(adapter_state, adapter_path)
    print(f"[recur_train] adapter → {adapter_path}  ({len(adapter_state)} tensors)")

    # ------------------------------------------------------------------
    # 10. Save recur config
    # ------------------------------------------------------------------
    recur_cfg_dict = {
        "P": args.P,
        "Q": args.Q,
        "mu_min": args.mu_min,
        "mu_max": args.mu_max,
        "n_max": args.n_max,
        "tokens": tokens_seen,
        "base": args.base,
        "kd_weight": args.kd_weight,
        "max_seq_len": args.max_seq_len,
        "optim": args.optim,
        "steps": step,
        "tiny_random": args.tiny_random,
    }
    with open(os.path.join(args.out, "recur_config.json"), "w", encoding="utf-8") as fh:
        json.dump(recur_cfg_dict, fh, indent=2)

    # ------------------------------------------------------------------
    # 11. Metrics
    # ------------------------------------------------------------------
    recurrence_histogram = {str(k): v for k, v in sorted(recurrence_hist.items())}
    metrics: dict = {
        "lm_loss": final_lm,
        "kd_loss": final_kd,
        "recurrence_histogram": recurrence_histogram,
        "steps": step,
        "tokens": tokens_seen,
    }
    if peak_vram_gb is not None:
        metrics["peak_vram_gb"] = peak_vram_gb
    if _steps_per_sec is not None:
        metrics["steps_per_sec"] = round(_steps_per_sec, 3)
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[recur_train] metrics → {metrics_path}")

    # ------------------------------------------------------------------
    # 12. Smoke / GATE check
    # ------------------------------------------------------------------
    lm_finite = not math.isnan(final_lm)
    recur_varies = len(recurrence_hist) > 1
    adapter_saved = os.path.exists(adapter_path)

    print(
        f"\n[recur_train] GATE: lm_finite={lm_finite}  "
        f"recur_varies={recur_varies}  adapter_saved={adapter_saved}"
    )
    if not (lm_finite and recur_varies and adapter_saved):
        raise RuntimeError("Smoke-test gate FAILED — see GATE line above.")
    print("[recur_train] GATE PASSED")


if __name__ == "__main__":
    main()
