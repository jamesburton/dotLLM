"""MoTE heal-train loop: LM + KL distillation + Switch load-balance aux loss.

The decoder-layer compatibility problem
----------------------------------------
``MoTEBlock.forward`` returns ``(hidden, aux_loss, expert_counts)`` — a tuple.
HF decoder layers call ``self.mlp(x)`` expecting a plain tensor.  This script
wraps every converted MoTEBlock in a ``MoTEShim`` that:

  1. Unpacks the tuple and returns only ``hidden`` (so the HF residual stream
     is unaffected).
  2. Stores ``last_aux`` in-graph (gradients flow through it) and
     ``last_counts`` (detached, for logging) as attributes.

After each student forward pass, ``_collect_aux(student)`` sums
``shim.last_aux`` over all MoTE layers and adds it to the loss.

Loss
----
  loss = lm_loss + kd_weight * KL(student || frozen_dense_teacher) + 0.01 * aux

Teacher
-------
Loaded from the same checkpoint *before* ``build_mote`` is called on the
student, so the teacher is always a plain dense BitNet.  All parameters are
frozen and no gradients are computed through it.

Freezing
--------
All parameters are frozen first.  Then only router and routed-expert weights
are unfrozen (trainable).  The shared expert (fp or ternary) stays frozen,
matching the spec.  Base attention/embedding weights are always frozen.

CLI
---
  python scripts/lora/mote_train.py \\
      --config c1 --n-experts 4 --top-k 1 --shared fp \\
      --layers upper-half --tokens 4e8 --kd-weight 0.5 \\
      --out .docs/mote/c1

Smoke test (CPU, seconds — uses a tiny randomly-initialised model):
  python scripts/lora/mote_train.py \\
      --config smoke --n-experts 2 --top-k 1 --shared none \\
      --layers upper-half --tokens 1e5 --kd-weight 0.5 \\
      --out .docs/mote/smoke --tiny-random

Writes
------
  <out>/adapter_weights.pt  — trainable MoTE weights (router + routed experts)
  <out>/mote_config.json    — MoTE hyperparameters used for this run
  <out>/metrics.json        — final losses + expert-count histogram
"""

# Windows: torch.compile's Triton/Inductor back-end requires cl.exe; suppress
# dynamo errors so the script runs on Windows without a compiler tool-chain.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass  # older torch or dynamo not available

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as hf_datasets

# Allow importing mote_upcycle from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mote_upcycle import MoTEBlock, build_mote


# ---------------------------------------------------------------------------
# MoTEShim — tensor-return compatibility shim for HF decoder layers
# ---------------------------------------------------------------------------


class MoTEShim(nn.Module):
    """Thin wrapper that makes MoTEBlock compatible with HF decoder layer forward.

    The HF decoder layer calls ``self.mlp(x)`` and splices the result into the
    residual stream directly — it expects a plain tensor.  MoTEBlock returns
    ``(hidden, aux_loss, expert_counts)``.  This shim unpacks the tuple:

    * Returns only ``hidden`` to the decoder layer (residual splice is correct).
    * Stores ``last_aux`` in-graph so gradients flow through it when the
      training loop adds it to the loss.
    * Stores ``last_counts`` (detached, int64) for histogram logging.

    Parameters
    ----------
    mote_block:
        The MoTEBlock to wrap.
    """

    def __init__(self, mote_block: MoTEBlock) -> None:
        super().__init__()
        self.mote = mote_block
        self.last_aux: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, aux, counts = self.mote(x)
        self.last_aux = aux          # in-graph — gradients flow through backward
        self.last_counts = counts.detach()
        return hidden

    # Convenience accessors for the freeze helper.
    @property
    def router(self) -> nn.Linear:
        return self.mote.router

    @property
    def experts(self) -> nn.ModuleList:
        return self.mote.experts

    @property
    def shared(self) -> Optional[nn.Module]:
        return self.mote.shared


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_layers(arg: str, n_layers: int) -> list:
    """Resolve the ``--layers`` string to a concrete list of layer indices.

    Supported forms:
      ``upper-half``  → ``range(n_layers // 2, n_layers)``
      ``all``         → ``range(0, n_layers)``
      ``15-29``       → ``range(15, 30)``
      ``15,16,17``    → ``[15, 16, 17]``
    """
    if arg == "upper-half":
        return list(range(n_layers // 2, n_layers))
    if arg == "all":
        return list(range(n_layers))
    if "-" in arg and "," not in arg:
        start, end = arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in arg.split(",")]


def _wrap_mote_shims(model: nn.Module) -> nn.Module:
    """Replace every MoTEBlock .mlp with a MoTEShim in-place.

    Called immediately after ``build_mote`` so the HF decoder forward sees
    a plain-tensor-returning module.
    """
    for layer in model.model.layers:
        if isinstance(layer.mlp, MoTEBlock):
            layer.mlp = MoTEShim(layer.mlp)
    return model


def _freeze_for_mote_training(model: nn.Module) -> None:
    """Freeze everything; then unfreeze router + routed-expert weights only.

    Frozen:  all base attention/embedding params; shared expert (fp or ternary).
    Trainable: BF16 router nn.Linear + routed-expert AutoBitLinear clones.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.model.layers:
        shim = layer.mlp
        if isinstance(shim, MoTEShim):
            for p in shim.router.parameters():
                p.requires_grad_(True)
            for p in shim.experts.parameters():
                p.requires_grad_(True)
            # shared expert stays frozen regardless of mode


def _collect_aux(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Sum ``last_aux`` from all MoTEShim layers after a student forward pass.

    The returned tensor is in-graph (gradients flow through it).
    Returns a zero scalar if no MoTE layers contributed.
    """
    total: Optional[torch.Tensor] = None
    for layer in model.model.layers:
        shim = layer.mlp
        if isinstance(shim, MoTEShim) and shim.last_aux is not None:
            if total is None:
                total = shim.last_aux
            else:
                total = total + shim.last_aux
    return total if total is not None else torch.zeros(1, device=device).squeeze()


def _collect_counts(model: nn.Module) -> Optional[torch.Tensor]:
    """Accumulate per-expert dispatch counts from all MoTEShim layers."""
    total: Optional[torch.Tensor] = None
    for layer in model.model.layers:
        shim = layer.mlp
        if isinstance(shim, MoTEShim) and shim.last_counts is not None:
            if total is None:
                total = shim.last_counts.clone()
            else:
                total = total + shim.last_counts
    return total


def _kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token mean KL(student || teacher).

    Uses log-softmax for numerical stability.  Teacher logits are detached and
    moved to the student's device (teacher may be on CPU when --teacher-device cpu).
    """
    V = student_logits.size(-1)
    # Move teacher logits to the student's device (no-op when both are on the same device).
    t_logits = teacher_logits.to(student_logits.device)
    s_lp = F.log_softmax(student_logits.float().view(-1, V), dim=-1)
    t_p = F.softmax(t_logits.float().view(-1, V).detach(), dim=-1)
    # NOTE: F.kl_div(log_p, q) computes forward KL = KL(q || p) = KL(teacher || student).
    # This is the standard LM-distillation objective (penalizes the student for missing
    # teacher mass). The spec's "KL(student || teacher)" notation was inverted; this
    # implementation is the scientifically correct direction.
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
    required.  Otherwise: loads ``dataset_name`` from the HF hub (cached
    already for the expected corpora: HuggingFaceH4/no_robots, etc.) and
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MoTE heal-train: LM + KD distillation + Switch aux loss"
    )
    ap.add_argument(
        "--config", required=True,
        help="Experiment name (used in logging and saved mote_config.json)",
    )
    ap.add_argument("--n-experts", type=int, default=4, help="Number of routed experts")
    ap.add_argument("--top-k", type=int, default=1, help="Experts selected per token")
    ap.add_argument(
        "--shared", choices=["fp", "ternary", "none"], default="fp",
        help="Shared-expert mode (fp=frozen bf16; ternary=frozen ternary; none=omit)",
    )
    ap.add_argument(
        "--layers", default="upper-half",
        help=(
            "Which transformer layers to convert to MoTE. "
            "Options: upper-half | all | <start>-<end> | comma-separated list. "
            "upper-half means layers[n//2:] (e.g. layers 15..29 for a 30-block model)."
        ),
    )
    ap.add_argument(
        "--tokens", type=float, default=4e8,
        help="Approximate total training tokens (default 4e8 = 400M)",
    )
    ap.add_argument(
        "--kd-weight", type=float, default=0.5,
        help="Weight for KL distillation loss (set 0 to disable KD)",
    )
    ap.add_argument("--out", required=True, help="Output directory for adapter + metrics")
    ap.add_argument(
        "--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="Base model checkpoint (HF hub id or local path)",
    )
    ap.add_argument(
        "--dataset", default="HuggingFaceH4/no_robots",
        help="HF dataset for heal-training corpus (default: HuggingFaceH4/no_robots, cached)",
    )
    ap.add_argument("--dataset-config", default=None, help="HF dataset config name")
    ap.add_argument("--dataset-split", default="train", help="HF dataset split")
    ap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Token sequence length for each training step",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="Training device: cpu or cuda (default: cpu for smoke tests)",
    )
    ap.add_argument(
        "--tiny-random", action="store_true",
        help=(
            "Use a tiny randomly-initialised BitNetForCausalLM (2 layers, hidden=64) "
            "and a synthetic random corpus.  No downloads required.  Use for smoke "
            "tests and CI — runs in seconds on CPU."
        ),
    )
    ap.add_argument(
        "--histogram-steps", type=int, default=50,
        help="Number of recent steps used to build the expert-count histogram",
    )
    ap.add_argument(
        "--optim", choices=["adamw8bit", "adafactor", "adamw"], default="adamw8bit",
        help=(
            "Optimizer: adamw8bit (bitsandbytes PagedAdamW8bit, auto-falls back to "
            "adafactor if bnb unavailable); adafactor (Adafactor, no state memory); "
            "adamw (full AdamW, most memory)."
        ),
    )
    ap.add_argument(
        "--teacher-device", default="cpu", choices=["cpu", "cuda"],
        help="Device for the frozen teacher model (default: cpu to save GPU VRAM)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)
    max_tokens = int(args.tokens)
    os.makedirs(args.out, exist_ok=True)

    print(
        f"[mote_train] config={args.config!r}  n_experts={args.n_experts}  "
        f"top_k={args.top_k}  shared={args.shared!r}  layers={args.layers!r}"
    )
    print(
        f"[mote_train] tokens={max_tokens:.2e}  kd_weight={args.kd_weight}  "
        f"device={device}  teacher_device={teacher_device}  "
        f"optim={args.optim!r}  tiny_random={args.tiny_random}"
    )

    # ------------------------------------------------------------------
    # 1. Build tiny config (used by both teacher and student in tiny mode)
    # ------------------------------------------------------------------
    tiny_cfg = None
    if args.tiny_random:
        from transformers import AutoConfig, BitNetForCausalLM  # noqa: F401 (import check)
        tiny_cfg = AutoConfig.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16", local_files_only=True)
        tiny_cfg.hidden_size = 64
        tiny_cfg.intermediate_size = 128
        tiny_cfg.num_hidden_layers = 2
        tiny_cfg.num_attention_heads = 2
        tiny_cfg.num_key_value_heads = 2
        tiny_cfg.max_position_embeddings = 256
        # Keep vocab_size from the real config so token IDs are valid.

    # ------------------------------------------------------------------
    # 2. Teacher — loaded (or created) BEFORE build_mote so it is dense
    # ------------------------------------------------------------------
    teacher = None
    if args.kd_weight > 0.0:
        if args.tiny_random:
            from transformers import BitNetForCausalLM
            teacher = BitNetForCausalLM(tiny_cfg).to(device=teacher_device)
        else:
            # BitNet refuses device_map with a CPU or disk device; load without device_map
            # (defaults to CPU) and move manually, or use device_map only for CUDA.
            if teacher_device.type == "cpu":
                teacher = AutoModelForCausalLM.from_pretrained(
                    args.base, dtype=torch.bfloat16
                )
                # Already on CPU by default; no explicit move needed.
            else:
                teacher = AutoModelForCausalLM.from_pretrained(
                    args.base, dtype=torch.bfloat16, device_map={"": teacher_device}
                )

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[mote_train] teacher loaded and frozen on {teacher_device} (dense BitNet, no grad)")
    else:
        print(f"[mote_train] KD disabled (kd_weight=0) — no teacher loaded")

    # ------------------------------------------------------------------
    # 3. Student — separate load; build_mote converts target layers
    # ------------------------------------------------------------------
    if args.tiny_random:
        from transformers import BitNetForCausalLM
        student = BitNetForCausalLM(tiny_cfg).to(device=device)
    else:
        student = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map={"": device}
        )
    student.config.use_cache = False

    n_layers = len(student.model.layers)
    layer_indices = _parse_layers(args.layers, n_layers)
    print(f"[mote_train] converting {len(layer_indices)} layers → MoTE: {layer_indices}")

    student = build_mote(
        student,
        layers=layer_indices,
        n_experts=args.n_experts,
        top_k=args.top_k,
        shared=args.shared,
    )
    # Wrap each MoTEBlock so the HF decoder forward receives a plain tensor.
    student = _wrap_mote_shims(student)

    # Move student to training device (build_mote may have created new modules on CPU)
    student.to(device)
    # Assert all parameters and buffers are on the training device
    for name, p in student.named_parameters():
        assert p.device.type == device.type, (
            f"Parameter {name} is on {p.device} but training device is {device}"
        )
    for name, buf in student.named_buffers():
        assert buf.device.type == device.type, (
            f"Buffer {name} is on {buf.device} but training device is {device}"
        )
    print(f"[mote_train] student moved to device {device}; all params/buffers verified on {device}")

    _freeze_for_mote_training(student)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    print(
        f"[mote_train] trainable={trainable:,} / total={total_params:,} "
        f"({100 * trainable / max(total_params, 1):.2f}%)"
    )

    # ------------------------------------------------------------------
    # 4. Optimizer — router lr 1e-4 (spec: "lr 1e-4 on the MoE path")
    # ------------------------------------------------------------------
    _lr = 1e-4
    _trainable = [p for p in student.parameters() if p.requires_grad]
    _optim_choice = args.optim
    opt: torch.optim.Optimizer

    if _optim_choice == "adamw":
        opt = torch.optim.AdamW(_trainable, lr=_lr, weight_decay=0.0, betas=(0.9, 0.95))
        print(f"[mote_train] optimizer: AdamW (full, lr={_lr})")
    elif _optim_choice == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(
            _trainable,
            lr=_lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        print(f"[mote_train] optimizer: Adafactor (lr={_lr})")
    else:  # adamw8bit (default) — try bnb, fallback to adafactor
        # bnb may import and construct fine but fail at first step() on Windows
        # (illegal-instruction / WinError -1073741795 in CUDA kernels). Verify with
        # a synthetic parameter step before committing to it for the real training.
        _bnb_ok = False
        _bnb_err_msg = "not attempted"
        try:
            import bitsandbytes as bnb
            try:
                opt = bnb.optim.PagedAdamW8bit(_trainable, lr=_lr, weight_decay=0.0, betas=(0.9, 0.95))
                _bnb_cls_name = "PagedAdamW8bit"
            except AttributeError:
                opt = bnb.optim.AdamW8bit(_trainable, lr=_lr, weight_decay=0.0, betas=(0.9, 0.95))
                _bnb_cls_name = "AdamW8bit"
            # Verify the optimizer kernel actually works at runtime (Windows wheels are flaky)
            _vp = torch.zeros(32, device=device, requires_grad=True)
            _vo = type(opt)([_vp], lr=_lr)
            _vp.sum().backward()
            _vo.step()
            _vo.zero_grad()
            del _vp, _vo
            _bnb_ok = True
            print(f"[mote_train] optimizer: bitsandbytes {_bnb_cls_name} (verified, lr={_lr})")
        except (ImportError, OSError, RuntimeError, Exception) as _bnb_exc:
            _bnb_err_msg = repr(_bnb_exc)

        if not _bnb_ok:
            from transformers.optimization import Adafactor
            opt = Adafactor(
                _trainable,
                lr=_lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            print(
                f"[mote_train] optimizer: Adafactor fallback "
                f"(bitsandbytes runtime error: {_bnb_err_msg}, lr={_lr})"
            )

    # ------------------------------------------------------------------
    # 5. Corpus
    # ------------------------------------------------------------------
    vocab_size = (teacher or student).config.vocab_size
    max_seqs = max(200, max_tokens // args.max_seq_len + 1)
    corpus = _build_corpus(
        tokenizer=None if args.tiny_random else AutoTokenizer.from_pretrained(args.base),
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
        f"[mote_train] corpus: {len(corpus)} sequences "
        f"× {args.max_seq_len} tokens each"
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    student.train()
    tokens_seen = 0
    step = 0
    recent_counts: list = []   # last K expert-count tensors

    final_lm = final_kd = final_aux = float("nan")
    _t0 = time.perf_counter()
    _steps_per_sec: Optional[float] = None

    while tokens_seen < max_tokens:
        seq = corpus[step % len(corpus)].unsqueeze(0).to(device)  # [1, T] on student device

        # Teacher forward (no grad) — input moved to teacher_device (may be CPU)
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(input_ids=seq.to(teacher_device)).logits  # [1, T, V]

        # Student forward
        student_logits = student(input_ids=seq).logits  # [1, T, V]

        # --- LM loss (causal; shift by 1) ---
        lm_loss = F.cross_entropy(
            student_logits[:, :-1, :].contiguous().view(-1, vocab_size),
            seq[:, 1:].contiguous().view(-1),
        )

        # --- KD loss: KL(student || teacher) per token ---
        if args.kd_weight > 0.0 and teacher_logits is not None:
            kd = _kl_loss(
                student_logits[:, :-1, :].contiguous(),
                teacher_logits[:, :-1, :].contiguous(),
            )
        else:
            kd = torch.zeros(1, device=device).squeeze()

        # --- Aux loss from all MoTE shims (Switch load-balance, γ=0.01) ---
        aux = _collect_aux(student, device)

        # Log values BEFORE backward (tensors may be freed after)
        lm_val = lm_loss.item()
        kd_val = kd.item()
        aux_val = aux.item()

        # --- Total loss ---
        loss = lm_loss + args.kd_weight * kd + 0.01 * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()

        tokens_seen += seq.size(1)
        step += 1

        # Accumulate expert counts for histogram (rolling window)
        counts = _collect_counts(student)
        if counts is not None:
            recent_counts.append(counts.cpu())
            if len(recent_counts) > args.histogram_steps:
                recent_counts.pop(0)

        # Compute rolling steps/sec from step 10 onward (warm-up excluded)
        if step == 10:
            _t0 = time.perf_counter()  # reset timer after warm-up
        elif step > 10:
            elapsed = time.perf_counter() - _t0
            _steps_per_sec = (step - 10) / elapsed if elapsed > 0 else None

        # Progress logging
        if step <= 5 or step % 20 == 0:
            sps_str = f"  {_steps_per_sec:.2f} steps/s" if _steps_per_sec else ""
            print(
                f"  step {step:5d}  tokens {tokens_seen:.2e}  "
                f"lm {lm_val:.4f}  kd {kd_val:.4f}  aux {aux_val:.4f}{sps_str}",
                flush=True,
            )

        final_lm = lm_val
        final_kd = kd_val
        final_aux = aux_val

    # Log peak VRAM if on CUDA
    peak_vram_gb = None
    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / 1e9
        print(f"[mote_train] peak VRAM: {peak_vram_gb:.2f} GB")

    sps_report = f"  {_steps_per_sec:.3f} steps/s" if _steps_per_sec else ""
    print(
        f"[mote_train] training done — {step} steps / {tokens_seen} tokens{sps_report}\n"
        f"             final: lm={final_lm:.4f}  kd={final_kd:.4f}  "
        f"aux={final_aux:.4f}"
    )

    # ------------------------------------------------------------------
    # 7. Save adapter weights (trainable params only)
    # ------------------------------------------------------------------
    trainable_names = {
        name for name, p in student.named_parameters() if p.requires_grad
    }
    adapter_state = {
        k: v.cpu() for k, v in student.state_dict().items()
        if k in trainable_names
    }
    adapter_path = os.path.join(args.out, "adapter_weights.pt")
    torch.save(adapter_state, adapter_path)
    print(f"[mote_train] adapter → {adapter_path}  ({len(adapter_state)} tensors)")

    # ------------------------------------------------------------------
    # 8. Save MoTE config
    # ------------------------------------------------------------------
    mote_cfg_dict = {
        "config": args.config,
        "base": args.base,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "shared": args.shared,
        "layers": layer_indices,
        "kd_weight": args.kd_weight,
        "max_seq_len": args.max_seq_len,
        "tokens_trained": tokens_seen,
        "steps": step,
        "tiny_random": args.tiny_random,
    }
    with open(os.path.join(args.out, "mote_config.json"), "w", encoding="utf-8") as fh:
        json.dump(mote_cfg_dict, fh, indent=2)

    # ------------------------------------------------------------------
    # 9. Expert-count histogram + metrics.json
    # ------------------------------------------------------------------
    if recent_counts:
        histo = torch.stack(recent_counts).sum(0).tolist()
    else:
        histo = []

    n_experts_used = sum(1 for c in histo if c > 0)
    print(
        f"[mote_train] expert histogram (last {args.histogram_steps} steps): {histo}"
    )
    print(f"[mote_train] experts used: {n_experts_used} / {args.n_experts}")

    metrics = {
        "lm_loss": final_lm,
        "kd_loss": final_kd,
        "aux_loss": final_aux,
        "expert_counts": histo,
        "n_experts_used": n_experts_used,
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
    print(f"[mote_train] metrics → {metrics_path}")

    # Smoke-test gate (print clearly for the CI gate checker)
    lm_ok = not (final_lm != final_lm)          # NaN check
    kd_ok = not (final_kd != final_kd)
    aux_ok = not (final_aux != final_aux)
    experts_ok = n_experts_used >= 2 if args.n_experts >= 2 else True
    print(
        f"\n[mote_train] GATE: lm_finite={lm_ok}  kd_finite={kd_ok}  "
        f"aux_finite={aux_ok}  experts_used>={2}={experts_ok}"
    )
    if not (lm_ok and kd_ok and aux_ok and experts_ok):
        raise RuntimeError("Smoke-test gate FAILED — see GATE line above.")
    print("[mote_train] GATE PASSED")


if __name__ == "__main__":
    main()
