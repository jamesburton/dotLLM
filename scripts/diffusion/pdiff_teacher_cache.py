#!/usr/bin/env python3
"""pdiff_teacher_cache.py — Dump top-K teacher logits for offline ternary KD-QAT.

The teacher is the base BitNet model with a PEFT adapter attached WITHOUT merging
(the hybrid form that evaluates at PPL ~12.4). The student will consume these logits
during KD-QAT in pdiff_ternary_qat.py --kd-teacher-cache.

Two sequential GPU passes keep memory within 12 GB:
  Pass 1 (this script):  teacher-only, forward over a FIXED corpus -> write top-K cache
  Pass 2 (qat script):   student-only, reads the cache, no teacher in memory

Produces:
  <out>/topk_idx.npy   -- int32   [N_SEQS, SEQ_LEN, K]  top-K vocab indices
  <out>/topk_logit.npy -- float16 [N_SEQS, SEQ_LEN, K]  teacher RAW logits (not softmaxed)
  <out>/meta.json      -- {"model_id", "adapter", "K", "seq_len", "n_seqs", "vocab_size"}
  <out>/progress.json  -- {"next_row": int}  (updated per batch, enables --resume)

Alignment contract
------------------
  topk_idx[i, t, :]   = top-K vocab indices from teacher at position t
                         predicting the NEXT token ids[i, t+1] (teacher-forcing).
  Final position t=SEQ_LEN-1 predicts a pad; trainer ignores it via [:-1] slice.
  Corpus (--corpus) MUST be the EXACT same ids.npy produced by pdiff_kd_corpus.py.

Memory safety
-------------
  Logits are computed per-batch and written to memmaps immediately.
  Only one batch of logits [B, L, V] lives in VRAM at a time.
  B=4 × L=512 × V=32000 × 2B (fp16) ≈ 128 MB per batch — safe on 12 GB.

CLI:
    # Validate flags (no model load):
    python scripts/diffusion/pdiff_teacher_cache.py --help

    # GPU run — full corpus:
    python scripts/diffusion/pdiff_teacher_cache.py \\
        --corpus E:/kd/corpus_no_robots \\
        --adapter .docs/diffusion/pdiff_boot_fixedinit/pdiff_func_out38/gen_adapter \\
        --out E:/kd/teacher_cache \\
        --device cuda

    # Resume interrupted run:
    python scripts/diffusion/pdiff_teacher_cache.py \\
        --corpus E:/kd/corpus_no_robots \\
        --adapter .docs/diffusion/pdiff_boot_fixedinit/pdiff_func_out38/gen_adapter \\
        --out E:/kd/teacher_cache \\
        --device cuda --resume

    # Smoke (tiny subset, CPU, validates end-to-end without a real model):
    python scripts/diffusion/pdiff_teacher_cache.py \\
        --corpus /tmp/kd_smoke \\
        --adapter .docs/diffusion/pdiff_boot_fixedinit/pdiff_func_out38/gen_adapter \\
        --out /tmp/kd_cache_smoke \\
        --device cpu --max-seqs 2 --batch-size 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure UTF-8 stdout/stderr on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_DEFAULT_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"
_DEFAULT_K = 64
_DEFAULT_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pdiff_teacher_cache] device=auto -> {dev}")
        return dev
    return torch.device(requested)


def _open_or_create_memmap(
    path: str, shape: tuple, dtype: np.dtype, resume: bool, start_row: int
) -> np.ndarray:
    if resume and start_row > 0 and os.path.isfile(path):
        mm = np.lib.format.open_memmap(path, mode="r+", dtype=dtype, shape=shape)
        print(
            f"[pdiff_teacher_cache] opened existing memmap "
            f"{os.path.basename(path)} (resume, shape={shape})"
        )
    else:
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        print(
            f"[pdiff_teacher_cache] created memmap "
            f"{os.path.basename(path)} shape={shape} dtype={dtype}"
        )
    return mm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Dump top-K teacher logits (base + PEFT adapter, NOT merged) to disk "
            "for offline KD-QAT in pdiff_ternary_qat.py --kd-teacher-cache. "
            "Run on GPU before the student KD-QAT pass — the two models never "
            "co-exist in memory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--corpus",
        required=True,
        help="Corpus directory from pdiff_kd_corpus.py (contains ids.npy + meta.json).",
    )
    ap.add_argument(
        "--adapter",
        required=True,
        help=(
            "PEFT LoRA adapter directory to load on top of the base model. "
            "The adapter is attached WITHOUT merging (PeftModel.from_pretrained), "
            "so the teacher runs in the hybrid bf16 form (PPL ~12.4). "
            "Pass 'none' to use the base model alone (PPL ~13.72) — useful for baselines."
        ),
    )
    ap.add_argument("--out", required=True, help="Output cache directory.")
    ap.add_argument(
        "--k",
        type=int,
        default=_DEFAULT_K,
        help="Top-K logits to store per token position.",
    )
    ap.add_argument(
        "--base",
        default=_DEFAULT_BASE,
        help="HF model id or local path for the base BitNet model.",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
        help="Inference device. 'auto' picks CUDA if available, else CPU.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Sequences per forward pass. 4 is safe on 12 GB with seq_len=512.",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Process at most this many sequences (smoke / partial run). Default: all.",
    )
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Model compute dtype. bf16 is default and halves memory vs fp32.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an interrupted run (reads progress.json).",
    )
    args = ap.parse_args()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Device + dtype
    # ------------------------------------------------------------------
    device = _resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Read corpus metadata
    # ------------------------------------------------------------------
    corpus_meta_path = os.path.join(args.corpus, "meta.json")
    if not os.path.isfile(corpus_meta_path):
        raise FileNotFoundError(f"Corpus meta.json not found at {corpus_meta_path!r}")
    with open(corpus_meta_path, encoding="utf-8") as fh:
        corpus_meta = json.load(fh)
    total_seqs = corpus_meta["n_seqs"]
    seq_len = corpus_meta["seq_len"]
    print(
        f"[pdiff_teacher_cache] corpus: {total_seqs} seqs × {seq_len} tokens "
        f"from {args.corpus!r}"
    )

    n_seqs = total_seqs
    if args.max_seqs is not None:
        n_seqs = min(total_seqs, args.max_seqs)
        print(
            f"[pdiff_teacher_cache] --max-seqs={args.max_seqs}: "
            f"processing {n_seqs} of {total_seqs}"
        )

    # ------------------------------------------------------------------
    # 3. Resume: find next unwritten row
    # ------------------------------------------------------------------
    progress_path = os.path.join(args.out, "progress.json")
    start_row = 0
    if args.resume and os.path.isfile(progress_path):
        with open(progress_path, encoding="utf-8") as fh:
            prog = json.load(fh)
        start_row = int(prog.get("next_row", 0))
        if start_row >= n_seqs:
            print(
                f"[pdiff_teacher_cache] Already complete "
                f"(progress={start_row}, n_seqs={n_seqs}). Nothing to do."
            )
            return
        print(f"[pdiff_teacher_cache] resuming from row {start_row}/{n_seqs}")
    elif not args.resume and os.path.isfile(progress_path):
        print(
            "[pdiff_teacher_cache] NOTE: progress.json exists but --resume not set. "
            "Starting fresh (overwriting)."
        )

    # ------------------------------------------------------------------
    # 4. Load corpus ids (memory-mapped)
    # ------------------------------------------------------------------
    ids_path = os.path.join(args.corpus, "ids.npy")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"ids.npy not found at {ids_path!r}")
    ids = np.load(ids_path, mmap_mode="r")  # [N_TOTAL, SEQ_LEN] int32
    print(f"[pdiff_teacher_cache] ids.npy mmap: shape={ids.shape}  dtype={ids.dtype}")

    # ------------------------------------------------------------------
    # 5. Open / create output memmaps
    # ------------------------------------------------------------------
    topk_shape = (n_seqs, seq_len, args.k)
    topk_idx_path = os.path.join(args.out, "topk_idx.npy")
    topk_logit_path = os.path.join(args.out, "topk_logit.npy")

    topk_idx = _open_or_create_memmap(
        topk_idx_path, topk_shape, np.int32, resume=args.resume, start_row=start_row
    )
    topk_logit_mm = _open_or_create_memmap(
        topk_logit_path, topk_shape, np.float16, resume=args.resume, start_row=start_row
    )

    # ------------------------------------------------------------------
    # 6. Load teacher: base + PEFT adapter (NO merge)
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM

    print(f"[pdiff_teacher_cache] loading base model {args.base!r} ...")
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=dtype, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=dtype,
            device_map={"": device},
            local_files_only=True,
        )

    use_adapter = args.adapter.strip().lower() != "none"
    if use_adapter:
        from peft import PeftModel

        adapter_path = args.adapter.strip()
        print(
            f"[pdiff_teacher_cache] attaching PEFT adapter from {adapter_path!r} "
            "(NO merge — teacher runs in hybrid bf16 form) ..."
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        # Do NOT call merge_and_unload() — keep the adapter live so the teacher runs
        # at its full hybrid PPL (~12.4), not at base PPL (~13.72).
    else:
        print("[pdiff_teacher_cache] --adapter=none: using base model only (PPL ~13.72)")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False  # disable KV cache for memory efficiency

    vocab_size: int = model.config.vocab_size
    print(
        f"[pdiff_teacher_cache] teacher ready: "
        f"vocab_size={vocab_size}  device={device}  dtype={dtype}"
    )

    # ------------------------------------------------------------------
    # 7. Write meta.json (now that vocab_size is known)
    # ------------------------------------------------------------------
    meta = {
        "model_id": args.base,
        "adapter": args.adapter.strip() if use_adapter else None,
        "K": args.k,
        "seq_len": seq_len,
        "n_seqs": n_seqs,
        "vocab_size": vocab_size,
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[pdiff_teacher_cache] meta.json: {json.dumps(meta)}")

    # ------------------------------------------------------------------
    # 8. Main dump loop (batched, memory-safe, resumable)
    # ------------------------------------------------------------------
    print(
        f"\n[pdiff_teacher_cache] dumping rows {start_row}..{n_seqs-1} "
        f"in batches of {args.batch_size} ..."
    )
    batch_times: list = []
    row = start_row
    while row < n_seqs:
        batch_end = min(row + args.batch_size, n_seqs)
        t_batch = time.perf_counter()

        # Load batch of input ids (mmap slice, no full copy)
        batch_ids_np = np.array(ids[row:batch_end], dtype=np.int64)
        input_tensor = torch.from_numpy(batch_ids_np).to(device)

        with torch.no_grad():
            out = model(input_ids=input_tensor)
        logits = out.logits  # [B, SEQ_LEN, V]  in fp16/bf16

        # Top-K per position: [B, SEQ_LEN, K]
        topk_vals, topk_inds = torch.topk(logits, args.k, dim=-1)

        # Cast and write to memmaps immediately (free GPU memory after)
        topk_idx[row:batch_end] = topk_inds.cpu().to(torch.int32).numpy()
        topk_logit_mm[row:batch_end] = topk_vals.cpu().to(torch.float16).numpy()

        # Flush each batch so partial runs are usable on --resume
        topk_idx.flush()
        topk_logit_mm.flush()

        # Update progress (enables --resume after interruption)
        row = batch_end
        with open(progress_path, "w", encoding="utf-8") as fh:
            json.dump({"next_row": row}, fh)

        # Logging + ETA
        elapsed_batch = time.perf_counter() - t_batch
        batch_times.append(elapsed_batch)
        pct = 100.0 * row / n_seqs
        recent = batch_times[-10:]
        mean_t = sum(recent) / len(recent)
        batches_left = (n_seqs - row) / max(args.batch_size, 1)
        eta_s = mean_t * batches_left
        eta_str = f"  ETA ~{eta_s/60:.1f}min" if eta_s > 5 else ""
        print(
            f"  row {row:6d}/{n_seqs}  ({pct:5.1f}%)  "
            f"batch {elapsed_batch:.2f}s{eta_str}",
            flush=True,
        )

        # Explicit VRAM cleanup between batches
        del logits, topk_vals, topk_inds, input_tensor, out, batch_ids_np
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t_start
    print(f"\n[pdiff_teacher_cache] DONE in {total_elapsed / 60:.1f} min")
    print(f"  topk_idx.npy   : {topk_idx_path}")
    print(f"    shape={topk_idx.shape}  dtype={topk_idx.dtype}")
    print(f"  topk_logit.npy : {topk_logit_path}")
    print(f"    shape={topk_logit_mm.shape}  dtype={topk_logit_mm.dtype}")
    print(f"  meta.json      : {meta_path}")
    print(f"  progress.json  : {progress_path}  (next_row={n_seqs}, complete)")
    print()
    print("[pdiff_teacher_cache] Contract verification:")
    print(f"  topk_idx[i, t, :]   = top-{args.k} vocab indices, teacher logits at position t")
    print(f"                        predicting ids[i, t+1]  (standard teacher-forcing)")
    print(f"  topk_logit[i, t, :] = raw logits (NOT softmaxed) for those indices, float16")
    print(f"  Final position t={seq_len-1} predicts pad; trainer ignores via [:-1] slice.")
    print()
    print("[pdiff_teacher_cache] Next step: run pdiff_ternary_qat.py with:")
    print(f"  --kd-teacher-cache {args.out}")
    print(f"  --kd-corpus {args.corpus}")


if __name__ == "__main__":
    main()
