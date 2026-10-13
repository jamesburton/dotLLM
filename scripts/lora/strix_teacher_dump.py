"""Dump top-K teacher logits to disk for offline knowledge distillation.

Reads the corpus produced by materialize_kd_corpus.py and writes:

  <out>/topk_idx.npy   -- int32   [N_SEQS, SEQ_LEN, K]  top-K vocab indices
  <out>/topk_logit.npy -- float16 [N_SEQS, SEQ_LEN, K]  teacher RAW logits (not softmaxed)
  <out>/meta.json      -- {"model_id", "K", "seq_len", "n_seqs", "vocab_size"}
  <out>/progress.json  -- {"next_row": int}  (resumable — updated per batch)

Alignment contract
------------------
  topk_idx[i, t, :] = top-K vocab indices from teacher logits at position t.
  These are the teacher's predictions for the NEXT token ids[i, t+1], which is
  the standard causal language model teacher-forcing convention. The logit at
  the final position (t = SEQ_LEN-1) predicts a pad token; values are written
  but the trainer ignores them.

Memory safety
-------------
  Logits are computed per-batch and written to memory-mapped arrays immediately.
  No more than one batch of logits ([B, L, V]) lives in RAM at a time.
  On Strix (128 GB, AMD CPU), B=4 x L=512 x V=32000 x 4B ≈ 256 MB per batch.

Resumable
---------
  Pass --resume to skip rows already written (tracked via progress.json).
  Interrupted runs can be restarted with the same command + --resume.

Usage (on Strix, CPU — non-CUDA box)
--------------------------------------
  # 1. smoke test (8 seqs, validates end-to-end without a full model load delay)
  python strix_teacher_dump.py \\
      --corpus /path/to/corpus --out /path/to/cache \\
      --max-seqs 8 --device cpu --batch-size 2

  # 2. full pass (resumable)
  python strix_teacher_dump.py \\
      --corpus /path/to/corpus --out /path/to/cache \\
      --device cpu --batch-size 4 --resume

See STRIX_TEACHER.md for the full step-by-step runbook.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Must match mote_train.py's --base (the weights the student is upcycled from).
_DEFAULT_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
_DEFAULT_K = 64
_DEFAULT_BATCH_SIZE = 4


def _resolve_device(requested: str) -> torch.device:
    """Resolve 'auto' or validate a literal device string."""
    if requested == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[teacher_dump] device=auto -> {dev}")
        return dev
    return torch.device(requested)


def _load_model_safe(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load the teacher with CPU fallback.

    BitNet models may not support device_map on CPU paths; if placement on
    the requested device fails we fall back to CPU with a clear message.
    Returns (model, actual_device).
    """
    from transformers import AutoModelForCausalLM

    if device.type == "cpu":
        print(f"[teacher_dump] loading {model_id!r} on CPU (dtype={dtype}) ...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        except Exception as exc:
            print(
                f"[teacher_dump] WARNING: failed to load with dtype={dtype} ({exc}). "
                "Retrying with fp32 ..."
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        return model, torch.device("cpu")

    # Non-CPU path (CUDA etc.)
    try:
        print(f"[teacher_dump] loading {model_id!r} on {device} (dtype={dtype}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map={"": device}
        )
        return model, device
    except Exception as exc:
        print(
            f"[teacher_dump] WARNING: failed to load on {device} ({exc}). "
            "Falling back to CPU."
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        return model, torch.device("cpu")


def _open_or_create_memmap(
    path: str, shape: tuple, dtype: np.dtype, resume: bool, start_row: int
) -> np.ndarray:
    """Open an existing memmap (resume) or create a fresh one."""
    if resume and start_row > 0 and os.path.isfile(path):
        mm = np.lib.format.open_memmap(path, mode="r+", dtype=dtype, shape=shape)
        print(f"[teacher_dump] opened existing memmap {os.path.basename(path)} (resume, shape={shape})")
    else:
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        print(f"[teacher_dump] created memmap {os.path.basename(path)} shape={shape} dtype={dtype}")
    return mm


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Dump top-K teacher logits to disk for offline KD. "
            "Designed to run on Strix (CPU, non-CUDA, 128 GB). "
            "See STRIX_TEACHER.md for the full runbook."
        )
    )
    ap.add_argument(
        "--corpus",
        required=True,
        help="Corpus directory produced by materialize_kd_corpus.py (contains ids.npy + meta.json)",
    )
    ap.add_argument("--out", required=True, help="Output cache directory")
    ap.add_argument(
        "--k",
        type=int,
        default=_DEFAULT_K,
        help=f"Top-K logits to store per token position (default: {_DEFAULT_K})",
    )
    ap.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"HF model id or local path (default: {_DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help=(
            "Inference device. 'auto' picks CUDA if available, else CPU. "
            "On Strix (non-CUDA, AMD) this always resolves to 'cpu'. (default: auto)"
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=(
            f"Sequences per forward pass. "
            f"Larger values are faster but use more RAM for intermediate logits. "
            f"On Strix CPU, 4-8 is a safe default. (default: {_DEFAULT_BATCH_SIZE})"
        ),
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
        help=(
            "Model compute dtype. bf16 halves memory vs fp32 and is supported "
            "on modern AMD CPUs. Use fp32 if you see NaN outputs. (default: bf16)"
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an interrupted run. "
            "Reads progress.json to find the next unwritten row and skips it. "
            "Existing memmaps are opened in r+ mode; no data is overwritten."
        ),
    )
    args = ap.parse_args()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Resolve device + dtype
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
        f"[teacher_dump] corpus: {total_seqs} seqs x {seq_len} tokens "
        f"from {args.corpus!r}"
    )

    # Respect --max-seqs (smoke / partial run)
    n_seqs = total_seqs
    if args.max_seqs is not None:
        n_seqs = min(total_seqs, args.max_seqs)
        print(f"[teacher_dump] --max-seqs={args.max_seqs}: processing {n_seqs} of {total_seqs} seqs")

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
            print(f"[teacher_dump] Already complete (progress={start_row}, n_seqs={n_seqs}). Nothing to do.")
            return
        print(f"[teacher_dump] resuming from row {start_row} ({start_row}/{n_seqs})")
    elif not args.resume and os.path.isfile(progress_path):
        print(
            f"[teacher_dump] NOTE: progress.json exists but --resume not set. "
            "Starting fresh (existing files will be overwritten)."
        )

    # ------------------------------------------------------------------
    # 4. Load corpus ids (memory-mapped; never fully read into RAM)
    # ------------------------------------------------------------------
    ids_path = os.path.join(args.corpus, "ids.npy")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"ids.npy not found at {ids_path!r}")
    ids = np.load(ids_path, mmap_mode="r")  # [N_TOTAL, SEQ_LEN] int32
    print(f"[teacher_dump] ids.npy mmap: shape={ids.shape}  dtype={ids.dtype}")

    # ------------------------------------------------------------------
    # 5. Open / create output memmaps
    # ------------------------------------------------------------------
    topk_shape = (n_seqs, seq_len, args.k)
    topk_idx_path = os.path.join(args.out, "topk_idx.npy")
    topk_logit_path = os.path.join(args.out, "topk_logit.npy")

    topk_idx = _open_or_create_memmap(
        topk_idx_path, topk_shape, np.int32, resume=args.resume, start_row=start_row
    )
    topk_logit = _open_or_create_memmap(
        topk_logit_path, topk_shape, np.float16, resume=args.resume, start_row=start_row
    )

    # ------------------------------------------------------------------
    # 6. Load teacher model
    # ------------------------------------------------------------------
    model, actual_device = _load_model_safe(args.model, device, dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vocab_size: int = model.config.vocab_size
    print(f"[teacher_dump] teacher ready: vocab_size={vocab_size}  actual_device={actual_device}")

    # ------------------------------------------------------------------
    # 7. Write meta.json (now that vocab_size is known)
    # ------------------------------------------------------------------
    meta = {
        "model_id": args.model,
        "K": args.k,
        "seq_len": seq_len,
        "n_seqs": n_seqs,
        "vocab_size": vocab_size,
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[teacher_dump] meta.json: {json.dumps(meta)}")

    # ------------------------------------------------------------------
    # 8. Main dump loop (batched, memory-safe, resumable)
    # ------------------------------------------------------------------
    print(
        f"\n[teacher_dump] dumping rows {start_row}..{n_seqs-1} "
        f"in batches of {args.batch_size} ..."
    )
    rows_done = start_row
    batch_times: list = []

    row = start_row
    while row < n_seqs:
        batch_end = min(row + args.batch_size, n_seqs)
        t_batch = time.perf_counter()

        # Load batch of input ids and run forward
        batch_ids = ids[row:batch_end]  # [B, SEQ_LEN] int32 (mmap slice, no copy)
        input_tensor = torch.from_numpy(
            np.array(batch_ids, dtype=np.int64)  # int64 required by HF models
        ).to(actual_device)

        with torch.no_grad():
            out = model(input_ids=input_tensor)
        logits = out.logits  # [B, SEQ_LEN, V]

        # Top-K per position: [B, SEQ_LEN, K] for both indices and values
        topk_vals, topk_inds = torch.topk(logits, args.k, dim=-1)

        # Cast and write to memmaps
        topk_idx[row:batch_end] = topk_inds.cpu().to(torch.int32).numpy()
        topk_logit[row:batch_end] = topk_vals.cpu().to(torch.float16).numpy()

        # Flush each batch so partial runs are usable on resume
        topk_idx.flush()
        topk_logit.flush()

        # Update progress file atomically (write-then-rename would be safer on Unix;
        # simple overwrite is fine here since partial writes are tiny)
        row = batch_end
        with open(progress_path, "w", encoding="utf-8") as fh:
            json.dump({"next_row": row}, fh)

        # Logging
        elapsed_batch = time.perf_counter() - t_batch
        batch_times.append(elapsed_batch)
        rows_done = row
        pct = 100.0 * rows_done / n_seqs
        # Rolling ETA from last 10 batches
        recent = batch_times[-10:]
        mean_batch_t = sum(recent) / len(recent)
        rows_per_batch = args.batch_size
        batches_left = (n_seqs - rows_done) / max(rows_per_batch, 1)
        eta_s = mean_batch_t * batches_left
        eta_str = f"  ETA ~{eta_s/60:.1f}min" if eta_s > 5 else ""
        print(
            f"  row {rows_done:6d}/{n_seqs}  ({pct:5.1f}%)  "
            f"batch {elapsed_batch:.1f}s{eta_str}",
            flush=True,
        )

        # Free GPU memory explicitly between batches
        del logits, topk_vals, topk_inds, input_tensor, out
        if actual_device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 9. Final summary + contract verification printout
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t_start
    print(f"\n[teacher_dump] DONE in {total_elapsed/60:.1f} min")
    print(f"  topk_idx.npy   : {topk_idx_path}")
    print(f"    shape={topk_idx.shape}  dtype={topk_idx.dtype}")
    print(f"  topk_logit.npy : {topk_logit_path}")
    print(f"    shape={topk_logit.shape}  dtype={topk_logit.dtype}")
    print(f"  meta.json      : {meta_path}")
    print(f"  progress.json  : {progress_path}  (next_row={n_seqs}, complete)")
    print()
    print("[teacher_dump] Contract verification:")
    print(f"  topk_idx[i, t, :]   = top-{args.k} vocab indices, teacher logits at position t")
    print(f"                        predicting ids[i, t+1]  (standard teacher-forcing)")
    print(f"  topk_logit[i, t, :] = raw logits (NOT softmaxed) for those indices, float16")
    print(f"  Final position t={seq_len-1} predicts pad; trainer is expected to ignore it.")


if __name__ == "__main__":
    main()
