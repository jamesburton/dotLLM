#!/usr/bin/env python3
"""Standalone GPTQ warm-start: build the ternary student, calibrate, and save ckpt_warmstart.

No teacher cache needed. Designed to run on a big-memory box (e.g. the Strix iGPU, ~96 GB
addressable) that sidesteps the 12 GB memory wall the 3060 hits mid-warm-start; the resulting
checkpoint is handed back for `train_from_cache --resume-from` on the CUDA host.

  python scripts/lora/make_warmstart.py --base Qwen/Qwen3-1.7B --device cuda \
      --out F:/shared-assets/warmstart/qwen3-1p7b   # all-GPU (default; wants big VRAM)
      [--hessian-cpu]                                # park Hessians on CPU to spare VRAM
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitdistill as bd            # noqa: E402
import bitdistill_data as bdata    # noqa: E402
import gptq_init                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="dir to write ckpt_warmstart/ (student_state.pt + state.json)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--calib-batches", type=int, default=16)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--hessian-cpu", action="store_true", help="park Hessians on CPU (spare VRAM)")
    ap.add_argument("--cpt-dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--cpt-config", default="sample-10BT")
    ap.add_argument("--cpt-local-parquet", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"[warmstart] loading {args.base} ({dtype}) on {device} ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    student.config.use_cache = False
    info = bd.convert_to_bitnet_student(student)
    student.to(device).train()
    print(f"[warmstart] {info['bitlinears']} BitLinears + {info['subnorms']} SubLNs; calibrating "
          f"({args.calib_batches}x{args.calib_batch_size} seqs, hessian_device="
          f"{'cpu' if args.hessian_cpu else str(device)}) ...", flush=True)

    calib = bdata.cpt_token_stream(tok, args.seq_len, dataset_name=args.cpt_dataset,
                                   dataset_config=args.cpt_config,
                                   local_parquet=args.cpt_local_parquet, seed=4242)
    t0 = time.time()
    stats = gptq_init.gptq_warmstart_init(
        student, calib, device,
        n_calib_batches=args.calib_batches, calib_batch_size=args.calib_batch_size,
        hessian_device=("cpu" if args.hessian_cpu else None))
    print(f"[warmstart] DONE in {time.time() - t0:.0f}s; aggregate MSE "
          f"{stats['mse_absmean']:.3e} -> {stats['mse_warmstart']:.3e} "
          f"({100 * stats['mse_reduction']:.1f}%)", flush=True)

    ck = os.path.join(args.out, "ckpt_warmstart")
    bd.save_checkpoint(student, ck, 0, 0,
                       extra={"warmstart": True, "base": args.base,
                              "mse_reduction": stats.get("mse_reduction")})
    print(f"[warmstart] saved -> {ck}/student_state.pt "
          f"({os.path.getsize(os.path.join(ck, 'student_state.pt')) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
