#!/usr/bin/env python3
"""build_blend.py ROLL_DIR DATA_DIR OUT_DIR — interleave a rollout cache 50/50 with data shards.

Rollout shards carry loss_mask (completion-masked); data shards don't (full-token general LM).
train_from_cache --completion-mask then trains completions on rollouts + all tokens on data,
so the generation-distribution signal is added *without* forgetting general text.
"""
import glob
import json
import os
import shutil
import sys

import torch

roll_dir, data_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
roll = sorted(glob.glob(f"{roll_dir}/batch_*.pt"))
data = sorted(glob.glob(f"{data_dir}/batch_*.pt"))[:len(roll)]
if not roll:
    raise SystemExit(f"no rollout shards in {roll_dir}")
os.makedirs(out, exist_ok=True)
r0 = torch.load(roll[0], map_location="cpu")
idx = 0
for r, d in zip(roll, data):
    shutil.copy(r, f"{out}/batch_{idx:06d}.pt"); idx += 1
    shutil.copy(d, f"{out}/batch_{idx:06d}.pt"); idx += 1
json.dump({"n_batches": idx, "batch_size": int(r0["input_ids"].shape[0]),
           "seq_len": int(r0["input_ids"].shape[1]), "top_k": 128, "tau": 1.0,
           "base": "Qwen/Qwen3-1.7B", "blend": f"{os.path.basename(roll_dir)}+data 50/50"},
          open(f"{out}/meta.json", "w"))
print(f"[blend] {idx} shards -> {out} ({len(roll)} rollout + {len(data)} data)", flush=True)
