"""Check if dotLLM's per-head attn_output is a permutation of llama.cpp's."""
import sys, struct
from pathlib import Path
import numpy as np


def read_bin(path):
    with open(path, "rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape) if shape else data


la = read_bin(sys.argv[1]).flatten()
da = read_bin(sys.argv[2]).flatten()

NVHead = 32
DState = 128

# For each dotLLM head, find best matching llama head by cosine similarity
print(f"For each dotLLM head dvh, best-matching llama head lvh (by cosine sim):")
print(f"{'dvh':>4s} {'lvh':>4s} {'cos':>10s} {'ratio_llama/dotllm':>18s}")
for dvh in range(NVHead):
    di = da[dvh*DState:(dvh+1)*DState].astype(np.float64)
    di_norm = di / (np.linalg.norm(di) + 1e-30)
    best_cos = -2.0
    best_lvh = -1
    for lvh in range(NVHead):
        li = la[lvh*DState:(lvh+1)*DState].astype(np.float64)
        li_norm = li / (np.linalg.norm(li) + 1e-30)
        c = float(np.dot(di_norm, li_norm))
        if c > best_cos:
            best_cos = c
            best_lvh = lvh
    li = la[best_lvh*DState:(best_lvh+1)*DState].astype(np.float64)
    # Ratio with same-direction sign
    li_n = np.linalg.norm(li); di_n = np.linalg.norm(di)
    ratio = li_n / di_n if di_n > 1e-12 else float('inf')
    print(f"{dvh:4d} {best_lvh:4d} {best_cos:10.6f} {ratio:18.4f}")
