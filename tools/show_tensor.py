"""Print first/last 3 elements and sum of a dotLLM tensor dump.

Usage: python show_tensor.py <bin_path>
"""
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


for path in sys.argv[1:]:
    data = read_bin(path)
    flat = data.flatten()
    print(f"{Path(path).name}: shape={list(data.shape)} n={flat.size}")
    print(f"  first3: {flat[0]:.4f}, {flat[1]:.4f}, {flat[2]:.4f}")
    print(f"  last3:  {flat[-3]:.4f}, {flat[-2]:.4f}, {flat[-1]:.4f}")
    print(f"  sum:    {flat.astype(np.float64).sum():.6f}")
    print(f"  min={flat.min():.4f} max={flat.max():.4f} rms={np.sqrt((flat.astype(np.float64)**2).mean()):.4f}")
