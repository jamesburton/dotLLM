"""Check specific tensors element-by-element between llama.cpp and dotLLM dumps."""
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
    return data.reshape(shape) if shape else data, tuple(shape)


def show(name, path):
    data, shape = read_bin(path)
    flat = data.flatten()
    print(f"{name}: shape={list(shape)} n={flat.size}")
    if flat.size <= 16:
        print(f"  values: {flat.tolist()}")
    else:
        print(f"  first8: {flat[:8].tolist()}")
        print(f"  last8: {flat[-8:].tolist()}")
    print(f"  sum={float(flat.astype(np.float64).sum()):.6f} min={flat.min():.4f} max={flat.max():.4f} rms={np.sqrt((flat.astype(np.float64)**2).mean()):.4f}")


for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    show(name, path)
    print()
