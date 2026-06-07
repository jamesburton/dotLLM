"""Per-head diff of attn_output for layer 0."""
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


la = read_bin(sys.argv[1])
da = read_bin(sys.argv[2])

la_flat = la.flatten()
da_flat = da.flatten()
NVHead = 32
DState = 128

print(f"llama shape={la.shape} dotllm shape={da.shape}")
print(f"per-head diff (first8/last8/sum/rel_err):")
for vh in range(NVHead):
    li = la_flat[vh*DState:(vh+1)*DState]
    di = da_flat[vh*DState:(vh+1)*DState]
    diff = (li - di).astype(np.float64)
    err_rms = float(np.sqrt((diff ** 2).mean()))
    ref_rms = float(np.sqrt((li.astype(np.float64) ** 2).mean()))
    cos = float(np.dot(li, di) / (np.linalg.norm(li) * np.linalg.norm(di) + 1e-30))
    print(f"  vh={vh:2d}: llama_rms={ref_rms:.5f} dotllm_rms={float(np.sqrt((di.astype(np.float64)**2).mean())):.5f} err_rms={err_rms:.4e} cos={cos:.4f}")
