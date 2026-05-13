"""Per-head ratio analysis of attn_output."""
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

ratios = []
for vh in range(NVHead):
    li = la[vh*DState:(vh+1)*DState].astype(np.float64)
    di = da[vh*DState:(vh+1)*DState].astype(np.float64)
    # Compute ratio llama / dotllm (where dotllm is nonzero)
    nz = np.abs(di) > 1e-10
    if nz.sum() == 0:
        ratios.append(float('nan'))
        continue
    r = li[nz] / di[nz]
    ratios.append((float(r.min()), float(r.max()), float(r.mean()), float(r.std()), nz.sum()))

print(f"{'vh':>3s}  {'ratio_mean':>12s} {'std':>10s} {'min':>10s} {'max':>10s}  nonzero")
for vh, r in enumerate(ratios):
    if r is None or not isinstance(r, tuple):
        print(f"{vh:3d}  -- nan --")
        continue
    rmin, rmax, rmean, rstd, nnz = r
    print(f"{vh:3d}  {rmean:12.6f} {rstd:10.6f} {rmin:10.4f} {rmax:10.4f}  {nnz}/128")
