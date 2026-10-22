"""KL of dumps vs a reference dump, binned by the REFERENCE entropy at each position (#873: the entropy confound).

usage (run in the dump directory): klentropy.py ref x1 x2 ...
Short chat prompts end on confident tokens (entropy ~0) and score KL ~0.001; document text has many 1-4 nat positions where the same
perturbation scores 0.03-0.1. Comparing KL across text types without this binning reads as "drift with context".
"""
import sys, numpy as np
from klpair import perpos, lsm, V

ref = sys.argv.pop(1)
A = np.memmap(ref + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, V)
pos = np.loadtxt(ref + ".pos.txt", dtype=np.int64)
ent = np.array([-(np.exp(lp) * lp).sum() for lp in (lsm(A[i]) for i in range(len(pos)))])
edges = [0, 0.1, 0.5, 1.5, 3, 20]
print("KL(ref||X) binned by the reference entropy (nats) at that position; n per bin:")
print(" " * 14 + "".join(f"{f'[{lo},{hi})':>12}" for lo, hi in zip(edges[:-1], edges[1:])))
print(" " * 14 + "".join(f"{int(((ent >= lo) & (ent < hi)).sum()):>12}" for lo, hi in zip(edges[:-1], edges[1:])))
for x in sys.argv[1:]:
    r = perpos(ref, x)
    n = min(len(r), len(ent))
    row = ""
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ent[:n] >= lo) & (ent[:n] < hi)
        row += f"{r[:n][m, 1].mean():12.4f}" if m.any() else f"{'-':>12}"
    print(f"{x:14}{row}")
