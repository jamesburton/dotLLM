#!/usr/bin/env python
"""Position-bucketed KL between two KlCurve dumps (#873).

usage: klcmp.py <prefixA> <prefixB> <vocab> [csv]
KL(A||B) per sampled position over the FULL vocab, then mean per position bucket, cumulative mean up to each length,
top-1 agreement, and the NLL (PPL) of each side per bucket from the all-position nll files.
"""
import sys
import numpy as np

pa, pb, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
pos = np.loadtxt(pa + ".pos.txt", dtype=np.int64, ndmin=1)
A = np.memmap(pa + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, vocab)
B = np.memmap(pb + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, vocab)
n = min(len(A), len(B), len(pos))
pos = pos[:n]


def lsm(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


kl = np.zeros(n)
t1 = np.zeros(n, dtype=bool)
for i in range(n):
    la, lb = lsm(A[i]), lsm(B[i])
    kl[i] = float((np.exp(la) * (la - lb)).sum())
    t1[i] = A[i].argmax() == B[i].argmax()

edges = [0, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
na = np.fromfile(pa + ".nll.f32", dtype=np.float32)
nb = np.fromfile(pb + ".nll.f32", dtype=np.float32)
print(f"{pa} vs {pb}: {n} sampled positions, vocab {vocab}")
print(f"{'bucket':>12} {'n':>4} {'meanKL':>9} {'medKL':>9} {'top1':>6} {'PPL_A':>8} {'PPL_B':>8} {'cumKL':>9}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (pos >= lo) & (pos < hi)
    if not m.any():
        continue
    cum = (pos < hi)
    sl = slice(lo, min(hi, len(na)))
    ppa = np.exp(na[sl].mean()) if sl.stop > sl.start else float("nan")
    ppb = np.exp(nb[sl].mean()) if sl.stop > sl.start else float("nan")
    print(f"{lo:>5}-{hi:<6} {m.sum():>4} {kl[m].mean():9.5f} {np.median(kl[m]):9.5f} {t1[m].mean():6.3f} {ppa:8.3f} {ppb:8.3f} {kl[cum].mean():9.5f}")
if len(sys.argv) > 4:
    np.savetxt(sys.argv[4], np.stack([pos, kl, t1.astype(float)], 1), fmt="%d %.8g %.0f")
