"""Pairwise KL matrix between KlCurve dumps, split dense (<2048) | sparse (>=2048) regime (#873).

usage (run in the dump directory): klpair.py a b c ...   -> KL(row||col), mean over the sampled positions of each regime.
Per-pair per-position results are cached as _kl_<a>__<b>.npy.
"""
import sys, os, numpy as np
V = 248320


def lsm(x):
    x = x.astype(np.float64); m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def perpos(a, b):
    cache = f"_kl_{a}__{b}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    pa = np.loadtxt(a + ".pos.txt", dtype=np.int64, ndmin=1)
    pb = np.loadtxt(b + ".pos.txt", dtype=np.int64, ndmin=1)
    n = min(len(pa), len(pb))
    A = np.memmap(a + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, V)
    B = np.memmap(b + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, V)
    out = np.zeros((n, 2))
    for i in range(n):
        la, lb = lsm(A[i]), lsm(B[i])
        out[i] = (pa[i], float((np.exp(la) * (la - lb)).sum()))
    np.save(cache, out)
    return out


if __name__ == "__main__":
    names = sys.argv[1:]
    print("pairwise KL(row||col): dense(<2048) | sparse(>=2048)")
    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append("     -     "); continue
            r = perpos(a, b)
            row.append(f"{r[r[:,0]<2048,1].mean():.3f}|{r[r[:,0]>=2048,1].mean():.3f}")
        print(f"{a:12}", "  ".join(row))
