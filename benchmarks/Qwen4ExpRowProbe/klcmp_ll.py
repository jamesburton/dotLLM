#!/usr/bin/env python
"""KL between a KlCurve dump and a llama.cpp `--save-all-logits` file (#873).

usage: klcmp_ll.py <prefix> <llama.bin> <vocab> [refIsLlama=0]
llama-perplexity saves log-probs (uint16-quantised per row) for the SECOND HALF of each window only (positions n_ctx/2 .. n_ctx-2).
Prints per-bucket mean KL(ours||llama) (or KL(llama||ours) with refIsLlama=1) over our sampled positions inside that range.

CAVEATS (#873): (1) llama.cpp tokenizes the text file ITSELF. On the 8192-token held-out window it matched dotLLM's ids only for the first 2005
tokens (it then splits some text differently), so a comparison is valid ONLY at positions below the first token mismatch: use a window whose
scored half lies below it (-c 2048, positions 1024..2003, which is what the #873 numbers use, n=61) and check `toks != ids` first.
(2) The record offset is empirical, not derived from the format: SHIFT is the number of records by which the file is offset from
`first = n_ctx/2`. For the matched 2048 window it is 0 (argmax agreement 27/30); on the mismatched 8192 file a "fit" of 4 and later 8 is only
the token-stream offset, and is meaningless. Re-fit with an argmax-agreement scan for every new file and pass LL_SHIFT / LL_PMAX explicitly.
"""
import os
import sys
import numpy as np

prefix, llf, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
ref_ll = len(sys.argv) > 4 and sys.argv[4] == "1"
raw = open(llf, "rb")
assert raw.read(8) == b"_logits_", "bad magic"
n_ctx, n_vocab, n_chunk = np.frombuffer(raw.read(12), dtype=np.int32)
assert n_vocab == vocab, (n_vocab, vocab)
toks = np.frombuffer(raw.read(4 * n_ctx * n_chunk), dtype=np.int32)
first = n_ctx // 2
SHIFT = int(os.environ['LL_SHIFT'])   # empirical record offset (see caveats); required, no default
PMAX = int(os.environ['LL_PMAX'])     # last position known to share the token stream with dotLLM's ids; required, no default
nrec = n_ctx - 1 - first
nv = 2 * ((vocab + 1) // 2) + 4  # uint16 units per record
rec = np.memmap(llf, dtype=np.uint16, mode="r", offset=raw.tell()).reshape(-1, nv)[:nrec]
print(f"llama file: n_ctx={n_ctx} chunks={n_chunk} records={len(rec)} first scored pos={first}")

pos = np.loadtxt(prefix + ".pos.txt", dtype=np.int64, ndmin=1)
L = np.memmap(prefix + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, vocab)


def ll_logprobs(r):
    h = r[:4].view(np.float32)
    # the header holds a positive scale and a negative min log-prob; order is detected
    scale, mn = (h[0], h[1]) if h[0] > 0 else (h[1], h[0])
    return mn + scale * r[4:4 + vocab].astype(np.float64)


def lsm(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


rows = []
for i, p in enumerate(pos):
    if p < first or p > PMAX or p - first + SHIFT >= nrec:
        continue
    lo = lsm(L[i])
    ll = ll_logprobs(rec[p - first + SHIFT])
    a, b = (ll, lo) if ref_ll else (lo, ll)
    kl = float((np.exp(a) * (a - b)).sum())
    rows.append((p, kl, float(np.argmax(lo) == np.argmax(ll))))
rows = np.array(rows)
edges = [0, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
print(f"{'bucket':>12} {'n':>4} {'meanKL':>9} {'medKL':>9} {'top1':>6}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    if m.any():
        print(f"{lo:>5}-{hi:<6} {m.sum():>4} {rows[m, 1].mean():9.5f} {np.median(rows[m, 1]):9.5f} {rows[m, 2].mean():6.3f}")
print(f"all scored: n={len(rows)} meanKL {rows[:, 1].mean():.5f}")
if len(sys.argv) > 5:
    np.savetxt(sys.argv[5], rows, fmt="%d %.8g %.0f")
