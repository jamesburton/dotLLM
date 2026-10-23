#!/usr/bin/env python
"""Per-position KL / top-1 between NemotronProbe logits and a llama.cpp `--save-all-logits` file (#375).

usage:
  nemprobe_kl.py ids  <kld.bin> <ids-out.txt>          # token ids llama.cpp scored (feed to `NemotronProbe logits`)
  nemprobe_kl.py cmp  <kld.bin> <probe-prefix>          # compare
llama-perplexity stores uint16-quantised log-probs for positions n_ctx/2 .. n_ctx-2 of EACH window (chunk-major);
the probe writes the same rows from the same ids, so the comparison is position-for-position with no shift.
Prints mean KL(llama || ours), top-1 agreement, and the NLL of the actual next token for both sides.
"""
import sys
import numpy as np

mode, kld = sys.argv[1], sys.argv[2]
raw = open(kld, "rb")
assert raw.read(8) == b"_logits_", "bad magic"
n_ctx, n_vocab, n_chunk = (int(x) for x in np.frombuffer(raw.read(12), dtype=np.int32))
ids = np.frombuffer(raw.read(4 * n_ctx * n_chunk), dtype=np.int32)
off = raw.tell()
if mode == "ids":
    open(sys.argv[3], "w").write(" ".join(map(str, ids)))
    print(f"wrote {len(ids)} ids (n_ctx={n_ctx} chunks={n_chunk} vocab={n_vocab})")
    sys.exit(0)

prefix = sys.argv[3]
ctx, chunks, vocab, first = (int(x) for x in open(prefix + ".meta.txt").read().split())
assert (ctx, chunks, vocab) == (n_ctx, n_chunk, n_vocab), ((ctx, chunks, vocab), (n_ctx, n_chunk, n_vocab))
nrec = n_ctx - 1 - first
nv = 2 * ((n_vocab + 1) // 2) + 4
rec = np.memmap(kld, dtype=np.uint16, mode="r", offset=off).reshape(-1, nv)
L = np.memmap(prefix + ".logits.f32", dtype=np.float32, mode="r").reshape(-1, vocab)
assert len(rec) == len(L) == nrec * n_chunk, (len(rec), len(L), nrec * n_chunk)


def ll_logprobs(r):
    h = r[:4].view(np.float32)
    scale, mn = (h[0], h[1]) if h[0] > 0 else (h[1], h[0])
    return mn + scale * r[4:4 + n_vocab].astype(np.float64)


def lsm(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


kls, top1, nll_l, nll_o, rank_flip = [], [], [], [], 0
for i in range(len(L)):
    c, j = divmod(i, nrec)
    pos = first + j
    tgt = ids[c * n_ctx + pos + 1]
    ours = lsm(L[i])
    ll = ll_logprobs(rec[i])
    # llama.cpp CLAMPS every stored log-prob at -16 (min of the quantiser range): the clipped tail carries a fake
    # ~1.5% of mass (131072 * e^-16), so do NOT renormalise and restrict the sum to the unclipped support.
    sup = ll > -15.0
    kls.append(float((np.exp(ll[sup]) * (ll[sup] - ours[sup])).sum()))
    top1.append(float(np.argmax(ours) == np.argmax(ll)))
    nll_l.append(-ll[tgt])
    nll_o.append(-ours[tgt])
kls, top1 = np.array(kls), np.array(top1)
print(f"positions={len(kls)}  meanKL(llama||ours)={kls.mean():.5f}  medKL={np.median(kls):.5f}  p95KL={np.percentile(kls,95):.5f}  maxKL={kls.max():.4f}")
print(f"top1 agreement={top1.mean():.4f} ({int(top1.sum())}/{len(top1)})  PPL llama={np.exp(np.mean(nll_l)):.4f} ours={np.exp(np.mean(nll_o)):.4f} "
      f"ratio={np.exp(np.mean(nll_o) - np.mean(nll_l)):.5f}")
