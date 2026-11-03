"""Compare dotLLM full-vocab logits (DOTLLM_PPL_LOGITS_DUMP) against llama.cpp --kl-divergence-base (kld.bin).

usage: kld_compare.py kld.bin dot_logits.bin [max_chunks]
Reports per-window and overall: top-1 agreement, mean KL(llama || dotLLM), mean |dNLL|, PPL of both on the targets.
llama.cpp stores log-probs as uint16 levels between (max_logit-16) and max_logit, so KL is exact up to that 16-nat floor.
"""
import struct, sys
import numpy as np

kld, dot = sys.argv[1], sys.argv[2]
with open(kld, 'rb') as f:
    assert f.read(8) == b'_logits_'
    n_ctx, n_vocab, n_chunk = struct.unpack('<iii', f.read(12))
    toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32).reshape(n_chunk, n_ctx)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    rows = n_ctx - 1 - first
    data = np.fromfile(f, dtype=np.uint16, count=n_chunk * rows * nv).reshape(n_chunk, rows, nv)
print(f'n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk} rows/chunk={rows}')
dl = np.memmap(dot, dtype=np.float32, mode='r')
nwin = dl.size // (n_ctx * n_vocab)
dl = dl.reshape(nwin, n_ctx, n_vocab)
nchunk = min(n_chunk, nwin, int(sys.argv[3]) if len(sys.argv) > 3 else 10**9)
tot = dict(n=0, top1=0, kl=0.0, dn=0.0, nll_l=0.0, nll_d=0.0)
for c in range(nchunk):
    hdr = data[c, :, :4].copy().view(np.float32)  # [rows, 2] scale, min_log_prob
    scale = hdr[:, 0:1]; minlp = hdr[:, 1:2]
    q = data[c, :, 4:4 + n_vocab].astype(np.float32)
    lp_l = scale * q + minlp                      # llama log-probs (floored at max-16)
    x = dl[c, first:first + rows, :].astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    lp_d = x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
    p_l = np.where(q > 0, np.exp(lp_l.astype(np.float64)), 0.0)  # level 0 = below the max-16 floor: mass unknown, ~1e-7 each
    # renormalise llama's truncated distribution
    p_l /= p_l.sum(axis=1, keepdims=True)
    kl = (p_l * (np.log(np.maximum(p_l, 1e-30)) - lp_d)).sum(axis=1)
    top_l = lp_l.argmax(axis=1); top_d = lp_d.argmax(axis=1)
    tgt = toks[c, first + 1:first + 1 + rows]
    nll_l = -lp_l[np.arange(rows), tgt]; nll_d = -lp_d[np.arange(rows), tgt]
    tot['n'] += rows; tot['top1'] += int((top_l == top_d).sum()); tot['kl'] += kl.sum()
    tot['dn'] += np.abs(nll_l - nll_d).sum(); tot['nll_l'] += nll_l.sum(); tot['nll_d'] += nll_d.sum()
    print(f'chunk {c:3d}: top1 {np.mean(top_l == top_d):.4f}  KL {kl.mean():.3e}  ppl llama {np.exp(nll_l.mean()):9.4f} dot {np.exp(nll_d.mean()):9.4f}')
n = tot['n']
print(f"OVERALL n={n}  top1={tot['top1']/n:.5f}  meanKL={tot['kl']/n:.4e}  mean|dNLL|={tot['dn']/n:.4e}  "
      f"ppl llama={np.exp(tot['nll_l']/n):.4f} dot={np.exp(tot['nll_d']/n):.4f}")
