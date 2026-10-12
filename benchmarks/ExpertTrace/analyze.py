"""Analyse ExpertTrace .bin routing traces (#938, parent #825).

usage: python -I analyze.py name=path.bin [name=path.bin ...]

Prints, per workload and pooled: decode-phase hit rates of per-layer LRU and LFU-with-decay caches (capacity 64/128/256 of 512 experts),
static-profile transfer (profile built on the OTHER workloads, evaluated on this one; plus the same-workload oracle), distinct experts per
1..5-row verify window, per-chunk prefill coverage, and the implied SSD traffic per decoded token.
"""
import sys, struct, collections
import numpy as np

EXP_MIB = 2.93          # qwen4exp UD-Q4_K_XL gate+up+down per expert (0.879+0.879+1.172 MiB)
NUM_EXPERTS = 512


def load(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'RTR1'
        layers, k, n, npre = struct.unpack('<4i', f.read(16))
        a = np.frombuffer(f.read(), dtype='<u2').reshape(n, layers, k).astype(np.int32)
    return a, npre


def lru_hits(seq, cap, warm):
    """seq: [N, K] expert ids of one layer. Returns (hits, lookups, misses_per_token list) counted on rows >= warm."""
    cache = collections.OrderedDict()
    hits = look = 0
    miss_tok = []
    for i in range(seq.shape[0]):
        row = set(seq[i].tolist())
        h = sum(1 for e in row if e in cache)
        if i >= warm:
            hits += h; look += len(row); miss_tok.append(len(row) - h)
        for e in row:
            if e in cache: cache.move_to_end(e)
            else: cache[e] = 1
        while len(cache) > cap:
            for victim in cache:                      # oldest first, never one of this token's own experts
                if victim not in row:
                    del cache[victim]; break
            else:
                break
    return hits, look, miss_tok


def lfu_hits(seq, cap, warm, decay=0.995):
    cnt = np.zeros(NUM_EXPERTS, dtype=np.float64)
    cache = set()
    hits = look = 0
    for i in range(seq.shape[0]):
        row = set(seq[i].tolist())
        cnt *= decay
        h = sum(1 for e in row if e in cache)
        if i >= warm:
            hits += h; look += len(row)
        for e in row: cnt[e] += 1.0
        cache |= row
        while len(cache) > cap:
            victim = min((e for e in cache if e not in row), key=lambda e: cnt[e], default=None)
            if victim is None: break
            cache.discard(victim)
    return hits, look


def static_hits(seq, resident, warm):
    hits = look = 0
    for i in range(warm, seq.shape[0]):
        row = set(seq[i].tolist()); hits += sum(1 for e in row if e in resident); look += len(row)
    return hits, look


def profile(traces, layer, cap):
    c = np.zeros(NUM_EXPERTS)
    for a, _ in traces:
        np.add.at(c, a[:, layer, :].ravel(), 1)
    return set(np.argsort(-c)[:cap].tolist())


def pct(h, l): return 100.0 * h / max(l, 1)


def main():
    items = [x.split('=', 1) for x in sys.argv[1:]]
    tr = {n: load(p) for n, p in items}
    caps = (64, 128, 256)
    print(f"traces: " + ", ".join(f"{n}: {a.shape[0]} tok ({npre} prefill, {a.shape[0]-npre} decode) x {a.shape[1]} layers x top-{a.shape[2]}" for n, (a, npre) in tr.items()))

    # ---- LRU / LFU, per workload and pooled ----
    for mode in ('warm (cache filled by the prompt prefill)', 'cold (empty at first decode token)'):
        print(f"\n== LRU / LFU-decay hit rate on DECODE tokens, {mode}")
        print(f"{'workload':<14}{'cap':>5}{'%exp':>6} {'LRU all':>8} {'LRU minL':>9} {'LRU maxL':>9} {'layers>60%':>11} {'LFU':>7} {'miss/tok/layer':>15}")
        pooled = {c: [0, 0, 0, 0, 0, 0] for c in caps}
        for n, (a, npre) in tr.items():
            L = a.shape[1]
            for cap in caps:
                lh = ll = 0; per = []; fh = fl = 0; miss = []
                for l in range(L):
                    seq = a[:, l, :]
                    if mode.startswith('warm'):
                        h, k, mt = lru_hits(seq, cap, npre)
                        h2, k2 = lfu_hits(seq, cap, npre)
                    else:
                        h, k, mt = lru_hits(seq[npre:], cap, 0)
                        h2, k2 = lfu_hits(seq[npre:], cap, 0)
                    lh += h; ll += k; per.append(pct(h, k)); fh += h2; fl += k2; miss.append(np.mean(mt) if mt else 0)
                per = np.array(per)
                print(f"{n:<14}{cap:>5}{100*cap/NUM_EXPERTS:>5.1f}% {pct(lh,ll):>7.1f}% {per.min():>8.1f}% {per.max():>8.1f}% {int((per>60).sum()):>6}/{L:<4} {pct(fh,fl):>6.1f}% {np.mean(miss):>14.2f}")
                p = pooled[cap]; p[0] += lh; p[1] += ll; p[2] += fh; p[3] += fl
        for cap in caps:
            p = pooled[cap]
            print(f"{'POOLED':<14}{cap:>5}{100*cap/NUM_EXPERTS:>5.1f}% LRU {pct(p[0],p[1]):.1f}%  LFU {pct(p[2],p[3]):.1f}%")

    # ---- static-profile transfer ----
    print("\n== Static resident set per layer (top-N by frequency), evaluated on DECODE tokens")
    print(f"{'test on':<14}{'cap':>5} {'self-oracle':>12} {'profile from others':>20}")
    for n, (a, npre) in tr.items():
        others = [v for k2, v in tr.items() if k2 != n]
        L = a.shape[1]
        for cap in (64, 128):
            so = [0, 0]; ot = [0, 0]
            for l in range(L):
                h, k = static_hits(a[:, l, :], profile([(a[npre:], 0)], l, cap), npre); so[0] += h; so[1] += k
                if others:
                    h, k = static_hits(a[:, l, :], profile(others, l, cap), npre); ot[0] += h; ot[1] += k
            print(f"{n:<14}{cap:>5} {pct(*so):>11.1f}% {pct(*ot) if others else float('nan'):>19.1f}%")

    # ---- verify window ----
    print("\n== Distinct experts per layer touched by a window of W consecutive decode rows (max W*10)")
    print(f"{'workload':<14}" + "".join(f"{'W='+str(w):>16}" for w in (1, 2, 3, 4, 5)))
    for n, (a, npre) in tr.items():
        d = a[npre:]
        row = f"{n:<14}"
        for w in (1, 2, 3, 4, 5):
            if d.shape[0] < w: row += f"{'-':>16}"; continue
            vals = []
            for l in range(d.shape[1]):
                s = d[:, l, :]
                for i in range(0, s.shape[0] - w + 1, max(1, w)):
                    vals.append(len(set(s[i:i + w].ravel().tolist())))
            v = np.array(vals); row += f"  {v.mean():5.1f} (p95 {np.percentile(v,95):4.0f})"
        print(row)
    print("window-batched LRU-64 misses per layer per window (cache updated once per window, warm):")
    for n, (a, npre) in tr.items():
        res = []
        for w in (1, 5):
            tot = 0; cnt = 0
            for l in range(a.shape[1]):
                s = a[:, l, :]; cache = collections.OrderedDict()
                for i in range(0, npre, 1):
                    for e in s[i].tolist():
                        cache[e] = 1; cache.move_to_end(e)
                        while len(cache) > 64: cache.popitem(last=False)
                for i in range(npre, s.shape[0] - w + 1, w):
                    u = set(s[i:i + w].ravel().tolist()); tot += sum(1 for e in u if e not in cache); cnt += 1
                    for e in u:
                        cache[e] = 1; cache.move_to_end(e)
                    while len(cache) > max(64, len(u)):
                        cache.popitem(last=False)
            res.append(tot / max(cnt, 1))
        print(f"  {n:<14} W=1: {res[0]:.2f} misses/layer/row-window   W=5: {res[1]:.2f} misses/layer/window ({res[1]/5:.2f}/row)")

    # ---- prefill coverage ----
    print("\n== Prefill: distinct experts per layer touched by a chunk of C prompt tokens (of 512)")
    for n, (a, npre) in tr.items():
        p = a[:npre]
        out = []
        for c in (16, 64, 256, 1024):
            if p.shape[0] < c: continue
            vals = [len(set(p[i:i + c, l, :].ravel().tolist())) for l in range(p.shape[1]) for i in range(0, p.shape[0] - c + 1, c)]
            out.append(f"C={c}: {np.mean(vals):.0f}")
        print(f"  {n:<14}" + "  ".join(out))

    # ---- traffic ----
    print("\n== Implied SSD traffic per decoded token (all layers, qwen4exp 2.93 MiB/expert)")
    for n, (a, npre) in tr.items():
        L = a.shape[1]
        for cap in (64, 128):
            m = 0.0
            for l in range(L):
                _, _, mt = lru_hits(a[:, l, :], cap, npre); m += np.mean(mt) if mt else 0
            mib = m * EXP_MIB
            print(f"  {n:<14} LRU-{cap}: {m:.1f} misses/token = {mib:.0f} MiB/token -> {1.0e3*1.0/ (mib*1.048576/1.5):.1f} tok/s ceiling @1.5 GB/s random-read (no overlap)")


if __name__ == '__main__':
    main()
