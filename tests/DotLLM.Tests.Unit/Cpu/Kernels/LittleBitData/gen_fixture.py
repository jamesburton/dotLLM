"""Offline fixture generator for LittleBitTests (issue #832). Written from the paper's equations
(arXiv 2506.13771, W ~= diag(h) Us diag(l) Vs^T diag(g)); no third-party LittleBit code is used or run.
Usage: python -I gen_fixture.py > littlebit_fixture.json   (needs numpy)."""
import json
import numpy as np

rng = np.random.default_rng(832)

def path(d_out, d_in, r):
    us = rng.choice([-1, 1], size=(d_out, r)).astype(np.int8)
    vs = rng.choice([-1, 1], size=(d_in, r)).astype(np.int8)
    # FP16-representable scales (the model stores h, g, l as FP16)
    h = rng.uniform(0.2, 1.5, d_out).astype(np.float16)
    g = rng.uniform(0.2, 1.5, d_in).astype(np.float16)
    l = rng.uniform(0.2, 1.5, r).astype(np.float16)
    return us, vs, h, g, l

def layer_apply(paths, x):
    """y = sum_p h .* (Us (l .* (Vs^T (g .* x)))), in float64."""
    y = 0
    for us, vs, h, g, l in paths:
        t = l.astype(np.float64) * (vs.T.astype(np.float64) @ (g.astype(np.float64) * x))
        y = y + h.astype(np.float64) * (us.astype(np.float64) @ t)
    return y

cases = []
# d_out != d_in, r not a multiple of 8/32, several activation vectors; second case has a residual path
for (d_out, d_in, r, npaths, nx) in [(5, 19, 11, 1, 2), (24, 40, 9, 2, 2), (9, 35, 13, 2, 1)]:
    paths = [path(d_out, d_in, r) for _ in range(npaths)]
    xs = [np.round(rng.normal(size=d_in), 3).astype(np.float32) for _ in range(nx)]
    cases.append({
        "d_out": d_out, "d_in": d_in, "r": r,
        "paths": [{"us": us.flatten().tolist(), "vs": vs.flatten().tolist(),  # vs is [d_in x r] row-major
                   "h": h.astype(np.float32).tolist(), "g": g.astype(np.float32).tolist(),
                   "l": l.astype(np.float32).tolist()} for us, vs, h, g, l in paths],
        "x": [[round(float(v), 3) for v in x] for x in xs],
        "y": [layer_apply(paths, x.astype(np.float64)).tolist() for x in xs],
    })
print(json.dumps({"cases": cases}, separators=(",", ":")))
