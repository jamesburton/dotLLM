"""Pure-Python reimplementation of the dotLLM Mamba-3 kernel algorithms.

Runs against the captured fixture.json (produced by capture_fixtures.py) with no
torch dependency. A pass here means the kernel math, as we implemented it in C#,
is equivalent to the VikramKarLex reference. A remaining C# test failure after
this would point at a translation bug, not an algorithmic one.

Run:
    .mamba3-python-venv\\Scripts\\python.exe tests\\DotLLM.Tests.Integration\\Fixtures\\Mamba3\\verify_algorithm.py
"""

import json
import math
from pathlib import Path

ABS_TOL = 1e-5
REL_TOL = 1e-4

FIXTURE_PATH = Path(__file__).parent / "fixture.json"


def load_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def close(a, b, label):
    """Element-wise compare two flat lists; report first mismatch."""
    if len(a) != len(b):
        return f"FAIL {label}: length {len(a)} != {len(b)}"
    max_abs = 0.0
    max_rel = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(x - y)
        r = d / (abs(x) + 1e-12)
        max_abs = max(max_abs, d)
        max_rel = max(max_rel, r)
        if d > ABS_TOL and r > REL_TOL:
            return f"FAIL {label}[{i}]: expected={x:.8f} got={y:.8f} absDiff={d:.3e} relDiff={r:.3e}"
    return f"PASS {label}  (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"


def idx(flat, shape, *indices):
    """Row-major flat[idx]. len(indices) must equal len(shape)."""
    assert len(indices) == len(shape), (len(indices), shape)
    i = 0
    stride = 1
    for s, ix in zip(reversed(shape), reversed(indices)):
        i += ix * stride
        stride *= s
    return flat[i]


# ─── Kernels (match C# line-for-line) ─────────────────────────────────────────

def mamba3_discretize(dt, a, lam, seqlen, nheads):
    """Mirrors Mamba3Discretize.Execute: α=exp(dt·A), β=(dt−γ)·α, γ=λ·dt."""
    alpha = [0.0] * (seqlen * nheads)
    beta = [0.0] * (seqlen * nheads)
    gamma = [0.0] * (seqlen * nheads)
    for t in range(seqlen):
        for h in range(nheads):
            ix = t * nheads + h
            dA = dt[ix] * a[h]
            alpha[ix] = math.exp(dA)
            gamma[ix] = lam[ix] * dt[ix]
            beta[ix] = (dt[ix] - gamma[ix]) * alpha[ix]   # == (1-λ)·dt·α
    return alpha, beta, gamma


def rms_norm(x, weight, eps, n):
    """One-slice RMSNorm on a flat list of length n."""
    sq = sum(v * v for v in x) / n
    scale = 1.0 / math.sqrt(sq + eps)
    return [v * scale * weight[j] for j, v in enumerate(x)]


def mamba3_qknorm(bc, weight, eps, seqlen, ngroup, dstate):
    """Mirrors Mamba3QkNorm.Execute: RMSNorm on each [dstate] slice."""
    out = []
    for i in range(seqlen * ngroup):
        slice_ = bc[i * dstate:(i + 1) * dstate]
        out.extend(rms_norm(slice_, weight, eps, dstate))
    return out


def mamba3_selective_scan(x, alpha, beta, gamma, b, c, nhead, headdim, dstate, ngroup, seqlen):
    """Mirrors Mamba3SelectiveScan.Execute: trapezoidal recurrence with prev_Bx memory."""
    dinner = nhead * headdim
    heads_per_group = nhead // ngroup
    state_stride_head = headdim * dstate

    state = [0.0] * (nhead * headdim * dstate)
    prev_bx = [0.0] * (nhead * headdim * dstate)
    y = [0.0] * (seqlen * dinner)

    for t in range(seqlen):
        for h in range(nhead):
            a_th = alpha[t * nhead + h]
            b_th = beta[t * nhead + h]
            g_th = gamma[t * nhead + h]
            g = h // heads_per_group
            b_row_start = (t * ngroup + g) * dstate
            c_row_start = (t * ngroup + g) * dstate

            for p in range(headdim):
                x_thp = x[t * dinner + h * headdim + p]
                state_off = h * state_stride_head + p * dstate
                sumf = 0.0
                for k in range(dstate):
                    cur_bx = b[b_row_start + k] * x_thp
                    s = state[state_off + k] * a_th + prev_bx[state_off + k] * b_th + cur_bx * g_th
                    state[state_off + k] = s
                    prev_bx[state_off + k] = cur_bx
                    sumf += s * c[c_row_start + k]
                y[t * dinner + h * headdim + p] = sumf

    return y, state, prev_bx


# ─── Run comparators ──────────────────────────────────────────────────────────

def main():
    f = load_fixture()
    cfg = f["config"]
    seqlen = cfg["seqlen"]
    nhead = cfg["nheads"]
    headdim = cfg["headdim"]
    dstate = cfg["d_state"]

    print(f"Config: seqlen={seqlen} nhead={nhead} headdim={headdim} dstate={dstate}")

    # 1. Discretize
    dt = f["activated"]["dt"]["data"]
    lam = f["activated"]["lam"]["data"]
    a = f["inputs"]["A"]["data"]
    alpha, beta, gamma = mamba3_discretize(dt, a, lam, seqlen, nhead)
    print(close(f["expected"]["alpha"]["data"], alpha, "alpha"))
    print(close(f["expected"]["beta"]["data"], beta, "beta"))
    print(close(f["expected"]["gamma"]["data"], gamma, "gamma"))

    # 2. QkNorm (SISO: ngroup=1, bc_dim=dstate)
    b_raw = f["inputs"]["B_raw"]["data"]
    c_raw = f["inputs"]["C_raw"]["data"]
    b_w = f["inputs"]["B_norm_weight"]["data"]
    c_w = f["inputs"]["C_norm_weight"]["data"]
    b_qkn = mamba3_qknorm(b_raw, b_w, 1e-5, seqlen, 1, dstate)
    c_qkn = mamba3_qknorm(c_raw, c_w, 1e-5, seqlen, 1, dstate)
    print(close(f["expected"]["B_qkn"]["data"], b_qkn, "B_qkn"))
    print(close(f["expected"]["C_qkn"]["data"], c_qkn, "C_qkn"))

    # 3. SelectiveScan: ngroup = nhead (post-BC-bias broadcast)
    x = f["inputs"]["x"]["data"]
    b_roped = f["expected"]["B_roped"]["data"]
    c_roped = f["expected"]["C_roped"]["data"]

    # Reference alpha/beta/gamma to feed in (bypass our Discretize for this kernel's check)
    y, state, prev_bx = mamba3_selective_scan(
        x,
        f["expected"]["alpha"]["data"],
        f["expected"]["beta"]["data"],
        f["expected"]["gamma"]["data"],
        b_roped, c_roped,
        nhead, headdim, dstate, ngroup=nhead, seqlen=seqlen,
    )
    print(close(f["expected"]["y_scan"]["data"], y, "y_scan"))
    print(close(f["expected"]["ssm_state"]["data"], state, "ssm_state"))
    print(close(f["expected"]["last_Bx"]["data"], prev_bx, "prev_Bx"))


if __name__ == "__main__":
    main()
