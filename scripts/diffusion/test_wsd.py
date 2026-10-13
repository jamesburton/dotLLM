"""Plain-assert tests for weight-space diffusion Exp 0. Run:
    C:\\Python311\\python.exe scripts\\diffusion\\test_wsd.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from wsd_ddpm import (
    make_schedule, q_sample, DenoiseMLP, train_ddpm, sample_ddpm,
    cosine, rel_l2, nearest_target,
)

def test_schedule():
    s = make_schedule(T=100)
    assert s.betas.shape == (100,) and s.alpha_bars.shape == (100,)
    assert torch.all(s.alpha_bars[1:] < s.alpha_bars[:-1])          # strictly decreasing
    assert float(s.alpha_bars.min()) > 0.0 and float(s.alpha_bars[0]) <= 1.0
    print("[pass] test_schedule")

def test_q_sample_identity_at_t0():
    s = make_schedule(T=100)
    x0 = torch.randn(2, 8)
    noise = torch.randn(2, 8)
    t0 = torch.zeros(2, dtype=torch.long)
    xt = q_sample(x0, t0, noise, s)
    # at t=0 alpha_bar≈alphas[0]=1-1e-4, so xt ≈ x0 (not exact); assert dominated by x0
    assert cosine(xt[0], x0[0]) > 0.999
    print("[pass] test_q_sample_identity_at_t0")

def test_metrics():
    v = torch.randn(40)
    assert abs(cosine(v, v) - 1.0) < 1e-5
    assert rel_l2(v, v) < 1e-6
    tgts = torch.stack([v, torch.randn(40), torch.randn(40)])
    idx, cos, _ = nearest_target(v, tgts)
    assert idx == 0 and cos > 0.999
    print("[pass] test_metrics")

def test_sample_shape_and_determinism():
    s = make_schedule(T=20)
    m = DenoiseMLP(dim=8, hidden=32)
    a = sample_ddpm(m, s, n=3, dim=8, seed=1)
    b = sample_ddpm(m, s, n=3, dim=8, seed=1)
    c = sample_ddpm(m, s, n=3, dim=8, seed=2)
    assert a.shape == (3, 8)
    assert torch.allclose(a, b)               # same seed → identical
    assert not torch.allclose(a, c)           # different seed → different
    print("[pass] test_sample_shape_and_determinism")

def test_synthetic_end_to_end_smoke():
    # Gate-0a proxy at tiny dim: memorize ONE random vector, recover it.
    torch.manual_seed(0)
    s = make_schedule(T=100)
    dim = 64
    target = torch.randn(1, dim)
    m = DenoiseMLP(dim=dim, hidden=256)
    loss = train_ddpm(m, target, s, epochs=4000, lr=1e-3, seed=0)
    samples = sample_ddpm(m, s, n=5, dim=dim, seed=0)
    cosines = [cosine(samples[i], target[0]) for i in range(5)]
    assert min(cosines) > 0.99, f"min cosine {min(cosines):.4f}, loss {loss:.5f}"
    print(f"[pass] test_synthetic_end_to_end_smoke (min cos {min(cosines):.4f})")

from wsd_targets import extract_targets, denormalize, normalize

_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"


def test_extract_targets_shapes():
    tgts, meta = extract_targets(_BASE, k=5, seed=0)
    assert tgts.shape == (5, 40960) and tgts.dtype == torch.float32
    assert meta["dim"] == 40960 and meta["block_shape"] == [16, 2560]
    assert len(meta["locations"]) == 5 and len(set(map(tuple, meta["locations"]))) == 5  # distinct
    print("[pass] test_extract_targets_shapes")


def test_normalize_roundtrip():
    tgts, meta = extract_targets(_BASE, k=2, seed=1)
    raw = denormalize(tgts, meta)
    back = normalize(raw, meta)
    assert torch.allclose(tgts, back, atol=1e-4)
    # normalized targets should be roughly unit-scale
    assert 0.3 < float(tgts.std()) < 3.0
    print("[pass] test_normalize_roundtrip")


from wsd_exp0 import run_gate

def test_run_gate_synthetic():
    # synthetic mode: no base model; validates harness wiring + report shape + pass logic
    rep = run_gate(k=1, epochs=3000, n_samples=5, seed=0, base_id="", synthetic=True)
    assert set(rep) >= {"mean_nearest_cosine", "min_nearest_cosine", "coverage", "passed", "per_sample"}
    assert len(rep["per_sample"]) == 5
    assert rep["passed"] is True and rep["min_nearest_cosine"] > 0.99
    print("[pass] test_run_gate_synthetic")

if __name__ == "__main__":
    test_schedule()
    test_q_sample_identity_at_t0()
    test_metrics()
    test_sample_shape_and_determinism()
    test_synthetic_end_to_end_smoke()
    test_extract_targets_shapes()
    test_normalize_roundtrip()
    test_run_gate_synthetic()
    print("ALL wsd_ddpm TESTS PASSED")
