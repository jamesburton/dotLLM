"""Plain-assert tests for pdiff Gate P1 (weight-block autoencoder).

Run:
    C:\\Python311\\python.exe scripts\\diffusion\\pdiff_test.py

No pytest required.  Tests cover:
  1. AE encoder/decoder output shapes
  2. Normalization round-trip accuracy (including realistic 8-sigma outliers)
  3. Tanh squash keeps all values strictly inside (-1, 1)
  4. Synthetic AE reconstruction smoke (tiny dim, small batch -> cos > 0.99)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from pdiff_ae import (
    WeightBlockAE,
    WeightBlockLinearAE,
    normalize_block,
    denormalize_block,
    normalize_population,
    denormalize_population,
    normalize_global,
    apply_normalize_global,
    denormalize_global,
    SQUASH_C,
)


# ---------------------------------------------------------------------------
# Test 1: AE shape correctness
# ---------------------------------------------------------------------------

def test_ae_shapes():
    """Encoder, decoder, and forward output shapes are correct."""
    dim, latent, hidden = 128, 16, 32
    ae  = WeightBlockAE(dim=dim, latent=latent, hidden=hidden)

    x   = torch.randn(6, dim)
    z   = ae.encode(x)
    xr  = ae.decode(z)
    xr2 = ae(x)

    assert z.shape   == (6, latent), f"encoder shape {z.shape} != (6, {latent})"
    assert xr.shape  == (6, dim),    f"decoder shape {xr.shape} != (6, {dim})"
    assert xr2.shape == (6, dim),    f"forward shape {xr2.shape} != (6, {dim})"

    # n_params sanity: should be > 0
    np = ae.n_params()
    assert np > 0, f"n_params={np}"

    print(f"[PASS] test_ae_shapes  (n_params={np:,})")


# ---------------------------------------------------------------------------
# Test 2: Normalization round-trip
# ---------------------------------------------------------------------------

def test_norm_roundtrip_typical():
    """normalize_block -> denormalize_block is near-exact for typical weight values."""
    torch.manual_seed(3)
    # Simulate a typical BitNet weight block: small values, unit-ish std
    x = torch.randn(40960) * 0.4   # typical weight magnitude

    s, meta = normalize_block(x)
    xr      = denormalize_block(s, meta)

    max_err = float((xr - x).abs().max())
    rel_err = float((xr - x).abs().max() / (x.abs().max() + 1e-8))
    assert max_err < 1e-3, f"round-trip max_err={max_err:.2e} > 1e-3 (typical block)"
    print(f"[PASS] test_norm_roundtrip_typical  (max_err={max_err:.2e}  rel={rel_err:.2e})")


def test_norm_roundtrip_with_outliers():
    """Round-trip stays accurate for realistic 8-sigma outliers (not extreme synthetic ones).

    BitNet raw weights span roughly ±26 with per-block std ~2-4.  After per-block
    z-score, outliers reach ~8-9σ → tanh(9/4)=tanh(2.25)≈0.978 — not saturated →
    atanh recovers the value exactly.
    """
    torch.manual_seed(5)
    x = torch.randn(40960) * 3.0     # broad distribution, std≈3
    # Inject realistic 8σ outliers (8 × local std ≈ 8 × 3 = 24)
    sigma_x = float(x.std())
    x[0]    =  8.0 * sigma_x
    x[1]    = -8.0 * sigma_x

    s, meta = normalize_block(x)

    # After squash, outliers should be well inside ±1 (not saturated)
    z_outlier = 8.0 * sigma_x / (meta["sigma"])   # per-block z-score of outlier
    s_outlier = float(torch.tanh(torch.tensor(z_outlier / SQUASH_C)))
    assert abs(s_outlier) < 0.999, (
        f"outlier saturated: z={z_outlier:.2f}  s={s_outlier:.4f}.  "
        f"Increase SQUASH_C if BitNet outliers exceed this."
    )

    xr      = denormalize_block(s, meta)
    max_err = float((xr - x).abs().max())
    assert max_err < 0.5, (       # loose tolerance: outlier may have small float32 error
        f"round-trip max_err={max_err:.4f} > 0.5 with 8σ outliers"
    )
    print(
        f"[PASS] test_norm_roundtrip_with_outliers  "
        f"(z_outlier={z_outlier:.2f}  s_outlier={s_outlier:.4f}  max_err={max_err:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 3: Squash range
# ---------------------------------------------------------------------------

def test_norm_squashes_all_values():
    """tanh-squash keeps ALL values strictly inside (-1, 1)."""
    torch.manual_seed(9)
    x = torch.randn(40960) * 5.0   # heavy-tailed
    x[0] = 50.0; x[1] = -50.0     # extreme synthetic outlier to test clamp path

    s, _ = normalize_block(x)
    max_abs = float(s.abs().max())
    assert max_abs < 1.0, f"squash output has |value| >= 1: max_abs={max_abs}"
    print(f"[PASS] test_norm_squashes_all_values  (max_abs={max_abs:.6f})")


# ---------------------------------------------------------------------------
# Test 4: Population normalize / denormalize inverse
# ---------------------------------------------------------------------------

def test_population_normalize_roundtrip():
    """normalize_population -> denormalize_population is consistent row-wise."""
    torch.manual_seed(11)
    raw  = torch.randn(8, 512) * 2.0   # 8 blocks, 512-D (small for speed)

    normed, metas = normalize_population(raw)
    assert normed.shape == raw.shape, f"normed shape {normed.shape} != {raw.shape}"

    rec = denormalize_population(normed, metas)
    max_err = float((rec - raw).abs().max())
    assert max_err < 1e-3, f"population round-trip max_err={max_err:.2e}"
    print(f"[PASS] test_population_normalize_roundtrip  (max_err={max_err:.2e})")


# ---------------------------------------------------------------------------
# Test 5: Synthetic AE reconstruction smoke
# ---------------------------------------------------------------------------

def test_ae_synthetic_reconstruction():
    """Tiny AE (dim=64, latent=8) overfits to k=4 synthetic blocks: cos > 0.99 in raw space.

    Uses heavy-tailed synthetic data (to exercise the full norm → AE → denorm pipeline).
    """
    torch.manual_seed(17)
    dim, latent, hidden = 64, 8, 32
    k = 4

    # Synthetic heavy-tailed data resembling real weight blocks
    raw        = torch.randn(k, dim) * 2.5
    raw[0, 0]  = float(raw[0].std()) * 7.0   # realistic 7σ outlier in block 0

    normed, metas = normalize_population(raw)

    ae  = WeightBlockAE(dim=dim, latent=latent, hidden=hidden)
    opt = torch.optim.Adam(ae.parameters(), lr=5e-3)
    ae.train()
    for _ in range(3000):
        out  = ae(normed)
        loss = F.mse_loss(out, normed)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ae.eval()
    with torch.no_grad():
        recon_normed = ae(normed)
        recon_raw    = denormalize_population(recon_normed, metas)

    cosines = [
        float(F.cosine_similarity(recon_raw[i].unsqueeze(0), raw[i].unsqueeze(0)))
        for i in range(k)
    ]
    mean_cos = sum(cosines) / len(cosines)

    assert mean_cos > 0.99, (
        f"Synthetic AE mean_cos={mean_cos:.4f} < 0.99  (per-block: {cosines})"
    )
    print(
        f"[PASS] test_ae_synthetic_reconstruction  "
        f"(mean_cos={mean_cos:.4f}  min={min(cosines):.4f})"
    )


# ---------------------------------------------------------------------------
# Test 6: AE param count sanity for real-scale weights
# ---------------------------------------------------------------------------

def test_ae_real_scale_params():
    """WeightBlockAE at real scale (dim=40960, latent=128, hidden=256) has expected param count."""
    ae  = WeightBlockAE(dim=40960, latent=128, hidden=256)
    np_ = ae.n_params()
    # Enc: 40960*256 + 256 + 256*128 + 128 = 10,551,680
    # Dec: 128*256 + 256 + 256*40960 + 40960 = 10,551,680
    # Total ~21 M
    assert 20_000_000 < np_ < 25_000_000, f"param count {np_:,} out of expected range"
    print(f"[PASS] test_ae_real_scale_params  (n_params={np_:,})")


# ---------------------------------------------------------------------------
# Test 7: Linear AE shapes and param count
# ---------------------------------------------------------------------------

def test_linear_ae_shapes_and_params():
    """WeightBlockLinearAE shapes and parameter count are correct."""
    dim, latent = 128, 16
    ae  = WeightBlockLinearAE(dim=dim, latent=latent)
    x   = torch.randn(6, dim)
    z   = ae.encode(x)
    xr  = ae.decode(z)
    xr2 = ae(x)

    assert z.shape   == (6, latent), f"encoder shape {z.shape}"
    assert xr.shape  == (6, dim),    f"decoder shape {xr.shape}"
    assert xr2.shape == (6, dim),    f"forward shape {xr2.shape}"

    np_ = ae.n_params()
    assert np_ == 2 * dim * latent, f"expected {2*dim*latent} params, got {np_}"

    # Real scale: dim=40960, latent=64 -> 2*40960*64 = 5,242,880
    ae_big = WeightBlockLinearAE(dim=40960, latent=64)
    assert ae_big.n_params() == 5_242_880, f"real-scale params {ae_big.n_params()}"

    print(f"[PASS] test_linear_ae_shapes_and_params  (n_params={np_}  real_scale={ae_big.n_params():,})")


# ---------------------------------------------------------------------------
# Test 8: Linear AE generalizes — overfits training but also reconstructs held-out
# ---------------------------------------------------------------------------

def test_linear_ae_generalization():
    """Linear AE trained on k=8 synthetic blocks generalizes to 2 held-out blocks.

    The key property: unlike MLP AE, the linear AE finds a shared subspace
    so held-out data lying in that subspace is well-reconstructed (cos > 0.90).
    Uses dim=128 to keep the test fast.
    """
    torch.manual_seed(23)
    dim, latent, k_train = 128, 8, 8

    # Synthetic data: draw from a random 8-D subspace so ALL blocks share that structure
    basis      = torch.randn(dim, latent)                   # [128, 8] random basis
    coords_all = torch.randn(10, latent)                    # [10, 8] coordinates
    raw_all    = coords_all @ basis.T                       # [10, 128] all blocks in that subspace

    raw_train = raw_all[:k_train]                           # [8, 128]
    raw_held  = raw_all[k_train:]                           # [2, 128]

    normed_train, metas_train = normalize_population(raw_train)
    normed_held,  metas_held  = normalize_population(raw_held)

    ae  = WeightBlockLinearAE(dim=dim, latent=latent)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    ae.train()
    for _ in range(3000):
        out  = ae(normed_train)
        loss = F.mse_loss(out, normed_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ae.eval()
    with torch.no_grad():
        # Training reconstruction
        train_recon_n  = ae(normed_train)
        train_recon_r  = denormalize_population(train_recon_n, metas_train)
        train_cosines  = [
            float(F.cosine_similarity(train_recon_r[i].unsqueeze(0), raw_train[i].unsqueeze(0)))
            for i in range(k_train)
        ]
        # Held-out reconstruction
        held_recon_n   = ae(normed_held)
        held_recon_r   = denormalize_population(held_recon_n, metas_held)
        held_cosines   = [
            float(F.cosine_similarity(held_recon_r[i].unsqueeze(0), raw_held[i].unsqueeze(0)))
            for i in range(len(raw_held))
        ]

    mean_train_cos = sum(train_cosines) / len(train_cosines)
    mean_held_cos  = sum(held_cosines)  / len(held_cosines)

    assert mean_train_cos > 0.99, f"train cos {mean_train_cos:.4f} < 0.99"
    assert mean_held_cos  > 0.90, (
        f"linear AE held-out cos {mean_held_cos:.4f} < 0.90 "
        f"(MLP AE would get ~0 here due to overfitting)"
    )
    print(
        f"[PASS] test_linear_ae_generalization  "
        f"(train cos={mean_train_cos:.4f}  held cos={mean_held_cos:.4f})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 9: Global normalization round-trip and properties
# ---------------------------------------------------------------------------

def test_global_norm_roundtrip():
    """normalize_global -> denormalize_global round-trip is exact."""
    torch.manual_seed(31)
    raw  = torch.randn(8, 512) * 2.5
    raw[0, 0] = 25.0  # heavy-tailed outlier

    normed, meta = normalize_global(raw)
    rec = denormalize_global(normed, meta)

    max_err = float((rec - raw).abs().max())
    assert max_err < 1e-4, f"global norm round-trip max_err={max_err:.2e}"
    assert meta["mode"] == "global"
    print(f"[PASS] test_global_norm_roundtrip  (max_err={max_err:.2e})")


def test_global_norm_consistent_with_apply():
    """apply_normalize_global with training meta gives same result as normalize_global on same data."""
    torch.manual_seed(37)
    raw_train = torch.randn(10, 64)
    raw_new   = torch.randn(5, 64) * 1.5

    normed_train, meta = normalize_global(raw_train)
    applied             = apply_normalize_global(raw_train, meta)

    assert torch.allclose(normed_train, applied, atol=1e-6), "inconsistent normalization"

    # apply to new data should use training stats, not new-data stats
    applied_new     = apply_normalize_global(raw_new, meta)
    expected_new    = (raw_new - meta["mu"]) / meta["sigma"]
    assert torch.allclose(applied_new, expected_new, atol=1e-6)
    print("[PASS] test_global_norm_consistent_with_apply")


def test_linear_ae_global_norm_generalization():
    """Linear AE + global normalization generalizes to held-out: cos > 0.97.

    This is the key test that validates the Gate P1 design choice.
    Uses realistic-looking synthetic data (NOT from a single low-dim subspace,
    but correlated — as BitNet blocks are).
    """
    torch.manual_seed(41)
    dim, latent = 512, 16
    k_train, k_held = 20, 5

    # Synthetic: correlated blocks (shared random matrix + per-block noise)
    shared = torch.randn(dim, latent)                     # shared subspace
    coords = torch.randn(k_train + k_held, latent)        # coordinates
    raw_all = coords @ shared.T + 0.05 * torch.randn(k_train + k_held, dim)

    raw_train = raw_all[:k_train]
    raw_held  = raw_all[k_train:]

    # Global normalization from training set
    normed_train, meta = normalize_global(raw_train)
    normed_held        = apply_normalize_global(raw_held, meta)

    ae  = WeightBlockLinearAE(dim=dim, latent=latent)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    ae.train()
    for _ in range(2000):
        out  = ae(normed_train)
        loss = F.mse_loss(out, normed_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ae.eval()
    with torch.no_grad():
        held_recon_n = ae(normed_held)
        held_recon_r = denormalize_global(held_recon_n, meta)

    held_cosines = [
        float(F.cosine_similarity(held_recon_r[i].unsqueeze(0), raw_held[i].unsqueeze(0)))
        for i in range(k_held)
    ]
    mean_cos = sum(held_cosines) / len(held_cosines)

    assert mean_cos > 0.97, (
        f"Linear AE + global norm held cos={mean_cos:.4f} < 0.97\n"
        f"Per-block normalization would give ~0 here."
    )
    print(
        f"[PASS] test_linear_ae_global_norm_generalization  "
        f"(held mean_cos={mean_cos:.4f})"
    )


if __name__ == "__main__":
    print("Running pdiff Gate P1 tests ...")
    print()
    test_ae_shapes()
    test_norm_roundtrip_typical()
    test_norm_roundtrip_with_outliers()
    test_norm_squashes_all_values()
    test_population_normalize_roundtrip()
    test_ae_synthetic_reconstruction()
    test_ae_real_scale_params()
    test_linear_ae_shapes_and_params()
    test_linear_ae_generalization()
    test_global_norm_roundtrip()
    test_global_norm_consistent_with_apply()
    test_linear_ae_global_norm_generalization()
    print()
    print("ALL pdiff_test TESTS PASSED")
