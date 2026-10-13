"""Plain-assert tests for pdiff Gate P2 (latent diffusion pipeline).

Run:
    python scripts/diffusion/pdiff_p2_test.py

No pytest required.  Tests cover:
  1. Latent standardization round-trip (standardize -> unstandardize is exact)
  2. nearest_cosine_to_population: known inputs (identical -> 1.0, orthogonal -> 0.0)
  3. pairwise_l2_diversity: collapse (identical vectors) vs non-collapse (diverse)
  4. Synthetic manifold E2E control (Gate P2 smoke test):
       - 24 vectors on an 8-D manifold in 512-D + small Gaussian noise
       - quick WeightBlockLinearAE trained and saved to OS scratchpad
       - run pdiff_p2_run() with 1000 DDPM epochs
       - ASSERT: mean nearest cosine >= 0.90  AND  diversity > COLLAPSE_FLOOR
       - synthetic artifacts live in the scratchpad, NOT the repo

Synthetic manifold test rationale: the DDPM operates in a 16-D latent space
standardized to N(0,1)-ish scale.  The AE is a linear PCA-equivalent that
generalises perfectly to the 8-D manifold.  With the DDPM trained on only 19
latents and T=100 DDIM steps, samples converge to near-training latents.
Decoded weight vectors therefore land close to the training population ->
nearest cosine high, diversity non-zero.  This is the EXACTLY the regime Exp-0
identified as "well-behaved low-dim data".
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pdiff_ae import (
    WeightBlockLinearAE,
    normalize_global,
    train_ae,
)
from pdiff_p2_latent import (
    pdiff_p2_standardize,
    pdiff_p2_unstandardize,
    pdiff_p2_nearest_cosine,
    pdiff_p2_pairwise_l2,
    pdiff_p2_compute_metrics,
    pdiff_p2_run,
    COLLAPSE_FLOOR,
    PASS_NEAREST_COS,
)

# ---------------------------------------------------------------------------
# Scratchpad location for synthetic test artifacts (never committed to repo)
# ---------------------------------------------------------------------------

_SCRATCHPAD = (
    r"C:\Users\james\AppData\Local\Temp\claude"
    r"\E--Development-dotLLM\f5d62daa-0be0-4e86-81e4-cf8f16601e91\scratchpad"
)
_SYNTH_DIR = os.path.join(_SCRATCHPAD, "p2_synth_test")


# ---------------------------------------------------------------------------
# Test 1: Latent standardisation round-trip
# ---------------------------------------------------------------------------

def test_standardize_roundtrip():
    """pdiff_p2_standardize -> pdiff_p2_unstandardize is exact (within float32 eps)."""
    torch.manual_seed(1)
    latents = torch.randn(12, 16) * 3.0 + 5.0    # non-zero mean, non-unit std

    std_lat, mean, std = pdiff_p2_standardize(latents)

    # Standardized should have per-dimension mean ~0 and std ~1
    per_dim_mean_max = float(std_lat.mean(dim=0).abs().max())
    assert per_dim_mean_max < 1e-5, (
        f"standardized mean not ~0: max={per_dim_mean_max:.2e}"
    )
    per_dim_std_max = float(std_lat.std(dim=0).max())
    assert abs(per_dim_std_max - 1.0) < 0.1, (
        f"standardized std not ~1: max_per_dim={per_dim_std_max:.4f}"
    )

    # Round-trip
    rec     = pdiff_p2_unstandardize(std_lat, mean, std)
    max_err = float((rec - latents).abs().max())
    assert max_err < 1e-4, f"standardize round-trip max_err={max_err:.2e}"

    print(f"[PASS] test_standardize_roundtrip  (max_err={max_err:.2e})")


# ---------------------------------------------------------------------------
# Test 2: nearest_cosine_to_population metric correctness
# ---------------------------------------------------------------------------

def test_nearest_cosine_known():
    """pdiff_p2_nearest_cosine: identical -> ~1.0, orthogonal -> ~0.0."""
    # Identical vectors: nearest cosine must be ~1.0
    v       = torch.randn(4, 32)
    cos_id  = pdiff_p2_nearest_cosine(v, v)
    assert float(cos_id.min()) > 0.999, (
        f"identical vectors: min cos={float(cos_id.min()):.4f} expected >0.999"
    )

    # Orthogonal standard-basis pair: e1 vs e2 -> cos = 0
    dim = 16
    e1  = torch.zeros(1, dim); e1[0, 0] = 1.0
    e2  = torch.zeros(1, dim); e2[0, 1] = 1.0
    cos_orth = pdiff_p2_nearest_cosine(e1, e2)
    assert abs(float(cos_orth[0])) < 1e-5, (
        f"orthogonal vectors: cos={float(cos_orth[0]):.6f} expected ~0"
    )

    print("[PASS] test_nearest_cosine_known")


# ---------------------------------------------------------------------------
# Test 3: pairwise_l2_diversity collapse guard
# ---------------------------------------------------------------------------

def test_diversity_collapse_vs_diverse():
    """pdiff_p2_pairwise_l2: collapse (identical) -> ~0; random vectors -> > FLOOR."""
    # Collapse: all samples identical
    v          = torch.randn(1, 32).expand(8, -1).contiguous()
    d_collapse = pdiff_p2_pairwise_l2(v)
    assert d_collapse < 1e-4, (
        f"identical vectors: diversity={d_collapse:.6f} expected ~0"
    )

    # Diverse: random vectors in realistic dim
    torch.manual_seed(7)
    r         = torch.randn(8, 512)
    d_diverse = pdiff_p2_pairwise_l2(r)
    assert d_diverse > COLLAPSE_FLOOR, (
        f"random vectors: diversity={d_diverse:.4f} not > COLLAPSE_FLOOR={COLLAPSE_FLOOR}"
    )

    print(
        f"[PASS] test_diversity_collapse_vs_diverse  "
        f"(collapse={d_collapse:.2e}  diverse={d_diverse:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: Synthetic manifold E2E Gate P2 control
# ---------------------------------------------------------------------------

def _make_manifold_pop(dim: int, mdim: int, k: int, noise: float, seed: int) -> torch.Tensor:
    """Create k weight vectors on an (approx) mdim-D manifold in dim-D.

    Each vector = basis @ coord + Gaussian noise,
    where basis is [dim, mdim] random matrix.  Returns float32 [k, dim].
    """
    g      = torch.Generator().manual_seed(seed)
    basis  = torch.randn(dim, mdim, generator=g)      # [dim, mdim]
    coords = torch.randn(k, mdim, generator=g)         # [k, mdim]
    pop    = coords @ basis.T                          # [k, dim]
    pop   += noise * torch.randn(k, dim, generator=g)
    return pop.float()


def test_p2_synthetic_manifold_control():
    """E2E Gate P2 smoke test: samples from a trained DDPM land near the manifold.

    Steps
    -----
    1. Fabricate 24 vectors on an 8-D manifold in 512-D (tiny noise=0.01).
    2. Train WeightBlockLinearAE(dim=512, latent=16) for 800 epochs on train split.
    3. Save AE checkpoint and population .npy to OS scratchpad (NOT the repo).
    4. Call pdiff_p2_run() with 1000 DDPM epochs.
    5. Assert: mean nearest cosine >= PASS_NEAREST_COS (0.90)
               AND diversity > COLLAPSE_FLOOR (0.01).

    Why this must pass: the manifold is clean (SNR >>1), the linear AE is
    PCA-equivalent and reconstructs training vectors with cos ~1.0, and the
    DDPM on 19 standardized 16-D latents memorises the training distribution.
    Decoded samples therefore land near training weight vectors.
    """
    os.makedirs(_SYNTH_DIR, exist_ok=True)

    DIM          = 512
    MANIFOLD_DIM = 8
    K            = 24
    AE_LATENT    = 16
    AE_EPOCHS    = 800
    DDPM_EPOCHS  = 1000
    N_SAMPLES    = 16
    SEED         = 42

    t0 = time.time()
    print(f"  [synth] fabricating {K} vectors on {MANIFOLD_DIM}-D manifold in {DIM}-D ...")

    # ------------------------------------------------------------------
    # 1. Fabricate population
    # ------------------------------------------------------------------
    raw_all = _make_manifold_pop(DIM, MANIFOLD_DIM, K, noise=0.01, seed=SEED)
    # [24, 512]

    pop_path = os.path.join(_SYNTH_DIR, "synth_population.npy")
    np.save(pop_path, raw_all.numpy())

    # ------------------------------------------------------------------
    # 2. Train AE on training split  (same split logic as pdiff_p2_run)
    # ------------------------------------------------------------------
    g        = torch.Generator().manual_seed(SEED)
    perm     = torch.randperm(K, generator=g)
    k_train  = max(2, int(K * 0.8))     # 19
    raw_train = raw_all[perm[:k_train]]  # [19, 512]

    normed_train, _meta = normalize_global(raw_train)

    ae          = WeightBlockLinearAE(dim=DIM, latent=AE_LATENT)
    ae_ckpt_dir = os.path.join(_SYNTH_DIR, "ae_ckpt")
    ae, ae_loss = train_ae(
        ae, normed_train, ae_ckpt_dir,
        epochs=AE_EPOCHS,
        lr=1e-2,
        ckpt_every=AE_EPOCHS,    # save once at the end
        resume=False,
        verbose=False,
    )
    ae_ckpt = os.path.join(ae_ckpt_dir, f"ae_epoch_{AE_EPOCHS - 1:05d}.pt")
    assert os.path.exists(ae_ckpt), f"AE checkpoint missing: {ae_ckpt}"
    print(f"  [synth] AE trained  loss={ae_loss:.6f}  elapsed={time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Run Gate P2
    # ------------------------------------------------------------------
    metrics = pdiff_p2_run(
        ae_ckpt=ae_ckpt,
        population_path=pop_path,
        n_samples=N_SAMPLES,
        epochs=DDPM_EPOCHS,
        device="cpu",
        train_frac=0.8,
        seed=SEED,
        out_dir=os.path.join(_SYNTH_DIR, "p2_out"),
        verbose=False,
    )

    nn_cos    = metrics["mean_nearest_cosine"]
    diversity = metrics["diversity_mean_pairwise_l2"]
    elapsed   = time.time() - t0

    print(
        f"  [synth] P2 verdict={metrics['verdict']}"
        f"  nearest_cos={nn_cos:.4f}  diversity={diversity:.4f}"
        f"  elapsed={elapsed:.1f}s"
    )

    assert nn_cos >= PASS_NEAREST_COS, (
        f"Synthetic manifold P2 FAIL: mean nearest cosine {nn_cos:.4f} < {PASS_NEAREST_COS}\n"
        f"  DDPM samples are not landing near the training manifold.\n"
        f"  Try more DDPM epochs or a cleaner manifold (less noise)."
    )
    assert diversity > COLLAPSE_FLOOR, (
        f"Synthetic manifold P2 FAIL: diversity {diversity:.4f} <= {COLLAPSE_FLOOR}\n"
        f"  Mode collapse: all samples are converging to the same point."
    )

    print(
        f"[PASS] test_p2_synthetic_manifold_control"
        f"  (nearest_cos={nn_cos:.4f}  diversity={diversity:.4f}  elapsed={elapsed:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running pdiff Gate P2 tests ...")
    print()

    test_standardize_roundtrip()
    test_nearest_cosine_known()
    test_diversity_collapse_vs_diverse()
    test_p2_synthetic_manifold_control()

    print()
    print("ALL pdiff_p2_test TESTS PASSED")
