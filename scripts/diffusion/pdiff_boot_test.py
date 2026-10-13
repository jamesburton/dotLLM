"""Plain-assert tests for the pdiff bootstrap pipeline.

No pytest dependency.  Run directly:
    C:\\Python311\\python.exe scripts\\diffusion\\pdiff_boot_test.py

Tests:
  1. test_extract_format
       Fabricates a fake PEFT adapter_model.bin (torch state dict) with realistic
       lora_A key format, calls extract_lora_vec, asserts shape [rank*in_features]
       and dtype float32.

  2. test_population_save_resume
       Creates a temp directory, saves vec files for adapters 0 and 2 (skipping 1),
       verifies is_done()/load_vec() results, checks that build_population raises
       on a missing index, and verifies that saving a new vec enables a full stack.

  3. test_synthetic_manifold_gate_p1 (POSITIVE CONTROL)
       Fabricates a 20-vector population lying on a known 8-D linear manifold
       embedded in D=1024, with small Gaussian noise (scale=0.01).  Runs
       run_gate_p1() on it (latent=8, epochs=1000, CPU).  Asserts held-out
       mean cosine > 0.9 (positive control: manifold data should pass).

       This is the critical positive control that proves the P1-on-population
       path works correctly when the data genuinely lies on a low-dim manifold --
       which is what the real bootstrap will produce.  The base-weight Gate P1
       failed (held cos=0.038) because those blocks are near-isotropic in 40960-D;
       LoRA adapter vectors from the same task are NOT isotropic.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pdiff_bootstrap import (
    extract_lora_vec,
    is_done,
    save_vec,
    load_vec,
    build_population,
    vec_path,
)
from pdiff_gate_p1_adapters import run_gate_p1

# ---------------------------------------------------------------------------
# Helper: make a synthetic population on a low-dim manifold
# ---------------------------------------------------------------------------

def _make_manifold_population(
    N: int = 20,
    D: int = 1024,
    intrinsic_dim: int = 8,
    noise_scale: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Create N vectors that lie (approximately) on an intrinsic_dim-dimensional
    linear subspace of R^D, with small isotropic Gaussian noise added.

    The linear AE should learn to reconstruct held-out points with high cosine
    similarity when latent >= intrinsic_dim, because the signal concentrates on
    a low-dimensional manifold.
    """
    rng = np.random.default_rng(seed)
    # Orthonormal basis of the manifold
    raw_basis = rng.standard_normal((D, intrinsic_dim))
    Q, _ = np.linalg.qr(raw_basis)          # [D, intrinsic_dim], columns are orthonormal
    basis = Q.T                              # [intrinsic_dim, D]
    # Random coefficients per sample
    coeffs  = rng.standard_normal((N, intrinsic_dim)) * 5.0   # non-trivial scale
    points  = coeffs @ basis                 # [N, D] — on the manifold
    # Small noise
    noise   = rng.standard_normal((N, D)) * noise_scale
    return (points + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: vector extraction / flatten format
# ---------------------------------------------------------------------------

def test_extract_format() -> None:
    """Fabricate a fake PEFT adapter_model.bin and verify extract_lora_vec output."""
    RANK       = 16
    IN_FEATS   = 2560   # hidden_size of BitNet-2B
    OUT_FEATS  = 6912   # intermediate_size of BitNet-2B

    fake_lora_A = torch.randn(RANK, IN_FEATS)   # [rank, in_features]
    fake_lora_B = torch.randn(OUT_FEATS, RANK)  # [out_features, rank]

    state = {
        f"base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": fake_lora_A,
        f"base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": fake_lora_B,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "fake_adapter_000"
        adapter_dir.mkdir()
        bin_path = adapter_dir / "adapter_model.bin"
        torch.save(state, str(bin_path))

        vec = extract_lora_vec(adapter_dir, layer=0, proj="gate_proj", rank=RANK)

    expected_len = RANK * IN_FEATS   # 16 * 2560 = 40960
    assert vec.shape == (expected_len,), (
        f"Expected shape ({expected_len},), got {vec.shape}"
    )
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    # Verify values match the original tensor (flattened)
    ref = fake_lora_A.numpy().reshape(-1).astype(np.float32)
    assert np.allclose(vec, ref, atol=1e-6), "Extracted values do not match source tensor"

    print(
        f"[PASS] test_extract_format  "
        f"shape={vec.shape}  dtype={vec.dtype}  D=rank*in={RANK}*{IN_FEATS}={expected_len}"
    )


# ---------------------------------------------------------------------------
# Test 2: population save / resume (skip-existing) logic
# ---------------------------------------------------------------------------

def test_population_save_resume() -> None:
    """Verify the checkpoint/resume machinery: save, is_done, load, build."""
    D = 40960
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # --- Nothing done yet ---
        assert not is_done(out_dir, 0), "adapter 0 should not be done"
        assert not is_done(out_dir, 1), "adapter 1 should not be done"
        assert not is_done(out_dir, 2), "adapter 2 should not be done"

        # --- Save vecs for adapters 0 and 2 (skip 1) ---
        vec0 = np.ones(D, dtype=np.float32) * 0.1
        vec2 = np.ones(D, dtype=np.float32) * 0.3
        save_vec(out_dir, 0, vec0)
        save_vec(out_dir, 2, vec2)

        # --- is_done results ---
        assert is_done(out_dir, 0), "adapter 0 should be done after save"
        assert not is_done(out_dir, 1), "adapter 1 should still be missing"
        assert is_done(out_dir, 2), "adapter 2 should be done after save"

        # --- load_vec round-trip ---
        loaded0 = load_vec(out_dir, 0)
        loaded2 = load_vec(out_dir, 2)
        assert np.allclose(loaded0, vec0), "Loaded vec0 does not match saved"
        assert np.allclose(loaded2, vec2), "Loaded vec2 does not match saved"
        assert loaded0.dtype == np.float32
        assert loaded0.shape == (D,)

        # --- build_population raises when a required index is missing ---
        raised = False
        try:
            build_population(out_dir, [0, 1, 2])   # idx 1 missing
        except FileNotFoundError:
            raised = True
        assert raised, "build_population should raise FileNotFoundError for missing vec"

        # --- build_population succeeds on done subset ---
        pop = build_population(out_dir, [0, 2])    # only the two done adapters
        assert pop.shape == (2, D), f"Expected shape (2, {D}), got {pop.shape}"
        assert pop.dtype == np.float32
        assert np.allclose(pop[0], vec0)
        assert np.allclose(pop[1], vec2)

        # --- Save adapter 1 and build full population ---
        vec1 = np.ones(D, dtype=np.float32) * 0.2
        save_vec(out_dir, 1, vec1)
        pop_full = build_population(out_dir, [0, 1, 2])
        assert pop_full.shape == (3, D)
        assert np.allclose(pop_full[0], vec0)
        assert np.allclose(pop_full[1], vec1)
        assert np.allclose(pop_full[2], vec2)

        # --- Verify vec file is a proper .npy (not _tmp.npy) ---
        p = vec_path(out_dir, 0)
        assert p.exists() and p.suffix == ".npy", f"Expected .npy file, got {p}"
        tmp_p = out_dir / "adapter_000_vec_tmp.npy"
        assert not tmp_p.exists(), "Temp file should be cleaned up after rename"

    print(
        f"[PASS] test_population_save_resume  "
        f"D={D}  skip-logic=ok  round-trip=ok  build-raise=ok  full-stack=ok"
    )


# ---------------------------------------------------------------------------
# Test 3: synthetic manifold Gate P1 positive control
# ---------------------------------------------------------------------------

def test_synthetic_manifold_gate_p1() -> None:
    """Positive control: 20 vectors on an 8-D linear manifold in D=1024.

    The linear AE (latent=8) should reconstruct held-out vectors with
    mean cosine > 0.9 because the data genuinely lies on an 8-D subspace.

    This proves the pdiff_gate_p1_adapters P1 path works correctly when given
    data with real low-dimensional structure — which is exactly what the GPU
    bootstrap will produce from same-task LoRA adapters.

    Contrast: base-weight blocks (isotropic in 40960-D) give held cos=0.038.
    """
    N            = 20
    D            = 1024
    INTRINSIC    = 8
    NOISE_SCALE  = 0.01
    LATENT       = 8      # == intrinsic_dim: AE can represent the full manifold
    EPOCHS       = 1000
    LR           = 5e-3
    EXPECTED_COS = 0.90   # conservative threshold for the test

    population = _make_manifold_population(
        N=N, D=D, intrinsic_dim=INTRINSIC, noise_scale=NOISE_SCALE, seed=42
    )
    assert population.shape == (N, D), f"Bad population shape: {population.shape}"
    assert population.dtype == np.float32

    # Write to OS temp (not the repo) as required
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = str(Path(tmpdir) / "ae_ckpt")

        result = run_gate_p1(
            population,
            latent=LATENT,
            epochs=EPOCHS,
            lr=LR,
            ckpt_dir=ckpt_dir,
            train_frac=0.8,  # k_train=16, k_held=4
            verbose=True,
        )

    held_cos = result["held"]["mean_cos"]
    assert held_cos > EXPECTED_COS, (
        f"Synthetic manifold: held cos={held_cos:.4f} <= {EXPECTED_COS}. "
        f"The Gate P1 path is broken for low-dim manifold data."
    )

    print(
        f"[PASS] test_synthetic_manifold_gate_p1  "
        f"N={N}  D={D}  intrinsic={INTRINSIC}  latent={LATENT}  "
        f"held_cos={held_cos:.4f} > {EXPECTED_COS}  "
        f"verdict={result['verdict']}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Running pdiff boot tests ...")
    print("=" * 60)

    test_extract_format()
    test_population_save_resume()
    test_synthetic_manifold_gate_p1()

    print()
    print("=" * 60)
    print("ALL pdiff_boot_test TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
