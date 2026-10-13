"""Gate P1 runner for LoRA adapter populations.

Loads a population.npy produced by pdiff_bootstrap.py, applies global
weight-aware normalization, trains a linear autoencoder (imported from
pdiff_ae), and reports held-out reconstruction quality.

Pass criterion (Gate P1):
    held-out mean cosine >= 0.98
    held-out mean rel_l2 <= 0.15

Unlike pdiff_gate_p1.py (which fails on diverse base-weight blocks because
those are near-isotropic in 40960-D with pairwise cos ~0.005), this runner
operates on same-task LoRA adapter vectors which genuinely concentrate on a
low-dimensional manifold — the linear AE generalizes to held-out adapters.

CPU-only.  Imports from pdiff_ae; does NOT modify it.

Usage:
    C:\\Python311\\python.exe scripts\\diffusion\\pdiff_gate_p1_adapters.py \\
        --population .docs/diffusion/pdiff_boot/population.npy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from pdiff_ae import (
    WeightBlockLinearAE,
    normalize_global,
    apply_normalize_global,
    train_ae,
    eval_reconstruction,
)

# ---------------------------------------------------------------------------
# Defaults / thresholds
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent
_DEFAULT_POP = str(
    _REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot" / "population.npy"
)

COS_THRESHOLD    = 0.98
REL_L2_THRESHOLD = 0.15

# AE hyper-parameters
# latent=8: must be STRICTLY LESS THAN k_train (=int(N*train_frac)) to stay in
# the under-complete PCA regime.  For N=16, train_frac=0.8: k_train=12, so
# latent ∈ {1..11} is safe.  latent=8 is conservative and captures >99% of
# training variance (S_8/S_total > 0.99 empirically).
# Over-complete (latent >= k_train=12) causes Adam to stall at cos≈0.948
# and LBFGS to collapse to zero — both fixed by svd_init=True in train_ae,
# but staying under-complete is the correct long-term posture.
LATENT_DIM   = 8
EPOCHS       = 3000
LR           = 3e-3
WEIGHT_DECAY = 0.0
CKPT_EVERY   = 3000

SEED_SPLIT = 7
SEED_TRAIN = 0


# ---------------------------------------------------------------------------
# Core Gate-P1 logic (importable for tests and synthetic smoke runs)
# ---------------------------------------------------------------------------

def run_gate_p1(
    population: np.ndarray,
    latent: int       = LATENT_DIM,
    epochs: int       = EPOCHS,
    lr: float         = LR,
    weight_decay: float = WEIGHT_DECAY,
    ckpt_dir: str     = "",
    train_frac: float = 0.8,
    seed_split: int   = SEED_SPLIT,
    seed_train: int   = SEED_TRAIN,
    verbose: bool     = True,
    svd_init: bool    = True,
) -> dict:
    """Run Gate P1 on a [N, D] float32 population array.

    Performs:
        1. Train/held-out split (train_frac of N).
        2. Global z-score normalization (from training set only).
        3. Train WeightBlockLinearAE with MSE loss (PCA-equivalent).
        4. Evaluate reconstruction in raw (original) weight space.
        5. Report PASS/FAIL against thresholds.

    Args:
        population:   float32 array [N, D].
        latent:       LinearAE latent dimension (should be < k_train for PCA regime).
        epochs:       Training epochs.
        lr:           Adam learning rate.
        weight_decay: Adam weight decay.
        ckpt_dir:     AE checkpoint directory. Uses a temp dir if empty.
        train_frac:   Fraction of adapters used for training (rest = held-out).
        seed_split:   RNG seed for train/held split.
        seed_train:   RNG seed for AE weight init.
        verbose:      Print training and evaluation output.

    Returns:
        dict with keys:
            verdict    str       'PASS' or 'FAIL'
            passed     bool
            held       dict      per-block cosines/rel_l2s and means
            train      dict      per-block cosines/rel_l2s and means
            final_loss float
            N, D, k_train, k_held  ints
    """
    N, D = population.shape
    if N < 4:
        raise ValueError(f"Need at least 4 adapters in the population, got {N}")

    raw = torch.from_numpy(population.astype(np.float32))   # [N, D]

    # --- Train / held-out split ---
    k_train = max(2, int(N * train_frac))
    k_held  = N - k_train
    g       = torch.Generator().manual_seed(seed_split)
    perm    = torch.randperm(N, generator=g)
    train_idx = perm[:k_train]
    held_idx  = perm[k_train:]

    raw_train = raw[train_idx]   # [k_train, D]
    raw_held  = raw[held_idx]    # [k_held,  D]

    # --- Global normalization (training set statistics only) ---
    normed_train, global_meta = normalize_global(raw_train)
    normed_held               = apply_normalize_global(raw_held, global_meta)

    if verbose:
        print(f"Population: N={N}  D={D}  k_train={k_train}  k_held={k_held}")
        print(
            f"Global norm: mu={global_meta['mu']:.5f}  "
            f"sigma={global_meta['sigma']:.4f}  "
            f"normed range [{float(normed_train.min()):.2f}, {float(normed_train.max()):.2f}]"
        )

    # --- Build LinearAE (PCA-equivalent) ---
    ae = WeightBlockLinearAE(dim=D, latent=latent)
    if verbose:
        regime = "PCA regime" if latent < k_train else "OVER-COMPLETE (latent >= k_train)"
        print(
            f"LinearAE: dim={D}  latent={latent}  n_params={ae.n_params():,}"
            f"  [{regime}: latent={latent} vs k_train={k_train}]"
        )

    # --- Checkpointing ---
    if not ckpt_dir:
        import tempfile
        ckpt_dir = os.path.join(
            tempfile.gettempdir(), "pdiff_adapters_ckpt", f"D{D}_lat{latent}"
        )
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Train ---
    t0 = time.time()
    if verbose:
        print(f"Training for {epochs} epochs (ckpts -> {ckpt_dir}) ...")
    ae, final_loss = train_ae(
        ae,
        normed_train,
        ckpt_dir,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        ckpt_every=CKPT_EVERY,
        resume=True,
        seed=seed_train,
        verbose=verbose,
        svd_init=svd_init,
    )
    elapsed = time.time() - t0

    # --- Evaluate in raw weight space ---
    train_m = eval_reconstruction(ae, normed_train, raw_train, global_meta)
    held_m  = eval_reconstruction(ae, normed_held,  raw_held,  global_meta)

    passed = (
        held_m["mean_cos"]    >= COS_THRESHOLD
        and held_m["mean_rel_l2"] <= REL_L2_THRESHOLD
    )
    verdict = "PASS" if passed else "FAIL"

    if verbose:
        print()
        print("-" * 50)
        print(
            f"Train  mean_cos={train_m['mean_cos']:.4f}  "
            f"mean_rel_l2={train_m['mean_rel_l2']:.4f}  "
            f"min_cos={train_m['min_cos']:.4f}"
        )
        print(
            f"Held   mean_cos={held_m['mean_cos']:.4f}  "
            f"mean_rel_l2={held_m['mean_rel_l2']:.4f}  "
            f"min_cos={held_m['min_cos']:.4f}"
        )
        print()
        print("Per-adapter held-out cosines:")
        for idx, (c, r) in enumerate(zip(held_m["cosines"], held_m["rel_l2s"])):
            print(f"  adapter {idx}: cos={c:.4f}  rel_l2={r:.4f}")
        print()
        print("=" * 50)
        print(f"Gate P1: {verdict}")
        print(f"  held mean_cos    = {held_m['mean_cos']:.4f}  (threshold >= {COS_THRESHOLD})")
        print(f"  held mean_rel_l2 = {held_m['mean_rel_l2']:.4f}  (threshold <= {REL_L2_THRESHOLD})")
        print(f"  final train loss = {final_loss:.8f}")
        print(f"  elapsed: {elapsed:.1f}s")
        print("=" * 50)

    return {
        "verdict":    verdict,
        "passed":     passed,
        "held":       held_m,
        "train":      train_m,
        "final_loss": final_loss,
        "N":          N,
        "D":          D,
        "k_train":    k_train,
        "k_held":     k_held,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    ap = argparse.ArgumentParser(
        description=(
            "Gate P1 on a LoRA adapter population (population.npy from pdiff_bootstrap.py). "
            "Trains a linear autoencoder and reports held-out reconstruction quality. "
            "Pass: held-out mean cosine >= 0.98."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--population", default=_DEFAULT_POP,
        help="Path to population.npy [N, D] float32.",
    )
    ap.add_argument(
        "--latent", type=int, default=LATENT_DIM,
        help="LinearAE latent dimension (should be < n_train for PCA generalization).",
    )
    ap.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Training epochs for the linear AE.",
    )
    ap.add_argument(
        "--lr", type=float, default=LR,
        help="Adam learning rate.",
    )
    ap.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of adapters used for AE training (rest = held-out).",
    )
    ap.add_argument(
        "--ckpt-dir", default="",
        help="AE checkpoint directory. Defaults to a system temp subdirectory.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.population):
        print(f"ERROR: population file not found: {args.population}", file=sys.stderr)
        print("Run pdiff_bootstrap.py first to build the population.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Gate P1: LoRA adapter population")
    print("=" * 60)
    print(f"Loading: {args.population}")

    population = np.load(args.population).astype(np.float32)
    print(f"Loaded: shape={population.shape}  dtype={population.dtype}")

    # Print meta if available
    meta_path = Path(args.population).with_name("meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        print(
            f"Meta: task={meta.get('task')}  base={meta.get('base')}  "
            f"layer={meta.get('layer')}  proj={meta.get('proj')}  "
            f"lora_part={meta.get('lora_part')}  rank={meta.get('rank')}"
        )
    else:
        print("(no meta.json found alongside population.npy)")

    result = run_gate_p1(
        population,
        latent=args.latent,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_dir=args.ckpt_dir,
        train_frac=args.train_frac,
        verbose=True,
    )
    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["passed"] else 1)
