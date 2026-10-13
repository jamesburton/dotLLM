"""pdiff Gate P2 — diffusion in AE latent space.

Pipeline
--------
1. Load a trained AE checkpoint (pdiff_ae format: {model: state_dict, ...}).
2. Load a weight population (.npy float32 [k,D]  or  .pt with 'raw' key).
3. Apply global normalization (fitted to the TRAIN split) and encode the TRAIN
   split to AE latents.
4. Standardize the latents to zero mean / unit std per dimension.
5. Train a wsd_ddpm.DenoiseMLP on the standardized latents.
6. Sample N new latents, un-standardize, decode via AE to weight vectors.
7. Report Gate P2 metrics:
       (a) nearest-neighbour cosine to TRAIN population  (on-manifold test)
       (b) pairwise L2 diversity among samples            (collapse guard)
       (c) coverage of held-out modes if held-out adapters exist
   PASS = mean nearest cosine >= 0.90  AND  diversity > COLLAPSE_FLOOR.

CLI
---
  --ae-ckpt     Path to trained AE checkpoint (.pt from pdiff_ae.train_ae)
  --population  Path to weight population (.npy float32 [k,D] or .pt/'raw' key)
  --latent-dim  AE latent dimension (auto-detected from checkpoint if omitted)
  --n-samples   Number of new weight vectors to sample   [default: 16]
  --out         Output directory for decoded samples and stats
  --epochs      DDPM training epochs                     [default: 2000]
  --device      cpu or cuda                              [default: cpu]
  --train-frac  Fraction of population used as train     [default: 0.8]
  --seed        RNG seed                                  [default: 42]

Namespace: pdiff_p2_*  (see PASS / FAIL constants below).
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pdiff_ae import (
    WeightBlockAE,
    WeightBlockLinearAE,
    normalize_global,
    denormalize_global,
)
from wsd_ddpm import (
    DenoiseMLP,
    make_schedule,
    train_ddpm,
    sample_ddpm,
)

# ---------------------------------------------------------------------------
# Gate P2 pass constants
# ---------------------------------------------------------------------------

COLLAPSE_FLOOR       = 0.01   # mean pairwise L2 below this → mode collapse
PASS_NEAREST_COS     = 0.90   # Gate P2 on-manifold threshold
HELD_COV_THRESHOLD   = 0.50   # cosine threshold used in held-out coverage metric


# ---------------------------------------------------------------------------
# AE loading (auto-detects arch from checkpoint state-dict)
# ---------------------------------------------------------------------------

def pdiff_p2_detect_ae_arch(state_dict: dict):
    """Infer AE architecture and dimensions from checkpoint state dict.

    Returns (arch: str, dim: int, latent: int, hidden: int|None).

    arch='linear'  → WeightBlockLinearAE   (keys: encoder.weight, decoder.weight)
    arch='mlp'     → WeightBlockAE (MLP)   (keys: encoder.0.weight, ...)
    """
    keys = set(state_dict.keys())
    if "encoder.weight" in keys:
        enc_w  = state_dict["encoder.weight"]   # [latent, dim]
        latent, dim = enc_w.shape
        return "linear", int(dim), int(latent), None
    if "encoder.0.weight" in keys:
        enc0_w = state_dict["encoder.0.weight"]  # [hidden, dim]
        hidden, dim = enc0_w.shape
        enc2_w = state_dict["encoder.2.weight"]  # [latent, hidden]
        latent, _ = enc2_w.shape
        return "mlp", int(dim), int(latent), int(hidden)
    raise ValueError(
        f"Cannot identify AE arch from state dict keys: {sorted(keys)[:10]}"
    )


def pdiff_p2_load_ae(ckpt_path: str, device: str = "cpu"):
    """Load a pdiff_ae checkpoint and return (model, block_dim, latent_dim).

    Checkpoint formats accepted:
      {'model': state_dict, 'epoch': int, 'loss': float}  (pdiff_ae.train_ae)
      bare state_dict
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd  = raw["model"] if (isinstance(raw, dict) and "model" in raw) else raw

    arch, dim, latent, hidden = pdiff_p2_detect_ae_arch(sd)
    if arch == "linear":
        ae = WeightBlockLinearAE(dim=dim, latent=latent)
    else:
        ae = WeightBlockAE(dim=dim, latent=latent, hidden=hidden)
    ae.load_state_dict(sd)
    ae = ae.to(device)
    ae.eval()
    return ae, dim, latent


# ---------------------------------------------------------------------------
# Population loading
# ---------------------------------------------------------------------------

def pdiff_p2_load_population(path: str) -> torch.Tensor:
    """Load a weight population from .npy or .pt.

    .npy : direct float32 array [k, D].
    .pt  : dict with 'raw' key (pdiff_gate_p1 format) or a bare tensor.

    Returns float32 CPU tensor [k, D].
    """
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path).astype(np.float32))
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "raw" in data:
            return data["raw"].float()
        for v in data.values():
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                return v.float()
    if isinstance(data, torch.Tensor):
        return data.float()
    raise ValueError(f"Cannot load population from {path!r}: expected .npy or .pt with 'raw' key")


# ---------------------------------------------------------------------------
# Latent standardization
# ---------------------------------------------------------------------------

def pdiff_p2_standardize(latents: torch.Tensor):
    """Standardize latents to per-dimension zero mean / unit std.

    Dimensions with std < 1e-8 are treated as constant (std clamped to 1e-8).

    Returns (standardized [k, L], mean [L], std [L]).
    """
    mean = latents.mean(dim=0)               # [L]
    std  = latents.std(dim=0).clamp(min=1e-8)  # [L]
    return (latents - mean) / std, mean, std


def pdiff_p2_unstandardize(
    std_latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Invert pdiff_p2_standardize."""
    return std_latents * std + mean


# ---------------------------------------------------------------------------
# Gate P2 metric functions
# ---------------------------------------------------------------------------

def pdiff_p2_nearest_cosine(
    samples:    torch.Tensor,   # [n, D]
    population: torch.Tensor,   # [k, D]
) -> torch.Tensor:
    """Per-sample maximum cosine similarity to any vector in the population.

    Returns [n] float tensor in [-1, 1].  High → on-manifold.
    """
    s_norm = F.normalize(samples,    dim=-1)   # [n, D]
    p_norm = F.normalize(population, dim=-1)   # [k, D]
    return (s_norm @ p_norm.T).max(dim=1).values   # [n]


def pdiff_p2_pairwise_l2(samples: torch.Tensor) -> float:
    """Mean pairwise L2 distance among samples (diversity measure).

    High → diverse; near-zero → mode collapse.  Returns 0.0 for n < 2.
    """
    n = samples.shape[0]
    if n < 2:
        return 0.0
    diff = samples.unsqueeze(0) - samples.unsqueeze(1)   # [n, n, D]
    l2s  = diff.norm(dim=-1)                             # [n, n]
    idx  = torch.triu_indices(n, n, offset=1)
    return float(l2s[idx[0], idx[1]].mean())


def pdiff_p2_held_coverage(
    samples:         torch.Tensor,   # [n, D]
    held_out:        torch.Tensor,   # [k_held, D]
    cos_threshold:   float = HELD_COV_THRESHOLD,
) -> dict:
    """Fraction of held-out vectors covered by at least one sample.

    A held-out vector is 'covered' when its best sample cosine >= cos_threshold.
    """
    best_cos = pdiff_p2_nearest_cosine(held_out, samples)   # [k_held]
    covered  = float((best_cos >= cos_threshold).float().mean())
    return {
        "coverage_fraction":  covered,
        "per_held_max_cos":   best_cos.tolist(),
        "cos_threshold":      cos_threshold,
    }


def pdiff_p2_compute_metrics(
    sampled_weights: torch.Tensor,      # [n, D]  decoded samples
    train_weights:   torch.Tensor,      # [k_train, D]
    held_weights:    torch.Tensor = None,
) -> dict:
    """Compute Gate P2 pass/fail metrics.

    Returns dict with 'passed' (bool) and 'verdict' ('PASS'/'FAIL').

    Metrics:
      mean_nearest_cosine        — on-manifold score; must be >= PASS_NEAREST_COS
      diversity_mean_pairwise_l2 — collapse guard;    must be > COLLAPSE_FLOOR
      held_coverage              — (optional) if held_weights provided
    """
    nn_cos      = pdiff_p2_nearest_cosine(sampled_weights, train_weights)
    mean_nn_cos = float(nn_cos.mean())
    diversity   = pdiff_p2_pairwise_l2(sampled_weights)

    result = {
        "mean_nearest_cosine":        mean_nn_cos,
        "per_sample_nearest_cosine":  nn_cos.tolist(),
        "diversity_mean_pairwise_l2": diversity,
        "collapse_floor":             COLLAPSE_FLOOR,
        "pass_nearest_cos_threshold": PASS_NEAREST_COS,
    }
    if held_weights is not None:
        result["held_coverage"] = pdiff_p2_held_coverage(sampled_weights, held_weights)

    passed            = (mean_nn_cos >= PASS_NEAREST_COS) and (diversity > COLLAPSE_FLOOR)
    result["passed"]  = passed
    result["verdict"] = "PASS" if passed else "FAIL"
    return result


# ---------------------------------------------------------------------------
# Main P2 pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_normed(ae, normed: torch.Tensor, device: str) -> torch.Tensor:
    ae.eval()
    return ae.encode(normed.to(device)).cpu()


def pdiff_p2_run(
    ae_ckpt:         str,
    population_path: str,
    n_samples:       int   = 16,
    epochs:          int   = 2000,
    device:          str   = "cpu",
    train_frac:      float = 0.8,
    seed:            int   = 42,
    out_dir:         str   = None,
    verbose:         bool  = True,
) -> dict:
    """Full Gate P2 pipeline.

    Load AE + population → global-normalize → encode → standardize → DDPM train
    → DDIM sample → unstandardize → AE decode → denormalize → metrics.

    Returns the compute_metrics dict plus bookkeeping keys:
      'final_ddpm_loss', 'ae_block_dim', 'ae_latent_dim', 'n_train', 'n_held'.
    """
    t0 = time.time()
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # 1. Load AE
    # ------------------------------------------------------------------
    ae, block_dim, latent_dim = pdiff_p2_load_ae(ae_ckpt, device=device)
    if verbose:
        print(f"[P2] AE loaded: dim={block_dim}  latent={latent_dim}  device={device}")

    # ------------------------------------------------------------------
    # 2. Load population
    # ------------------------------------------------------------------
    raw = pdiff_p2_load_population(population_path)   # [k, D]  CPU
    k   = raw.shape[0]
    if verbose:
        print(f"[P2] Population: {k} vectors × {raw.shape[1]}-D")

    # ------------------------------------------------------------------
    # 3. Train / held split
    # ------------------------------------------------------------------
    g         = torch.Generator().manual_seed(seed)
    perm      = torch.randperm(k, generator=g)
    k_train   = max(2, int(k * train_frac))
    train_idx = perm[:k_train]
    held_idx  = perm[k_train:]
    raw_train = raw[train_idx]                          # [k_train, D]
    raw_held  = raw[held_idx] if len(held_idx) > 0 else None
    if verbose:
        print(f"[P2] Split: {k_train} train / {len(held_idx)} held")

    # ------------------------------------------------------------------
    # 4. Global-normalize training set then encode to latents
    # ------------------------------------------------------------------
    normed_train, global_meta = normalize_global(raw_train)
    latents_train = _encode_normed(ae, normed_train, device)   # [k_train, L]
    if verbose:
        print(
            f"[P2] Latents: shape={tuple(latents_train.shape)}"
            f"  range=[{float(latents_train.min()):.3f}, {float(latents_train.max()):.3f}]"
        )

    # ------------------------------------------------------------------
    # 5. Standardize latents
    # ------------------------------------------------------------------
    std_latents, lat_mean, lat_std = pdiff_p2_standardize(latents_train)
    if verbose:
        print(
            f"[P2] Std-latents: mean_abs={float(std_latents.abs().mean()):.4f}"
            f"  std={float(std_latents.std()):.4f}"
        )

    # ------------------------------------------------------------------
    # 6. Train DDPM on standardized latents
    # ------------------------------------------------------------------
    sched    = make_schedule(T=100, beta_start=1e-4, beta_end=0.10)
    denoiser = DenoiseMLP(dim=latent_dim, hidden=256)
    if verbose:
        print(f"[P2] Training DDPM: latent_dim={latent_dim}  epochs={epochs} ...")
    final_loss = train_ddpm(
        model=denoiser, targets=std_latents, sched=sched,
        epochs=epochs, lr=3e-4, seed=seed,
    )
    if verbose:
        print(f"[P2] DDPM done: loss={final_loss:.6f}  elapsed={time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 7. Sample new latents (DDIM, deterministic given seed+1)
    # ------------------------------------------------------------------
    sampled_std     = sample_ddpm(denoiser, sched, n=n_samples, dim=latent_dim, seed=seed + 1)
    sampled_latents = pdiff_p2_unstandardize(sampled_std, lat_mean, lat_std)
    if verbose:
        print(f"[P2] Sampled {n_samples} latents")

    # ------------------------------------------------------------------
    # 8. Decode latents → weight vectors
    # ------------------------------------------------------------------
    ae.eval()
    with torch.no_grad():
        sampled_normed = ae.decode(sampled_latents.to(device)).cpu()
    sampled_raw = denormalize_global(sampled_normed, global_meta)   # [n, D]
    if verbose:
        print(f"[P2] Decoded weight vectors: shape={tuple(sampled_raw.shape)}")

    # ------------------------------------------------------------------
    # 9. Gate P2 metrics
    # ------------------------------------------------------------------
    metrics = pdiff_p2_compute_metrics(
        sampled_weights=sampled_raw,
        train_weights=raw_train,
        held_weights=raw_held,
    )
    metrics.update(
        final_ddpm_loss=final_loss,
        ae_block_dim=block_dim,
        ae_latent_dim=latent_dim,
        n_train=k_train,
        n_held=int(len(held_idx)),
    )

    if verbose:
        print()
        print("=" * 60)
        print(f"Gate P2: {metrics['verdict']}")
        print(f"  mean nearest cosine (train)   = {metrics['mean_nearest_cosine']:.4f}"
              f"  (threshold >= {PASS_NEAREST_COS})")
        print(f"  diversity (mean pairwise L2)  = {metrics['diversity_mean_pairwise_l2']:.4f}"
              f"  (floor > {COLLAPSE_FLOOR})")
        if "held_coverage" in metrics:
            hc = metrics["held_coverage"]
            print(f"  held-out coverage"
                  f" (@cos>={hc['cos_threshold']}) = {hc['coverage_fraction']:.2%}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "p2_samples.pt")
        torch.save({
            "sampled_raw":     sampled_raw,
            "sampled_latents": sampled_latents,
            "lat_mean":        lat_mean,
            "lat_std":         lat_std,
            "global_meta":     global_meta,
            "metrics":         metrics,
        }, out_path)
        if verbose:
            print(f"[P2] Outputs → {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="pdiff_p2_latent",
        description="pdiff Gate P2 — diffusion in AE latent space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--ae-ckpt", required=True,
        help="Path to trained AE checkpoint (.pt saved by pdiff_ae.train_ae)",
    )
    ap.add_argument(
        "--population", required=True,
        help="Path to weight population (.npy float32 [k,D] or .pt with 'raw' key)",
    )
    ap.add_argument(
        "--latent-dim", type=int, default=None,
        help="AE latent dimension (auto-detected from checkpoint if omitted)",
    )
    ap.add_argument(
        "--n-samples", type=int, default=16,
        help="Number of new weight vectors to sample",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output directory for decoded samples and stats",
    )
    ap.add_argument(
        "--epochs", type=int, default=2000,
        help="DDPM training epochs",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="Device: cpu or cuda",
    )
    ap.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of population to use as training split",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    metrics = pdiff_p2_run(
        ae_ckpt=args.ae_ckpt,
        population_path=args.population,
        n_samples=args.n_samples,
        epochs=args.epochs,
        device=args.device,
        train_frac=args.train_frac,
        seed=args.seed,
        out_dir=args.out,
        verbose=True,
    )
    sys.exit(0 if metrics["passed"] else 1)


if __name__ == "__main__":
    main()
