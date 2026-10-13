"""pdiff_func_generate.py — Generate new LoRA module tensors via AE + latent diffusion.

For each module listed in the manifest (from pdiff_func_extract.py):
  1. Load the [N, D] population from modpops/<module_id>.npy.
  2. Hold out adapter at --held-out-idx (default 15); train on the remaining N-1.
  3. Fit a WeightBlockLinearAE (global normalization, PCA-equivalent).
  4. Standardize the train latents and train a DenoiseMLP DDPM.
  5. DDPM-sample 1 new latent → decode → denormalize  →  gen/<module_id>.npy
  6. AE-encode held-out → AE-decode → denormalize     →  recon/<module_id>.npy

CPU-only.  Imports AE/DDPM classes from pdiff_ae and wsd_ddpm.

Usage (small subset, fast validation):
    python pdiff_func_generate.py \\
        --manifest .docs/diffusion/pdiff_boot_fixedinit/modpops/manifest.json \\
        --out-dir  .docs/diffusion/pdiff_func_out \\
        --ae-epochs 50 --diff-steps 20 --device cpu

Full run:
    python pdiff_func_generate.py \\
        --manifest .docs/diffusion/pdiff_boot_fixedinit/modpops/manifest.json \\
        --out-dir  .docs/diffusion/pdiff_func_out \\
        --ae-epochs 200 --diff-steps 100 --device cpu
"""

import argparse
import json
import sys
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdiff_ae import (
    WeightBlockLinearAE,
    normalize_global,
    apply_normalize_global,
    denormalize_global,
    train_ae,
)
from pdiff_p2_latent import (
    pdiff_p2_standardize,
    pdiff_p2_unstandardize,
)
from wsd_ddpm import (
    DenoiseMLP,
    make_schedule,
    train_ddpm,
    sample_ddpm,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MANIFEST = str(
    _REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot_fixedinit" / "modpops" / "manifest.json"
)
_DEFAULT_OUT_DIR  = str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_func_out")

LATENT_DIM   = 8     # PCA regime: must be < N-1 (= 15 when N=16)
DDPM_T       = 100   # number of diffusion timesteps (beta schedule length)


# ---------------------------------------------------------------------------
# Per-module generate
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode(ae: WeightBlockLinearAE, x: torch.Tensor) -> torch.Tensor:
    ae.eval()
    return ae.encode(x)


@torch.no_grad()
def _decode(ae: WeightBlockLinearAE, z: torch.Tensor) -> torch.Tensor:
    ae.eval()
    return ae.decode(z)


def generate_module(
    module_info:    dict,
    pop_path:       Path,
    out_gen:        Path,
    out_recon:      Path,
    held_out_idx:   int,
    ae_epochs:      int,
    diff_steps:     int,
    device:         str,
    verbose:        bool = False,
) -> dict:
    """Run the full AE + DDPM pipeline for a single module.

    Args:
        module_info:   Entry from manifest['modules'], with keys 'module_id', 'D', etc.
        pop_path:      Path to the [N, D] population .npy file.
        out_gen:       Output path for the generated flat tensor (.npy).
        out_recon:     Output path for the AE-reconstructed flat tensor (.npy).
        held_out_idx:  Row index of the held-out adapter in the population.
        ae_epochs:     Training epochs for the linear AE.
        diff_steps:    Training epochs for the DDPM.
        device:        'cpu' or 'cuda'.
        verbose:       Print per-module progress.

    Returns:
        dict with 'module_id', 'ae_loss', 'ddpm_loss', 'gen_shape', 'recon_shape'.
    """
    mid = module_info["module_id"]
    pop = np.load(str(pop_path)).astype(np.float32)   # [N, D]
    N, D = pop.shape

    if N <= held_out_idx:
        raise IndexError(
            f"{mid}: held_out_idx={held_out_idx} out of range for N={N}"
        )

    # ----------------------------------------------------------------
    # Split: train = all rows except held_out_idx
    # ----------------------------------------------------------------
    train_rows = [i for i in range(N) if i != held_out_idx]
    raw_train  = torch.from_numpy(pop[train_rows]).to(device)  # [N-1, D]
    raw_held   = torch.from_numpy(pop[[held_out_idx]]).to(device)  # [1, D]
    k_train    = raw_train.shape[0]

    # ----------------------------------------------------------------
    # Global normalization (training-set statistics only)
    # ----------------------------------------------------------------
    normed_train, global_meta = normalize_global(raw_train.cpu())
    normed_train = normed_train.to(device)
    normed_held  = apply_normalize_global(raw_held.cpu(), global_meta).to(device)

    # ----------------------------------------------------------------
    # Linear AE (PCA-equivalent) — latent < k_train for generalization
    # ----------------------------------------------------------------
    latent = min(LATENT_DIM, k_train - 1)
    ae     = WeightBlockLinearAE(dim=D, latent=latent).to(device)

    ckpt_dir = os.path.join(
        tempfile.gettempdir(), "pdiff_func_gen_ae", mid
    )
    ae, ae_loss = train_ae(
        ae,
        normed_train.cpu() if device == "cpu" else normed_train,
        ckpt_dir,
        epochs=ae_epochs,
        lr=3e-3,
        weight_decay=0.0,
        ckpt_every=ae_epochs,
        resume=False,      # fresh per-module run
        seed=42,
        verbose=verbose,
    )
    ae = ae.to(device)

    # ----------------------------------------------------------------
    # Encode training vectors to latent space
    # ----------------------------------------------------------------
    latents_train = _encode(ae, normed_train)   # [k_train, latent]

    # ----------------------------------------------------------------
    # Standardize latents (per-dimension zero mean / unit std)
    # ----------------------------------------------------------------
    std_latents, lat_mean, lat_std = pdiff_p2_standardize(latents_train.cpu())

    # ----------------------------------------------------------------
    # DDPM in standardized latent space
    # ----------------------------------------------------------------
    sched    = make_schedule(T=DDPM_T, beta_start=1e-4, beta_end=0.10)
    denoiser = DenoiseMLP(dim=latent, hidden=64)
    ddpm_loss = train_ddpm(
        model=denoiser,
        targets=std_latents,
        sched=sched,
        epochs=diff_steps,
        lr=3e-4,
        seed=42,
    )

    # ----------------------------------------------------------------
    # Sample 1 new latent (GEN path)
    # ----------------------------------------------------------------
    sampled_std  = sample_ddpm(denoiser, sched, n=1, dim=latent, seed=43)   # [1, latent]
    sampled_lat  = pdiff_p2_unstandardize(sampled_std, lat_mean, lat_std)

    with torch.no_grad():
        gen_normed = _decode(ae, sampled_lat.to(device)).cpu()              # [1, D]
    gen_raw = denormalize_global(gen_normed, global_meta)                   # [1, D]

    # ----------------------------------------------------------------
    # AE-reconstruct held-out vector (RECON path — no DDPM)
    # ----------------------------------------------------------------
    with torch.no_grad():
        recon_normed = ae(normed_held.cpu() if device == "cpu" else normed_held).cpu()  # [1, D]
    recon_raw = denormalize_global(recon_normed, global_meta)               # [1, D]

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_gen.parent.mkdir(parents=True, exist_ok=True)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_gen),   gen_raw[0].numpy().astype(np.float32))
    np.save(str(out_recon), recon_raw[0].numpy().astype(np.float32))

    if verbose:
        print(
            f"  ae_loss={ae_loss:.6f}  ddpm_loss={ddpm_loss:.6f}"
            f"  gen={out_gen.name}  recon={out_recon.name}"
        )

    return {
        "module_id":   mid,
        "ae_loss":     float(ae_loss),
        "ddpm_loss":   float(ddpm_loss),
        "gen_shape":   list(gen_raw.shape),
        "recon_shape": list(recon_raw.shape),
        "D":           D,
        "latent":      latent,
        "k_train":     k_train,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_generate(
    manifest_path: Path,
    out_dir:       Path,
    held_out_idx:  int,
    ae_epochs:     int,
    diff_steps:    int,
    device:        str,
) -> dict:
    """Run generate for all modules listed in the manifest.

    Returns a results dict: {module_id: result_dict}.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    modpops_dir = manifest_path.parent
    modules     = manifest["modules"]
    n_total     = len(modules)

    gen_dir   = out_dir / "gen"
    recon_dir = out_dir / "recon"
    gen_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    t0      = time.time()
    for i, (mid, info) in enumerate(modules.items(), 1):
        pop_path  = modpops_dir / info["npy_file"]
        out_gen   = gen_dir   / f"{mid}.npy"
        out_recon = recon_dir / f"{mid}.npy"

        print(f"[{i}/{n_total}]  {mid}  D={info['D']} ...", end="  ", flush=True)
        t1 = time.time()

        res = generate_module(
            module_info=info,
            pop_path=pop_path,
            out_gen=out_gen,
            out_recon=out_recon,
            held_out_idx=held_out_idx,
            ae_epochs=ae_epochs,
            diff_steps=diff_steps,
            device=device,
            verbose=False,
        )
        elapsed = time.time() - t1
        print(f"ae_loss={res['ae_loss']:.4f}  ddpm_loss={res['ddpm_loss']:.4f}  {elapsed:.1f}s")
        results[mid] = res

    total = time.time() - t0
    print()
    print(f"Done: {len(results)}/{n_total} modules in {total:.1f}s")
    print(f"Gen   -> {gen_dir}")
    print(f"Recon -> {recon_dir}")

    # Save summary
    summary_path = out_dir / "generate_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "manifest":     str(manifest_path),
            "held_out_idx": held_out_idx,
            "ae_epochs":    ae_epochs,
            "diff_steps":   diff_steps,
            "device":       device,
            "results":      results,
        }, fh, indent=2)
    print(f"Summary -> {summary_path}")

    return results


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Generate new LoRA module tensors via AE + latent DDPM for each module "
            "in the manifest.  Outputs gen/<module_id>.npy and recon/<module_id>.npy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--manifest", default=_DEFAULT_MANIFEST,
        help="Path to manifest.json produced by pdiff_func_extract.py.",
    )
    ap.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help="Output directory; gen/ and recon/ subdirs are created inside.",
    )
    ap.add_argument(
        "--held-out-idx", type=int, default=15,
        help="Row index (0-based) of the held-out adapter in the population arrays.",
    )
    ap.add_argument(
        "--ae-epochs", type=int, default=100,
        help="Training epochs for the linear autoencoder per module.",
    )
    ap.add_argument(
        "--diff-steps", type=int, default=50,
        help="Training epochs for the DDPM denoiser per module.",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="Compute device (cpu recommended; CUDA not required).",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    manifest_path = Path(args.manifest)
    out_dir       = Path(args.out_dir)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        print("Run pdiff_func_extract.py first to build the manifest.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("pdiff_func_generate: AE + DDPM module generator")
    print("=" * 60)
    print(f"  manifest     : {manifest_path}")
    print(f"  out-dir      : {out_dir}")
    print(f"  held-out-idx : {args.held_out_idx}")
    print(f"  ae-epochs    : {args.ae_epochs}")
    print(f"  diff-steps   : {args.diff_steps}")
    print(f"  device       : {args.device}")
    print()

    run_generate(
        manifest_path=manifest_path,
        out_dir=out_dir,
        held_out_idx=args.held_out_idx,
        ae_epochs=args.ae_epochs,
        diff_steps=args.diff_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
