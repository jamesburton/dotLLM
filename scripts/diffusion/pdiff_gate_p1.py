"""Gate P1 runner: train weight-block linear autoencoder, evaluate held-out reconstruction.

Usage:
    C:\\Python311\\python.exe scripts\\diffusion\\pdiff_gate_p1.py

Pass criteria (Gate P1):
    held-out mean cosine >= 0.98
    held-out mean rel_l2 <= 0.15

Design choices (both informed by failed experiments):
  - Global z-score normalization (not per-block).  Per-block normalization in
    40960-D space creates near-orthogonal vectors, so NO AE can generalize.
    Global norm preserves inter-block correlation -> linear AE generalizes.
  - Linear AE (WeightBlockLinearAE), not MLP AE.  21 M-param MLP perfectly
    memorizes 32 training blocks but outputs garbage on held-out (cos~0.003).
    Linear AE finds shared PCA subspace -> generalizes by construction.

CPU-only.  Never downloads model weights (local_files_only=True).
New population cache and checkpoints go to C: scratchpad (E: is nearly full).
"""
import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from pdiff_ae import (
    WeightBlockLinearAE,
    extract_diverse_population,
    normalize_global,
    apply_normalize_global,
    denormalize_global,
    train_ae,
    eval_reconstruction,
    BLOCK_DIM,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
E_CKPT_DIR   = os.path.join(_REPO_ROOT, ".docs", "diffusion", "pdiff", "ckpt")

SCRATCHPAD   = r"C:\Users\james\AppData\Local\Temp\claude\E--Development-dotLLM\f5d62daa-0be0-4e86-81e4-cf8f16601e91\scratchpad"
BASE_CKPT    = os.path.join(SCRATCHPAD, "pdiff_ckpt")
LAE_CKPT_DIR = os.path.join(BASE_CKPT, "lae")   # Linear AE checkpoints

BASE_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_TOTAL  = 60   # 30 gate_proj + 30 up_proj, covering all layers
K_TRAIN  = 48
K_HELD   = 12

# latent < K_TRAIN -> PCA compression regime -> better generalization
LATENT_DIM = 32

EPOCHS       = 3000
LR           = 3e-3
WEIGHT_DECAY = 0.0
CKPT_EVERY   = 3000   # save only at the end (~10 MB model-only file)

SEED_EXTRACT = 42
SEED_SPLIT   = 7

COS_THRESHOLD    = 0.98
REL_L2_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(train_m, held_m, final_loss, t_start, cos_thr, rl2_thr):
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
    print("Per-block held-out cosines:")
    for i, (c, r) in enumerate(zip(held_m["cosines"], held_m["rel_l2s"])):
        print(f"  block {i}: cos={c:.4f}  rel_l2={r:.4f}")
    passed  = held_m["mean_cos"] >= cos_thr and held_m["mean_rel_l2"] <= rl2_thr
    verdict = "PASS" if passed else "FAIL"
    print()
    print("=" * 50)
    print(f"Gate P1: {verdict}")
    print(f"  held mean_cos    = {held_m['mean_cos']:.4f}  (threshold >= {cos_thr})")
    print(f"  held mean_rel_l2 = {held_m['mean_rel_l2']:.4f}  (threshold <= {rl2_thr})")
    if final_loss == final_loss:   # not nan
        print(f"  final train loss = {final_loss:.8f}")
    print(f"  total time: {time.time() - t_start:.1f}s")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    print("=" * 60)
    print("Gate P1: Linear AE + global normalization")
    print("=" * 60)
    t_start = time.time()

    os.makedirs(BASE_CKPT,    exist_ok=True)
    os.makedirs(LAE_CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Extract or load population
    # ------------------------------------------------------------------
    pop_cache  = os.path.join(BASE_CKPT, f"population_raw_k{K_TOTAL}.pt")
    locs_cache = os.path.join(BASE_CKPT, f"population_locs_k{K_TOTAL}.json")

    if os.path.exists(pop_cache):
        print(f"Loading cached population  ({pop_cache})")
        data      = torch.load(pop_cache, map_location="cpu", weights_only=True)
        raw       = data["raw"]
        with open(locs_cache) as fh:
            locations = json.load(fh)
        print(f"  loaded {raw.shape[0]} blocks x {raw.shape[1]}-D")
    else:
        print(f"Extracting {K_TOTAL} diverse weight blocks from {BASE_ID} ...")
        t_ex = time.time()
        raw, locations = extract_diverse_population(BASE_ID, k=K_TOTAL, seed=SEED_EXTRACT)
        print(f"  extracted in {time.time() - t_ex:.1f}s  shape={tuple(raw.shape)}")
        torch.save({"raw": raw}, pop_cache)
        with open(locs_cache, "w") as fh:
            json.dump(locations, fh, default=str)
        print(f"  cached to {pop_cache}")

    assert raw.shape == (K_TOTAL, BLOCK_DIM), f"unexpected population shape {raw.shape}"

    # ------------------------------------------------------------------
    # 2. Train / held-out split
    # ------------------------------------------------------------------
    g         = torch.Generator().manual_seed(SEED_SPLIT)
    perm      = torch.randperm(K_TOTAL, generator=g)
    train_idx = perm[:K_TRAIN]
    held_idx  = perm[K_TRAIN:]

    raw_train = raw[train_idx]   # [48, 40960]
    raw_held  = raw[held_idx]    # [12, 40960]

    # ------------------------------------------------------------------
    # 3. Global normalization (computed from training set only)
    # ------------------------------------------------------------------
    normed_train, global_meta = normalize_global(raw_train)
    normed_held               = apply_normalize_global(raw_held, global_meta)

    print(
        f"Global norm: mu={global_meta['mu']:.5f}  sigma={global_meta['sigma']:.4f}  "
        f"normed range [{float(normed_train.min()):.2f}, {float(normed_train.max()):.2f}]"
    )

    # ------------------------------------------------------------------
    # 4. Build linear AE  (latent < K_TRAIN -> PCA regime)
    # ------------------------------------------------------------------
    ae = WeightBlockLinearAE(dim=BLOCK_DIM, latent=LATENT_DIM)
    print(
        f"LinearAE: dim={BLOCK_DIM}  latent={LATENT_DIM}"
        f"  params={ae.n_params():,}  (latent={LATENT_DIM} < K_TRAIN={K_TRAIN}: PCA regime)"
    )

    # ------------------------------------------------------------------
    # 5. Resume from latest checkpoint if available
    # ------------------------------------------------------------------
    resume_from = None
    lae_ckpts = sorted(
        f for f in os.listdir(LAE_CKPT_DIR)
        if f.startswith("ae_epoch_") and f.endswith(".pt")
    )
    if lae_ckpts:
        resume_from = os.path.join(LAE_CKPT_DIR, lae_ckpts[-1])
        print(f"Will resume from {resume_from}")

        # Quick eval before more training
        try:
            tmp = torch.load(resume_from, map_location="cpu", weights_only=False)
            ae.load_state_dict(tmp["model"])
            del tmp
            q_held  = eval_reconstruction(ae, normed_held,  raw_held,  global_meta)
            q_train = eval_reconstruction(ae, normed_train, raw_train, global_meta)
            print(
                f"  Resume eval: train cos={q_train['mean_cos']:.4f}  "
                f"held cos={q_held['mean_cos']:.4f}  rel_l2={q_held['mean_rel_l2']:.4f}"
            )
            if q_held["mean_cos"] >= COS_THRESHOLD and q_held["mean_rel_l2"] <= REL_L2_THRESHOLD:
                print("  Gate P1 PASS at resume checkpoint -- skipping further training.")
                _report(q_train, q_held, float("nan"), t_start, COS_THRESHOLD, REL_L2_THRESHOLD)
                return {
                    "verdict": "PASS", "passed": True,
                    "held": q_held, "train": q_train, "final_loss": float("nan"),
                }
            else:
                print(f"  Not passing gate yet. Continuing training.")
                ae = WeightBlockLinearAE(dim=BLOCK_DIM, latent=LATENT_DIM)
        except Exception as ex:
            print(f"  Quick eval failed ({ex}); training from scratch.")
            ae = WeightBlockLinearAE(dim=BLOCK_DIM, latent=LATENT_DIM)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"Training for {EPOCHS} epochs (ckpts to {LAE_CKPT_DIR}) ...")
    ae, final_loss = train_ae(
        ae,
        normed_train,
        LAE_CKPT_DIR,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        ckpt_every=CKPT_EVERY,
        resume=True,
        resume_from=resume_from,
        seed=0,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    train_m = eval_reconstruction(ae, normed_train, raw_train, global_meta)
    held_m  = eval_reconstruction(ae, normed_held,  raw_held,  global_meta)

    _report(train_m, held_m, final_loss, t_start, COS_THRESHOLD, REL_L2_THRESHOLD)

    passed = (
        held_m["mean_cos"]    >= COS_THRESHOLD
        and held_m["mean_rel_l2"] <= REL_L2_THRESHOLD
    )
    return {
        "verdict":    "PASS" if passed else "FAIL",
        "passed":     passed,
        "held":       held_m,
        "train":      train_m,
        "final_loss": final_loss,
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["passed"] else 1)
