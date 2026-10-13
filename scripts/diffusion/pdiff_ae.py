"""pdiff Gate P1 — weight-block autoencoder with weight-aware normalization.

Architecture
------------
WeightBlockAE: shallow MLP, encoder [D→hidden→latent], decoder [latent→hidden→D].
Normalization: per-block z-score then tanh-squash (SQUASH_C=4.0). For real BitNet
blocks the max per-block z-score is ~8–9σ, mapping to tanh(≈2.2) ≈ 0.976 — well
below saturation.  Round-trip is exact (within float32 eps) for that range.

Extraction
----------
extract_diverse_population() draws from gate_proj + up_proj across all 30 layers.
Both projections are [6912, 2560]; a 16-row slice gives the same 40960-D block
shape as wsd_targets.py. Does NOT modify wsd_targets.py.

Device: CPU only.  Do NOT call .cuda() or change device anywhere.
"""
import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_ROWS  = 16
BLOCK_COLS  = 2560            # hidden_size
BLOCK_DIM   = BLOCK_ROWS * BLOCK_COLS  # 40 960
SQUASH_C    = 4.0             # tanh scale: maps ±9σ → tanh(±2.25) ≈ ±0.978


# ---------------------------------------------------------------------------
# Weight-aware normalization
#
# Two schemes are provided:
#
#   Per-block (normalize_block / normalize_population):
#     Each block gets its own mu/sigma.  Good for training stability but
#     DESTROYS inter-block correlation — in high-dimensional spaces (40960-D)
#     per-block-normalized blocks are nearly orthogonal to each other, so a
#     linear AE trained on them cannot generalize to held-out blocks.
#     Retained for ablation / synthetic tests.
#
#   Global (normalize_global / apply_normalize_global):
#     All blocks share a single mu/sigma computed from the training set.
#     Preserves inter-block correlation so a linear AE can generalize.
#     This is the correct scheme for Gate P1.
# ---------------------------------------------------------------------------

def normalize_block(x: torch.Tensor) -> tuple:
    """Per-block z-score then tanh-squash (ABLATION ONLY — see module docstring).

    Returns (s, meta) where s ∈ (-1, 1) and meta stores the inverse parameters.
    """
    mu    = float(x.mean())
    sigma = float(x.std()) + 1e-8
    z     = (x - mu) / sigma
    s     = torch.tanh(z / SQUASH_C)
    return s, {"mu": mu, "sigma": sigma}


def denormalize_block(s: torch.Tensor, meta: dict) -> torch.Tensor:
    """Exact inverse of normalize_block (within float32 precision for |z| ≤ ~12)."""
    z = torch.atanh(s.clamp(-0.9999, 0.9999)) * SQUASH_C
    return z * meta["sigma"] + meta["mu"]


def normalize_population(raw: torch.Tensor) -> tuple:
    """Per-block normalize each row of raw [k, D] (ABLATION ONLY)."""
    normed, metas = [], []
    for i in range(raw.shape[0]):
        s, m = normalize_block(raw[i])
        normed.append(s)
        metas.append(m)
    return torch.stack(normed), metas


def denormalize_population(normed: torch.Tensor, metas: list) -> torch.Tensor:
    """Invert per-block normalize_population (ABLATION ONLY)."""
    raw = [denormalize_block(normed[i], metas[i]) for i in range(normed.shape[0])]
    return torch.stack(raw)


# ---- Global normalization (the correct scheme for Gate P1) ----

def normalize_global(raw: torch.Tensor) -> tuple:
    """Global z-score computed from raw [k, D] (all values across all blocks).

    Returns (normed [k, D], meta dict) where meta contains 'mu' and 'sigma'.
    No tanh-squash — the outliers (±8σ of the global distribution) stay as-is.
    MSE loss then naturally emphasizes the large components, which is exactly
    what the cosine metric cares about.

    IMPORTANT: compute from the TRAINING set only; apply the same meta to
    held-out blocks via apply_normalize_global().
    """
    mu    = float(raw.mean())
    sigma = float(raw.std()) + 1e-8
    normed = (raw - mu) / sigma
    return normed, {"mu": mu, "sigma": sigma, "mode": "global"}


def apply_normalize_global(raw: torch.Tensor, meta: dict) -> torch.Tensor:
    """Apply pre-computed global normalization (training meta) to any set of blocks."""
    return (raw - meta["mu"]) / meta["sigma"]


def denormalize_global(normed: torch.Tensor, meta: dict) -> torch.Tensor:
    """Invert global normalization."""
    return normed * meta["sigma"] + meta["mu"]


# ---- Per-sample L2 normalization (optional pre-processing layer) ----
#
# Normalizes each adapter to unit L2 norm before global z-scoring.  Removes
# inter-adapter magnitude variation (~5 % std/mean for same-task LoRA adapters)
# so the AE trains purely on directional structure.
#
# For reconstruction: the stored per-adapter norms are multiplied back in at
# denorm time.  For generation: the decoder output must be re-scaled to some
# reference magnitude (e.g., training-set mean norm).
#
# Empirical finding on 16-adapter populations: per-sample L2 norm gives
# effectively identical PCA-ceiling quality to global-only z-score (the norm
# variation is small and the AE loss is dominated by directional error).
# Included here for completeness and future experimentation.

def normalize_per_sample_l2(raw: torch.Tensor) -> tuple:
    """Per-sample L2 normalization → unit sphere → global z-score.

    Computes per-adapter L2 norms, normalizes to unit sphere, then applies
    global z-score across the unit-sphere matrix.

    Returns (normed [k, D], meta dict).
    meta contains:
      'per_norms'  [k]    float32  — per-adapter L2 norms (restore at denorm)
      'mu'         float           — global z-score mean
      'sigma'      float           — global z-score std
      'mode'       str    'per_sample_l2'
    """
    per_norms = torch.linalg.vector_norm(raw, dim=1)             # [k]
    unit = raw / per_norms.unsqueeze(1).clamp(min=1e-12)         # [k, D]
    mu    = float(unit.mean())
    sigma = float(unit.std()) + 1e-8
    normed = (unit - mu) / sigma
    return normed, {"per_norms": per_norms.tolist(), "mu": mu, "sigma": sigma, "mode": "per_sample_l2"}


def apply_normalize_per_sample_l2(raw: torch.Tensor, meta: dict) -> tuple:
    """Apply per-sample-L2 + global-z normalization to new blocks.

    Returns (normed [k, D], norms [k]).  The caller must supply norms at
    denorm time (held-out: use returned norms; generation: use mean train norm).
    """
    per_norms = torch.linalg.vector_norm(raw, dim=1)
    unit = raw / per_norms.unsqueeze(1).clamp(min=1e-12)
    normed = (unit - meta["mu"]) / meta["sigma"]
    return normed, per_norms


def denormalize_per_sample_l2(normed: torch.Tensor, meta: dict, per_norms: torch.Tensor) -> torch.Tensor:
    """Invert per-sample L2 normalization.

    Args:
        normed:     [k, D] — AE output in the per-sample-L2 normalized space.
        meta:       dict returned by normalize_per_sample_l2.
        per_norms:  [k] — per-adapter norms to restore.  Use mean train norm
                    for generation (generation does not have a ground-truth norm).
    """
    unit_recon = normed * meta["sigma"] + meta["mu"]             # undo z-score
    return unit_recon * per_norms.unsqueeze(1)                   # restore magnitude


# ---------------------------------------------------------------------------
# Autoencoders
# ---------------------------------------------------------------------------

class WeightBlockAE(nn.Module):
    """MLP autoencoder over weight blocks with a bottleneck latent.

    Encoder: [dim] -> [hidden] -SiLU-> [latent]
    Decoder: [latent] -> [hidden] -SiLU-> [dim]

    WARNING: with typical population sizes (k=32) and dim=40960, this model has
    ~21 M parameters -- massively over-parameterized.  It will memorize training
    blocks but fail to generalize (held-out cos near 0).  Use WeightBlockLinearAE
    for reliable generalization.  Retained here for reference / ablation.
    """

    def __init__(self, dim: int = BLOCK_DIM, latent: int = 128, hidden: int = 256):
        super().__init__()
        self.dim    = dim
        self.latent = latent
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class WeightBlockLinearAE(nn.Module):
    """Linear (PCA-equivalent) autoencoder for weight blocks.

    Encoder: [dim] -> [latent]  (linear, no bias)
    Decoder: [latent] -> [dim]  (linear, no bias)

    With MSE loss, gradient descent approximates PCA: it finds the principal
    components of the training data.  Key generalization property: for any new
    block x, D(E(x)) is the projection of x onto the column space of D -- the
    best rank-latent approximation in the training span.  Unlike the MLP AE,
    this CANNOT memorize individual training examples; it learns a shared linear
    subspace.

    For weight blocks from the same trained model the intrinsic dimensionality
    is well below the latent dim (empirically 10-30 directions), so the linear
    AE generalizes to held-out blocks from the same model.

    IMPORTANT — latent must be STRICTLY LESS THAN k_train (number of training
    samples):
      - Under-complete (latent < k_train): unique PCA solution; well-conditioned
        gradient landscape; gradient descent converges reliably.
      - Over-complete (latent >= k_train): degenerate landscape; Adam gets stuck
        at a sub-optimal plateau (train cos ~0.95); LBFGS collapses to zero.
        Use svd_init=True in train_ae to bypass both failure modes.

    Recommended latent for LoRA-adapter populations: 8 (safe below typical
    k_train of 12-13 for N=16, train_frac=0.8).

    Parameter count: 2 * dim * latent (~655 K for dim=40960, latent=8).
    Training is fast; SVD init makes it effectively instant.
    """

    def __init__(self, dim: int = BLOCK_DIM, latent: int = 64):
        super().__init__()
        self.dim    = dim
        self.latent = latent
        self.encoder = nn.Linear(dim, latent, bias=False)
        self.decoder = nn.Linear(latent, dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# SVD initialization for WeightBlockLinearAE
# ---------------------------------------------------------------------------

def svd_init_linear_ae(ae: "WeightBlockLinearAE", normed_train: torch.Tensor) -> None:
    """Initialize WeightBlockLinearAE weights via exact PCA (thin SVD).

    Sets encoder.weight = Vt[:latent]  and  decoder.weight = Vt[:latent].T
    where X = U @ diag(S) @ Vt is the thin SVD of normed_train [k, D].

    After this call, ae(x) is the projection of x onto the top-latent principal
    components of normed_train.  This is the unique global optimum of the MSE
    loss for the under-complete (latent < k) case, reached in a single step
    instead of thousands of gradient-descent iterations.

    For the over-complete case (latent >= k), this initialization still produces
    a valid projection (ae(x) = x for all training points), avoiding the saddle-
    point collapse that random-init + Adam/LBFGS produces.

    Called by train_ae() when svd_init=True (the default) and start_epoch == 0
    (i.e., not resuming from a checkpoint).

    Args:
        ae:            WeightBlockLinearAE instance (modified in-place).
        normed_train:  [k, D] float32 tensor — the normalized training set.
    """
    with torch.no_grad():
        _, _, Vt = torch.linalg.svd(normed_train, full_matrices=False)
        L = ae.latent
        # Clamp in case latent > k (over-complete).
        L_eff = min(L, Vt.shape[0])
        if L_eff < L:
            # Pad remaining rows with small random noise to break symmetry.
            pad = torch.randn(L - L_eff, ae.dim, dtype=Vt.dtype) * 1e-4
            Vt_L = torch.cat([Vt[:L_eff], pad], dim=0)
        else:
            Vt_L = Vt[:L_eff]
        ae.encoder.weight.data.copy_(Vt_L)    # [L, D]
        ae.decoder.weight.data.copy_(Vt_L.T)  # [D, L]


# ---------------------------------------------------------------------------
# Diverse population extraction (gate_proj + up_proj)
# ---------------------------------------------------------------------------

def _resolve_shard(base_id: str, key: str) -> str:
    """Resolve cache-local path to the safetensors shard that holds `key`."""
    try:
        idx_path = hf_hub_download(
            base_id, "model.safetensors.index.json", local_files_only=True
        )
        weight_map = json.load(open(idx_path))["weight_map"]
        shard = weight_map[key]
    except Exception:
        shard = "model.safetensors"
    return hf_hub_download(base_id, shard, local_files_only=True)


def _list_proj_keys(base_id: str, proj_name: str) -> list:
    """Return sorted list of all weight keys ending in mlp.<proj_name>.weight."""
    try:
        idx_path = hf_hub_download(
            base_id, "model.safetensors.index.json", local_files_only=True
        )
        wm = json.load(open(idx_path))["weight_map"]
        return sorted(k for k in wm if k.endswith(f"mlp.{proj_name}.weight"))
    except Exception:
        path = hf_hub_download(base_id, "model.safetensors", local_files_only=True)
        with safe_open(path, framework="pt", device="cpu") as f:
            return sorted(k for k in f.keys() if k.endswith(f"mlp.{proj_name}.weight"))


def extract_diverse_population(base_id: str, k: int = 40, seed: int = 42) -> tuple:
    """Extract k weight blocks from gate_proj + up_proj across all layers.

    k//2 blocks drawn from gate_proj, k - k//2 from up_proj.
    Both projections have shape [6912, 2560]; a 16-row block = 40960-D vector.

    Returns:
        raw  [k, 40960]  float32  (un-normalized raw weights)
        locations        list of (proj_type_str, layer_int, row_off_int)

    Notes:
        - CPU-only; local_files_only=True; never downloads.
        - Each tensor is freed immediately after slicing to reduce peak memory.
        - Groups blocks by (proj, shard) to minimise file opens.
    """
    g         = torch.Generator().manual_seed(seed)
    gate_keys = _list_proj_keys(base_id, "gate_proj")
    up_keys   = _list_proj_keys(base_id, "up_proj")
    n_layers  = len(gate_keys)
    assert n_layers > 0, "no gate_proj weights found in base model"

    k_gate = k // 2
    k_up   = k - k_gate
    gate_rows = 6912  # intermediate_size

    # --- choose k distinct (proj, layer, row_off) locations ---
    locations: list = []
    seen: set      = set()

    for proj_type, n_want in [("gate", k_gate), ("up", k_up)]:
        count = 0
        while count < n_want:
            layer   = int(torch.randint(0, n_layers, (1,), generator=g))
            row_off = int(torch.randint(0, gate_rows - BLOCK_ROWS, (1,), generator=g))
            loc     = (proj_type, layer, row_off)
            if loc in seen:
                continue
            seen.add(loc)
            locations.append(loc)
            count += 1

    # --- group by (proj, shard) for efficient I/O ---
    # Build a map: key_str -> list of (locations_idx, row_off)
    key_to_locs: dict = {}
    for i, (proj_type, layer, row_off) in enumerate(locations):
        wkey = gate_keys[layer] if proj_type == "gate" else up_keys[layer]
        key_to_locs.setdefault(wkey, []).append((i, row_off))

    vecs = [None] * k
    for wkey, locs_for_key in key_to_locs.items():
        shard = _resolve_shard(base_id, wkey)
        with safe_open(shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(wkey)        # [6912, 2560]  bf16 or fp32
        w_f32 = w.to(torch.float32)
        del w                             # free bfloat16 copy
        for idx, row_off in locs_for_key:
            block     = w_f32[row_off : row_off + BLOCK_ROWS, :].reshape(-1).clone()
            vecs[idx] = block
        del w_f32

    raw = torch.stack(vecs)              # [k, 40960]
    return raw, locations


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ae(
    ae: WeightBlockAE,
    normed_train: torch.Tensor,
    ckpt_dir: str,
    epochs: int         = 5000,
    lr: float           = 3e-3,
    weight_decay: float = 1e-5,
    ckpt_every: int     = 1000,
    resume: bool        = True,
    resume_from: str    = None,
    seed: int           = 42,
    verbose: bool       = True,
    svd_init: bool      = True,
) -> tuple:
    """Train the AE with full-batch gradient descent.  Returns (ae, final_loss).

    Checkpoints saved to <ckpt_dir>/ae_epoch_NNNNN.pt every `ckpt_every` epochs.
    Only model weights are saved (no optimizer state) to keep checkpoint size small.

    resume_from: explicit path to a checkpoint to load before training.  If None
                 and resume=True, the latest checkpoint in ckpt_dir is used.

    svd_init: bool (default True)
        When True AND ae is a WeightBlockLinearAE AND no checkpoint is being
        resumed, initialize encoder/decoder via exact PCA (thin SVD of
        normed_train) before running Adam fine-tuning.

        WHY this matters: a random-init WeightBlockLinearAE trained with Adam on
        a LoRA-adapter population (N~16, latent~8–32) typically stalls at
        train cos ~0.948, far from the theoretical ceiling of ~0.975.  The root
        cause is a degenerate loss landscape when latent >= k_train (over-complete
        regime): Adam finds a sub-optimal plateau and LBFGS collapses to zero.
        SVD init bypasses gradient descent entirely for the initial solution,
        giving the optimal PCA projection at epoch 0.  Subsequent Adam steps
        fine-tune in the under-complete (latent < k) case and are no-ops in the
        over-complete case (already at MSE = 0 on training data).

        Set svd_init=False to replicate old (uninitialized) behaviour or when
        resuming from a checkpoint (resume=True automatically skips re-init).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(seed)

    opt       = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)

    start_epoch = 0

    # Determine which checkpoint to resume from
    ckpt_path_to_load = None
    if resume_from and os.path.exists(resume_from):
        ckpt_path_to_load = resume_from
    elif resume:
        existing = sorted(
            f for f in os.listdir(ckpt_dir) if f.startswith("ae_epoch_") and f.endswith(".pt")
        )
        if existing:
            ckpt_path_to_load = os.path.join(ckpt_dir, existing[-1])

    if ckpt_path_to_load:
        ckpt = torch.load(ckpt_path_to_load, map_location="cpu", weights_only=False)
        ae.load_state_dict(ckpt["model"])
        # optimizer state may be absent (model-only checkpoints)
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
            except Exception:
                pass
        # rebuild scheduler state to match resumed position
        resumed_epoch = ckpt.get("epoch", 0)
        for _ in range(resumed_epoch + 1):
            scheduler.step()
        start_epoch = resumed_epoch + 1
        if verbose:
            print(
                f"[pdiff_ae] Resumed from {ckpt_path_to_load}"
                f"  (epoch {resumed_epoch}  loss={ckpt.get('loss', float('nan')):.6f})"
            )
    elif svd_init and isinstance(ae, WeightBlockLinearAE):
        # SVD init: set encoder/decoder to the top-latent principal components of
        # normed_train.  This gives the optimal PCA projection at epoch 0 and
        # avoids the ~0.948 stall of random-init Adam on this tiny dataset.
        svd_init_linear_ae(ae, normed_train)
        if verbose:
            print("[pdiff_ae] SVD init: encoder/decoder set to top-latent PCA components.")

    ae.train()
    loss_val = float("nan")
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        out  = ae(normed_train)
        loss = F.mse_loss(out, normed_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        loss_val = float(loss.detach())

        save_now = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if save_now:
            ckpt_path = os.path.join(ckpt_dir, f"ae_epoch_{epoch:05d}.pt")
            try:
                # Save model weights only (no optimizer state) to keep file small (~84 MB)
                torch.save({"model": ae.state_dict(), "epoch": epoch, "loss": loss_val}, ckpt_path)
                save_note = f"saved {ckpt_path}"
            except Exception as exc:
                save_note = f"checkpoint save FAILED: {exc}"
            if verbose:
                elapsed = time.time() - t0
                print(
                    f"[pdiff_ae] epoch {epoch+1:>5d}/{epochs}"
                    f"  loss={loss_val:.6f}"
                    f"  lr={scheduler.get_last_lr()[0]:.2e}"
                    f"  elapsed={elapsed:.1f}s  {save_note}"
                )

    return ae, loss_val


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_reconstruction(
    ae,
    normed_blocks: torch.Tensor,
    raw_blocks: torch.Tensor,
    norm_meta,
) -> dict:
    """Evaluate reconstruction quality in raw (original) weight space.

    norm_meta can be:
      - list of per-block dicts  → per-block denormalization (ABLATION mode)
      - single dict with 'mode'='global' → global denormalization (GATE P1 mode)

    For each block:
      recon_normed = AE(normed_block)
      recon_raw    = denormalize(recon_normed)
      cosine(recon_raw, raw_block), rel_l2(recon_raw, raw_block)

    Returns dict with per-block lists and summary means.
    """
    ae.eval()
    recon_normed = ae(normed_blocks)   # [k, D]

    # Denormalize
    if isinstance(norm_meta, dict) and norm_meta.get("mode") == "global":
        recon_raw = denormalize_global(recon_normed, norm_meta)
    else:
        recon_raw = denormalize_population(recon_normed, norm_meta)

    cosines, rel_l2s = [], []
    for i in range(raw_blocks.shape[0]):
        a   = recon_raw[i]
        b   = raw_blocks[i]
        cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
        rl2 = float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12))
        cosines.append(cos)
        rel_l2s.append(rl2)

    return {
        "cosines":     cosines,
        "rel_l2s":     rel_l2s,
        "mean_cos":    sum(cosines) / len(cosines),
        "mean_rel_l2": sum(rel_l2s) / len(rel_l2s),
        "min_cos":     min(cosines),
        "max_rel_l2":  max(rel_l2s),
    }
