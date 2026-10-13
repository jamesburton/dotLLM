# Weight-Space Diffusion — Experiment 0 (Representation Gate) — Spec

**Date:** 2026-07-02
**Status:** Approved design → spec (ready for implementation plan)
**Parent exploration:** `.planning/2026-07-02-diffusion-weightspace-ternary-exploration.md`

## Goal

The cheapest possible kill/greenlight for the weight-space-diffusion idea: prove that a small
diffusion model can **encode and reconstruct a point in FFN weight space** at realistic
dimensionality (~41 K params, the shape of a rank-16 LoRA-A matrix for a BitNet FFN projection).
If a tiny diffusion cannot even memorize and regenerate one real weight vector, the entire
direction is dead and we stop before spending any compute on the expensive adapter-population
bootstrap.

**Non-goals (explicitly deferred):** ternary quantization of generated weights; the 25-adapter
population; task-conditioned generation; functional PPL evaluation of merged deltas; anything on
the GPU. Experiment 0 is **CPU-only and self-contained** so it runs with zero contention against
the in-progress JA `lm_head` heal on the 3060.

## Architecture

A standalone ε-prediction DDPM over flattened weight vectors, plus a target-extraction step that
draws realistic targets straight from the BitNet base weights, plus a harness that runs the two
gates (0a single-point, 0b five-point) and emits a pass/fail report.

Three focused modules under a new `scripts/diffusion/` directory, each independently testable:

### 1. `scripts/diffusion/wsd_ddpm.py` — the diffusion model (pure, testable)

- **Noise schedule:** linear β from `1e-4` to `0.02`, `T = 100`. Precompute `betas`, `alphas`,
  `alpha_bars` (cumulative product). Expose as a small dataclass/namedtuple.
- **Forward noising** `q_sample(x0, t, noise)`: `sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise`.
  Must return `x0` exactly when `t == 0`.
- **Model** `EpsMLP(dim, hidden=256, n_layers=3)`: input is the noisy vector `x_t` (`dim`-D) plus a
  sinusoidal timestep embedding (embed → small MLP → added into the first hidden activation);
  output is predicted noise `eps_hat` (`dim`-D). ReLU/SiLU activations. Pure `nn.Module`.
- **Training** `train_ddpm(model, targets, epochs, lr, seed)`: standard DDPM loss — for each step
  sample `t ~ U[0,T)`, `noise ~ N(0,I)`, predict `eps_hat`, `MSE(eps_hat, noise)`; Adam. `targets`
  is a `[K, dim]` tensor (K small; overfit deliberately). Deterministic given `seed`. Returns the
  trained model + final loss.
- **Sampling** `sample_ddpm(model, n, dim, seed)`: ancestral DDPM sampling from `x_T ~ N(0,I)` down
  to `x_0`; returns `[n, dim]`. Deterministic given `seed`.
- No file I/O, no argparse, no base-model dependency here — this module is unit-testable in
  isolation with tiny synthetic vectors.

### 2. `scripts/diffusion/wsd_targets.py` — target extraction (base weights → target vectors)

- `extract_targets(base_id, k, cols=16, seed) -> (targets: [k, 2560*16] float32, meta)`.
- Loads ONLY the needed weight tensors from the base safetensors (memory-mapped; NOT the full
  model). Base is `microsoft/bitnet-b1.58-2B-4T-bf16` (matches `mote_train.py --base`; use the HF
  cache, `local_files_only=True`).
- Each target = a `[16, 2560]` block of an FFN projection weight — 16 consecutive output-rows ×
  all 2560 input-cols (real weight statistics) — taken from distinct (layer, row-offset) locations
  so the K targets are genuinely different points. Flatten to a `[40960]` vector. Cast to float32.
- **Normalization:** compute a single global mean/std over the stacked targets; store in `meta` so
  samples can be denormalized before metric comparison. The DDPM trains in normalized space.
- Returns targets (normalized), plus `meta` = `{mean, std, dim, shape, source_locations}`.

### 3. `scripts/diffusion/wsd_exp0.py` — the harness (CLI)

- Flags: `--gate {0a,0b}` (or run both), `--k` (targets; default 1 for 0a, 5 for 0b),
  `--epochs`, `--n-samples`, `--seed`, `--base` (default the bf16 id), `--out` (report path),
  `--self-test` (fast synthetic mode: skip base extraction, use random 64-D targets to validate the
  whole pipeline in seconds).
- Flow: `extract_targets` → `train_ddpm` → `sample_ddpm(n)` → denormalize samples → compute metrics
  → decide verdict → write JSON report + print a clear summary.
- **Metrics** (in `wsd_ddpm.py`): `cosine(a,b)`, `rel_l2(a,b) = ||a-b|| /
  ||b||`, and `nearest_target(sample, targets) -> (idx, cosine, rel_l2)`. Report per-sample nearest
  cosine/rel_l2, the mean, and target-coverage (how many of the K targets are the nearest match for
  ≥1 sample).

## Data flow

```
base safetensors ──(mmap slice [2560,16]×K)──▶ targets [K,40960]
                                                    │ normalize (mean/std)
                                                    ▼
                                        train tiny ε-MLP DDPM (overfit)
                                                    │
                              sample_ddpm(n) ──▶ samples [n,40960] ──denorm──▶
                                                    │
                        nearest-target cosine / rel_l2 / coverage ──▶ verdict JSON
```

## Success / kill criteria

- **Gate 0a (represent one point):** K=1, sample n=10. **PASS** = every sample has cosine ≥ 0.99
  and rel_l2 ≤ 0.05 to the single target. **KILL (stop the entire direction)** if it cannot reach
  this after reasonable overfitting (e.g. cosine < 0.9): a diffusion that cannot memorize ONE
  weight vector cannot represent weight space.
- **Gate 0b (represent multiple points, no collapse):** K=5, sample n=50. **PASS** = mean
  nearest-target cosine ≥ 0.98 AND all 5 targets are covered (each is nearest to ≥1 sample).
  Coverage < 5 with high cosine indicates **mode collapse** — tune once (guidance/epochs); if still
  collapsed, that is a soft-fail flagged in the report (informs whether unconditional sampling can
  span multiple modes, which the population phase depends on).
- Both gates pass → greenlight the population-bootstrap phase (separate future spec). Report states
  the verdict explicitly.

## Testing (TDD)

Plain-python assert tests (run with `C:\Python311\python.exe scripts\diffusion\test_wsd.py`;
matches the repo's existing ad-hoc Python-test convention — no pytest dependency):

1. **Schedule:** `alpha_bars` monotonically decreasing, in `(0,1]`, length T; `q_sample(x0,0,·)==x0`.
2. **Metrics:** `cosine(v,v)==1`, `rel_l2(v,v)==0`; `nearest_target` picks the correct index on
   constructed inputs.
3. **Sampling shape/determinism:** `sample_ddpm` returns `[n, dim]`; identical seed → identical
   output; different seed → different output.
4. **End-to-end synthetic smoke (the pipeline gate):** overfit the DDPM to a single random 64-D
   vector for a modest number of epochs; assert sampled points reach cosine ≥ 0.99 to it. This
   validates train+sample+metric end to end in seconds and is the fast proxy for Gate 0a before
   the 41 K-D base-weight run.

## Hardware / cost

CPU-only. Gate 0a/0b at 40960-D: the ε-MLP first layer is `40960×256 ≈ 10.5 M` params — a ~10 M-param
model, comfortably CPU-trainable. Target extraction memory-maps a few base tensors (no full-model
load). Expected wall time ~1–2 h total on CPU; the synthetic smoke is seconds. **Runs alongside the
GPU JA heal with no contention.**

## File structure

- Create: `scripts/diffusion/wsd_ddpm.py` (schedule + model + train/sample + metrics helpers)
- Create: `scripts/diffusion/wsd_targets.py` (base-weight target extraction + normalization)
- Create: `scripts/diffusion/wsd_exp0.py` (CLI harness: gates 0a/0b + report + `--self-test`)
- Create: `scripts/diffusion/test_wsd.py` (plain-assert unit + synthetic-smoke tests)
- Output (git-ignored/scratch): `.docs/diffusion/wsd_exp0/report.json`
