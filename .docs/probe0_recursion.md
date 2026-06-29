# Probe 0 — Recursive ternary re-feed stability (zero-training spike)

**Question (make-or-break):** Is it numerically stable to recursively re-feed
activations through a slab of BitNet's *ternary* decoder layers (looped depth on a
ternary base)? This gates the *looped ternary MoE* research direction.

**Date:** 2026-06-29
**Branch:** `spike/recursion-stability` (no PR — throwaway spike)
**Script:** `scripts/spike/probe0_recursion.py` (+ `scripts/spike/probe0_results.json`)

## Setup

- Model: `microsoft/bitnet-b1.58-2B-4T-bf16` (the bf16 BitNet master), HF transformers
  `4.57.6`, `BitNetForCausalLM`, `attn_implementation="eager"`.
  - This is the **train==serve** form: `BitLinear` applies ternary absmean weight
    quant **+ int8 activation quant with STE on every forward**, so looping the
    decoder layers genuinely re-exercises the ternary re-feed path. No I2_S GGUF
    needed.
- Device: **CUDA (RTX 3060 12 GB)** — GPU was idle (727 MiB used, 5 % util, 11.8 GB
  free; only desktop/render processes present, no compute job). dtype = bf16.
- Architecture: **L = 30** decoder layers. Looped slab = **layers [7:22)** (15
  layers, i.e. `p=L//4`, `q=3L//4`).
- Forward: `embed → layers[0:7) → LOOP layers[7:22) ×N → layers[22:30) → norm → lm_head`.
- N ∈ {1, 2, 4, 8} (N=1 ≡ stock model). 5 fixed prompts. Greedy degen check =
  24 tokens.

## Sanity (validates the manual forward)

`N=1` manual forward vs stock `model(input_ids).logits`:
**max abs logit diff = 0.0000e+00, final-token argmax match = True.** Exact match —
the manual layer-iteration forward is correct, so the metrics below are trustworthy.

## Results

| N | slab_norm (1st pass) | slab_norm (last pass) | growth × | mean PPL | mean entropy | degenerate prompts |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 167469 | 167469 | 1.00 | 20.58 | 1.697 | 0/5 |
| 2 | 167469 | 292159 | 1.74 | 17.05 | 1.808 | 0/5 |
| 4 | 167469 | 537268 | 3.21 | 28.17 | 1.931 | 0/5 |
| 8 | 167469 | 1002743 | 5.99 | 150.33 | 2.167 | 1/5 |

- **(a) Residual-stream norm** grows **linearly/additively**, NOT exponentially:
  ~+123k per extra slab pass (each looped 15-layer pass adds a roughly constant
  chunk to the residual stream via `h = h + block(h)`). Growth at N=8 is ~6×, i.e.
  bounded growth *rate* — no runaway explosion, no collapse, no NaN/Inf at any N.
- **(b) Perplexity:** N=2 actually *improves* (20.6 → 17.0). Degrades at N=4 (28.2)
  and breaks down at N=8 (150.3) as the inflated residual stream overwhelms the
  final RMSNorm + ternary activation quant.
- **(c) Entropy / degeneration:** entropy drifts up gradually (1.70 → 2.17 → less
  confident). Greedy decode stays coherent through N=4 (0/5 degenerate); first
  degeneration appears only at N=8 (1/5).

## Verdict: **VIABLE (with an expected, fixable caveat)**

The gating question — *does recursively re-feeding through ternary layers blow up?* —
answers **No.** The ternary absmean weight quant + int8 STE activation quant did
**not** destabilise the recursion: behaviour is numerically stable and *predictable*
(linear residual drift), degrading gracefully rather than catastrophically.

- Recursion is genuinely stable for small loop counts. **N=2 even lowers
  perplexity** and shows zero degeneration — looped ternary depth is doing useful
  work, not just adding noise.
- The single failure mode is **residual-stream inflation** (linear norm growth from
  un-renormalised re-feeding), which only bites at N≥4 and breaks decode at N=8.
  This is exactly the kind of thing fixable with **per-step scale recalibration** —
  e.g. a residual rescale / extra norm between loop passes, or a learned per-loop
  gate. It is a tuning problem, not a fundamental instability.

**Recommendation:** the *looped ternary MoE* direction is worth pursuing. Next step
is to add a per-loop residual rescale/norm and re-measure whether N=4–8 can be held
flat-or-improving; if so, this becomes a real lever.

## Caveats / simplifications (what a faithful follow-up would change)

- **No KV cache.** Single full-sequence forward; metrics are teacher-forcing PPL +
  short greedy decode that recomputes the whole sequence each step. A faithful
  follow-up would loop the slab *with* a KV cache during incremental decode.
- **Position handling.** Every looped pass re-uses the natural position ids
  `[0..seq)` (the slab re-sees the same RoPE positions each loop). An alternative
  worth testing is advancing/offsetting positions per loop, which could change
  drift behaviour.
- **bf16, not I2_S GGUF.** Master weights are ternary-quantised on the fly (the
  intended train==serve path); the deployed I2_S kernel path may differ slightly in
  rounding but should track this result.
- **Fixed slab [L//4 : 3L//4] and fixed prompts (5).** Norm growth and the N at
  which quality breaks will depend on slab choice and slab size; not swept here.
- Slab-output norms are large in absolute terms (~167k) — consistent with BitNet's
  known large-activation outliers; the exact N=1 logit match confirms these are the
  real residual magnitudes, and the *relative* growth is what matters.
