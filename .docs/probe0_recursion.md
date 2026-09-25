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

---

# Probe 0b — Does a per-loop renorm fix the residual-inflation failure?

**Question:** Probe 0 found one failure mode — the residual-stream norm inflates
~linearly per loop (~+123k/pass), breaking greedy decode by N=8. Does inserting a
**per-loop residual renormalization between slab passes** hold the loop
flat-or-improving out to N=4, N=8 (and N=16)?

**Date:** 2026-06-29
**Branch:** `spike/recursion-stability` (no PR — same throwaway spike)
**Script:** `scripts/spike/probe0b_renorm.py` (+ `scripts/spike/probe0b_results.json`)

## Setup

Identical to Probe 0 — same model (`microsoft/bitnet-b1.58-2B-4T-bf16`, HF
`BitNetForCausalLM`, eager attn, bf16, transformers 4.57.6), same **CUDA RTX 3060**,
same **L=30**, same looped slab **[7:22)** (15 layers), same 5 prompts, same
greedy-degen check (24 tokens). Extended to **N ∈ {1, 2, 4, 8, 16}**.

Renorm is applied to the hidden state **strictly between passes** (after each
non-final pass, before re-feeding), so **N=1 ≡ stock model for every variant**
(sanity confirmed: max-abs logit diff = 0.0, argmax match = True, all four variants).

Variants:
- **none** — baseline, no renorm (reproduces Probe 0, now extended to N=16).
- **(a) rescale** — norm-preserving: after each non-final pass, rescale `h` per-token
  so its L2 norm equals the *slab-input* norm of that pass (holds residual norm
  constant across loops).
- **(b) rmsnorm** — weightless (unit-weight) RMSNorm applied between passes.
- **(c) damped** — under-relaxed residual: for passes ≥2,
  `h ← h_in + γ·(slab(h_in) − h_in)`, **γ = 0.5**.

## Results (mean over 5 prompts)

**none** (baseline, extended):

| N | norm 1st→last | growth× | mean PPL | entropy | degenerate |
|--:|--:|--:|--:|--:|--:|
| 1 | 167469→167469 | 1.00 | 20.58 | 1.697 | 0/5 |
| 2 | 167469→292159 | 1.74 | **17.05** | 1.807 | 0/5 |
| 4 | 167469→537268 | 3.21 | 28.17 | 1.931 | 0/5 |
| 8 | 167469→1002743 | 5.99 | 150.33 | 2.167 | 1/5 |
| 16 | 167469→1837072 | 10.97 | 1301.34 | 2.292 | 4/5 |

**(a) rescale** — norm-preserving rescale to slab-input norm:

| N | norm 1st→last | growth× | mean PPL | entropy | degenerate |
|--:|--:|--:|--:|--:|--:|
| 1 | 167469→167469 | 1.00 | 20.58 | 1.697 | 0/5 |
| 2 | 25510→144762 | 5.67 | 28.13 | 3.765 | 2/5 |
| 4 | 25510→124107 | 4.86 | 553.81 | 5.668 | 5/5 |
| 8 | 25510→121108 | 4.75 | 94209.21 | 5.051 | 2/5 |
| 16 | 25510→123345 | 4.84 | 321184.27 | 5.406 | 0/5 |

**(b) rmsnorm** — weightless RMSNorm between passes:

| N | norm 1st→last | growth× | mean PPL | entropy | degenerate |
|--:|--:|--:|--:|--:|--:|
| 1 | 167469→167469 | 1.00 | 20.58 | 1.697 | 0/5 |
| 2 | 51→95722 | 1891.82 | 620948 | 6.428 | 2/5 |
| 4 | 51→96986 | 1916.81 | 236577 | 7.159 | 2/5 |
| 8 | 51→98799 | 1952.64 | 323367 | 5.308 | 1/5 |
| 16 | 51→97643 | 1929.80 | 8274421 | 5.744 | 1/5 |

**(c) damped** — under-relaxed residual, γ=0.5:

| N | norm 1st→last | growth× | mean PPL | entropy | degenerate |
|--:|--:|--:|--:|--:|--:|
| 1 | 167469→167469 | 1.00 | 20.58 | 1.697 | 0/5 |
| 2 | 167469→229041 | 1.37 | **17.89** | 1.769 | 0/5 |
| 4 | 167469→351812 | 2.10 | **18.19** | 1.876 | 0/5 |
| 8 | 167469→593326 | 3.54 | 35.93 | 2.000 | **0/5** |
| 16 | — | — | broken† | — | degenerate† |

†**damped N=16 is broken** and not cleanly tabulated: per-prompt teacher-forcing
already explodes (prompt0 norm 156315→1022183, PPL 79.4; prompt1 norm 172946→1104221,
PPL 409.2) and greedy decode is degenerate (prompt0 `looped=True`). The N=16 ×
24-step greedy decode (~255 layer evals/step) then **hard-crashes the CUDA context**
(exit 255/127, no Python traceback — Windows TDR / resource limit), so a 5-prompt mean
was not collected. The combo is already past breakdown at the teacher-forcing level,
so the crash does not change the conclusion.

## Findings

- **The two "hard renorm" fixes Probe 0 hypothesized — (a) rescale and (b) RMSNorm —
  both make things *catastrophically worse*, not better.** They fail from N=2: PPL
  jumps to hundreds → hundreds-of-thousands and entropy blows up (3.8–7.2 nats),
  with widespread greedy degeneration. Forcing the residual stream back to a fixed
  small magnitude every pass (rescale → ~25k, RMSNorm → RMS=1, i.e. ‖h‖≈√d≈51)
  **strips the very scale BitNet's downstream RMSNorm + int8 activation quant are
  calibrated for**. The accumulation *is* the signal; hard-normalizing it destroys it.
  (Their low/erratic "degenerate" counts at high N are an artifact — output is so
  flat/random that the repetition heuristic doesn't trip, not that decode is healthy;
  PPL of 10⁴–10⁶ is the real story.)
- **(c) damped is the clear winner.** γ=0.5 does *not* reset the scale; it just slows
  accumulation (norm growth roughly halved: 3.54× at N=8 vs 5.99× baseline). It keeps
  **PPL flat-or-improving and decode non-degenerate through N=8**: N=2 PPL 17.9 and
  N=4 PPL 18.2 both **beat baseline N=1 (20.6)**, N=8 is 35.9 with **0/5 degenerate**
  (baseline N=8: 150.3, 1/5). It extends the usable loop depth from baseline's N≈2–4
  out to **N=8**.
- **But damping only delays, not eliminates, the failure.** Residual norm still grows
  (just ~2× slower), and by **N=16 damped also breaks** (PPL exploding, degenerate
  decode). Under-relaxation buys depth; it is not an unconditional fix.

## Verdict

**A per-loop *renormalization* does NOT fix the failure mode — it is actively
harmful.** Both naive renorm forms hypothesized in Probe 0 (rescale-to-input-norm,
RMSNorm-between-passes) collapse quality from N=2 because they discard the residual
magnitude BitNet depends on. **Do not bake a hard between-loop renorm into the looped
design.**

**A per-loop *gated/damped residual* IS the right lever.** The under-relaxed update
(`h ← h_in + γ·Δ`, γ<1) is the only variant that holds the loop flat-or-improving and
non-degenerate through N=8 — it tames the inflation without destroying scale. It does
not reach N=16 at fixed γ=0.5.

**Recommendation for the trained looped design:** bake in a **learned per-loop gate
on the residual increment** — `h ← h_in + g·(slab(h_in) − h_in)` with `g` a learned
scalar (or per-channel vector), initialized around 0.5, optionally per-loop-step.
This is the trainable generalization of the damped variant that worked, and unlike a
fixed γ it can adapt the increment magnitude with depth (likely needed to push past
N=8). Explicitly **avoid** a fixed rescale or a weightless RMSNorm between loops. A
mild learned RMSNorm *with* a learned scale γ could be revisited, but only with a
learned scale large enough to preserve BitNet's native activation magnitude — the
weightless form tested here is disqualified.

## Caveats (carried from Probe 0, all still apply)

- **No KV cache;** single full-sequence forward, teacher-forcing PPL + short greedy
  recompute. **Single fixed slab [L//4:3L//4]; fixed 5 prompts.** **bf16 master,
  ternary-in-forward** (not the deployed I2_S GGUF kernel path).
- Position ids reused as `[0..seq)` every looped pass (slab re-sees the same RoPE
  positions each loop).
- **New 0b caveat:** the renorm variants are *zero-training* — applied to a network
  never trained to expect them. The hard-renorm collapse is partly that mismatch; it
  argues for a *learned* gate, not that normalization is fundamentally impossible. And
  damped N=16's GPU hard-crash is a Windows-TDR/resource artifact of the no-cache
  24-step deep decode, not a numerical result (the numerical breakdown is already
  visible in teacher forcing).
