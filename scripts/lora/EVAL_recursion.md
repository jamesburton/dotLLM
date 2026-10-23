# EVAL_recursion.md — Track-R Looped Ternary Depth: Honest Negative Result (#118)

**Date:** 2026-06-30
**Branch:** `issue/trackR-recursion`
**Base model:** `microsoft/bitnet-b1.58-2B-4T-bf16`
**Hardware:** RTX 3060 12 GB (CUDA, sm_86)

---

## Summary

Track-R tested whether retrofitting BitNet-b1.58-2B-4T with a looped recurrent slab — and
specifically whether a live-perplexity/entropy exit gate — yields adaptive-depth benefit over a
single forward pass. After two training runs (r1 and a control r2), the answer at this scale is
**no**. A properly-trained single-pass model is strictly better than a looped one. The apparent
benefit in r1 was a curriculum artifact, overturned by the r2 control. The gate mechanism is
mechanically correct; it simply has nothing useful to gate on.

---

## Experimental Setup

### Architecture

BitNet-b1.58-2B-4T (2 B parameters, ternary weights, 30 transformer layers) was retrofitted as a
looped (P, R, C) model:

- **Prelude P:** layers 0–6 (7 layers, run once)
- **Recurrent slab R:** layers 7–21 (15 layers, looped N times)
- **Coda C:** layers 22–29 (8 layers, run once)

A **learned per-loop residual gate** (scalar, init 0.5, sigmoid-constrained) was inserted at each
slab exit to attenuate the residual stream between passes. A **fusion adapter** (low-rank,
`r=16`) was placed at the prelude→slab and slab→coda boundaries to adapt the frozen base model
to the looped regime.

**Trained parameters:** slab FFN LoRA, fusion adapter, gate scalars. Attention weights frozen.
**Objective:** LM loss only. Knowledge distillation from a teacher was infeasible on the 3060
(teacher model does not fit alongside the student at training batch size).

### Training Runs

| Run | Curriculum μ_min | μ_max | n_max | Tokens | LM loss | Peak VRAM |
|-----|-----------------|-------|-------|--------|---------|-----------|
| r1  | 2.0             | 8.0   | 8     | 5.0 M  | 2.47    | 10.1 GB   |
| r2  | 1.0             | 3.0   | 4     | 5.0 M  | 2.36    | 10.1 GB   |

Curriculum-N: each training step samples N ~ Poisson(μ), clamped to [μ_min, n_max]. r1 never
sampled N=1 by design (μ_min=2). r2 was designed specifically to train N=1 properly as a control.

### Eval Protocol

30 sequences × 512 tokens = 15,330 tokens from `HuggingFaceH4/no_robots` test split.
Three criteria assessed (see `task-r5-report.md` for full methodology):

- **C1 — PPL vs fixed N:** does PPL decrease as N increases?
- **C2 — Hard-token benefit:** do high-entropy (hard) tokens gain more from extra passes?
- **C3 — Gate vs fixed-N:** does the live-entropy/margin gate match or beat the best fixed-N?

---

## Results

### PPL vs Fixed Recurrence N

| N  | r1 PPL        | r2 PPL        |
|----|---------------|---------------|
| 1  | 121.01 (\*)   | **14.06** (best) |
| 2  | 17.83         | 14.75         |
| 3  | **16.75** (best) | 16.28      |
| 4  | 17.01         | 19.62         |
| 5  | 17.88         | —             |
| 6  | 19.25         | —             |
| 7  | 21.52         | —             |
| 8  | 24.21         | —             |

(\*) r1 N=1 is catastrophic because μ_min=2 never exposed the fusion adapter to single-pass
operation. This is a training artifact, not a model property.

**r1 gate verdict: PASS** (PPL(N=8) < PPL(N=1) due to the broken N=1 baseline).
**r2 gate verdict: FAIL** (PPL increases monotonically with N; looping strictly hurts).

The decisive comparison: **r2's properly-trained N=1 (14.06) beats r1's best looped result
(N=3, 16.75) by 16%.**

### Hard-Token Benefit (C2)

| Metric | r1 | r2 |
|--------|----|----|
| Pearson r (entropy vs improvement) | **+0.226** | **−0.041** |
| p-value | < 1e-175 | 4.2e-7 |
| Very-hard vs easy mean improvement | +2.24 vs +0.94 | −0.39 vs −0.26 |
| Verdict | PASS | FAIL |

In r1, hard tokens appeared to benefit 2.4× more from extra passes than easy tokens. This
correlation was statistically significant at 15,330 tokens. In r2, the sign reverses: hard tokens
are *more* damaged by extra passes than easy ones. The r1 signal was driven by hard tokens
suffering more from the broken N=1 baseline — not by genuine benefit from looping.

### Gate vs Fixed-N (C3)

Both runs fail this criterion.

**r1** — the oracle gate (binary: easy tokens→N=1, hard tokens→N=8) produces PPLs ranging
from 34.5 (aggressive) to 82.1 (conservative), all far above the best fixed N=3 (16.75). Root
cause: the gate directs easy tokens to N=1 (PPL=121, broken), making every calibration
catastrophically worse than fixed N=3. The live-entropy/margin adaptive demo runs correctly
(mean 1.74 loops/token, range [1,8], loops vary across tokens), confirming the gate mechanism
is mechanically sound.

**r2** — oracle gate PPL 14.77–17.81, above the best fixed N=1 (14.06). No calibration matches
single-pass quality. Live adaptive demo: mean 2.26 loops/token, range [1,4]. Gate works;
there is simply no per-token benefit to gate on.

---

## Why r1 Looked Positive but Was an Artifact

r1 passed 2/3 criteria (C1, C2). The original interpretation — "looped ternary depth is partially
confirmed; the gate failed because of the N range, not the hypothesis" — was plausible at the
time. The r2 control disproves it:

1. **C1 in r1** passed only because the N=1 baseline was broken by μ_min=2 training. The model
   was never trained to run the fusion adapter at N=1, so N=1 PPL was 121 — an untrained
   failure mode, not a genuine single-pass result. N=2 fixing that breakage was mistaken for
   "one extra loop helps."

2. **C2 in r1** measured per-token improvement as `L_t(N=1) − L_t(N=8)`. Hard tokens showed
   larger values because hard tokens are the ones most catastrophically broken at N=1 (they
   have high entropy, so the degraded fusion adapter harms them most). Once N=1 is trained
   properly (r2), the correlation becomes −0.04: essentially null, with a slight negative slope
   meaning hard tokens are marginally *more* damaged by extra passes.

3. r2 converged to lower LM loss (2.36 vs 2.47) in the same number of steps — the curriculum
   covering N=1 trained the adapter to better overall quality.

---

## Conclusion

**Null result.** At 5M-token retrofit scale, a BitNet-b1.58-2B-4T model with a looped recurrent
slab shows no adaptive-depth benefit. The model specializes to whatever recurrence distribution
it was trained on. A clean, properly-trained single forward pass (N=1, PPL=14.06) is better than
any number of recurrent slab passes. The live-perplexity/entropy exit gate is mechanically correct
but has no quality signal to exploit.

The novel hypothesis — that hard tokens should benefit more from extra slab passes — is **not
confirmed** once the curriculum artifact is removed.

---

## Caveats — Why This Is Not the Final Word

1. **Scale.** 5M tokens is tiny. Recurrent-depth literature (Ouroboros, Universal Transformers,
   retrofit-recurrence on GPT-2/LLaMA) uses billions of tokens before emergent adaptive-depth
   behaviour appears. The 3060 at 0.3–0.6 steps/sec makes multi-billion-token runs infeasible.

2. **Single fixed slab.** The 15-layer recurrent slab is a fixed architectural choice. Models
   designed for recurrence from scratch (not retrofitted) may behave differently.

3. **LM-only training.** KD from a teacher that can express "this token is computationally hard"
   might convey a cleaner signal. That was infeasible on the 3060.

4. **Frozen attention.** Attention patterns for cross-pass information flow could not adapt.

5. **The gate mechanism is verified and reusable.** Correctness tests pass: n_max=1 produces
   greedy token-exact output, loop counts and determinism hold across calibrations. If a future
   experiment on larger hardware — with a base model that already benefits from looping — needed
   a live-entropy gate, the implementation in `recur_gate.py` is a sound foundation.

---

## dotLLM Engine Implication

**Do not wire looped-depth/recursion into the DotLLM.Engine on this evidence.**

The experiment was designed precisely to check whether this was worth building. It is not — not
at the scale we can test. Revisit only if:
- A base or fine-tuned model regime is identified where repeated slab passes demonstrably reduce
  per-token perplexity over a single pass (not an artifact of undertrained N=1), and
- Sufficient compute (>1B tokens of recurrence-aware training) is available.

The architectural interfaces (`IAttentionStrategy`, `IInferenceHook`) are sufficient to add
adaptive-depth looping later as a plugin. No engine changes are needed now.

---

## Artifacts

| Path | Description |
|------|-------------|
| `.docs/recursion/r1/eval.json` | Full eval results, r1 (curriculum N=2–8) |
| `.docs/recursion/r2/eval.json` | Full eval results, r2 control (N=1–4) |
| `.docs/recursion/r1/metrics.json` | Training histograms, r1 |
| `.docs/recursion/r2/metrics.json` | Training histograms, r2 |
| `scripts/lora/recur_eval.py` | Eval harness (3 criteria + adaptive demo) |
| `scripts/lora/recur_train.py` | Curriculum-N training script |
| `scripts/lora/recur_model.py` | Looped (P,R,C) model wrapper + gate |
| `scripts/lora/recur_gate.py` | Live-entropy/margin adaptive gate |

---

*Closes #118*

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
