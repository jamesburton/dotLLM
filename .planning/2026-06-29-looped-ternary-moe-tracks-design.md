# Growing BitNet capacity — Track R (recursion) + Track M (MoTE), parallel single-axis experiments

**Date:** 2026-06-29
**Status:** design approved (2026-06-29). Spec for TWO parallel single-axis experiments. The combined **looped ternary MoE** is a deliberate follow-on spec, written only after both tracks show signal.
**Origin:** verified deep-research + 3 parallel agent briefs (recursion / MoTE / diffusion) + two zero-train spikes (Probe 0, Probe 0b). Durable record in handoff + memory `[[task-lora-stacking]]`.

## Goal
Test, on a single RTX 3060 (12 GB) from the **Microsoft BitNet b1.58-2B-4T** base, whether ternary-model capacity/capability can be grown by (R) **adaptive looped depth** and (M) **ternary MoE upcycling** — each a novel-on-ternary direction. Then converge them into looped ternary MoE.

## Why these two (validated foundations)
- **Ternary MoE is mechanically sound on CPU:** kernel `moe_indexed_matmul_i2_s` (#116, branch `issue/i2s-moe-kernel`, commit `8fe276f`, **PR held**) passes its numerical gate (N identical I2_S clones == dense I2_S; per-expert scale honored) and is purely additive (0 deletions → dense path untouched).
- **Ternary recursion is stable:** Probe 0 (spike `9b25c64`) — looping a ternary slab does not blow up; **N=2 improves PPL 20.6→17.0**, no degeneration; residual norm grows ~linearly (not exponential), inflating only at N≥4 and breaking decode at N=8. Probe 0b is validating the per-loop renorm that fixes that.
- **Off the table:** diffusion. Weight-generation that works at LLM scale is feedforward hypernets (Text-to-LoRA, Drag-and-Drop), FP-LoRA only, ternary speculative; the masked-diffusion *objective* is ~2–3× slower and only wins in low-data. Not pursued here.

## Decisions (locked, approved)
- **Decompose:** two parallel single-axis tracks now; **looped ternary MoE = separate later spec.**
- **Venue:** Python-first (HF transformers + the existing `scripts/lora/` QLoRA heal-train harness). dotLLM serving deferred until a track wins (kernel #116 already covers MoTE serving).
- **Base:** bf16 `microsoft/bitnet-b1.58-2B-4T-bf16` master — its BitLinear forward applies ternary absmean weight quant + int8 STE activations, so training on it *is* the ternary path (train==serve faithful).
- **Loss:** LM + **KD** = KL to the frozen dense BitNet's logits. KD weight ≈ 0.5.
- **N_max (recursion cap):** 8 (from Probe 0/0b).
- **Eval:** reuse the U2/U5 harness — val-PPL + `scripts/lora/eval_*.py` (tool-use/coding/instruction) — plus per-track metrics below.

## Track R — recursion-only retrofit
**Topology (demonstrated path, arXiv:2511.07384):** prelude `[0:P)` / looped recurrent slab `[P:Q)` / coda `[Q:L)`, with a linear adapter `2h→h` fusing prelude output + recurrent state each pass. Default slab ≈ Probe 0's `[7:22)` of 30 layers (sweepable).

**Per-loop renorm — RESOLVED by Probe 0b (commit f6c08ee):** use a **learned per-loop residual gate** `h ← h_in + g·(slab(h_in) − h_in)`, `g` init ≈ 0.5 (scalar or per-channel, optionally per-loop-step). 0b showed **fixed** renorms are *harmful* (rescale-to-input-norm and weightless RMSNorm collapse quality from N=2 — they discard the residual magnitude BitNet's downstream RMSNorm + int8 activation quant expect); the **fixed damped** version (γ=0.5) holds PPL flat-or-improving + decode clean through N=8 (N=2/4 beat N=1) but breaks by N=16. The learned gate is the trainable generalization of damping, aiming to extend past N=8.

**Training:** retrofit the bf16 master; **curriculum-N** (recurrence count sampled Poisson-Lognormal, mean μ ramped up to N_max over training); QLoRA on the recurrent slab + the fusion adapter + the FP renorm/gate params; freeze prelude/coda/attention/embeddings; + KD.

**Novel contribution (the headline):** an **inference-time live-perplexity/entropy exit gate** — loop while the current-token signal (output entropy, or logit top-1−top-2 margin) is above a threshold, up to N_max; stop early when confident. This gates adaptive compute on a *measured* signal with **no learned gate** — unclaimed in the literature (Ouro learns λ; the retrofit line samples N and calls adaptive compute future work).

**Success criteria:**
1. Val-PPL **decreases with recurrence** up to the training max, ideally extrapolating one step (N=train_max+1 still helps).
2. **High-entropy tokens benefit more from extra loops than easy ones** — per-token loss-improvement `Iₜ = L_{t−1}−L_t` correlates with token difficulty. This is the direct test of the hypothesis.
3. The **live gate** matches/exceeds fixed-N quality at lower *average* recurrence (compute saved on easy tokens).

## Track M — MoTE upcycle ablation
**Upcycle recipe (MoTE arXiv:2506.14435 + Sparse Upcycling arXiv:2212.05055):** clone the dense BitNet FFN into N routed experts (identical at init — routing breaks symmetry, no noise); fresh **BF16 router** (init σ=0.02); freeze base (and the shared expert when it is FP); **Switch load-balance aux loss γ=0.01** (do NOT normalize router combine weights — hurts upcycled LMs); lr 1e-4 on the MoE path; ~0.3–0.5B-token heal + KD. Convert the **upper ~half** of the 30 blocks first.

**Ablation grid (ordered; each cell = one heal run):**
| # | experts | top-k | shared expert | purpose |
|--|--|--|--|--|
| 0 | 4 ternary | 1 | **ternary** | **fast-fail** — cheaper anchor, or confirm FP anchor is needed |
| 1 | 4 ternary | 1 | **FP (BF16)** | MoTE-faithful baseline |
| 2 | 4 ternary | 2 | **FP (BF16)** | top-2 mixing |
| 3 | 4 ternary | 1 | **none** | does the FP anchor earn its cost? |
| 4 | 4 ternary | 2 | **none** | top-2, no anchor |
| 5 | **8 ternary** | best of 1–4 | best of 1–4 | capacity scaling |

**Per-cell metrics:** expert-utilization entropy (collapse check — primary go/no-go), val-PPL vs the dense BitNet baseline, router output input-dependence (not constant argmax), active-params/FLOPs. Expect *recovery* to ~dense at this budget, with specialization/headroom as the upside; text-domain ternary-MoE quality is unverified (MoTE's demonstrated domain is multimodal), so cell #0 fast-fail and the entropy metric are the early signals.

**Serving:** kernel #116 (CPU, validated) serves the winning config. CUDA + Vulkan `moe_indexed_matmul_i2_s`, the loader for I2_S expert banks + per-expert scale tensors, and the MoTE block wiring (frozen FP shared FFN + MoE-sum via the `DispatchGemm` I2_S case) are **follow-up issues**, built once a cell wins.

## Convergence (OUT OF SCOPE — its own later spec)
Once R and M each show signal: **looped ternary MoE** = the winning Track-M block placed inside the Track-R recurrent slab, with per-loop **re-routing** on the evolved hidden state, a **cross-loop diversity aux-loss** (penalize re-selecting the same expert across consecutive loops; loop-step embedding into the router) to convert "loop more" into "touch more experts," plus the live-perplexity gate. Both agents flagged the two empirical unknowns to settle there: per-loop router diversification vs fixpoint collapse, and cross-loop load balance.

## Components & boundaries
| Unit | Responsibility | New vs reuse |
|------|----------------|--------------|
| `scripts/lora/train_task_lora.py` (+ KD option) | heal-train loop + KD term | extend |
| Track-R retrofit module (Python) | `(P,R,C)` surgery, fusion adapter, per-loop renorm, curriculum-N | new (spike-derived) |
| Track-R live exit gate (Python, inference) | entropy/margin-gated looping to N_max | new |
| Track-M upcycle module (Python) | clone FFN→experts, FP router, Switch aux | new |
| Eval (`scripts/lora/eval_*.py`, val-PPL) | quality + per-track metrics | reuse |
| `moe_indexed_matmul_i2_s` (#116) | serve ternary experts | done (CPU); GPU/loader/block = follow-up |

## Risks
- Renorm form for Track R — resolved by Probe 0b (in flight).
- Track-M expert collapse at low heal budget — mitigated by Switch aux + entropy monitoring + cell #0 fast-fail; KD counters capacity loss.
- Text-domain ternary-MoE quality unverified — the ablation IS the test; honest expectation is parity-class.
- Compute: Track M = ~6 sub-1B-token heal runs; sequence them; confirm 3060 idle / coordinate GPU before each (user has a parallel GPU workload).

## Workflow
Issue-driven; one issue/branch/PR per track (Track M's serving follow-ups are separate issues). Commits ref the issue; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Spikes (`probe0-recursion`) stay throwaway. Each heal run gated on its success metrics before proceeding.
