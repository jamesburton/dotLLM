# MoTE accuracy experiment — spec (2026-07-07)

## Key realization
dotLLM ALREADY has the full-FFN-expert MoTE pipeline (Track-M): `scripts/lora/mote_upcycle.py`
(dense→N full-FFN routed experts + BF16 router + optional shared, identical-init), `mote_train.py`
(KD-from-dense-teacher + Switch load-balance aux), `mote_eval.py` (entropy/PPL/router-dependence), plus
C# CPU/CUDA/Vulkan MoE inference and a Kaggle ablation grid. This is the research-recommended design.
Our ternary-LoRA thread was a weaker parallel (low-rank adapters). **We pivot to USING Track-M.**

## The gap this experiment closes
Track-M was validated on PPL + expert-entropy, NOT task accuracy — the exact metric the research
(2508.18672, 2604.07030) says misleads. Nobody has accuracy-tested the real MoTE. That is the decisive
cheap experiment: **does a per-layer full-ternary-FFN-expert MoTE add TASK ACCURACY (GSM8K/HumanEval)
where ternary low-rank adapters did not (2B: 47.4%==47.4%)?**

## Design (from research 2026-07-07 + Track-M defaults)
- **Per-layer full-FFN experts** (mote_upcycle) — proven design (OLMoE). Layers: start upper-half
  (Track-M's 24-29 on 2B) or all; placement is flexible.
- **Expert count / granularity**: research says finer=better with steep diminishing returns
  (8→32 ≈ +10%, 32→64 ≈ +1-2%). Track-M grid does 4/8. START at 8 experts top-2; have a 16/32 arm ready.
- **Shared expert**: config-dependent (DeepSeek +, OLMoE −, collapses routing combos). Test {none, 1
  shared} as an arm, don't assume.
- **Router**: BF16 first (research: router logits are the FRAGILE part under quant → keep higher
  precision). Then the ROUTER PRECISION-ANNEAL (below) for a fully-ternary endpoint.
- **Load-balance**: Switch aux (present). Research: balancing SCOPE matters more than the formula; use
  wide scope + enough tokens-per-expert so experts specialize, not memorize.
- **Upcycle from the fixed dense BitNet base** — fine in our LOW additional-compute regime (head-start
  dominates before the from-scratch crossover ~500B tokens; we're far below that).

## Router precision-anneal / KD-to-ternary (the user's idea — the fully-ternary path)
Missing piece per the code map. Add to `mote_train.py`:
1. Phase A: train MoTE with BF16 router + ternary experts until routing is good + experts specialize
   (accuracy-selected).
2. Phase B: anneal the router BF16→ternary via STE (a λ-ramp on the router weights, mirroring our expert
   λ-ramp), with a KD term: student(ternary router) logits distilled toward the frozen teacher(BF16
   router) logits (soft top-k distribution), experts fixed or lightly co-adapted.
3. Report: routing agreement (ternary vs bf16 router argmax match), accuracy, entropy — does fully-ternary
   hold quality? Router is tiny (~hidden×n_experts×n_layers) so this is cheap.

## THE MISSING TOOL: accuracy eval on a MoTE model
mote_eval.py does PPL/entropy only. Build/adapt an ACCURACY eval that loads a trained MoTE (MoTEBlock/
MoTEShim structure) and runs GSM8K/HumanEval generation (reuse tlora_capability_hard scorers + the
enable_inference_cache 11.6x speedup; note MoTE forward returns tuples via MoTEShim, and experts are
AutoBitLinear so the cache patch applies to them). This is the one real build.

## Training protocol (bakes in the overfitting + divergence lessons)
1. SMOKE FIRST: a few-step upcycle+train run before any full run (the 8B adapter diverged at full ternary
   with AdamW lr 1e-4 and wasted ~1.5h — the gate caught it but late). Watch loss stability at lam=1.
2. LR: lower than the 2B adapter used; add grad clipping; shorter/gentler λ-ramp if needed. Tune on smoke.
3. SELECT ON ACCURACY, never PPL. Add a small held-out ACCURACY check during training; early-stop / pick
   best-accuracy checkpoint. Report train-PPL, held-PPL, held-ACCURACY, expert-entropy together.
4. FORMAT-MATCHED data (GSM8K-CoT for GSM8K eval; completion data for HumanEval) — no Q&A→completion
   mismatch (that was the code-expert failure).
5. Monitor expert-entropy (dead-router) + per-expert token counts (balance).

## Plan
- **P1 (fast):** check for existing trained Track-M MoTE checkpoints (kaggle-results branch / .docs/mote)
  — if any, ACCURACY-eval one directly (fastest possible answer).
- **P2:** upcycle 2B → 8-expert top-2 MoTE, heal-train (KD + aux, accuracy-selected, format-matched,
  smoke-first) on a domain with headroom (GSM8K-CoT), ACCURACY-eval vs base. THE decisive test.
- **P3:** if it adds accuracy → (a) shared-expert arm, (b) finer-grained (16/32) arm, (c) router
  precision-anneal to fully-ternary, (d) scale to 8B.
- **P4 (optional novelty):** grouped-4/8 "one expert per block of layers" — research says UNBUILT and that
  layer-sharing tends to cost quality; run only as a cheap side-probe, not the main path.

## Success metric
2B MoTE GSM8K accuracy vs 2B base (base ≈ 45% at this eval). Adapters gave +0.0. If the full-FFN MoTE
shows a clear positive delta beyond noise, the capability case is real and we scale + go fully-ternary.
