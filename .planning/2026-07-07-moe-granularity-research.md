# MoE granularity/structure research — findings (2026-07-07)

Deep-research pass (104 agents, 22 sources fetched, 25 claims adversarially verified, 22 confirmed).
Source-quality caveat: OLMoE + DeepSeekMoE are the strongest (released models, task-accuracy ablations);
several grouping/sparsity findings are single recent preprints — promising, not settled.

## Verdicts by question

**Q1 — Per-layer FFN experts is the proven default, and it gives real TASK accuracy (not just PPL).**
OLMoE (64 experts/layer, top-8, 7B/1B-active) replaces every layer's FFN and beats all similar-active
models incl. larger dense ones on MMLU/HellaSwag etc. Alternate-layer placement also works. Routing
DEPTH is not a dominant knob — placement is flexible. [arXiv:2409.02060, 2604.07030]

**Q2 — The grouped "one router / one expert spanning a block of layers" is NOT demonstrated (would-be-novel).**
Existing "coarse" work shares PARAMETERS or ROUTER params across consecutive layers but KEEPS per-layer
routing: Megrez2 (cross-layer expert sharing), Expert-Tying (shared FFN, per-layer routers), PathMoE
(block-shared ROUTER params, per-layer experts). PathMoE reports gains + removes aux-loss need (single
preprint). CAUTION: "grouped/tied sharing is quality-neutral" was REFUTED twice (Megrez2 0-3, Expert-Tying
1-2) — layer-sharing tends to COST quality. So the grouped-4/8 idea is novel + cheap to probe, but the
evidence leans against it being free. [2507.17728, 2606.16825, 2603.18297]

**Q3 — Many small fine-grained experts > few large, on loss AND accuracy, with steep diminishing returns.**
OLMoE: 8→32 experts ≈ +10% (HellaSwag/MMLU); 32→64 ≈ +1-2%. DeepSeekMoE fine-grained segmentation matches
LLaMA2-7B at ~40% compute. "Granularity" is a compute-optimal axis. → design ~32 small ternary experts.
[2409.02060, 2401.06066, Ludziejewski ICML24]

**Q4 — Shared/always-on expert is config-dependent.** DeepSeek/Qwen: helps (isolates common knowledge).
OLMoE: slightly WORSE — reserving a shared slot collapsed ~90% of routing combinations. Test, don't assume.
[2401.06066, 2409.02060]

**Q5 (MOST decision-relevant) — MoE expert FFN weights are FAR more robust to ultra-low-bit quant than dense.**
MoQE: 2-bit experts (QAT) BEAT the dense baseline; 3-bit dense FFN dropped ~23%, 2-bit dense ~zero. Strong
support that FULL ternary FFN experts are viable where a dense ternary FFN collapses. BUT: only 2-bit (not
1.58/ternary), encoder-decoder MT (not decoder 8B). Router LOGITS are the fragile part → keep router
higher-precision. No demonstrated ternary-MoE / BitNet-MoE / sparse-upcycle-to-ternary task accuracy exists
— our target is genuinely novel territory. [2310.02410]

**Q6 — PPL-optimal is NOT accuracy-optimal (this is the literature-level confirmation of OUR finding).**
Controlled studies: reasoning accuracy (GSM8K/HumanEval) needs DENSER routing / more active compute at
matched loss; memorization + PPL improve with sparsity; "optimal sparsity set jointly by active FLOPs and
tokens-per-parameter, not by minimizing loss." Broad-mix validation loss hides specialization; gains show
on SPECIALIZED sets. → our +16% PPL / flat GSM8K is a KNOWN pattern, not a fluke. Fix: select on accuracy,
use denser routing / more active compute for reasoning. [2508.18672, 2604.07030]

**Q6 router/data —** Balancing SCOPE (tokens/sequences balanced over) matters more than aux-loss method;
wide scope + sufficient tokens-per-expert → specialization not memorization. Upcycling gives a head start
that is NOT durable (from-scratch catches up ~500B tokens) — attractive only in the LOW-additional-compute
regime (which is ours). [2604.07030, 2409.02060]

## Refuted (don't rely on these)
- Grouped/cross-layer expert SHARING is quality-neutral — REFUTED (Megrez2 0-3, Expert-Tying 1-2).
- Reasoning quality is just top-k — REFUTED 0-3; active-parameter budget is the driver.

## Open questions (no direct evidence — our experiments would be first)
1. Do full ternary per-layer FFN experts recover TASK accuracy where ternary low-rank LoRA did not?
2. Does the coarse grouped (one expert / block of layers) design hold up vs per-layer?
3. Tokens + balancing scope for ternary experts to specialize (not memorize) before the from-scratch crossover?
4. Shared expert in a fine-grained TERNARY regime — help or hurt?

## Implications for our direction (see synthesis)
- Our adapter negative is EXPECTED (PPL-not-accuracy is known); wrong tool + wrong objective.
- Full ternary FFN experts, per-layer, per-token, ~32 fine-grained (+shared TBD), wide balancing scope,
  higher-precision router, ACCURACY-based selection — this is the evidence-backed bet, and matches dotLLM
  Track-M kernels. Genuinely novel at ternary; MoQE+OLMoE give real reason to expect it differs from adapters.
