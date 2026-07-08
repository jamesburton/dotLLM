# Identity-expert routed-MoTE + supervised routing — design note (2026-07-08)

Campaign: **trackM-mote**. Supersedes the FFN-expert-in-existing-layers / learned-routing
MoTE (which produced a robust NULL at 2B — see `.docs/handoff.md`). This note describes the
code delivered this session (BUILD-ONLY; GPU runs launched separately).

## Problem being fixed
The prior MoTE campaign added **no task accuracy** at 2B. Two diagnosed failure modes:
1. Converting an *existing* FFN layer to MoTE gives the model no "do nothing" option — it can
   only match or degrade the base.
2. Learned (unsupervised) routing left experts **homogeneous** (73% identical predictions).

## Fix = three ingredients
1. **Depth expansion (LLaMA-Pro, ternary).** `scripts/bitnet_depth_expand.py` inserts
   zero-residual identity blocks (o_proj + down_proj zeroed) → 2B becomes ~4.5B of NEW capacity,
   model logits **bit-for-bit unchanged at init** (already validated: max_logit_diff=0.0000).
2. **Each inserted block's FFN → IdentityMoTEBlock** = `[skip expert] + K capability experts + router`.
   The **skip expert** (index 0) has `down_proj == 0` and is **frozen** → always outputs 0 →
   routing a token to it = "skip this inserted layer" = exact base path. This is the permanent
   **no-regression fallback** that fixes failure mode 1.
3. **Supervised routing.** Each task's data is tagged with a routing label and the router is
   trained with a cross-entropy loss to send that data to ITS OWN expert (math→expert 1,
   instruction→expert 2, tool→expert 3). This *forces* specialization → fixes failure mode 2.

## The identity-MoTE layer (`scripts/lora/identity_mote.py`)
`IdentityMoTEBlock(experts, router, top_k=1)`:
- `experts[0]` = skip/identity expert (frozen, `down_proj==0`).
- `experts[1..K]` = K trainable capability experts (full BitNet ternary FFNs / BitLinear-consistent).
- `router` = `nn.Linear(hidden, K+1, bias=True)`, **top-1**. Weight zero-init, `bias[0] = router_identity_bias`.
- forward → **plain tensor** (drop-in for HF `self.mlp(x)`); stashes `block.last_logits`
  (pre-softmax router logits, in-graph) and `block.last_counts` (detached) for the trainer.

### Identity-at-init invariant (THE critical property)
- **`capability_init="zero"` (default):** every expert (skip AND capability) has `down_proj` zeroed →
  the FFN branch is 0 **for every token regardless of routing** → the inserted block is an exact
  identity for an **arbitrary** router. Bulletproof. Capability experts keep the template's real
  `gate_proj/up_proj` (warm start) and grow `down_proj` away from 0 during training (per-expert
  LLaMA-Pro). Verified: `max_logit_diff == 0.0` under both a skip-favoring AND a random router.
- **`capability_init="template"` (experimental warm-start):** capability experts restore a
  neighbouring layer's full real FFN (real `down_proj`). Identity then holds only because the router
  is initialised to route every token to the skip expert (`router_identity_bias` large, e.g. 30).
  Verified: `max_logit_diff == 0.0` (skip expert output is exactly 0).

`assert_identity_at_init(base, mote, ids, randomize_router=...)` returns
`{max_abs, mean_abs, argmax_match, ok}` and is used by the test + trainer gate.

`build_identity_mote(model, inserted_indices, n_capability_experts, capability_init, ...)` converts
each inserted block in place and records `model._identity_mote_layers`. `inserted_indices` comes from
`bitnet_depth_expand.expand_model`'s `info["inserted_indices"]` (an **additive** field added this
session — does not change existing behaviour; existing depth-expand tests still pass 5/5).

## Supervised-routing training (`scripts/lora/identity_mote_train.py`)
Per step (one labeled sequence `(seq, label)`):
```
logits     = model(seq)                                   # routed full-model forward
lm_loss    = CE(logits[:-1], seq[1:])                     # LM on the routed output
route_loss = mean_over_layers CE(block.last_logits, label)# supervised routing (label broadcast to all tokens)
loss       = lm_loss + route_weight * route_loss
```
- Freeze base; unfreeze **routers + capability experts only** (skip expert stays frozen; inserted
  attention frozen-at-zero unless `--train-inserted-attn`).
- Separate LRs: `--router-lr` (default 1e-3, router must move logits fast) vs `--lr` (experts).
- `--grad-checkpoint` for the 12 GB 3060 (non-reentrant + `enable_input_require_grads`). `use_cache=False`
  throughout (also sidesteps the KV-cache "N vs N-1" layer-count mismatch on the changed stack).
- Identity-at-init is **verified before training** and **refuses to train** if broken (the whole
  no-regression guarantee rests on it). Zero-init additionally re-checked under a random router.
- Optimizers: `adamw` (default), `adafactor`, `adamw8bit` (bnb with runtime verify + adafactor fallback).

## Task-labeled corpus (`scripts/lora/multitask_routed_data.py`)
`build_routed_corpus(tokenizer, capabilities, ...) -> (sequences, labels, label_map)`.
- Emits `(token_ids[seq_len], routing_label>=1)`; **label 0 reserved** for the skip expert.
- Capabilities → labels 1..K in requested order, **compacted over the caps that actually load**.
- Loaders: `math` = openai/gsm8k CoT (`"Question: ..\nAnswer: ..#### N"`); `instruction` =
  no_robots and `tooluse` = hermes glaive (reuse `capability_data.py`); `coding` =
  python_code_instructions (cached stand-in axis).
- A capability whose dataset is missing is **skipped with a warning** (graceful) so dev/smoke runs
  work on whatever is cached. `--tiny-random` yields synthetic ids+labels (no downloads).

## Local cache status (checked this session, offline)
Cached: openai/gsm8k, HuggingFaceH4/no_robots, iamtarun/python_code_instructions_18k_alpaca,
microsoft/orca-math, TIGER-Lab/math_instruct, openai_humaneval.
**NOT cached: NousResearch/hermes-function-calling-v1 (tool-use)** — stage it (like orca-math was
staged) before the tool-use arm, or substitute `coding` as the third capability.

## Smoke-test results (CPU, tiny synthetic BitNet, `CUDA_VISIBLE_DEVICES=`)
- `scripts/test_depth_expand.py` — 5/5 pass (additive `inserted_indices` change is non-breaking).
- `scripts/test_identity_mote.py` — 5/5 pass. Identity `max_abs == 0.0` under favoring AND random
  router (zero-init) and under template-init with skip bias.
- `identity_mote_train.py --tiny-random` — identity@init 0.000e+00 (both routers); one+ steps run
  end-to-end; route CE drops from ln(K+1) toward ~0.26; all K+1 experts receive dispatch; GATE PASSED.
- `multitask_routed_data.py` on real cache — math/instruction/coding load, tooluse gracefully
  skipped, label_map compacted, shapes/labels correct.

## Open questions for the GPU phase
- **Route weight & router LR** to balance LM vs specialization without the router collapsing to skip.
- **Where to insert** (`--every 4` interleave vs upper-half only) and **K** (one expert per capability
  vs a shared "general" default expert at index 0 fed a general-web slice).
- **capability_init**: bulletproof `zero` (smooth grow) vs `template` warm-start (faster but needs
  the identity bias) — try `zero` first.
- Whether to also lightly train inserted attention (`--train-inserted-attn`) once FFN experts are set.
