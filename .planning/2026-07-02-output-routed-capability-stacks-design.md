# Output-Routed Capability Stacks — Design Spec

**Date:** 2026-07-02  
**Issue:** #117 (MoTE Track-M / Track-R)  
**Status:** Go-forward design; motivating experiments complete, first validation running  
**Author:** James Burton (with Claude)

---

## Background and Motivation

Track-M (MoTE — Mixture-of-Ternary-Experts) upcycled `microsoft/bitnet-b1.58-2B-4T` by cloning the
final four FFN layers into 4 routed ternary experts while freezing everything else (embeddings,
attention, final norm, `lm_head`). After fixing a gate-normalization bug (commit `a74bab3`) and
running 10M tokens of LM-only training on Kaggle T4×2, both runs converged to dense parity on the
chat eval set (MoTE PPL 13.45, dense 13.44; delta +0.01, within noise). A KD run at `kd_weight=0.5`
produced the same uniform routing histogram and no PPL improvement. The architecture is sound and
the routing is health (H/log(N) = 0.9999, router_dependence = 1.0, no collapse). The mid-FFN MoTE
route is not wrong; it is complete for what it can show. This spec describes the next step.

---

## Validation Results (2026-07-02)

The Step 0 experiment (described in §5) has now completed. Results confirm the output-layer
bottleneck hypothesis.

### A/B comparison — frozen vs. trainable `lm_head`

| Condition | JA PPL (step 0) | JA PPL (step 500) | JA PPL (step 1000) | EN PPL |
|-----------|-----------------|--------------------|--------------------|--------|
| Dense base (reference) | 74.4 | — | — | 13.44 |
| FFN-only MoTE (frozen `lm_head`) | 79 | ~79 (flat) | ~79 (flat) | ~13.44 |
| + trainable `lm_head` (grad-checkpointed) | 74.4 | **69.5** | **61.7** (falling) | ~14.6 |

**FFN-only:** JA PPL was flat at ~79 across all steps — worse than the dense 74.4 baseline —
confirming that tuning the FFN interior cannot project through an unchanged output head.

**Trainable `lm_head`:** JA PPL fell monotonically from 74.4 → 69.5 (step 500) → 61.7 (step 1000),
still descending at run termination. EN PPL rose mildly to ~14.6 (8.6% above base) but was stable —
the EN capability is not destroyed, only slightly perturbed.

**Conclusion:** The output layer was the bottleneck. Training `lm_head` unlocks new-capability
learning that FFN-only upcycling cannot provide. This is the direct in-domain replication of
the vocabulary-expansion literature (EEVE-Korean and others).

### Memory limit — 3060 12 GB

The run stalled after step ~1180. Root cause: the 128k-vocab `lm_head` in bf16 (~256 MB weights)
generates ~12 GB of gradient + optimizer state at the batch sizes needed for stable training.
The RTX 3060 12 GB cannot sustain this. **Full `lm_head` training requires ≥16 GB VRAM.**
Suitable hardware: Kaggle P100 (16 GB), Kaggle T4 (15 GB with head on its own partition),
Kaggle T4×2 with teacher-split, or a 24 GB consumer card. The Strix Halo (128 GB unified)
can run it in CPU-offload mode but CUDA-based PyTorch training is unproven there.

### Impact on the design

The bottleneck confirmation removes the gate condition on Step 0 and advances the experiment sequence
to Step 1 (single capability stack, bf16). The memory limit on the 3060 means the JA stack training
must be done on a bigger GPU (see §5 and the companion EXPERIMENT-PLAN.md). Prose and tool-call
stacks — whose `lm_head` gradient is bounded by the same 128k vocab but on smaller batches with
stronger base-teacher signal — may still fit the 3060 at reduced batch size; this should be probed
before committing to a cloud run.

---

## 1. Diagnosis: Why Mid-FFN MoTE Cannot Add New Capabilities

Two independent root causes explain the result, both backed by the research evidence.

### 1a. Ternary adds no free in-distribution capacity without a teacher

Deep-research (wm1og66ic) synthesized 24 confirmed claims from primary sources on ternary training
methodology. The key result:

> "Ternary/1.58-bit training adds no free in-distribution capacity: LM parity with 16-bit typically
> requires roughly doubling hidden size (encoder hidden 192 ternary to match 96 in 16-bit; 12M-param
> LM ppl 60.0–68.2 ternary vs 36.7 in 16-bit at equal size)." (arXiv:2411.05882, arXiv:2407.09527)

Cloning the base FFN into ternary expert copies does not produce new representational capacity — it
produces K copies of the same ternary FFN. Without (a) added effective width (doubling hidden size),
or (b) a higher-precision teacher providing gradient signal the ternary student cannot compute for
itself, or (c) a hard OOD specialization signal, the experts converge to the dense baseline.

Confirming evidence from the same thread:

> "Naive direct QAT-to-ternary of an existing full-precision LLM is unstable and fails to preserve
> performance, and the gap WIDENS with scale (13.9% at 0.6B to 15.3% at 4B) unless a supporting
> distillation pipeline is used." (arXiv:2510.13998 — Microsoft BitNet Distillation, Oct 2025)

We trained LM-only, no teacher, cloned experts — the exact configuration this warns against.

The demonstrated recipe that works is: train in 16-bit for ~20–40% of the run, then transition to
1.58-bit QAT with a straight-through estimator, gradual lambda warmup, and SubLN/extra-RMSNorm. A
fine-tuned ternary model reaches WikiText PPL 12.2 using this approach vs. 26 for best
from-scratch ternary (HF blog: 1_58_llm_extreme_quantization; arXiv:2502.11895). With KD from a
full-precision teacher, ternary students reach FP parity on MNLI at 0.6B (88.01 vs. 88.17) and 4B
(91.48 vs. 91.40) (arXiv:2510.13998, TernaryBERT arXiv:2009.12812).

**Track-M summary:** No capacity gain is the correct result, not a bug. The base is broadly strong
(code/math reversed: Python 3.75 PPL, C# 4.22, math 3.67–3.98 vs. chat 13.44); only German
(18.42, 1.37x) and Japanese (74.37, 5.53x) show real data-distribution gaps.

### 1b. A frozen `lm_head` is a hard output-side ceiling for new languages

The cross-lingual probe (EVAL_mote.md `cross-lingual spike`) shows:

| Language | Dense PPL | Ratio vs chat |
|----------|-----------|---------------|
| chat | 13.44 | 1.00x |
| German (wiki_de) | 18.42 | 1.37x |
| Japanese (wiki_ja) | 74.37 | 5.53x |

Japanese is 4x harder than German and 5.5x harder than chat. The FFN-only MoTE was unable to reduce
this: JA PPL stayed flat at ~79 even with load-balancing disabled (brakes-off) — *worse* than the
dense 74.37 baseline, because tuning the FFN interior cannot project new tokens through an unchanged
output head.

The vocabulary expansion literature confirms this is a structural constraint:

> "Newly introduced tokens require embeddings in both the input embedding matrix and the language
> model head (decoder matrix). For frozen-backbone vocabulary extension, only the embedding and
> un-embedding (lm_head) are learned with a high learning rate."
> (arXiv:2402.14714 EEVE-Korean, arXiv:2406.11477, arXiv:2604.16656)

> "A frozen shared lm_head is a hard capability ceiling for new languages."

EEVE-Korean achieved strong Korean performance within 2B tokens — specifically by training the
`lm_head` and input embeddings, not mid-layer FFN. Our JA failure (74→79 flat with frozen `lm_head`)
is the direct in-domain replication of this theoretical prediction. If retraining JA with `lm_head`
unfrozen brings the gap down (experiment running as of 2026-07-02), the output-layer bottleneck
hypothesis is confirmed.

---

## 2. Architecture: Output-Routed Capability Stacks

### 2a. Core idea

Instead of routing between mid-FFN experts (which cannot add output-side capability), route the
**output** itself. K "top stacks," each consisting of:

- A full transformer block's FFN (the final block, bf16)
- A final RMSNorm
- A dedicated `lm_head` projection (bf16)

sit above the frozen ternary BitNet-2B backbone. At inference, exactly one stack is active per
sequence, selected by a lightweight router. The backbone is never updated.

```
Input tokens
    │
[Frozen ternary BitNet-2B backbone, layers 0–28]
    │  hidden state h ∈ R^(seq × d_model)
    ├──────────────────────────────────────────┐
    │                                          │
    │    Router (MLP on h at commit token)     │
    │    ─────────────────────────────────     │
    │    → selects stack k ∈ {0 … K-1}        │
    │                                          │
    ├─ Stack 0: FFN_0 + Norm_0 + lm_head_0  (prose)
    ├─ Stack 1: FFN_1 + Norm_1 + lm_head_1  (tool-call / JSON)
    ├─ Stack 2: FFN_2 + Norm_2 + lm_head_2  (Japanese)
    └─ Stack k …
```

Each stack is independently trained on type-specific data. The backbone is the universal shared
representation layer; the stacks are the per-capability output surfaces.

### 2b. Router: format-prefix commit routing

The key causal constraint is that the output type is not known before the first token is generated.
Three solutions exist (prompt-level prediction, format-prefix token, soft per-token mixture — see
`research-output-routing.md §5`). The recommended mechanism for dotLLM is **format-prefix commit
routing**:

1. Generation begins with the default (prose) stack.
2. The model is trained to emit a special structural token at the first type-commitment point:
   `<tool_call>`, `{`, `[`, a Unicode language-onset character, or a domain-specific sentinel.
3. On observing this token, the router reads the hidden state `h_t` and commits to one stack for
   all remaining positions `t+1 …`.
4. All subsequent logits come exclusively from the committed stack's `lm_head`.

This is:
- **Causally valid** — the router sees a real, already-generated token, not a prediction.
- **Zero-latency on subsequent tokens** — one routing decision, then a pure stack forward pass.
- **Already aligned with dotLLM's constrained-decoding FSM** — the format-prefix token IS the FSM
  trigger (e.g. `{` triggers JSON schema masking). The same structural token activates both the
  output stack and the constrained-decoding mode, requiring no additional trigger logic.

This tight integration makes format-prefix routing a natural extension of the existing swappable
task-LoRA design: per-capability LoRAs (U0–U5) become per-capability top stacks.

### 2c. Memory budget — RTX 3060 12 GB

| Component | Params | Memory (bf16) |
|-----------|--------|---------------|
| Frozen BitNet-2B backbone | ~1.9B ternary | ~380 MB |
| K=3 dedicated lm_heads (prose, tool, JA) | 3 × ~128M | ~768 MB |
| K=3 final FFN blocks (bf16) | 3 × ~50M | ~300 MB |
| Router MLP | ~1M | ~2 MB |
| **Total stack overhead** | | **~1.45 GB** |
| KV-cache (32K context, 4-bit) | — | ~1–2 GB |
| **Total VRAM at inference** | | **~3–4 GB** |

Fits easily on a 12 GB RTX 3060. Training requires bf16 activations + optimizer state for the
active stack only — also within the 12 GB envelope for a stack of ~180M parameters.

---

## 3. Training Recipe

### 3a. Phase 1 — bf16 warm-up with KD (16-bit majority of run)

The demonstrated recipe (arXiv:2502.11895 "Continual QAT"; HF blog 1_58_llm_extreme_quantization)
establishes that starting in 16-bit for 20–40% of the run before transitioning to ternary is
critical. For capability stacks, this translates directly:

1. Initialize each stack by copying the final block (FFN + norm) and `lm_head` from the base model.
2. **Train the stack in bf16** against type-specific data, with KD from the frozen bf16 base model
   as teacher. The teacher signal closes the gap that ternary cannot close on its own.
   - KD objective: KL-divergence on output logits (proven approach; see TernaryBERT arXiv:2009.12812,
     Apprentice arXiv:1711.05852).
   - The backbone is the teacher — it is always resident in bf16 for inference, so no extra model
     load is needed. This makes full KD feasible on the 3060 without a separate teacher process.
3. Train for ~60–80% of the total budget in bf16.

For Japanese (the hardest capability), the teacher requirement is the primary constraint:
a frozen English-centric BitNet base is a weak Japanese teacher. A Japanese-capable full-precision
teacher (e.g. a multilingual Llama-3-8B or Qwen2.5-7B) would provide stronger gradient signal but
does not fit alongside the student on a 3060. This is the reason the **full Japanese stack
experiment wants T4×2 (teacher on one card, student on the other) or Strix (128 GB, teacher
CPU-offloaded)** rather than the 3060.

For prose and tool/JSON stacks, the base itself is a strong teacher and 3060 training is feasible.

### 3b. Phase 2 — ternary compression (optional, later)

Once the bf16 stack has converged, it can be quantized to 1.58-bit using:
- Straight-through estimator (STE) for gradient flow through rounding.
- Gradual lambda warmup (linear ramp from 0 to 1 over 10–20% of phase-2 tokens).
- An extra RMSNorm before each linear layer (arXiv:2505.08823 "An Extra RMSNorm is All You Need
  for Fine Tuning to 1.58 Bits" — note: the strong form "alone closes most of the gap" was refuted
  3-0; SubLN is a stabilizer, not a silver bullet).
- Continue KD from the bf16 version of the stack itself (now the phase-2 teacher).

Whether ternary compression of a trained bf16 capability stack preserves the added capability is
**an open question** (see Section 6). It is not on the critical path; the bf16 stack is the primary
deliverable.

### 3c. Hardware requirements per stack

| Stack | Teacher | Training hardware |
|-------|---------|-------------------|
| Prose (English general) | Base BitNet bf16 (same card) | 3060 12 GB |
| Tool-call / JSON | Base BitNet bf16 (same card) | 3060 12 GB |
| Japanese | Base BitNet bf16 (weak teacher) OR JA-capable 7B (needs split) | Prefer T4×2 or Strix |

---

## 4. Novel vs. Demonstrated Claims

Being precise about novelty avoids overclaiming.

### 4a. Demonstrated precedents

- **Mixture of Softmaxes / Mixtape (Yang et al. ICLR 2018; NeurIPS 2019):** K specialized
  `lm_head` matrices mixed per token. Demonstrated to break the softmax bottleneck for
  expressivity/perplexity. The mechanism (multiple parallel output projections on the final hidden
  state) is exactly the stack architecture. The distinction is purpose: MoS/Mixtape route for
  *expressivity*, not *capability-type specialization*.

- **Growing Transformers (arXiv:2507.07129):** Trains new top-layer transformer blocks above a
  fully frozen substrate and trains `lm_head` jointly with the new blocks. Quote: "Useful continued
  learning can proceed above a frozen minimal interface." Most structurally similar to this design.

- **CORAL (arXiv:2502.16880):** Weight-grouping of the `lm_head` by vocabulary, activates only a
  subset per forward pass. Demonstrates output-layer routing is computationally tractable
  (2.5–4x speedup). Efficiency-driven, not capability-driven.

- **Medusa (arXiv:2401.10774):** K parallel output projection heads on the final hidden state for
  speculative decoding. Proves multiple parallel `lm_head`-equivalent projections are fast.

- **Multilingual routing in MoE (arXiv:2510.04694, arXiv:2605.28306):** Routes FFN experts to
  languages at middle layers. Demonstrates per-language routing works. Shared `lm_head` is the
  bottleneck these papers do not fix — which is the gap this design addresses.

- **Vocabulary expansion literature (arXiv:2402.14714, 2406.11477, 2604.16656):** Establishes that
  training the `lm_head` is necessary and sufficient for adding new-language output capability. The
  direct theoretical backing for the JA bottleneck hypothesis.

- **LLaDA-MoE (arXiv:2509.24389):** Masked-diffusion LM trained with MoE (64 experts, top-8, ~7B
  total / 1.4B active, 20T tokens). Proves diffusion + MoE is compatible from scratch.

- **Diffusion beats AR in data-constrained regimes (arXiv:2507.15857, NeurIPS 2025):** Diffusion
  outperforms autoregressive training when compute is abundant and data is scarce/repeated, because
  masked-token randomization acts as data augmentation.

### 4b. Novel combinations (clearly flagged)

**NOVEL (arm A): Output-type routing to type-specialized stacks.**
Existing output routing (MoS, Mixtape, CORAL, Medusa) is driven by expressivity, efficiency, or
position-offset speculation — not by output *type* or *capability domain*. Routing to a
capability-specialized stack (prose vs. tool-call vs. Japanese) where each stack is independently
trained on domain-pure data, and the routing trigger is the format-prefix commit token, has no
direct precedent in the literature. It is a synthesis of: MoS-style parallel output heads +
Growing Transformers-style frozen-backbone expansion + format-prefix causal routing.

**NOVEL (arm B): Diffusion-trained capability stacks for data-constrained domains (Japanese).**
No checkpoint-backed precedent exists for: (a) routing a chosen dataset deterministically to a
designated stack/expert and training that expert via a masked-diffusion objective, or (b) using a
diffusion objective as a rescue for a domain where AR fine-tuning of a ternary expert failed.
LLaDA-MoE demonstrates diffusion+MoE from scratch; the continual-learning / capability-grafting
variant is unproven. The diffusion advantage (data-constrained regimes; arXiv:2507.15857) is
motivated by the JA setting (limited high-quality Japanese data relative to training compute), but
the specific combination remains novel and should be labeled as a speculative arm, not a first
experiment.

---

## 5. Validation Plan and Experiment Sequence

### Step 0 (running): `lm_head`-trainable JA retrain

Retrain the JA-domain LoRA / FFN upcycle with `lm_head` unfrozen. This is the cheap confirmation
gate: if JA PPL drops below the dense 74.37 baseline where FFN-only training stalled flat at ~79,
the output-layer bottleneck is confirmed and the fundamental motivation for this design is
empirically grounded.

**Expected outcome:** JA PPL decreases measurably (target <60). If it does not, re-examine whether
the bottleneck is elsewhere (tokenizer fragmentation: 0.60 tok/char for JA may inflate raw PPL
independent of the lm_head — needs disentangling).

**Hardware:** 3060 (lm_head is ~128M params bf16; fits with backbone frozen).

### Step 1 (first stack): Single capability stack, bf16, KD from base

Train one output-type stack (tool-call or prose as the easiest case; JA as the most motivating).
Architecture: frozen backbone + one additional final FFN block + one dedicated `lm_head`, bf16.
No router yet — run the target domain through the new stack always, evaluate on held-out type-
specific data and on the original chat eval to confirm no regression.

**Success criterion:**
- Target domain PPL improves vs. dense baseline OR task-accuracy metric improves.
- Chat (original capability) does not regress vs. dense.

**Hardware:** 3060 for prose/tool stack; T4×2 or Strix preferred for JA (strong teacher).

### Step 2: Router + K=2 stacks

Add the format-prefix commit router. Train router + two stacks jointly (e.g. prose + tool-call) on
a labeled mixture. Evaluate:
- Routing accuracy: does the router correctly commit to the tool-call stack on `<tool_call>` / `{`?
- Task accuracy per type: does each type get the benefit of its dedicated stack?
- Routing timing: measure latency added by the router at the format-prefix token position.

### Step 3: K=3+ stacks, JA, load balancing

Scale to three or more stacks including the Japanese stack. Introduce load-balancing auxiliary loss
(same aux_weight lever as MoTE) if any stack gets starved. Cross-validate: does the JA stack help
JA without hurting prose or tool-call?

### Step 4 (optional): Ternary compression of trained stacks

Apply Phase 2 (Section 3b) to a trained bf16 stack. Verify the capability gain survives ternary
compression. This is the most uncertain step; prioritize bf16 stacks first.

---

## 6. Open Questions and Risks

### Q1: Does the output-layer bottleneck hypothesis hold? (Step 0 gate)

The JA `lm_head`-trainable experiment is the gate. If JA PPL does not improve with `lm_head`
unfrozen, the bottleneck may be elsewhere:
- Tokenizer fragmentation (0.60 tok/char inflates raw PPL without reflecting semantic competence).
- Middle-layer representation quality (the base may not encode JA semantics even if the `lm_head`
  can map them — this is the question the deep-research thread 3 left unanswered; no surviving
  claim covered which layers gate cross-lingual output).
- Insufficient Japanese pretraining data to learn a meaningful `lm_head` distribution.

If Step 0 confirms the bottleneck is the `lm_head`, proceed. If not, the layer-gating hypothesis
needs a different investigation (e.g. probing middle-layer representations for JA vocabulary).

### Q2: Does ternary compression preserve added capability?

The distillation literature achieves FP parity on MNLI classification (arXiv:2510.13998), but
those are fixed-label tasks. Whether a bf16 stack trained to produce fluent Japanese tokens survives
quantization to 1.58-bit — particularly for the low-frequency Japanese subword distribution — has
no direct precedent in the sources. This is an empirical question deferred to Step 4.

### Q3: Router timing — what if no format-prefix token appears?

If the output type is ambiguous (user prompt allows either prose or JSON), no format-prefix token
appears at a predictable position and the router must fall back to prompt-level inference. The
fallback is a prompt-level router (reads the hidden state at the last instruction token before
generation begins). This adds ~1 ms latency per request and is not a correctness concern, but must
be implemented before Step 2.

### Q4: Per-capability data availability

Training effective per-type stacks requires type-pure data:
- Prose: `no_robots`, standard instruction tuning datasets (abundant).
- Tool-call / JSON: synthetic tool-call datasets (moderate; already used in U4–U5 LoRA experiments).
- Japanese: `wikimedia/wikipedia 20231101.ja`, JSTS, JSQuAD, etc. (moderate; real gap at 2B token
  scale relative to compute).

Low-resource stacks (JA) may not reach parity with the prose stack on the first run. The diffusion
training arm (Section 4b novel arm B) is the experimental mitigation, not the default.

### Q5: Compatibility with dotLLM inference engine

The output-type stacks are an additional set of weight tensors loaded alongside the backbone. In
dotLLM's architecture, the `lm_head` is already a named weight in `DotLLM.Models`. The stack
extension requires:
- Multiple `lm_head` tensors keyed by capability type in the model weights.
- A router module (MLP or attention-pool) with its own weights.
- An inference-time dispatch: at the format-prefix token position, the router selects the active
  stack; all subsequent forward passes use that stack's FFN + norm + `lm_head`.

The constrained-decoding FSM in `DotLLM.Engine` already fires on the format-prefix token — sharing
the trigger event with the router is the intended tight integration. No new trigger mechanism needed.

---

## Summary

Mid-FFN MoTE is the correct result: the architecture works (routing is healthy, no collapse), but
ternary experts without a teacher and without added width cannot add new in-distribution capacity
(research-backed) and cannot route output tokens through a frozen `lm_head` (vocab-expansion
literature + JA flat-79 evidence). The go-forward design routes the *output*, not the mid-FFN:
K small bf16 top stacks (final FFN block + RMSNorm + dedicated `lm_head`, ~50M params each) above
the frozen ternary backbone, selected by a format-prefix commit router that fires on the same
structural token that triggers dotLLM's constrained-decoding FSM. Each stack is trained in bf16
with KD from the frozen base teacher before optional ternary compression. The design is empirically
motivated by the JA bottleneck probe (74 dense vs. 79 FFN-only flat), theoretically backed by MoS/
Growing Transformers / vocab-expansion literature, and has a clear VRAM budget (~1.45 GB stack
overhead, fits 3060 12 GB). Output-type routing to capability-specialized stacks is novel; diffusion
training for the Japanese stack is novel and speculative.

---

## References

| Paper | arXiv / URL | Role in this spec |
|-------|-------------|-------------------|
| Breaking the Softmax Bottleneck (Yang et al., ICLR 2018) | arXiv:1711.03953 | Architectural precedent: parallel lm_heads |
| Mixtape: Efficient MoS (NeurIPS 2019) | NeurIPS 9723 | Tractable multi-head output routing |
| Growing Transformers (2026) | arXiv:2507.07129 | Closest structural precedent: top stacks above frozen substrate |
| CORAL LM Head Router | arXiv:2502.16880 | Output-head routing is computationally tractable |
| Medusa (2024) | arXiv:2401.10774 | K parallel output projections on final hidden state |
| EEVE-Korean vocab expansion | arXiv:2402.14714 | lm_head must be trained for new language |
| Vocab expansion survey | arXiv:2406.11477, 2604.16656 | Frozen lm_head = output ceiling |
| Multilingual MoE routing | arXiv:2510.04694, 2605.28306 | Middle-layer routing, shared lm_head = bottleneck |
| BitNet b1.58 2B4T | arXiv:2504.12285 | Base model; native ternary at 2B scale |
| BitNet Distillation | arXiv:2510.13998 | Naive QAT unstable; distillation pipeline for parity |
| Continual QAT (Nielsen et al., ACL 2025) | arXiv:2502.11895 | 16-bit → ternary recipe; ppl 12.2 vs 26 from-scratch |
| HF 1.58-bit fine-tuning blog | hf.co/blog/1_58_llm_extreme_quantization | Demonstrated recipe + checkpoints |
| TernaryBERT | arXiv:2009.12812 | Distillation reaches FP parity at 14.9x smaller |
| Apprentice (Hinton / Mishra, ICLR 2018) | arXiv:1711.05852 | KD improves low-precision networks |
| Extra RMSNorm | arXiv:2505.08823 | SubLN stabilizes ternary QAT (alone insufficient) |
| Ternary capacity ceiling | arXiv:2411.05882, 2407.09527 | 2x width for LM parity |
| LLaDA-MoE | arXiv:2509.24389 | Diffusion + MoE from scratch demonstrated |
| Diffusion beats AR (NeurIPS 2025) | arXiv:2507.15857 | Diffusion wins in data-constrained regimes |
