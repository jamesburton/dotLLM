# Exploration: Masked Diffusion LM Objective for Ternary MoTE Experts

**Date:** 2026-07-02
**Status:** Design gate — no implementation code
**Context:** `trackM-mote` worktree; base model `microsoft/bitnet-b1.58-2B-4T`

---

## 0. Executive Summary

A masked-diffusion (LLaDA-style) training objective for ternary expert layers is a
**technically sound and genuinely novel** idea, but it **does not address the specific
Japanese-capability bottleneck we already measured**. The bottleneck was the output
layer (`lm_head`), not the training objective for hidden FFN layers. FFN-only AR training
gave Δ PPL = −0.18 on Japanese; lm_head training gave Δ ≈ −12. A diffusion objective
applied to the same FFN-only parameters will be approximately as ineffective for Japanese.

The idea has real merit as a research direction into "can ternary weights be trained
well under a non-AR, bidirectional objective" and "can a diffusion-trained ternary LM
match an AR-trained one at small scale." The correct de-risk ladder is:

1. **Dense 117M, WikiText-103, MDLM objective** — does the loss setup even work? (RTX 3060, ~1 day)
2. If yes → **Ternary 117M from scratch, same setup** — does QAT interfere with diffusion gradient? (~2 days)
3. If yes → consider integration into MoTE architecture

**Go/No-Go: Conditional Go** for the de-risk experiment only. Do NOT attempt to plug a
diffusion objective directly into the BitNet 2B MoTE before steps 1–2 pass.

**First experiment:** dense 117M masked-diffusion baseline on WikiText-103. Kill criterion:
BPD > 2.5 bits/token after 100M training tokens.

---

## 1. Concrete Mechanism: Absorbing-Mask Discrete Diffusion

### 1.1 Forward process

Given a sequence x = (x₁, …, x_L) of discrete tokens from vocabulary V, the absorbing
(masking) forward process at noise level t ∈ [0, 1] independently replaces each token
with a special [MASK] token with probability t:

```
q(x_t | x_0) = Π_i [t · δ(x_i^t = MASK) + (1-t) · δ(x_i^t = x_i)]
```

At t = 0, no tokens are masked (original sequence). At t = 1, all tokens are [MASK].
This is a "one-way" absorbing-state Markov chain: once a token is masked it cannot
un-mask in the forward direction.

Key property: the posterior q(x₀ | x_t, x) is a product of independent Bernoulli
distributions — the masked tokens came from the original and the unmasked ones are
known exactly. This makes the ELBO tractable.

### 1.2 Training loss (MDLM / LLaDA recipe)

MDLM (Sahoo et al., arXiv 2406.07524) showed the ELBO simplifies to a **weighted
mixture of masked-language-modeling losses**:

```
L(θ) = E_{t ~ U[0,1]} E_{m ~ Mask(t)} [
    (1/t) · Σ_{i: m_i = MASK} -log p_θ(x_i | x_m)
]
```

where:
- t is sampled uniformly from [0, 1]
- m is a binary mask with each position masked independently with probability t
- p_θ(x_i | x_m) is the model's predicted distribution over x_i given x_m (the
  partially masked sequence, with bidirectional attention over all positions)
- The 1/t weighting comes from the ELBO derivation and upweights predictions at
  low noise levels (where fewer tokens are masked and the task is easier)

In practice: sample t, mask the sequence, forward pass with FULL (bidirectional, non-
causal) attention, compute cross-entropy only at the masked positions, scale by 1/t.

LLaDA (Nie et al., arXiv 2502.09992) uses the same objective with an additional
simplification: discard the 1/t weighting (treat it as uniform over diffusion steps),
which they call the "simple loss" and show empirically works as well or better.

### 1.3 Architecture requirement: bidirectional attention

**Critical difference from AR:** The model must use full (bidirectional, non-causal)
attention — no causal mask. This is why you cannot trivially plug the diffusion loss
into an existing causal LM without architectural modification. DiffuLLaMA (arXiv
2410.17891, ICLR 2025) handles this via "attention mask annealing" when adapting a
pre-trained causal model: gradually remove the causal mask over the first phase of
training.

### 1.4 Inference: iterative unmasking

Sampling from a trained masked-diffusion LM works by iterative refinement over T steps:

```
1. x^T = [MASK, MASK, ..., MASK]  (all masked)
2. For step τ from T down to 0:
     a. p_θ(x | x^τ) = forward pass (full attention over current sequence)
     b. For each still-masked position i:
          - Sample or argmax x_i from p_θ(x_i | x^τ)
          - Unmask x_i with probability proportional to confidence and schedule
     c. Re-mask some positions (optional, for iterative refinement)
3. Return x^0 (fully unmasked sequence)
```

The simplest strategy (greedy MDLM): at each step unmask the L/T highest-confidence
tokens; T steps to generate L tokens (T typically 10–100). This is the core efficiency
advantage: parallel decoding in O(T) forward passes vs O(L) for AR.

**LLaDA codebase is in HF cache and DiffusionGemma is running on Strix** — the
iterative unmasking loop is already implemented in those codebases and can be reused
directly.

---

## 2. Unit-of-Training Question — Critical

**Can a single FFN expert be "trained as a diffusion LM" in isolation?**

No. The diffusion LM objective requires a full model forward pass to produce predictions
over the vocabulary (a softmax over |V| ≈ 32K tokens). A single FFN expert receives
hidden states from an attention block and emits hidden states back to the residual
stream. It has no projection to vocabulary. You cannot compute a masked-token prediction
loss from an FFN expert in isolation.

**Honest architecture of what this experiment actually is:**

> "Train a small standalone transformer — embedding + attention layers + FFN layers +
> lm_head — under the masked diffusion objective, where the FFN layers use AutoBitLinear
> (ternary weights). The 'ternary expert' is the FFN capacity within this standalone
> transformer."

This is **not** the same as plugging a diffusion objective into one MoTE block inside
the existing BitNet 2B backbone. It is closer to:

- Option A: Train a **small ternary transformer from scratch** with a masked diffusion
  objective (standalone, e.g., 125M params). The result is a ternary diffusion LM, not
  a MoTE expert. To use it in MoTE you'd then have to adapt/distill its weights into
  MoTE expert slots — an additional, non-trivial step.

- Option B: Adapt the **full BitNet 2B** (not just FFN layers) to a diffusion objective
  by removing the causal mask. This modifies the shared backbone, not just the experts.
  Freezing the backbone is difficult because the causal masking is baked into the
  attention, and removing it changes the effective attention distributions throughout.

- Option C: The most tractable integration (if you want to stay within the MoTE
  training framework): use the diffusion objective **only at the lm_head output** while
  the backbone remains causal. This is a semi-AR approach (not true diffusion). The
  model would generate all tokens in one-shot at the output layer, attending to the
  full causal context. This is the "block diffusion" approach (LLaDA 2.0).

The cleanest experimental unit is **Option A: train a small ternary transformer from
scratch with masked diffusion objective**.

---

## 3. Where Ternary Enters — Interaction with QAT

### 3.1 Is diffusion orthogonal to ternary QAT?

Mostly yes, with one important caveat.

**Orthogonal aspects:**
- The loss function (masked cross-entropy at masked positions, weighted by 1/t) is
  defined at the model output level. It does not care about how hidden-layer weights
  are quantized.
- The STE (straight-through estimator) used for ternary QAT passes gradients through
  the rounding operation: backward pass sees dL/d(quantized_w) as dL/d(w_bf16). This
  is independent of the loss shape.
- AutoBitLinear in our MoTE codebase (`mote_upcycle.py`) already implements online
  absmean ternary quantization in its forward pass and STE in backward. No modification
  needed to use it with a diffusion loss.

**Important difference from AR QAT:**
- AR training provides gradient signal at ONE position per forward pass (the next token).
  The gradient is sparse and sequential: each parameter update comes from one token
  prediction position.
- Diffusion training provides gradient signal at MULTIPLE positions per forward pass
  (all masked positions). With typical mask rate t ~ 0.5, roughly L/2 positions
  contribute gradient per step. This is a **denser gradient signal**, closer to
  masked-LM pre-training (BERT-style).
- For ternary QAT, denser gradient might be beneficial (more stable STE gradient
  estimate per step) or harmful (more conflicting gradient directions, more noise).
  This is genuinely unknown and is one reason to run the de-risk experiment.

### 3.2 Recommended integration

If de-risk passes:

```python
# In training loop, replace AR LM loss:
# lm_loss = cross_entropy(logits[:, :-1], tokens[:, 1:])

# With MDLM loss:
t = torch.rand(1).item()   # sample noise level
mask = torch.bernoulli(torch.full_like(tokens.float(), t)).bool()
masked_tokens = tokens.clone()
masked_tokens[mask] = MASK_TOKEN_ID

# Forward pass with BIDIRECTIONAL attention (caution: requires removing causal mask)
logits = student(input_ids=masked_tokens, attention_mask=torch.ones_like(masked_tokens)).logits

# Loss only at masked positions, weighted by 1/t
masked_logits = logits[mask]   # [n_masked, V]
masked_targets = tokens[mask]  # [n_masked]
diffusion_loss = cross_entropy(masked_logits, masked_targets) / t
```

The switch from causal to bidirectional attention is the gating blocker. For a
standalone small ternary LM (Option A), this is straightforward (just don't add a
causal mask to the attention). For the existing BitNet 2B, you would need to modify
the attention implementation or use the attention-mask-annealing approach from
DiffuLLaMA.

---

## 4. The Japanese / Hard-Capability Hypothesis — Skeptical Check

**The measured bottleneck:**

From our MoTE eval logs:

| Condition | JA PPL (wiki_ja) | EN PPL (no_robots) |
|---|---|---|
| Dense BitNet 2B (baseline) | 73.4 | 13.5 |
| MoTE FFN-only, 4 experts, AR obj., ~400M tokens | 73.2 (Δ=−0.18) | 13.5 |
| MoTE + lm_head training, AR obj., ~256K tokens | 61.7 (Δ=−11.7) | 14.6 |

**Interpretation:** The dominant bottleneck for Japanese is the output layer (`lm_head`,
the unembedding matrix that maps from hidden states to vocabulary logits). The dense
FFN layers trained with AR objective showed essentially no improvement (Δ=−0.18). The
lm_head, which maps from a 2048-dimensional hidden state to 32K vocabulary positions,
cannot represent Japanese tokens well when trained primarily on English data.

**Does a diffusion objective address this bottleneck?**

No. Here is why:

1. **What diffusion changes:** the objective function (bidirectional prediction of masked
   tokens vs. sequential prediction of next token). Both still route gradient through
   the lm_head output layer.

2. **What diffusion does NOT change:** which parameters are updated. If you train MoTE
   with a diffusion loss but keep `lm_head` frozen (as in the FFN-only experiments),
   the lm_head bottleneck persists.

3. **The specific failure mode for Japanese:** BitNet 2B was trained on primarily
   English data. The lm_head has poor resolution in the Japanese vocabulary subspace.
   Training signal from Japanese text flows through the FFN layers but cannot improve
   the lm_head output mapping if lm_head is frozen. This is an **output-layer
   representation bottleneck**, not a context-integration bottleneck.

4. **What diffusion could theoretically offer:** because diffusion sees ALL positions
   bidirectionally, it might integrate context from Japanese tokens appearing elsewhere
   in the sequence more effectively than causal AR. This could help with within-sequence
   consistency. But the primary bottleneck (vocabulary projection) is unchanged.

**Honest verdict:** A diffusion objective is orthogonal to the Japanese-capability gap
we measured. It will not fix the lm_head bottleneck any more than AR FFN-only training
did. If the goal is improving Japanese performance, the correct intervention remains:
train lm_head + final norm alongside the FFN experts, as ja_lmhead4.log confirms.

**However**, the diffusion objective has a legitimate separate motivation:
- It forces bidirectional context awareness in the ternary FFN layers
- This might produce better feature representations for tasks that benefit from
  full-sequence context (e.g., tasks with reversal-type dependencies, where AR models
  struggle and LLaDA has been shown to excel)
- It is a novel training signal for ternary weights that has not been studied

These are research questions separate from the Japanese capability gap.

---

## 5. Dense-From-Scratch Minimal Experiment — Concrete Specification

The user's proposed de-risk: **first test with a dense model on a known-good dataset,
reproduce similar quality to AR before attempting the ternary case.**

### 5.1 Sanity check (must pass first — hours, not days)

**Goal:** Verify the masked diffusion loss implementation actually decreases.

- Model: 2-layer, hidden=256, 4-head transformer (randomly initialized), ~1.5M params
- Dataset: WikiText-2 (a few MB, already a standard benchmark)
- Training: 10K steps, batch=4, seqlen=256, MDLM loss (simple, no 1/t weighting)
- Expected: loss should decrease from ~ln(V) ≈ 10.4 (random init) to < 3.0
- Kill: loss does not decrease by step 1000 → implementation bug

This sanity check does NOT require a GPU. It runs on CPU in ~30 minutes.

### 5.2 Minimal de-risk experiment — RTX 3060 12GB

**Goal:** Can a dense 117M model trained with masked diffusion objective on WikiText-103
reach quality comparable to AR training on the same data?

- **Model:** 12-layer, hidden=768, FFN=3072, 12-head transformer (GPT-2 Small equivalent)
  - ~117M parameters
  - FULL bidirectional attention (no causal mask)
  - BF16 weights
  - Memory: ~0.5GB weights, ~3GB activations at batch=8, seq=512 → well within 12GB

- **Dataset:** WikiText-103 (103M token Wikipedia corpus, standard benchmark)
  - Known AR reference: GPT-2 124M → PPL ~18-20 on WikiText-103
  - MDLM 110M → PPL ~21-23 (within ~20% of AR; Sahoo et al. Table 2)
  - This gives us a quantitative "comparable quality" target

- **Objective:** MDLM simple loss (uniform 1/t weighting, not the ELBO-exact variant)

- **Training budget:** 1B tokens (~10 hours on RTX 3060 at ~27M tok/s for 117M model)
  - Or subset: 300M tokens (~3 hours) for a first signal

- **Quantitative success threshold:** BPD (bits per dimension) ≤ 2.0 on WikiText-103
  test set, OR generative PPL via greedy unmasking (T=10 steps) ≤ 40. MDLM achieves
  ~1.78 BPD; AR GPT-2 achieves ~1.57 BPD. "Within 30% of AR" = acceptable.

- **Measurement:** Compare against a GPT-2-Small equivalent trained AR on the same
  data for the same token budget (easy reference: use HF GPT-2 as the baseline since
  it's pre-trained on similar web text).

**What "comparable quality" means operationally:**
> The masked diffusion 117M model trained from scratch on WikiText-103 achieves
> ≤ 2.5 BPD (or PPL ≤ 50) after 1B tokens of training. GPT-2-Small achieves ~1.57 BPD.
> Within 60% degradation is acceptable for this de-risk (diffusion models have
> inference advantages that partially offset PPL gaps).

### 5.3 Ternary extension (if 5.2 passes)

**Goal:** Does QAT (ternary weights) interfere with masked diffusion training?

- Same model architecture, but replace nn.Linear layers with AutoBitLinear (online
  absmean ternary quantization, STE backward)
- Expect 10–30% PPL degradation from ternary quantization at 117M scale (consistent
  with BitNet b1.58 findings at similar scale — ternary 1B ≈ fp16 360M)
- Kill criterion: ternary 117M achieves > 2× the BPD of dense 117M after same token
  budget (suggests QAT gradient noise + diffusion gradient interaction is destructive)

---

## 6. Hardware Feasibility

### RTX 3060 12GB (T1 — primary)

**Fits:**
- Sanity check (1.5M model, any seq len)
- De-risk experiment: 117M dense diffusion LM on WikiText-103 (fits in ~4GB total)
- Ternary 117M variant (same memory as dense for QAT with online quant)

**Memory breakdown for 117M @ batch=8, seq=512, BF16:**
- Weights: 117M × 2B = 234MB
- Activations (no gradient checkpointing): ~3GB
- Optimizer (AdamW): ~470MB (2 moments × 117M × 4B)
- Total: ~4GB — comfortable within 12GB

**Estimated throughput:** at ~27M tokens/second for a 117M forward pass on 3060,
1B tokens takes ~10 hours. With the diffusion objective (each forward covers the whole
sequence including masked positions, so no 2× overhead for mask), throughput is similar.

**Does NOT fit on 3060:**
- BitNet 2B full adaptation to diffusion (14GB+ for training)
- Any DiffuLLaMA-style adaptation of the full 2B base (requires bidirectional attention
  in all 32 layers = same model memory + optimizer states = 16-20GB)

### Strix Halo (T2 — non-CUDA, AMD Vulkan/CPU)

DiffusionGemma is already running on Strix. This means:
- The iterative unmasking sampling loop is already implemented and tested
- The diffusion model inference infrastructure is available for reuse
- But Strix is CPU/Vulkan only — cannot do efficient QAT training (no CUDA STE kernels)

**Strix is useful for:**
- Running inference on a trained ternary diffusion LM checkpoint (evaluation)
- Running the LLaDA checkpoints already in HF cache for reference comparisons
- NOT for training the de-risk experiment (do that on 3060)

### Kaggle T4×2 or L4 (T3)

**Needed for:**
- BitNet 2B full-model diffusion adaptation (if ever pursued)
- Larger ternary diffusion LM (>500M params) training
- Teacher-student pipeline if a dense diffusion teacher is used

**Coordination requirement:** confirm before launching Kaggle jobs per project protocol.

---

## 7. Risks, Failure Modes, Kill Criteria

### Risk 1: Ternary QAT + diffusion gradient interact destructively
**Mechanism:** The dense, multi-position gradient from masked diffusion loss may cause
high variance in the STE gradient estimate for ternary weights, leading to oscillation
or collapse of the ternary quantization (all weights → 0 or all → ±1).
**Signal:** Training loss diverges or BPD fails to decrease below 5.0 after 50M tokens.
**Kill criterion:** If loss > 5.0 after 50M tokens of ternary 117M training, abandon.

### Risk 2: Diffusion objective provides no JA capability improvement
**Mechanism:** As analyzed in §4, the lm_head bottleneck is orthogonal to the training
objective. A MoTE with diffusion-trained FFN experts but frozen lm_head will still
show flat JA PPL.
**This is expected, not a surprise.** Only matters if the goal was specifically Japanese.
**Kill criterion for JA hypothesis:** JA PPL Δ ≤ −2.0 after 400M tokens with diffusion
FFN-only training. (We expect ~−0.18, consistent with AR result.)

### Risk 3: Dense 117M de-risk fails (diffusion training doesn't converge)
**Mechanism:** Implementation bug, wrong learning rate, wrong 1/t weighting, causal
mask not fully removed, wrong mask token handling.
**Signal:** BPD does not decrease from initial ~10.4 after 10K steps.
**Kill criterion:** If de-risk sanity check (§5.1) fails → stop and debug before
proceeding to any further experiments.

### Risk 4: Causal mask removal breaks the existing BitNet 2B architecture
**Mechanism:** The BitNet model's attention layers have causal masking hard-coded in
the HF implementation. Removing it changes the behavior of KV caching and may cause
numerical instability or incorrect attention patterns.
**Mitigation:** Do NOT attempt causal mask removal on the full 2B model before the
standalone 117M experiment succeeds. For the standalone experiment, use a fresh
encoder-only (BERT-style) transformer, not a modified causal LM.

### Risk 5: Memory OOM during bidirectional attention at long seqlens
**Mechanism:** Full attention (bidirectional) with seq=512 on a 117M model has
O(L²·d) memory. At batch=8, seq=512: attention matrix = 8 × 12 × 512 × 512 × 2B ≈ 0.6GB.
**Mitigation:** Gradient checkpointing. Start at seq=256 if OOM.

---

## 8. Precedents and Reuse Potential

| Paper | arXiv | Relevance | Checkpoint/Code Available |
|---|---|---|---|
| LLaDA | 2502.09992 | Core method: absorbing-mask forward process, simple loss | HF cache (user confirmed) |
| MDLM | 2406.07524 | Loss derivation, ELBO simplification, code on GitHub | github.com/kuleshov-group/mdlm |
| MD4 | 2407.04121 | Same ELBO from different angle; code from Google DeepMind | Code in paper supplement |
| SEDD | 2310.16834 | Score entropy discrete diffusion (different, harder) | Aaron Lou's GitHub |
| DiffuLLaMA | 2410.17891 | Adapting AR→diffusion: attention mask annealing trick | HKUNLP/DiffuLLaMA on HF |
| LLaDA-MoE | 2509.24389 | MoE + diffusion from scratch (7B total, 1.4B active) | inclusionAI/LLaDA-MoE-7B-A1B-Base |
| LLaDA 2.0 | 2512.15745 | AR→diffusion "knowledge inheritance", scales to 100B | — |
| LLaDA 2.0 Uni | 2604.20796 | Adds MoE backbone to LLaDA 2.0 | — |

**Immediate reuse:** The MDLM training loop (github.com/kuleshov-group/mdlm) is the
cleanest reference implementation. It implements:
- `MaskingSchedule` — the t ~ U[0,1] sampler and token masking
- `MDLMLoss` — the 1/t weighted cross-entropy at masked positions
- Training harness for GPT-2 / DiT models on text8 / One-Billion-Word

The LLaDA demo code (in HF cache from the user's existing LLaDA model) provides the
iterative unmasking sampler, reusable for evaluation on Strix.

DiffusionGemma on Strix is useful for: validating iterative unmasking sampling quality
by running reference comparisons against a known-good diffusion LM.

---

## 9. Go/No-Go Recommendation

**Verdict: CONDITIONAL GO for the de-risk ladder only.**

**Do:** Run §5.1 sanity check (CPU, ~30 min) followed by §5.2 dense 117M de-risk on
RTX 3060 (~10 hours). Both must PASS before any integration into MoTE or BitNet 2B.

**Do not** (yet):
- Attempt to add a diffusion objective to the existing `mote_train.py` pipeline (the
  causal mask issue alone makes this non-trivial)
- Target the Japanese capability gap with diffusion (the bottleneck is the lm_head,
  not the objective)
- Run BitNet 2B full-model adaptation before the small-scale sanity is confirmed

**The single first experiment:**

Train a 117M dense bidirectional transformer on WikiText-103 using the MDLM simple
loss for 300M tokens on the RTX 3060. Use the MDLM reference code directly
(github.com/kuleshov-group/mdlm). Measure BPD on the WikiText-103 test set and compare
against the published MDLM 110M result (~1.78 BPD). If your run reaches ≤ 2.5 BPD
after 300M tokens, the machinery works. Then proceed to the ternary variant (§5.3).

**Kill criterion for the first experiment:**
> BPD > 2.5 after 300M training tokens → something is wrong (implementation, LR,
> architecture). Debug before proceeding. Do not assume the ternary QAT is at fault
> if the dense baseline hasn't passed first.

---

## 10. What Success Would Actually Prove

If all three de-risk stages pass (dense 117M → ternary 117M → integration):

1. **Ternary weights can be trained under masked diffusion objective** — this is novel;
   no prior paper has done it at >10M parameter scale.

2. **The diffusion objective produces different representations** in the ternary FFN —
   measurable by comparing perplexity on tasks with reversal properties (where LLaDA
   beats AR models) vs standard next-token prediction.

3. **A new route to ternary layer initialization** — a ternary diffusion LM trained
   from scratch could serve as a weight-initialization source for MoTE experts,
   replacing the current "deep clone from dense bitnet" upcycle. The hypothesis is
   that experts initialized from diffusion training may specialize differently under
   subsequent AR fine-tuning.

None of these outcomes directly improve the Japanese PPL gap. They are independently
interesting research contributions.

---

## Appendix: Key Numbers for Hardware Planning

| Experiment | Params | VRAM (BF16, batch=8, seq=512) | Est. time (3060) | Device |
|---|---|---|---|---|
| Sanity check | 1.5M | <1GB | 30 min (CPU) | Any |
| Dense 117M de-risk | 117M | ~4GB | ~10h (300M tokens) | RTX 3060 |
| Ternary 117M de-risk | 117M | ~4GB (QAT overhead minimal) | ~12h | RTX 3060 |
| BitNet 2B adaptation | 2B | ~16GB+ (training) | Not on 3060 | Kaggle L4 |
| Full MoTE + diffusion | 3B (student+teacher) | >12GB | Not on 3060 | Kaggle |

---

*References (arXiv IDs):*
- *LLaDA: 2502.09992 | MDLM: 2406.07524 | MD4: 2407.04121 | SEDD: 2310.16834*
- *DiffuLLaMA: 2410.17891 | LLaDA-MoE: 2509.24389 | LLaDA 2.0: 2512.15745*
- *LLaDA 2.0 Uni: 2604.20796*
