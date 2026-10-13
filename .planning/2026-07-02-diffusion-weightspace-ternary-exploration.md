# Weight-Space Diffusion for Ternary Expert Generation: Design Exploration

**Date:** 2026-07-02  
**Status:** Design gate — no code; written exploration only  
**Scope:** Feasibility and first-experiment design for using weight-space diffusion to generate ternary FFN experts for MoTE upcycle of `microsoft/bitnet-b1.58-2B-4T`

---

## TL;DR (read this first)

**Recommended mechanism:** Diffuse rank-16 LoRA *deltas* (continuous bf16) over the gate_proj of a single FFN layer, then ternarize post-sampling. Start with ONE expert, ONE layer, ONE LoRA projection.

**The crux risk:** Training data. Weight-space diffusion needs a *population* of trained weight variants. We currently have at most 4–6 distinct expert checkpoints. Minimum viable is ~20–30. Every hour we spend without a training population is wasted planning.

**One first experiment:** Overfitting test — train a tiny MLP diffusion model to reconstruct a single existing LoRA adapter's gate_proj A matrix (~41K params). If sampled adapter recovers within 20% of original eval PPL, proceed to population building.

**Go/No-Go:** Conditional Go. The task-conditioned generation payoff is genuinely novel. But the training data bootstrap is a real obstacle. Run the overfit test (2–3 hours on 3060) before any further investment.

---

## 1. Prior Art and Grounding

### 1.1 Core Papers

| Paper | arXiv | What was generated | Scale actually demonstrated | Code |
|-------|-------|--------------------|-----------------------------|------|
| Neural Network Diffusion (p-diff) — Wang et al. 2024 | 2402.13144 | Full parameters of *tiny* nets; last-2-norm-layers of large nets | MLP-3: ~40K params; ConvNet-3: ~25K params; ViT/ResNet: only ~few-thousand normalization params | Yes (NUS-HPC-AI-Lab/Neural-Network-Diffusion) |
| G.pt — Peebles et al. 2022 | 2209.12892 | Full parameters of small supervised + RL nets, conditioned on target loss/return | MNIST MLPs, small CNNs, RL policy nets — all well under 1M params | Yes (wpeebles/G.pt) |
| HyperDiffusion — Erkoç et al. 2023 | (Semantic Scholar) | Weights of implicit neural fields (INFs) — very small MLPs representing 3D shapes | ~100K param INF MLPs | Not widely released |
| D2NWG — 2024 | 2402.18153 | Transfer-learning weights conditioned on target dataset; claims "LLM scale" but no specifics in abstract | Unknown from abstract; methodology suggests subset/head layers only | No (CC-BY-4.0, no repo found) |
| Recurrent Diffusion — 2025 | 2501.11587 | LoRA parameter vectors for vision models, generated autoregressively in chunks | Up to 200M params on a single GPU | Unknown |
| ORAL — 2025 | 2503.24354 | LoRA weights for LLMs across 7 language + 4 vision + 3 multimodal tasks | Scales to "billions-of-parameter LLMs"; 99.9% storage saving vs per-task LoRA storage | Unknown |
| DiffLoRA — 2024 | 2408.06740 | LoRA weights for Stable Diffusion XL for image personalization | SD-XL LoRA size (~tens of millions) | Unknown |

**No papers found** on diffusion generating ternary or quantized ({-1,0,+1}) weight tensors. This is a genuine gap — and therefore a genuine research risk.

### 1.2 The Critical p-diff Finding

p-diff's paper explicitly states its limitation: *"We cannot synthesize the entire parameters of large architectures, such as ResNet, ViT, and ConvNeXt series, primarily due to GPU memory limitations."* For large nets, they generated ONLY the last two normalization layers — a few hundred parameters, not the full network.

This has direct implications for our use case (Section 2).

---

## 2. Concrete Mechanism for Our Case

### 2.1 Verified Dimensions (from `config.json`)

```
hidden_size:        2560
intermediate_size:  6912
num_hidden_layers:  30
activation:         relu2 (gated MLP: gate_proj × up_proj → relu²(gate) × up → down_proj)
```

**One FFN expert (BitNetMLP) = three ternary linear projections:**
- `gate_proj`:  `[6912, 2560]` = 17,694,720 ternary weights  
- `up_proj`:    `[6912, 2560]` = 17,694,720 ternary weights  
- `down_proj`:  `[2560, 6912]` = 17,694,720 ternary weights  
- **Total per expert: ~53.1M ternary params**

Storage: ~53 MB int8 per expert; ~106 MB bf16 master per expert.

**Scale mismatch vs prior work:**  
- p-diff fully generated: ~40K params  
- Our full FFN expert: ~53M params  
- Ratio: **1,350×**

This is not a marginal gap. It means naive p-diff-style generation of full FFN experts is non-viable on any hardware we have.

### 2.2 Four Options and Analysis

#### Option A: Diffuse only a LoRA/low-rank delta (RECOMMENDED)

Rather than generating the full 53M-param weight matrix, diffuse a rank-*r* low-rank delta on top of the frozen ternary base weight. At rank 16:

| Projection | LoRA A shape | LoRA B shape | Params |
|------------|-------------|-------------|--------|
| gate_proj | [2560, 16] | [16, 6912] | 40,960 + 110,592 = 151,552 |
| up_proj   | [2560, 16] | [16, 6912] | 151,552 |
| down_proj | [6912, 16] | [16, 2560] | 151,552 |
| **Total delta per expert** | | | **~454K params** |

In bf16: ~0.9 MB per expert delta. This is within p-diff's demonstrated range and within ORAL/Recurrent Diffusion's range.

**Why this works conceptually:** The frozen ternary base weight already encodes general language capability. The LoRA delta encodes task-specific specialization. Diffusion learns the distribution over task-specific deltas — a far smaller space.

**Why this is the right choice:**
1. Aligns with ORAL and Recurrent Diffusion — both demonstrated LoRA generation for LLMs at this scale.
2. Our existing LoRA infrastructure (`scripts/lora/train_task_lora.py`) generates exactly this format.
3. Ternarizing a LoRA delta is simpler (see Section 4).
4. The diffusion model itself stays small enough for 3060.

#### Option B: Diffuse per-channel/tile blocks

Generate the 53M-param matrix in 2560×128 tiles, diffusing each tile independently. Problems: (1) tiles are statistically independent but physically they are not — cross-tile coherence is lost; (2) still requires 54 diffusion samples per expert; (3) no evidence this works. Not recommended.

#### Option C: Autoencoder in weight space, then diffuse latent

Train a VAE to compress each 17.7M-param projection matrix to a ~4K-dim latent. Diffuse in latent space. Decode to full weight. Problems: (1) requires training the VAE itself, which requires the full population of weight matrices to begin with — circular dependency; (2) we'd need to learn the VAE before the diffusion model; (3) two-stage training adds complexity. Potentially viable at Kaggle-scale but not for a first experiment.

#### Option D: Shrink the expert

Use a bottleneck expert: hidden_size=512, intermediate_size=1024. Total params: 3 × 512 × 1024 = 1.57M. This falls into p-diff territory (25K–200M). But: (1) a bottleneck expert is not a full-capacity BitNetMLP clone; (2) adding it to the MoTE model requires architectural changes to handle the dimension mismatch (projection layers needed); (3) defeats the purpose of starting from the trained dense FFN.

---

## 3. The Training-Data Bootstrap Problem — The Crux

**This is the hardest part. Everything else is engineering; this is the physics.**

Weight-space diffusion learns a distribution *P(weights)*. Learning a distribution requires samples from it. For our LoRA-delta diffusion approach, each "sample" is a fully-trained LoRA adapter for one FFN expert. We need enough to define a non-trivial distribution.

### 3.1 What we currently have

From the MoTE experiments and task-LoRA work: at most **4–6 distinct expert checkpoints** (one per task: instruction, tool-use, coding, Japanese, etc.). This is far too few. The diffusion model would memorize, not learn a distribution.

### 3.2 How to assemble a training population

| Strategy | N possible | Cost estimate (3060) | Quality |
|----------|-----------|---------------------|---------|
| One LoRA per task domain (existing) | 4–6 | Already done | Low diversity — different tasks |
| Same task, varied seeds + data slices | 20–50 | 20–50 × ~10 min = 3–8 hours | Best: same function space, varied trajectories |
| Checkpoints along ONE training trajectory | 20–50 | 1 run + dense saves | High correlation — not independent |
| Permutation augmentation of one checkpoint | ∞ | Minutes | For ternary weights, permutations that preserve function are limited and hard to enumerate correctly |
| Random init + short fine-tune (5K tokens each) | 50–100 | 50 × ~3 min = ~2.5 hours | Cheap but noisy — short fine-tunes may not converge |

**Honest assessment:**

The most rigorous approach (varied seeds + data slices, same task) requires 20–50 training runs. At rank-16 LoRA with `no_robots` slices of 50K tokens each on the 3060, each run takes roughly 5–10 minutes. So 30 runs = 2.5–5 hours. This is feasible.

**However**: the diffusion model trained on "one task, 30 random seeds" will learn the distribution over *random seed / data-ordering noise* in one task, not over *tasks*. If the goal is task-conditioned generation (Section 5), we need a *cross-task* population (one LoRA per task × N tasks). That requires significantly more effort.

**What is NOT viable:** Using checkpoints from a single training trajectory (steps 1000, 2000, 3000, ...). These are highly correlated — essentially one sample with noise added. The diffusion model trained on this data learns to undo training noise, not to generate new adapters.

**Permutation augmentation is unreliable for ternary.** For continuous weights, random permutation of neurons preserves functional equivalence. For ternary weights, the discrete structure means permutation changes inter-layer alignment. Do not use permutation augmentation for this.

---

## 4. Generating Ternary Specifically

### 4.1 Two approaches

**Approach (i): Continuous latent → post-sampling quantization (RECOMMENDED)**

1. Diffuse over the bf16 master weight of the LoRA A and B matrices (continuous, float).
2. After sampling a (A, B) pair, compute the effective weight delta: `W_delta = B @ A`.
3. Apply online absmean ternary quantization to `W_base + W_delta` (our existing `online_quant` logic).
4. The resulting ternary expert is the sum of the frozen ternary base and the quantized delta.

This preserves the continuous geometry in which diffusion operates best. The quantization step is a post-processing step. The diffusion model never needs to know about {-1, 0, +1}.

**Approach (ii): Discrete diffusion directly over {-1, 0, +1}**

Use a discrete diffusion framework (D3PM, MDLM) with a 3-token alphabet. Each position is independently corrupted by transitions between {-1, 0, +1} at training time, and the model learns to reverse this.

Problems: (1) the tooling is far less mature than continuous diffusion; (2) the ternary alphabet is severely low-capacity — 3 states per coordinate vs continuous; (3) no demonstrated precedent for generating functional weight tensors; (4) the denoising target (exact ternary recovery) is harder to optimize than a smooth L2 loss.

### 4.2 Recommendation: Approach (i)

The STE (straight-through estimator) principle already used in BitNet QAT is the right mental model here: maintain bf16 masters, quantize for forward passes. The diffusion model generates bf16 deltas; quantization happens at inference time. This decouples the generative modeling from the quantization and lets us use standard continuous diffusion infrastructure.

**The subtle risk of Approach (i):** A bf16 LoRA delta that looks good in continuous space might not quantize well. Two different continuous deltas that look similar to the diffusion model might quantize to very different ternary patterns. This is the ternary-specific risk: the quantization function is piecewise constant with flat regions, and the diffusion model has no signal about quantization boundaries. Mitigation: add quantization-error loss terms during diffusion training (not standard, requires experimentation).

---

## 5. The Honest "What Is the Actual Win?" Question

If diffusion just regenerates a layer we already trained, it is a slow and expensive copy. The win only materializes under specific conditions.

### 5.1 Candidate payoffs, ranked

**Rank 1 — Task-conditioned generation (compelling if buildable):**
Train the diffusion model conditioned on a task embedding (e.g., mean activation of the base model on task examples, or a CLIP-style text embedding of the task description). At inference: give the diffusion model "Japanese language" or "tool-calling" as a condition vector → sample a specialized ternary expert for that task *without running any gradient training*. 

This is the payoff ORAL demonstrated for standard LoRA — zero-shot task-specific adapter generation. Adapting it to ternary experts is the novel contribution. If it works: one diffusion model spawns per-task ternary experts on demand. The MoTE router would then dynamically select among task-conditioned experts at inference time.

**Why compelling:** The marginal cost of a new expert drops from "hours of fine-tuning" to "seconds of diffusion sampling."  
**Why hard:** Requires a cross-task training population (one adapter per task × many tasks), which costs real compute.

**Rank 2 — Stochastic diversity (marginal):**
Sample 8 experts from the same distribution → assemble as a MoTE ensemble. Diffusion gives "free" expert diversity without 8 separate fine-tunes. Marginal win: we could also get diversity by simply using different random seeds.

**Rank 3 — Escaping SGD local minima (speculative, probably weak):**
Diffusion samples from the learned distribution, which theoretically can include weight configurations unreachable by gradient descent from a specific init. For ternary weights in particular, the loss landscape has many flat regions that SGD cannot navigate. Diffusion sampling is not gradient-dependent.

This is theoretically interesting but practically unverifiable without extensive eval. Do not use this as a primary motivation.

**Honest summary:** If task-conditioned generation works, this is a real contribution. If the diffusion model can only regenerate variants of adapters it was trained on (no conditioning, no cross-task generalization), the win over "just fine-tune another expert" is marginal. The experiment design must distinguish these cases.

---

## 6. The Dense-First Minimal Experiment — Made Concrete

### 6.1 Why "dense first" is right

Testing with a dense (bf16) LoRA delta before attempting ternary removes one variable. If diffusion can't regenerate a continuous LoRA adapter reliably, there is zero hope for the ternary case. Dense first is the correct de-risking order.

### 6.2 The "hello world" that MUST pass before anything else

**Singleton overfitting test:**

1. Take ONE existing trained LoRA adapter from our task-LoRA runs (e.g., the instruction-following adapter already trained in the task-LoRA pipeline).
2. Extract the gate_proj A matrix: shape [2560, 16], ~41K params in bf16 (~80 KB).
3. Train a tiny diffusion model (3-layer MLP, hidden dim 256, T=100 noise steps, simple linear schedule) on this SINGLE sample.
4. Let it massively overfit — we want it to memorize this one vector.
5. Sample 5 adapters from the diffusion. Apply each to the frozen base model (merge LoRA delta, do not ternarize yet).
6. Evaluate: PPL on `HuggingFaceH4/no_robots` test split.

**Pass threshold:** All 5 sampled adapters must achieve PPL within **20%** of the original adapter's eval PPL (not dense base — the adapter we trained).

**Meaning of this test:** Can a diffusion model at all represent and reconstruct a parameter vector in FFN weight space? If it cannot overfit to ONE sample, the entire research direction is dead. This takes 2 hours on CPU (no GPU needed).

**If this passes:** Proceed to the actual smallest experiment below.

### 6.3 The actual smallest viable experiment

**Population assembly:**
- Fine-tune 25 rank-16 LoRA adapters on 25 different 50K-token slices of `HuggingFaceH4/no_robots` (varied by random seed and data window). Target: 25 adapters × gate_proj + up_proj A/B matrices for ONE specific FFN layer (e.g., layer 20, mid-upper half).
- Each fine-tune: ~5–8 minutes on 3060 at rank 16. Total: ~2.5–3.5 hours.
- Reserve 5 adapters as test set.

**Diffusion training:**
- Each training sample: flattened [A.flatten() || B.flatten()] for gate_proj of layer 20 = [2560×16 + 16×6912].flatten() = [151,552] float32 vector.
- Model: 1D U-Net (p-diff architecture) or simple MLP with noise embedding, operating over this 151K-dim space.
- Training: 10K steps, batch size 4 (uses 4 of the 25 training adapters per batch — very small dataset, use aggressive augmentation with Gaussian noise).
- Time: ~30 minutes on 3060 GPU.

**Evaluation:**
- Sample 20 new adapters from the trained diffusion.
- Apply each to frozen base, eval PPL on no_robots test.
- Compare: (a) sampled adapter mean PPL vs (b) training population mean PPL vs (c) dense base PPL.

**Quantitative pass criteria:**
- PASS: Mean sampled PPL ≤ training population mean PPL + 0.5 × (dense_base_PPL - training_population_mean_PPL). In plain language: sampled adapters are at least halfway between base and trained-adapter quality.
- FAIL: Sampled PPL ≥ dense base PPL (diffusion produces adapters no better than random init).
- STRONG PASS: Mean sampled PPL within 10% of training population mean — generalization is working.

**Dense base reference PPL:** Known from prior eval scans: no_robots test ≈ **13.44** (from the `_DENSE_EN_PPL_REF` constant in `mote_train.py`).

---

## 7. Hardware Feasibility

### 7.1 3060 (12 GB VRAM) — suitable for all of the minimal experiment

| Task | VRAM estimate | Time estimate |
|------|--------------|---------------|
| Load BitNet 2B bf16 base model | ~4.5 GB | 30s |
| Rank-16 LoRA training (base + optimizer) | ~8–9 GB with grad checkpoint | 5–8 min per run |
| 25-run population assembly | fits comfortably | 2.5–3.5 hours total |
| Diffusion model training (151K-dim, 25 samples) | <1 GB (model is tiny) | 20–40 min |
| Eval loop (5 samples × no_robots PPL) | ~4.5 GB (no grad) | 5–10 min |
| **Total minimal experiment** | **within 12 GB** | **~4–5 hours** |

**Important:** Do not run while any other GPU job is active. Check `nvidia-smi` first.

### 7.2 What needs Strix or Kaggle

| Task | Why more than 3060 |
|------|--------------------|
| Cross-task population (50+ adapters across 10+ tasks) | Time: 50 × 8 min = 6+ hours. Strix can run multiple CPU-based fine-tune jobs in parallel (Strix has 128 GB unified — can hold multiple model instances). |
| Full-FFN latent autoencoder (Option C) | Memory: VAE over 17.7M param matrices needs >12 GB for reasonable batch. Kaggle T4 16 GB or Strix. |
| MoTE-scale diffusion training (all 3 projections, all 30 layers, 100+ adapters) | Multi-GPU or Kaggle T4×2. |
| Ternary-specific experiments after dense validation | Same as above. |

**Strix coordination:** Confirm with user before launching parallel LoRA fine-tune jobs. Strix runs PyTorch CPU/DirectML — standard HF training works but no CUDA kernel acceleration. LoRA fine-tunes at rank 16 will be slower per job than 3060 but can run many jobs concurrently.

---

## 8. Risks, Failure Modes, and Kill Criteria

### 8.1 Risk inventory

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Training data too small → memorization, no distribution | High (if we stop at 25 samples) | Fatal — diffusion model is useless | Need 50+ diverse samples for real distribution learning; treat 25 as lower bound |
| Post-sampling ternarization destroys functional similarity | Medium | Major — continuous-to-ternary loss can be >20% PPL degradation | Add QE-aware loss term during diffusion training; test ternarization gap explicitly |
| Diffusion mode collapse (all samples near same point) | Medium | Fatal for diversity payoff | Use classifier-free guidance; check sample diversity metric (L2 pairwise distance) |
| 151K-dim latent space too large for simple MLP diffusion | Low (Recurrent Diffusion handles 200M) | Solvable | Use chunked/recurrent architecture if MLP fails |
| Task conditioning requires far more data than available | High | Fatal for Rank-1 payoff | Reduces scope to unconditional generation only |
| Diffusion on LoRA deltas doesn't capture expert specialization | Medium | Major | Validate by checking if sampled adapters show any task specialization |

### 8.2 Explicit kill criteria (stop here, do not continue)

1. **Singleton overfit test fails** (sampled adapters >20% worse PPL than the single training adapter): the approach cannot represent FFN weight space at all. **Stop. Do not proceed.**

2. **After 25-adapter experiment**: Mean sampled PPL ≥ dense base PPL (adapters are no better than random): diffusion learned nothing useful. **Stop.**

3. **Ternarization gap >40%** (sampled adapter PPL continuous vs ternary-quantized differs by >40%): the continuous-to-ternary pipeline is broken. Pivot to discrete diffusion or abandon ternary. Do not continue as-is.

4. **Sample diversity near-zero** (L2 pairwise distance between 10 sampled adapters < 1% of intra-population distance): mode collapse. Tune hyperparameters once; if still collapsed, stop.

5. **Time to 50+ diverse adapters (cross-task) exceeds 20 hours**: reassess whether the training data problem is solvable in this project without dedicated training infrastructure.

---

## 9. Go/No-Go Recommendation

### 9.1 Factors in favor (Go)

- The task-conditioned generation payoff is genuinely novel and has no exact prior art for ternary experts.
- We have all infrastructure: MoTE upcycle, LoRA training, eval scripts, known reference PPL values.
- The minimal experiment is 4–5 hours of wall time on hardware we own (3060).
- The field (ORAL, Recurrent Diffusion) has validated that LoRA-space diffusion works for LLMs — we are applying it to a novel target (ternary expert specialization).
- Even a negative result (diffusion fails for ternary) would be informative and publishable.

### 9.2 Factors against (caution)

- The training data bootstrap problem is unsolved and requires 20–50 LoRA fine-tunes just to get a minimal dataset.
- No prior work on generating quantized/ternary weights via diffusion — high unknowns.
- The gap between generating LoRA deltas and generating useful *expert-diverse* adapters is not obvious.
- The compelling payoff (task-conditioned generation) requires a cross-task adapter population that costs 10–20× more compute than the minimal experiment.

### 9.3 Verdict: Conditional Go

**Run the singleton overfit test first** (2–3 hours, CPU, no GPU conflict). This is a prerequisite with a clear pass/fail. If it passes, invest in the 25-adapter experiment. If that passes, build the case for cross-task population assembly on Strix/Kaggle.

**Do not skip directly to architecture design for the diffusion model** — the training data problem must be resolved in parallel with the technical design.

---

## 10. The Single First Experiment to Run

**If you are reading this and deciding whether to proceed:**

```
EXPERIMENT 0: Singleton overfit test
Hardware:  CPU (or 3060 if free)
Time:      ~2 hours
Dataset:   One existing task-LoRA adapter (instruction-tuning, gate_proj A matrix only)
Model:     3-layer MLP diffusion, hidden=256, T=100 steps, linear noise schedule
Training:  5K steps, single-sample batch, overfit deliberately
Eval:      5 sampled adapters → merged with frozen BitNet base → no_robots test PPL
Pass:      All 5 samples within 20% of original adapter PPL (reference: ~13.44 base PPL)
Kill:      Any sample worse than base PPL (14.x or above) → stop entirely
```

This experiment costs nothing except time. It validates the most primitive claim: a diffusion model can encode and decode a functional parameter vector in FFN weight space.

---

## Appendix A: Detailed Parameter Math

```
gate_proj weight: [6912 × 2560] = 17,694,720 ternary params
up_proj weight:   [6912 × 2560] = 17,694,720 ternary params
down_proj weight: [2560 × 6912] = 17,694,720 ternary params
Total per expert: 53,084,160 ternary params

int8 storage:     53 MB per expert
bf16 master:      106 MB per expert

LoRA rank-16 delta (gate_proj only):
  A matrix: [2560, 16] = 40,960 float32 = 160 KB
  B matrix: [16, 6912] = 110,592 float32 = 432 KB
  Combined: ~151,552 params = 591 KB bf16

Full 3-projection LoRA rank-16 per expert: ~454K params = ~1.8 MB bf16

MoTE with 4 experts × 15 layers = 60 experts:
  LoRA deltas only:  60 × 1.8 MB = ~108 MB (trivial)
  bf16 masters:      60 × 106 MB = ~6.4 GB (fits in VRAM alongside base)
```

## Appendix B: References

- arXiv:2402.13144 — Neural Network Diffusion (p-diff), Wang et al. 2024. [Code: NUS-HPC-AI-Lab/Neural-Network-Diffusion]
- arXiv:2209.12892 — G.pt, Peebles et al. 2022. [Code: wpeebles/G.pt]
- arXiv:2501.11587 — Recurrent Diffusion for Large-Scale Parameter Generation, 2025.
- arXiv:2503.24354 — ORAL: Conditional Recurrent Diffusion for LoRA Generation, 2025.
- arXiv:2402.18153 — D2NWG: Diffusion-Based Neural Network Weights Generation, 2024.
- arXiv:2408.06740 — DiffLoRA: Generating LoRA Weights with Diffusion, 2024.
- `microsoft/bitnet-b1.58-2B-4T` config.json — hidden_size=2560, intermediate_size=6912, num_hidden_layers=30, activation=relu2.
- `scripts/lora/mote_upcycle.py` — MoTEBlock and build_mote implementation.
- `scripts/lora/mote_train.py` — MoTE heal-train loop; reference dense base PPL: 13.44 (no_robots test).
