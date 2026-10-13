# Ternary Conversion Research: bf16 LoRA → Ternary-Only Endpoint

**Date:** 2026-07-05  
**Context:** BitNet-b1.58-2B-4T base; generated bf16 LoRA recovering ~93% of a real adapter's task gain.  
**Goal:** Ternary-only (no bf16 delta at inference) endpoint matching bf16-adapted capability, trained FASTER than ternary-from-scratch by using the generated adapter as a warm start.  
**Hardware:** Single RTX 3060 12 GB.

---

## Problem Framing

The one-jump conversion failure is well-predicted by the literature: merging a ~10% task signal (bf16 LoRA delta) into weights and then applying absmean ternarization produces an error budget roughly 5× larger than the signal. Ternary quantization of a typical LLM weight matrix has an MSE on the order of 40–60% of the weight's variance; the LoRA delta (rank-16 or rank-32, low-magnitude) is disproportionately damaged. Short QAT from this broken init cannot recover because the gradient signal is not strong enough to re-carve signal out of the noise floor.

The literature addresses this from multiple angles — all of which are surveyed below.

---

## Area 1: Warm-Starting QAT from a Higher-Precision Model

### LSQ — Learned Step Size Quantization
**Paper:** "Learned Step Size Quantization" (ICLR 2020)  
**arXiv:** 1902.08153  
**Method:** Adds a learnable per-layer (or per-channel) step size parameter to the quantizer. The step-size gradient is estimated directly through the rounding operation. The step size is initialized from the first batch of data (or from weight statistics: `2·E[|w|] / sqrt(Q_p)` where Q_p is the positive clipping level). QAT always starts from a full-precision pretrained model — the step size initialization is itself a form of warm-starting.  
**Demonstrated:** Achieves FP32 baseline accuracy on ImageNet at 3-bit weights + 3-bit activations. At 2-bit, degradation exists but is measurably smaller than other methods of the era.  
**Relevance here:** LSQ is a component technique. If we do QAT on the merged (ternary init) model, using learned absmean step sizes instead of fixed ones could reduce the static quantization bias that causes the plateau. However, BitNet already uses absmean (fixed per-tensor scale), and LSQ would require changing the inference path unless scales are frozen after training. Thin evidence that LSQ alone closes a 5× signal/noise problem.

### PACT — Parameterized Clipping Activation
**Paper:** "PACT: Parameterized Clipping Activation for Quantized Neural Networks" (2018)  
**arXiv:** 1805.06085  
**Method:** Learns a clipping parameter α for activations during QAT, avoiding the saturation problem at low bit-widths. Weights are still quantized with fixed rounding.  
**Demonstrated:** 4-bit weights + 4-bit activations match FP32 on ImageNet for AlexNet/ResNet. Works best when combined with learned weight step sizes (like LSQ).  
**Relevance here:** BitNet uses 8-bit activations already; the main quantization error in our problem is ternary weights. PACT addresses activations. Indirect relevance: the key lesson is that letting the clipping boundary be a trainable parameter dramatically stabilizes QAT convergence. For BitNet fine-tuning, the equivalent would be allowing the absmean scale to train (making it more like LSQ). Not a standalone fix.

### DoReFa-Net — Low Bitwidth Weights, Activations, and Gradients
**Paper:** "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients" (2016)  
**arXiv:** 1606.06160  
**Method:** Quantizes weights, activations, and *gradients* to low bit-widths during training. Enables training-time simulation of full quantization pipeline. Not a warm-start method per se — it trains from scratch or from FP init with quantization on from step 1.  
**Demonstrated:** 1-bit weights + 2-bit activations from ImageNet training (from FP init), achieving 46.1% top-1 on ImageNet with AlexNet. The FP init matters: from-scratch 1-bit training was significantly worse in the 2016 era.  
**Relevance here:** Establishes the foundational STE (Straight-Through Estimator) approach that all subsequent work builds on. The lesson: starting QAT from a FP pretrained checkpoint (warm start) was crucial even in 2016 for binary/ternary models. Directly relevant precedent.

### Compute-Optimal QAT
**Paper:** "Compute-Optimal Quantization-Aware Training" (2025)  
**arXiv:** 2509.22935  
**Method:** Derives scaling laws predicting the optimal split of compute between full-precision pretraining and QAT fine-tuning. Training protocol: FP phase → QAT fine-tuning with learning rate re-warmup. Proposes a "cooldown and QAT fusion" approach to eliminate redundant updates.  
**Demonstrated:** The optimal ratio of QAT-to-FP compute *increases* with total compute budget. The warm-start protocol (FP init → QAT) consistently outperforms QAT-only. Provides tokens-per-parameter-byte statistic for predicting optimal QAT fractions.  
**Relevance here:** Directly validates our hypothesis that warm-starting QAT from the bf16-adapted model is better than ternary-from-scratch. The paper gives a principled method to estimate how many QAT steps are needed given our compute budget (RTX 3060 time constraints).

### Reasoning-QAT — Systematic Study
**Paper:** "What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study" (2026)  
**arXiv:** 2601.14888  
**Method:** Ablates four factors: (1) KD objective, (2) PTQ initialization, (3) RL feasibility, (4) domain alignment between PTQ calibration and QAT training data.  
**Demonstrated:** PTQ initialization (using a GPTQ/AWQ quantized checkpoint as the QAT starting point rather than the FP model) improves accuracy *while reducing training cost* — the closest thing to a controlled warm-start-vs-cold-start experiment in the LLM QAT literature. On Qwen3-0.6B at 2-bit, outperforms GPTQ by 44.53% on MATH-500. KD is "robust" regardless of training regime.  
**Relevance here:** This is highly relevant. The PTQ-init insight maps cleanly onto our scenario: use GPTQ on the merged bf16-adapted weights (or the adapted + ternarized weights) as a warm start rather than raw random-ternary. The convergence benefit is real and documented.

**Scaling Laws for Precision caveat:**  
**Paper:** "Scaling Laws for Precision" (2024)  
**arXiv:** 2411.04330  
**Key finding:** For long-horizon *pretraining* from scratch, lower-precision training may be compute-optimal because PTQ degradation worsens as pretraining extends. This is a concern for our scenario only if we're doing tens of billions of tokens; for fine-tuning a domain adapter (thousands to millions of tokens), the warm-start advantage dominates.

**Summary for Area 1:** Strong evidence that warm-starting QAT from FP is better than from-scratch ternary. The critical question is *which* FP checkpoint to warm-start from — the unadapted base, the bf16-adapted model, or a GPTQ-quantized version of the adapted model. Literature (Reasoning-QAT) suggests PTQ-quantized adapted weights as the QAT init gives the best of both worlds.

---

## Area 2: Progressive / Gradual Quantization

### GPTQ — Block-wise Second-Order Post-Training Quantization
**Paper:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)  
**arXiv:** 2210.17323  
**Method:** Layer-by-layer (block-wise) second-order (Hessian-based) weight quantization. Quantizes one weight at a time within a block, compensating unquantized weights to reduce cumulative error. No gradient descent — PTQ only.  
**Demonstrated:** 175B GPT models quantized to 3-4 bit in ~4 GPU-hours with "negligible accuracy degradation." 2-bit shows meaningful perplexity increase but remains usable.  
**Relevance here:** GPTQ on the bf16-adapted model gives the best available initialization for subsequent QAT. It's the "best one-shot starting point" before any fine-tuning. Memory-efficient to run on 12 GB at 2B scale. Does not give ternary directly (GPTQ targets INT2/3/4), but the principle of Hessian-compensated block updates applies.

### AWQ — Activation-Aware Weight Quantization
**Paper:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (MLSys 2024 Best Paper)  
**arXiv:** 2306.00978  
**Method:** Identifies salient weight channels via activation magnitude statistics. Applies per-channel scaling to protect those 1% of critical weights, then quantizes uniformly. No backpropagation needed.  
**Demonstrated:** Strong 4-bit results; competitive with GPTQ at 3-bit. Better generalization than GPTQ on out-of-distribution tasks due to not overfitting calibration set.  
**Relevance here:** Alternative to GPTQ for the PTQ initialization step. AWQ is faster (no second-order Hessian) and may generalize better for task-adapted weights. Worth trying as QAT warm-start init alongside GPTQ.

### EfficientQAT — Block-wise QAT
**Paper:** "EfficientQAT: Efficient Quantization-Aware Training for Large Language Models" (2024)  
**arXiv:** 2407.11062  
**Method:** Two phases: (1) Block-AP — block-wise training of all parameters, one block at a time, avoiding full-model gradient computation; (2) E2E-QP — end-to-end training of quantization step sizes only. Enables 2-bit QAT on large models with limited memory.  
**Demonstrated:** 2-bit Llama-2-70B at 69.48 accuracy vs 72.41 full-precision (3-point gap at 2-bit). Requires 41 hours on a single A100-80GB. For 2B-scale, block-wise approach fits on a 3060 12GB.  
**Relevance here:** The block-wise training approach is directly applicable to our memory-constrained setup. Instead of fitting the full 2B model's QAT in 12 GB (which barely fits), do one transformer block at a time. Each block's QAT takes a fraction of total VRAM. The block-by-block approach also localizes the quantization error correction, which is particularly important for ternary conversion.

### Bit-by-Bit — Progressive QAT with Outlier Channel Splitting
**Paper:** "Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs" (ACL 2026)  
**arXiv:** 2604.07888  
**Method:** Staged bit-width reduction (8-bit → 4-bit → 2-bit), with each stage warm-starting from the previous. Adds "outlier channel splitting" — detects high-magnitude channels and halves them (with paired doubling elsewhere) to suppress rounding noise without changing the quantized output value. Addresses convergence instability from accumulated quantization noise.  
**Demonstrated:** Stable convergence to 2-bit for LLMs where one-shot QAT fails. Published at ACL 2026, suggesting state-of-the-art results at the time.  
**Relevance here:** The staged approach directly addresses our "5× signal/noise" problem. Instead of bf16-adapted → ternary (one jump), do: bf16-adapted → 8-bit QAT (stable) → 4-bit QAT (warm-started from 8-bit) → ternary QAT (warm-started from 4-bit). Each step halves the quantization error jump the model must absorb. The outlier channel splitting is compatible with BitNet's absmean scheme.

### UPQ — Unified Progressive Quantization
**Paper:** "Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs" (2026)  
**arXiv:** 2506.09104  
**Method:** Three-stage pipeline — FP16 → INT4 (block-wise PTQ) → INT2 (distillation-based QAT). Minimizes generalized Jensen-Shannon divergence (JSD) between teacher and quantized student logits. Designed for instruction-tuned models without proprietary training data.  
**Demonstrated:** State-of-the-art on MMLU and IFEval at 2-bit quantization. Combines the PTQ-init advantage (from Reasoning-QAT) with distillation-based QAT.  
**Relevance here:** The FP→INT4→INT2 cascade is the closest published precedent to what we want (bf16-adapted → ternary). The JSD distillation loss is a better-calibrated version of standard KL for generation tasks. Does not go to ternary, but ternary (~1.58-bit) can be viewed as a slightly more aggressive 2-bit.

### HuggingFace Lambda Scheduler for Gradual Ternarization
**Source:** "Fine-tuning LLMs to 1.58bit: extreme quantization made easy" (Hugging Face blog, 2024)  
**Not a paper, but a directly applicable recipe:**  
The approach introduces a dynamic `lambda_` parameter: `lambda_ = min(training_step / 1000, 1)`. The quantized output is computed as `lambda_ * quantized_weight + (1 - lambda_) * fp_weight`. This allows the model to start training with full FP weights and gradually commits to ternary over 1000 steps. Prevents catastrophic loss of pretrained knowledge.  
**Demonstrated:** Llama3-8B quantized to 1.58-bit with 10B tokens on FineWeb-edu outperformed BitNet 7B (trained on 100B tokens) on MMLU. Implemented in HuggingFace Nanotron (PR #180) and Transformers (PR #33410).  
**Relevance here:** This is the most directly applicable technique for our exact problem. The lambda scheduler can be applied to our bf16-adapted model, running the ternary ramp-up on task-specific data (or even the same data used to generate the LoRA adapter). The ramp-up is essentially a soft-to-hard annealing: the model learns ternary quantization without being shocked by it.

**Temperature annealing / soft-to-hard quantization:**  
The broader family (see arXiv:2104.05168 "Soft then Hard: Rethinking the Quantization in Neural Image Compression") uses a Gumbel-softmax relaxation with temperature annealing to achieve differentiable discrete assignments. The lambda scheduler above is a simpler linear version of this principle. For BitNet specifically, the lambda scheduler is the right tool — the Gumbel approach adds complexity with marginal benefit at ternary bit-width.

**Summary for Area 2:** Progressive quantization (staged bit-width reduction with warm starts between stages) consistently outperforms one-shot conversion. Key finding: block-wise execution (EfficientQAT) enables this on memory-constrained hardware. The HuggingFace lambda scheduler is the only published recipe specifically for ternary ramp-up from an FP checkpoint.

---

## Area 3: Making a Small Adaptation Survive Extreme Quantization

### LoftQ — LoRA-Fine-Tuning-Aware Quantization
**Paper:** "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models" (NeurIPS 2023 workshop, arXiv Oct 2023)  
**arXiv:** 2310.08659  
**Method:** Jointly quantizes the base LLM weights AND finds a low-rank initialization for LoRA adapters via alternating SVD. The idea: instead of quantizing and then finding a random LoRA init, solve for the quantized weight + low-rank complement such that their sum approximates the original FP weight. Minimizes the approximation error at the initialization stage.  
**Demonstrated:** Outperforms QLoRA significantly, especially at 2-bit and 2/4-bit mixed precision, on NLU, QA, summarization, and NLG tasks.  
**Relevance here:** This solves the *initialization* problem for a quantized base model. In our scenario, we need the inverse: starting from a bf16-adapted model and finding a ternary weight + LoRA complement that approximates the bf16 weights. LoftQ's alternating SVD is directly applicable: run it on the merged `W_base + ΔW_LoRA` matrix to find `Q(W_merged) + A·B` where Q(·) is absmean ternary quantization. This gives a better ternary init than naive round-and-ternarize, and the A·B component can be further fine-tuned or zeroed post-QAT. **This is likely the best initialization technique for our specific problem.**

### LQ-LoRA — Low-Rank Plus Quantized Matrix Decomposition
**Paper:** "LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning" (2023)  
**arXiv:** 2311.12023  
**Method:** Decomposes each pretrained weight matrix into a frozen quantized component + a trainable low-rank component. The quantized component uses integer linear programming to optimally allocate bit-widths per matrix within a memory budget. The low-rank component is initialized via SVD on the quantization error (similar to LoftQ but with dynamic bit allocation).  
**Demonstrated:** Aggressive quantization to sub-3 bits with "only minor performance degradations." A 2.75-bit LLaMA-2-70B model fits in 27 GB. Outperforms QLoRA and GPTQ-LoRA baselines.  
**Relevance here:** The SVD-of-quantization-error initialization is crucial. If we apply LQ-LoRA's initialization to our adapted model: decompose `W_merged = Q_ternary(W_merged) + SVD(W_merged - Q_ternary(W_merged))`, then the low-rank component captures the adaptation signal that ternary quantization dropped. The question is whether ternary (3-value) is too aggressive for the SVD residual to converge — this is untested territory but theoretically sound.

### QA-LoRA — Quantization-Aware Low-Rank Adaptation
**Paper:** "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models" (ICLR 2024)  
**arXiv:** 2309.14717  
**Method:** Uses group-wise quantization (more degrees of freedom than per-tensor) combined with LoRA to ensure the adapter weights can be merged into the quantized base without a precision mismatch. After training, the LLM + auxiliary weights integrate naturally into the quantized model.  
**Demonstrated:** Enables efficient INT4 inference without keeping LoRA adapters separate. Resolves the "bf16 delta at inference" problem that our goal targets.  
**Relevance here:** The "merge without precision mismatch" property is exactly what we need. However, QA-LoRA operates at 4-bit, not ternary. The group-wise quantization approach may not generalize to absmean ternary, which has a fixed 3-value codebook. Partial relevance: the principle of designing LoRA updates that align with the target quantization grid is directly applicable.

### RILQ — Rank-Insensitive LoRA Quantization Error Compensation
**Paper:** "RILQ: Rank-Insensitive LoRA-based Quantization Error Compensation for Boosting 2-bit Large Language Model Accuracy" (AAAI 2025)  
**arXiv:** 2412.01129  
**Method:** Uses a "model-wise activation discrepancy loss" that is theoretically rank-insensitive — meaning even very low-rank LoRA adapters can effectively compensate for quantization error if trained with this loss. The adapters adjust cooperatively across layers rather than independently.  
**Demonstrated:** Consistent improvements on LLaMA-2 and LLaMA-3 at 2-bit quantization; task-specific fine-tuning on quantized models also benefits. Compatible with existing quantizers.  
**Relevance here:** The rank-insensitivity property is critical for memory-constrained hardware. If a rank-8 or rank-16 adapter can effectively compensate ternary quantization error, the training cost is much lower than full QAT. The cooperative (cross-layer) loss function is more aligned with how quantization error propagates than per-layer reconstruction. Tested at 2-bit, not ternary, but ternary is approximately 1.58-bit — very close. The gap might be larger.

### LoTA-QAF — Lossless Ternary Adaptation for QAT Fine-Tuning
**Paper:** "LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning" (NeurIPS 2025)  
**arXiv:** 2505.18724  
**Method:** Three components: (1) Ternary Adaptation (TA) — adaptation weights are ternary-valued themselves, aligned to the quantization grid; (2) lossless merging — because both base and adapter are ternary, the merge is exact with no precision truncation; (3) t-SignSGD — ternary signed gradient descent for updating the TA weights.  
**Demonstrated:** Applied to Llama-3.1/3.3 and Qwen-2.5 families. Surpasses 16-bit LoRA on MMLU by up to 5.14% for *quantized* base models. The improvement over standard LoRA on quantized bases is because standard LoRA's bf16 adapter introduces a precision mismatch on merge.  
**Relevance here:** **This is the most directly relevant paper to our goal.** LoTA-QAF ensures that post-training, the model is purely ternary with no bf16 delta. The ternary adapter aligns with the base model's ternary quantization grid. Two caveats: (a) the paper tests on 4-bit quantized bases (Llama/Qwen with 4-bit GPTQ), not on BitNet's native ternary; (b) t-SignSGD is a new optimizer requiring implementation. But the concept maps: initialize the ternary adapter from SVD of the bf16 LoRA delta, then fine-tune using t-SignSGD on task data.

### Recover-LoRA — LoRA + KD for Aggressive Quantization Recovery
**Paper:** "Recover-LoRA for Aggressive Quantization: Reclaiming Accuracy in 2-Bit Language Models via Low-Rank Adaptation with Knowledge Distillation on Synthetic Data" (2026)  
**arXiv:** 2606.04238  
**Method:** Post-quantization recovery: trains LoRA adapters on the aggressively quantized model using logit distillation from the full-precision model. Uses synthetic data (10k samples model-generated). Selective mixed-precision: only 2-bit quantizes certain projection layers.  
**Demonstrated:** 80–95% accuracy recovery on 9/12 benchmarks for Qwen3-4B at 2-bit, using only 10k synthetic training samples. From AMD, tested on 4B–20B models.  
**Relevance here:** Very applicable as a lightweight recovery step. After our ternary one-jump conversion fails, Recover-LoRA style training — LoRA on the ternary model, distilling from bf16-adapted teacher — could recover the 80–95% accuracy with minimal compute. BUT note: this produces a ternary base + bf16 LoRA adapter, which is not a ternary-only endpoint. The LoRA adapter must then be merged (QA-LoRA/LoTA-QAF approach) or the base must be further QAT'd to absorb the signal.

**Summary for Area 3:** A pipeline combining LoftQ's initialization + LoTA-QAF's ternary adapter + distillation training is likely the best approach. The chain is: (1) LoftQ on merged bf16 weights → ternary init + low-rank complement; (2) train the low-rank complement as a ternary adapter (LoTA-QAF style) using KD from bf16-adapted teacher; (3) merge. Evidence is strong for each component individually but the combined pipeline has not been published for BitNet-scale ternary models.

---

## Area 4: BitNet / 1.58-bit Specific Fine-Tuning

### BitNet b1.58 2B4T — Native Ternary Training Recipe
**Paper:** "BitNet b1.58 2B4T Technical Report" (Microsoft, April 2025)  
**arXiv:** 2504.12285  
**Method:** The model was trained natively in 1.58-bit from scratch over three phases: large-scale ternary pretraining, supervised fine-tuning (SFT), and direct preference optimization (DPO). BitLinear layers ternarize weights via absmean quantization at every forward pass during training. STE passes gradients through the quantization step.  
**Key architecture details:** BitLinear replaces nn.Linear. During training, weights are stored in FP32 but are ternarized (absmean → {-1,0,1} × scale) for the forward pass. Gradients flow through STE. Scale is per-tensor, not per-row.  
**Relevance here:** The SFT phase of BitNet b1.58 2B4T IS ternary fine-tuning. The model fine-tunes task capability natively in ternary without needing any bf16 adapter — the adaption is absorbed into the ternary weights through continued gradient descent. This is the gold standard "ternary-from-scratch" approach we are trying to beat with warm-starting.

### BitNet Distillation (BitDistill)
**Paper:** "BitNet Distillation" (Microsoft, October 2025)  
**arXiv:** 2510.13998  
**Method:** A three-technique pipeline to distill a full-precision LLM into 1.58-bit ternary: (1) SubLN module from BitNet architecture; (2) multi-head attention distillation based on MiniLM (intermediate feature alignment, not just logit matching); (3) continual pre-training warm-up step to address performance gaps before distillation.  
**Demonstrated:** "Performance comparable to full-precision counterpart models across model size." 10× memory savings, 2.65× faster CPU inference.  
**Relevance here:** The three-step recipe is directly applicable: (a) apply SubLN to our bf16-adapted model; (b) use the bf16-adapted model as the teacher for MiniLM-style attention distillation into the ternary student; (c) the continual pre-training warm-up maps to our "run QAT on task data" step. The key difference from our problem: BitDistill starts from a full-precision FP model, not a LoRA-adapted one. But since our bf16-adapted model IS the teacher, the pipeline maps cleanly.

### HuggingFace 1.58-bit Fine-Tuning Recipe
**Source:** "Fine-tuning LLMs to 1.58bit: extreme quantization made easy" (HuggingFace blog, 2024)  
**Implementation:** HuggingFace Nanotron (PR #180), Transformers (PR #33410)  
**Method:** Lambda scheduler for gradual ternary ramp-up:
```python
lambda_ = min(training_step / 1000, 1)
effective_weight = lambda_ * absmean_ternary(w) + (1 - lambda_) * w
```
Trains Llama3-8B on 10B tokens (FineWeb-edu) with lr=1e-4, batch=2M tokens.  
**Demonstrated:** After 10B tokens, the resulting model outperforms BitNet 7B (trained on 100B tokens) and surpasses FBI LLM (1.26T tokens) on MMLU. The ratio of efficiency gain is striking: ~10× fewer tokens needed for comparable capability when warm-starting from FP vs. native ternary from scratch.  
**Relevance here:** **This is the single most applicable published recipe.** Apply it to our bf16-adapted model:
- Replace Linear → BitLinear in the adapted model
- Run lambda-scheduled ternary ramp-up on task data for ~1000 steps before full commitment
- Continue QAT on task data (or task-adjacent data) after lambda=1
- The 10B vs 100B token efficiency ratio is the quantitative answer to "does warm-starting beat from-scratch": yes, ~10×.

### QVAC Fabric — LoRA Fine-Tuning BitNet b1.58 on Edge
**Source:** "LoRA Fine-Tuning BitNet b1.58 LLMs on Heterogeneous Edge GPUs via QVAC Fabric" (HuggingFace blog, 2025/2026)  
**Method:** Applies LoRA to frozen BitNet b1.58 base weights (ternary, TQ1_0/TQ2_0 format). LoRA adapters remain in FP16 during training and inference — the base is ternary, the adapter is not. Uses AdamW + linear LR decay, sequence length 512.  
**Demonstrated:** Successfully fine-tuned a 13B BitNet model on an iPhone 16 (A17 GPU). Instruction tuning on assistant responses.  
**Relevance here:** This is NOT a ternary-only endpoint — it's a ternary base + bf16 LoRA approach, which is what we're trying to avoid. However, it confirms the engineering feasibility of adapting BitNet on memory-constrained hardware. The approach can serve as a fast upper-bound checkpoint: run this to get the best possible adapted model, then apply LoTA-QAF / ternary distillation to convert the bf16 adapter into ternary.

### Falcon-Edge — 1.58-bit Models with Instruction Tuning
**Source:** Falcon-Edge blog (TII / TGIDM, 2026)  
**Models:** Falcon-E-1B and Falcon-E-3B (base + instruction-tuned)  
**Method:** "New pre-training paradigm that delivers a full-scope output from a single training process." Standard fine-tuning with HuggingFace `trl` library, then re-quantization back to ternary.  
**Demonstrated:** Falcon-E-3B achieved 18.32 average score vs. 18.19 for its bf16 counterpart — 0.13-point degradation from ternary. This is remarkably close to parity.  
**Caveat:** The blog explicitly states PEFT for BitNet models "remains an exciting and impactful open question." No PEFT recipe is provided.  
**Relevance here:** The 0.13-point gap confirms that native ternary fine-tuning CAN match bf16 near-perfectly — the question is how many tokens are required, and Falcon-Edge suggests it's feasible at 3B scale.

**Summary for Area 4:** Three published BitNet fine-tuning approaches exist: (1) native ternary SFT (BitNet b1.58 2B4T — gold standard but slow); (2) FP→ternary distillation (BitDistill); (3) lambda-scheduled ramp-up (HuggingFace blog — most efficient, 10× token advantage over from-scratch). No published ternary LoRA merge recipe exists yet, making this a gap in the literature.

---

## Area 5: Distillation as the Bridge

### LLM-QAT — Data-Free Quantization Aware Training
**Paper:** "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models" (2023)  
**arXiv:** 2305.17888  
**Method:** Uses model-generated data (from the pretrained FP model itself) for QAT distillation. Teacher: full-precision LLM. Student: quantized LLM. Distillation loss = KL divergence on token probabilities. Also quantizes KV cache. No original training data needed.  
**Demonstrated:** LLaMA 7B, 13B, 30B at 4-bit weights + 4-bit activations + 4-bit KV cache. Enables quantization of models where training data is unavailable.  
**Relevance here:** Exactly our scenario: teacher = bf16-adapted model, student = ternary model being QAT'd. Synthetic data from the bf16-adapted teacher is perfectly suited (it captures the task distribution). The distillation loss is superior to plain LM-loss QAT because it transfers the teacher's *output distribution*, not just the training token labels. Recovers more task gain than LM-loss alone.

### BitDistiller — CAKLD Self-Distillation for Sub-4-bit
**Paper:** "BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation" (ACL 2024)  
**arXiv:** 2402.10631  
**Method:** Combines QAT + KD with a novel loss: Confidence-Aware KL Divergence (CAKLD). Teacher and student are the same model architecture (self-distillation — the FP model teaches the quantized model). CAKLD weights the KL loss by the teacher's confidence (entropy), focusing on tokens where the teacher is certain. Uses asymmetric quantization + clipping for better weight fidelity.  
**Demonstrated:** Significantly surpasses existing methods at 3-bit and 2-bit on language understanding and complex reasoning. "More cost-effective, demanding fewer data and training resources."  
**Relevance here:** CAKLD is a better distillation loss than plain KL for our use case. The confidence-weighting concentrates the learning signal on tokens where the bf16-adapted teacher is decisive, which aligns with where task adaptation matters most. On a 3060 12GB at 2B scale, this is feasible without the "self-distillation" overhead (teacher and student fit simultaneously at 2B scale in different precision).

### Reasoning-QAT — KD + PTQ Init Pipeline
**Already described in Area 1:** The finding that KD is "a robust objective for reasoning models" regardless of training regime, combined with PTQ initialization as warm-start, is the most evidence-backed recipe for our target scenario.

### Recover-LoRA — LoRA + Logit Distillation on Synthetic Data
**Already described in Area 3:** 80–95% accuracy recovery on 2-bit quantized models using LoRA + logit distillation with 10k synthetic samples. Lightweight, fast.

### NVFP4 Quantization-Aware Distillation
**Paper:** "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery" (2026)  
**arXiv:** 2601.20088  
**Method:** NVIDIA's approach to recover FP4 quantization loss: minimize forward KL divergence (teacher logit → student logit) during QAT, applied specifically to the output logit distribution. Uses Muon optimizer for distillation.  
**Demonstrated:** Strong recovery of accuracy in FP4 models; outperforms LM-loss QAT.  
**Relevance here:** FP4 ≈ ternary in quantization aggressiveness. The forward KL (teacher→student) distillation during QAT is the standard recipe, and this paper confirms it's the right objective at very low bit-widths. The Muon optimizer may be worth evaluating alongside AdamW for ternary QAT convergence.

**Key question: Does distillation beat plain LM-loss QAT for ternary?**

Evidence from multiple papers (LLM-QAT, BitDistiller, Reasoning-QAT, NVFP4 QAD) consistently says **yes**: KD-augmented QAT outperforms plain LM-loss QAT at 2-bit and below. The gap widens as quantization becomes more aggressive. At ternary (~1.58-bit), this gap is large enough that LM-loss QAT alone is likely inadequate to recover the task adaptation signal. Distillation from the bf16-adapted teacher is not optional — it is the standard route.

**Summary for Area 5:** Distillation from bf16-adapted teacher to ternary student is strongly validated as the correct approach. Use CAKLD (BitDistiller) or plain KL (LLM-QAT) as the primary loss, with optional intermediate feature alignment (attention maps, MiniLM-style, as in BitDistill). Data-free operation (using teacher-generated synthetic data) is viable and removes the dependency on task-labeled data.

---

## Honest Assessment: What the Literature Does NOT Cover

1. **BitNet absmean + LoRA merge at ternary:** No paper has published a recipe for merging a bf16 LoRA adapter directly into absmean-ternarized weights while preserving task accuracy. LoTA-QAF (2505.18724) comes closest but is tested on 4-bit GPTQ bases, not BitNet ternary.

2. **Signal/noise ratio at ternary:** The quantitative claim "~10% task gain vs ~50% ternary error" is our empirical observation, not from literature. The literature gives only general guidance (more KD helps, staged reduction helps) without quantifying the threshold below which recovery fails.

3. **t-SignSGD for BitNet:** LoTA-QAF's ternary optimizer is not validated on BitNet's absmean quantization scheme. May need adaptation.

4. **Memory budget for combined approaches:** EfficientQAT's 41-hour A100 estimate for 70B does not extrapolate cleanly to 3060 + 2B. Block-wise QAT for 2B on 12GB should be feasible in 1–4 hours per block depending on sequence length, but this is an estimate.

---

## Ranked Shortlist: Top 5 Techniques

### Rank 1: KD-Augmented QAT with PTQ Warm-Start + Lambda Scheduler

**Evidence:** LLM-QAT (2305.17888), BitDistiller (2402.10631), Reasoning-QAT (2601.14888), HuggingFace 1.58-bit recipe  
**How to apply:**  
1. Start from the bf16-adapted model (the LoRA already merged into bf16 weights).  
2. Apply GPTQ or AWQ to get a well-quantized (4-bit or even ternary) initialization — this is the PTQ warm-start.  
3. Apply the HuggingFace lambda scheduler: ramp `lambda_` from 0 → 1 over 1000 steps, making the transition gradual.  
4. Use CAKLD (BitDistiller) as the training loss: the bf16-adapted model is the teacher, the ternary model is the student.  
5. Train on task-adjacent data (or teacher-generated synthetic data via LLM-QAT recipe) for 2,000–10,000 steps.  
6. Optional: include MiniLM-style intermediate attention distillation (BitDistill approach) for deeper feature alignment.  

**Expected payoff:** Highest expected accuracy recovery. The combination of PTQ warm-start + gradual ternarization + distillation is the convergence of all five evidence streams. The HuggingFace recipe already demonstrates ~10× token efficiency over native ternary training. With the bf16-adapted model as teacher (instead of the unadapted base), the task signal should be substantially preserved. RTX 3060 feasibility: the teacher can be offloaded to CPU or run in 4-bit while the student trains in ternary QAT mode.

**Expected timeline:** 1,000–5,000 steps at batch size 8–16 (sequence length 512). Estimated 4–12 hours on RTX 3060 for 2B model.

---

### Rank 2: LoftQ Initialization Applied to Merged Weights

**Evidence:** LoftQ (2310.08659), LQ-LoRA (2311.12023)  
**How to apply:**  
1. Merge the bf16 LoRA delta into the base weights: `W_merged = W_base + A·B`.  
2. Apply LoftQ's alternating SVD algorithm to `W_merged`: find `Q_ternary(W_merged) + A'·B'` that minimizes `||W_merged - Q_ternary - A'B'||_F`. Use 5–10 alternating iterations.  
3. This gives: (a) a ternary weight matrix `Q_ternary(W_merged)` that incorporates the adaptation signal; (b) a low-rank residual `A'·B'` that captures what ternary quantization dropped.  
4. Option A: Zero out `A'·B'` and continue with pure ternary QAT (Rank 1 recipe) from this better init.  
5. Option B: Keep `A'·B'` as a LoTA-QAF ternary adapter and train it with t-SignSGD + distillation until the adapter is negligible.  

**Expected payoff:** Better initialization than naive absmean ternarization of `W_merged`. LoftQ's alternating SVD explicitly redistributes the task signal from the low-rank adapter into the quantized matrix's available capacity. At 2B scale with ternary quantization, the rank of the informative subspace is small enough that SVD captures most of it. This is a nearly zero-cost improvement over the current one-jump approach — the SVD initialization adds one preprocessing step with no additional training required.  

**Caveat:** LoftQ was validated at 2-bit and 4-bit, not ternary. The alternating SVD convergence at ternary bit-width (3 discrete values) is theoretically valid but may require more iterations. Implementation exists at github.com/yxli2123/LoftQ.

**Expected timeline:** Preprocessing step only (minutes on RTX 3060); then feed into Rank 1 recipe for QAT.

---

### Rank 3: Block-wise Progressive QAT (EfficientQAT + Bit-by-Bit Cascade)

**Evidence:** EfficientQAT (2407.11062), Bit-by-Bit (2604.07888), UPQ (2506.09104)  
**How to apply:**  
1. Start from bf16-adapted model.  
2. Stage 1: Block-wise 8-bit QAT — using EfficientQAT's Block-AP approach, quantize one transformer block at a time to INT8 while fine-tuning only that block's parameters (16–32 blocks, each taking ~10 minutes on 3060).  
3. Stage 2: Block-wise 4-bit QAT — repeat from the Stage 1 checkpoint.  
4. Stage 3: Ternary QAT — repeat from Stage 2 checkpoint with lambda scheduler and distillation (Rank 1).  

**Expected payoff:** Avoids the catastrophic shock of one-jump conversion. Each stage's QAT loss is ~2–5× smaller than the jump from bf16 to ternary directly. The block-wise execution keeps peak VRAM at ~6–8 GB even for full-parameter fine-tuning. The Bit-by-Bit paper demonstrates that this staged approach enables convergence where one-shot QAT fails.  

**Caveat:** Total training time is 3× longer than single-stage QAT. On the 3060, the three-stage pipeline may take 12–36 hours. If the ternary step still fails, it indicates the fundamental signal/noise ratio is the bottleneck, not initialization quality.

---

### Rank 4: LoTA-QAF — Ternary Adapter with Lossless Merge

**Evidence:** LoTA-QAF (2505.18724)  
**How to apply:**  
1. Start from the bf16-adapted model.  
2. Apply standard absmean ternarization to get the ternary base (this is our current failing step — proceed despite poor accuracy).  
3. Initialize a ternary adapter (`A'·B'` where the values are {-1,0,1}) by ternarizing the SVD of the adaptation component (the difference between merged and base ternary weights).  
4. Train the ternary adapter with t-SignSGD using KD loss from bf16-adapted teacher.  
5. Merge: ternary base + ternary adapter → sum → re-ternarize → pure ternary model.  

**Expected payoff:** Solves the "no bf16 delta at inference" constraint cleanly. The ternary adapter can capture the task adaptation within the ternary grid, and the merge is lossless. The 5.14% MMLU gain over bf16 LoRA on quantized bases is encouraging. However, LoTA-QAF was not tested on BitNet's absmean ternary — it was tested on 4-bit GPTQ bases. The extrapolation to ternary introduces uncertainty.  

**Caveat:** t-SignSGD requires implementation (not in HuggingFace mainline). The convergence of ternary-valued adapters on ternary bases has very thin published evidence.

---

### Rank 5: Recover-LoRA Style — Quick Accuracy Floor

**Evidence:** Recover-LoRA (2606.04238), LLM-QAT (2305.17888)  
**How to apply:**  
1. Apply naive absmean ternarization to the bf16-adapted model (current failing step).  
2. Train a standard bf16 LoRA adapter on the ternary model using logit distillation from the bf16-adapted teacher, with 10k teacher-generated synthetic samples.  
3. This gives: ternary base + bf16 LoRA (NOT a ternary-only endpoint, but a fast recovery baseline).  
4. Use this checkpoint as the teacher for a subsequent LoTA-QAF / Rank 1 run.  

**Expected payoff:** Recover 80–95% of accuracy in a lightweight pass (a few hours on 3060). Does NOT achieve the ternary-only endpoint directly, but establishes an upper bound on what's recoverable and serves as a diagnostic tool. If Recover-LoRA cannot recover >80%, the task signal may be too fragile for any ternary approach, and the effort should shift to reducing the LoRA rank (less delta to survive quantization).

---

## Synthesis: Recommended Execution Order

```
Step 0 (1 hour):   LoftQ alternating SVD on W_merged → better ternary init
Step 1 (4–12 hr):  Rank 1 (KD-QAT + lambda scheduler) from LoftQ init
    - If accuracy > 85% of bf16-adapted target: DONE (ternary-only endpoint achieved)
    
Step 2 (4 hr):     If Step 1 fails, run Recover-LoRA (Rank 5) as diagnostic
    - Establishes recovery ceiling; confirms whether >80% is achievable at all
    
Step 3 (12–36 hr): If Step 1 partially works, try Rank 3 (staged block-wise cascade)
    - More compute but higher probability of convergence

Step 4 (ongoing):  Implement LoTA-QAF (Rank 4) if a ternary adapter approach
                   converges better than full QAT on this signal/noise ratio
```

The most important single technique is the distillation objective (CAKLD or plain KL from bf16-adapted teacher). Without it, any amount of QAT will plateau below the unadapted base on the task. With it, the HuggingFace lambda scheduler recipe gives strong evidence that the bf16 → ternary conversion can be done in <10B tokens with near-parity accuracy.

---

## Key Paper Reference Table

| Paper | arXiv ID | Year | Core Contribution | Area |
|-------|----------|------|-------------------|------|
| LSQ | 1902.08153 | 2020 | Learnable step sizes for QAT | 1 |
| PACT | 1805.06085 | 2018 | Learned activation clipping | 1 |
| DoReFa-Net | 1606.06160 | 2016 | Low-bit weights/activations/gradients training | 1 |
| Compute-Optimal QAT | 2509.22935 | 2025 | Optimal FP→QAT compute split | 1 |
| Reasoning-QAT | 2601.14888 | 2026 | PTQ init + KD = best recipe | 1,5 |
| Scaling Laws for Precision | 2411.04330 | 2024 | Lower-precision training may be compute-optimal | 1 |
| GPTQ | 2210.17323 | 2023 | Block-wise second-order PTQ | 2 |
| AWQ | 2306.00978 | 2024 | Activation-aware channel scaling | 2 |
| EfficientQAT | 2407.11062 | 2024 | Block-wise QAT for large models | 2 |
| Bit-by-Bit | 2604.07888 | 2026 | Progressive 8→4→2-bit QAT | 2 |
| UPQ | 2506.09104 | 2026 | PTQ + distillation QAT cascade | 2,5 |
| HuggingFace 1.58-bit | blog | 2024 | Lambda scheduler for ternary ramp-up | 2,4 |
| LoftQ | 2310.08659 | 2023 | Joint quant + LoRA init via SVD | 3 |
| LQ-LoRA | 2311.12023 | 2023 | Low-rank + quantized decomposition | 3 |
| QA-LoRA | 2309.14717 | 2024 | Group-wise quant-aware LoRA (merge-compatible) | 3 |
| RILQ | 2412.01129 | 2025 | Rank-insensitive LoRA error compensation | 3 |
| LoTA-QAF | 2505.18724 | 2025 | Ternary adapter weights, lossless merge | 3,4 |
| Recover-LoRA | 2606.04238 | 2026 | LoRA + KD on synthetic data for 2-bit | 3,5 |
| BitNet b1.58 2B4T | 2504.12285 | 2025 | Native ternary pretraining + SFT | 4 |
| BitNet Distillation | 2510.13998 | 2025 | FP→ternary via SubLN + attention distillation | 4,5 |
| LLM-QAT | 2305.17888 | 2023 | Data-free KD for quantized LLMs | 5 |
| BitDistiller | 2402.10631 | 2024 | CAKLD self-distillation for sub-4-bit | 5 |
| NVFP4 QAD | 2601.20088 | 2026 | Forward KL distillation for FP4 | 5 |
