# DiffusionGemma support in dotLLM — Epic Overview

> Research + planning lead deliverable. Branch: `dev-diffusiongemma` (base HEAD `467ea38`,
> "dev: merge feature/qwen3.6 (216 commits — full WIP integration)").
> Research only — **no GPU operations were run** to produce this document. Any GPU-heavy
> validation or benchmark step in the plans below is gated on an explicit
> "is-the-GPU-free?" check (see [VALIDATION.md](VALIDATION.md)).

---

## 1. What DiffusionGemma is (verified specs + sources)

DiffusionGemma is a **real, released** Google open-weights model: an **experimental text-diffusion
Gemma** built on the **Gemma 4 26B-A4B Mixture-of-Experts** backbone. It generates a whole
256-token block ("canvas") in parallel by iterative **discrete masked diffusion** denoising,
rather than one autoregressive token at a time.

### Sources
- DeepMind model page: <https://deepmind.google/models/gemma/diffusiongemma/>
- Google AI overview: <https://ai.google.dev/gemma/docs/diffusiongemma>
- Google AI HF-inference guide: <https://ai.google.dev/gemma/docs/diffusiongemma/inference-diffusiongemma-with-hf>
- HF model: <https://huggingface.co/google/diffusiongemma-26B-A4B-it> (Apache-2.0, ~25.8B params, `model_type=diffusion_gemma`, `architectures=["DiffusionGemmaForBlockDiffusion"]`)
- GGUF community port: <https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF>
- NVFP4 port: <https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4>

### Verified `config.json` (text tower) — `google/diffusiongemma-26B-A4B-it`
Top level: `model_type=diffusion_gemma`, `architectures=["DiffusionGemmaForBlockDiffusion"]`,
`canvas_length=256`, `dtype=bfloat16`, `tie_word_embeddings=true`, multimodal (`vision_config` present,
`use_bidirectional_attention="vision"`).

`text_config` (`model_type=diffusion_gemma_text`):

| Field | Value | Notes |
|---|---|---|
| `hidden_size` | 2816 | |
| `num_hidden_layers` | 30 | |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 8 | GQA |
| `head_dim` | 256 | sliding layers; `global_head_dim` 512 on full-attn layers |
| `num_global_key_value_heads` | 2 | KV heads for the full-attention layers |
| `intermediate_size` | 2112 | dense FFN width |
| `moe_intermediate_size` | 704 | per-expert FFN width |
| `num_experts` | 128 | |
| `top_k_experts` | 8 | active experts/token |
| `vocab_size` | 262144 | Gemma SentencePiece |
| `hidden_activation` | `gelu_pytorch_tanh` | GeGLU |
| `rms_norm_eps` | 1e-6 | |
| `final_logit_softcapping` | 30.0 | (no `attn_logit_softcapping`) |
| `sliding_window` | 1024 | |
| `layer_types` | 30-entry list | full-attention at layers 5,11,17,23,29 (every 6th, 1-indexed); rest sliding |
| `max_position_embeddings` | 262144 | |
| `rope_parameters` | per-attn-type | full: theta 1e6, `partial_rotary_factor=0.25`, `rope_type=proportional`; sliding: theta 1e4, default |

### Verified `generation_config.json` (diffusion sampler)
- `canvas_length` / `max_new_tokens` = 256
- `max_denoising_steps` = 48 (adaptive early-stop typically lands at 12–16)
- Sampler = `EntropyBoundSamplerConfig`, `entropy_bound` = 0.1 (lowest-entropy unmask selection)
- Adaptive stop: `confidence_threshold` / entropy threshold = 0.005; `stability_threshold` = 1
- Temperature schedule: linear `t_max=0.8 → t_min=0.4`
- `pad_token_id`=0, `eos_token_id`=[1,106,50], `bos_token_id`=2

### Mechanism (from the docs, terminology verified)
- **Discrete masked (absorbing-state) diffusion.** A 256-token canvas starts fully masked; each
  denoise step the model predicts the canvas bidirectionally and **unmasks** the lowest-entropy
  tokens (those whose mutual-information bound stays under `entropy_bound`). Remaining tokens stay
  masked for the next step. Stops early when average canvas entropy < 0.005.
- **Block-autoregressive / multi-canvas.** Prompt context is processed by an **autoregressive
  encoder/prefill that fills a KV cache**; the decoder applies **bidirectional attention over the
  canvas** and reads cached prompt context. Multiple canvases are produced block-autoregressively
  for sequences longer than 256.
- **Mask token id**: not published in `config.json`/`generation_config.json`. Must be read from
  `tokenizer_config.json` / `special_tokens_map.json` / the reference modelling code at load time
  (see issue 07). Do **not** hardcode.

> Note: the public page also says the *vision* tower (not the text tower) is the part flagged
> `use_bidirectional_attention="vision"`. The **text** tower's bidirectionality comes from the
> **diffusion decode** path operating over the canvas, not from a config flag — important for the
> mask-path design (issue 06).

---

## 2. NEW vs REUSABLE against dev's CURRENT state (HEAD `467ea38`)

Re-verified directly on this branch (not the older fork state in the brief). Key correction to the
prior gap analysis: **a partial Gemma 3 implementation already exists on dev** — enum, config
fields, attention-side mechanisms, and a synthetic forward test. But it is *mechanism-level only*:
the FFN still runs SwiGLU, the loader is the 2-norm Llama layout, and there is **no embedding
scaling, no 4-norm layout, no per-head QK-norm load, no `(1+w)` RMSNorm absorption, and no GeGLU
wiring into the forward pass**. DiffusionGemma additionally needs the **Gemma 4 MoE** shape and the
entire **diffusion decode seam**, neither of which exists.

### Already present / reusable
- **`Architecture.Gemma3` enum** with full doc of Gemma 2/3 features — `src/DotLLM.Core/Configuration/Architecture.cs:147-194`.
- **Gemma config fields on `ModelConfig`**: `PerLayerSlidingWindow`, `AttnLogitSoftcap`,
  `FinalLogitSoftcap`, `QueryPreAttnScalar` — `src/DotLLM.Core/Models/ModelConfig.cs:71-97`.
- **HF config extraction for Gemma 3**: per-layer `layer_types`, softcaps, QPAS, activation
  mapping, `text_config` hoist, tie-default — `src/DotLLM.Models/SafeTensors/HfConfigExtractor.cs:58-168, 637-669`.
- **Attention mechanisms wired in CPU forward**: per-layer sliding-window dispatch, `QueryPreAttnScalar`
  scale, `AttnLogitSoftcap`, `FinalLogitSoftcap` — `src/DotLLM.Models/Architectures/TransformerModel.cs:354,734-764,1052-1073,1428-1475`.
- **GeGLU (tanh) fused kernel** (SIMD + scalar reference) — `src/DotLLM.Cpu/Kernels/FusedOps.cs:95-211`.
  Already validated by `tests/DotLLM.Tests.Unit/Cpu/Kernels/FusedOpsTests.cs`. **Not yet called by the model.**
- **`ActivationFunction` enum** has `SiLU, GELU, GELUTanh, ReluSquared` — `src/DotLLM.Core/Configuration/ActivationFunction.cs`.
- **Per-head Q/K RMSNorm path** exists (`QNormWeight`/`KNormWeight`, `ApplyPerHeadNorm`) —
  `src/DotLLM.Models/Architectures/TransformerModel.cs:734-737,1428-1431`; weight slot at
  `TransformerWeights.cs:324`. Loader does not yet populate it for Gemma.
- **MoE forward** (routed top-k + shared experts, SwiGLU experts, router gate) —
  `src/DotLLM.Models/Architectures/TransformerModel.cs:832-863`; config `MoeConfig` at `src/DotLLM.Core/Models/MoeConfig.cs`.
- **`IModel.Forward` returns all-position logits** `[seqLen, vocab]` (not just last token) —
  `src/DotLLM.Core/Models/IModel.cs:25,53`; confirmed by the Gemma3 forward test asserting `[seqLen, vocab]`.
- **Architecture extension template** (config extractor + loader dispatch + dedicated model class):
  Mamba3 — `ModelLoader.cs:77-89,147-151`, `Mamba3ConfigExtractor.cs`, `Mamba3TransformerModel.cs`.
- **Synthetic Gemma 3 forward test** harness pattern — `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelGemma3ForwardTests.cs`.

### Net-new (does not exist on dev)
- **Numerically-correct Gemma backbone**: embedding scale (`× sqrt(hidden_size)`), GeGLU wired into
  FFN, Gemma 4-norm layout (`input` / `post_attention` / `pre_feedforward` / `post_feedforward`),
  `(1+w)` RMSNorm absorption, per-head QK-norm loaded. (The existing Gemma3 path is mechanism-only;
  the synthetic test explicitly notes the FFN uses SwiGLU and the loader uses the 2-norm layout —
  `TransformerModelGemma3ForwardTests.cs:297-301,316-321,339`.)
- **Gemma 4 MoE shape**: `num_global_key_value_heads` (different KV head count on full-attn layers),
  partial-rotary RoPE per attention type, `gelu_pytorch_tanh` MoE experts (current MoE experts are
  SwiGLU-only). No `Architecture.Gemma4` / `Gemma4Moe` enum exists.
- **The whole non-AR/diffusion decode seam**: bidirectional/hybrid attention mask path; mask-token
  plumbing; iterative denoise loop (`DiffusionTextGenerator`); remask/unmask scheduler; parallel
  confidence/entropy-based unmasking sampler. There is **zero** `bidirectional` code anywhere
  (grep: no matches in `src/`). Causal masking is hardcoded — `ApplyCausalMask` at
  `src/DotLLM.Cpu/Kernels/Attention.cs:160,381,727`.
- **SafeTensors dispatch** for `model_type=diffusion_gemma` / `diffusion_gemma_text` — `ModelLoader.cs:77-89` throws `NotSupportedException` for anything outside the current arch list.
- **Validation + throughput harness** for a small diffusion LM.

See [proposed-issues/](proposed-issues/) for the full decomposition.

---

## 3. Small-model validation strategy

Validating against the full 26B MoE on the dev box is impractical and contends for the GPU.
We validate the **operations** against the smallest masked-diffusion LM that exercises the same
core ops (bidirectional attention + iterative masked denoise), then scale up.

- **Primary**: [`diffusionfamily/diffugpt-s`](https://huggingface.co/diffusionfamily/diffugpt-s)
  — 124M, GPT-2 backbone adapted to **absorbing-state masked diffusion** (DiffuLLaMA / arXiv 2410.17891),
  safetensors (F16), Apache-2.0. Exercises bidirectional attention + masked iterative denoise +
  confidence unmasking. Does **not** exercise: Gemma norms/GeGLU/softcap/QPAS, MoE, partial-rotary RoPE.
- **Fallback / architecturally-closest**: [`inclusionAI/LLaDA2.1-mini`](https://huggingface.co/inclusionAI/LLaDA2.1-mini)
  — `model_type=llada2_moe`, **MoE + masked diffusion** (closest single model to DiffusionGemma's
  MoE-diffusion combo), safetensors, Apache-2.0. Heavier; use once the MoE diffusion path is in.

Full rationale, ops-coverage matrix, and the AR-baseline comparison are in [VALIDATION.md](VALIDATION.md).

---

## 4. Capability + throughput comparison approach

Once the decode loop works (M3), every milestone is validated by comparing the dotLLM diffusion
path against the HF reference of the **same small model** and against an **AR baseline of comparable
size**:

- **Numerical correctness**: per-step canvas-logit parity vs the HF reference forward
  (cosine sim / max-abs-diff on a fixed prompt+mask pattern), on **CPU** where possible to avoid the GPU.
- **Capability**: coherence/quality on a fixed prompt set (code-gen, OCR-correction, short QA — the
  DiffusionGemma demo tasks), scored against the HF reference output.
- **Throughput**: tokens/sec, **denoise-steps/sec**, canvas-latency, and effective tokens/sec vs an
  AR model of similar size (the diffusion selling point: parallel-canvas speedup at fixed quality).

> **GPU guard (repeat):** the dev box is the Strix Halo target and shares its iGPU with concurrent
> work. **Every GPU-heavy validation/benchmark run MUST first confirm the GPU is idle** (see the
> concrete check in [VALIDATION.md §GPU-free guard](VALIDATION.md)). Prefer CPU for correctness; gate
> only the throughput numbers on GPU availability.

---

## 5. Plan index

- **Proposed issues** (12): [proposed-issues/](proposed-issues/) — `01`..`12`.
- **PR plan & milestones**: [PR-PLAN.md](PR-PLAN.md) — 6 milestones (M1 Gemma-AR backbone → M6 generalize).
- **Validation methodology**: [VALIDATION.md](VALIDATION.md).

### Biggest risk
The **encoder-decoder / hybrid attention seam**: dotLLM's `IModel.Forward` + KV-cache + scheduler are
built end-to-end around **causal** attention (hardcoded `ApplyCausalMask`). DiffusionGemma needs an
AR-prefill (causal, KV-cached) **and** a bidirectional canvas decode that cross-attends to that cache,
within one model. Threading a non-causal mask through the attention kernels, KV-cache, and a new
`DiffusionTextGenerator` without regressing the AR hot path is the highest-effort, highest-uncertainty
piece — and it is the gate for all of M3–M6.
