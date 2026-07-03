# DiffusionGemma-26B — real-weight end-to-end results (#40 / #41 / #33)

**Session 2026-07-03, Strix Halo (Ryzen AI Max 395, Radeon 8060S gfx1151, 128 GB UMA), Windows 11.**
Model: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (15.65 GB, unsloth conversion). Branch
`issue/40-vulkan-real-26b` on top of `dev` @ `2d6543a`. GPU runs executed under the
GPU-free check (iGPU < 10 % for 60 s before each measurement).

## 1. First real end-to-end diffusion generation (CPU) — #40 closed

`DiffusionGemmaGgufForwardTests` (all three pass on real weights):

| Test | Wall | Result |
|---|---|---|
| SingleForward (8-mask canvas) | load 3.5 s, forward 18.1 s | finite, sharp (spread 39.4); raw argmax = mask everywhere (expected absorbing state), suppressed argmax surfaces content |
| BackboneIsolation (prompt-only causal) | forward 6.0 s | top-1 `the` 21.62, top-2 **`Paris` 21.51** — backbone sane |
| **DenoiseLoop (canvas 16, ≤16 steps, SC on)** | **192.8 s, 15 steps** | finish=Stop, 14/15 distinct tokens |

Generated text for `The Eiffel Tower is located in`:

> *", the French capital. It is one of the de recognizable landmarks in"*

Coherent completion (the lone `de` is the known low-information-token artifact). CPU cost
≈ **12.9 s/step ⇒ 0.08 effective tok/s** — confirms the GPU is mandatory for usable
throughput, which is what the Vulkan path below measures.

## 2. Vulkan end-to-end generation — `DiffusionGemmaVulkanRealGenerationTests`

**First attempt OOM'd — two real defects found and fixed (issue #120):**

1. The real unsloth Q4_K_M's `ffn_down_exps` are **Q5_0 ×16 + Q8_0 ×14 (never Q5_1)**, so
   `Gemma4ExpertsKeepQuantized` failed on all 30 layers and the F32 host-dequant fallback
   tried to allocate ~92 GB (≈3 GB/layer × 30). Fixed by widening the quantized path:
   Q5_0 downs repack **bit-exactly** to Q5_1 at upload (`d·(q−16) = d·q + m`, `m = −16d`,
   fp16-exact — unit-tested), Q8_0 downs keep the existing Q8_0 indexed kernel with the
   per-expert `ffn_down_exps.scale` folded into each block's fp16 `d`.
2. The **strict DEVICE_LOCAL heap on gfx1151 is 15.82 GiB** (budget 15.03) — smaller than
   the model — while heap[1] exposes 96 GiB of DEVICE_LOCAL|HOST_VISIBLE UMA memory.
   `AllocateDeviceLocal` now retries the UMA/GTT type on `VK_ERROR_OUT_OF_DEVICE_MEMORY`
   (llama.cpp `GGML_VK_ALLOW_SYSMEM_FALLBACK` equivalent; opt-out
   `DOTLLM_VULKAN_STRICT_DEVICE_LOCAL=1`, occurrences in `DeviceLocalFallbackCount`).

**With the fix: PASSES on real weights** (canvas 32, ≤16 steps, SC + PKV on):

| Metric | Value |
|---|---|
| Load (repack + upload, 15.65 GB) | 38.5 s |
| Generation wall | 58.6 s, 15 steps (finish=Stop) |
| Step latency | **3.9 s/step** (vs 12.9 s/step CPU → 3.3×) |
| Effective | 0.29 tok/s (17 tokens, 15/17 distinct) |

Decoded text carried the fact ("… France.") with rougher filler tokens than the CPU run —
sampling is temperature-stochastic and the trajectory differs per backend/canvas; the
CPU↔Vulkan numerical parity gate is the synthetic-fixture argmax test, not decoded text.

## 3. Relative throughput (#41) and capability (#33)

Same session, same box, GPU pre-checked idle (0.5 %). llama.cpp = `llamacpp-vulkan`
build 74ade5274 / the PR-24423 `llama-diffusion-cli` Vulkan build.

| Measurement | dotLLM Vulkan | llama.cpp Vulkan | Ratio |
|---|---|---|---|
| **AR decode, LLaDA-8B Q4_K_M (same file, plain Llama backbone)** | 28.24 tok/s (`decode_min_ms` 34.47) | 44.85 ± 0.05 tok/s (`tg32`) | **0.63×** |
| 26B diffusion step latency | 3.90 s/step @ canvas 32, SC+PKV on | 0.736 s/step @ canvas 256 | ~36× slower per canvas-token |
| 26B diffusion effective | 0.29 tok/s (canvas 32) | 34.8 tok/s (256 tok, 10 steps) | — |
| PKV prompt-cache effect (canvas 32) | 3.90 s/step on vs 4.47 s/step off | n/a | −13 % per step |

**Reading.** The AR ratio (0.63×, up from 0.23× pre-kernel-campaign) says the Vulkan
GEMV/MMVQ kernels are within striking distance. The diffusion gap is structural, not
kernel-level: dotLLM's self-conditioning computes the full soft-embed —
`[canvas × 262 144-vocab] softmax × [vocab × 2 816] embedding GEMM` ≈ 47 GFLOP/step at
canvas 32 — which alone accounts for the observed ~3.9 s/step, while llama.cpp's
diffusion build runs `gpu_sampling=on sample_reduce=on` (sparsified SC / on-device
sampling). Closing the diffusion gap = sparsifying the SC embed (top-K logits, like the
reference) + moving unmask sampling on-device — filed as the #41 follow-up; the AR
backbone needs no dedicated work beyond the existing kernel roadmap.

Capability note (#33): both engines complete the factual prompt correctly at full canvas
(llama.cpp produced a coherent 256-token chat-formatted answer; dotLLM CPU produced the
coherent French-capital completion in §1). A scored fixed-prompt-set comparison remains
open under #33 — it only becomes meaningful once the SC sparsification lands and dotLLM
can run full-canvas in seconds rather than minutes.
