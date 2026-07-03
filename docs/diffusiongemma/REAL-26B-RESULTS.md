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

_(bench battery: full-canvas 256 run, PKV A/B, same-session llama.cpp Vulkan diffusion
baseline, and the LLaDA-8B (plain Llama backbone) cross-engine AR decode ratio — results
pending)_
