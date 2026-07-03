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

_(pending — run in flight)_

## 3. Relative throughput (#41) and capability (#33)

_(pending)_
