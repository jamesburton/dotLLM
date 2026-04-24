---
date: 2026-04-24
target-tag: v0.2.0-alpha.2 (or v0.2.0-beta if we beat llama.cpp)
baseline: v0.2.0-alpha at commit 5d04691
baseline-perf: 4.94 ms/forward = 202 tok/s decode on SmolLM-135M.Q8_0 (AMD Radeon 8060S, RDNA3.5)
target: ≥349 tok/s decode (match llama.cpp Vulkan d0a6dfeb2 on same hardware)
gap: 1.73×
---

## Current state of the 4.94 ms forward

Dispatch count per decode step on SmolLM-135M (30 layers):

| Category | Per layer | × layers | + outer | Total |
|---|---:|---:|---:|---:|
| matmul_q8_0 (GEMV) | 7 | 210 | 1 (lm_head) | 211 |
| rmsnorm | 2 | 60 | 1 (output) | 61 |
| rope | 1 | 30 | 0 | 30 |
| kv-cache-update | 1 | 30 | 0 | 30 |
| attention | 1 | 30 | 0 | 30 |
| add (residual) | 2 | 60 | 0 | 60 |
| copy-buffer | 4 | 120 | 2 | 122 |
| swiglu | 1 | 30 | 0 | 30 |
| **Total** | **19** | **570** | **4** | **~574** |

~9 µs per dispatch on average. Only 40% of that is matmul compute; the rest is attention work (1.5-1.8 ms total), non-matmul overhead, and copies.

## Reference: what llama.cpp's Vulkan backend does differently

From `C:/Development/llama.cpp/llama.cpp-repo/ggml/src/ggml-vulkan/vulkan-shaders/`:

- **Coopmat matmul:** `mul_mm_cm2.comp` (KHR_coopmat2 / NV) and coopmat-1 path via `mul_mat_vecq.comp`. 16×16×16 int8/fp16 tiles via `VK_KHR_cooperative_matrix`. AMD RDNA3.5 iGPU supports this — `llama-bench --verbose` reports `matrix cores: KHR_coopmat`.
- **Coopmat flash attention:** `flash_attn_cm1.comp` / `flash_attn_cm2.comp` + `flash_attn_split_k_reduce.comp`. Tile-by-tile QK^T, online softmax, tile-by-tile AV — all in coopmat tiles.
- **Split-K reduction:** both GEMM and attention have split-K variants with a `*_split_k_reduce.comp` second pass. More parallelism for small-output-dim decode shapes.

Their decode on this hardware: 349.4 tok/s. These are the two unlocks we're missing.

## Plan (ROI-ordered, parallelism-aware)

### Round 1 (parallel, independent files)

**A1. Coopmat Q8_0 GEMM** — issue [#2](https://github.com/jamesburton/dotLLM/issues/2).
- New shader `matmul_q8_0_gemm_coopmat.comp` + new `MatMulQ8_0GemmCoopmatKernel.cs` + new test file.
- Runtime probe `VK_KHR_cooperative_matrix` (pattern already in `VulkanDevice.cs` subgroup probe).
- Dispatch coopmat when available, scalar fallback otherwise.
- **Expected: 1.0 ms saving** (≈202 → 255 tok/s).
- **Risk:** medium. Coopmat tile shape + Q8_0 dequant-to-tile is new territory but llama.cpp reference exists.
- **Does NOT modify:** existing `matmul_q8_0_gemm.comp` (scalar), existing `matmul_q8_0.comp` (GEMV).

**A2. Coopmat Flash Attention** — new issue (to be filed).
- New shader `flash_attention_f32_coopmat.comp` + new `FlashAttentionCoopmatKernel.cs` + tests.
- Also file-independent: does NOT modify existing `attention_f32.comp` or `attention_f32_sg.comp`.
- **Expected: 1.2 ms saving** (≈255 → 340 tok/s).
- **Risk:** high. Online softmax + coopmat + numerical stability.

**B3. Upload embedding table once** — issue [#12](https://github.com/jamesburton/dotLLM/issues/12).
- Touches `VulkanWeights.cs` + small bit of `VulkanTransformerModel.Forward`.
- **Expected: 0.05 ms saving** (noise-level on decode; more meaningful on prefill).
- **Risk:** low. Pattern identical to the perf-wave's device-local weight migration.
- Runs parallel with A1/A2 — the files touched don't overlap enough to block.

### Round 2 (after Round 1 lands)

**B1. Eliminate copy dispatches via buffer swap** — new issue (to be filed).
- `VulkanTransformerModel.Forward` currently does 4 `RecordCopyBuffer` per layer (HiddenState↔Residual↔AddScratch). Replace with pointer/handle swap — the buffers are identical shape so a ping-pong works.
- **Expected: 0.3 ms saving** (~120 copy dispatches eliminated).
- **Risk:** low. Pure refactor of forward loop.

**B2. Fuse rmsnorm + matmul** — new issue (to be filed).
- New shader `rmsnorm_matmul_q8_0.comp` doing rmsnorm → in-register matmul without round-tripping through `NormOutput` buffer. Attention and FFN in-rmsnorms both feed a Q8_0 matmul immediately after.
- **Expected: 0.2 ms saving** (4 fused dispatches per layer × 30).
- **Risk:** medium. New shader, reorderable numerics.

### Round 3 (validation + tag)

- Re-run `VulkanForwardPerfHarness` (32 decode steps, 4 warmup).
- Re-run llama.cpp `llama-bench` for fresh reference.
- Update `.continue-here.md` with final numbers.
- Tag `v0.2.0-alpha.2` (or `v0.2.0-beta` if we match/beat llama.cpp).

## Expected final

| | Decode (tok/s) | ms/forward |
|---|---:|---:|
| Baseline (v0.2.0-alpha) | 202 | 4.94 |
| + A1 (coopmat GEMM) | 255 | 3.9 |
| + A2 (coopmat attention) | 340 | 2.9 |
| + B1 (buffer swap) | 385 | 2.6 |
| + B2 (fused norm+matmul) | 430 | 2.3 |
| + B3 (embedding) | 435 | 2.3 |

Target: ≥349 (llama.cpp parity). Stretch: >435 (decisively beat llama.cpp).
