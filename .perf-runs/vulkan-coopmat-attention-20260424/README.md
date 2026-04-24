# Vulkan attention_f32_coopmat — AMD Radeon 8060S iGPU (gfx1151, RDNA3.5)

## Status

Landed — shader compiles, probe works, runtime dispatch picks coopmat when
available, numerical parity validated to abs 5e-4 / rel 5e-3 against the
scalar CPU reference on five representative shapes (SmolLM decode, SmolLM
prefill, Llama hd=128 decode, multi-tile online-softmax, single-head decode).

**Perf: coopmat path is SLOWER than shared-mem / subgroup on this iGPU.**

The target was ≥2× at the prefill shape; reality is a ~1.3× regression.
Root cause appears to be RDNA3.5 WMMA + wave=64 shared-memory pressure plus
the Q/K f32→f16 conversion overhead, not outweighed by the 16×16×16 tile
throughput on the shape sizes tested. See perf table below.

## Measured

Host: Ryzen AI Max+ 395 iGPU (AMD Radeon 8060S, gfx1151). 50 iters per
shape, warmed up 3×. Timer: `Stopwatch` around `AttentionF32Kernel.Launch`
(includes `vkQueueSubmit` + `vkQueueWaitIdle`). Env-var toggling selects
the variant.

| Shape                                                | shared-mem | subgroup | coopmat | coopmat vs shared |
|------------------------------------------------------|-----------:|---------:|--------:|------------------:|
| SmolLM decode     (sq=1,  sk=128, nh=9,  nkv=3, hd=64)  |   0.166 ms |  0.164 ms | 0.251 ms |           1.51× slower |
| SmolLM prefill    (sq=64, sk=64,  nh=9,  nkv=3, hd=64)  |   0.222 ms |  0.216 ms | 0.282 ms |           1.27× slower |
| Llama decode      (sq=1,  sk=128, nh=32, nkv=8, hd=128) |   0.275 ms |  0.264 ms | 0.429 ms |           1.56× slower |

## Why is coopmat slower?

The 16×16×16 tile shape RDNA3 exposes via `VK_KHR_cooperative_matrix` runs
wave32 WMMA instructions even though the driver reports subgroupSize=64.
That halves the usable lanes per coopmat multiply. Plus the shader does:

- Explicit f32 → f16 Q/K/V conversion into shared memory (load, cast,
  store, barrier). The shared-mem shader reads f32 straight through.
- Zero-pad of unused `[headDim..MAX_HEAD_DIM]` tail (to match the coopmat
  stride). Wasted work when `headDim` is 64 or 128.
- Extra barriers for the per-row softmax update + correction feedback.

For prefill (sq=64) the coopmat path processes 4 query tiles × 9 heads = 36
workgroups, each doing 16 Q rows × N KV tiles. The shared-mem path does
64 × 9 = 576 workgroups, each doing 1 Q row. RDNA3.5's 8 WGPs prefer more
smaller workgroups for this shape; coopmat's fewer-but-wider dispatch
under-fills the GPU.

## Decisions

- **Dispatch priority kept as `coopmat > subgroup > shared`** per the task.
  NVIDIA tensor cores + AMD discrete (gfx110x) will make this the right
  choice; leaving the dispatch default gives those paths the speedup and
  lets AMD iGPU users escape via `DOTLLM_VULKAN_FORCE_NO_COOPMAT=1`.
- Tolerance relaxed to **abs 5e-4 / rel 5e-3** on the coopmat-path arm of
  the numeric-parity tests. The f16 input quantization adds ~3× abs noise
  over the shared-mem baseline's 1e-4 / 1e-3; shared and subgroup stay on
  the strict envelope.
- `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` sType is `1000506000`
  (NOT `1000506001` — `1000506001` is the property struct). Had this
  swapped in the first commit and the feature probe reported
  `cooperativeMatrix = 0` on a device that clearly exposed the tile.

## Next steps (out of scope for this commit)

- Re-bench on an NVIDIA GPU (tensor cores at 16×16×16 tf32/f16) and on
  AMD discrete (gfx110x actual wave32 WMMA). Expect coopmat to win cleanly
  there.
- Larger prefill shapes — as seq_q grows past ~256, the shared-mem path's
  per-query-row workgroup overhead starts dominating and coopmat's
  16-row-per-WG dispatch becomes more favorable.
- Use `VK_EXT_subgroup_size_control` to force wave32 on RDNA3 — that's
  what llama.cpp does to match the WMMA instruction width and halves the
  LDS footprint. Changing WG_SIZE from 64 to 32 plus explicit wave32 would
  likely close the gap on AMD iGPU.
