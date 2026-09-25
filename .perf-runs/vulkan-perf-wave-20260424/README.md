# Vulkan Perf Wave — 2026-04-24

## Host

- GPU: **AMD Radeon(TM) 8060S Graphics** (Strix Halo iGPU, RDNA 3.5, gfx1151)
- CPU: Ryzen AI Max+ 395 (Zen 5, 32T)
- Memory: UMA DDR5
- Vulkan driver: AMD proprietary (`vulkan-1.dll`)
- Build: `-c Release`, `net10.0`, JIT
- Branch: `feature/vulkan-perf-wave` (branched off `feature/mamba-3` at `c256d19`)

## Goal

Take the CPU↔Vulkan end-to-end forward from "seconds per forward" (host-visible
everything, one `vkQueueWaitIdle` per kernel, no descriptor reuse) to
something useful — without touching kernel quality or shaders.

Ground truth: `VulkanForwardPerfHarness.MeasureDecodeLatency` on
SmolLM-135M.Q8_0, 16 decode steps after 3 warmup. Wall-clock per
`VulkanTransformerModel.Forward` call.

Reproduction:

```bash
cd .claude/worktrees/agent-a74786ce  # or wherever feature/vulkan-perf-wave is checked out
DOTLLM_VULKAN_PERF=1 DOTLLM_VULKAN_PERF_DECODE_STEPS=16 DOTLLM_VULKAN_PERF_WARMUP=3 \
  dotnet test tests/DotLLM.Tests.Integration \
  --filter "FullyQualifiedName~VulkanForwardPerfHarness" \
  -c Release --logger "console;verbosity=detailed"
```

Correctness oracle at every step:

```bash
dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~VulkanTransformerModelTests" -c Release
dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~Vulkan" -c Release
```

## Summary

| Stage                              | Commit    | decode avg (ms) | tok/s | Δ vs prev |
| ---------------------------------- | --------- | --------------: | ----: | --------: |
| Baseline (c256d19 as checked out)  | c256d19   | 49.36           | 20.26 | —         |
| Step 1: device-local weights + KV  | e8266d0   | 42.97           | 23.27 | -13%      |
| Step 2: fence-pipelined forward    | d43ff71   | 15.70           | 63.71 | -63%      |
| Step 3: descriptor-set cache       | 7566ee8   | 13.21           | 75.69 | -16%      |

**End-to-end: 49.36 → 13.21 ms/forward (3.7× speedup), 20.26 → 75.69 tok/s (3.7×).**

All three steps kept the CPU↔Vulkan parity test at 8/9 strict argmax
matches (the single step-1 swap is Q8_0-vs-F32 noise, unchanged from the
baseline). Every Vulkan unit test (53/53) stays green across all three
commits. Full unit suite stays at 1396/0/38.

## Step-by-step

### Step 1 — Device-local weights + KV cache (commit `e8266d0`)

**Before.** Every `VulkanDevice.Allocate` returned a buffer backed by
`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT`. Weights lived
in the same host-coherent linear memory a compute shader would read via
the same HOST memory bandwidth as a `memcpy` would. KV cache was
similarly host-coherent and updated via host-mapped `System.Buffer.MemoryCopy`.

**After.** Two allocator paths:

- `VulkanDevice.Allocate(long)` — still host-visible host-coherent, for
  activation / scratch buffers the forward pass reads from the host
  (embedding upload, position upload, bias-add fallback, logits
  download).
- `VulkanDevice.AllocateDeviceLocal(long)` — device-local only (prefers
  a `DEVICE_LOCAL_BIT` memory type that is *not* `HOST_VISIBLE` so the
  driver picks its tiled / swizzled native layout). Falls back to
  `DEVICE_LOCAL + HOST_VISIBLE` on older Intel / mobile GPUs that only
  expose the combined pool.

Weight upload (`VulkanWeights.Upload`) pre-allocates one reusable
host-visible staging buffer sized to the widest matrix (lm_head
49152×576 on SmolLM-135M). Each weight matrix:

1. Map staging, dequantize / copy FP32 rows into it, unmap.
2. `vkCmdCopyBuffer(staging → device-local)` on a one-shot command
   buffer, fenced.

KV cache (`VulkanKvCache`) now allocates device-local from the start.
`UpdateDevice` records `vkCmdCopyBuffer` from the current forward's
host-visible K/V scratch into the cache (one contiguous region when
positions are consecutive, per-row otherwise) and fences.

Measured decode avg: **49.36 → 42.97 ms** (13% drop).

The gain is modest on this iGPU because the bytes still sit in the same
DDR5 — there's no PCIe bus to cross. A discrete dGPU would see a much
bigger jump on this step because the weights physically move off the
system-memory bus into VRAM. The ~13% we do see on UMA is the driver
switching from host-coherent linear memory to its tiled compute layout.

### Step 2 — Fence-pipelined forward (commit `d43ff71`)

**Before.** Each of the 10 kernels' `Launch(...)` allocated its own
command buffer, submitted one dispatch, called `vkQueueWaitIdle`, and
freed the command buffer. SmolLM-135M with 30 layers dispatches ~15
kernels per layer for a total of ~450 per forward, plus many host-side
`vkMapMemory` round-trips for the per-layer residual copies. Every one
of those 450+ round-trips went host ↔ driver ↔ GPU ↔ driver ↔ host.

**After.** Refactored every kernel into two entry points:

- `Record(cmdBuf, ...)` — appends the dispatch to a caller-owned command
  buffer. No submit, no fence.
- `Launch(...)` — wraps `Record` in a one-shot submit context for
  standalone unit tests and API compatibility.

`VulkanTransformerModel` now owns one persistent `SubmitContext`
(command buffer + fence). `Forward` calls `Begin`, records the entire
transformer (with `COMPUTE→COMPUTE` memory barriers between kernels,
`TRANSFER→COMPUTE` after each KV update, `HOST→COMPUTE` after the
embedding/position uploads, `COMPUTE→HOST` before the logit download),
then `SubmitAndWait` once. The per-kernel residual copies that used to
be `vkMapMemory + Buffer.MemoryCopy` become `vkCmdCopyBuffer` in the
same command buffer.

Optional bias adds remain host-mapped (SmolLM-135M has no biases, so
the whole forward lands in one submit on the test model). When a model
*does* have biases, the record loop breaks into multiple submits around
the host step — each submit still batches many dispatches behind one
fence.

`KernelSupport` centralizes descriptor-pool creation (now
`maxSets=1024` per kernel) and the barrier shapes.

Measured decode avg: **42.97 → 15.70 ms** (63% drop, 2.7× speedup).

This is the biggest single lever. The scaffold wasn't slow because the
kernels were slow — it was slow because every dispatch round-tripped
through the host.

### Step 3 — Descriptor-set cache per kernel (commit `7566ee8`)

**Before.** Each `Record` call ran
`vkAllocateDescriptorSets` + `vkUpdateDescriptorSets` to build a fresh
descriptor set, then used it for exactly one dispatch. Across a SmolLM
forward the matmul kernel alone does this 211 times per forward,
RMSNorm 61 times, and so on.

**After.** `DescriptorSetCache` (one per kernel) keys on the tuple of
buffer handles bound to the set. First call with a new tuple allocates
and writes; every subsequent call with the same tuple re-uses the
cached handle and skips both the allocate and the update.

Cache lifetime spans forward passes: weights and activation scratch
have stable `VkBuffer` handles once the first forward has grown
`VulkanForwardState` to the longest seqLen the caller will hit. So the
cache warms through the prefill and stays hot for every decode step.
`VulkanForwardState.EnsureCapacity` now returns a bool so the model can
invalidate every kernel's cache on the rare scratch-regrow path; no
reset on the steady-state decode loop.

Capacity is 256 slots per kernel (SmolLM-135M hits ~211 distinct matmul
tuples per forward). On overflow the pool resets and the cache drops
every entry — a slow fallback that never fires in normal inference.

Measured decode avg: **15.70 → 13.21 ms** (16% drop). Steady-state min
/ max 11.90 ms / 15.00 ms.

The remaining headroom from here is in kernel quality, not dispatch
machinery — see "Not in scope" below.

## Per-step latency distribution (after Step 3)

Steady-state decode on SmolLM-135M.Q8_0:

```
warmup  [0..2]: 17.6..21.4 ms   (cache warming)
decode  [0]:    15.00 ms        (one distinct tuple vs warmup — branch on kvCache.CurrentLength crossing 8)
decode  [1..15]: 11.90..14.44 ms (cache fully warm)
```

## Limits

On this UMA box (Ryzen AI Max+ 395 iGPU), steady-state decode is
~**75 tok/s** on the F32 Vulkan path. llama.cpp Vulkan's reference is
~392 tok/s on the same hardware — but that path uses Q8_0 end-to-end
weights, not the FP32-dequantized path we have today. Closing that gap
is the Q8_0-end-to-end + cooperative-matrix work, tracked on the
existing follow-up issues and explicitly out of scope for this wave.

## Not in scope (follow-ups)

1. **Q8_0 end-to-end** — a parallel agent owns this. Once Q8_0 blocks
   stay on device and matmuls route through `matmul_q8_0` (GEMV) /
   `matmul_q8_0_gemm` (batched), the weight memory and the kernel op
   count both drop ~4×.
2. **Cooperative-matrix Q8 GEMM** — RDNA3 WMMA 16×16×16 int8 via
   `VK_KHR_cooperative_matrix`. Expected ~10× on the batched kernel.
3. **`bias_add_f32` kernel** — pulls Phi / Qwen bias adds out of the
   host-mapped bracket; removes the per-layer submit split when biases
   are present.
4. **Async scratch pool** — if a future model grows scratch mid-decode
   (long-context), the current invalidate-and-reset cache flush is a
   submit-wide stall. A small async-allocated scratch pool would keep
   the cache warm across resizes.
5. **Staging ring for prefill** — currently one reusable staging buffer
   sized to the largest matrix. A small ring would let weight uploads
   pipeline during model load; not load-path-critical today.

## Files changed

- `src/DotLLM.Vulkan/VulkanDevice.cs` — `AllocateDeviceLocal`,
  `UploadToDeviceLocal`, `CopyBufferRangeSynchronous`, `SubmitContext`.
- `src/DotLLM.Vulkan/VulkanWeights.cs` — staging-buffered device-local
  weight upload.
- `src/DotLLM.Vulkan/VulkanKvCache.cs` — device-local buffers,
  `RecordUpdate(cmdBuf, ...)` for the pipelined forward plus the
  legacy synchronous `UpdateDevice` fallback.
- `src/DotLLM.Vulkan/VulkanForwardState.cs` — `EnsureCapacity` now
  returns a `bool` to signal scratch regrow.
- `src/DotLLM.Vulkan/VulkanTransformerModel.cs` — refactored
  `Forward` to record the whole transformer into one persistent
  command buffer + fence.
- `src/DotLLM.Vulkan/Kernels/*.cs` — every kernel gained a
  `Record(cmdBuf, ...)` entry point; `Launch` kept as a legacy
  wrapper; each kernel owns a `DescriptorSetCache`.
- `src/DotLLM.Vulkan/Kernels/KernelSupport.cs` (new) — shared
  descriptor-pool / set / barrier helpers.
- `src/DotLLM.Vulkan/Kernels/DescriptorSetCache.cs` (new) — per-kernel
  buffer-handle-keyed descriptor set cache.
- `src/DotLLM.Vulkan/Interop/VulkanApi.cs` +
  `src/DotLLM.Vulkan/Interop/VulkanStructs.cs` — `vkCmdCopyBuffer`,
  `vkCmdPipelineBarrier`, fence APIs, memory-barrier / buffer-copy /
  fence structs, pipeline-stage / access flags.
- `tests/DotLLM.Tests.Integration/Vulkan/VulkanForwardPerfHarness.cs`
  (new) — `DOTLLM_VULKAN_PERF=1`-gated timing harness.

## Raw measurements

Each run: 3 warmup decodes (timed but reported separately) + 16 steady
decodes. Wall time per `VulkanTransformerModel.Forward` via
`System.Diagnostics.Stopwatch`. Host `dotnet test` default Release GC
/ JIT, `SustainedLowLatency` not forced.

### Baseline (c256d19)

```
prefill_len=5 prefill_ms=80.77
warmup_avg_ms=51.74
decode_avg_ms=49.36 min=39.87 max=107.99 tok/s=20.26
```

### After Step 1 (e8266d0)

```
prefill_len=5 prefill_ms=114.79   # includes staging copies
warmup_avg_ms=46.61
decode_avg_ms=42.97 min=38.27 max=49.18 tok/s=23.27
```

### After Step 2 (d43ff71)

```
prefill_len=5 prefill_ms=54.70
warmup_avg_ms=22.79
decode_avg_ms=15.70 min=14.55 max=18.07 tok/s=63.71
```

### After Step 3 (7566ee8)

```
prefill_len=5 prefill_ms=45.62
warmup_avg_ms=18.31
decode_avg_ms=13.21 min=11.90 max=15.00 tok/s=75.69
```
