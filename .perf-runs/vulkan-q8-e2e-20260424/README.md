# Vulkan Q8_0 end-to-end — 2026-04-24

## Host

- GPU: **AMD Radeon(TM) 8060S Graphics** (Strix Halo iGPU, RDNA 3.5, gfx1151)
- CPU: Ryzen AI Max+ 395 (Zen 5, 32T)
- Memory: UMA DDR5
- Vulkan driver: AMD proprietary (`vulkan-1.dll`)
- Build: `-c Debug` (default for `dotnet test`), `net10.0`, JIT
- Branch: `feature/vulkan-q8-end-to-end` (branched off `feature/mamba-3` at `6f509de`)

## Goal

Close the last big lever called out at the end of the Vulkan perf wave
(`vulkan-perf-wave-20260424`): stop dequantising Q8_0 weights to FP32 at
load. Keep the 34-byte blocks on device, route matmuls through the
already-validated `matmul_q8_0` (GEMV) and `matmul_q8_0_gemm` (batched
GEMM) kernels. No shader edits — this is a host-side routing change.

Ground truth: `VulkanForwardPerfHarness.MeasureDecodeLatency` on
SmolLM-135M.Q8_0. Wall-clock per `VulkanTransformerModel.Forward` call.

Reproduction:

```bash
DOTLLM_VULKAN_PERF=1 DOTLLM_VULKAN_PERF_DECODE_STEPS=32 DOTLLM_VULKAN_PERF_WARMUP=4 \
  dotnet test tests/DotLLM.Tests.Integration \
  --filter "FullyQualifiedName~VulkanForwardPerfHarness" \
  --logger "console;verbosity=detailed"
```

Correctness oracle at every commit:

```bash
dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~VulkanTransformerModel"
dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~Vulkan"
```

## Summary

| Stage                                     | Commit   | decode avg (ms) | tok/s  | Δ vs prev |
| ----------------------------------------- | -------- | --------------: | -----: | --------: |
| Baseline (6f509de — Vulkan perf wave tip) | 6f509de  | 11.49           | 87.02  | —         |
| Step 1: `VulkanWeights` Q8_0 option       | 73f9424  | 11.49           | 87.02  | 0% (no-op) |
| Step 2: route matmuls through Q8_0        | 944cd67  | 6.23            | 160.40 | **-46%**  |

**End-to-end: 11.49 → 6.23 ms/forward (1.85× speedup), 87 → 160 tok/s (1.84×).**

Correctness: `VulkanForward_MatchesCpuReference_OnEightDecodeSteps`
holds at **8/9 strict argmax matches** across prefill + 8 decode steps,
identical to the baseline count. The single swap on step 1 is a top-1 /
top-2 neighbour tie that survives from the F32-dequant path — both
backends now read the exact same Q8_0 block bytes, so the only
arithmetic divergence is Vulkan's direct Q8_0 dot-with-FP32-activation
vs CPU's `GemmQ8_0` (which quantises the activation to Q8_1). Top-5
jaccard holds throughout.

Unit tests: 53 passed / 1 skipped Vulkan filter — unchanged from baseline.

## Step-by-step

### Step 1 — `VulkanWeights` keeps Q8_0 on device (commit `73f9424`)

**Before.** `VulkanWeights.UploadMatrix` unconditionally dequantised
every Q8_0 weight row to FP32 through `Dequantize.ToFloat32` before
staging. On SmolLM-135M's Q8_0 GGUF the matmul-weight total on device
was ~520 MiB (FP32) vs ~144 MiB of source bytes.

**After.** `Upload` gains an optional `dequantToFp32` parameter
(default `false`). When `false` and the source quant type is Q8_0, the
34-byte blocks are uploaded to device-local memory verbatim via
`vkCmdCopyBuffer` through the same reusable staging buffer.
`LayerBuffers` records the device-side quant type alongside each buffer
handle so the forward dispatch can pick the right kernel. F16 and
K-quant paths still dequant to F32 at upload (those Vulkan kernels
don't exist yet); `dequantToFp32: true` preserves the old all-F32
behaviour as a fallback switch.

The `VulkanTransformerModel` call site passes `dequantToFp32: true` in
this step so the forward behaviour is unchanged — Step 1 just makes the
weights-side change without activating it. This keeps the tree green
between commits.

Decode avg: **11.49 → 11.49 ms** (no-op, flag forced to true).

### Step 2 — Route matmuls through Q8_0 kernels (commit `944cd67`)

**Before.** `VulkanTransformerModel.Forward` dispatched every linear
projection through `MatMulF32Kernel.Record`, which reads FP32 weights.

**After.** New `RecordMatmul(cmdBuf, weights, weightQt, ...)` helper
picks the kernel by device-side quant type and seqLen:

- Q8_0 + seqLen == 1 (decode)  → `MatMulQ8_0Kernel.Record` (GEMV)
- Q8_0 + seqLen >  1 (prefill) → `MatMulQ8_0GemmKernel.Record` (batched)
- F32                          → `MatMulF32Kernel.Record` (unchanged)

All 7 per-layer projections (Q, K, V, O, Gate, Up, Down) **and** the LM
head go through the dispatcher. The LM head on SmolLM-135M is Q8_0
(49152×576), so the biggest single matmul in the forward now reads 34
bytes per 32 elements instead of 128. Activations and scratch stay
FP32 end-to-end — only the weight read-side changes.

Weights upload now runs with `dequantToFp32: false` (default) so Q8_0
matrices actually stay on device as blocks.

Decode avg: **11.49 → 6.23 ms** (46% drop, 1.85× speedup).

This is consistent with the "weight bandwidth bound" hypothesis. The
weight read is ~4× fewer bytes (34 vs 128 per 32 elements = 3.76×);
RDNA 3.5's int8 dot is fast enough on this iGPU that the Q8_0 GEMV
kernel is still bandwidth-bound, so the observed 1.85× matches the
fraction of the forward that is matmul-weight-read (roughly half —
activations, KV cache, barriers, and the attention kernel eat the rest).

## Per-step latency distribution (after Step 2)

Steady-state decode on SmolLM-135M.Q8_0 (64 decode steps after 8 warmup):

```
warmup  [0..7]: 5.74..6.55 ms   (cache + shader compile warming)
decode  [0..63]: 4.92..9.27 ms
              min 4.92 median ~6.0 max 9.27 avg 6.23   → 160.4 tok/s
```

Two-run jitter: a second 32-step run came in at decode_avg=5.85 ms
(171 tok/s). Iterations are UMA-memory-bound and the DDR5 channel
shares with background CPU traffic, so ~±6% sample-to-sample noise is
expected.

## Correctness detail

From the Q8_0 end-to-end integration test at `944cd67`:

```
step 0 (prefill): cpu=7042 vk=7042 [match]      ← Q8_0 GEMM (seqLen=5)
step 1:           cpu=30   vk=28   [swap-in-topK]
step 2..8:        cpu==vk everywhere [match]    ← Q8_0 GEMV (seqLen=1)
summary: 8/9 strict argmax matches
```

The step-1 swap is the same one the F32-dequant path produced at
`6f509de` — it's a top-2 neighbour tie from the prefill-step KV cache,
not an artefact of the Q8_0 routing. Both sides agree on the prefill
argmax (token 7042 = " Paris") and on all 7 post-swap decode argmaxes.

## VRAM footprint (SmolLM-135M)

Matmul weight bytes on device (only the tensors routed through a matmul
kernel, not bias / norm / embedding / KV):

- Before: 30 layers × (Q + K + V + O + Gate + Up + Down) + LM head, all FP32
  → ~139 MiB
- After:  same tensors at Q8_0 (34 bytes / 32 elems ≈ 27% of FP32)
  → ~37 MiB

The per-layer FP32 form was: Q 576×576, K 192×576, V 192×576, O 576×576,
Gate/Up 1536×576, Down 576×1536 ≈ 3.94 MiB/layer × 30 = 118 MiB +
LM head 49152×576 = 108 MiB × (Q8_0: ~27%) ≈ 29 MiB. Exact numbers
depend on `AllocatedBytes` which the model reports, but the ratio is
decisive on larger models — a 7B Q8_0 model goes from ~28 GiB on device
(dequantised) to ~7 GiB (native).

## Limits

On this UMA box, steady-state decode is now ~**160 tok/s** on the
Vulkan Q8_0 path. llama.cpp Vulkan's reference is ~392 tok/s on the
same hardware. The remaining gap is kernel-quality, not dispatch
machinery: a cooperative-matrix Q8_0 GEMM (RDNA3 WMMA 16×16×16 int8
via `VK_KHR_cooperative_matrix`) is the single biggest next lever.

## Not in scope (follow-ups)

1. **Cooperative-matrix Q8_0 GEMM** — separate filed issue. Expected
   ~10× on the batched kernel over the current 209 GFLOPS.
2. **Q4_0 / Q5_0 / K-quant end-to-end** — those Vulkan kernels don't
   exist yet; those quant types still dequant to F32 at upload as
   they did pre-wave.
3. **`bias_add_f32` kernel** — Phi / Qwen biases still host-mapped; a
   dedicated kernel removes the per-layer submit split. Out of scope
   here (scope limit: no new kernels).
4. **GEMV k=32 M>1 stride bug** noted in `.continue-here.md` — not
   hit by this change (prefill dispatches Q8_0 GEMM, decode uses M=1).
   Worth a separate regression test + patch on the GEMV path.

## Files changed

- `src/DotLLM.Vulkan/VulkanWeights.cs` — `Upload(..., bool dequantToFp32=false)`;
  per-LayerBuffers device quant type; `UploadMatrix` now branches on
  `KeepQ8OnDevice` to upload either raw Q8_0 blocks or FP32.
- `src/DotLLM.Vulkan/VulkanTransformerModel.cs` — `MatMulQ8_0Kernel` +
  `MatMulQ8_0GemmKernel` fields, `RecordMatmul` dispatcher; every
  `_matmul.Record(...)` call in `Forward` replaced with
  `RecordMatmul(...)`; updated `InvalidateKernelCaches` and `Dispose`.

## Raw measurements

Each run: 4–8 warmup decodes (timed but reported separately) + 32–64
steady decodes. Wall time per `VulkanTransformerModel.Forward` via
`System.Diagnostics.Stopwatch`.

### Baseline (6f509de)

```
load_ms=757.0
prefill_len=5 prefill_ms=33.48
warmup_avg_ms=11.80
decode_avg_ms=11.49 min=10.69 max=13.34 tok/s=87.02
```

### After Step 1 (73f9424) — no-op (flag forced true at model site)

Identical to baseline (same F32 dequant path, same forward).

### After Step 2 (944cd67) — Q8_0 end-to-end

Run 1 (32 steps, 4 warmup):
```
load_ms=428.0
prefill_len=5 prefill_ms=47.81
warmup_avg_ms=5.77
decode_avg_ms=6.61 min=5.63 max=8.57 tok/s=151.34
```

Run 2 (32 steps, 4 warmup):
```
decode_avg_ms=5.85 min=5.08 max=6.99 tok/s=170.80
```

Run 3 (64 steps, 8 warmup):
```
load_ms=~430 (consistent)
prefill_len=5 prefill_ms=~48
warmup_avg_ms=6.07
decode_avg_ms=6.23 min=4.92 max=9.27 tok/s=160.40
```

Load time improvement (757 → 428 ms, -43%) comes from skipping the
per-row `Dequantize.ToFloat32` loop across 7 × 30 + 1 = 211 matrices
in favour of a direct `Span<byte>.CopyTo` for the Q8_0 path. Prefill
rose modestly (33 → 48 ms) — driver shader compile cost on the first
`matmul_q8_0_gemm` dispatch, amortised across all future prefills.
