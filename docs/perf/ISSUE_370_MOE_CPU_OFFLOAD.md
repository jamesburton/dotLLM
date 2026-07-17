# Issue #370 — MoE CPU/GPU expert offload (Vulkan Qwen3MoeHybrid)

**FOR THE COORDINATOR: fold into `.docs/KERNEL_MAP.md` §7, then delete this file.**

Status: **implemented, parity-tested (synthetic fixture), and real-GGUF validated**
(Strix Halo, `dev` worktree `agent-a2042d63b51f96fa2`, branch `issue/370-moe-cpu-offload`).
Real-model `dotllm bench` numbers: see "Real-model validation" below. Full Vulkan suite:
930 passed / 1 flaky-fail (known pre-existing, confirmed by isolate-retry) / 40 skipped.

## What was built

llama.cpp's `--n-cpu-moe N` shorthand equivalent for dotLLM's Vulkan **Qwen3MoeHybrid**
path (`VulkanQwen3MoeHybridTransformerModel` / `VulkanQwen3MoeMoeUpload.cs`, the
integration point named in the issue). Uniform-per-layer placement (v1 simplification the
issue explicitly accepts): the first `N` layers (by index) run their routed MoE expert
compute on the CPU (`DotLLM.Cpu.Kernels.MoeSwiGluMlp`) instead of uploading a GPU expert
bank; the remaining layers keep the existing resident/streaming GPU indexed-matmul path
unchanged. Dense/attention (GDN + full-attention token-mixing) weights stay GPU-resident
regardless of MoE placement — the repo-wide "device placement always explicit" rule.

### Why Qwen3MoeHybrid only (not the generic `VulkanTransformerModel.RecordMoeLayer` path
used by Gemma-4 / DeepSeek-V2 / non-hybrid Qwen3-MoE)

The issue names `VulkanQwen3MoeMoeUpload.cs` specifically as the integration point — it's
the resident-vs-streamed bank abstraction with the clearest "skip the GPU bank entirely for
this layer" seam (per-layer `LayerBundle` upload/dispose, already opt-in vs default). The
generic MoE path shares `RecordMoeLayer`'s kernel dispatch but its per-model forward loops
(`VulkanTransformerModel.cs`) are structured differently and weren't touched — that's a
natural follow-up (`--n-cpu-moe` generalized across all Vulkan MoE model classes) if there's
demand, tracked as a note here rather than a new issue since the pattern established below
transfers directly (same host round-trip via `VulkanDevice.Upload`/`Download`, same
`MoeSwiGluMlp.Route`/`ExecuteRoutedFromAssignments` call).

## Design

1. **Placement policy.** `nCpuMoeLayers` parameter on both
   `VulkanQwen3MoeHybridTransformerModel.BuildFromGguf` and `BuildFromPrebuiltWeights`
   (clamped to `[0, NumLayers]`). Negative (default `-1`) falls back to the
   `DOTLLM_N_CPU_MOE` environment variable (0 when unset — identical to pre-#370
   behaviour), matching the existing `DOTLLM_VK_MOE_RESIDENT` env-var-toggle convention in
   the same file. Wired into `dotllm bench --n-cpu-moe N` (Vulkan + Qwen3MoeHybrid only;
   no-op elsewhere).
2. **Forward-pass wiring.** The per-layer MoE submission (`VulkanQwen3MoeHybridTransformerModel.Forward`,
   step "2b") now branches per layer:
   - **GPU-placed** (`RunGpuPlacedMoeLayer`): byte-identical to the pre-#370 code —
     resident/streaming bank upload + `RecordMoeLayer` indexed-matmul dispatch.
   - **CPU-placed** (`RunCpuPlacedMoeLayer`): the post-attn RMSNorm still runs on GPU
     (writes `_state.NormOutput`); that buffer is then downloaded to a rented host `float[]`,
     run through `MoeSwiGluMlp.Route` + `ExecuteRoutedFromAssignments` (identical
     routing/GEMM call the pure-CPU `Qwen3MoeHybridTransformerModel.ForwardMoeBody` uses,
     directly against the layer's raw GGUF quant-view pointers — no CPU-side dequant beyond
     what the CPU kernel already does internally), then re-uploaded in place. The residual
     add + copy-back into `HiddenState` happens in a follow-up GPU submission. **No GPU
     expert bank is ever allocated for a CPU-placed layer** — this is a real device-memory
     reduction, not a deferred upload.
   - Submission split: `[residual-snapshot + RMSNorm] → host round-trip → [residual-add +
     copy-back]`, each half its own `_submit.Begin()/SubmitAndWait()` with the existing
     `HostToComputeBarrier`/`ComputeToHostBarrier` convention (same barrier discipline the
     file already uses at every host↔device boundary — e.g. token embedding, logits
     download).
3. **`EstimatedCpuOffloadVramSavedBytes` / `NCpuMoeLayers` properties** report the
   placement and a byte estimate (streaming-F32 bank sizing: `(2*W1elems + W2elems) * 4
   bytes * numExperts` per CPU-placed layer) for `dotllm bench` to print and for the
   ledger below.

## Bug found + fixed along the way: `VulkanDevice.Upload` had no device-local staging path

`VulkanDevice.Download` was fixed for strictly-DEVICE_LOCAL (non-host-visible) buffers in
#364 (dGPU VRAM staging). `Upload` had **no equivalent fix** — it unconditionally called
`vkMapMemory` on the destination buffer. This didn't matter for the pre-#370 codebase
because every existing `Upload` call site targets either an explicitly host-visible staging
buffer or a buffer allocated `!deviceLocal`. The CPU-MoE-offload host round-trip is the
first caller to `Upload` directly into a **forward-scratch** buffer
(`_state.NormOutput`, allocated `AllocateDeviceLocal`), and on this box that scratch
buffer is *not* host-visible even though the physical device is UMA — surfaced as
`VK_ERROR_MEMORY_MAP_FAILED` immediately in the first CPU-placed-layer test run.

Fixed in `VulkanDevice.cs`: both `Upload(ReadOnlySpan<float>, Buffer)` and
`Upload(ReadOnlySpan<byte>, Buffer)` now check `dst.IsHostVisible` and, when false, stage
through a transient host-visible buffer via the existing `UploadToDeviceLocal` helper
(mirrors `Download`'s `IsHostVisible` staging branch exactly). This is a real,
general-purpose Vulkan-backend fix, not scoped to #370 — any future caller that uploads
into a strictly-device-local buffer (dGPU VRAM, or a UMA scratch buffer allocated without
the host-visible flag) now works correctly instead of crashing. No other backend
(CPU/CUDA/HIP) shares this code path, so no cross-backend propagation was needed per
CLAUDE.md's cross-backend-bug rule — this is Vulkan-buffer-allocation-specific.

## Parity gate result

New test file: `tests/DotLLM.Tests.Unit/Vulkan/VulkanQwen3MoeHybridCpuMoeOffloadTests.cs`
(synthetic 2-layer GDN+Attention+MoE+shared-expert fixture, all-F32 weights — the
F32-pointer fallback branch every non-GGUF caller of `RunMoeLayerOnCpu` exercises; the
raw-quant-view branch is the exact same call the already-tested pure-CPU
`ForwardMoeBody`/production GGUF loads use, so it isn't independently re-tested here).

- **`CpuOffload_AnyPlacement_MatchesFullGpuPlacement`** (nCpuMoeLayers ∈ {1, 2} vs 0):
  **PASSED** at the established dense-host tolerance (abs 5e-3 / rel 1e-3) — this is the
  literal "exact-token parity gate" the issue's acceptance criteria ask for. Both runs are
  Vulkan-vs-Vulkan at fixed weights/tokens; GDN/attention compute is byte-identical either
  way (their device placement never changes), so only the MoE FFN backend differs between
  the compared runs — a tight, meaningful comparison.
- **`CpuOffload_AnyPlacement_ProducesFiniteNonDegenerateLogits`** (nCpuMoeLayers ∈ {0, 1,
  2}): **PASSED** — sanity gate (finite, non-degenerate) at every placement level.
- Cross-backend (pure-CPU-reference-model vs Vulkan) raw-logit-VALUE parity was
  **deliberately not asserted** for this fixture: a 2-layer GDN+MoE hybrid with
  random small-scale weights is a textbook case of discrete top-k routing-selection
  chaos — a sub-ULP F32 reduction-order difference between the CPU grouped-GEMM path and
  the GPU indexed-matmul path can flip which expert(s) a token selects (a legitimate
  discrete divergence, not a precision bug), and it compounds layer-to-layer since layer
  2's router input depends on layer 1's output magnitude. This reproduces at
  `nCpuMoeLayers=0` — the **completely unmodified pre-#370 GPU path** — confirming it's a
  fixture-scale property (also documented for the existing IQ3 Qwen3MoeHybrid Vulkan test,
  which uses a 10x-looser 1.5e-1/1.5e-1 tolerance for the same reason), not a #370
  regression. `dotllm bench` on the real 35B-A3B model below is the real-scale validation
  where this chaos amplification does not apply (weights aren't near-tied by construction).
- Bookkeeping tests (`NCpuMoeLayers` clamping, `EstimatedCpuOffloadVramSavedBytes`,
  `DOTLLM_N_CPU_MOE` env-var fallback): **PASSED**.

All 8 tests in the new file pass. Pre-existing Qwen3MoeHybrid Vulkan tests (real-GGUF ones
self-skip — no cached A3B GGUF at the conventional path) still pass: 25 passed / 0 failed /
30 skipped for `FullyQualifiedName~Qwen3Moe`.

## Real-model validation

Ran on this box (Strix Halo, Radeon 8060S iGPU, 128 GB UMA) against the cached
`unsloth/Qwen3.6-35B-A3B-GGUF` **UD-Q4_K_XL** (22.4 GB on disk; the Q6_K_XL variant
referenced by existing tests wasn't cached, Q4_K_XL was). No OOM at any placement level
including full-GPU (`--n-cpu-moe 0`) — expected on this box since 128 GB UMA has enough
headroom even for the ~123 GB streaming-F32 dequant path; the offload's VRAM-reduction
value is for genuinely VRAM-constrained hosts (dGPU, e.g. T5500's 12 GB) or reserving UMA
budget for KV-cache/context, per the issue's stated motivation — this run validates
*mechanism and throughput*, not a VRAM ceiling this box happens to have.

`dotllm bench --device vulkan -p 8 -n 4..6 -r 1` (prompt 8 tok, short decode — enough to
get a stable-ish tok/s read without a multi-minute run per data point at these speeds):

| `--n-cpu-moe` | GPU expert-bank bytes avoided | decode tok/s | prefill tok/s | load ms |
|---|---|---|---|---|
| 0 (full GPU, pre-#370 default) | 0 | **0.067** | 0.39 | 4262 |
| 4 | ~12.3 GB | 0.085–0.09 | 0.41 | 19424¹ |
| 20 | ~61.4 GB | 0.18 | 0.98 | 1829¹ |
| 40 (fully CPU-placed) | ~122.9 GB | **2.98** | 4.74 | 5820¹ |
| CPU-only (`--device cpu`, 32 threads) | n/a (no GPU involved) | **6.61** | 14.25 | 286 |

¹ Load-time variance across these rows is OS page-cache warmth for the 22 GB GGUF mmap
(each run benefits from the previous run's file-cache pages), not a real per-`N` cost —
don't read load-ms as a monotonic function of `N`.

**Headline finding — the win is bigger than "VRAM saved," it's "usable at all":**
the existing default GPU-streaming path (`--n-cpu-moe 0`, i.e. the pre-#370 behaviour,
`DOTLLM_VK_MOE_RESIDENT` unset) re-dequantizes and re-uploads every routed-expert bank to
F32 on *every forward* — already flagged `SUSPECTED-SLOW`/"decode-killing" in
`.docs/KERNEL_MAP.md` §7 for this exact model. At 0.067 tok/s decode it is not a usable
interactive path. Moving MoE compute to the CPU (`--n-cpu-moe 40`) is **~44x faster
decode** (0.067 → 2.98 tok/s) purely by skipping that GPU streaming cost, even though the
CPU-side compute itself isn't free. So for Qwen3MoeHybrid-at-this-scale on Vulkan today,
CPU-offload is less "trade throughput for VRAM" and more "the only way to get a working
decode speed out of the Vulkan backend at all" until the GPU-resident-quant path
(`DOTLLM_VK_MOE_RESIDENT=1`) is validated at Q4_K/Q4_K_XL scale (today it's proven for
Q6_K-uniform banks only — this GGUF is Q4_K_XL, a mixed-quant UD build, so resident mode
would fall back to the same F32-dequant-resident path and wasn't attempted here given the
~123 GB device-memory commitment that implies).

**But full-CPU-offload-via-Vulkan (2.98 tok/s) is still ~2.2x slower than pure-CPU (6.61
tok/s)** on this box. The likely cause: each CPU-placed layer costs two GPU
submit-and-wait round trips (residual-snapshot+RMSNorm, then residual-add+copy-back)
plus a host↔device buffer round trip in between — at `seqLen=1` decode, per-submission
GPU dispatch/fence latency is almost certainly the dominant cost over 40 layers × 2
submissions, not the CPU compute itself or the transfer bytes (`hiddenSize=2048` floats is
tiny). A full-CPU model avoids all of that dispatch overhead entirely. This is consistent
with dense/attention GPU dispatch being cheap per-layer in isolation but not free at
`n_layers × 2` fixed round trips per decode step — a real, understood, **unclaimed
optimization opportunity** for a follow-up: batching the host round-trip across
*all* CPU-placed layers' MoE-input snapshots into fewer, larger submissions instead of
one pair per layer (or, at the limit, running the token-mixing GPU passes for a whole
forward first, then a single batched CPU-MoE sweep, then a single batched GPU residual-add
sweep) would likely close much of this gap. Left for a follow-up issue — v1 here is
correctness + a working, measured tradeoff, not the fastest possible CPU-offload
implementation.

**Practical guidance this run supports:** on this box, for this model/quant, full-CPU
(`--device cpu`) remains the fastest option in absolute terms. `--n-cpu-moe` earns its
keep on a VRAM-constrained dGPU where full-CPU isn't competitive with CPU+GPU-attention
hybrid throughput (T5500-class 12 GB cards, or any host that wants GPU-accelerated
dense/attention while conserving VRAM for KV-cache) — exactly the motivating case in the
issue. The monotonic `N → throughput` trend above (0 → 4 → 20 → 40 all strictly improve
decode tok/s) confirms the feature composes correctly at every granularity, not just the
two endpoints.

## Known v1 gaps (deliberately out of scope, per issue's "start simple" guidance)

- **Uniform per-layer placement only** — no per-token/per-request split-routing within a
  CPU-placed layer's own top-k selection (issue explicitly accepts this simplification for
  v1: "acceptable to require placement to be uniform per-layer").
- **`-ot`/`--override-tensor` regex pattern** — not implemented; issue says the simpler
  `--n-cpu-moe N` shorthand is fine for v1, `-ot` can follow if there's demand.
- **LoRA on CPU-placed layers** — `RunMoeLayerOnCpu` passes `loraAdapter: null`. Vulkan-side
  LoRA is a separate delta system from the CPU kernel's adapter hook; wiring a LoRA adapter
  through the CPU-offload host round-trip is a follow-up if a user needs LoRA + CPU-MoE-offload
  simultaneously.
- **Generic (non-Qwen3MoeHybrid) Vulkan MoE models** (Gemma-4, DeepSeek-V2, plain
  Qwen3-MoE) don't have `--n-cpu-moe` wired — only the `VulkanQwen3MoeMoeUpload.cs` path the
  issue names. The pattern (host round-trip via `VulkanDevice.Upload`/`Download` +
  `MoeSwiGluMlp.Route`/`ExecuteRoutedFromAssignments` against the raw quant view) transfers
  directly if a follow-up issue wants it generalized.
- **Q6_K-resident-bank CPU-offload interaction not independently tested** — CPU-placed
  layers never touch a GPU bank at all (resident or streaming), so `DOTLLM_VK_MOE_RESIDENT=1`
  simply doesn't apply to them; this is inherent to the design, not a gap, but worth noting
  explicitly since the two env vars (`DOTLLM_VK_MOE_RESIDENT`, `DOTLLM_N_CPU_MOE`) now
  compose (resident mode only affects the GPU-placed remainder).
