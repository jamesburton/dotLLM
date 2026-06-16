# Cross-device offload (iGPU + eGPU) — scoping (2026-06-15)

Multi-week feature scoping for **item (4)** of the local-platform campaign: split a model's layers across
the **Intel Arc iGPU (Vulkan)** and the **NVIDIA RTX 3060 (CUDA, eGPU)** on this Meteor Lake box, and/or
keep the KV cache on a different device than compute. Goal: compare **iGPU-only vs iGPU+eGPU**.

> **Status: SCOPED, NOT STARTED.** This is the durable record of the milestone breakdown so the build can
> begin later with explicit go. Nothing here is implemented. DP4a (item 1) is the foreground item; offload is
> the parallel *scoping-only* track per the user's "both in parallel" — build does not start until item 1 is
> unblocked and the user greenlights the multi-week commitment.

## What already exists and is reusable

**Core abstractions — ready as-is:**
- `IBackend` (`src/DotLLM.Core/Backends/IBackend.cs`): `AllocateOnDevice`, `CopyBetweenDevices`, `Send`/`Receive` already specified.
- `DevicePlacement` (`src/DotLLM.Core/Backends/DevicePlacement.cs`): `-1 = CPU`, `0..N = GPU`; used by every tensor.
- `ITensor` / `TensorMetadata` (`src/DotLLM.Core/Tensors/`): `DeviceId` on every tensor — device identity already tracked everywhere.
- `BackendCapabilities` (`src/DotLLM.Core/Backends/BackendCapabilities.cs`): `VulkanIgpu` + `CudaDiscrete` profiles defined, Meteor-Lake aware.

**Working templates to clone:**
- `HybridTransformerModel` (`src/DotLLM.Cuda/HybridTransformerModel.cs`): CUDA-layers-then-CPU-layers split with FP16 D2H boundary + `TensorPrimitives.ConvertToSingle`. **Exact structural template** for a Vulkan+CUDA split.
- `HybridKvCache` (`src/DotLLM.Cuda/HybridKvCache.cs`): routes `Update()`/`GetKeysRef()`/`GetValuesRef()` by layer index.
- `HybridPrefillDecodeStrategy` (`src/DotLLM.Engine/Strategies/HybridPrefillDecodeStrategy.cs`): `HybridKvHandoff` already wired to `VulkanKvCache.IngestFromHost` — the CPU→Vulkan KV upload path is built.

**Transfer glue present:**
- `VulkanKvCache.IngestFromHost(layerIndex, length, ReadOnlySpan<float> keys, values)`: stages FP32 host → device-local VkBuffer.
- `SimpleKvCache.KeysSpan/ValuesSpan(layerIndex)`: host-side KV read surface.
- `CudaBackend.CopyBetweenDevices`: H2D/D2H/D2D via `cuMemcpy*` — the host-staging primitive.

## New components required

| Component | Where | What |
|-----------|-------|------|
| **`VulkanBackend : IBackend`** | `src/DotLLM.Vulkan/VulkanBackend.cs` | **Prerequisite zero — currently absent.** `AllocateOnDevice`, `CopyBetweenDevices` (host↔Vulkan via staging, same pattern as `VulkanKvCache.IngestFromHost`), `DeviceCount=1`. `Send`/`Receive`/`AllReduce` throw initially. |
| **`HybridVulkanCudaTransformerModel`** | `src/DotLLM.Cuda/` | Mirrors `HybridTransformerModel`: Phase 1 Vulkan layers `0..V-1`; Phase 2 host-staged FP32 handoff (D2H Vulkan, H2D CUDA); Phase 3 CUDA layers `V..L-1`. |
| **`HybridVulkanCudaKvCache`** | `src/DotLLM.Cuda/` | `IKvCache` routing: `0..V-1` → `VulkanKvCache`, `V..L-1` → `CudaKvCache`. |
| **Vulkan→host hidden download** | inside hybrid model | `vkCmdCopyBuffer` device-local → HOST_VISIBLE staging + fence wait. FP32 native (no half conversion on Vulkan side). |
| **Host→CUDA hidden upload** | inside hybrid model | `cuMemcpyHtoD_v2` of FP32 staging into `_gpuState.HiddenState`. **Precision decision:** keep FP32 through the boundary for the first milestone (avoid a 2nd conversion). |
| **Weight partition loader** | Vulkan + CUDA | `CudaWeights.LoadFromGguf` needs a `firstLayer` offset (today loads `0..N-1`); Vulkan side loads `0..V-1`. |

## Single biggest technical risk

**No Vulkan↔CUDA direct path; all handoffs stage through host RAM, and the Vulkan forward is a single
`vkQueueSubmit` + fence-wait per call — so Phase 1 (Vulkan) and Phase 3 (CUDA) cannot overlap.**

- Bandwidth is survivable: Arc Xe-LPG is UMA (device "copy" is into HOST_VISIBLE LPDDR5X, ~free once fence fires); RTX 3060 H2D over PCIe ≈16 GB/s → decode hidden (16 KB) <1 µs, 2048-token prefill (32 MB) ~2 ms.
- The risk is **serialization latency, not bandwidth.** `VkFenceWait → H2D → CUDA kernel` is strictly sequential; no async double-buffering today. This caps the split's speedup until async submission (Vulkan timeline semaphore + CUDA external semaphore) exists.
- Secondary: Vulkan outputs FP32, CUDA prefill internally expects FP16 — needs a deliberate precision decision at the boundary or it silently degrades quality.

## Phase breakdown

- **M0 — `VulkanBackend : IBackend`** (~2 d). Prerequisite; without it no `IBackend`-level coordination.
- **M1 — concept proof: 2-layer split Vulkan(0)+CUDA(1), FP32 host-staged handoff** (~3–4 d). Minimal hybrid model; reuse `SimpleKvCache` (Vulkan layer) + `CudaKvCache`. Validate: SmolLM-135M logit argmax matches Vulkan-only and CUDA-only within bf16 tolerance.
- **M2 — `HybridVulkanCudaKvCache` + configurable split point** (~2–3 d). Arbitrary `numVulkanLayers`; `CudaWeights.LoadFromGguf` `firstLayer` offset. Validate on Llama-3.2-1B (split 8/16).
- **M3 — async pipelining** (~1 wk, separate issue). Timeline/external semaphores so CUDA H2D overlaps Vulkan recording. This is what makes the split *faster*, not just correct.

**Out of scope until later:** NCCL/ParallelismConfig, non-contiguous layer→device maps, DiffusionGemma (needs the diffusion family *and* this offload).
