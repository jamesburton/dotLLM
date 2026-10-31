# Vulkan Backend Architecture — dotLLM

## Why a Vulkan Backend

The `DotLLM.Cuda` backend is NVIDIA-only: it P/Invokes the CUDA Driver API
(`libcuda.so` / `nvcuda.dll`) and cuBLAS, and loads PTX text files. On
non-NVIDIA GPUs — AMD Radeon, Intel Arc, Apple Silicon (via MoltenVK),
mobile Adreno/Mali — that path returns nothing.

`DotLLM.Vulkan` exists to cover that gap. It uses the same P/Invoke
philosophy as the CUDA backend — no custom C shared library — but targets
the Vulkan loader (`vulkan-1.dll` / `libvulkan.so.1`) and loads SPIR-V
compute shaders instead of PTX. The architectural parallel is exact:

| | CUDA backend | Vulkan backend |
|---|---|---|
| Native loader | `libcuda.so` / `nvcuda.dll` | `libvulkan.so.1` / `vulkan-1.dll` |
| Shader IR | PTX (text) | SPIR-V (u32 binary) |
| Kernel source | CUDA C++ (`.cu`) | GLSL compute (`.comp`) |
| Compiler | `nvcc -ptx` | `glslc --target-env=vulkan1.2 -O` |
| Module type | `CUmodule` | `VkShaderModule` |
| Launch | `cuLaunchKernel` | `vkCmdDispatch` |
| Vendor reach | NVIDIA only | AMD, NVIDIA, Intel, Apple (MoltenVK), Qualcomm, ARM |

The existing gap-analysis in `docs/CUDA.md` §1 concludes that Vulkan
compute is the only cross-vendor path with full custom-kernel expressivity
and a credible route to Tensor-Core-class throughput via
`VK_NV_cooperative_matrix2` (October 2024). That extension adds
dequantization callbacks specifically for quantized LLM inference and is
implemented by NVIDIA and AMD; Intel support is tracked in the Mesa ANV
driver. Recent benchmarks in llama.cpp's `ggml-vulkan` backend reach
~70–95% of native CUDA throughput on RTX 4090.

## Scope of This PR

**Proof of pipeline only.** This PR establishes the plumbing:

- `DotLLM.Vulkan` project with raw `[LibraryImport("vulkan-1")]` P/Invoke
  — no Silk.NET, no external bindings package. ~15 Vulkan entry points
  covering instance, physical device, logical device, memory, buffer,
  shader module, compute pipeline, descriptor sets, command buffer,
  queue submit.
- Shader build pipeline: `native/vulkan/shaders/*.comp` → `glslc` →
  `native/vulkan/spv/*.spv`, driven by `build.sh` / `build.ps1` and also
  wired into MSBuild (`CompileVulkanShaders` target) so shaders rebuild
  incrementally when sources change and `glslc` is on PATH.
- One working kernel: `add.comp` implements `c[i] = a[i] + b[i]` over
  FP32 buffers.
- `VulkanDevice` — instance creation, physical-device selection
  (discrete > integrated; prefer AMD/NVIDIA over Intel integrated),
  compute queue + command pool, host-visible buffer allocation,
  upload/download.
- `VulkanModule` — loads one `.spv`, creates compute pipelines by
  entry-point name.
- `AddKernel` — wraps `add.comp`; records a one-shot command buffer,
  binds descriptor set, pushes `n` via push constant, dispatches,
  waits.
- Smoke test `VulkanAddKernelTests` — verifies 1024-element addition
  round-trips correctly; skips when no Vulkan loader/device is present.

**Not in scope.** Real LLM kernels, multi-GPU, staging ring, fence-based
pipelining, `VK_NV_cooperative_matrix2`, descriptor-set pooling across
launches, memory-type heuristics beyond HOST_VISIBLE|HOST_COHERENT.

## SPIR-V Compilation Pipeline

End users need only the compiled `.spv` blobs, which ship as MSBuild
`Content` alongside the managed DLL. Shader *authors* need the Vulkan
SDK (https://vulkan.lunarg.com/) for `glslc`. The MSBuild target is
committed `.spv` files.

**Shader build rule (#1254, #1310):** always `glslc --target-env=vulkan1.2 -O` (build scripts and the MSBuild target
all pass it); committed `.spv` files are byte-reproducible with that command. Stay on `vulkan1.2` (SPIR-V 1.5, matching
the 1.2 instance) until a shader needs a 1.3 feature, then raise the instance requirement too. Bytes depend on the
`glslc` version: the committed set was verified with shaderc v2026.1 (spirv-tools v2026.1), so record the version when
regenerating a shader on a newer toolchain.

**glslc pin (#1364):** the verified toolchain is recorded in `native/vulkan/glslc.version` (`shaderc v2026.1`).
An ordinary build runs `CompileVulkanShaders` only when `glslc --version` starts with that string; on any other
glslc it **warns and skips**, so a build never rewrites tracked `.spv` with version-drifted bytes (previously ~30
files dirtied every worktree and PR). Without glslc it also just uses the committed blobs.
- `dotnet msbuild src/DotLLM.Vulkan -t:RegenerateVulkanShaders` rebuilds every `.spv` unconditionally (the
  explicit authoring path; warns when the glslc differs from the pin - then review the diff and bump the pin).
- `-p:DotLLMStrictSpv=true` turns the mismatch warning into an error.
- `VulkanSpvFreshnessTests` (non-GPU, runs in the unit-test job) recompiles every `.comp` with the pinned glslc
  and fails if any tracked `.spv` differs; it skips when glslc is absent or not the pinned version. The MSBuild
  incremental check is timestamp-based, and timestamps are arbitrary after a checkout, so only this test (not the
  build) proves a tracked `.spv` matches its `.comp`. Always run it after editing a shader.
committed `.spv` files.

```
native/vulkan/shaders/add.comp         (author edits)
            │
            │   glslc --target-env=vulkan1.2 -O -o add.spv add.comp
            ▼
native/vulkan/spv/add.spv              (checked in — ships to users)
            │
            │   <Content Include> in DotLLM.Vulkan.csproj
            ▼
bin/Debug/net10.0/spv/add.spv          (loaded at runtime)
            │
            │   File.ReadAllBytes → vkCreateShaderModule
            ▼
VkShaderModule handle
            │
            │   vkCreateComputePipelines
            ▼
VkPipeline → vkCmdBindPipeline → vkCmdDispatch
```

The driver compiles SPIR-V to vendor ISA at `vkCreateComputePipelines`
time and caches the result on disk (AMDGPU-PRO cache, Mesa shader cache,
NVIDIA internal blob). First-launch cost is amortized across process
restarts, same as PTX JIT under CUDA.

## P/Invoke Strategy

Identical to `DotLLM.Cuda`:

- `[LibraryImport("vulkan-1")]` with source-generated marshalling.
- `VulkanLibraryResolver` rewrites "vulkan-1" to the correct OS binary
  (`vulkan-1.dll`, `libvulkan.so.1`, `libvulkan.dylib`).
- Handles (`VkInstance`, `VkDevice`, `VkBuffer`, `VkDeviceMemory`, etc.)
  cross the boundary as opaque `nint` — tensor bytes never traverse
  P/Invoke.
- `VkResult` returned as `int`; negative values are errors, zero is
  `VK_SUCCESS`, positive values are non-error status codes
  (`VK_INCOMPLETE`).
- Structs declared `[StructLayout(LayoutKind.Sequential)]`. We only
  declare the fields actually used; extension tails are left as
  `fixed byte` padding (see `VkPhysicalDeviceProperties.limits`).

No Silk.NET dependency. Adding Silk.NET.Vulkan would give us ergonomic
bindings but ~15 MB of transitive DLLs and another layer to audit; the
~15 Vulkan entry points we actually need fit in a single file.

## Target Kernel Catalog

Future sessions port the `DotLLM.Cuda` catalog (see
`docs/CUDA.md` §Kernel Catalog) one kernel at a time. Order is chosen so
each lands a testable end-to-end slice:

| Phase | Kernel | CUDA file | Notes |
|---|---|---|---|
| 1 | `add`           | `add.cu`            | ✅ Done |
| 2 | `rmsnorm_f32`   | `rmsnorm_f32.cu`    | ✅ Done; warp reduction → subgroup shuffle (`GL_KHR_shader_subgroup`); fused rmsnorm+matmul_q8_0 sub-tile variant landed |
| 2 | `rope_f32`      | `rope_f32.cu`       | ✅ Done; sin/cos lookup; no reduction |
| 2 | `swiglu_f32`    | `swiglu_f32.cu`     | ✅ Done; pointwise |
| 3 | `embedding_*`   | `embedding.cu`      | ✅ Done; gather, F32 + Q8_0 in tree |
| 3 | `bias_add_f32`  | `bias_add_f32.cu`   | ✅ Done |
| 3 | `softmax`       | `softmax.cu`        | ✅ Done; two-pass (max, exp-sum), subgroup reduction |
| 4 | `attention_f32` | `attention_f32.cu`  | ✅ Done; standard + MLA (`AttentionMlaF32Kernel`) variants |
| 5 | `dequant_q8_0`  | `dequant.cu`        | ✅ Done (CPU-side); Vulkan reads source bytes inline |
| 5 | `dequant_q4_k` / `q5_k` / `q6_k`  | `dequant.cu`        | ✅ Done (CPU-side); Vulkan kernels read source bytes inline (Phase 1 K-quant work) |
| 6 | `matmul_q8_0_*` | `quantized_gemv.cu` | ✅ Done; GEMV (`MatMulQ8_0Kernel`) + GEMM (`MatMulQ8_0GemmKernel`) + coopmat (`MatMulQ8_0GemmCoopmatKernel`) |
| 6 | `matmul_q4_k_*` / `q5_k_*` / `q6_k_*` | `quantized_gemv.cu` | ✅ Done (Phase 1, commits `afb2272`/`b1ee6bc` Q4_K, `15099b9`/`83e0732` Q5_K, `29a1459`/`39b7646` Q6_K); GEMV + GEMM each |
| 7 | F16 / BF16 pipeline | `*_f16.cu` | ✅ Done (Phase 8 brought forward, commits `c9c08c5`/`6787492`); F16 GEMV + GEMM + coopmat (F16xF16→F32, gfx1151), BF16 GEMV + GEMM (`uintBitsToFloat(bits<<16)`); F32 activations + native F16/BF16 weights on device |
| 8 | Cooperative matrix | (new) | ✅ Partial (F16 + Q8_0 GEMM coopmat tiles via `GL_KHR_cooperative_matrix`; gfx1151 verified). MoE Strategy C now has a gated F16 grouped path: expert-count/offset, group-by-expert, grouped F16 coopmat matmul, ungroup scatter, and end-to-end synthetic forward parity. K-quant coopmat variants and non-F16 grouped MoE remain follow-ups |

Milestone 7 (FP16) is a significant enabler: most Vulkan drivers expose
`shaderFloat16` only through those two extensions. `VK_KHR_16bit_storage`
lets us read/write FP16 in storage buffers; `shader_float16_int8` lets
shaders operate on `float16_t` natively.

Milestone 8 (cooperative matrix) is the cross-vendor equivalent of
NVIDIA's `mma.sync` / Tensor Cores. `VK_NV_cooperative_matrix2` (Oct
2024) is the ticket to ~90% of cuBLAS FP16 GEMM throughput on a 4090,
and AMD's equivalent is on its RDNA3+ driver roadmap.

## GLSL Conventions

All kernels live under `native/vulkan/shaders/*.comp`. Conventions
mirror the CUDA kernels where possible:

- One `.comp` file per kernel; file name matches the CUDA file name
  (e.g. `rmsnorm_f32.comp` ↔ `rmsnorm_f32.cu`).
- `#version 450` for the baseline; bump to `#version 460` when
  cooperative matrix is needed.
- `layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;`
  matches the CUDA block size of 256.
- Storage buffers use `std430` layout and `readonly`/`writeonly`
  qualifiers as appropriate.
- Scalar uniforms (sizes, strides, configuration) are passed through
  push constants (max 128 bytes per pipeline). Larger config goes in
  a uniform buffer.
- Entry point is always `void main()`. The C# `AddKernel` passes
  `"main"` as the entry-point name to match.

## Physical-Device Selection

`VulkanDevice.Create` enumerates all physical devices and scores them:

- Device type: discrete (+1000), integrated (+500), virtual (+100), other (0).
- Vendor: NVIDIA/AMD (+20), Intel (+10), other (+5).

Highest score wins. Tie-breaking is first-enumerated. This prioritizes
a discrete GPU over an integrated one on hybrid laptops, and prefers
AMD/NVIDIA discrete over the (rare) Intel Arc discrete when both are
present. The heuristic is deliberately dumb — real placement policy
(explicit `--device` CLI flag, per-request device hints) is
infrastructure for later.

## Deferred Work

Tracked for future sessions:

1. **Staging buffers for large uploads.** The scaffold allocates
   host-visible buffers directly, which is OK for 1024 floats but will
   burn bandwidth on multi-GB model weights. Real uploads need a
   device-local destination plus a host-visible staging ring and
   `vkCmdCopyBuffer`.
2. **Descriptor-set pool reuse.** `AddKernel` currently allocates a
   descriptor set per launch. Port the per-kernel cache pattern from
   `CudaKernels`.
3. **Fence-based pipelining.** Every launch currently does
   `vkQueueWaitIdle` — synchronous, no overlap with host work.
   Replace with `VkFence` + per-in-flight command-buffer arena.
4. **`IBackend` integration.** `DotLLM.Vulkan` does not yet implement
   `DotLLM.Core.Backends.IBackend`. Hooking it up needs the
   `VulkanTransformerModel` equivalent of `CudaTransformerModel`, which
   in turn needs the attention/rmsnorm/rope kernels.
5. **Model loading.** GGUF weights → Vulkan buffers requires the mmap
   + staging path above.
6. **Validation layers.** `VK_LAYER_KHRONOS_validation` at
   instance-create time gives llama.cpp-style debug output. Opt-in via
   env var (`DOTLLM_VULKAN_VALIDATION=1`) once the SDK-install-check
   story is sorted.

## Building

```bash
# .NET build — compiles C# and (if glslc is on PATH) shaders.
dotnet build src/DotLLM.Vulkan/DotLLM.Vulkan.csproj

# Shader-only rebuild.
./native/vulkan/build.sh          # Linux / macOS / WSL / Git Bash
pwsh ./native/vulkan/build.ps1    # Windows PowerShell

# Run the scaffold test — passes if a Vulkan device is present, else skips.
dotnet test tests/DotLLM.Tests.Unit --filter FullyQualifiedName~Vulkan
```

Opt out on CI without a usable driver: `DOTLLM_SKIP_VULKAN=1`.

## Gemma-4 dense variant (E2B/E4B) (issue #734)

Supported: per-layer embeddings (PLE), trailing shared-KV layers, proportional `rope_freqs`, mixed head dim 256/512, dense GeGLU FFN, layer output scale, final logit soft-cap.

- PLE inputs are computed on the host with the CPU-oracle code (`Gemma4PerLayerInputs`) and uploaded layer-major; each layer copies its `[seq, pleDim]` slice and runs gate matmul -> GeGLU -> proj matmul -> post-norm -> add on the device.
- Shared layers read the donor layer's `VulkanKvCache` lines (cacheless: a 2-slot donor stash).
- `rope_freqs` is accepted only in the released `{1.0 x n, 1e30 x rest}` form (mapped onto partial rotary); any other table throws at load.
- Not supported: layer-split pipeline stages, TurboQuant KV, and the fused `ForwardBatch` path (falls back to per-sequence forwards).
- Measured on Strix Halo (gfx1151), Q4_K_M: decode 36.5 tok/s, prefill 381 tok/s at 128 tokens; wikitext PPL within 0.04% of CPU (see PR).

## Expert-grouped coopmat MoE prefill (issue #789, builds on #773/#785)

For prefill (S >= 16 tokens; `DOTLLM_VK_MOE_GROUPED_MIN_TOKENS`) the generic MoE layer (`VulkanTransformerModel.RecordMoeLayer`) no longer runs the
scalar indexed expert matmuls when every routed bank is a packed Q4_K / Q5_K / Q6_K / Q8_0 / MXFP4 tensor: it sorts the (token, slot) rows by expert once,
builds an indirect tile list (`moe_build_tile_list`) and runs one cooperative-matrix GEMM per (expert, 16-row tile) with a 64-row weight tile
(`moe_grouped_matmul_{q4_k,q5_k,q6_k,q8_0,mxfp4}_coopmat_m64`). The activation (SiLU, or gpt-oss's clamped OAI SwiGLU) is pointwise so it runs in packed
order; the down GEMM reads that packed activation directly; one ungroup restores the (token, slot) order for the unchanged weighted scatter. gpt-oss's
per-expert gate/up/down bias is added in the MXFP4 shader epilogue (`applyScale == 2`), the same single F32 add the scalar path does as a separate pass.
Operands are F16 (dequantised weights and activations), so logits differ from the scalar path by ~1% of their rms; see the tests for the measured envelopes.

* Opt-out: `DOTLLM_VK_MOE_GROUPED=0` (load time) or `VulkanTransformerModel.GroupedMoeEnabled = false` (runtime, in-process A/B).
* Decode (S < gate) and LoRA keep the indexed MMVQ kernels. Wave64 + cooperative matrix devices only; others fall back to the scalar path.
* Measured on gfx1151 (same session, split-submit profile): gpt-oss-20b pp512 74-82 -> 730-780 tok/s, OLMoE Q4_K_M pp512 `VK_ERROR_DEVICE_LOST` (#787) -> ~2900.

## Dense Q4_0 prefill on Q8_1 activations (issue #1447)

Dense Q4_0 projections at `n > 1` quantise the activations to Q8_1 (shared per q/k/v and gate/up group) and run
`matmul_q4_0_mmq_cm1` (int8 cooperative matrices, AMD wave64; `mmq_cm1_main.glsl` MMQ_Q4_0 arm, codes recentred to `q - 8` at
staging so there is no min term) or the dp4a `matmul_q4_0_mmq` fallback (Arc/NVIDIA; parity-tested on the 8060S only). Weights are read
packed; no expansion. `DOTLLM_VK_Q4_0_PREFILL_Q8_1=0` restores the F16 coopmat / F32 GEMMs.

Measured on the 8060S (paired in-process-env A/B, both orders, 3 rounds, `dotllm bench`, pure-Q4_0 Llama-3.2 requantised from Q8_0):
pp128 1B 4.0-4.2k -> 6.5-6.9k tok/s, pp512 1B 6.1-6.6k -> 6.9-7.3k, pp512 3B 2.37-2.42k -> 2.78-2.82k, pp2048 1B 5.1k -> 5.4-5.5k;
decode unchanged (3B 88-91, 1B 197-210). The kernel-level probe (`DOTLLM_GEMM_SHAPE_PROBE=1`) is only ~equal to the F16 coopmat GEMM
(+-10 %), so the end-to-end gain comes from outside the GEMM itself (the F16 path stages F32 activations to F16 per call, and the MMQ group shares one quantize); the split was not isolated.
wikitext-2 PPL (c512, 40k tokens, 1B): MMQ 16.614, CPU 16.614, F16 path 16.583. A 64x128 token tile did not help (unlike Q4_1).
The GPU matches the exact `d * d_x * dot(q - 8, x)`; the CPU/llama.cpp fold the -8 into the fp16-rounded `s_x`, so a GPU-vs-CPU GEMM
differs by up to ~1e-3 relative where activation block sums are large (a test artefact; real activations ~1e-5).
