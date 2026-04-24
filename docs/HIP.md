# HIP/ROCm Backend Architecture — dotLLM

## Overview

The HIP backend provides the AMD GPU execution path for dotLLM. It is an architectural mirror of [docs/CUDA.md](CUDA.md): same layering, same conventions, same approach — the only differences are library names (`libamdhip64` / `hipblas`) and the code-object format (`.co`) instead of PTX.

HIP is AMD's source-level CUDA clone. The kernels in `native/kernels/*.cu` translate to HIP with a mechanical rewrite (`hipify-perl` or manual search-and-replace of `cuda_` → `hip_`), and the runtime API reads like the CUDA Driver API with `cu` replaced by `hip`. This repository adopts the same design choice as the CUDA backend: **P/Invoke directly against AMD's runtime libraries; no custom C/C++ shared library.**

## Status

This is a **proof-of-pipeline scaffold**. The RmsNorm kernel is ported and a round-trip test confirms the P/Invoke → module-load → kernel-launch → memcpy pipeline. The full LLM forward pass is deferred — see [Kernel Catalog](#kernel-catalog-to-port) for the port list.

## Chosen Architecture: Code-Object Loading via HIP Module API

Kernels are written in HIP (`.hip` / `.cu`) and compiled to **code objects** (AMDGPU ELF, `.co` / `.hsaco`) using `hipcc --genco`. The HIP runtime loads them at application startup via `hipModuleLoadData` and exposes kernel entry points via `hipModuleGetFunction`. Launches go through `hipModuleLaunchKernel`.

```
┌──────────────────┐   hipcc --genco    ┌──────────────┐
│  rmsnorm.hip     │ ──────────────────► │ rmsnorm.co   │  (fat ELF; ships with app)
│  rope.hip        │   (bundled ISA      │ rope.co      │
│  attention.hip   │    for each         │ attention.co │
│  dequant.hip     │    --offload-arch)  │ dequant.co   │
└──────────────────┘                     └──────┬───────┘
                                                │ loaded at runtime
┌───────────────────────────────────────────────▼──────────────────┐
│  C# application                                                  │
│                                                                  │
│  [LibraryImport("amdhip64")]       ← AMD HIP runtime (on system) │
│  hipModuleLoadData(coBytes)        ← loads code object           │
│  hipModuleGetFunction(module)      ← gets kernel handle          │
│  hipModuleLaunchKernel(func, ...)  ← launches on GPU             │
│                                                                  │
│  [LibraryImport("hipblas")]        ← hipBLAS (on system)         │
│  hipblasHgemm(...)                 ← FP16 GEMM, matrix cores     │
└──────────────────────────────────────────────────────────────────┘
```

Unlike NVIDIA's PTX (text IR, JIT'd per GPU), a `.co` file is a fat ELF bundle containing pre-compiled AMDGPU ISA for each `--offload-arch` target. Cross-architecture distribution therefore requires compiling for every target at build time. For the scaffold we target `gfx1030` (RDNA2, RX 6000) and `gfx1100` (RDNA3, RX 7000); override with the `HIP_ARCHS` env var in `native/hip/build.sh` / `build.ps1`.

## Library Layering

```
DotLLM.Hip (managed, this project)
├── Interop/
│   ├── HipDriverApi.cs       — [LibraryImport("amdhip64")]  ~25 fns
│   ├── HipBlasApi.cs         — [LibraryImport("hipblas")]   ~6  fns
│   ├── HipLibraryResolver.cs — DllImportResolver: versioned .so / .dll
│   ├── HipException.cs
│   └── HipErrorHelper.cs     — ThrowOnError / ThrowOnHipBlasError
├── HipContext.cs  — RAII context (hipCtxCreate/hipCtxDestroy)
├── HipStream.cs   — RAII stream  (hipStreamCreate/hipStreamDestroy)
├── HipModule.cs   — Code-object loader (hipModuleLoadData + cache)
├── HipDevice.cs   — Enumeration, device props, host/device memcpy
└── Kernels/
    └── RmsNormKernel.cs      — Wraps rmsnorm.co, exposes typed Launch()
```

Exactly mirrors `DotLLM.Cuda`. Every type here has a direct counterpart there.

## P/Invoke Approach

No custom C library. `[LibraryImport]` source-generated P/Invoke declarations call AMD's runtime directly:

- **`amdhip64`** — the HIP runtime. `libamdhip64.so[.6]` on Linux, `amdhip64.dll` on Windows. Installed with ROCm (Linux) or AMD's ROCm for Windows release.
- **`hipblas`** — hipBLAS, AMD's cuBLAS counterpart. `libhipblas.so` / `hipblas.dll`.

`HipLibraryResolver.Register()` is called before the first P/Invoke and routes requests through a `NativeLibrary.SetDllImportResolver` that tries versioned Linux names (`libamdhip64.so.6`, `.5`, etc.) — the same pattern as the CUDA resolver.

`HipDevice.IsAvailable()` probes for the runtime library with `NativeLibrary.TryLoad` **before** touching any generated P/Invoke stub. This lets the assembly load cleanly on systems without ROCm installed and lets callers gracefully degrade.

## HIP vs CUDA: Key API Mappings

Mechanical translation covers ~95% of call sites. A reference for anyone porting a kernel or runtime call:

| CUDA Driver API          | HIP Runtime API          | Notes |
|--------------------------|--------------------------|-------|
| `cuInit`                 | `hipInit`                | |
| `cuDeviceGet(Count)`     | `hipDeviceGet` / `hipGetDeviceCount` | |
| `cuCtxCreate_v2`         | `hipCtxCreate`           | HIP drops the `_v2` suffixes |
| `cuMemAlloc_v2`          | `hipMalloc`              | |
| `cuMemcpyHtoD_v2`        | `hipMemcpyHtoD`          | |
| `cuModuleLoadData`       | `hipModuleLoadData`      | takes .co instead of PTX |
| `cuModuleGetFunction`    | `hipModuleGetFunction`   | |
| `cuLaunchKernel`         | `hipModuleLaunchKernel`  | name differs, signature identical |
| `cuStreamCreate(flags)`  | `hipStreamCreate()`      | HIP's variant has no flags arg |
| `cuGetErrorString`       | `hipGetErrorString`      | returns `const char*` directly, no out-param |
| `cublasHgemm`            | `hipblasHgemm`           | |
| `cublasGemmEx`           | `hipblasGemmEx`          | constant values differ — see below |

For **kernel source**, the hipify mapping is just as mechanical:

| CUDA                       | HIP                             |
|----------------------------|---------------------------------|
| `<cuda_fp16.h>`            | `<hip/hip_fp16.h>`              |
| `<cuda_runtime.h>`         | `<hip/hip_runtime.h>`           |
| `__global__` / `__device__` | unchanged                      |
| `__syncthreads()`          | unchanged                       |
| `__shfl_down_sync(mask,v,n)`| `__shfl_down(v,n)` — HIP drops the mask (implicit full wave) |
| `cudaStream_t`             | `hipStream_t`                   |
| `warpSize`                 | unchanged (32 on RDNA wave32 / CDNA, 64 on older wave64) |

**Note on constants**: `hipblas` uses enum values that do NOT match cuBLAS. `HIPBLAS_OP_N = 111` (not 0), `HIPBLAS_R_16F = 150` (not 2), etc. These are wired correctly in `HipBlasApi.cs`.

## Kernel Catalog (to port)

All kernels from [docs/CUDA.md §Kernel Catalog](CUDA.md#kernel-catalog) have a hipify-perl-ready counterpart. Ported kernels live in `native/hip/kernels/*.hip`, compile to `native/hip/co/*.co`, and are wrapped by typed classes under `src/DotLLM.Hip/Kernels/`.

**Ported in this scaffold:**

| Kernel | File | Entry Points | Wrapper |
|--------|------|--------------|---------|
| RMS Norm | `rmsnorm.hip` | `rmsnorm_f32`, `rmsnorm_f16` | `RmsNormKernel` |

**To port (mechanical hipify pass):**

| Kernel | Source (CUDA) | Entry Points |
|--------|---------------|--------------|
| Fused Add + RmsNorm | `fused_add_rmsnorm.cu` | `fused_add_rmsnorm_f16` |
| Per-Head RmsNorm | `per_head_rmsnorm.cu` / `per_head_rmsnorm_f32.cu` | `per_head_rmsnorm_f16`, `per_head_rmsnorm_f32` |
| RmsNorm (FP32-in→FP16-out) | `rmsnorm_f32in.cu` | `rmsnorm_f32_in_f16_out` |
| RoPE | `rope.cu` / `rope_f32.cu` | `rope_f16`, `rope_f32` |
| Attention (naive) | `attention.cu` / `attention_f32.cu` | `attention_f16`, `attention_f32` |
| SwiGLU | `swiglu.cu` / `swiglu_f32.cu` | `swiglu_f16`, `swiglu_f32` |
| Add | `add.cu` / `add_f32.cu` | `add_f16`, `add_f32`, `add_f32_f16` |
| Bias Add | `bias_add.cu` / `bias_add_f32.cu` | `bias_add_f16`, `bias_add_f32` |
| Softmax | `softmax.cu` | `softmax_f16` |
| Embedding | `embedding.cu` / `embedding_f32out.cu` | `embedding_lookup_{f32,f16,q8_0}`, plus f32out variants |
| Convert | `convert.cu` | `convert_f16_to_f32`, `convert_f32_to_f16` |
| Dequant | `dequant.cu` | `dequant_{q8_0,q4_0,q5_0,q4_k,q5_k,q6_k}_f16` |
| Quantized GEMV | `quantized_gemv.cu` / `quantized_gemv_f32in.cu` | `quantized_gemv_{q8_0,q4_k,q5_0,q5_k,q6_k}` + f32-in variants |
| KV quant | `quant_kv.cu` | `quant_kv_*` family |

**hipBLAS for GEMM** — prefill path uses `hipblasHgemm` / `hipblasGemmEx` directly, mirroring cuBLAS. The P/Invoke declarations already exist in `HipBlasApi.cs`; a `HipGemm.cs` wrapper (mirror of `CudaGemm.cs`) lands with the first GEMM-consuming caller.

## Build Workflow

### Developer with ROCm installed

```bash
# one-off: compile all kernels
bash native/hip/build.sh
# or on Windows:
pwsh native/hip/build.ps1

# then a normal solution build copies .co files into the output directory
dotnet build
```

`DotLLM.Hip.csproj` also runs an incremental `hipcc --genco` MSBuild target before build when `hipcc` is in `PATH`, matching the CUDA project's `CompileCudaPtx` target.

### Downstream user without ROCm

Pre-compiled `.co` files ship alongside the .NET binaries under `runtimes/*/native/` (to be wired up when distribution tooling lands). The assembly builds and loads cleanly without ROCm; `HipDevice.IsAvailable()` returns `false` and the backend is skipped.

## Runtime Requirements

- **AMD GPU**: supported architectures depend on which `--offload-arch` targets are compiled in. Default: `gfx1030` (RDNA2 — RX 6000 series) and `gfx1100` (RDNA3 — RX 7000 series). Add `gfx942` for MI300, `gfx90a` for MI250, `gfx900` for Vega, etc.
- **ROCm**: 6.x recommended (5.x likely works but untested). Linux: install `rocm-hip-runtime`, `hipblas`, and the `amdgpu` kernel driver. Windows: install AMD's ROCm for Windows release (HIP SDK).
- **Driver**: `amdgpu` kernel driver (Linux) / AMD Adrenalin with ROCm support (Windows).

## hipify-perl Workflow

When porting a new kernel:

1. Copy the `.cu` file to `native/hip/kernels/` renamed to `.hip`.
2. Run `hipify-perl <file.hip> -inplace` (if available) — translates headers, atomics, shuffle intrinsics, typedefs. The tool is part of the ROCm HIPIFY package.
3. Manual fix-ups: any `__shfl_*_sync` calls simplify to `__shfl_*` (drop the mask arg). Shared-memory sizes that assume 32-wave reduction may need to grow to 64 to cover wave64 GPUs (see `rmsnorm.hip` for an example).
4. Compile with `hipcc --genco --offload-arch=<target> -o out.co in.hip`.
5. Verify numerical parity against the CPU reference in a unit test (pattern: `HipRmsNormKernelTests`).

Reference: [ROCm HIPIFY docs](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/).

## Future Work

- Port remaining kernels per [Kernel Catalog (to port)](#kernel-catalog-to-port).
- `HipGemm.cs` — row-major wrapper around `hipblasHgemm` using the same transpose trick as cuBLAS.
- `HipKvCache.cs`, `HipWeights.cs`, `HipForwardState.cs` — mirror the CUDA counterparts.
- `HipTransformerModel.cs` full implementation (currently a NotImplemented stub).
- CI job on an AMD runner to exercise the GPU tests.
- Distribution: package pre-built `.co` files under `runtimes/{rid}/native/` for NuGet consumption.
- RCCL (ROCm's NCCL counterpart) for multi-AMD-GPU tensor parallelism — same P/Invoke pattern, same library layering.
