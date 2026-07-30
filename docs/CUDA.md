# CUDA Backend Architecture — dotLLM

## GPU Acceleration from .NET: Alternatives Research

Before committing to the native C/CUDA shared library approach (CMake → `libdotllm_native.so` → P/Invoke), we evaluated every viable path to GPU compute from C#/.NET. The goal: determine whether dotLLM can avoid creating a C/C++ library project while maintaining competitive inference performance.

### Evaluated Approaches

**ILGPU** (v1.5.3, ilgpu.net) is the strongest pure-C# option. It JIT-compiles C# kernel methods through .NET IL → SSA IR → CUDA PTX at runtime, entirely in managed code. Sponsored by G-Research, actively maintained, ~1,700 GitHub stars. Key strength: built-in cuBLAS wrapper in `ILGPU.Algorithms` providing FP16 GEMM with automatic Tensor Core usage. Key gaps: no Tensor Core access from custom kernels (no `wmma`/`mma.sync` API), no bfloat16 support, no published LLM workload benchmarks (one third-party benchmark showed ~3.7× slower than native for a physics workload). Flash Attention would be technically possible but impractical without direct Tensor Core access.

**ManagedCuda** (v13.0.64 by Michael Kunz) wraps the CUDA Driver API 1:1 in C# — loads PTX modules, calls cuBLAS, manages device memory. No kernel compilation, just orchestration. Critical issue: **switched to GPLv3/Commercial dual license from CUDA 12 onward**, making it incompatible with dotLLM's GPL v3 license without careful verification, and problematic for any future relicensing. Single-maintainer continuity risk. However, the architectural pattern it uses — P/Invoke against NVIDIA's own driver libraries + PTX loading — is freely replicable.

**ComputeSharp** (v3.2.0 by Sergio Pedri / Microsoft) transpiles C# to HLSL via Roslyn source generators, runs on DirectX 12. Production-proven (Microsoft Store, Paint.NET 5.0) but **fundamentally unsuitable**: Windows-only (DX12 hard dependency, cross-platform explicitly rejected), no CUDA/cuBLAS, no FP16, no tensor cores.

**Silk.NET** (v2.23.0, .NET Foundation) provides low-level Vulkan/OpenCL/DirectX bindings. **No CUDA bindings** (GitHub issue #558 never implemented). Vulkan compute path is viable — `VK_NV_cooperative_matrix2` (Oct 2024) adds dequantization callbacks specifically for quantized LLM inference, and recent benchmarks show Vulkan approaching CUDA parity on RTX 4090. But you'd write GLSL shaders, not C#, and build an entire compute framework — similar effort to CUDA with less ecosystem support.

**NVIDIA provides no official .NET SDK.** Confirmed by NVIDIA staff (March 2025): "None of them are directly provided by or supported by NVIDIA." Historical projects Alea GPU and Hybridizer are both defunct.

**OpenCL** lacks Tensor Core access entirely. **WebGPU** has buffer size limits (256MB–1GB, insufficient for model weights). **No Roslyn source generator compiles C# to PTX.** **.NET 9/10 add zero GPU compute** — `System.Numerics.Tensors` is CPU-only SIMD.

### Conclusion

No pure-C# approach matches native CUDA for LLM inference. However, the ManagedCuda pattern — P/Invoke directly against NVIDIA's system libraries + load PTX text files — achieves full native CUDA performance **without creating any C/C++ shared library**. We adopt this approach with our own minimal P/Invoke declarations (~30 functions), avoiding the ManagedCuda dependency and its GPLv3 license.

### Capability Comparison

| Capability | ILGPU (pure C#) | ManagedCuda + PTX | Own P/Invoke + PTX | Vulkan via Silk.NET |
|---|---|---|---|---|
| **cuBLAS GEMM** | ✅ Built-in wrapper | ✅ Full wrapper | ✅ Direct calls | ❌ No (use coopmat) |
| **Custom GPU kernels** | ✅ C# kernels | ✅ CUDA C → PTX | ✅ CUDA C → PTX | ✅ GLSL → SPIR-V |
| **Tensor Cores (custom)** | ❌ Only via cuBLAS | ✅ Full access | ✅ Full access | ✅ Via cooperative matrix |
| **FP16 / BF16** | FP16 only, no BF16 | ✅ Full | ✅ Full | ✅ FP16 (BF16 varies) |
| **Flash Attention** | ⚠️ Very difficult | ✅ Native quality | ✅ Native quality | ✅ Proven in research |
| **Memory management** | ✅ Full control | ✅ Full control | ✅ Full control | ✅ Full control |
| **Linux support** | ✅ | ✅ | ✅ | ✅ |
| **No C/C++ build system** | ✅ | ✅ (nvcc only) | ✅ (nvcc only) | ✅ (glslc only) |
| **License risk** | NCSA (permissive) | GPLv3/Commercial | None (own code) | MIT (Silk.NET) |
| **Multi-vendor GPU** | ❌ | ❌ NVIDIA only | ❌ NVIDIA only | ✅ Cross-vendor |
| **Perf vs native CUDA** | ~60–80% estimated | ~98–100% | ~98–100% | ~70–95% (improving) |

---

## Chosen Architecture: PTX Loading via CUDA Driver API

dotLLM uses NVIDIA's **CUDA Driver API** (`libcuda.so` / `nvcuda.dll`) and **cuBLAS** (`libcublas.so` / `cublas64_*.dll`) directly via P/Invoke. CUDA kernels are written in `.cu` files, compiled to PTX with a single `nvcc -ptx` command (no CMake, no shared library project), and loaded at runtime. The application is entirely C# — PTX files ship alongside the .NET assemblies as content files.

### How It Works

```
┌─────────────────┐     nvcc -ptx      ┌──────────────┐
│  rmsnorm.cu     │ ──────────────────► │ rmsnorm.ptx  │  (text file, ships with app)
│  rope.cu        │   (one command,     │ rope.ptx     │
│  attention.cu   │    no build system) │ attention.ptx│
│  dequant.cu     │                     │ dequant.ptx  │
└─────────────────┘                     └──────┬───────┘
                                               │ loaded at runtime
┌──────────────────────────────────────────────▼───────────────────┐
│  C# application                                                  │
│                                                                  │
│  [LibraryImport("cuda")]     ← NVIDIA's driver (on system)      │
│  cuModuleLoadData(ptxBytes)  ← loads PTX text into module        │
│  cuModuleGetFunction(module) ← gets kernel handle                │
│  cuLaunchKernel(func, ...)   ← launches on GPU                  │
│                                                                  │
│  [LibraryImport("cublas")]   ← NVIDIA's cuBLAS (on system)      │
│  cublasHgemm(...)            ← Tensor Core FP16 GEMM            │
└──────────────────────────────────────────────────────────────────┘
```

### What Libraries Are Involved

**`libcuda.so` / `nvcuda.dll`** — the CUDA Driver API. Installed with every NVIDIA GPU driver. Provides: device enumeration, context management, memory allocation, PTX module loading, kernel launching, stream management. This is what ManagedCuda wraps, and what we P/Invoke directly.

**`libcublas.so` / `cublas64_*.dll`** — cuBLAS. Installed with the CUDA Toolkit. Provides FP16 GEMM (`cublasHgemm`) with automatic Tensor Core usage when matrix dimensions are multiples of 8 — the single most important operation in LLM inference.

No dotLLM-authored `.so` or `.dll` is ever created.

### PTX: Text-Based GPU Intermediate Representation

PTX (Parallel Thread Execution) is NVIDIA's virtual instruction set — a text-based intermediate representation that the GPU driver JIT-compiles to native SASS instructions for the specific GPU at load time. It is architecturally analogous to SPIR-V for Vulkan or DXIL for DirectX — a shader file, not a compiled binary.

Key properties:
- **Architecture-independent**: the same PTX file runs on any GPU from sm_50 (Maxwell) through sm_90 (Hopper). The driver handles ISA translation.
- **Text-based**: human-readable, diffable, embeddable as string constants or .NET embedded resources.
- **JIT-compiled**: first load incurs ~100–500ms JIT compilation per module. The driver caches compiled kernels across runs (`~/.nv/ComputeCache`).
- **Fatbin option**: `nvcc` can produce fatbin files bundling PTX with pre-compiled SASS for specific architectures, eliminating JIT overhead entirely.

### Comparison with Original Plan (Shared Library Approach)

The original Step 31 plan creates `native/CMakeLists.txt` → builds `libdotllm_native.so`/`.dll` → wraps all CUDA operations behind a flat C API header (`dotllm_native.h`) → P/Invokes through `NativeMethods.cs`.

| Aspect | Shared Library (original) | PTX Loading (chosen) |
|---|---|---|
| **Build system** | CMake project, multi-target | `nvcc -ptx` (one command) |
| **Output artifact** | `libdotllm_native.so/.dll` | `*.ptx` text files |
| **C wrapper layer** | ~30 C functions in header | None — P/Invoke NVIDIA's API directly |
| **Memory/stream mgmt** | Custom C wrappers around cudaMalloc | Direct cuMemAlloc_v2 P/Invoke |
| **cuBLAS access** | Through custom C wrapper | Direct cublasHgemm P/Invoke |
| **Kernel code** | Same `.cu` kernels | Same `.cu` kernels (with `extern "C"`) |
| **Error handling** | Custom error codes | CUDA Driver API error codes (CUresult) |
| **Cross-compilation** | CMake cross-compile + RID packaging | PTX is arch-independent, no cross-compile |
| **CI complexity** | CUDA Toolkit + CMake + native build | CUDA Toolkit + single nvcc command |
| **Runtime dependency** | libdotllm_native + libcuda + libcublas | libcuda + libcublas (system-installed) |

The CUDA kernel code is **identical** in both approaches — the same RMSNorm, RoPE, attention, dequantization kernels. The difference is purely in how they're compiled (shared library vs PTX) and how C# calls them (through a custom C wrapper vs directly through NVIDIA's Driver API).

---

## P/Invoke Layer

### CUDA Driver API Declarations

~25 function declarations against `libcuda.so` / `nvcuda.dll`, covering initialization, device queries, context management, PTX module loading, kernel launching, memory operations, streams, and error handling. Following existing `CpuAffinity.cs` conventions with `[LibraryImport]` source generator.

```csharp
// src/DotLLM.Cuda/Interop/CudaDriverApi.cs
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal P/Invoke declarations against NVIDIA's CUDA Driver API.
/// libcuda.so (Linux) / nvcuda.dll (Windows) — installed with GPU driver.
/// All functions return CUresult (int): 0 = CUDA_SUCCESS, non-zero = error.
/// </summary>
internal static partial class CudaDriverApi
{
    // .NET resolves "cuda" to libcuda.so (Linux) / nvcuda.dll (Windows)
    private const string LibName = "cuda";

    // ── Initialization ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuInit(uint flags);

    // ── Device ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceTotalMem_v2(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuCtxCreate_v2(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int cuCtxDestroy_v2(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetCurrent(out nint ctx);

    // ── Module (PTX loading) ────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadData(out nint module, nint ptxImage);

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadDataEx(
        out nint module, nint ptxImage, uint numOptions,
        nint options, nint optionValues);

    [LibraryImport(LibName)]
    internal static partial int cuModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int cuModuleUnload(nint module);

    // ── Kernel launch ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuMemAlloc_v2(out nint devicePtr, nuint bytesize);

    [LibraryImport(LibName)]
    [SuppressGCTransition] // trivially short — just cudaFree
    internal static partial int cuMemFree_v2(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoD_v2(
        nint dstDevice, nint srcHost, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoH_v2(
        nint dstHost, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoD_v2(
        nint dstDevice, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoDAsync_v2(
        nint dstDevice, nint srcHost, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoHAsync_v2(
        nint dstHost, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemsetD8_v2(nint dstDevice, byte value, nuint n);

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuStreamDestroy_v2(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    // ── Error ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorName(int error, out nint str);

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorString(int error, out nint str);
}
```

### cuBLAS Declarations

~6 function declarations for GEMM operations:

```csharp
// src/DotLLM.Cuda/Interop/CublasApi.cs
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal cuBLAS P/Invoke. libcublas.so / cublas64_*.dll — from CUDA Toolkit.
/// </summary>
internal static partial class CublasApi
{
    private const string LibName = "cublas";

    [LibraryImport(LibName)]
    internal static partial int cublasCreate_v2(out nint handle);

    [LibraryImport(LibName)]
    internal static partial int cublasDestroy_v2(nint handle);

    [LibraryImport(LibName)]
    internal static partial int cublasSetStream_v2(nint handle, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cublasSetMathMode(nint handle, int mode);
    // CUBLAS_TENSOR_OP_MATH = 1 — enable Tensor Core usage

    // FP16 GEMM — C = alpha * op(A) * op(B) + beta * C, all FP16
    // Tensor Cores used automatically when dims are multiples of 8
    // Row-major trick: compute C^T = B^T @ A^T via swapped args
    [LibraryImport(LibName)]
    internal static partial int cublasHgemm(
        nint handle,
        int transa, int transb,     // cublasOperation_t: 0=N, 1=T, 2=C
        int m, int n, int k,
        in ushort alpha,            // __half passed as ushort
        nint A, int lda,
        nint B, int ldb,
        in ushort beta,
        nint C, int ldc);

    // GemmEx — mixed precision (FP16 input, FP32 accumulate)
    [LibraryImport(LibName)]
    internal static partial int cublasGemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        nint alpha,
        nint A, int Atype, int lda,
        nint B, int Btype, int ldb,
        nint beta,
        nint C, int Ctype, int ldc,
        int computeType, int algo);
    // cudaDataType: CUDA_R_16F=2, CUDA_R_32F=0
    // cublasComputeType: CUBLAS_COMPUTE_16F=64, CUBLAS_COMPUTE_32F=68
}
```

### Error Handling

```csharp
// src/DotLLM.Cuda/Interop/CudaException.cs
namespace DotLLM.Cuda.Interop;

public sealed class CudaException : Exception
{
    public int ErrorCode { get; }

    public CudaException(int errorCode, string message)
        : base($"CUDA error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }
}

// src/DotLLM.Cuda/Interop/CudaErrorHelper.cs
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

internal static class CudaErrorHelper
{
    internal static void ThrowOnError(this int result)
    {
        if (result == 0) return; // CUDA_SUCCESS

        string message = "Unknown CUDA error";
        CudaDriverApi.cuGetErrorString(result, out nint strPtr);
        if (strPtr != 0)
            message = Marshal.PtrToStringAnsi(strPtr) ?? message;

        throw new CudaException(result, message);
    }
}
```

---

## PTX Kernel Conventions

All CUDA kernels use `extern "C"` linkage to prevent C++ name mangling, enabling `cuModuleGetFunction` lookup by simple string name:

```cuda
// native/kernels/rmsnorm.cu
#include <cuda_fp16.h>

extern "C" __global__ void rmsnorm_f16(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int n,
    const float eps)
{
    // Standard warp-reduction RMS normalization
    // FP16 in/out, FP32 accumulation for numerical stability
    // One block per row, warp shuffle for reduction
}
```

Compiled to PTX:

```bash
nvcc -ptx -arch=compute_80 -o rmsnorm.ptx rmsnorm.cu
```

Or for multi-architecture support (Ampere + Ada + Hopper):

```bash
nvcc -ptx \
    --generate-code arch=compute_80,code=compute_80 \
    --generate-code arch=compute_89,code=compute_89 \
    --generate-code arch=compute_90,code=compute_90 \
    -o rmsnorm.ptx rmsnorm.cu
```

### Kernel Launch from C#

The CUDA Driver API passes kernel arguments as an array of pointers to the actual values:

```csharp
public void LaunchRmsNorm(
    nint input, nint weight, nint output,
    int hiddenSize, float eps,
    uint rows, nint stream)
{
    unsafe
    {
        nint inputArg = input;
        nint weightArg = weight;
        nint outputArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void*[] args = [&inputArg, &weightArg, &outputArg, &nArg, &epsArg];

        fixed (void** argsPtr = args)
        {
            CudaDriverApi.cuLaunchKernel(
                _rmsnormFunc,
                gridDimX: rows, gridDimY: 1, gridDimZ: 1,
                blockDimX: 256, blockDimY: 1, blockDimZ: 1,
                sharedMemBytes: 0,
                stream: stream,
                kernelParams: (nint)argsPtr,
                extra: 0).ThrowOnError();
        }
    }
}
```

### Module Loading

PTX files are loaded once at startup and cached for the lifetime of the process. The CUDA driver caches JIT-compiled SASS in `~/.nv/ComputeCache` across process restarts.

```csharp
// src/DotLLM.Cuda/CudaModule.cs
public sealed class CudaModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = new();

    public static CudaModule LoadFromFile(string ptxPath)
    {
        byte[] ptxBytes = File.ReadAllBytes(ptxPath);
        var module = new CudaModule();

        unsafe
        {
            fixed (byte* ptxPtr = ptxBytes)
            {
                // Null-terminated — PTX is text, File.ReadAllBytes is fine
                // if the file ends without null, append one
                CudaDriverApi.cuModuleLoadData(out module._module, (nint)ptxPtr)
                    .ThrowOnError();
            }
        }
        return module;
    }

    public nint GetFunction(string name)
    {
        if (!_functions.TryGetValue(name, out nint func))
        {
            CudaDriverApi.cuModuleGetFunction(out func, _module, name)
                .ThrowOnError();
            _functions[name] = func;
        }
        return func;
    }

    public void Dispose()
    {
        if (_module != 0)
        {
            CudaDriverApi.cuModuleUnload(_module);
            _module = 0;
        }
    }
}
```

---

## Build System

### Kernel Compilation

A simple shell script replaces the entire CMake build system:

```bash
#!/bin/bash
# native/build.sh — Compile all .cu kernels to PTX

CUDA_ARCHS="compute_61"  # Pascal and newer; PTX is forward-compatible
OUT_DIR="$(dirname "$0")/ptx"
mkdir -p "$OUT_DIR"

for cu_file in "$(dirname "$0")"/kernels/*.cu; do
    base=$(basename "$cu_file" .cu)

    GENCODE_FLAGS=""
    for arch in $CUDA_ARCHS; do
        GENCODE_FLAGS="$GENCODE_FLAGS --generate-code arch=$arch,code=$arch"
    done

    nvcc -ptx $GENCODE_FLAGS \
         --use_fast_math \
         -o "$OUT_DIR/$base.ptx" \
         "$cu_file"

    echo "  $base.cu → $base.ptx"
done
```

### .NET Integration

PTX files are included as content files in the project, copied to output directory:

```xml
<!-- src/DotLLM.Cuda/DotLLM.Cuda.csproj -->
<ItemGroup>
    <Content Include="..\..\native\ptx\*.ptx" Link="ptx\%(Filename)%(Extension)">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
</ItemGroup>
```

Or embedded as resources for single-file deployment:

```xml
<ItemGroup>
    <EmbeddedResource Include="..\..\native\ptx\*.ptx" Link="ptx\%(Filename)%(Extension)" />
</ItemGroup>
```

### NativeLibrary Resolution

.NET's `NativeLibrary` resolution handles platform differences automatically. For cases where library names differ across platforms, use a resolver:

```csharp
// src/DotLLM.Cuda/Interop/CudaLibraryResolver.cs
using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

internal static class CudaLibraryResolver
{
    internal static void Register()
    {
        NativeLibrary.SetDllImportResolver(
            typeof(CudaLibraryResolver).Assembly,
            ResolveCudaLibrary);
    }

    private static nint ResolveCudaLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "cuda")
        {
            // Linux: libcuda.so.1 (symlink from driver install)
            // Windows: nvcuda.dll (in System32)
            string osLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "nvcuda.dll"
                : "libcuda.so.1";

            if (NativeLibrary.TryLoad(osLib, out nint handle))
                return handle;
        }

        if (libraryName == "cublas")
        {
            // Linux: libcublas.so.XX (versioned, from CUDA Toolkit)
            // Windows: cublas64_XX.dll
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Try common versions: CUDA 12.x, 13.x
                foreach (var ver in new[] { "12", "11" })
                {
                    if (NativeLibrary.TryLoad($"cublas64_{ver}.dll", out nint h))
                        return h;
                }
            }
            else
            {
                if (NativeLibrary.TryLoad("libcublas.so", out nint h))
                    return h;
            }
        }

        return 0; // fall through to default resolution
    }
}
```

---

## PTX JIT Compilation Overhead

First-time PTX loading incurs ~100–500ms per module as the driver compiles PTX → SASS for the specific GPU. Mitigations:

1. **Driver kernel cache** (`~/.nv/ComputeCache`): enabled by default, persists compiled SASS across process restarts. Second launch is near-instant.
2. **Fatbin compilation**: pre-compile to SASS for target architectures alongside PTX fallback. Zero JIT overhead on matching GPUs:
   ```bash
   nvcc -fatbin \
       --generate-code arch=compute_80,code=sm_80 \
       --generate-code arch=compute_89,code=sm_89 \
       --generate-code arch=compute_90,code=sm_90 \
       --generate-code arch=compute_80,code=compute_80 \
       -o rmsnorm.fatbin rmsnorm.cu
   ```
   Load with `cuModuleLoadData` — same API, auto-selects SASS if available, falls back to PTX.
3. **Startup amortization**: all modules loaded during model initialization (which already takes 1–10s for GGUF mmap + weight upload). PTX JIT adds negligible overhead relative to weight transfer.

---

## Kernel Catalog

All kernels compiled to PTX, loaded via `cuModuleLoadData`, launched via `cuLaunchKernel`:

**FP16 pipeline (primary):**

| Kernel | File | Function Name | Block Size | Grid Size | Shared Mem |
|---|---|---|---|---|---|
| RMS Norm | `rmsnorm.cu` | `rmsnorm_f16` | 256 | rows | Warp reduction |
| Fused Add+RmsNorm | `fused_add_rmsnorm.cu` | `fused_add_rmsnorm_f16` | 256 | rows | Warp reduction |
| Per-Head RmsNorm | `per_head_rmsnorm.cu` | `per_head_rmsnorm_f16` | 256 | heads × seqLen | Warp reduction |
| RoPE | `rope.cu` | `rope_f16` | 256 | seqLen × numHeads | None |
| Attention | `attention.cu` | `attention_f16` | 256 | numHeads × seqQ | Per-head scores |
| SwiGLU | `swiglu.cu` | `swiglu_f16` | 256 | ceil(n/256) | None |
| Add | `add.cu` | `add_f16` | 256 | ceil(n/256) | None |
| Bias Add | `bias_add.cu` | `bias_add_f16` | 256 | ceil(n/256) | None |
| Softmax | `softmax.cu` | `softmax_f16` | 256 | rows | Warp reduction |
| Embedding (F32) | `embedding.cu` | `embedding_lookup_f32` | 256 | seqLen | None |
| Embedding (F16) | `embedding.cu` | `embedding_lookup_f16` | 256 | seqLen | None |
| Embedding (Q8_0) | `embedding.cu` | `embedding_lookup_q8_0` | 256 | seqLen | None |

**Dequantization (quantized weights → FP16 scratch):**

| Kernel | File | Function Name |
|---|---|---|
| Dequant Q8_0 | `dequant.cu` | `dequant_q8_0_f16` |
| Dequant Q4_0 | `dequant.cu` | `dequant_q4_0_f16` |
| Dequant Q5_0 | `dequant.cu` | `dequant_q5_0_f16` |
| Dequant Q4_K | `dequant.cu` | `dequant_q4_k_f16` |
| Dequant Q5_K | `dequant.cu` | `dequant_q5_k_f16` |
| Dequant Q6_K | `dequant.cu` | `dequant_q6_k_f16` |
| Dequant IQ4_NL / IQ4_XS | `dequant_iquants.cu` | `dequant_iq4_{nl,xs}_{f16,f32}` |
| Dequant IQ2_XXS / IQ2_XS / IQ2_S | `iq2.cu` | `dequant_iq2_{xxs,xs,s}_{f16,f32}` (IQ2_S also stores `MOSTLY_IQ2_M` file-type tensors) |

**Quantized GEMV (decode path — operate directly on quantized weights):**

| Kernel | File | Function Name | Notes |
|---|---|---|---|
| Q8_0 GEMV | `quantized_gemv.cu` | `quantized_gemv_q8_0` | FP fmuladd, legacy |
| Q4_K GEMV | `quantized_gemv.cu` | `quantized_gemv_q4_k` | FP fmuladd, legacy |
| Q5_K GEMV | `quantized_gemv.cu` | `quantized_gemv_q5_k` | FP fmuladd, legacy |
| Q6_K GEMV | `quantized_gemv.cu` | `quantized_gemv_q6_k` | FP fmuladd, legacy |
| Q4_K MMQ | `quantized_gemv_mmq.cu` | `quantized_gemv_q4_k_mmq` | dp4a, 4 rows/block; default for k<1024 |
| Q5_K MMQ | `quantized_gemv_mmq.cu` | `quantized_gemv_q5_k_mmq` | dp4a, 4 rows/block |
| Q6_K MMQ | `quantized_gemv_mmq.cu` | `quantized_gemv_q6_k_mmq` | dp4a, 4 rows/block; lifts LmHead in Q4_K_M GGUFs |
| Q{4,5,6}_K MMVQ-large | `quantized_gemv_mmq.cu` | `quantized_gemv_q*_k_mmvq_large` | dp4a, 1 row/block × 128 threads; default for k≥1024 (paired with `_preq` variants) |
| Q{4,5,6}_K MMQ/MMVQ `_preq` | `quantized_gemv_mmq.cu` | `*_preq` suffix | Read pre-quantized x from scratch; skips Stage 1 input quant |
| Pre-Q8_1 input quant | `quantize_x.cu` | `quantize_x_to_q8_1` | Quantizes activation x[k] once per fused-GEMV-input — feeds the `_preq` variants. Auto-engages for k≥1024 |
| IQ4_NL / IQ4_XS GEMV | `quantized_gemv.cu` | `quantized_gemv_iq4_{nl,xs}` | FP fmuladd, legacy |
| IQ2_XXS / IQ2_XS / IQ2_S GEMV | `iq2.cu` | `quantized_gemv_iq2_{xxs,xs,s}` | 2-bit codebook-lookup decode; one block per output row; warp-reduction accumulator. IQ2_S handles `MOSTLY_IQ2_M` GGUFs (Qwen3.6-A3B-IQ2_M ~11.5 GB). MMQ-preq / MMVQ-large / MoE-grouped variants deferred — prefill falls back to dequant→cuBLAS via `dequant_iq2_*_f32`. |

**Embedding lookup (per-row dequant, no full-table FP16):**

| Kernel | File | Function Name | Notes |
|---|---|---|---|
| F32 / F16 / Q8_0 | `embedding.cu` | `embedding_lookup_*` | Existing fast paths |
| Q4_K / Q5_K / Q6_K | `embedding.cu` | `embedding_lookup_q{4,5,6}_k_f16` | Per-row dequant, llama.cpp `get_rows_q*_K` pattern. Saves ~1.16 GiB on Qwen3-8B vocab×hidden |

**Conversion:**

| Kernel | File | Function Name |
|---|---|---|
| FP16→FP32 | `convert.cu` | `convert_f16_to_f32` |
| FP32→FP16 | `convert.cu` | `convert_f32_to_f16` |

FP32 diagnostic variants also exist (`*_f32.cu` files) for debugging precision issues.

GEMM/GEMV for prefill use cuBLAS (`cublasHgemm` / `cublasGemmEx`) directly — no custom PTX kernel needed.

---

## cuBLAS Row-Major Convention

cuBLAS uses column-major layout (Fortran convention). Since dotLLM tensors are row-major, we use the standard transposition trick:

To compute `C = A × B` (all row-major), we call `cublasHgemm` with swapped operands:
- Pass `B` as first matrix, `A` as second matrix
- Set `transa = CUBLAS_OP_N`, `transb = CUBLAS_OP_N`
- cuBLAS computes `C_colmajor = B_colmajor × A_colmajor`
- Because `X_colmajor` is equivalent to `X^T_rowmajor`, this yields the correct row-major result

This is well-proven — llama.cpp, vLLM, and every CUDA inference engine uses the same trick.

---

## Prerequisites

### Runtime Requirements

- **NVIDIA GPU**: Compute capability 6.1+ (Pascal and newer). Recommended: 7.0+ (Volta) for Tensor Core acceleration.
- **NVIDIA GPU driver**: 525.60+ (for CUDA 12.x compatibility).
- **CUDA Runtime**: not required — the Driver API (`libcuda.so`) is sufficient and ships with the driver.
- **cuBLAS**: required for GEMM. Installed with CUDA Toolkit or available as a standalone redistributable.

### Build Requirements (kernel compilation only)

- **CUDA Toolkit 12.x+**: provides `nvcc` compiler. Only needed to recompile `.cu` → `.ptx` files. Pre-compiled PTX files can be distributed without requiring the toolkit on end-user machines.
- **No CMake, no C/C++ compiler, no build system** beyond `nvcc`.

---

## Future Work

- **CUDA MoE FFN port**: top-k routing + per-expert grouped-GEMM on GPU. CPU has `MoeSwiGluMlp`; CUDA equivalent doesn't exist yet. **Concrete blocker for end-to-end DeepSeek-V2/V3 on CUDA** (MLA attention Phase 1 primitives landed; they need a matching MoE FFN to complete the layer forward).
- **MLA Phase B + C** (CUDA decode efficiency): latent KV cache + W_UK absorption (`MlaAttention.ExecuteLatent`/`ExecuteLatentHybrid` CPU equivalents). Phase A naive expanded forward is already in tree (`CudaMlaAttention.Forward`).
- **MLA FP16/quantized weight paths**: current Phase A is F32 throughout. FP16 follow-up; quantized extends `Project` patterns from the GQA path.
- **Flash Attention**: replace naive attention kernel with tiled flash attention (shared memory, online softmax). Full Tensor Core access via `wmma` intrinsics in PTX. **Elevated `ncu --set full` data (2026-07-27, RTX 3060, `attention_f32` decode launches, Bonsai-27B, grid=(24,1,1)/block=(256,1,1))**: Compute (SM) Throughput ~4.2-4.3%, Memory Throughput ~8.6-8.7%, Achieved Occupancy 16.5-16.8% vs. 100% Theoretical (ncu's own analysis flags ~83% Est. Speedup from occupancy alone), Waves Per SM only 0.14 (grid=24=`numHeads` badly underfills the 3060's 28 SMs), Warp Cycles Per Issued Instruction ~42.3-42.8 — i.e. the kernel is genuinely **latency-bound**, not compute- or memory-bandwidth-bound; both throughput metrics sit near-idle simultaneously. This is the same root cause #180/#182's "grid too small" fixes addressed elsewhere in the decode path (see the `prismml-bonsai-model` project memory), now confirmed quantitatively for attention specifically rather than inferred from category-level `DOTLLM_HYBRID_PROFILE` percentages. **This also explains, not just motivates, issue #183's inconclusive real-world result**: #183 shipped an opt-in split-KV ("Flash-Decoding") kernel that splits the KV-tile loop across more cooperating blocks specifically to raise occupancy — the real A/B (depth 256-1024) came back within noise (+0.5% to +2% best-of, not a clean win). Given this ncu data, that outcome makes sense: splitting into more blocks helps an occupancy-bound kernel, but a kernel this deep into the latency-bound regime (42 cycles stalled per issued instruction, both compute AND memory sitting under 9%) needs fewer, larger, better-pipelined memory transactions — i.e. an actual flash-attention rewrite (tiled shared-memory staging, online softmax, deeper software pipelining to hide the per-access latency) — not just more parallel blocks replaying the same latency-bound access pattern. Treat #183's split-KV kernel as a proven-safe but low-value stepping stone, not a substitute for this item.

  **2026-07-30 re-profile, post-#197/#198 (PR #201), at realistic depth — corrects the picture above.**
  The 2026-07-27 numbers were measured at shallow depth (`-p 8 -n 12`, no `--depth`, so `seq_kv` was
  effectively ~8-20) with no launch-skip verification of which kernel/phase was actually captured —
  worth flagging since a first re-profile attempt this session initially mis-captured **prefill**
  `attention_f32` launches (`seq_q=8`, grid=192) while believing it had captured decode (a live
  instance of exactly the "verify a flag/kernel is actually wired into the code path you're testing"
  lesson from the BitNet session's GQA-split false-correlation — see `[[bitnet-support]]`). Corrected
  methodology: `--kernel-name` filtering (not `--launch-skip` guessing) plus `--depth 512` (clears
  `AttentionGqaSplitMinSeqKv=256`, the opt-in gate `DOTLLM_ATTN_GQA_SPLIT=1` needs to even dispatch
  `attention_f32_gqa_split_kv` instead of falling through to the plain kernel) confirmed by grid-shape
  before trusting any number.

  | Metric | Baseline `attention_f32` (grid=24) | `attention_f32_gqa_split_kv` (grid=4×8, `DOTLLM_ATTN_GQA_SPLIT=1`) |
  |---|---:|---:|
  | Duration | 191.97 us | 180.48 us (**-6.0%**) |
  | Compute (SM) Throughput | 9.74% | 20.61% (**+112%**) |
  | Memory Throughput | 48.93% | 25.15% (**-48.6%**, i.e. real reduction in redundant KV traffic) |
  | Achieved Occupancy | 16.65% | 19.00% |
  | Theoretical Occupancy | 100% | 83.33% (grouped-warp kernel needs more registers/shared mem per block) |
  | Waves Per SM | 0.14 | 0.23 |
  | Warp Cycles Per Issued Instruction | 24.60 | 15.24 (**-38%**) |
  | Dominant stall reason | 75.7% L1TEX/global-memory scoreboard wait | 44.2% CTA-barrier wait (6.7 of 15.24 cycles) |

  Reading: #197/#198 (the GQA-group + tuned split-KV kernel) is real and working as designed — it
  substantially cuts redundant KV reads (compute throughput +112%, memory traffic -49%, per-instruction
  stall -38%) — but the wall-clock win per launch is modest (~6%) because occupancy barely moves
  (16.65%→19.00%) and the theoretical ceiling actually drops (100%→83.33%) from the grouped kernel's
  higher per-block resource cost. Critically, **the dominant stall reason changed category**: the
  plain kernel is bottlenecked on raw memory-load latency (waiting for K/V reads to land), the
  GQA-split kernel is now bottlenecked on **CTA-barrier synchronization** (warps waiting for siblings
  in the same block, "commonly caused by diverging code paths before a barrier" per `ncu`'s own
  analysis) — a different, more specific, more actionable diagnosis than the original "generically
  latency-bound" read. This means #199's own stated precondition ("implement after #198 lands and
  the kernel is out of the latency-bound regime") is only **partially** satisfied: #198 helped, but
  the kernel is still deep in an underutilized regime, just via a different mechanism. Worth
  considering a smaller, lower-risk fix first — reducing warp divergence/barrier count in
  `attention_f32_gqa_split_kv`'s grouped-warp code specifically — before committing to #199's full
  FP16 tensor-core rewrite (HIGH risk, precision-sensitive, per that issue's own scoping). Not yet
  investigated at the SASS/source level; the barrier-stall *hypothesis* itself should get the same
  "verify before trusting" treatment #218 gave a superficially similar `ncu` hypothesis (which turned
  out wrong) before either path is chosen. Raw `.ncu-rep` reports and this session's methodology notes
  are in `.perf-runs/ncu-2026-07-30-post197198/` (not committed — binary, ~250 MB combined).

  **Follow-up (issues #197+#198, same session): GQA-group register-blocking + tuned split-KV,
  composed into one kernel.** `attention_f32_gqa_split_kv` grids `(numKvHeads, kv_split)` instead
  of `(numHeads, ATTN_KV_SPLIT=4)` -- each block owns one KV head and register-blocks the QK/PV
  loops across the `group=numHeads/numKvHeads` query heads sharing it (Bonsai-27B: group=6), so
  each K/V element is read from global memory once per tile and reused `group` times, instead of
  `group` independent blocks each re-reading the same rows. `kv_split` is now a runtime parameter
  (an occupancy-target heuristic, `CudaKernels.ComputeAttentionKvSplit`, form ported from Vulkan's
  `ComputeSplits`/#347 but re-derived for CUDA's cooperative-launch co-residency ceiling), not
  #183's hardcoded 4. Correctness: bit-exact (0 ULP) per query head vs `attention_f32` at
  `kv_split==1` across 5 shapes including the real Bonsai-27B shape, validated directly (not
  assumed) -- see `CudaAttentionF32GqaSplitTests.cs`. Real occupancy query on this RTX 3060 for
  Bonsai-27B's shape: `MaxSafeAttentionGqaSplit(numKvHeads=4, headDim=256, group=6) = 35` (up to
  140 co-resident blocks vs today's grid=24). **Real A/B (dotnet bench, same host, 2 rounds/depth,
  MEDIAN not best-of): depth 256 -2.1%, depth 512 +0.9%, depth 768 +2.1% (consistent both rounds),
  depth 1024 +0.8%, depth 2048 +0.9%** -- directionally flat-to-positive at every depth except the
  shallowest (256, where per-split KV rows are too few to amortize the grid.sync+combine overhead,
  same shape as #183's own depth-256 result), but no depth clears this project's documented 2-8%
  run-to-run noise floor as a decisive win. An honest inconclusive result, matching #183's own
  precedent exactly -- shipped opt-in (`DOTLLM_ATTN_GQA_SPLIT=1`, default OFF), not because it's
  unsafe (correctness-validated, zero-risk to any default path) but because the real-world gain
  is not yet demonstrated beyond noise. `ncu` re-profiling to confirm the occupancy/waves-per-SM
  metrics actually moved could not be completed this session (`ERR_NVGPUCTRPERM` -- this host
  requires elevated/UAC PowerShell for `ncu`, unavailable to this non-interactive session); the
  register-level evidence available instead (`ptxas -v`: 40 registers, 0 spill loads/stores, same
  register count as the baseline `attention_f32` kernel) is consistent with the kernel being a
  clean regrid rather than something register-pressure-bound, but does not substitute for a real
  `ncu` capture. A future session with UAC access should re-run `ncu --set full` on this kernel
  before considering the occupancy claim independently confirmed.
  **Follow-up investigation (issues #199/#200/#219/#220/#222/#226/#227, 2026-07-27→28): tensor
  cores parked, a real kv_split bug found + fixed, and #183's precision disqualified it from
  default-on despite being the strongest performer found.**
  - **#199 (tensor-core decode kernel): not attempted, correctly blocked at the design stage.** A
    literal port of Vulkan's coopmat decode kernel would have regressed #197/#198's KV-head-grid
    design (Vulkan grids per query-head with padding; #197/#198 deliberately grids per KV-head to
    amortize reads across Bonsai's group=6 sibling heads — porting Vulkan's scheme would throw that
    away). Bonsai's `headDim=256` also doesn't fit the existing prefill mma kernel's `headDim=64`
    hardcoding, so this would be from-scratch design work, not a port. Correctly deferred pending
    confirmation that the kernel is still latency-bound after #197/#198 (see #219 below — it
    isn't, in the way assumed).
  - **#200 (native paged-KV decode kernel): not attempted, no producer exists.** CUDA has no paged
    KV-cache implementation at all — `PagedKvCache` is CPU-only; every CUDA path (`run`/`chat`/
    `serve`) that requests paging silently falls back to the native non-paged cache. Nothing for
    a "read paged blocks directly" kernel to attach to. Real prerequisite: a CUDA paged KV-cache
    implementation, worth building once continuous batching (already tracked) makes multi-sequence
    paged decode the common case.
  - **#219: found and fixed a real kv_split grid-sizing bug in #197/#198's own heuristic.**
    `ComputeAttentionKvSplit`'s `AttnSplitMinKvPerSplit` term (default 128) was the binding clamp
    at Bonsai's actual decode depth (seqKv~258-270), forcing `kv_split=3` (grid=12 — *half* the
    unsplit baseline's grid=24) instead of the intended `byOccupancy=8`. Fixed via #220 (merged):
    lowered to 32. `ncu`-confirmed: grid 12→32, duration 174-176us→144-146us (~17% faster),
    occupancy 16.5%→18.8%. Still ~37% slower than the unsplit baseline at this depth even fixed —
    stays opt-in/default-OFF. The "32" target is depth-specific, not universal (item 4 of #219: at
    depth 768, target=96 helps further, but even matched-grid the GQA-split design still trails
    #183's simpler kernel by ~20% — its register-blocking approach carries overhead grid size alone
    doesn't close).
  - **#219 also re-tested #183 (`DOTLLM_ATTN_SPLIT_KV`) at the depth range its original A/B never
    reached** (the depth>=768 hang above is gone on current `dev`). Result: the cleanest win in
    this whole investigation family — 194us vs 274-276us (~29% faster), occupancy 16.6%→57%
    (3.4x), Waves/SM 0.14→0.69. Structurally simpler (fixed `SPLIT=4`, no GQA register-blocking)
    than #197/#198/#220's design, and beats it even after that design is grid-size-tuned.
  - **#222: that win is precision-unsafe — #183 changes generated output for any context beyond
    256 tokens, not a rare edge case.** Real end-to-end generation-parity test (Bonsai-27B,
    deterministic sampling) found the first divergence at generated-step 225 (decode depth 257 —
    the earliest point the kernel can engage): a genuine argmax flip (baseline margin 0.076 vs
    split-KV margin 0.011) that fully compounds via the sampled-token feedback loop — 774/775
    subsequent tokens differ. Perplexity confirms independently: flat pre-gate (-0.02%, as expected
    since the kernel can't engage there), +0.30% post-gate. **Recommendation: #183 stays opt-in/
    default-OFF** — the fastest kernel found here is not a safe default despite the win being real.
  - **#226/#227: tried fp64 accumulation in the cross-split combine step to reduce the
    reassociation error — no improvement, clean negative result.** Root cause is *not* `fast_exp_neg`
    (this file's header already documents why precise `expf` was rejected — CPU/GPU parity, both
    the baseline and split-KV kernels already use the same approximation identically) and *not* the
    final 4-way merge arithmetic: the double-precision combine variant (`attention_f32_split_kv_hp`,
    opt-in `DOTLLM_ATTN_SPLIT_KV_HP=1`, PR #227 open/not merged) diverges at the *identical* step
    225 with an essentially identical margin and perplexity delta. The reassociation error is
    already baked into each split's own independent partial accumulation (computed in float, over a
    different KV sub-range/order than the baseline's single pass) *before* the combine runs — fixing
    that would need higher precision in the per-split accumulation itself, a materially bigger
    change than scoped, and an open question rather than a next task.
  - **Net for this line of work**: three kernel variants now exist for decode attention
    (`attention_f32` baseline, `attention_f32_split_kv` #183, `attention_f32_gqa_split_kv` #197/
    #198/#220), all correctness-validated and zero-default-risk, but only the baseline is safe as
    default — #183 is faster but changes output; the GQA-split design is neither faster nor safer.
    A tensor-core rewrite (#199) remains the only lever not yet tried that could plausibly beat
    #183's raw speed *and* stay numerically closer to baseline (different algorithm shape, not a
    reassociated split of the same one) — worth reconsidering once continuous batching or another
    driver makes the investment timing right.

  **#230: checked whether the GQA-split kernel's CTA-barrier stall (44.2% of 15.24 cycles, per the
  2026-07-30 re-profile above) has a cheaper fix than #199's rewrite — SASS-verified per-barrier
  diagnosis, one genuine barrier removed, honest negative result on wall-clock impact.**
  Per this project's #218 precedent (an `ncu` barrier-stall hypothesis that direct SASS inspection
  proved wrong), the stall label was not trusted at face value. Compiled with
  `nvcc -lineinfo` + `ptxas -lineinfo -arch=sm_86` + `nvdisasm -g` (no elevation needed) to get an
  exact PC→source-line map for every `BAR.SYNC` in `attention_f32_gqa_split_kv`: 16 static sites
  across 13 source-level `__syncthreads()` call sites, of which 6 belong to the per-head max/sum
  reduction loop (attention_f32.cu lines 827-878) that the kernel's own header already flags as
  "the one place this design does not parallelize across the group" — run sequentially `group`=6
  times per KV tile for Bonsai-27B, i.e. this one loop accounts for the large majority of the
  kernel's *dynamic* barrier count. Dependency analysis of what each of those 6 barriers actually
  protects found one (the loop-tail sync, guarding `warp_scratch` reuse across the next head's
  max-phase write) to be fully redundant: the WAR hazard it exists for is already closed
  transitively by the loop's own real cross-warp barriers (the ones a few lines into the next
  iteration, which every thread must reach — and reaching them requires every thread to have
  already finished the current iteration, scratch-buffer read included). Removed it with a
  same-file comment documenting the argument. SASS confirms a clean, minimal, single-barrier
  removal (`BAR.SYNC` count 16→15; PTX diff is exactly one deleted `bar.sync` instruction) at
  **zero register/shared-memory cost** (`ptxas`: `REG:40` unchanged, so the 83.33% theoretical
  occupancy ceiling is untouched) — the lowest-risk category of change this investigation could
  make. The other 4 of the 6 per-head barriers are real cross-warp producer/consumer dependencies
  (broadcasting a warp-0-computed cross-warp max/sum to the rest of the block) inherent to the
  shuffle-tree reduction algorithm shared verbatim with `attention_f32`/`attention_f32_split_kv`,
  and removing them would mean changing that reduction's structure — which would break the
  kv_split==1 bit-exactness contract this kernel is tested against, a materially bigger and riskier
  change than scoped here.
  **Correctness**: all 21 `CudaAttentionF32GqaSplitTests` pass post-fix, including bit-exact
  (0 ULP) at every tested group size (1, 4, 6, 8, including the MAX_GQA_GROUP boundary) and the
  300-consecutive-decode-step drift characterization; both `CudaAttentionSplitKvGenerationParityTests`
  (real Bonsai-27B generation + perplexity, exercising the same compiled PTX module) pass unchanged.
  Ran the GQA-split unit-test suite 3x (once concurrently with itself) with no flakiness, the kind
  of check a subtle removed-barrier race would likely surface.
  **Wall-clock**: real `dotnet bench` A/B on this RTX 3060, interleaved pre-fix/post-fix rounds (not
  blocked sequentially, after an initial blocked run showed a same-direction decline in BOTH
  configurations — a session-level thermal/clock effect, not a kernel difference; interleaving and
  shortening `-n` from 48→24 per rep controls for it), `DOTLLM_ATTN_GQA_SPLIT=1`, real Bonsai-27B:
  at **depth 512** (matching the re-profile's own depth), 3 interleaved rounds, median decode tok/s
  pre-fix {17.48, 17.30, 17.36} vs post-fix {17.47, 17.37, 17.30} — mean 17.38 vs 17.38, a dead
  tie. At **depth 1024**, 2 interleaved rounds, pre-fix {16.80, 16.37} vs post-fix {15.73, 16.05} —
  post-fix trends ~2-4% lower by median (though best-of-rep is within ~1%), still inside this
  project's documented 2-8% run-to-run noise floor and with isolated outlier reps (a single rep
  dropping to ~14-15 tok/s with no GPU contention visible in `nvidia-smi`) present at both configs.
  **Reading**: the fix is real (verified at the SASS/instruction level, not just asserted) and free
  (no register/occupancy cost, no precision cost), but its wall-clock effect is not distinguishable
  from noise at either depth tested. This is consistent, not surprising, given `attention_f32*`'s
  own scoping note elsewhere in this file that attention is only ~3-10% of total decode-step time
  even at depth 256-1024 — removing 1 of 6 barriers in one loop of one kernel that is itself a
  small slice of the token budget was never likely to clear this host's noise floor, and the other
  5 barriers are not cheaply removable without a materially bigger, riskier change. **Recommendation
  for #199**: this specific "cheap fix" avenue is exhausted — kept as a real, zero-risk,
  correctness-preserving cleanup (worth keeping in the opt-in kernel regardless), but it does not
  answer #199's precondition question in the affirmative. #199's tensor-core rewrite (or accepting
  the GQA-split kernel's current form as a non-default, marginal-value opt-in alongside #183)
  remain the only levers left that could plausibly move the needle further on this kernel.
- **BitNet decode CUDA-graph capture, and the generic `attention_f16_dyn` slowdown behind it** (issues #212/#213/#218/#221, PRs #214/#217/#223/#224, 2026-07-28): BitNet was the only supported architecture excluded from the project's default-on decode CUDA-graph capture (the generic captured body omitted BitNet's FP32-residual/Sub-LN/ReLU² ops). #214 ported those ops in and removed the exclusion — bit-exact vs eager, +9-11% decode at shallow depth on both real BitNet models (2B-4T, `bitnet_b1_58-xl`). This surfaced a real depth-dependent regression, fixed for BitNet specifically via #217's depth ceiling (`BitNetGraphCaptureMaxDepth`, default 384).

  **#218 then found and fixed the underlying kernel-level cause, and generalized the mitigation to every architecture.** An elevated `ncu --set full` capture (`.perf-runs/ncu-2026-07-28/README.md`) first suggested a CTA-barrier-stall hypothesis (the `seq_kv`/`position_offset` device-pointer reads landing too close to a sync point) — **this was refuted** by direct SASS inspection (`ptxas -arch=sm_86` + `cuobjdump --dump-sass`, no elevation needed): `ptxas` already schedules both loads as the first two real instructions, with 50-90 independent instructions before first use. The actual cause: all 256 threads in `attention_f16_dyn` each independently re-read the same block-uniform pointer values — 8x the redundant memory-latency exposure `attention_f16` doesn't pay (it reads from the near-free constant/parameter bank instead). Fixed by templating the shared kernel body on a `DeviceIndirect` compile-time bool; the `_dyn` instantiation has only thread 0 dereference the pointers once, broadcasting via the dead tail of the existing `warp_scratch[32]` shared buffer — no new shared memory, no new barrier, and `attention_f16`'s SASS is byte-for-byte unaffected (verified via SASS diff). This closed roughly a third of the regression on its own; the rest was closed by generalizing #217's pattern into a new `GraphCaptureMaxDepth` (default 512, `DOTLLM_GRAPH_MAX_DEPTH` override) covering every graph-capable architecture, with BitNet keeping its own tighter, separately-validated ceiling. Falcon-E-3B/Falcon3-3B regression fully closed at every depth ≥512 tested; shallow-depth graph win preserved (+2.8% to +7.9%).

  **#221 separately confirmed** (via a real regression test plus `nsys --cuda-graph-trace=node` kernel-launch traces, not just code reading) that the I2_S QKV/GateUp fused-GEMV decode dispatch (`CanFuseI2SDecode`) already engages identically for Llama-arch I2_S models (Falcon-E-3B, Falcon3-3B) and BitNet-arch — confirmed working as designed, no fix needed there. (A separate, orthogonal, non-performance-relevant finding from the same investigation: the load-time QKV VRAM buffer-packing, `TryUploadPackedThree`, never engages for any I2_S model since I2_S has its own dedicated kernel family rather than the generic quantized-GEMV path that buffer feeds — documented so it isn't mistaken for a regression later.)
- **Fused quantized GEMM for prefill**: Marlin-style dequant-in-register. Decode is now MMQ + MMVQ-large + pre-Q8_1 (Qwen3-8B Q4_K_M decode hits 33 tok/s eager on RTX 3060 — inside llama.cpp's reported range); prefill still uses dequant→cuBLAS HGEMM.
- **Continuous batching scheduler** (engine-layer prerequisite for tensor-core mma kernel value — see `docs/perf/MMA_BATCHED_MMQ.md` for the design analysis).
- **Tensor-core (mma) batched MMQ**: only valuable once batched decode is the call shape. See `docs/perf/MMA_BATCHED_MMQ.md` for thresholds.
- **Multi-stream pipelining** (Step 32): overlap H2D transfer with compute across layers.
- **NCCL integration** (Step 51): multi-GPU tensor parallelism. NCCL is another system library — same P/Invoke pattern, no shared library needed.
- **Fatbin distribution**: ship pre-compiled SASS for common architectures to eliminate JIT overhead.
- **NVRTC runtime compilation**: compile `.cu` source to PTX at application startup using NVIDIA's Runtime Compilation library, eliminating the nvcc build step entirely. NVRTC is available as `libnvrtc.so` / `nvrtc64_*.dll`.
