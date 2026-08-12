# NemotronH CUDA Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give NemotronH (`Architecture.NemotronH`, GGUF `general.architecture = nemotron_h`) a real CUDA host — `CudaNemotronHTransformerModel` — wired into `CudaModelLoader.CreateFromGguf`, replacing the interim `NotSupportedException` guard, with a dedicated Mamba2 selective-scan CUDA kernel and real-model CPU-vs-CUDA parity coverage (prefill + decode).

**Architecture:** Mirror the already-shipped `VulkanNemotronHTransformerModel` structurally (same per-layer dispatch on `HybridLayerLayout.LayerKind`, same 12-step Mamba2 SSM sub-layer sequence, same sparse KV-cache-by-attention-slot design) but implemented with CUDA driver-API kernel launches instead of Vulkan compute-shader dispatch, and with F32 activations throughout (matching every other CUDA hybrid host in this codebase). The CUDA loader reuses the CPU `NemotronHTransformerModel.LoadFromGguf` for all GGUF tensor-name resolution and per-layer shape validation (reflection-extracting its already-loaded `NemotronHLayerWeights[]`, exactly the pattern `VulkanNemotronHTransformerModel.BuildFromGguf` already uses) rather than re-deriving NemotronH's tensor-naming conventions a third time. Three small new CUDA kernels are required: a bit-order-faithful Mamba2 selective-scan kernel (`mamba2_selective_scan_f32`), a per-group RMSNorm kernel (`group_rmsnorm_f32`), and a plain elementwise squared-ReLU kernel (`relu_squared_inplace_f32`). The existing generic `conv1d_causal_f32` CUDA kernel (already used by Qwen3MoeHybrid/Qwen3HybridDense's GDN conv step) is reused as-is — its channel/kernel-width parameterization already covers NemotronH's Mamba2 conv1d shape with no changes.

**Tech Stack:** C# / .NET 10, CUDA Driver API (`cuLaunchKernel`, no runtime API), PTX kernels compiled via `native/build_ptx.bat`, cuBLAS (`cublasGemmEx`) for dense F32/F16 GEMM, xUnit + `SkippableFact`/`SkippableTheory` for tests.

## Global Constraints

- **Do NOT share the SSM-scan kernel with Mamba3 (issue #346).** This session's own research (independently re-verified against `Mamba2SelectiveScan.Execute`, the CPU reference read in full for this plan) confirms NemotronH's Mamba2 recurrence and Mamba3's SSD/trapezoidal recurrence are mathematically different families: Mamba2 has no RoPE on B/C, no trapezoidal lookahead, and applies the D-skip and z-gate as separate post-scan ops; Mamba3 rotates B/C through RoPE before the recurrence and fuses a pre-RoPE skip-dot + gate into the recurrence itself. Vulkan already ships them as two separate kernel families (`Mamba2SelectiveScanF32Kernel` vs. whatever Mamba3 lands). Do not attempt to generalize `mamba2_selective_scan_f32` into a shared kernel — write it as its own, NemotronH/Mamba2-specific implementation.
- **F32 activations throughout** — matches `CudaQwen3HybridDenseTransformerModel`/CPU/Vulkan; no F16 activation staging except where an existing generic helper (the `Gemm` dispatcher's prefill dequant-to-F16-then-cuBLAS-HGEMM path) already does it for weights lacking a native F32 kernel.
- **GGUF-only.** NemotronH has no CUDA safetensors path (mirrors the existing CPU/Vulkan restriction — `ModelLoader.LoadFromSafetensors` does not enumerate `NemotronH` either). Do not touch `CudaModelLoader.LoadFromSafetensors`.
- **Bit-order fidelity is required only for the SSM recurrence kernel** (`mamba2_selective_scan_f32`) and its fused softplus/decay/D-skip math — mirror `Mamba2SelectiveScan.Execute`'s per-thread sequential accumulation order exactly, including its **guarded** softplus (`x>20→x`, `x<-20→exp(x)`, else `log(1+exp(x))` — NOT the unguarded form `gdn_decay_f32` uses; that kernel's CPU oracle has no guard, `Mamba2SelectiveScan.SoftPlus` does). RMSNorm-family kernels (`group_rmsnorm_f32`) use warp-shuffle tree reduction like the existing `per_head_rmsnorm_f32` — tolerance-based parity, not bit-exact, consistent with every other RMSNorm kernel in this codebase.
- **Shared-file hotspots** — `CudaKernels.cs`, `CudaModelLoader.cs`, `native/build_ptx.bat`, and `CudaUnsupportedArchitectureGuardTests.cs` are touched by this plan's sibling tickets in the same batch (#346/#348/#349). Every edit to these files in this plan is additive (new function-pointer fields, new `case` arms, one new `NO_FMA` list entry, one removed `InlineData`) — never restructure or reformat surrounding code in these files.
- **Real-model fixture:** the env var is `DOTLLM_NEMOTRON_H_GGUF` (confirmed from `tests/DotLLM.Tests.Integration/Engine/NemotronHTextGeneratorTests.cs` — NOT a `TestFixtureResolver`-style HF org/repo/filename triple, which does not exist for this model). Follow that file's simple `Environment.GetEnvironmentVariable` + `File.Exists` + `Skip.If` pattern, not `TestFixtureResolver.ResolveFile`.
- Every new/modified C# file follows CLAUDE.md conventions: file-scoped namespaces, `<Nullable>enable</Nullable>`, `NativeMemory`/`cuMemAlloc`-only for tensor-adjacent buffers (no managed arrays on the hot path), XML doc comments on public/internal API surfaces that other tasks depend on.

---

### Task 1: Mamba2 selective-scan CUDA kernel + launcher + unit test

**Files:**
- Create: `native/kernels/mamba2_selective_scan.cu`
- Modify: `native/build_ptx.bat` (append `mamba2_selective_scan` to the `NO_FMA` list)
- Modify: `src/DotLLM.Cuda/CudaKernels.cs` (module load + `LaunchMamba2SelectiveScanF32`)
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaMamba2SelectiveScanF32Tests.cs`

**Interfaces:**
- Produces: `CudaKernels.LaunchMamba2SelectiveScanF32(nint state, nint x, nint dtRaw, nint dtBias, nint a, nint d, nint b, nint c, nint y, int nHead, int headDim, int dState, int nGroup, int seqLen, nint stream)` — consumed by Task 9 (`ForwardSsmBody`).
- Consumes: nothing from other tasks (pure kernel + launcher + test).

**Reference (read in full for this plan):** `src/DotLLM.Cpu/Kernels/Mamba2SelectiveScan.cs` (CPU oracle), `native/kernels/gated_delta_net_scan.cu`'s `gdn_scan_step_f32` (structural template for a per-thread-owns-a-state-row recurrence kernel with shared-memory staging), `src/DotLLM.Cuda/CudaKernels.cs`'s `LaunchGdnScanStepF32` (launcher template).

Unlike `gdn_scan_step_f32` (one launch per token, host loops over `seqLen`), this kernel takes ONE launch per SSM-layer-forward call and loops over `seqLen` internally — matching how Vulkan's `Mamba2SelectiveScanF32Kernel.Record` is dispatched once per SSM layer in `VulkanNemotronHTransformerModel.RecordSsmLayer` (`_mamba2Scan.Record(cmdBuf, ssmStateBuf, _state.SsmX, _state.DtBuf, ssmW.A, _state.SsmB, _state.SsmC, _state.SsmY, nHead, headDim, dState, nGroup, seqLen)`). This kernel additionally fuses two things the CPU/Vulkan implementations keep separate, both verified as safe fusions because they add no cross-thread accumulation-order ambiguity:
1. The `dt = dtRaw + dtBias` bias-add and guarded softplus (mirrors the *shape* of `gdn_decay_f32`'s bias+softplus fusion, but with `Mamba2SelectiveScan.SoftPlus`'s guarded body — see Global Constraints).
2. The D-skip term `y += x * D[h]` (CPU step 7, `ForwardSsmBody`'s block starting `// 7. y += x * D[h]`) — a per-element op with no reduction, safe to fuse into the same per-thread final store.

- [ ] **Step 1: Write the new kernel file**

```cuda
// native/kernels/mamba2_selective_scan.cu
//
// Mamba2 selective state-space scan (NVIDIA Nemotron-H). Bit-order-faithful port of
// DotLLM.Cpu.Kernels.Mamba2SelectiveScan.Execute, fused with:
//   - the raw-dt + dt_bias add and GUARDED softplus (Mamba2SelectiveScan.SoftPlus's exact
//     three-branch form: x>20 -> x; x<-20 -> exp(x); else log(1+exp(x))) — NOT the unguarded
//     form gdn_decay_f32 uses (that kernel's own CPU oracle has no guard; this one's does).
//   - the D-skip term (CPU step 7 of NemotronHTransformerModel.ForwardSsmBody: y += x*D[h]),
//     a per-element op safe to fuse since it adds no cross-thread reduction.
//
// Do NOT generalize this into a shared kernel with any future Mamba3/SSD scan (issue #346) —
// Mamba3's recurrence rotates B/C through RoPE and fuses a pre-RoPE skip-dot + gate that this
// Mamba2 recurrence does not have. Confirmed independently by reading both CPU references.
//
// ── Layouts (matches the CPU reference exactly) ─────────────────────────────
//   state  : [n_head, head_dim, d_state]  row-major, in/out
//   x      : [seq_len, d_inner]           row-major (d_inner = n_head*head_dim)
//   dt_raw : [seq_len, n_head]            row-major, NOT yet bias-added
//   dt_bias: [n_head]
//   a      : [n_head]                     scalar A per head (GGUF stores it negative)
//   d      : [n_head]                     scalar D per head (skip connection)
//   b, c   : [seq_len, n_group, d_state]  row-major
//   y      : [seq_len, d_inner]           row-major, out (already includes the D-skip term)
//
// ── Parallelization ──────────────────────────────────────────────────────────
// One block per head (gridDim.x = n_head). One thread per head-channel
// (blockDim.x = head_dim; host launch requires head_dim <= 256, this kernel is
// __launch_bounds__(256)). Each thread i owns state row state[h, i, 0..d_state) and walks
// t = 0..seq_len-1 sequentially (state depends on t-1), and for each t walks
// k = 0..d_state-1 sequentially in-register — the SAME nesting and accumulation order as the
// CPU's `for t / for h / for i / for k` loop nest (h is the block, i is the thread, t and k are
// the two sequential loops each thread runs). B/C are shared across every head in a group, so
// each block (== one head) stages this token's b/c group slice into shared memory once per t,
// exactly like gdn_scan_step_f32 stages k_shared/q_shared once per call.
//
// group index: g = h / (n_head / n_group) (heads_per_group heads share one B/C group).

extern "C" __global__ void __launch_bounds__(256) mamba2_selective_scan_f32(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ dt_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    const float* __restrict__ d,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ y,
    const int n_head, const int head_dim, const int d_state, const int n_group, const int seq_len)
{
    int h = blockIdx.x;
    if (h >= n_head) return;
    int i = threadIdx.x; // this thread's channel within the head; valid range [0, head_dim)

    int d_inner = n_head * head_dim;
    int heads_per_group = n_head / n_group;
    int g = h / heads_per_group;

    extern __shared__ float smem[];
    float* b_shared = smem;             // [d_state]
    float* c_shared = smem + d_state;   // [d_state]

    float a_h = a[h];
    float d_h = d[h];
    float dtb_h = dt_bias[h];

    // state row for this (h, i) — only meaningful while i < head_dim, which host launch
    // guarantees (blockDim.x == head_dim exactly).
    float* state_row = state + ((size_t)h * head_dim + i) * (size_t)d_state;

    for (int t = 0; t < seq_len; t++)
    {
        const float* b_row = b + ((size_t)t * n_group + g) * (size_t)d_state;
        const float* c_row = c + ((size_t)t * n_group + g) * (size_t)d_state;
        for (int k = threadIdx.x; k < d_state; k += blockDim.x)
        {
            b_shared[k] = b_row[k];
            c_shared[k] = c_row[k];
        }
        __syncthreads();

        // dt = dt_raw + dt_bias, then GUARDED softplus (Mamba2SelectiveScan.SoftPlus exactly).
        float dt_val = dt_raw[(size_t)t * n_head + h] + dtb_h;
        float dt_sp;
        if (dt_val > 20.0f)      dt_sp = dt_val;
        else if (dt_val < -20.0f) dt_sp = expf(dt_val);
        else                      dt_sp = logf(1.0f + expf(dt_val));

        // A is stored negative by the GGUF converter; exp(dt_sp * a_h) is in (0,1) -> decay.
        float dA = expf(dt_sp * a_h);

        float x_val = x[(size_t)t * d_inner + (size_t)h * head_dim + i];
        float x_dt = x_val * dt_sp;

        float sumf = 0.0f;
        for (int k = 0; k < d_state; k++)
        {
            float s = state_row[k] * dA + b_shared[k] * x_dt;
            state_row[k] = s;
            sumf += s * c_shared[k];
        }

        // D-skip fused in (CPU step 7): elementwise, no reduction, safe to fuse.
        y[(size_t)t * d_inner + (size_t)h * head_dim + i] = sumf + x_val * d_h;

        __syncthreads(); // before next t's shared-memory overwrite
    }
}
```

- [ ] **Step 2: Register the kernel for bit-parity compilation**

In `native/build_ptx.bat`, find the `NO_FMA` line (currently `set "NO_FMA=conv1d_causal gated_delta_net_scan elementwise_f32 turboquant"`) and append this kernel's base filename:

```bat
set "NO_FMA=conv1d_causal gated_delta_net_scan elementwise_f32 turboquant mamba2_selective_scan"
```

This kernel needs no `CXX17`/cooperative-groups flag (no `grid.sync()` — single-block-per-head, `__syncthreads()` only) and no `ARCH_86` override (no `mma.sync`/`cp.async`).

- [ ] **Step 3: Wire the loader + launcher into `CudaKernels.cs`**

Find the module-loading region (near where `conv1dCausalF32Path`/`_conv1dCausalF32Module` are loaded, around line 920-935) and add a sibling module load, following the exact pattern used for `_gdnScanF32Module`/`_gdnScanStepF32Func`:

```csharp
string mamba2ScanF32Path = Path.Combine(ptxDir, "mamba2_selective_scan.ptx");
if (File.Exists(mamba2ScanF32Path))
{
    _mamba2ScanF32Module = CudaModule.LoadFromFile(mamba2ScanF32Path);
    _mamba2ScanF32Func = _mamba2ScanF32Module.TryGetFunction("mamba2_selective_scan_f32");
}
```

Add the two backing fields alongside the other `_*Module`/`_*Func` fields (near `_gdnScanF32Module`/`_gdnScanStepF32Func`):

```csharp
private CudaModule? _mamba2ScanF32Module;
private nint _mamba2ScanF32Func;
```

Add the launcher near `LaunchGdnScanStepF32` (~line 3019), following its exact parameter-marshalling and shared-memory-sizing pattern:

```csharp
/// <summary>
/// Mamba2 selective-scan (NVIDIA Nemotron-H), one launch per SSM-layer forward call —
/// bit-order-faithful port of <see cref="DotLLM.Cpu.Kernels.Mamba2SelectiveScan.Execute"/>,
/// fused with the raw-dt + dtBias guarded-softplus decay and the D-skip term. See
/// native/kernels/mamba2_selective_scan.cu for the full layout and fusion documentation.
/// </summary>
/// <param name="state">Device pointer, <c>[nHead, headDim, dState]</c> F32, updated in place.</param>
/// <param name="x">Device pointer, <c>[seqLen, dInner]</c> F32 (dInner = nHead*headDim).</param>
/// <param name="dtRaw">Device pointer, <c>[seqLen, nHead]</c> F32, NOT yet bias-added.</param>
/// <param name="dtBias">Device pointer, <c>[nHead]</c> F32.</param>
/// <param name="a">Device pointer, <c>[nHead]</c> F32 (stored negative by the GGUF converter).</param>
/// <param name="d">Device pointer, <c>[nHead]</c> F32 (D skip parameter).</param>
/// <param name="b">Device pointer, <c>[seqLen, nGroup, dState]</c> F32.</param>
/// <param name="c">Device pointer, <c>[seqLen, nGroup, dState]</c> F32.</param>
/// <param name="y">Device pointer, <c>[seqLen, dInner]</c> F32, overwritten (includes D-skip).</param>
/// <param name="nHead">Number of Mamba2 heads.</param>
/// <param name="headDim">Channels per head (dInner / nHead). Must be in (0, 256] — the kernel
/// is compiled with <c>__launch_bounds__(256)</c> and launches with blockDim.x == headDim.</param>
/// <param name="dState">SSM state width.</param>
/// <param name="nGroup">Number of B/C groups (must divide nHead evenly).</param>
/// <param name="seqLen">Number of tokens in this call.</param>
/// <param name="stream">CUDA stream handle.</param>
/// <exception cref="ArgumentOutOfRangeException">headDim outside (0, 256].</exception>
public void LaunchMamba2SelectiveScanF32(nint state, nint x, nint dtRaw, nint dtBias,
                                           nint a, nint d, nint b, nint c, nint y,
                                           int nHead, int headDim, int dState, int nGroup,
                                           int seqLen, nint stream)
{
    if (headDim <= 0 || headDim > 256)
        throw new ArgumentOutOfRangeException(nameof(headDim),
            $"headDim={headDim}; mamba2_selective_scan_f32 is compiled with __launch_bounds__(256) " +
            "and launches with blockDim.x == headDim.");

    nint sArg = state, xArg = x, dtArg = dtRaw, dtbArg = dtBias;
    nint aArg = a, dArg = d, bArg = b, cArg = c, yArg = y;
    int nhArg = nHead, hdArg = headDim, dsArg = dState, ngArg = nGroup, slArg = seqLen;

    void** args = stackalloc void*[] {&sArg, &xArg, &dtArg, &dtbArg,
                    &aArg, &dArg, &bArg, &cArg, &yArg,
                    &nhArg, &hdArg, &dsArg, &ngArg, &slArg};

    // Shared memory: b_shared[dState] + c_shared[dState]
    uint sharedBytes = (uint)(2 * dState * sizeof(float));

    CudaDriverApi.cuLaunchKernel(_mamba2ScanF32Func,
            (uint)nHead, 1, 1, (uint)headDim, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
}
```

- [ ] **Step 4: Write the failing unit test**

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaMamba2SelectiveScanF32Tests.cs
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba2SelectiveScanF32"/>
/// (native/kernels/mamba2_selective_scan.cu) against its CPU oracle,
/// <see cref="Mamba2SelectiveScan.Execute"/>. Mirrors
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanMamba2SelectiveScanF32KernelTests"/>'s shapes and
/// tolerance (abs 1e-3 / rel 1e-3 — softplus + exp + the inner k-loop recurrence accumulate F32
/// rounding faster than pointwise kernels; the per-thread loop order matches the CPU's, but
/// exp/log emission can shift the last bits across iterations) and
/// <see cref="CudaGdnScanStepF32Tests"/>'s CUDA device-buffer idiom.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMamba2SelectiveScanF32Tests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _out;
    public CudaMamba2SelectiveScanF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    [SkippableTheory]
    [InlineData(2, 4, 8, 1, 1)]        // smallest decode shape
    [InlineData(4, 8, 16, 2, 1)]       // groups, single token
    [InlineData(4, 8, 16, 2, 4)]       // groups, multi-token prefill
    [InlineData(10, 80, 128, 10, 1)]   // Nemotron-H-realistic decode
    [InlineData(10, 80, 128, 10, 8)]   // Nemotron-H-realistic prefill
    public void Launch_MatchesCpuReference(int nHead, int headDim, int dState, int nGroup, int seqLen)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0x4A31 ^ (nHead * 131) ^ (headDim * 71) ^ (dState * 53) ^ (nGroup * 23) ^ seqLen);
        int dInner = nHead * headDim;

        float[] state0 = SmallRandom(rng, nHead * headDim * dState);
        float[] x = SmallRandom(rng, seqLen * dInner);
        float[] dtRaw = SmallRandom(rng, seqLen * nHead);
        float[] dtBias = SmallRandom(rng, nHead);
        float[] a = NegativeRandom(rng, nHead);
        float[] d = SmallRandom(rng, nHead);
        float[] b = SmallRandom(rng, seqLen * nGroup * dState);
        float[] c = SmallRandom(rng, seqLen * nGroup * dState);

        // CPU reference: pre-add dtBias (the CUDA kernel does this internally), guarded softplus
        // is inside Mamba2SelectiveScan itself; add D-skip manually after, matching
        // NemotronHTransformerModel.ForwardSsmBody steps 6-7 exactly.
        float[] dtBiased = new float[seqLen * nHead];
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < nHead; h++)
                dtBiased[t * nHead + h] = dtRaw[t * nHead + h] + dtBias[h];

        float[] stateCpu = (float[])state0.Clone();
        float[] yCpu = new float[seqLen * dInner];
        Mamba2SelectiveScan.Execute(stateCpu, x, dtBiased, a, b, c, yCpu, nHead, headDim, dState, nGroup, seqLen);
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < nHead; h++)
                for (int i = 0; i < headDim; i++)
                    yCpu[t * dInner + h * headDim + i] += x[t * dInner + h * headDim + i] * d[h];

        nint dState_ = 0, dX = 0, dDt = 0, dDtb = 0, dA = 0, dD = 0, dB = 0, dC = 0, dY = 0;
        try
        {
            long stateBytes = (long)state0.Length * sizeof(float);
            long xBytes = (long)x.Length * sizeof(float);
            long dtBytes = (long)dtRaw.Length * sizeof(float);
            long headBytes = (long)nHead * sizeof(float);
            long bcBytes = (long)b.Length * sizeof(float);
            long yBytes = (long)yCpu.Length * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)dtBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDtb, (nuint)headBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)headBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)headBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dC, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)yBytes).ThrowOnError();

            unsafe
            {
                float[] state0Copy = (float[])state0.Clone();
                fixed (float* p = state0Copy) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)xBytes).ThrowOnError();
                fixed (float* p = dtRaw) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)dtBytes).ThrowOnError();
                fixed (float* p = dtBias) CudaDriverApi.cuMemcpyHtoD_v2(dDtb, (nint)p, (nuint)headBytes).ThrowOnError();
                fixed (float* p = a) CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)headBytes).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)headBytes).ThrowOnError();
                fixed (float* p = b) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = c) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)bcBytes).ThrowOnError();
            }

            kernels.LaunchMamba2SelectiveScanF32(dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                nHead, headDim, dState, nGroup, seqLen, stream.Handle);
            stream.Synchronize();

            float[] yGpu = new float[yCpu.Length];
            float[] stateGpu = new float[stateCpu.Length];
            unsafe
            {
                fixed (float* p = yGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)yBytes).ThrowOnError();
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            for (int i = 0; i < yCpu.Length; i++)
            {
                float diff = MathF.Abs(yCpu[i] - yGpu[i]);
                float bar = AbsTol + RelTol * MathF.Abs(yCpu[i]);
                Assert.True(diff <= bar, $"y[{i}]: cpu={yCpu[i]:F6} vs cuda={yGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
            for (int i = 0; i < stateCpu.Length; i++)
            {
                float diff = MathF.Abs(stateCpu[i] - stateGpu[i]);
                float bar = AbsTol + RelTol * MathF.Abs(stateCpu[i]);
                Assert.True(diff <= bar, $"state[{i}]: cpu={stateCpu[i]:F6} vs cuda={stateGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
            _out.WriteLine($"nHead={nHead} headDim={headDim} dState={dState} nGroup={nGroup} seqLen={seqLen}: within tolerance.");
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dDtb != 0) CudaDriverApi.cuMemFree_v2(dDtb);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dC != 0) CudaDriverApi.cuMemFree_v2(dC);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
        }
    }

    /// <summary>Splitting a seqLen=8 scan into two seqLen=4 calls on the same state buffer must
    /// match a single seqLen=8 call — the property the decode loop relies on. Ports
    /// <c>VulkanMamba2SelectiveScanF32KernelTests.Launch_StatePersistsAcrossCalls</c>.</summary>
    [SkippableFact]
    public void Launch_StatePersistsAcrossCalls()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        const int nHead = 4, headDim = 8, dState = 16, nGroup = 2, seqLen = 8;
        int dInner = nHead * headDim;

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(unchecked((int)0xBEEFCAFE));
        float[] state0 = SmallRandom(rng, nHead * headDim * dState);
        float[] x = SmallRandom(rng, seqLen * dInner);
        float[] dtRaw = SmallRandom(rng, seqLen * nHead);
        float[] dtBias = SmallRandom(rng, nHead);
        float[] a = NegativeRandom(rng, nHead);
        float[] d = SmallRandom(rng, nHead);
        float[] b = SmallRandom(rng, seqLen * nGroup * dState);
        float[] c = SmallRandom(rng, seqLen * nGroup * dState);

        long stateBytes = (long)state0.Length * sizeof(float);
        nint dState_ = 0, dX = 0, dDt = 0, dDtb = 0, dA = 0, dD = 0, dB = 0, dC = 0, dY = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)((long)x.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)((long)dtRaw.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDtb, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)((long)b.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dC, (nuint)((long)c.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)((long)seqLen * dInner * sizeof(float))).ThrowOnError();

            unsafe
            {
                fixed (float* p = dtBias) CudaDriverApi.cuMemcpyHtoD_v2(dDtb, (nint)p, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
                fixed (float* p = a) CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            }

            // 1. One-shot seqLen=8.
            unsafe
            {
                float[] s0 = (float[])state0.Clone();
                fixed (float* p = s0) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)((long)x.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = dtRaw) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)((long)dtRaw.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = b) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)((long)b.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = c) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)((long)c.Length * sizeof(float))).ThrowOnError();
            }
            kernels.LaunchMamba2SelectiveScanF32(dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                nHead, headDim, dState, nGroup, seqLen, stream.Handle);
            stream.Synchronize();
            float[] yOneShot = new float[seqLen * dInner];
            float[] stateOneShot = new float[state0.Length];
            unsafe
            {
                fixed (float* p = yOneShot) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)((long)yOneShot.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = stateOneShot) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            // 2. Two seqLen=4 calls on the same state buffer.
            unsafe
            {
                float[] s0 = (float[])state0.Clone();
                fixed (float* p = s0) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
            }
            RunHalf(kernels, stream, dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                x, dtRaw, b, c, 0, 4, nHead, headDim, dState, nGroup, dInner, out float[] yFirstHalf);
            RunHalf(kernels, stream, dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                x, dtRaw, b, c, 4, 4, nHead, headDim, dState, nGroup, dInner, out float[] ySecondHalf);
            float[] stateSplit = new float[state0.Length];
            unsafe
            {
                fixed (float* p = stateSplit) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            for (int i = 0; i < 4 * dInner; i++) Assert.Equal(yOneShot[i], yFirstHalf[i]);
            for (int i = 0; i < 4 * dInner; i++) Assert.Equal(yOneShot[4 * dInner + i], ySecondHalf[i]);
            for (int i = 0; i < state0.Length; i++) Assert.Equal(stateOneShot[i], stateSplit[i]);
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dDtb != 0) CudaDriverApi.cuMemFree_v2(dDtb);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dC != 0) CudaDriverApi.cuMemFree_v2(dC);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
        }
    }

    private static unsafe void RunHalf(CudaKernels kernels, CudaStream stream,
        nint dState_, nint dX, nint dDt, nint dDtb, nint dA, nint dD, nint dB, nint dC, nint dY,
        float[] x, float[] dtRaw, float[] b, float[] c,
        int tokenOffset, int halfLen, int nHead, int headDim, int dState, int nGroup, int dInner,
        out float[] yHalf)
    {
        float[] xSlice = x.AsSpan(tokenOffset * dInner, halfLen * dInner).ToArray();
        float[] dtSlice = dtRaw.AsSpan(tokenOffset * nHead, halfLen * nHead).ToArray();
        float[] bSlice = b.AsSpan(tokenOffset * nGroup * dState, halfLen * nGroup * dState).ToArray();
        float[] cSlice = c.AsSpan(tokenOffset * nGroup * dState, halfLen * nGroup * dState).ToArray();

        fixed (float* px = xSlice) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)px, (nuint)((long)xSlice.Length * sizeof(float))).ThrowOnError();
        fixed (float* pdt = dtSlice) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)pdt, (nuint)((long)dtSlice.Length * sizeof(float))).ThrowOnError();
        fixed (float* pb = bSlice) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)pb, (nuint)((long)bSlice.Length * sizeof(float))).ThrowOnError();
        fixed (float* pc = cSlice) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)pc, (nuint)((long)cSlice.Length * sizeof(float))).ThrowOnError();

        kernels.LaunchMamba2SelectiveScanF32(dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
            nHead, headDim, dState, nGroup, halfLen, stream.Handle);
        stream.Synchronize();

        yHalf = new float[halfLen * dInner];
        fixed (float* py = yHalf) CudaDriverApi.cuMemcpyDtoH_v2((nint)py, dY, (nuint)((long)yHalf.Length * sizeof(float))).ThrowOnError();
    }

    private static float[] SmallRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return arr;
    }

    private static float[] NegativeRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(-(rng.NextDouble() * 0.45 + 0.05));
        return arr;
    }
}
```

- [ ] **Step 5: Build PTX and run the test**

Run (from `native/`): `build_ptx.bat` (requires `CUDA_PATH` set). Verify `native/ptx/mamba2_selective_scan.ptx` is produced.
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaMamba2SelectiveScanF32Tests"`
Expected: PASS on a CUDA-capable host (or `Skipped` without one).

- [ ] **Step 6: Commit**

```bash
git add native/kernels/mamba2_selective_scan.cu native/build_ptx.bat src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaMamba2SelectiveScanF32Tests.cs
git commit -m "feat(cuda): add mamba2_selective_scan_f32 kernel for NemotronH (#347)"
```

---

### Task 2: Group RMSNorm CUDA kernel + launcher + unit test

**Files:**
- Create: `native/kernels/group_rmsnorm.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs` (module load + `LaunchGroupRmsNormF32`)
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaGroupRmsNormF32Tests.cs`

**Interfaces:**
- Produces: `CudaKernels.LaunchGroupRmsNormF32(nint x, nint weight, float eps, int seqLen, int nGroup, int groupDim, nint stream)` — consumed by Task 9 (`ForwardSsmBody`, CPU step 9 / Vulkan step 11, "Group RMSNorm").
- Consumes: nothing from other tasks.

**Reference:** `native/kernels/per_head_rmsnorm_f32.cu` (read in full — structural base: warp-shuffle tree reduction, one block per (row, group)). NemotronH's group RMSNorm differs from `per_head_rmsnorm_f32` in exactly one way: `per_head_rmsnorm_f32` broadcasts ONE shared `weight[head_dim]` array across every head (`vec[i] = vec[i] * ri * weight[i]`), but NemotronH's `ssm_norm.weight` is `[d_inner]` = `[n_group * group_dim]` — **each group owns its own weight slice** (CPU: `ssmW.NormWeight.AsSpan(g * groupDim, groupDim)` in `NemotronHTransformerModel.ForwardSsmBody` step 9). Reusing `per_head_rmsnorm_f32` as-is would silently apply the wrong group's gain. This kernel is `per_head_rmsnorm_f32` with the weight offset changed from `weight[i]` to `weight[g*group_dim + i]`, dims renamed, and layout is `x[t, n_group, group_dim]` — matches `NemotronHForwardState.SsmY`'s `[T, dInner]` buffer where `dInner = nGroup*groupDim` is treated as `nGroup` contiguous groups.

Precision: warp-shuffle tree reduction, same as `per_head_rmsnorm_f32` — tolerance-based parity vs. `RmsNorm.Execute`'s sequential CPU reduction, not bit-exact. This matches every other RMSNorm kernel in this codebase (Global Constraints).

- [ ] **Step 1: Write the new kernel file**

```cuda
// native/kernels/group_rmsnorm.cu
//
// Per-group RMS normalization (NVIDIA Nemotron-H Mamba2 SSM output norm). Structural copy of
// per_head_rmsnorm_f32.cu with ONE change: weight is indexed per-GROUP (weight[g*group_dim+i]),
// not shared across all groups (per_head_rmsnorm_f32's weight[i] broadcast is WRONG for this use
// — NemotronH's ssm_norm.weight is [n_group*group_dim], each group has its own gain slice; see
// NemotronHTransformerModel.ForwardSsmBody step 9, ssmW.NormWeight.AsSpan(g*groupDim, groupDim)).
//
// Layout: x is [seq_len, n_group, group_dim] row-major (n_group*group_dim == d_inner).
// weight is [n_group, group_dim] row-major (== ssm_norm.weight, GGUF shape [d_inner]).
//
// Warp-shuffle tree reduction (same precision philosophy as per_head_rmsnorm_f32 and every other
// RMSNorm-family CUDA kernel in this codebase) — tolerance-based parity with the CPU's sequential
// RmsNorm.Execute reduction, not bit-exact.

extern "C" __global__ void __launch_bounds__(256) group_rmsnorm_f32(
    float* __restrict__ x, const float* __restrict__ weight,
    const float eps, const int seq_len, const int n_group, const int group_dim)
{
    int block_id = blockIdx.x;
    int t = block_id / n_group, g = block_id % n_group;
    if (t >= seq_len) return;

    float* vec = x + (size_t)t * n_group * group_dim + (size_t)g * group_dim;
    const float* w = weight + (size_t)g * group_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x) { float v = vec[i]; sum_sq += v * v; }
    for (int off = warpSize / 2; off > 0; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float ws[32]; int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) ws[wid] = sum_sq; __syncthreads();
    if (wid == 0) { int nw = (blockDim.x + warpSize - 1) / warpSize; sum_sq = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off); }
    __shared__ float ri; if (threadIdx.x == 0) ri = rsqrtf(sum_sq / (float)group_dim + eps); __syncthreads();
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x)
        vec[i] = vec[i] * ri * w[i];
}
```

No `build_ptx.bat` change needed — not in `NO_FMA`, `FAST_MATH`, `ARCH_86`, or `CXX17` (matches `per_head_rmsnorm_f32.cu`'s own untouched default-flag treatment).

- [ ] **Step 2: Wire the loader + launcher into `CudaKernels.cs`**

Module load, alongside the `conv1d_causal.ptx`/`mamba2_selective_scan.ptx` loads added in Task 1:

```csharp
string groupRmsNormF32Path = Path.Combine(ptxDir, "group_rmsnorm.ptx");
if (File.Exists(groupRmsNormF32Path))
{
    _groupRmsNormF32Module = CudaModule.LoadFromFile(groupRmsNormF32Path);
    _groupRmsNormF32Func = _groupRmsNormF32Module.TryGetFunction("group_rmsnorm_f32");
}
```

Backing fields:

```csharp
private CudaModule? _groupRmsNormF32Module;
private nint _groupRmsNormF32Func;
```

Launcher, following `LaunchPerHeadRmsNormF32`'s exact shape:

```csharp
/// <summary>Per-group F32 RmsNorm (NVIDIA Nemotron-H Mamba2 SSM output norm). Each group has
/// its own weight slice — unlike <see cref="LaunchPerHeadRmsNormF32"/>'s single shared weight
/// array. See native/kernels/group_rmsnorm.cu.</summary>
/// <param name="x">Device pointer, <c>[seqLen, nGroup, groupDim]</c> F32, normalized in place.</param>
/// <param name="weight">Device pointer, <c>[nGroup, groupDim]</c> F32 (== ssm_norm.weight).</param>
public void LaunchGroupRmsNormF32(nint x, nint weight, float eps,
                                    int seqLen, int nGroup, int groupDim, nint stream)
{
    nint xArg = x, wArg = weight;
    float epsArg = eps;
    int slArg = seqLen, ngArg = nGroup, gdArg = groupDim;

    void** args = stackalloc void*[] {&xArg, &wArg, &epsArg, &slArg, &ngArg, &gdArg};

    CudaDriverApi.cuLaunchKernel(_groupRmsNormF32Func,
            (uint)(seqLen * nGroup), 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
}
```

- [ ] **Step 3: Write the failing unit test**

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaGroupRmsNormF32Tests.cs
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchGroupRmsNormF32"/>
/// (native/kernels/group_rmsnorm.cu) against a per-group application of
/// <see cref="RmsNorm.Execute"/> (the CPU reference NemotronH's ForwardSsmBody step 9 uses,
/// once per group, with each group's own weight slice).
/// </summary>
[Trait("Category", "GPU")]
public class CudaGroupRmsNormF32Tests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-4f;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableTheory]
    [InlineData(1, 2, 8)]     // smallest: 1 token, 2 groups
    [InlineData(4, 4, 32)]    // multi-token prefill
    [InlineData(1, 10, 8)]    // Nemotron-H-realistic: nGroup=10, groupDim=8 (dInner=80)
    public void Launch_MatchesCpuReference(int seqLen, int nGroup, int groupDim)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0x9E3 ^ seqLen ^ (nGroup << 8) ^ (groupDim << 16));
        int dInner = nGroup * groupDim;
        const float eps = 1e-5f;

        float[] x = new float[seqLen * dInner];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        float[] weight = new float[dInner];
        for (int i = 0; i < weight.Length; i++) weight[i] = 1.0f + (float)(rng.NextDouble() * 0.1 - 0.05);

        // CPU reference: apply RmsNorm.Execute once per (t, g) over its groupDim-wide slice with
        // that group's own weight slice — exactly NemotronHTransformerModel.ForwardSsmBody step 9.
        float[] cpuOut = new float[x.Length];
        for (int t = 0; t < seqLen; t++)
        {
            for (int g = 0; g < nGroup; g++)
            {
                int off = t * dInner + g * groupDim;
                RmsNorm.Execute(
                    x.AsSpan(off, groupDim),
                    weight.AsSpan(g * groupDim, groupDim),
                    eps,
                    cpuOut.AsSpan(off, groupDim));
            }
        }

        nint dX = 0, dW = 0;
        try
        {
            long xBytes = (long)x.Length * sizeof(float);
            long wBytes = (long)weight.Length * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dW, (nuint)wBytes).ThrowOnError();
            unsafe
            {
                fixed (float* px = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)px, (nuint)xBytes).ThrowOnError();
                fixed (float* pw = weight) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)pw, (nuint)wBytes).ThrowOnError();
            }

            kernels.LaunchGroupRmsNormF32(dX, dW, eps, seqLen, nGroup, groupDim, stream.Handle);
            stream.Synchronize();

            float[] gpuOut = new float[x.Length];
            unsafe { fixed (float* p = gpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dX, (nuint)xBytes).ThrowOnError(); }

            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                float bar = AbsTol + RelTol * MathF.Abs(cpuOut[i]);
                Assert.True(diff <= bar, $"[{i}]: cpu={cpuOut[i]:F6} vs cuda={gpuOut[i]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }
        finally
        {
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dW != 0) CudaDriverApi.cuMemFree_v2(dW);
        }
    }
}
```

- [ ] **Step 4: Build PTX and run the test**

Run: `native/build_ptx.bat`; verify `native/ptx/group_rmsnorm.ptx` is produced.
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaGroupRmsNormF32Tests"`
Expected: PASS (or `Skipped` without a CUDA GPU).

- [ ] **Step 5: Commit**

```bash
git add native/kernels/group_rmsnorm.cu src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaGroupRmsNormF32Tests.cs
git commit -m "feat(cuda): add group_rmsnorm_f32 kernel for NemotronH (#347)"
```

---

### Task 3: Squared-ReLU elementwise CUDA kernel + launcher + unit test

**Files:**
- Create: `native/kernels/relu_squared_inplace.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs` (module load + `LaunchReluSquaredInplaceF32`)
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaReluSquaredInplaceF32Tests.cs`

**Interfaces:**
- Produces: `CudaKernels.LaunchReluSquaredInplaceF32(nint x, int n, nint stream)` — consumed by Task 10 (`ForwardFfnBody`).
- Consumes: nothing from other tasks.

**Reference:** `src/DotLLM.Cpu/Kernels/ReluSquared.cs` (CPU oracle: `y = max(0,x)^2`, single buffer, no gating). NemotronH's FFN is explicitly non-gated squared-ReLU (`NemotronHTransformerModel.ForwardFfnBody`: `ReluSquared.Execute(ffnMid, ffnMid)`), unlike the existing CUDA `relu2_f32`/`relu2glu_f32` kernels (`native/kernels/relu2_f32.cu`) which compute the GLU-fused form `relu(gate)^2 * up` over TWO input buffers — those are for BitNet's gated MoE FFN, not applicable here (confirmed by reading `relu2_f32.cu`: its only entry points take `gate`+`up`, there is no plain single-buffer squared-ReLU kernel anywhere in `native/kernels/`).

- [ ] **Step 1: Write the new kernel file**

```cuda
// native/kernels/relu_squared_inplace.cu
//
// Plain elementwise squared-ReLU, single buffer, in place: x = max(0, x)^2.
// Bit-perfect-shape port of DotLLM.Cpu.Kernels.ReluSquared.Execute (used, un-gated, by
// NVIDIA Nemotron-H's FFN sub-layer — up -> relu_squared -> down, no gate). Distinct from
// relu2_f32.cu's relu2_f32/relu2glu_f32 (GLU-fused, two input buffers, BitNet MoE FFN) — this
// kernel takes exactly one buffer, matching ReluSquared.Execute's signature.

extern "C" __global__ void relu_squared_inplace_f32(float* __restrict__ x, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    v = v > 0.0f ? v : 0.0f;
    x[idx] = v * v;
}
```

No `build_ptx.bat` change required (pure multiply/compare, no transcendentals — safe under either default or fast-math flags; leaving it out of every special list, like `add_f32`'s sibling elementwise kernels that don't need `--fmad=false`, keeps the `build_ptx.bat` diff to zero for this kernel).

- [ ] **Step 2: Wire the loader + launcher into `CudaKernels.cs`**

Module load:

```csharp
string reluSquaredInplaceF32Path = Path.Combine(ptxDir, "relu_squared_inplace.ptx");
if (File.Exists(reluSquaredInplaceF32Path))
{
    _reluSquaredInplaceF32Module = CudaModule.LoadFromFile(reluSquaredInplaceF32Path);
    _reluSquaredInplaceF32Func = _reluSquaredInplaceF32Module.TryGetFunction("relu_squared_inplace_f32");
}
```

Backing fields:

```csharp
private CudaModule? _reluSquaredInplaceF32Module;
private nint _reluSquaredInplaceF32Func;
```

Launcher, following `LaunchSiluF32`'s exact shape (single-buffer elementwise, grid-sized off `n`):

```csharp
/// <summary>Plain elementwise squared-ReLU in place: x = max(0,x)^2. NVIDIA Nemotron-H's
/// non-gated FFN activation — distinct from <see cref="LaunchReLU2F32"/>'s GLU-fused form.
/// See native/kernels/relu_squared_inplace.cu.</summary>
public void LaunchReluSquaredInplaceF32(nint x, int n, nint stream)
{
    if (n <= 0) return;
    nint xArg = x;
    int nArg = n;
    void** args = stackalloc void*[] {&xArg, &nArg};
    uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
    CudaDriverApi.cuLaunchKernel(_reluSquaredInplaceF32Func,
            gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
}
```

- [ ] **Step 3: Write the failing unit test**

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaReluSquaredInplaceF32Tests.cs
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>Correctness anchor for <see cref="CudaKernels.LaunchReluSquaredInplaceF32"/>
/// against <see cref="ReluSquared.Execute"/>.</summary>
[Trait("Category", "GPU")]
public class CudaReluSquaredInplaceF32Tests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(37)]
    [InlineData(4096)]
    public void Launch_MatchesCpuReference(int n)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0x517 ^ n);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 4.0 - 2.0); // includes negatives

        float[] cpuOut = (float[])x.Clone();
        ReluSquared.Execute(cpuOut, cpuOut);

        nint dX = 0;
        try
        {
            long bytes = (long)n * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)bytes).ThrowOnError();
            unsafe { fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)bytes).ThrowOnError(); }

            kernels.LaunchReluSquaredInplaceF32(dX, n, stream.Handle);
            stream.Synchronize();

            float[] gpuOut = new float[n];
            unsafe { fixed (float* p = gpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dX, (nuint)bytes).ThrowOnError(); }

            for (int i = 0; i < n; i++)
                Assert.Equal(cpuOut[i], gpuOut[i], precision: 5);
        }
        finally
        {
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
        }
    }
}
```

- [ ] **Step 4: Build PTX and run the test**

Run: `native/build_ptx.bat`; verify `native/ptx/relu_squared_inplace.ptx` is produced.
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaReluSquaredInplaceF32Tests"`
Expected: PASS (or `Skipped`).

- [ ] **Step 5: Commit**

```bash
git add native/kernels/relu_squared_inplace.cu src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaReluSquaredInplaceF32Tests.cs
git commit -m "feat(cuda): add relu_squared_inplace_f32 kernel for NemotronH (#347)"
```

---

### Task 4: `CudaNemotronHSsmStateCache` — device-side per-sequence Mamba2 recurrent state

**Files:**
- Create: `src/DotLLM.Cuda/Architectures/CudaNemotronHSsmStateCache.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHSsmStateCacheTests.cs`

**Interfaces:**
- Produces: `internal sealed unsafe class CudaNemotronHSsmStateCache : ISsmState` with constructor `CudaNemotronHSsmStateCache(MambaSsmConfig ssm, int numSsmLayers)`, `nint GetConvStatePtr(int ssmLayerIndex)`, `nint GetSsmStatePtr(int ssmLayerIndex)`, `void Reset()`, `long AllocatedBytes`, `int NumSsmLayers`. Consumed by Task 9 (`ForwardSsmBody`) and Task 11 (`Forward`'s `_activeSsm`/`ResolveSsm`, `CreateSequenceState`).
- Consumes: `DotLLM.Core.Models.MambaSsmConfig` (already exists — `src/DotLLM.Core/Models/MambaSsmConfig.cs`, read in full for this plan: `ConvStateElements => (DConv-1)*ConvDim`, `SsmStateElements => DInner*DState`), `DotLLM.Core.Models.ISsmState` (already exists).

**Reference (read in full for this plan):** `src/DotLLM.Cuda/Architectures/CudaGdnStateCache.cs` — this class is a near-verbatim structural copy, substituting `MambaSsmConfig` for `GatedDeltaNetConfig` and `ISsmState`/`NumSsmLayers` for `IGdnState`/`NumGdnLayers`. Every `cuMemAlloc_v2`/`cuMemsetD8_v2`/`cuMemFree_v2`/`cuMemcpyDtoD_v2` call is copied unchanged.

- [ ] **Step 1: Write the class**

```csharp
// src/DotLLM.Cuda/Architectures/CudaNemotronHSsmStateCache.cs
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side per-sequence recurrent state cache for the Mamba2 SSM layers of a NemotronH
/// model. Mirror of <see cref="DotLLM.Models.Architectures.SsmStateCache"/> (CPU) and
/// <see cref="CudaGdnStateCache"/> (same pattern, different recurrence config), allocating GPU
/// memory via <c>cuMemAlloc_v2</c>.
/// </summary>
/// <remarks>
/// One cache instance covers all SSM layers for a single sequence. Per SSM layer the cache
/// stores two device buffers, both zero-initialised at construction:
/// <c>conv_state</c> (<c>(DConv-1) * ConvDim</c> F32 elements — rolling conv1d history) and
/// <c>ssm_state</c> (<c>DInner * DState</c> F32 elements — the Mamba2 recurrent state matrix,
/// shape <c>[n_head, head_dim, d_state]</c>). <c>mamba2_selective_scan_f32</c>
/// (native/kernels/mamba2_selective_scan.cu) and <c>conv1d_causal_f32</c> consume and mutate
/// these pointers directly.
/// </remarks>
internal sealed unsafe class CudaNemotronHSsmStateCache : ISsmState
{
    private readonly MambaSsmConfig _ssm;
    private readonly int _numSsmLayers;
    private readonly int _convStateElements;
    private readonly int _ssmStateElements;

    // Contiguous per-layer blocks. SSM layer ordinal i occupies:
    //   conv:  _convState + i * _convStateElements * sizeof(float)
    //   state: _ssmState  + i * _ssmStateElements  * sizeof(float)
    private nint _convState;
    private nint _ssmState;

    private bool _disposed;

    /// <inheritdoc/>
    public int NumSsmLayers => _numSsmLayers;

    /// <summary>Elements per layer in the conv rolling buffer.</summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>Elements per layer in the SSM matrix state.</summary>
    public int SsmStateElements => _ssmStateElements;

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numSsmLayers * (_convStateElements + _ssmStateElements) * sizeof(float);

    /// <summary>
    /// Device-to-device deep-copies this cache's current contents into a freshly-allocated
    /// <see cref="CudaNemotronHSsmStateCache"/> of the same shape — mirrors
    /// <see cref="CudaGdnStateCache.Clone"/> (used there for speculative-decoding state
    /// rollback; kept here for the same future use).
    /// </summary>
    public CudaNemotronHSsmStateCache Clone()
    {
        ThrowIfDisposed();
        var clone = new CudaNemotronHSsmStateCache(_ssm, _numSsmLayers);
        CopyTo(clone);
        return clone;
    }

    /// <summary>Device-to-device overwrites <paramref name="destination"/>'s buffers with this
    /// cache's current contents via <c>cuMemcpyDtoD_v2</c>. Both caches must share the same shape.</summary>
    public void CopyTo(CudaNemotronHSsmStateCache destination)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(destination);
        destination.ThrowIfDisposed();
        if (destination._numSsmLayers != _numSsmLayers
            || destination._convStateElements != _convStateElements
            || destination._ssmStateElements != _ssmStateElements)
        {
            throw new ArgumentException(
                "Destination CudaNemotronHSsmStateCache shape does not match this cache's shape.", nameof(destination));
        }

        if (_numSsmLayers == 0) return;

        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);
        if (convBytes > 0)
            CudaDriverApi.cuMemcpyDtoD_v2(destination._convState, _convState, (nuint)convBytes).ThrowOnError();
        if (stateBytes > 0)
            CudaDriverApi.cuMemcpyDtoD_v2(destination._ssmState, _ssmState, (nuint)stateBytes).ThrowOnError();
    }

    /// <summary>Creates a new SSM state cache for the given config and layer count. All buffers
    /// are zero-initialised (zero state = no prior history) using <c>cuMemsetD8_v2</c>.</summary>
    public CudaNemotronHSsmStateCache(MambaSsmConfig ssm, int numSsmLayers)
    {
        if (numSsmLayers < 0) throw new ArgumentOutOfRangeException(nameof(numSsmLayers));

        _ssm = ssm;
        _numSsmLayers = numSsmLayers;
        _convStateElements = ssm.ConvStateElements; // (DConv-1) * ConvDim
        _ssmStateElements = ssm.SsmStateElements;   // DInner * DState

        if (numSsmLayers == 0)
        {
            _convState = 0;
            _ssmState = 0;
            return;
        }

        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out _convState, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _ssmState, (nuint)stateBytes).ThrowOnError();

        CudaDriverApi.cuMemsetD8_v2(_convState, 0, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)stateBytes).ThrowOnError();
    }

    /// <summary>Device pointer to SSM layer <paramref name="ssmLayerIndex"/>'s conv rolling
    /// state buffer. Length: <see cref="ConvStateElements"/> floats.</summary>
    public nint GetConvStatePtr(int ssmLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerIndex >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerIndex));
        return _convState + (nint)((long)ssmLayerIndex * _convStateElements * sizeof(float));
    }

    /// <summary>Device pointer to SSM layer <paramref name="ssmLayerIndex"/>'s matrix-state
    /// buffer. Length: <see cref="SsmStateElements"/> floats (shape
    /// <c>[n_head, head_dim, d_state]</c> row-major).</summary>
    public nint GetSsmStatePtr(int ssmLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerIndex >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerIndex));
        return _ssmState + (nint)((long)ssmLayerIndex * _ssmStateElements * sizeof(float));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numSsmLayers == 0) return;
        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);
        CudaDriverApi.cuMemsetD8_v2(_convState, 0, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)stateBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_convState != 0) { CudaDriverApi.cuMemFree_v2(_convState); _convState = 0; }
        if (_ssmState != 0) { CudaDriverApi.cuMemFree_v2(_ssmState); _ssmState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaNemotronHSsmStateCache));
    }

    ~CudaNemotronHSsmStateCache()
    {
        if (_disposed) return;
        if (_convState != 0) CudaDriverApi.cuMemFree_v2(_convState);
        if (_ssmState != 0) CudaDriverApi.cuMemFree_v2(_ssmState);
    }
}
```

- [ ] **Step 2: Write the failing unit test**

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHSsmStateCacheTests.cs
using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public class CudaNemotronHSsmStateCacheTests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    [SkippableFact]
    public void Construct_ZeroInitializesBothBuffers()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        var ssm = new MambaSsmConfig(DConv: 4, DInner: 16, DState: 8, NGroup: 2, NHead: 2);
        using var cache = new CudaNemotronHSsmStateCache(ssm, numSsmLayers: 3);

        Assert.Equal(3, cache.NumSsmLayers);
        Assert.Equal((4 - 1) * ssm.ConvDim, cache.ConvStateElements);
        Assert.Equal(16 * 8, cache.SsmStateElements);

        for (int layer = 0; layer < 3; layer++)
        {
            float[] conv = Download(cache.GetConvStatePtr(layer), cache.ConvStateElements);
            float[] state = Download(cache.GetSsmStatePtr(layer), cache.SsmStateElements);
            Assert.All(conv, v => Assert.Equal(0f, v));
            Assert.All(state, v => Assert.Equal(0f, v));
        }
    }

    [SkippableFact]
    public void Reset_ZeroesNonZeroState()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        var ssm = new MambaSsmConfig(DConv: 4, DInner: 8, DState: 4, NGroup: 1, NHead: 2);
        using var cache = new CudaNemotronHSsmStateCache(ssm, numSsmLayers: 1);

        float[] ones = new float[cache.SsmStateElements];
        Array.Fill(ones, 1.0f);
        unsafe
        {
            fixed (float* p = ones)
                CudaDriverApi.cuMemcpyHtoD_v2(cache.GetSsmStatePtr(0), (nint)p,
                    (nuint)(ones.Length * sizeof(float))).ThrowOnError();
        }

        cache.Reset();
        float[] afterReset = Download(cache.GetSsmStatePtr(0), cache.SsmStateElements);
        Assert.All(afterReset, v => Assert.Equal(0f, v));
    }

    private static unsafe float[] Download(nint devicePtr, int count)
    {
        float[] result = new float[count];
        fixed (float* p = result)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(float))).ThrowOnError();
        return result;
    }
}
```

- [ ] **Step 3: Build and run the test**

Run: `dotnet build src/DotLLM.Cuda`
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaNemotronHSsmStateCacheTests"`
Expected: PASS (or `Skipped`).

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHSsmStateCache.cs tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHSsmStateCacheTests.cs
git commit -m "feat(cuda): add CudaNemotronHSsmStateCache device recurrent state (#347)"
```

---

### Task 5: `CudaNemotronHKvCache` — sparse F32 attention KV cache

**Files:**
- Create: `src/DotLLM.Cuda/Architectures/CudaNemotronHKvCache.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHKvCacheTests.cs`

**Interfaces:**
- Produces: `internal sealed class CudaNemotronHKvCache : IKvCache` with constructor `CudaNemotronHKvCache(int attentionLayerCount, int numKvHeads, int headDim, int maxSeqLen, int deviceId)`. Consumed by Task 10 (`ForwardAttentionBody`) and Task 11 (`CreateKvCache`).
- Consumes: `DotLLM.Core.Attention.IKvCache`, `DotLLM.Core.Tensors.TensorRef`/`DType` (all already exist, read in full for this plan).

**Design rationale:** the existing generic `CudaKvCache` (`src/DotLLM.Cuda/CudaKvCache.cs`) is **F16-only** (`GetKeysRef`/`GetValuesRef` hardcode `DType.Float16`, row bytes computed as `Stride(layerIndex) * sizeof(ushort)`) and its `Update(TensorRef,...)` throws `NotSupportedException` in favor of an F16-specific `UpdateDevice` — reusing it would force an F32→F16 conversion step this plan's F32-throughout design (Global Constraints) doesn't want, and would fight its device-pointer-based internal API. `CudaQwen3HybridDenseTransformerModel` instead owns a bespoke internal F16 KV store (`EnsureF16KvCache`/`WriteF16KvRows`) — also not reusable and considerably more machinery (host staging, WDDM-safe D2H readback) than a first correct NemotronH port needs. This class is a small, dedicated F32 cache sized to `attentionLayerCount` (not `config.NumLayers` — matches the CPU/Vulkan sparse-KV-slot design: `kvSlotForLayer[layer]` maps a physical layer index to a slot in `[0, attentionLayerCount)`, `-1` for non-attention layers) that implements the real `IKvCache` interface (`TensorRef`-based `Update`/`GetKeysRef`/`GetValuesRef`) so `ForwardAttentionBody` can call it exactly the way `NemotronHTransformerModel.ForwardAttentionBody` (CPU) does: `kvCache.Update(kRef, vRef, positions, kvSlot)`.

`Update` assumes **contiguous ascending positions starting at `positions[0]`** — the only pattern the CPU/Vulkan NemotronH hosts ever produce (prefill writes `[0..seqLen)`, decode writes a single position). This is documented, not silently assumed; a non-contiguous call throws.

- [ ] **Step 1: Write the class**

```csharp
// src/DotLLM.Cuda/Architectures/CudaNemotronHKvCache.cs
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Sparse, F32, per-attention-layer-slot device KV cache for <see cref="CudaNemotronHTransformerModel"/>.
/// Sized to <c>attentionLayerCount</c> (the number of <see cref="DotLLM.Core.Models.HybridLayerKind.Attention"/>
/// layers), not the model's total layer count — the model maps a physical layer index to a slot via
/// its own <c>kvSlotForLayer</c> array (mirrors <c>NemotronHTransformerModel</c>/
/// <c>VulkanNemotronHKvCache</c>'s sparse-KV-slot design exactly).
/// </summary>
/// <remarks>
/// F32 storage (not F16, unlike the generic <see cref="CudaKvCache"/>) to keep the model's
/// activation precision uniform end-to-end — see this plan's Global Constraints.
/// <see cref="Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/> requires
/// <paramref name="positions"/> to be contiguous and ascending starting at
/// <c>positions[0]</c> — the only write pattern NemotronH's CPU/Vulkan hosts ever produce
/// (prefill writes <c>[0, seqLen)</c>; decode writes one position). A non-contiguous call throws.
/// </remarks>
internal sealed class CudaNemotronHKvCache : IKvCache
{
    private readonly nint[] _keys;   // per-slot device buffers, [maxSeqLen, kvStride] F32
    private readonly nint[] _values;
    private readonly int _kvStride;  // numKvHeads * headDim
    private readonly int _maxSeqLen;
    private readonly int _deviceId;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of attention-layer slots this cache covers.</summary>
    public int AttentionLayerCount => _keys.Length;

    /// <summary>Total bytes allocated across every slot's key + value buffers.</summary>
    public long AllocatedBytes =>
        2L * _keys.Length * _maxSeqLen * _kvStride * sizeof(float);

    public CudaNemotronHKvCache(int attentionLayerCount, int numKvHeads, int headDim, int maxSeqLen, int deviceId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(attentionLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _deviceId = deviceId;
        _keys = new nint[attentionLayerCount];
        _values = new nint[attentionLayerCount];

        long bytesPerSlot = (long)maxSeqLen * _kvStride * sizeof(float);
        for (int i = 0; i < attentionLayerCount; i++)
        {
            CudaDriverApi.cuMemAlloc_v2(out _keys[i], (nuint)bytesPerSlot).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _values[i], (nuint)bytesPerSlot).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        ThrowIfDisposed();
        if (positions.IsEmpty)
            throw new ArgumentException("positions must be non-empty.", nameof(positions));
        int startPos = positions[0];
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] != startPos + i)
                throw new NotSupportedException(
                    $"{nameof(CudaNemotronHKvCache)}.Update requires contiguous ascending positions " +
                    $"starting at positions[0]={startPos}; got positions[{i}]={positions[i]}.");
        }

        int seqLen = keys.Dim0;
        if (startPos + seqLen > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positions),
                $"positions extend to {startPos + seqLen}, exceeding MaxLength={_maxSeqLen}.");

        long rowBytes = (long)_kvStride * sizeof(float);
        long bytesToCopy = (long)seqLen * rowBytes;
        nint dstK = _keys[layerIndex] + (nint)((long)startPos * rowBytes);
        nint dstV = _values[layerIndex] + (nint)((long)startPos * rowBytes);

        CudaDriverApi.cuMemcpyDtoD_v2(dstK, keys.DataPointer, (nuint)bytesToCopy).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(dstV, values.DataPointer, (nuint)bytesToCopy).ThrowOnError();

        int newLength = startPos + seqLen;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        ThrowIfDisposed();
        return new TensorRef(_currentLength, _kvStride, DType.Float32, _deviceId, _keys[layerIndex]);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        ThrowIfDisposed();
        return new TensorRef(_currentLength, _kvStride, DType.Float32, _deviceId, _values[layerIndex]);
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        ThrowIfDisposed();
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            $"{nameof(CudaNemotronHKvCache)}.Update(ITensor) not supported. Use Update(TensorRef).");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException(
            $"{nameof(CudaNemotronHKvCache)}.GetKeys(ITensor) not supported. Use GetKeysRef.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException(
            $"{nameof(CudaNemotronHKvCache)}.GetValues(ITensor) not supported. Use GetValuesRef.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        for (int i = 0; i < _keys.Length; i++)
        {
            if (_keys[i] != 0) CudaDriverApi.cuMemFree_v2(_keys[i]);
            if (_values[i] != 0) CudaDriverApi.cuMemFree_v2(_values[i]);
        }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaNemotronHKvCache));
    }
}
```

- [ ] **Step 2: Write the failing unit test**

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHKvCacheTests.cs
using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public class CudaNemotronHKvCacheTests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    [SkippableFact]
    public unsafe void Update_ThenGetKeysRef_RoundTripsWrittenRows()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        const int numKvHeads = 2, headDim = 4, maxSeqLen = 16;
        int kvStride = numKvHeads * headDim;
        using var cache = new CudaNemotronHKvCache(attentionLayerCount: 1, numKvHeads, headDim, maxSeqLen, deviceId: 0);

        // Prefill: write 3 rows at positions [0,1,2).
        float[] kHost = new float[3 * kvStride];
        for (int i = 0; i < kHost.Length; i++) kHost[i] = i + 1;
        nint dK = 0, dV = 0;
        try
        {
            long bytes = (long)kHost.Length * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)bytes).ThrowOnError();
            fixed (float* p = kHost)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)bytes).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)bytes).ThrowOnError();
            }

            var kRef = new TensorRef(3, kvStride, DType.Float32, 0, dK);
            var vRef = new TensorRef(3, kvStride, DType.Float32, 0, dV);
            cache.Update(kRef, vRef, new[] { 0, 1, 2 }, layerIndex: 0);

            Assert.Equal(3, cache.CurrentLength);
            TensorRef stored = cache.GetKeysRef(0);
            Assert.Equal(3, stored.Dim0);
            Assert.Equal(kvStride, stored.Dim1);

            float[] readBack = new float[3 * kvStride];
            fixed (float* p = readBack)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, stored.DataPointer, (nuint)bytes).ThrowOnError();
            Assert.Equal(kHost, readBack);
        }
        finally
        {
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
        }
    }

    [SkippableFact]
    public void Update_NonContiguousPositions_Throws()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        using var cache = new CudaNemotronHKvCache(attentionLayerCount: 1, numKvHeads: 1, headDim: 4, maxSeqLen: 8, deviceId: 0);
        var kRef = new TensorRef(2, 4, DType.Float32, 0, (nint)1);
        var vRef = new TensorRef(2, 4, DType.Float32, 0, (nint)1);
        Assert.Throws<NotSupportedException>(() => cache.Update(kRef, vRef, new[] { 0, 2 }, layerIndex: 0));
    }

    [SkippableFact]
    public void Rollback_ReducesCurrentLength()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        using var cache = new CudaNemotronHKvCache(attentionLayerCount: 1, numKvHeads: 1, headDim: 4, maxSeqLen: 8, deviceId: 0);
        // Directly exercise Rollback's bound check without a real Update (0-length cache).
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Rollback(1));
        cache.Rollback(0);
        Assert.Equal(0, cache.CurrentLength);
    }
}
```

- [ ] **Step 3: Build and run the test**

Run: `dotnet build src/DotLLM.Cuda`
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaNemotronHKvCacheTests"`
Expected: PASS (or `Skipped`).

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHKvCache.cs tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHKvCacheTests.cs
git commit -m "feat(cuda): add CudaNemotronHKvCache sparse F32 attention cache (#347)"
```

---

### Task 6: `CudaNemotronHForwardState` — device forward-pass scratch buffers

**Files:**
- Create: `src/DotLLM.Cuda/Architectures/CudaNemotronHForwardState.cs`

**Interfaces:**
- Produces: `internal sealed class CudaNemotronHForwardState : IDisposable` with constructor `CudaNemotronHForwardState(int hiddenSize, int maxIntermediateSize, int vocabSize, int qElems, int kvElems, int inputProjectionDim, int convDim, int dConv, int dInner, int nHead, int nGroup, int dState, int maxSeqLen)`, `bool EnsureCapacity(int seqLen)` (returns whether a reallocation happened), and `nint` fields for every scratch buffer plus `long AllocatedBytes`. Consumed by Task 7 (construction) and Tasks 9-11 (every forward-body method).
- Consumes: nothing new — plain `cuMemAlloc_v2`/`cuMemFree_v2`.

**Reference (read in full for this plan):** `src/DotLLM.Models/Architectures/NemotronHForwardState.cs` — same buffer inventory, same power-of-two growth strategy (`EnsureCapacity`), same sizing formula for `maxIntermediateSize`; ported from `NativeMemory.AlignedAlloc`/`Span<float>` to `cuMemAlloc_v2`/raw `nint` device pointers. Two buffers the CPU version doesn't need are added: `TokenIdsDevice`/`PositionsDevice` (int32 device arrays — the CUDA embedding-lookup and RoPE kernels read positions/token ids from device memory, unlike the CPU host which reads `ReadOnlySpan<int>` directly). The CPU version's `InputQ8Scratch` (pre-quantization scratch for the CPU's own Q8/Q8_1/Q8_K activation quantizer) has NO CUDA equivalent here — the CUDA `Gemm` dispatcher (Task 8) manages its own F16 activation-conversion scratch internally (mirrors `CudaQwen3HybridDenseTransformerModel`'s `_activF16InScratch`/`_activF16OutScratch`, which are model-level fields, not part of this per-call forward state).

- [ ] **Step 1: Write the class**

```csharp
// src/DotLLM.Cuda/Architectures/CudaNemotronHForwardState.cs
using System.Numerics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side, power-of-two-growth scratch buffers for the NemotronH hybrid forward pass on
/// CUDA. Mirrors <see cref="DotLLM.Models.Architectures.NemotronHForwardState"/> (CPU) —
/// same buffer inventory and growth strategy, ported to <c>cuMemAlloc_v2</c> device pointers.
/// Adds <see cref="TokenIdsDevice"/>/<see cref="PositionsDevice"/> (int32 device arrays the
/// embedding-lookup and RoPE CUDA kernels read from device memory) which the CPU host doesn't
/// need (it reads <c>ReadOnlySpan&lt;int&gt;</c> directly).
/// </summary>
internal sealed class CudaNemotronHForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _maxIntermediateSize;
    private readonly int _vocabSize;
    private readonly int _qElems;
    private readonly int _kvElems;

    private readonly int _inputProjectionDim;
    private readonly int _convDim;
    private readonly int _dConv;
    private readonly int _dInner;
    private readonly int _nHead;
    private readonly int _bcDim; // n_group * d_state

    private int _currentSeqLen;
    private readonly int _maxSeqLen; // cap for TokenIdsDevice/PositionsDevice (int32, not resized per-call)

    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint FfnIntermediate;
    public nint Logits;

    public nint QScratch;
    public nint KScratch;
    public nint VScratch;
    public nint AttnOutput;

    public nint Zxbcdt;
    public nint ConvInput;
    public nint XBC;
    public nint DtBuffer;
    public nint SsmX;
    public nint SsmB;
    public nint SsmC;
    public nint SsmY;
    /// <summary>Extraction scratch for the SwiGLU gate `z` slice of Zxbcdt (CPU/Vulkan step
    /// 8/10) — needed because Zxbcdt's per-token row stride is <c>inputProjectionDim</c>, not
    /// <c>dInner</c>, so `z` cannot be passed directly to a single fused
    /// <c>LaunchSwiGLUF32</c> call over all seqLen tokens at once without first being copied
    /// into a contiguous dInner-strided buffer. See Task 9.</summary>
    public nint SsmZ;

    /// <summary>Device int32 array of the current call's token ids, length &gt;= seqLen. Plain
    /// field (not a property) like every other buffer here, so it can be passed by
    /// <c>ref</c> to <see cref="FreeIfNonZero"/> in <see cref="Dispose"/>.</summary>
    public nint TokenIdsDevice;

    /// <summary>Device int32 array of the current call's positions, length &gt;= seqLen.</summary>
    public nint PositionsDevice;

    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;
            long floats = 0;
            floats += s * _hiddenSize * 3;             // HiddenState, Residual, NormOutput
            floats += s * _maxIntermediateSize;         // FfnIntermediate
            floats += s * _vocabSize;                   // Logits
            floats += s * _qElems;                      // QScratch
            floats += s * _kvElems * 2;                 // KScratch, VScratch
            floats += s * _qElems;                       // AttnOutput
            floats += s * _inputProjectionDim;           // Zxbcdt
            floats += (_dConv - 1 + s) * _convDim;        // ConvInput
            floats += s * _convDim;                       // XBC
            floats += s * _nHead;                         // DtBuffer
            floats += s * _dInner;                        // SsmX
            floats += s * _bcDim * 2;                     // SsmB, SsmC
            floats += s * _dInner;                        // SsmY
            floats += s * _dInner;                        // SsmZ
            long bytes = floats * sizeof(float);
            bytes += (long)_maxSeqLen * sizeof(int) * 2;  // TokenIdsDevice, PositionsDevice
            return bytes;
        }
    }

    public CudaNemotronHForwardState(
        int hiddenSize, int maxIntermediateSize, int vocabSize, int qElems, int kvElems,
        int inputProjectionDim, int convDim, int dConv, int dInner, int nHead, int nGroup,
        int dState, int maxSeqLen)
    {
        _hiddenSize = hiddenSize;
        _maxIntermediateSize = maxIntermediateSize;
        _vocabSize = vocabSize;
        _qElems = qElems;
        _kvElems = kvElems;
        _inputProjectionDim = inputProjectionDim;
        _convDim = convDim;
        _dConv = dConv;
        _dInner = dInner;
        _nHead = nHead;
        _bcDim = nGroup * dState;
        _maxSeqLen = maxSeqLen;

        TokenIdsDevice = AllocInts(maxSeqLen);
        PositionsDevice = AllocInts(maxSeqLen);

        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    /// <summary>Grows every seqLen-dependent buffer to at least <paramref name="seqLen"/> rows
    /// (rounded up to the next power of two), freeing and reallocating if it grows. Returns
    /// true iff a reallocation happened (callers that cache descriptor/graph state keyed on
    /// buffer identity must invalidate on true).</summary>
    public bool EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return false;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSeqBuffers();

        HiddenState = AllocFloats((long)cap * _hiddenSize);
        Residual = AllocFloats((long)cap * _hiddenSize);
        NormOutput = AllocFloats((long)cap * _hiddenSize);
        FfnIntermediate = AllocFloats((long)cap * _maxIntermediateSize);
        Logits = AllocFloats((long)cap * _vocabSize);

        QScratch = AllocFloats((long)cap * _qElems);
        KScratch = AllocFloats((long)cap * _kvElems);
        VScratch = AllocFloats((long)cap * _kvElems);
        AttnOutput = AllocFloats((long)cap * _qElems);

        Zxbcdt = AllocFloats((long)cap * _inputProjectionDim);
        ConvInput = AllocFloats((long)(_dConv - 1 + cap) * _convDim);
        XBC = AllocFloats((long)cap * _convDim);
        DtBuffer = AllocFloats((long)cap * _nHead);
        SsmX = AllocFloats((long)cap * _dInner);
        SsmB = AllocFloats((long)cap * _bcDim);
        SsmC = AllocFloats((long)cap * _bcDim);
        SsmY = AllocFloats((long)cap * _dInner);
        SsmZ = AllocFloats((long)cap * _dInner);

        _currentSeqLen = cap;
        return true;
    }

    private static nint AllocFloats(long count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)(count * sizeof(float))).ThrowOnError();
        return ptr;
    }

    private static nint AllocInts(long count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)(count * sizeof(int))).ThrowOnError();
        return ptr;
    }

    private void FreeSeqBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref FfnIntermediate);
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref QScratch);
        FreeIfNonZero(ref KScratch);
        FreeIfNonZero(ref VScratch);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref Zxbcdt);
        FreeIfNonZero(ref ConvInput);
        FreeIfNonZero(ref XBC);
        FreeIfNonZero(ref DtBuffer);
        FreeIfNonZero(ref SsmX);
        FreeIfNonZero(ref SsmB);
        FreeIfNonZero(ref SsmC);
        FreeIfNonZero(ref SsmY);
        FreeIfNonZero(ref SsmZ);
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    public void Dispose()
    {
        FreeSeqBuffers();
        FreeIfNonZero(ref TokenIdsDevice);
        FreeIfNonZero(ref PositionsDevice);
        _currentSeqLen = 0;
    }
}
```

- [ ] **Step 2: Build**

Run: `dotnet build src/DotLLM.Cuda`
Expected: builds cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHForwardState.cs
git commit -m "feat(cuda): add CudaNemotronHForwardState device scratch buffers (#347)"
```

---

### Task 7: `CudaNemotronHTransformerModel` — fields, structs, constructor, `LoadFromGguf`, `Dispose`

**Files:**
- Create: `src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs`

**Interfaces:**
- Produces: `public sealed unsafe class CudaNemotronHTransformerModel : IModel` (partial build-out — this task adds fields, the private constructor, `LoadFromGguf`, per-layer device structs, and `Dispose`; Tasks 8-11 add the rest of the class body to the SAME file). Consumed by Task 12 (`CudaModelLoader`).
- Consumes: `NemotronHTransformerModel.LoadFromGguf` + its `internal` fields (`_layers`, `_outputNormWeight`, `_tokenEmbedWeight`, `_tokenEmbedQuantType`, `_outputWeight`, `_outputQuantType`, `_outputOutputDim`, `_outputInputDim` — all `private`, extracted via reflection, exactly as `VulkanNemotronHTransformerModel.BuildFromGguf` already does for the identical fields), `NemotronHLayerWeights`/`NemotronHSsmWeights`/`NemotronHAttentionWeights`/`NemotronHFfnWeights` (all `internal`, visible to `DotLLM.Cuda` via the existing `InternalsVisibleTo` in `DotLLM.Models.csproj`), `CudaContext`/`CudaStream`/`CudaCublasHandle`/`CudaKernels` (all already exist), `CudaNemotronHForwardState` (Task 6), `CudaNemotronHSsmStateCache` (Task 4).

**Design decision — reuse the CPU loader instead of re-deriving tensor names:** confirmed by reading `DotLLM.Models.csproj` (`<InternalsVisibleTo Include="DotLLM.Cuda" />` already present, added for the exact same reason Vulkan needed it). `VulkanNemotronHTransformerModel.BuildFromGguf` deliberately calls `NemotronHTransformerModel.LoadFromGguf` (CPU) first and reflects out its already-loaded `NemotronHLayerWeights[]`, rather than re-parsing GGUF tensor names — its own doc comment states why: "duplicating the tensor-name mapping here would be a second place for the Nemotron-H naming conventions... to drift." This CUDA loader follows the identical pattern.

- [ ] **Step 1: Write the file — usings, class header, fields, device-layer structs**

```csharp
// src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the NemotronH (<c>nemotron_h</c>) hybrid Mamba2-SSM + Transformer
/// model — e.g. NVIDIA's Nemotron-3-Nano-4B. F32 activations throughout, mirrors
/// <c>DotLLM.Models.Architectures.NemotronHTransformerModel</c> (CPU) and
/// <see cref="DotLLM.Vulkan.VulkanNemotronHTransformerModel"/> on the GPU.
/// </summary>
/// <remarks>
/// Each of the <c>config.NumLayers</c> layers is exactly one of Mamba2 SSM, GQA attention, or
/// squared-ReLU FFN (<see cref="ModelConfig.HybridLayout"/>'s <c>LayerKind</c> per layer), with a
/// single pre-sublayer RMSNorm and one residual add shared by all three kinds — see
/// <c>NemotronHTransformerModel.Forward</c> (CPU) for the authoritative per-layer sequence this
/// class's <c>Forward</c> (Task 11) mirrors.
/// </remarks>
public sealed unsafe class CudaNemotronHTransformerModel : IModel
{
    private readonly CudaNemotronHForwardState _state;
    private readonly CudaNemotronHSsmStateCache _ssmCache;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly int _deviceId;

    private readonly DeviceLayer[] _layers;

    // NOT device memory — the embedding LOOKUP is done as a per-call host-side row dequant in
    // Embed() (Task 8), reading directly from this retained HOST pointer (mmap'd GGUF data, or
    // a synthetic fixture's unmanaged host buffer), then H2D-copying only the seqLen rows
    // actually needed per call. This mirrors CudaQwen3HybridDenseTransformerModel's identical
    // "_embedDataBase" pattern and its documented rationale (only ever need seqLen rows, never
    // the whole vocab table) — see Task 8 for the full design note. Uniformly covers every GGUF
    // quant type (F32/F16/Q8_0/Q4_K/Q5_K/Q6_K/...) with one code path instead of a device-kernel
    // fast path for 3 formats plus a host fallback for the rest.
    private readonly nint _tokenEmbedHostPtr;
    private readonly QuantizationType _tokenEmbedQt;
    private readonly long _tokenEmbedRowBytes;
    private readonly nint _outputNormDevice;   // F32 [hiddenSize]
    private readonly nint _outputDevice;       // lm_head raw quant bytes (may alias _tokenEmbedDevice)
    private readonly QuantizationType _outputQt;
    private readonly int _outputOutputDim;     // vocab size
    private readonly int _outputInputDim;      // hidden size
    private readonly bool _ownsOutputDevice;

    private readonly HybridLayerLayout _layout;
    private readonly MambaSsmConfig _ssm;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _ssmLayerOrdinal;
    private readonly int _numSsmLayers;

    private readonly float _ropeTheta;
    private readonly int _ropeDim;

    // Prefill dequant-to-F16 + cuBLAS-HGEMM scratch, shared by every projection whose weight
    // has no native F32 CUDA kernel — see Task 8's Gemm dispatcher.
    private nint _dequantScratchF16Weight;
    private nint _activF16InScratch;
    private long _activF16InScratchElems;
    private nint _activF16OutScratch;
    private long _activF16OutScratchElems;

    // Caller-supplied per-sequence SSM state for the in-flight Forward — see Task 11.
    private CudaNemotronHSsmStateCache? _activeSsm;

    /// <summary>The CPU model <see cref="LoadFromGguf"/> reused to resolve GGUF tensor names;
    /// disposed with this model so its dequantised F32 norm arrays are released. The
    /// <see cref="GgufFile"/> itself stays caller-owned (mirrors
    /// <see cref="DotLLM.Vulkan.VulkanNemotronHTransformerModel"/>'s identical field). Null on
    /// the <see cref="BuildFromPrebuiltWeights"/> (synthetic-fixture) path.</summary>
    private NemotronHTransformerModel? _cpuModel;

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _ssmCache.AllocatedBytes;

    /// <summary>Number of attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>Creates a <see cref="CudaNemotronHKvCache"/> sized for this model.</summary>
    public CudaNemotronHKvCache CreateKvCache(int maxSeqLen)
        => new(_attentionLayerCount, Config.NumKvHeads, Config.HeadDim, maxSeqLen, _deviceId);

    // ── Device-side per-layer weight structs ────────────────────────────────

    private readonly struct DeviceSsm
    {
        public required nint InWeight { get; init; }
        public required QuantizationType InQt { get; init; }
        public required int InInputDim { get; init; }
        public required int InOutputDim { get; init; }
        public required nint Conv1dWeightDevice { get; init; }  // F32 [dConv, convDim]
        public required nint Conv1dBiasDevice { get; init; }    // F32 [convDim]
        public required nint ADevice { get; init; }             // F32 [nHead]
        public required nint DDevice { get; init; }             // F32 [nHead]
        public required nint DtBiasDevice { get; init; }        // F32 [nHead]
        public required nint NormWeightDevice { get; init; }    // F32 [dInner]
        public required nint OutWeight { get; init; }
        public required QuantizationType OutQt { get; init; }
        public required int OutInputDim { get; init; }
        public required int OutOutputDim { get; init; }
    }

    private readonly struct DeviceAttn
    {
        public required nint QWeight { get; init; }
        public required QuantizationType QQt { get; init; }
        public required int QInputDim { get; init; }
        public required int QOutputDim { get; init; }
        public required nint KWeight { get; init; }
        public required QuantizationType KQt { get; init; }
        public required int KInputDim { get; init; }
        public required int KOutputDim { get; init; }
        public required nint VWeight { get; init; }
        public required QuantizationType VQt { get; init; }
        public required int VInputDim { get; init; }
        public required int VOutputDim { get; init; }
        public required nint OWeight { get; init; }
        public required QuantizationType OQt { get; init; }
        public required int OInputDim { get; init; }
        public required int OOutputDim { get; init; }
        public required int NumKvHeads { get; init; }
    }

    private readonly struct DeviceFfn
    {
        public required nint UpWeight { get; init; }
        public required QuantizationType UpQt { get; init; }
        public required int UpInputDim { get; init; }
        public required int UpOutputDim { get; init; }
        public required nint DownWeight { get; init; }
        public required QuantizationType DownQt { get; init; }
        public required int DownInputDim { get; init; }
        public required int DownOutputDim { get; init; }
    }

    private readonly struct DeviceLayer
    {
        public required nint AttnNormWeightDevice { get; init; } // F32 [hiddenSize]
        public required HybridLayerKind Kind { get; init; }
        public DeviceSsm? Ssm { get; init; }
        public DeviceAttn? Attention { get; init; }
        public DeviceFfn? Ffn { get; init; }
    }
```

- [ ] **Step 2: Constructor**

```csharp
    private CudaNemotronHTransformerModel(
        ModelConfig config,
        DeviceLayer[] layers,
        nint tokenEmbedHostPtr, QuantizationType tokenEmbedQt, long tokenEmbedRowBytes,
        nint outputNormDevice,
        nint outputDevice, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        bool ownsOutputDevice,
        int[] kvSlotForLayer, int attentionLayerCount,
        float ropeTheta, int ropeDim,
        CudaNemotronHForwardState state, CudaNemotronHSsmStateCache ssmCache,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context, CudaKernels kernels,
        int deviceId, nint dequantScratchDevice)
    {
        Config = config;
        _layers = layers;
        _tokenEmbedHostPtr = tokenEmbedHostPtr;
        _tokenEmbedQt = tokenEmbedQt;
        _tokenEmbedRowBytes = tokenEmbedRowBytes;
        _outputNormDevice = outputNormDevice;
        _outputDevice = outputDevice;
        _outputQt = outputQt;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _ownsOutputDevice = ownsOutputDevice;
        _layout = config.HybridLayout!;
        _ssm = config.SsmConfig!.Value;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _state = state;
        _ssmCache = ssmCache;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _deviceId = deviceId;
        _dequantScratchF16Weight = dequantScratchDevice;

        _ssmLayerOrdinal = new int[config.NumLayers];
        int ssmOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _ssmLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.Ssm
                ? ssmOrdinal++
                : -1;
        }
        _numSsmLayers = ssmOrdinal;
    }
```

- [ ] **Step 3: `LoadFromGguf` — reuse the CPU loader, reflect out its weights, upload to device**

```csharp
    /// <summary>
    /// Loads a NemotronH model from an opened GGUF file onto the given CUDA device. Reuses
    /// <see cref="NemotronHTransformerModel.LoadFromGguf"/> (CPU) for all GGUF tensor-name
    /// resolution and shape validation, then uploads the resulting weights to device memory —
    /// see this task's "Design decision" note for why.
    /// </summary>
    public static CudaNemotronHTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.NemotronH)
            throw new ArgumentException(
                $"CudaNemotronHTransformerModel requires Architecture.NemotronH, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));

        var cpuModel = NemotronHTransformerModel.LoadFromGguf(gguf, config);
        try
        {
            var cpuLayers = ExtractCpuLayers(cpuModel);
            var outputNormWeight = ExtractOutputNormWeight(cpuModel);
            var (tokenEmbedPtr, tokenEmbedQt) = ExtractTokenEmbed(cpuModel);
            var (outputPtr, outputQt, outputM, outputK) = ExtractOutput(cpuModel);

            var model = BuildFromPrebuiltWeights(
                config, cpuLayers, outputNormWeight,
                outputPtr, outputQt, outputM, outputK,
                tokenEmbedPtr, tokenEmbedQt,
                deviceId, ptxDir);
            model._cpuModel = cpuModel;
            return model;
        }
        catch
        {
            cpuModel.Dispose();
            throw;
        }
    }

    // ── CPU-model field extraction via reflection (mirrors VulkanNemotronHTransformerModel's
    // identical Field()/Extract* helpers — the fields are `private`, not `internal`, so
    // InternalsVisibleTo alone doesn't expose them). ─────────────────────────────────────────

    private static NemotronHLayerWeights[] ExtractCpuLayers(NemotronHTransformerModel m)
        => (NemotronHLayerWeights[])Field("_layers").GetValue(m)!;

    private static float[] ExtractOutputNormWeight(NemotronHTransformerModel m)
        => (float[])Field("_outputNormWeight").GetValue(m)!;

    private static (nint ptr, QuantizationType qt) ExtractTokenEmbed(NemotronHTransformerModel m)
        => ((nint)Field("_tokenEmbedWeight").GetValue(m)!,
            (QuantizationType)Field("_tokenEmbedQuantType").GetValue(m)!);

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) ExtractOutput(
        NemotronHTransformerModel m)
        => ((nint)Field("_outputWeight").GetValue(m)!,
            (QuantizationType)Field("_outputQuantType").GetValue(m)!,
            (int)Field("_outputOutputDim").GetValue(m)!,
            (int)Field("_outputInputDim").GetValue(m)!);

    private static System.Reflection.FieldInfo Field(string name)
        => typeof(NemotronHTransformerModel).GetField(
               name,
               System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
           ?? throw new InvalidOperationException($"NemotronHTransformerModel.{name} field missing.");
```

- [ ] **Step 4: `Dispose`**

```csharp
    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _state.Dispose();
        _ssmCache.Dispose();
        FreeIfNonZero(ref _dequantScratchF16Weight);
        FreeIfNonZero(ref _activF16InScratch);
        FreeIfNonZero(ref _activF16OutScratch);

        for (int i = 0; i < _layers.Length; i++)
            FreeLayer(_layers[i]);

        // _tokenEmbedHostPtr is a borrowed host pointer (mmap'd GGUF data or a caller-owned
        // fixture buffer) — never device memory, never freed here. See its field doc.
        nint outputNorm = _outputNormDevice;
        FreeIfNonZero(ref outputNorm);
        if (_ownsOutputDevice)
        {
            nint outputDevice = _outputDevice;
            FreeIfNonZero(ref outputDevice);
        }

        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
        _cpuModel?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private static void FreeLayer(in DeviceLayer layer)
    {
        nint p = layer.AttnNormWeightDevice; FreeIfNonZero(ref p);
        if (layer.Ssm is { } s)
        {
            nint a = s.InWeight; FreeIfNonZero(ref a);
            nint b = s.Conv1dWeightDevice; FreeIfNonZero(ref b);
            nint c = s.Conv1dBiasDevice; FreeIfNonZero(ref c);
            nint d = s.ADevice; FreeIfNonZero(ref d);
            nint e = s.DDevice; FreeIfNonZero(ref e);
            nint f = s.DtBiasDevice; FreeIfNonZero(ref f);
            nint g = s.NormWeightDevice; FreeIfNonZero(ref g);
            nint h = s.OutWeight; FreeIfNonZero(ref h);
        }
        if (layer.Attention is { } at)
        {
            nint q = at.QWeight; FreeIfNonZero(ref q);
            nint k = at.KWeight; FreeIfNonZero(ref k);
            nint v = at.VWeight; FreeIfNonZero(ref v);
            nint o = at.OWeight; FreeIfNonZero(ref o);
        }
        if (layer.Ffn is { } ff)
        {
            nint up = ff.UpWeight; FreeIfNonZero(ref up);
            nint down = ff.DownWeight; FreeIfNonZero(ref down);
        }
    }

    // ── Shared device-upload primitives (reused by BuildFromPrebuiltWeights, Task 8) ────────

    private static nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        return ptr;
    }

    private static void CopyHtoD(nint dst, nint src, long bytes)
    {
        CudaDriverApi.cuMemcpyHtoD_v2(dst, src, (nuint)bytes).ThrowOnError();
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    /// <summary>Uploads a projection's raw quantized (or F32/F16) bytes from a host pointer
    /// (mmap'd GGUF data, or a synthetic fixture's unmanaged buffer) to a fresh device buffer.
    /// No PQ2_0 repack (unlike <c>CudaQwen3HybridDenseTransformerModel.UploadRawTensor</c>) —
    /// NemotronH GGUFs never carry PQ2_0 tensors.</summary>
    private static nint UploadRawTensorFromHost(nint hostPtr, QuantizationType qt, int outputDim, int inputDim)
    {
        long bytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        nint device = AllocDevice(bytes);
        CopyHtoD(device, hostPtr, bytes);
        return device;
    }

    /// <summary>Uploads an already-dequantised managed float array (e.g. <c>NormWeight</c>,
    /// <c>Conv1dWeight</c>, <c>AttnNormWeight</c> — every small per-layer F32 array
    /// <see cref="NemotronHLayerWeights"/>'s CPU loader already materialised) to device memory.</summary>
    private static nint UploadF32ArrayFrom(float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        nint device = AllocDevice(bytes);
        fixed (float* p = data)
        {
            CopyHtoD(device, (nint)p, bytes);
        }
        return device;
    }
```

- [ ] **Step 5: Build (will not fully compile yet — `BuildFromPrebuiltWeights`, the `Gemm`/`Embed` helpers, and the `Forward`/body methods are added in Tasks 8-11; this task's file is intentionally incomplete until Task 11 lands)**

Run: `dotnet build src/DotLLM.Cuda` — expect compile errors referencing `BuildFromPrebuiltWeights` (not yet defined) and `Dequantize` (confirm the `using DotLLM.Models.Gguf;` import at the top of the file resolves it — `Dequantize` lives in `DotLLM.Models.Gguf`, same namespace as `GgufFile`/`GgufTensorDescriptor`, already imported). This is expected and resolved by Task 8; do not attempt to make Task 7 alone compile standalone.

- [ ] **Step 6: Commit (WIP, part of a multi-task file build-out)**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
git commit -m "feat(cuda): CudaNemotronHTransformerModel scaffolding + GGUF loader (#347) [1/5]"
```

---

### Task 8: `CudaNemotronHTransformerModel` — `BuildFromPrebuiltWeights`, `Gemm` dispatcher, `Embed`

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs` (append to the class body started in Task 7)

**Interfaces:**
- Produces: `internal static CudaNemotronHTransformerModel BuildFromPrebuiltWeights(ModelConfig config, NemotronHLayerWeights[] cpuLayers, float[] outputNormWeight, nint outputWeight, QuantizationType outputQt, int outputOutputDim, int outputInputDim, nint tokenEmbedWeight, QuantizationType tokenEmbedQt, int deviceId = 0, string? ptxDir = null)` — consumed by Task 7's `LoadFromGguf` and Task 13's synthetic-fixture test. `private void Gemm(nint weight, QuantizationType qt, nint x, nint y, int m, int k, int seqLen)` and `private void Embed(ReadOnlySpan<int> tokenIds, nint hiddenDevice, int hiddenSize)` — consumed by Tasks 9-11.
- Consumes: `CudaGemm.LinearF32`/`GemvF32`/`LinearF16`/`GemvF16` (`src/DotLLM.Cuda/CudaGemm.cs`, already exist), `CudaKernels.LaunchQuantizedGemvF32In`/`LaunchQuantizedGemv`/`LaunchQuantizedGemvMmq`/`LaunchDequantToF16`/`LaunchConvertF32ToF16`/`LaunchConvertF16ToF32`/`HasMmq`/`HasQuantizedGemvKernel`/`ForceDirectGemv` (`src/DotLLM.Cuda/CudaKernels.cs`, already exist).

**Design note — `Embed` always uses the host-dequant path (never a device embedding-lookup kernel):** `CudaKernels.LaunchEmbeddingLookupF32` only supports `F32`/`F16`/`Q8_0` embed dtypes; real NemotronH GGUFs commonly quantize `token_embd.weight` as `Q4_K`/`Q5_K`/`Q6_K` (per `DESIGN.md`'s target `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf`). Rather than a device-kernel fast path for 3 formats plus a host-dequant fallback for the rest (two code paths to verify), this uses ONE code path for every quant type — the same choice `CudaQwen3HybridDenseTransformerModel` already made and documents (`LoadFromGguf`'s remarks: "the embedding LOOKUP... is done as a per-call host-side row dequant in Forward() instead — only `seqLen` rows are ever needed, never all [vocab]"). `_tokenEmbedHostPtr` (Task 7) is the retained, never-freed, caller-owned host pointer this reads from.

**Reference (read in full for this plan):** `CudaQwen3HybridDenseTransformerModel.Gemm` (lines 2503-2592) — this task's `Gemm` is that method verbatim minus its `I2_S`/`PQ2_0` branches (NemotronH GGUFs never carry those formats); `CudaQwen3HybridDenseTransformerModel.LoadFromGguf`'s embedding-lookup remarks (lines ~381-405) for the `Embed` design rationale; `NemotronHTransformerModel.EmbedTokens` (CPU, already read in Task 1's research) for the exact F32/F16/else dequant branching `Embed` mirrors.

- [ ] **Step 1: `BuildFromPrebuiltWeights` — validate, allocate device context, upload output/embed metadata**

Append to the class body (after the upload primitives from Task 7 Step 4):

```csharp
    /// <summary>
    /// Builds a NemotronH CUDA model from caller-owned, pre-built <see cref="NemotronHLayerWeights"/> —
    /// the shared entry point for <see cref="LoadFromGguf"/> (Task 7) and the synthetic-fixture
    /// parity test (Task 13). Caller retains ownership of every host <see cref="nint"/> pointer
    /// (token embed, output, plus every projection inside <paramref name="cpuLayers"/>) — this
    /// method only reads from them to build fresh device buffers plus the retained
    /// <c>_tokenEmbedHostPtr</c> for per-call embedding dequant.
    /// </summary>
    internal static CudaNemotronHTransformerModel BuildFromPrebuiltWeights(
        ModelConfig config,
        NemotronHLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt,
        int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);
        if (config.Architecture != Architecture.NemotronH)
            throw new ArgumentException(
                $"CudaNemotronHTransformerModel requires Architecture.NemotronH, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));
        if (cpuLayers.Length != config.NumLayers)
            throw new ArgumentException(
                $"cpuLayers length {cpuLayers.Length} != config.NumLayers {config.NumLayers}.", nameof(cpuLayers));

        var layout = config.HybridLayout!;
        var ssm = config.SsmConfig!.Value;
        int hiddenSize = config.HiddenSize;

        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        long maxTileFloats = 0;

        // Output norm — always F32 [hiddenSize].
        nint outputNormDevice = UploadF32ArrayFrom(outputNormWeight);

        // lm_head — always uploaded as its own fresh device buffer (no tied-embedding aliasing
        // optimization; NemotronH's CPU loader already resolves the tied case at the host-pointer
        // level, so outputWeight here is already the correct source regardless).
        nint outputDevice = UploadRawTensorFromHost(outputWeight, outputQt, outputOutputDim, outputInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)outputOutputDim * outputInputDim);

        long tokenEmbedRowBytes = Dequantize.RowByteSize(hiddenSize, tokenEmbedQt);

        // Per-layer upload.
        var layers = new DeviceLayer[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        int numSsmLayers = 0;
        int maxIntermediate = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            var cpuLayer = cpuLayers[i];
            nint attnNormDevice = UploadF32ArrayFrom(cpuLayer.AttnNormWeight);

            DeviceSsm? ssmDev = null;
            DeviceAttn? attnDev = null;
            DeviceFfn? ffnDev = null;

            switch (layout.LayerKind[i])
            {
                case HybridLayerKind.Ssm:
                    ssmDev = UploadDeviceSsmLayer(cpuLayer.Ssm!, ref maxTileFloats);
                    numSsmLayers++;
                    break;
                case HybridLayerKind.Attention:
                    attnDev = UploadDeviceAttentionLayer(cpuLayer.Attention!, ref maxTileFloats);
                    kvSlotForLayer[i] = attentionLayerCount++;
                    break;
                case HybridLayerKind.Ffn:
                    ffnDev = UploadDeviceFfnLayer(cpuLayer.Ffn!, ref maxTileFloats);
                    if (cpuLayer.Ffn!.UpOutputDim > maxIntermediate) maxIntermediate = cpuLayer.Ffn.UpOutputDim;
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {layout.LayerKind[i]} at layer {i}.");
            }
            if (layout.LayerKind[i] != HybridLayerKind.Attention) kvSlotForLayer[i] = -1;

            layers[i] = new DeviceLayer
            {
                AttnNormWeightDevice = attnNormDevice,
                Kind = layout.LayerKind[i],
                Ssm = ssmDev,
                Attention = attnDev,
                Ffn = ffnDev,
            };
        }
        if (maxIntermediate == 0) maxIntermediate = hiddenSize;

        // RoPE config — allow ropeDim==0 only when there are no attention layers (mirrors
        // NemotronHTransformerModel.BuildFromPrebuiltWeights exactly).
        int ropeDim = config.RoPEConfig?.DimensionCount ?? 0;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        if (attentionLayerCount > 0)
        {
            if (ropeDim <= 0 || (ropeDim & 1) != 0)
                throw new ArgumentException(
                    $"NemotronH attention layers require an even rope_dim > 0 (got {ropeDim}).", nameof(config));
            if (ropeDim > config.HeadDim)
                throw new ArgumentException(
                    $"rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.", nameof(config));
        }

        var state = new CudaNemotronHForwardState(
            hiddenSize: hiddenSize,
            maxIntermediateSize: maxIntermediate,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            inputProjectionDim: ssm.InputProjectionDim,
            convDim: ssm.ConvDim,
            dConv: ssm.DConv,
            dInner: ssm.DInner,
            nHead: ssm.NHead,
            nGroup: ssm.NGroup,
            dState: ssm.DState,
            maxSeqLen: config.MaxSequenceLength);

        var ssmCache = new CudaNemotronHSsmStateCache(ssm, numSsmLayers);

        UpdateMaxTile(ref maxTileFloats, maxIntermediate); // dequant scratch floor for tiny models
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        return new CudaNemotronHTransformerModel(
            config, layers,
            tokenEmbedWeight, tokenEmbedQt, tokenEmbedRowBytes,
            outputNormDevice,
            outputDevice, outputQt, outputOutputDim, outputInputDim, ownsOutputDevice: true,
            kvSlotForLayer, attentionLayerCount,
            ropeTheta, ropeDim,
            state, ssmCache, stream, cublas, context, kernels, deviceId, dequantScratchDevice);
    }

    private static void UpdateMaxTile(ref long max, long candidate)
    {
        if (candidate > max) max = candidate;
    }

    private static DeviceSsm UploadDeviceSsmLayer(NemotronHSsmWeights w, ref long maxTileFloats)
    {
        nint inDevice = UploadRawTensorFromHost(w.InWeight, w.InQuantType, w.InOutputDim, w.InInputDim);
        nint outDevice = UploadRawTensorFromHost(w.OutWeight, w.OutQuantType, w.OutOutputDim, w.OutInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.InOutputDim * w.InInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.OutOutputDim * w.OutInputDim);

        return new DeviceSsm
        {
            InWeight = inDevice, InQt = w.InQuantType, InInputDim = w.InInputDim, InOutputDim = w.InOutputDim,
            Conv1dWeightDevice = UploadF32ArrayFrom(w.Conv1dWeight),
            Conv1dBiasDevice = UploadF32ArrayFrom(w.Conv1dBias),
            ADevice = UploadF32ArrayFrom(w.A),
            DDevice = UploadF32ArrayFrom(w.D),
            DtBiasDevice = UploadF32ArrayFrom(w.DtBias),
            NormWeightDevice = UploadF32ArrayFrom(w.NormWeight),
            OutWeight = outDevice, OutQt = w.OutQuantType, OutInputDim = w.OutInputDim, OutOutputDim = w.OutOutputDim,
        };
    }

    private static DeviceAttn UploadDeviceAttentionLayer(NemotronHAttentionWeights w, ref long maxTileFloats)
    {
        nint qDevice = UploadRawTensorFromHost(w.QWeight, w.QQuantType, w.QOutputDim, w.QInputDim);
        nint kDevice = UploadRawTensorFromHost(w.KWeight, w.KQuantType, w.KOutputDim, w.KInputDim);
        nint vDevice = UploadRawTensorFromHost(w.VWeight, w.VQuantType, w.VOutputDim, w.VInputDim);
        nint oDevice = UploadRawTensorFromHost(w.OWeight, w.OQuantType, w.OOutputDim, w.OInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.QOutputDim * w.QInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.KOutputDim * w.KInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.VOutputDim * w.VInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.OOutputDim * w.OInputDim);

        return new DeviceAttn
        {
            QWeight = qDevice, QQt = w.QQuantType, QInputDim = w.QInputDim, QOutputDim = w.QOutputDim,
            KWeight = kDevice, KQt = w.KQuantType, KInputDim = w.KInputDim, KOutputDim = w.KOutputDim,
            VWeight = vDevice, VQt = w.VQuantType, VInputDim = w.VInputDim, VOutputDim = w.VOutputDim,
            OWeight = oDevice, OQt = w.OQuantType, OInputDim = w.OInputDim, OOutputDim = w.OOutputDim,
            NumKvHeads = w.NumKvHeads,
        };
    }

    private static DeviceFfn UploadDeviceFfnLayer(NemotronHFfnWeights w, ref long maxTileFloats)
    {
        nint upDevice = UploadRawTensorFromHost(w.UpWeight, w.UpQuantType, w.UpOutputDim, w.UpInputDim);
        nint downDevice = UploadRawTensorFromHost(w.DownWeight, w.DownQuantType, w.DownOutputDim, w.DownInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.UpOutputDim * w.UpInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.DownOutputDim * w.DownInputDim);

        return new DeviceFfn
        {
            UpWeight = upDevice, UpQt = w.UpQuantType, UpInputDim = w.UpInputDim, UpOutputDim = w.UpOutputDim,
            DownWeight = downDevice, DownQt = w.DownQuantType, DownInputDim = w.DownInputDim, DownOutputDim = w.DownOutputDim,
        };
    }
```

- [ ] **Step 2: `Gemm` dispatcher**

```csharp
    /// <summary>
    /// Dispatches one linear projection <c>Y[seqLen,m] = X[seqLen,k] @ W[m,k]^T</c> by quant
    /// type. F32 weights go straight through cuBLAS. Q8_0 decode (seqLen==1) uses the direct
    /// F32-in/F32-out quantized GEMV kernel. Every other quant format (F16, K-quants, IQ-quants)
    /// funnels through an F16 dequant-then-cuBLAS-HGEMM round trip. Verbatim copy of
    /// <c>CudaQwen3HybridDenseTransformerModel.Gemm</c> minus its I2_S/PQ2_0 branches (NemotronH
    /// GGUFs never carry ternary/PQ2_0 tensors).
    /// </summary>
    private void Gemm(nint weight, QuantizationType qt, nint x, nint y, int m, int k, int seqLen)
    {
        nint streamH = _stream.Handle;

        if (qt == QuantizationType.F32)
        {
            CudaGemm.LinearF32(_cublas.Handle, x, weight, y, seqLen, k, m, streamH);
            return;
        }

        if (seqLen == 1)
        {
            if (qt == QuantizationType.Q8_0)
            {
                _kernels.LaunchQuantizedGemvF32In(weight, x, y, m, k, streamH);
                return;
            }

            if (qt == QuantizationType.F16
                || _kernels.HasMmq(qt)
                || _kernels.HasQuantizedGemvKernel(qt))
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);

                if (qt == QuantizationType.F16)
                {
                    CudaGemm.GemvF16(_cublas.Handle, weight, _activF16InScratch,
                        _activF16OutScratch, m, k, streamH);
                }
                else if (_kernels.HasMmq(qt) && !CudaKernels.ForceDirectGemv)
                {
                    _kernels.LaunchQuantizedGemvMmq(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, preqScratch: 0, streamH);
                }
                else
                {
                    _kernels.LaunchQuantizedGemv(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, streamH);
                }

                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }
        }

        // Prefill (seqLen > 1) and decode fallback for any quant format without a direct path.
        long totalElems = (long)m * k;
        int totalElemsI = checked((int)totalElems);
        int activInElems = checked((int)((long)seqLen * k));
        int activOutElems = checked((int)((long)seqLen * m));
        EnsureActivF16InScratch(activInElems);
        EnsureActivF16OutScratch(activOutElems);

        _kernels.LaunchDequantToF16(weight, qt, _dequantScratchF16Weight, totalElemsI, streamH);
        _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, activInElems, streamH);
        CudaGemm.LinearF16(_cublas.Handle, _activF16InScratch, _dequantScratchF16Weight,
            _activF16OutScratch, seqLen, k, m, streamH);
        _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, activOutElems, streamH);
    }

    private void EnsureActivF16InScratch(long halfs)
    {
        if (halfs <= _activF16InScratchElems) return;
        FreeIfNonZero(ref _activF16InScratch);
        CudaDriverApi.cuMemAlloc_v2(out _activF16InScratch, (nuint)(halfs * sizeof(ushort))).ThrowOnError();
        _activF16InScratchElems = halfs;
    }

    private void EnsureActivF16OutScratch(long halfs)
    {
        if (halfs <= _activF16OutScratchElems) return;
        FreeIfNonZero(ref _activF16OutScratch);
        CudaDriverApi.cuMemAlloc_v2(out _activF16OutScratch, (nuint)(halfs * sizeof(ushort))).ThrowOnError();
        _activF16OutScratchElems = halfs;
    }
```

- [ ] **Step 3: `Embed`**

```csharp
    /// <summary>
    /// Embedding lookup via per-call host-side row dequant (see this task's design note) —
    /// writes <paramref name="tokenIds"/>.Length rows of <paramref name="hiddenSize"/> floats
    /// each into the device buffer <paramref name="hiddenDevice"/>. Mirrors
    /// <c>NemotronHTransformerModel.EmbedTokens</c> (CPU) exactly: F32 straight copy, F16
    /// widened via <c>TensorPrimitives.ConvertToSingle</c>, everything else via
    /// <c>Dequantize.ToFloat32</c>.
    /// </summary>
    private void Embed(ReadOnlySpan<int> tokenIds, nint hiddenDevice, int hiddenSize)
    {
        int seqLen = tokenIds.Length;
        float[] hostRows = new float[seqLen * hiddenSize];

        for (int t = 0; t < seqLen; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {Config.VocabSize}).");

            Span<float> dst = hostRows.AsSpan(t * hiddenSize, hiddenSize);
            nint rowPtr = _tokenEmbedHostPtr + (nint)((long)tokenId * _tokenEmbedRowBytes);

            if (_tokenEmbedQt == QuantizationType.F32)
            {
                new ReadOnlySpan<float>((float*)rowPtr, hiddenSize).CopyTo(dst);
            }
            else if (_tokenEmbedQt == QuantizationType.F16)
            {
                var src = new ReadOnlySpan<Half>((Half*)rowPtr, hiddenSize);
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(src, dst);
            }
            else
            {
                Dequantize.ToFloat32(rowPtr, hiddenSize, _tokenEmbedQt, dst);
            }
        }

        fixed (float* p = hostRows)
        {
            CopyHtoD(hiddenDevice, (nint)p, (long)hostRows.Length * sizeof(float));
        }
    }
```

- [ ] **Step 4: Build**

Run: `dotnet build src/DotLLM.Cuda`
Expected: builds cleanly (still missing `Forward`/body methods — Tasks 9-11 — this is normal for this WIP file).

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
git commit -m "feat(cuda): CudaNemotronHTransformerModel BuildFromPrebuiltWeights + Gemm + Embed (#347) [2/5]"
```

---

### Task 9: `CudaNemotronHTransformerModel` — `ForwardSsmBody`

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs` (append)

**Interfaces:**
- Produces: `private void ForwardSsmBody(in DeviceSsm ssmW, int absoluteLayerIndex, int seqLen, int hiddenSize, float eps)` — consumed by Task 11's per-layer dispatch loop.
- Consumes: `CudaKernels.LaunchConv1dCausalF32` (existing), `CudaKernels.LaunchSiluF32` (existing), `CudaKernels.LaunchMamba2SelectiveScanF32` (Task 1), `CudaKernels.LaunchSwiGLUF32` (existing), `CudaKernels.LaunchGroupRmsNormF32` (Task 2), `Gemm` (Task 8), `CudaNemotronHSsmStateCache.GetConvStatePtr`/`GetSsmStatePtr` (Task 4), `CudaDriverApi.cuMemcpyDtoDAsync_v2` (already exists — confirmed present at `src/DotLLM.Cuda/Interop/CudaDriverApi.cs:176`, already used by `CudaKvCache.cs`/`CudaKvBlockPool.cs`).

**Reference (read in full for this plan):** `NemotronHTransformerModel.ForwardSsmBody` (CPU, the 12-numbered-step block) and `VulkanNemotronHTransformerModel.RecordSsmLayer` (the Vulkan 12-step sequence with explicit compute/transfer barriers between every step). This CUDA version is the SAME 11-operation sequence (CPU steps 6 "dt = bias-add" and 7 "y += x\*D" are fused into the Task 1 scan kernel, so there are 2 fewer discrete steps than the CPU/Vulkan numbering) with **no explicit barriers between operations** — unlike Vulkan's `KernelSupport.ComputeToComputeBarrier`/`ComputeToTransferBarrier` calls between every step, CUDA's driver-API stream-ordering guarantees every operation queued on the same `nint stream` (here, `_stream.Handle`, used for every launch and every `cuMemcpyDtoDAsync_v2` in this method) executes in submission order with no explicit fence needed — this matches how the sibling `CudaQwen3HybridDenseTransformerModel.ForwardGdnBody` sequences its own kernel launches (no manual barriers anywhere in that method either).

One structural difference from Vulkan worth calling out: Vulkan's step 10-11 extracts `z` into a scratch buffer, runs `SwiGLU(gate=SsmZ, up=SsmY, output=SsmZ)` into a THIRD location, then device-copies `SsmZ -> SsmY` — its own comment says "The kernel requires distinct buffers" (a Vulkan descriptor-binding constraint). `swiglu_f32.cu`'s actual per-element body (`output[idx] = (g/(1+expf(-g))) * u`) reads `gate[idx]`/`up[idx]` into registers before writing `output[idx]`, so aliasing `up == output` is a plain data-race-free in-place update on CUDA — this plan's version calls `LaunchSwiGLUF32(gate: SsmZ, up: SsmY, output: SsmY, ...)` directly, in place, skipping the extra copy-back Vulkan needs.

- [ ] **Step 1: Write `ForwardSsmBody`**

```csharp
    /// <summary>
    /// Mamba2 SSM sub-layer forward — reads pre-normed activations from
    /// <c>_state.NormOutput</c> and writes the ssm_out projection back into the same buffer.
    /// Advances the per-layer conv/SSM recurrent state in place. See this task's reference note
    /// for how this 11-operation CUDA sequence maps onto the CPU/Vulkan 12-step numbering (two
    /// steps are fused into the Task 1 scan kernel).
    /// </summary>
    private void ForwardSsmBody(in DeviceSsm ssmW, int absoluteLayerIndex, int seqLen, int hiddenSize, float eps)
    {
        int dInner = _ssm.DInner;
        int dConv = _ssm.DConv;
        int nHead = _ssm.NHead;
        int headDim = _ssm.HeadDim;
        int dState = _ssm.DState;
        int nGroup = _ssm.NGroup;
        int convDim = _ssm.ConvDim;
        int groupDim = dInner / nGroup;
        int inProjDim = _ssm.InputProjectionDim;
        int bcDim = nGroup * dState;
        int dtOffset = 2 * dInner + 2 * nGroup * dState;
        nint streamH = _stream.Handle;

        int ssmOrdinal = _ssmLayerOrdinal[absoluteLayerIndex];
        var activeSsm = _activeSsm ?? _ssmCache;
        nint convStatePtr = activeSsm.GetConvStatePtr(ssmOrdinal);
        nint ssmStatePtr = activeSsm.GetSsmStatePtr(ssmOrdinal);

        // 1. ssm_in GEMM: NormOutput[seqLen, hiddenSize] -> Zxbcdt[seqLen, inProjDim].
        Gemm(ssmW.InWeight, ssmW.InQt, _state.NormOutput, _state.Zxbcdt, inProjDim, hiddenSize, seqLen);

        // 2. ConvInput = concat(conv_state, xBC rows sliced out of Zxbcdt).
        long convDimBytes = (long)convDim * sizeof(float);
        long inProjRowBytes = (long)inProjDim * sizeof(float);
        if (dConv > 1)
        {
            long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.ConvInput, convStatePtr, (nuint)convStateBytes, streamH).ThrowOnError();
        }
        for (int t = 0; t < seqLen; t++)
        {
            nint src = _state.Zxbcdt + (nint)((long)t * inProjRowBytes + (long)dInner * sizeof(float));
            nint dst = _state.ConvInput + (nint)((long)(dConv - 1 + t) * convDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)convDimBytes, streamH).ThrowOnError();
        }

        // 3. Conv1d causal -> XBC. Reuses the existing generic conv1d_causal_f32 kernel as-is.
        _kernels.LaunchConv1dCausalF32(_state.ConvInput, ssmW.Conv1dWeightDevice, ssmW.Conv1dBiasDevice, _state.XBC,
            dConv, convDim, seqLen, streamH);

        // 4. SiLU on XBC in place.
        _kernels.LaunchSiluF32(_state.XBC, (long)seqLen * convDim, streamH);

        // 5. Save the trailing (dConv-1) rows of ConvInput (pre-SiLU) back into conv_state.
        if (dConv > 1)
        {
            long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
            nint src = _state.ConvInput + (nint)((long)seqLen * convDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(convStatePtr, src, (nuint)convStateBytes, streamH).ThrowOnError();
        }

        // 6. Extract the RAW dt slice (bias-add + guarded softplus are fused into the scan
        // kernel launched in step 8 — see Task 1).
        long dtRowBytes = (long)nHead * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            nint src = _state.Zxbcdt + (nint)((long)t * inProjRowBytes + (long)dtOffset * sizeof(float));
            nint dst = _state.DtBuffer + (nint)((long)t * dtRowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)dtRowBytes, streamH).ThrowOnError();
        }

        // 7. Split XBC[t,:] = [x | B | C] into SsmX / SsmB / SsmC.
        long xRowBytes = (long)dInner * sizeof(float);
        long bcRowBytes = (long)bcDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            nint rowBase = _state.XBC + (nint)((long)t * convDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.SsmX + (nint)((long)t * xRowBytes), rowBase,
                (nuint)xRowBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.SsmB + (nint)((long)t * bcRowBytes), rowBase + (nint)xRowBytes,
                (nuint)bcRowBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.SsmC + (nint)((long)t * bcRowBytes),
                rowBase + (nint)(xRowBytes + bcRowBytes), (nuint)bcRowBytes, streamH).ThrowOnError();
        }

        // 8. Mamba2 selective scan — dt bias-add, guarded softplus, decay, and the D-skip term
        // are ALL fused into this one launch (see Task 1's kernel documentation).
        _kernels.LaunchMamba2SelectiveScanF32(
            ssmStatePtr, _state.SsmX, _state.DtBuffer, ssmW.DtBiasDevice, ssmW.ADevice, ssmW.DDevice,
            _state.SsmB, _state.SsmC, _state.SsmY,
            nHead, headDim, dState, nGroup, seqLen, streamH);

        // 9. Extract z = Zxbcdt[t, 0..dInner) into SsmZ (strided source row, contiguous dest).
        for (int t = 0; t < seqLen; t++)
        {
            nint src = _state.Zxbcdt + (nint)((long)t * inProjRowBytes);
            nint dst = _state.SsmZ + (nint)((long)t * xRowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)xRowBytes, streamH).ThrowOnError();
        }

        // 10. SwiGLU gating in place: SsmY = SiLU(SsmZ) * SsmY. Safe to alias up==output — see
        // this task's reference note.
        _kernels.LaunchSwiGLUF32(_state.SsmZ, _state.SsmY, _state.SsmY, dInner, seqLen, streamH);

        // 11. Group RMSNorm on SsmY in place.
        _kernels.LaunchGroupRmsNormF32(_state.SsmY, ssmW.NormWeightDevice, eps, seqLen, nGroup, groupDim, streamH);

        // 12. ssm_out projection into NormOutput.
        Gemm(ssmW.OutWeight, ssmW.OutQt, _state.SsmY, _state.NormOutput, hiddenSize, dInner, seqLen);
    }
```

- [ ] **Step 2: Build**

Run: `dotnet build src/DotLLM.Cuda`
Expected: builds cleanly (still missing `ForwardAttentionBody`/`ForwardFfnBody`/`Forward` — Tasks 10-11).

- [ ] **Step 3: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
git commit -m "feat(cuda): CudaNemotronHTransformerModel ForwardSsmBody (#347) [3/5]"
```

---

### Task 10: `CudaNemotronHTransformerModel` — `ForwardAttentionBody`, `ForwardFfnBody`

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs` (append)

**Interfaces:**
- Produces: `private void ForwardAttentionBody(in DeviceAttn attn, int absoluteLayerIndex, int seqLen, ReadOnlySpan<int> positions, int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)` and `private void ForwardFfnBody(in DeviceFfn ffn, int seqLen, int hiddenSize)` — both consumed by Task 11's per-layer dispatch loop.
- Consumes: `CudaKernels.LaunchRoPEF32`/`LaunchAttentionF32`/`ToCudaRopeType` (existing), `CudaKernels.LaunchReluSquaredInplaceF32` (Task 3), `Gemm` (Task 8), `CudaNemotronHKvCache` (Task 5) via the generic `IKvCache`/`TensorRef` interface (not the concrete type — any `IKvCache` implementation honoring `Update(TensorRef,...)`/`GetKeysRef`/`GetValuesRef` works, exactly like the CPU host's `ForwardAttentionBody`).

**Reference (read in full for this plan):** `NemotronHTransformerModel.ForwardAttentionBody` (CPU) for the cached-vs-uncached branch structure (`if (kvCache is not null) {...} else {...}`) and the RoPE/attention parameter mapping; `CudaKernels.LaunchRoPEF32`'s doc comment (already confirms `ropeType=0`/`freqDim=0`/`neoxPairOffset=0` is the correct call shape for "Qwen3 / NemotronH / Llama" partial rotation — this is a direct, already-documented match, not a new derivation); `CudaKernels.ToCudaRopeType` for the `RoPEType.Norm -> 0` mapping (matches CPU's `RoPE.Execute(..., RoPEType.Norm)` call in `NemotronHTransformerModel.ForwardAttentionBody`); `NemotronHTransformerModel.ForwardFfnBody` (CPU) for the FFN's up -> squared-ReLU -> down sequence (no gate).

- [ ] **Step 1: `ForwardAttentionBody`**

```csharp
    /// <summary>
    /// GQA attention sub-layer forward — reads pre-normed activations from
    /// <c>_state.NormOutput</c> and writes the o_proj result back into the same buffer.
    /// <paramref name="positions"/> is the HOST-side position span (needed by
    /// <see cref="IKvCache.Update"/>'s signature); <c>_state.PositionsDevice</c> (uploaded once
    /// per <c>Forward</c> call by Task 11) is the device-side copy RoPE reads from.
    /// </summary>
    private void ForwardAttentionBody(
        in DeviceAttn attn, int absoluteLayerIndex, int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int kvStride = numKvHeads * headDim;
        nint streamH = _stream.Handle;

        Gemm(attn.QWeight, attn.QQt, _state.NormOutput, _state.QScratch, attn.QOutputDim, attn.QInputDim, seqLen);
        Gemm(attn.KWeight, attn.KQt, _state.NormOutput, _state.KScratch, attn.KOutputDim, attn.KInputDim, seqLen);
        Gemm(attn.VWeight, attn.VQt, _state.NormOutput, _state.VScratch, attn.VOutputDim, attn.VInputDim, seqLen);

        // Partial RoPE: RoPEType.Norm (GPT-J interleaved pairing) over the first _ropeDim dims
        // of each head — matches NemotronHTransformerModel.ForwardAttentionBody's
        // RoPE.Execute(..., RoPEType.Norm) call. LaunchRoPEF32's own doc comment already
        // documents ropeType=0/freqDim=0/neoxPairOffset=0 as the correct shape for NemotronH.
        _kernels.LaunchRoPEF32(_state.QScratch, _state.KScratch, _state.PositionsDevice,
            seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta,
            CudaKernels.ToCudaRopeType(RoPEType.Norm), streamH);

        if (kvCache is not null)
        {
            int kvSlot = _kvSlotForLayer[absoluteLayerIndex];
            if (kvSlot < 0)
                throw new InvalidOperationException(
                    $"Layer {absoluteLayerIndex} has no KV-cache slot (not an attention layer).");

            var kRef = new TensorRef(seqLen, kvStride, DType.Float32, _deviceId, _state.KScratch);
            var vRef = new TensorRef(seqLen, kvStride, DType.Float32, _deviceId, _state.VScratch);
            kvCache.Update(kRef, vRef, positions, kvSlot);

            int seqKv = kvCache.CurrentLength;
            TensorRef cachedK = kvCache.GetKeysRef(kvSlot);
            TensorRef cachedV = kvCache.GetValuesRef(kvSlot);

            _kernels.LaunchAttentionF32(_state.QScratch, cachedK.DataPointer, cachedV.DataPointer, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv, numHeads, numKvHeads, headDim,
                positionOffset: positions[0], slidingWindow: 0, streamH);
        }
        else
        {
            _kernels.LaunchAttentionF32(_state.QScratch, _state.KScratch, _state.VScratch, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqLen, numHeads, numKvHeads, headDim,
                positionOffset: 0, slidingWindow: 0, streamH);
        }

        Gemm(attn.OWeight, attn.OQt, _state.AttnOutput, _state.NormOutput, attn.OOutputDim, attn.OInputDim, seqLen);
    }
```

- [ ] **Step 2: `ForwardFfnBody`**

```csharp
    /// <summary>Squared-ReLU FFN sub-layer forward (no gate) — up -> relu² -> down, reads/writes
    /// <c>_state.NormOutput</c>. Matches <c>NemotronHTransformerModel.ForwardFfnBody</c> exactly.</summary>
    private void ForwardFfnBody(in DeviceFfn ffn, int seqLen, int hiddenSize)
    {
        int intermediateSize = ffn.UpOutputDim;
        nint streamH = _stream.Handle;

        Gemm(ffn.UpWeight, ffn.UpQt, _state.NormOutput, _state.FfnIntermediate, ffn.UpOutputDim, ffn.UpInputDim, seqLen);
        _kernels.LaunchReluSquaredInplaceF32(_state.FfnIntermediate, seqLen * intermediateSize, streamH);
        Gemm(ffn.DownWeight, ffn.DownQt, _state.FfnIntermediate, _state.NormOutput, ffn.DownOutputDim, ffn.DownInputDim, seqLen);
    }
```

- [ ] **Step 3: Build**

Run: `dotnet build src/DotLLM.Cuda`
Expected: builds cleanly (still missing `Forward` and its orchestration — Task 11).

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
git commit -m "feat(cuda): CudaNemotronHTransformerModel ForwardAttentionBody + ForwardFfnBody (#347) [4/5]"
```

---

### Task 11: `CudaNemotronHTransformerModel` — `Forward` orchestration + full `IModel`/recurrent-state API surface

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs` (append — this is the task that makes the file a complete, compiling `IModel` implementation)

**Interfaces:**
- Produces: `public ITensor Forward(ReadOnlySpan<int>, ReadOnlySpan<int>, int)`, `public ITensor Forward(ReadOnlySpan<int>, ReadOnlySpan<int>, int, IKvCache?)`, `public ITensor Forward(ReadOnlySpan<int>, ReadOnlySpan<int>, int, IKvCache?, ISsmState?)`, `public void ResetSequenceState()`, `public bool RequiresPerSequenceState => true`, `public bool SupportsThreadedSequenceState => true`, `public IRecurrentSequenceState? CreateSequenceState()`, `public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest>, int)`. Consumed by Task 12 (`CudaModelLoader`) and Task 13/14 (parity tests). This completes the class started in Tasks 7-10.
- Consumes: everything from Tasks 4-10 (this task is pure orchestration, wiring already-built pieces together) plus `CudaKernels.LaunchRmsNormF32`/`LaunchAddF32` (existing) and `UnmanagedTensor.Allocate`/`CudaDriverApi.cuMemcpyDtoH_v2` (existing, patterned after `CudaQwen3HybridDenseTransformerModel`'s identical D2H-logits tail, read in full for this plan — including its `_stream.Synchronize()`-before-`cuMemcpyDtoH_v2` requirement: "cuMemcpyDtoH_v2 does not implicitly wait for this model's non-default `_stream`").

**Reference (read in full for this plan):** `NemotronHTransformerModel.Forward`/`ResolveSsm`/`ForwardBatch`/`ResetSequenceState`/`CreateSequenceState` (CPU, lines 454-654 — the exact per-layer dispatch loop, the `_activeSsm`/`ResolveSsm` try/finally pattern, and `ForwardBatch`'s LoRA-rejection + null-state-rejection guards this task ports verbatim); `CudaQwen3HybridDenseTransformerModel`'s D2H logits tail (lines ~1232-1242) for the `_stream.Synchronize()` → `UnmanagedTensor.Allocate` → `cuMemcpyDtoH_v2` sequence.

- [ ] **Step 1: `UploadPositions` helper + `Forward` core**

```csharp
    /// <summary>Uploads the current call's host position span to <c>_state.PositionsDevice</c>
    /// (RoPE reads positions from device memory; <c>Embed</c> reads token ids from the host span
    /// directly, no upload needed there).</summary>
    private void UploadPositions(ReadOnlySpan<int> positions)
    {
        fixed (int* p = positions)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)p,
                (nuint)(positions.Length * sizeof(int))).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;

        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        _state.EnsureCapacity(seqLen);
        nint streamH = _stream.Handle;

        UploadPositions(positions);
        Embed(tokenIds, _state.HiddenState, hiddenSize);

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];

            // Save residual snapshot, then pre-sublayer RMSNorm into NormOutput — shared by all
            // three sub-layer kinds.
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
                (nuint)((long)seqLen * hiddenSize * sizeof(float)), streamH).ThrowOnError();
            _kernels.LaunchRmsNormF32(_state.HiddenState, lw.AttnNormWeightDevice, _state.NormOutput,
                hiddenSize, eps, seqLen, streamH);

            switch (lw.Kind)
            {
                case HybridLayerKind.Ffn:
                    ForwardFfnBody(lw.Ffn!.Value, seqLen, hiddenSize);
                    break;
                case HybridLayerKind.Attention:
                    ForwardAttentionBody(lw.Attention!.Value, layer, seqLen, positions,
                        numHeads, numKvHeads, headDim, kvCache);
                    break;
                case HybridLayerKind.Ssm:
                    ForwardSsmBody(lw.Ssm!.Value, layer, seqLen, hiddenSize, eps);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {lw.Kind} at layer {layer}.");
            }

            // Residual add: HiddenState = Residual + NormOutput (NormOutput holds this
            // sub-layer's output — every ForwardXBody writes back into NormOutput).
            _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
                seqLen * hiddenSize, streamH);
        }

        // Final RMSNorm over every row (matches the CPU host, which normalizes all seqLen rows
        // before the lm_head GEMM — unlike Qwen3HybridDense's optional lastTokenLogitsOnly this
        // model does not need since NemotronH's realistic vocab sizes don't hit the VRAM ceiling
        // that optimization exists for).
        _kernels.LaunchRmsNormF32(_state.HiddenState, _outputNormDevice, _state.HiddenState,
            hiddenSize, eps, seqLen, streamH);

        Gemm(_outputDevice, _outputQt, _state.HiddenState, _state.Logits,
             _outputOutputDim, _outputInputDim, seqLen);

        // cuMemcpyDtoH_v2 does not implicitly wait for this model's non-default _stream —
        // synchronize first (mirrors CudaQwen3HybridDenseTransformerModel's identical D2H tail).
        _stream.Synchronize();

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.Logits,
            (nuint)((long)seqLen * vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }
```

- [ ] **Step 2: Per-sequence SSM-state threading — `Forward(..., ISsmState?)`, `ResolveSsm`, `ResetSequenceState`, `CreateSequenceState`**

```csharp
    /// <summary>
    /// Forward with a caller-supplied per-sequence SSM state (the per-token recurrent state the
    /// continuous-batch scheduler threads so concurrent sequences don't share the model-owned
    /// default). Null falls back to the model-owned <c>_ssmCache</c> (single-sequence behaviour).
    /// Attention layers use <paramref name="kvCache"/> as usual. Mirrors
    /// <c>NemotronHTransformerModel.Forward(..., ISsmState?)</c> exactly.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ISsmState? ssmState)
    {
        CudaNemotronHSsmStateCache? prev = _activeSsm;
        _activeSsm = ResolveSsm(ssmState);
        try { return Forward(tokenIds, positions, deviceId, kvCache); }
        finally { _activeSsm = prev; }
    }

    private CudaNemotronHSsmStateCache? ResolveSsm(ISsmState? ssmState)
    {
        if (ssmState is null) return null; // use _ssmCache
        if (ssmState is CudaNemotronHSsmStateCache cache)
        {
            if (cache.NumSsmLayers != _numSsmLayers)
                throw new ArgumentException(
                    $"SsmState covers {cache.NumSsmLayers} SSM layers but this model has {_numSsmLayers}.",
                    nameof(ssmState));
            return cache;
        }
        throw new ArgumentException(
            $"CudaNemotronHTransformerModel requires a {nameof(CudaNemotronHSsmStateCache)} for its " +
            $"SSM state; got {ssmState.GetType().Name}.",
            nameof(ssmState));
    }

    /// <inheritdoc/>
    /// <remarks>Re-zeroes the model-owned SSM cache (conv history + hidden state) used by every
    /// forward that does not carry a caller-supplied <see cref="ISsmState"/>. Callers that treat
    /// each forward as an independent sequence (perplexity windows, growing-context reprefill
    /// parity tests — see Task 14) must call this between sequences.</remarks>
    public void ResetSequenceState() => _ssmCache.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => new CudaNemotronHSsmStateCache(_ssm, _numSsmLayers);
```

- [ ] **Step 3: `ForwardBatch`**

```csharp
    /// <summary>
    /// Batched forward across sequences. NemotronH SSM state is per-token recurrent, so this
    /// threads each request's per-seq <see cref="SequenceForwardRequest.SsmState"/> through a
    /// per-sequence <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, ISsmState?)"/>
    /// call (no cross-sequence fusion). For 2+ requests every entry must supply a per-seq
    /// <c>SsmState</c> (a null would silently share the model-owned default and corrupt
    /// concurrent decode). LoRA adapters are not supported. Mirrors
    /// <c>NemotronHTransformerModel.ForwardBatch</c> exactly.
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "CudaNemotronHTransformerModel.ForwardBatch does not support LoRA adapters.");
        }

        if (requests.Count >= 2)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                if (requests[i].SsmState is null)
                    throw new ArgumentException(
                        $"CudaNemotronHTransformerModel.ForwardBatch with {requests.Count} requests requires " +
                        $"every request to supply a per-seq SsmState; request[{i}] has none.",
                        nameof(requests));
            }
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.SsmState);
        }
        return results;
    }
}
```

(The closing `}` above closes the `CudaNemotronHTransformerModel` class itself — this is the last member added across Tasks 7-11.)

- [ ] **Step 4: Build**

Run: `dotnet build src/DotLLM.Cuda`
Expected: builds cleanly with ZERO errors — this is the first task where the whole file must compile end-to-end (Tasks 7-10 were intentionally incomplete WIP states).

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
git commit -m "feat(cuda): CudaNemotronHTransformerModel Forward orchestration + IModel surface (#347) [5/5]"
```

---

### Task 12: Wire into `CudaModelLoader.CreateFromGguf`, remove the guard

**Files:**
- Modify: `src/DotLLM.Cuda/CudaModelLoader.cs`
- Modify: `tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs`

**Interfaces:**
- Consumes: `CudaNemotronHTransformerModel.LoadFromGguf` (Task 7), `CudaNemotronHTransformerModel.CreateKvCache` (Task 7).
- Produces: nothing new — this task only rewires an existing dispatch point.

This is the literal acceptance-criterion edit from the issue: *"`CudaNemotronHTransformerModel.LoadFromGguf` + forward pass, wired into `CudaModelLoader.CreateFromGguf`'s switch (replacing the new guard)"*.

- [ ] **Step 1: Replace the `Architecture.NemotronH` guard case**

In `src/DotLLM.Cuda/CudaModelLoader.cs`, the current switch (read in full for this plan) has:

```csharp
            case Architecture.NemotronH:
                throw new NotSupportedException(
                    "CUDA has no dedicated loader for NemotronH yet — its Mamba2 SSM layers "
                    + "(A, dt_bias, conv1d, etc.) are not tensor-compatible with the generic "
                    + "CudaTransformerModel this would otherwise silently fall through to. Use "
                    + "the CPU or Vulkan backend for NemotronH checkpoints.");
```

Replace it with (matching the `Qwen3HybridDense`/`Qwen3MoeHybrid` cases immediately above it in the same switch — same shape, same `KvCacheFactory` pattern):

```csharp
            case Architecture.NemotronH:
            {
                var nemotronH = Architectures.CudaNemotronHTransformerModel
                    .LoadFromGguf(gguf, config, deviceId, ptxDir);
                return (nemotronH, size => nemotronH.CreateKvCache(size));
            }
```

Leave the `Architecture.Mamba3` and `Architecture.GptOss` guard cases in the same switch completely untouched — this task's scope is NemotronH only (Mamba3's own CUDA host is sibling issue #346, out of scope here per this plan's Global Constraints on shared-file hotspots).

- [ ] **Step 2: Remove the now-superseded guard test case**

In `tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs`, the theory currently has:

```csharp
    [SkippableTheory]
    [InlineData(Architecture.Mamba3)]
    [InlineData(Architecture.NemotronH)]
    [InlineData(Architecture.GptOss)]
    public void CreateFromGguf_UnsupportedArchitecture_ThrowsNotSupportedInsteadOfSilentFallthrough(
```

Remove only the `[InlineData(Architecture.NemotronH)]` line — `Mamba3` and `GptOss` remain guarded (out of this plan's scope) and must keep their coverage:

```csharp
    [SkippableTheory]
    [InlineData(Architecture.Mamba3)]
    [InlineData(Architecture.GptOss)]
    public void CreateFromGguf_UnsupportedArchitecture_ThrowsNotSupportedInsteadOfSilentFallthrough(
```

- [ ] **Step 3: Build**

Run: `dotnet build src/DotLLM.Cuda tests/DotLLM.Tests.Integration`
Expected: builds cleanly.

- [ ] **Step 4: Run the guard test to confirm NemotronH is no longer covered by it (and the other two still are)**

Run: `dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~CudaUnsupportedArchitectureGuardTests"`
Expected: PASS for `Mamba3`/`GptOss` (or `Skipped` without a GPU + fixture); no `NemotronH` case present.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cuda/CudaModelLoader.cs tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs
git commit -m "feat(cuda): wire CudaNemotronHTransformerModel into CudaModelLoader.CreateFromGguf (#347)"
```

---

### Task 13: Synthetic-fixture CPU-vs-CUDA forward parity test (cacheless + cached prefill/decode)

**Files:**
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHTransformerModelForwardTests.cs`

**Interfaces:**
- Consumes: `CudaNemotronHTransformerModel.BuildFromPrebuiltWeights` (Task 8), `CudaNemotronHTransformerModel.CreateKvCache`/`CreateSequenceState` (Task 11), `NemotronHTransformerModel.BuildFromPrebuiltWeights` (CPU, already exists), `DotLLM.Engine.KvCache.SimpleKvCache` (already exists — generic CPU `IKvCache`, confirmed present at `src/DotLLM.Engine/KvCache/SimpleKvCache.cs`, constructor `SimpleKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)`).

This is the primary, fast, CI-realistic correctness gate for this whole plan — it needs no multi-GB real GGUF, runs on any CUDA-capable CI box, and (per the second half of this task) is the ONLY test in this plan that actually exercises `CudaNemotronHKvCache` and `CudaNemotronHSsmStateCache` together end-to-end through cached prefill + decode, the way a real generation loop drives the model.

**Reference (read in full for this plan):** `tests/DotLLM.Tests.Unit/Vulkan/VulkanNemotronHTransformerModelForwardTests.cs` — Part 1 below is a near-verbatim port of that file (same `NemotronHFixtureBuilder`, same 6 test cases: 3 layer-kind combinations × {F32, Q8_0}), swapping the Vulkan model construction calls for CUDA ones. Part 2 is new (Vulkan's file has no cached-prefill-then-decode case).

- [ ] **Step 1: Write the failing test file — Part 1 (ported cacheless fixture + 6 test cases)**

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHTransformerModelForwardTests.cs
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end parity test for the CUDA NemotronH hybrid forward pass against the CPU reference.
/// Part 1 (this class's fixture + 6 cacheless test cases) is a near-verbatim port of
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanNemotronHTransformerModelForwardTests"/> — same
/// synthetic-model construction, same tolerance. Part 2 (bottom of the file) adds cached
/// prefill + decode coverage that exercises <see cref="CudaNemotronHKvCache"/> and
/// <see cref="CudaNemotronHSsmStateCache"/>, which the cacheless cases never touch.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaNemotronHTransformerModelForwardTests
{
    private const int HiddenSize = 16;
    private const int VocabSize = 8;
    private const int HeadDim = 8;
    private const int NumHeads = 2;
    private const int NumKvHeads = 2;
    private const int IntermediateSize = 24;
    private const int DInner = 16;
    private const int DConv = 4;
    private const int DState = 8;
    private const int NGroup = 2;
    private const int NHead = 2;
    private const int MaxSeqLen = 16;

    // Q8_0 quant-fixture dimensions — every contraction axis is a multiple of 32.
    private const int Q8HiddenSize = 32;
    private const int Q8IntermediateSize = 32;
    private const int Q8DInner = 32;
    private const int Q8NHead = 4;
    private const int Q8NumAttentionHeads = 4;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableFact]
    public void Forward_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 3, seed: 7);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 13);
    }

    [SkippableFact]
    public void Forward_Q8_0_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 142, quantize: true);
    }

    [SkippableFact]
    public void Forward_Q8_0_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 3, seed: 107, quantize: true);
    }

    [SkippableFact]
    public void Forward_Q8_0_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 113, quantize: true);
    }

    private void AssertCudaMatchesCpu(
        HybridLayerKind[] layerKinds, int seqLen, int seed, bool quantize = false)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed, quantize);
        var config = fixture.Config;

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        float[] cpuLogits;
        using (var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize))
        {
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] cudaLogits;
        using (var model = CudaNemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            deviceId: 0, ptxDir))
        {
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(seqLen, logits.Shape[0]);
            Assert.Equal(vocabSize, logits.Shape[1]);
            cudaLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < vocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * vocabSize + c];
            float cuda = cudaLogits[lastRow * vocabSize + c];
            float diff = MathF.Abs(cpu - cuda);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"layers={string.Join(',', layerKinds)}, seqLen={seqLen}, quant={quantize}, col={c}: " +
                $"cpu={cpu:F6} vs cuda={cuda:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>Owns a randomly-generated NemotronH "model" in unmanaged memory. Verbatim port of
    /// <c>VulkanNemotronHTransformerModelForwardTests.NemotronHFixtureBuilder</c> — see that class
    /// for the full design rationale (identical bytes fed to both backends, Q8_0-friendly dims in
    /// quant mode, F32-only token embedding).</summary>
    private sealed unsafe class NemotronHFixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;
        public QuantizationType OutputQuantType;

        private int _hiddenSize, _intermediateSize, _dInner, _nHead, _numAttentionHeads;
        private bool _quantize;

        public static NemotronHFixtureBuilder Build(HybridLayerKind[] layerKinds, int seed, bool quantize = false)
        {
            var b = new NemotronHFixtureBuilder();
            b.BuildInternal(layerKinds, seed, quantize);
            return b;
        }

        private void BuildInternal(HybridLayerKind[] layerKinds, int seed, bool quantize)
        {
            int numLayers = layerKinds.Length;
            var rng = new Random(seed);

            _quantize = quantize;
            _hiddenSize = quantize ? Q8HiddenSize : HiddenSize;
            _intermediateSize = quantize ? Q8IntermediateSize : IntermediateSize;
            _dInner = quantize ? Q8DInner : DInner;
            _nHead = quantize ? Q8NHead : NHead;
            _numAttentionHeads = quantize ? Q8NumAttentionHeads : NumHeads;
            OutputQuantType = quantize ? QuantizationType.Q8_0 : QuantizationType.F32;

            var headCountKv = new int[numLayers];
            var ffnLength = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                headCountKv[i] = layerKinds[i] == HybridLayerKind.Attention ? NumKvHeads : 0;
                ffnLength[i] = layerKinds[i] == HybridLayerKind.Ffn ? _intermediateSize : 0;
            }

            var layout = new HybridLayerLayout
            {
                LayerKind = layerKinds, HeadCountKv = headCountKv, FeedForwardLength = ffnLength,
            };
            var ssmConfig = new MambaSsmConfig(
                DConv: DConv, DInner: _dInner, DState: DState, NGroup: NGroup, NHead: _nHead);

            Config = new ModelConfig
            {
                Architecture = Architecture.NemotronH,
                VocabSize = VocabSize,
                HiddenSize = _hiddenSize,
                IntermediateSize = _intermediateSize,
                NumLayers = numLayers,
                NumAttentionHeads = _numAttentionHeads,
                NumKvHeads = NumKvHeads,
                HeadDim = HeadDim,
                MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                HybridLayout = layout,
                SsmConfig = ssmConfig,
                ChatTemplate = null,
            };

            TokenEmbedPtr = AllocAndFillUniform(VocabSize * _hiddenSize, rng, amplitude: 0.05f);
            OutputNormWeight = FillNormVec(_hiddenSize, rng);
            OutputWeightPtr = AllocProjection(VocabSize, _hiddenSize, rng, OutputQuantType);

            Layers = new NemotronHLayerWeights[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                float[] attnNorm = FillNormVec(_hiddenSize, rng);
                Layers[i] = layerKinds[i] switch
                {
                    HybridLayerKind.Ssm => new NemotronHLayerWeights { AttnNormWeight = attnNorm, Ssm = BuildSsm(rng, ssmConfig) },
                    HybridLayerKind.Attention => new NemotronHLayerWeights { AttnNormWeight = attnNorm, Attention = BuildAttn(rng, NumKvHeads) },
                    HybridLayerKind.Ffn => new NemotronHLayerWeights { AttnNormWeight = attnNorm, Ffn = BuildFfn(rng, _intermediateSize) },
                    _ => throw new InvalidOperationException(),
                };
            }
        }

        private NemotronHSsmWeights BuildSsm(Random rng, MambaSsmConfig ssm)
        {
            int convDim = ssm.ConvDim;
            int inProjDim = ssm.InputProjectionDim;
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHSsmWeights
            {
                InWeight = AllocProjection(inProjDim, _hiddenSize, rng, qt), InQuantType = qt,
                InInputDim = _hiddenSize, InOutputDim = inProjDim,
                Conv1dWeight = FillRandom(ssm.DConv * convDim, rng, 0.1f),
                Conv1dBias = FillRandom(convDim, rng, 0.1f),
                A = NegativeRandom(ssm.NHead, rng),
                D = FillRandom(ssm.NHead, rng, 0.1f),
                DtBias = FillRandom(ssm.NHead, rng, 0.1f),
                NormWeight = FillNormVec(ssm.DInner, rng),
                OutWeight = AllocProjection(_hiddenSize, ssm.DInner, rng, qt), OutQuantType = qt,
                OutInputDim = ssm.DInner, OutOutputDim = _hiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng, int numKvHeads)
        {
            int qOut = _numAttentionHeads * HeadDim;
            int kvOut = numKvHeads * HeadDim;
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHAttentionWeights
            {
                QWeight = AllocProjection(qOut, _hiddenSize, rng, qt), QQuantType = qt, QInputDim = _hiddenSize, QOutputDim = qOut,
                KWeight = AllocProjection(kvOut, _hiddenSize, rng, qt), KQuantType = qt, KInputDim = _hiddenSize, KOutputDim = kvOut,
                VWeight = AllocProjection(kvOut, _hiddenSize, rng, qt), VQuantType = qt, VInputDim = _hiddenSize, VOutputDim = kvOut,
                OWeight = AllocProjection(_hiddenSize, qOut, rng, qt), OQuantType = qt, OInputDim = qOut, OOutputDim = _hiddenSize,
                NumKvHeads = numKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng, int intermediateSize)
        {
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHFfnWeights
            {
                UpWeight = AllocProjection(intermediateSize, _hiddenSize, rng, qt), UpQuantType = qt, UpInputDim = _hiddenSize, UpOutputDim = intermediateSize,
                DownWeight = AllocProjection(_hiddenSize, intermediateSize, rng, qt), DownQuantType = qt, DownInputDim = intermediateSize, DownOutputDim = _hiddenSize,
                IntermediateSize = intermediateSize,
            };
        }

        private nint AllocProjection(int outputDim, int inputDim, Random rng, QuantizationType qt)
        {
            if (qt == QuantizationType.F32) return AllocAndFillUniform(outputDim * inputDim, rng, amplitude: 0.05f);
            if (qt != QuantizationType.Q8_0) throw new NotSupportedException($"AllocProjection only supports F32/Q8_0, got {qt}.");
            if ((inputDim % 32) != 0) throw new InvalidOperationException($"Q8_0 requires inputDim multiple of 32 (got {inputDim}).");

            int rowBytes = (inputDim / 32) * 34;
            long totalBytes = (long)rowBytes * outputDim;
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
            _allocs.Add(dst);

            float[] rowScratch = new float[inputDim];
            for (int row = 0; row < outputDim; row++)
            {
                for (int j = 0; j < inputDim; j++) rowScratch[j] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
                fixed (float* srcPtr = rowScratch)
                {
                    byte* rowDst = (byte*)dst + (long)row * rowBytes;
                    MatMul.QuantizeF32ToQ8_0(srcPtr, rowDst, inputDim);
                }
            }
            return dst;
        }

        private nint AllocAndFillUniform(int count, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++) dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return ptr;
        }

        private static float[] FillRandom(int count, Random rng, float amplitude)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return arr;
        }

        private static float[] FillNormVec(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        private static float[] NegativeRandom(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = -((float)rng.NextDouble() * 0.5f + 0.1f);
            return arr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
```

- [ ] **Step 2: Run and confirm Part 1 fails (nothing built yet — expected: compile error, `CudaNemotronHTransformerModel` etc. already exist from Tasks 1-12, so this should actually PASS once the file compiles; if it fails on real assertion mismatch, that is a genuine bug to fix, not an expected-red step)**

Run: `dotnet build tests/DotLLM.Tests.Unit`
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaNemotronHTransformerModelForwardTests"`
Expected: PASS for all 6 cases (or `Skipped` without a CUDA GPU) — since Tasks 1-12 already implemented the model, this task is verifying, not red-then-green from scratch.

- [ ] **Step 3: Add Part 2 — cached prefill + decode, exercising `CudaNemotronHKvCache` + `CudaNemotronHSsmStateCache` together**

Append this test method to the `CudaNemotronHTransformerModelForwardTests` class (before the closing brace, after the six `Forward_*` methods and `AssertCudaMatchesCpu`/`CopyLogits`, before `NemotronHFixtureBuilder`):

```csharp
    /// <summary>
    /// Runs a real cached prefill (3 tokens) followed by 3 cached decode steps on BOTH backends,
    /// using a real <see cref="IKvCache"/> (attention) and the model's own recurrent SSM state
    /// (not the cacheless <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/>
    /// overload every other test in this file uses). This is the only test in this plan that
    /// exercises <see cref="CudaNemotronHKvCache"/> and <see cref="CudaNemotronHSsmStateCache"/>
    /// end-to-end, together, the way a real generation loop drives the model.
    /// </summary>
    [SkippableFact]
    public void Forward_CachedPrefillThenDecode_MatchesCpuReference()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        var kinds = new[]
        {
            HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn, HybridLayerKind.Attention,
        };
        using var fixture = NemotronHFixtureBuilder.Build(kinds, seed: 271);
        var config = fixture.Config;
        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;
        int attentionLayerCount = kinds.Count(k => k == HybridLayerKind.Attention);

        int[] promptIds = { 1, 2, 3 };
        int[] promptPositions = { 0, 1, 2 };
        int[] decodeTokens = { 4, 5, 6 };

        var cpuSteps = new List<float[]>();
        using (var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize))
        using (var kvCache = new SimpleKvCache(attentionLayerCount, config.NumKvHeads, config.HeadDim, MaxSeqLen))
        {
            model.ResetSequenceState();
            using (ITensor logits = model.Forward(promptIds, promptPositions, deviceId: -1, kvCache))
                cpuSteps.Add(CopyLogits(logits));
            int pos = promptIds.Length;
            foreach (int tok in decodeTokens)
            {
                using ITensor logits = model.Forward(new[] { tok }, new[] { pos }, deviceId: -1, kvCache);
                cpuSteps.Add(CopyLogits(logits));
                pos++;
            }
        }

        var cudaSteps = new List<float[]>();
        using (var model = CudaNemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            deviceId: 0, ptxDir))
        using (var kvCache = model.CreateKvCache(MaxSeqLen))
        {
            model.ResetSequenceState();
            using (ITensor logits = model.Forward(promptIds, promptPositions, deviceId: -1, kvCache))
                cudaSteps.Add(CopyLogits(logits));
            int pos = promptIds.Length;
            foreach (int tok in decodeTokens)
            {
                using ITensor logits = model.Forward(new[] { tok }, new[] { pos }, deviceId: -1, kvCache);
                cudaSteps.Add(CopyLogits(logits));
                pos++;
            }
        }

        Assert.Equal(cpuSteps.Count, cudaSteps.Count);
        for (int step = 0; step < cpuSteps.Count; step++)
        {
            int rows = step == 0 ? promptIds.Length : 1;
            int lastRow = rows - 1;
            for (int c = 0; c < vocabSize; c++)
            {
                float cpu = cpuSteps[step][lastRow * vocabSize + c];
                float cuda = cudaSteps[step][lastRow * vocabSize + c];
                float diff = MathF.Abs(cpu - cuda);
                float bar = AbsTol + RelTol * MathF.Abs(cpu);
                Assert.True(diff <= bar,
                    $"step={step} col={c}: cpu={cpu:F6} vs cuda={cuda:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }
    }
```

Add `using System.Linq;` to the file's using block (for `.Count(...)`).

- [ ] **Step 4: Build and run**

Run: `dotnet build tests/DotLLM.Tests.Unit`
Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CudaNemotronHTransformerModelForwardTests"`
Expected: all 7 tests PASS (or `Skipped`).

- [ ] **Step 5: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Cuda/CudaNemotronHTransformerModelForwardTests.cs
git commit -m "test(cuda): synthetic-fixture NemotronH CPU-vs-CUDA parity, cacheless + cached prefill/decode (#347)"
```

---

### Task 14: Real-GGUF CPU-vs-CUDA parity test (gated on `DOTLLM_NEMOTRON_H_GGUF`) + docs update

**Files:**
- Modify: `tests/DotLLM.Tests.Integration/Cuda/RealGgufCudaParityTests.cs` (additive — new test method + private driver, does NOT touch the existing shared `RunGgufParityTestCore`)
- Modify: `docs/SUPPORTED_MODELS.md`
- Modify: `docs/ROADMAP.md` (only if NemotronH CUDA support is tracked there as a step — see Step 3)

**Interfaces:**
- Consumes: `CudaModelLoader.CreateFromGguf` (Task 12, now dispatches NemotronH for real), `NemotronHTransformerModel.LoadFromGguf` (CPU, already exists).

**Design note — why this is a NEW self-contained method, not an addition to `RunGgufParityTestCore`:** the acceptance criterion is real-model CPU-vs-CUDA parity for **prefill AND decode**. The existing shared `RunGgufParityTestCore` (used by 8 other architectures, including sibling-plan territory) does growing-context reprefill with **no KV cache and no `ResetSequenceState()` call** — fine for stateless attention-only architectures, but NemotronH's SSM layers mutate `_ssmCache` on every `Forward` call (issue #261's exact trap, called out in `NemotronHTransformerModel.ResetSequenceState`'s own doc comment: *"Callers that treat each forward as an independent sequence (perplexity windows) must call this between sequences"*). Reusing the shared driver unmodified for NemotronH would silently accumulate SSM state across the growing-context loop's repeated from-scratch recomputations, corrupting BOTH sides identically-but-coincidentally (not a real regression signal) rather than testing genuine prefill-then-decode parity. This plan's driver instead does a **real, non-reprefilled prefill followed by real KV/SSM-cached decode steps** — closer to the issue's literal "prefill + decode" wording, and it explicitly calls `ResetSequenceState()` on both models before starting (defensive — not strictly required for a from-scratch model, but matches the discipline every other stateful test in this codebase uses, and protects the test if construction order ever changes).

- [ ] **Step 1: Add the NemotronH real-GGUF parity test method**

Append to the `RealGgufCudaParityTests` class (`tests/DotLLM.Tests.Integration/Cuda/RealGgufCudaParityTests.cs`), after the existing `Qwen36A3B_IQ2_XXS_CudaForward_MatchesCpuReference` test and before the `// Driver` region comment:

```csharp
    // ────────────────────────────────────────────────────────────────────
    // NemotronH (NVIDIA Nemotron-3-Nano-4B or similar) — hybrid Mamba2-SSM +
    // GQA-attention coverage. Uses DOTLLM_NEMOTRON_H_GGUF (NOT
    // TestFixtureResolver — no known HF org/repo/filename triple is
    // registered for this model; mirrors NemotronHTextGeneratorTests'
    // simple env-var + File.Exists pattern). Real cached prefill + decode
    // (not the shared driver's KV-cache-less growing-context reprefill) —
    // see this task's design note for why.
    // ────────────────────────────────────────────────────────────────────

    private const string NemotronHModelPathEnvVar = "DOTLLM_NEMOTRON_H_GGUF";

    [SkippableFact]
    public void NemotronH_CudaForward_MatchesCpuReference_PrefillAndDecode()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? path = Environment.GetEnvironmentVariable(NemotronHModelPathEnvVar);
        Skip.If(string.IsNullOrWhiteSpace(path) || !File.Exists(path),
            $"Set {NemotronHModelPathEnvVar} to a Nemotron-H GGUF to run this test.");

        string ptxDir = ResolvePtxDir();
        Skip.If(!Directory.Exists(ptxDir), $"CUDA PTX directory not found (resolved: {ptxDir}).");

        bool prevAllowExpansion = CudaKernels.AllowQuantExpansion;
        CudaKernels.AllowQuantExpansion = true;
        try
        {
            RunNemotronHParityTest(path!, ptxDir);
        }
        finally
        {
            CudaKernels.AllowQuantExpansion = prevAllowExpansion;
        }
    }

    private void RunNemotronHParityTest(string path, string ptxDir)
    {
        _output.WriteLine($"[NemotronH] gguf: {path}");

        using var cpuGguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        Assert.Equal(Architecture.NemotronH, config.Architecture);
        Assert.NotNull(config.HybridLayout);
        Assert.NotNull(config.SsmConfig);

        using var cpuModel = NemotronHTransformerModel.LoadFromGguf(cpuGguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);

        using var cudaGguf = GgufFile.Open(path);
        IModel? cudaModel = null;
        try
        {
            var (model, kvCacheFactory) = CudaModelLoader.CreateFromGguf(cudaGguf, config, deviceId: 0, ptxDir);
            cudaModel = model;

            int attentionLayerCount = ((CudaNemotronHTransformerModel)model).AttentionLayerCount;

            int[] promptIds = tokenizer.Encode("The capital of France is").ToArray();
            Assert.NotEmpty(promptIds);
            _output.WriteLine($"[NemotronH] prompt tokens: [{string.Join(',', promptIds)}]");

            int[] promptPositions = new int[promptIds.Length];
            for (int i = 0; i < promptPositions.Length; i++) promptPositions[i] = i;

            cpuModel.ResetSequenceState();
            cudaModel.ResetSequenceState();

            using var cpuKv = new DotLLM.Engine.KvCache.SimpleKvCache(
                attentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var cudaKv = kvCacheFactory(config.MaxSequenceLength);

            int vocab = config.VocabSize;
            int strictArgmaxMatches = 0;
            int stepsChecked = 0;
            const int DecodeSteps = 8;
            const int StrictArgmaxFloor = 5;

            float[] cpuLogits = RunForwardLastRow(cpuModel, promptIds, promptPositions, vocab, deviceId: -1, cpuKv);
            float[] cudaLogits = RunForwardLastRow(cudaModel, promptIds, promptPositions, vocab, deviceId: -1, cudaKv);
            AssertLogitsMatch(cpuLogits, cudaLogits, step: 0, "NemotronH");
            int cpuArgmax0 = Argmax(cpuLogits), cudaArgmax0 = Argmax(cudaLogits);
            if (cpuArgmax0 == cudaArgmax0) strictArgmaxMatches++;
            stepsChecked++;
            _output.WriteLine($"[NemotronH] step 0 (prefill): cpu_argmax={cpuArgmax0} cuda_argmax={cudaArgmax0}" +
                (cpuArgmax0 == cudaArgmax0 ? " [match]" : " [diff]"));

            int pos = promptIds.Length;
            int nextCpuTok = cpuArgmax0, nextCudaTok = cudaArgmax0;
            for (int step = 1; step <= DecodeSteps; step++)
            {
                float[] cpuStep = RunForwardLastRow(cpuModel, new[] { nextCpuTok }, new[] { pos }, vocab, deviceId: -1, cpuKv);
                float[] cudaStep = RunForwardLastRow(cudaModel, new[] { nextCudaTok }, new[] { pos }, vocab, deviceId: -1, cudaKv);
                AssertLogitsMatch(cpuStep, cudaStep, step, "NemotronH");
                int cpuArgmax = Argmax(cpuStep), cudaArgmax = Argmax(cudaStep);
                if (cpuArgmax == cudaArgmax) strictArgmaxMatches++;
                stepsChecked++;
                _output.WriteLine($"[NemotronH] step {step}: cpu_argmax={cpuArgmax} cuda_argmax={cudaArgmax}" +
                    (cpuArgmax == cudaArgmax ? " [match]" : " [diff]"));
                nextCpuTok = cpuArgmax;
                nextCudaTok = cudaArgmax;
                pos++;
            }

            Assert.True(strictArgmaxMatches >= StrictArgmaxFloor,
                $"[NemotronH] strict argmax match floor {StrictArgmaxFloor}/{stepsChecked} not met: got {strictArgmaxMatches}/{stepsChecked}.");
            _output.WriteLine($"[NemotronH] strict argmax matches: {strictArgmaxMatches}/{stepsChecked}");
        }
        finally
        {
            cudaModel?.Dispose();
            cudaGguf.Dispose();
        }
    }

    private static unsafe float[] RunForwardLastRow(
        IModel model, int[] tokenIds, int[] positions, int vocab, int deviceId, DotLLM.Core.Attention.IKvCache kvCache)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId, kvCache);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }
```

Add these `using` statements to the top of the file if not already present: `using DotLLM.Cuda.Architectures;` (for the `CudaNemotronHTransformerModel` cast) and `using DotLLM.Models.Architectures;` (for `NemotronHTransformerModel`) — check the existing `using` block first; `DotLLM.Models.Architectures` is likely already absent (the file currently only needs `TransformerModel`/`CudaTransformerModel`, both in namespaces already imported) but `NemotronHTransformerModel` lives in `DotLLM.Models.Architectures` specifically, so add it if missing.

- [ ] **Step 2: Build and run**

Run: `dotnet build tests/DotLLM.Tests.Integration`
Run: `dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~NemotronH_CudaForward_MatchesCpuReference_PrefillAndDecode"`
Expected: `Skipped` (no `DOTLLM_NEMOTRON_H_GGUF` set in CI) locally without the env var, or PASS if a real Nemotron-H GGUF (e.g. NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf) is available and `DOTLLM_NEMOTRON_H_GGUF` points at it. **Do not block completion of this plan on acquiring that checkpoint** — the graceful-skip behavior is the acceptance bar for CI; a human with the checkpoint locally should run it once before merging to confirm real numeric parity, per this plan's biggest open risk (see the final report).

- [ ] **Step 3: Update `docs/SUPPORTED_MODELS.md`**

In the NemotronH row (the `| NVIDIA Nemotron-H | ... |` line), the current "verified" cell text ends with:

```
**Vulkan**: full forward path landed (`VulkanNemotronHTransformerModel` at `4abe4c2`) including SSM scan, attention, FFN, hybrid layer dispatch, plus Q8_0 / Q4_K / Q5_K / Q6_K / F16 / BF16 projection upload via the Phase 1 + Phase 8 Vulkan kernels.
```

Append a new sentence immediately after it (same cell, same row — do not restructure the table):

```
**CUDA**: full forward path landed (`CudaNemotronHTransformerModel`, issue #347) — dedicated `mamba2_selective_scan_f32`/`group_rmsnorm_f32`/`relu_squared_inplace_f32` CUDA kernels plus the existing generic `conv1d_causal_f32`, wired into `CudaModelLoader.CreateFromGguf`; synthetic-fixture CPU-vs-CUDA parity covers cacheless and cached-prefill/decode forward (`CudaNemotronHTransformerModelForwardTests`), real-GGUF parity is `RealGgufCudaParityTests.NemotronH_CudaForward_MatchesCpuReference_PrefillAndDecode` (gated on `DOTLLM_NEMOTRON_H_GGUF`, same as the CPU/Vulkan smoke tests).
```

Also update the last cell of the same row (the "Notes / gaps" column, currently ending `...Dedicated NemotronHTransformerModel (CPU) and VulkanNemotronHTransformerModel (Vulkan) with a hybrid Mamba-2 / attention forward. |`) by appending `CudaNemotronHTransformerModel (CUDA)` to that sentence:

```
Dedicated `NemotronHTransformerModel` (CPU), `VulkanNemotronHTransformerModel` (Vulkan), and `CudaNemotronHTransformerModel` (CUDA) with a hybrid Mamba-2 / attention forward.
```

- [ ] **Step 4: Check `docs/ROADMAP.md` for a NemotronH CUDA step**

Run: `grep -n "NemotronH" docs/ROADMAP.md`. If a roadmap line exists that specifically lists "NemotronH CUDA host" or similar as a pending step, mark it `:white_check_mark:` per CLAUDE.md's Development Workflow item 7. If no such line exists (this is an interim-bugfix-turned-feature issue, not necessarily a roadmap-tracked step), skip this step — do not invent a roadmap entry that wasn't already there.

- [ ] **Step 5: Commit**

```bash
git add tests/DotLLM.Tests.Integration/Cuda/RealGgufCudaParityTests.cs docs/SUPPORTED_MODELS.md
# add docs/ROADMAP.md too if Step 4 found and updated an entry
git commit -m "test(cuda): real-GGUF NemotronH CPU-vs-CUDA parity (prefill+decode), docs update (#347)"
```

---

## Self-Review

**1. Spec coverage:**
- "`CudaNemotronHTransformerModel.LoadFromGguf` + forward pass, wired into `CudaModelLoader.CreateFromGguf`'s switch (replacing the new guard)" — Tasks 7-12.
- "Real-model CPU vs CUDA parity test (prefill + decode) on a real NemotronH GGUF" — Task 14.
- "Consider sharing SSM-scan kernel infrastructure with #346 (Mamba3 CUDA)... verify against the CPU reference implementations first" — explicitly rejected in Global Constraints and Task 1, backed by an independent re-read of `Mamba2SelectiveScan.cs` in this plan's own research (not just trusting the issue's summary).
- HybridLayerLayout/Qwen3HybridDense-precedent research directions — confirmed already-generic (`TryExtractHybridLayout`, `TryExtractSsmConfig` in `GgufModelConfigExtractor.cs` need no changes) and cited in Task 7's design note.
- Vulkan PR #310 structural precedent — Tasks 1-11 cite `VulkanNemotronHTransformerModel`/`VulkanNemotronHWeights` throughout as the structural template.
- `gated_delta_net_scan.cu`/`conv1d_causal.cu` as coding-pattern templates — `gdn_scan_step_f32` cited in Task 1 (shared-memory staging pattern), `conv1d_causal_f32` reused as-is (no new kernel) after confirming its shape already covers NemotronH (Task 9).
- `CudaKernels.cs` GDN launcher conventions — every new launcher in Tasks 1-3 follows `LaunchGdnScanStepF32`'s exact parameter-marshalling/shared-memory pattern.
- `RealGgufCudaParityTests.cs` template + fixture-resolution caveat — Task 14, using the plain env-var pattern (not `TestFixtureResolver`) after reading `NemotronHTextGeneratorTests.cs` in full.
- Single-plan-vs-split decision — addressed in the final report (this plan proceeds as one coherent plan; see report for reasoning).

**2. Placeholder scan:** no `TBD`/`TODO`/"add error handling"/"similar to Task N" patterns anywhere in this plan; every code block is complete, real code with exact signatures drawn from files read in full during this planning session. The one self-corrected placeholder (`Unsafe_TokenIdsDevice` stub in an early draft of Task 6) was caught and replaced with real code in the same task before this plan was finalized — verify no other task has an equivalent stray stub.

**3. Type consistency:** `CudaKernels.LaunchMamba2SelectiveScanF32`/`LaunchGroupRmsNormF32`/`LaunchReluSquaredInplaceF32` signatures declared in Tasks 1-3 are called with matching parameter order/types in Task 9/10. `DeviceSsm`/`DeviceAttn`/`DeviceFfn`/`DeviceLayer` struct field names declared in Task 7 are consumed identically in Task 8's upload helpers and Tasks 9-10's forward bodies. `CudaNemotronHSsmStateCache.GetConvStatePtr`/`GetSsmStatePtr` (Task 4) match their call sites in Task 9. `CudaNemotronHKvCache`'s `IKvCache` surface (Task 5) matches Task 10's `TensorRef`-based call pattern. `_tokenEmbedHostPtr`/`_tokenEmbedQt`/`_tokenEmbedRowBytes` field names are consistent from their Task 7 declaration through Task 8's constructor assignment and `Embed`'s Task 8 usage.
