# Mamba3 CUDA Host (issue #346) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Mamba3 a dedicated CUDA host (`CudaMamba3TransformerModel`) that loads a safetensors checkpoint, runs prefill + streaming-decode with bit-parity-oriented F32 kernels, and replaces the interim `NotSupportedException` guards with real dispatch — closing the gap where CPU and Vulkan both support Mamba3 end-to-end but CUDA does not.

**Architecture:** Four new CUDA kernels (`mamba3_data_rope_f32`, `mamba3_chunk_boundary_f32`, `mamba3_ssd_scan_siso_f32`, `mamba3_ssd_scan_mimo_f32`) port the already-validated Vulkan GLSL compute shaders to CUDA C, translating `barrier()`→`__syncthreads()`, `gl_WorkGroupID.x`→`blockIdx.x`, shared arrays→`__shared__`. Per-token preprocessing (softplus/sigmoid/RMSNorm+bias/qk_pre_dot/scale) stays host-side C#, mirroring Vulkan's proven design decision rather than fusing it into a device kernel on day one. `CudaMamba3TransformerModel` reuses the existing CPU `Mamba3WeightLoader`/`Mamba3Weights` to resolve and validate safetensors tensors (no new tensor-mapping code), then uploads to device buffers via the same `cuMemAlloc_v2`/`cuMemcpyHtoD_v2` idiom `CudaGdnStateCache` and the Qwen3-hybrid loaders already use. In-proj/out-proj/lm_head GEMMs reuse `CudaGemm.LinearF32` (already used by the MLA F32 path for the same "byte-near-equivalence with CPU" reason). Embedding lookup, per-layer RMSNorm, final RMSNorm, and residual-add reuse the **existing generic** `rmsnorm_f32`/`add_f32`/`embedding_lookup_f32_f32out` kernels — no new kernel needed for those. SISO ships first (the only architecture with a real checkpoint); MIMO is added as a later task validated only against synthetic fixtures (no public MIMO checkpoint exists anywhere in the codebase).

**Tech Stack:** C# (.NET 10, `DotLLM.Cuda`), CUDA C (`native/kernels/*.cu` → PTX via `native/build_ptx.bat`), CUDA driver API (`cuLaunchKernel`, no `[LibraryImport]` per-kernel — see Global Constraints), xUnit (`DotLLM.Tests.Unit` / `DotLLM.Tests.Integration`).

## Deviations from the issue text — read before executing

The issue's acceptance criteria and research directions assume two things that turned out to be **false** on independent verification. Both are load-bearing for how this plan is scoped; do not "fix" the plan back toward the issue text without re-reading this section.

1. **No GGUF path exists for Mamba3 anywhere in the codebase — not CPU, not Vulkan, and no dotLLM-invented convention either.** `grep -r "Mamba3" src/DotLLM.Models/Gguf/` returns zero files. `Mamba3ConfigExtractor.cs`'s own doc comment says: *"Mamba-3 has no upstream GGUF support, so `Architecture.Mamba3` is parsed from HF JSON directly."* `docs/SUPPORTED_MODELS.md` states outright: *"No upstream GGUF mapping."* A Vulkan unit test (`VulkanNemotronHGgufLoaderTests.ParseArchitecture_Mamba3_IsNotAGgufArchitecture`) explicitly asserts Mamba3 is *not* a valid GGUF architecture string. The issue's acceptance criterion #1 ("`CudaMamba3TransformerModel.LoadFromGguf` ... wired into `CudaModelLoader.CreateFromGguf`'s switch, replacing the new guard") is therefore **unimplementable as literally written** — there is no tensor convention to load. **This plan implements `LoadFromSafetensors` instead** (mirroring the CPU `Mamba3TransformerModel.LoadFromSafetensors` / Vulkan `VulkanMamba3TransformerModel.LoadFromSafetensors` pattern exactly — the same pattern the issue itself calls "the GPU-porting pattern already validated once"). The `CreateFromGguf` guard added in the prior session (commit `ddd7a899`, PR #350) **stays in place**, reworded to explain that no backend has GGUF support for Mamba3 (not just CUDA) and to point at the new safetensors entry point. Inventing a dotLLM-only GGUF convention for Mamba3 is out of scope — flagged as a follow-up issue in Task 14, not attempted here.
2. **Mamba3 has no causal-conv1d step.** The issue's research directions describe Mamba3's SSM tensors as including `conv1d_weight` and ask whether `native/kernels/conv1d_causal.cu` can be reused. This is Mamba2/NemotronH-flavored language that does not apply to this codebase's Mamba3. Direct evidence: `Mamba3TensorMapping.cs` enumerates exactly 9 per-layer SISO tensors (`norm, in_proj, out_proj, B_norm, C_norm, B_bias, C_bias, D, dt_bias`) — no conv weight. `Mamba3State.cs` has exactly four buffers (`ssm_state, cum_angle, k_state, v_state`) — no conv-state buffer. `Mamba3Config.cs`'s own doc comment states outright: *"Mamba-3 is a pure SSM: there is no convolution (Mamba-3's trapezoidal two-input recurrence makes conv1d redundant — see DESIGN_MAMBA_3.md §1.1)."* Grepping `Mamba3Block.cs` and the entire Vulkan Mamba3 port for `conv1d`/`Conv1d`/`conv_state` returns zero hits. `conv1d_causal.cu` is real and used — by NemotronH's Mamba2 layers and Qwen3-Hybrid's GDN layers, a structurally different recurrence family (also the subject of sibling issue #347, which this plan does not touch per the issue's own warning against kernel sharing). **No task in this plan touches `conv1d_causal.cu` or adds a conv1d kernel.**

## Global Constraints

- **F32 only.** Mirrors the CPU model, which throws `NotSupportedException` for any non-F32 tensor dtype (`Mamba3TransformerModel.SpanFromHandle`) — the only real checkpoint (`ib-ssm/mamba3-370M-10BT`) is F32. Quantized/F16 Mamba3 weights are out of scope; Vulkan's Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16 overlay support (`Mamba3Weights`'s "Vulkan-only quant overlay" fields) is explicitly not ported here.
- **SISO before MIMO.** The only real-weight checkpoint anywhere (`ib-ssm/mamba3-370M-10BT`, gated via `DOTLLM_IBSSM_CHECKPOINT_PATH`) is SISO (`is_mimo=false`). MIMO (Tasks 13) is validated only against synthetic fixtures, exactly as CPU/Vulkan's own MIMO coverage is (`docs/ROADMAP.md` step 60f: "No public MIMO checkpoint — real-weight MIMO verification deferred").
- **Bit-exactness discipline for the recurrent kernels.** Per the GDN scan kernel's established precedent (`native/kernels/gated_delta_net_scan.cu`, issues #173/#180): the SSD scan kernels must default to the exact, sequential accumulation order that matches the CPU scalar reference — same accumulator nesting, same reduction order — not a faster-but-reordered variant. `mamba3_ssd_scan_siso_f32`, `mamba3_ssd_scan_mimo_f32`, `mamba3_data_rope_f32`, and `mamba3_chunk_boundary_f32` all go in `native/build_ptx.bat`'s `NO_FMA` list (alongside `gated_delta_net_scan`) so nvcc's default `--fmad=true` does not introduce FMA-contraction drift against the CPU's plain `MathF` operations. None of the four go in the `FAST_MATH` list.
- **No `[LibraryImport]` per kernel.** This codebase's CUDA kernels compile to PTX and load via the CUDA **driver API** (`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`) at runtime, not per-kernel P/Invoke declarations — `[LibraryImport]` wraps the driver/cuBLAS API itself (`CudaDriverApi.cs`, `CublasApi.cs`), which already exists and needs no changes. New kernels need: a `.cu` file (auto-picked-up by `build_ptx.bat`'s glob), a `CudaKernels.cs` loader block + launcher method using `stackalloc void*[]` + `cuLaunchKernel`.
- **Optional-kernel loading pattern.** Every new kernel loads via the "Tier 2" pattern in `CudaKernels.cs` (`File.Exists(path)` guard, `TryGetFunction` not `GetFunction`, nullable module field) — a stale PTX build without the new symbols must not break unrelated models. Mirrors the GDN scan kernel block (`CudaKernels.cs:931-948`) verbatim in structure.
- **Reuse the CPU weight loader, do not reimplement tensor resolution.** `CudaMamba3TransformerModel.LoadFromSafetensors` calls `Mamba3WeightLoader.Load(config, file)` to get a host-side `Mamba3Weights` (mmap-backed `Mamba3TensorHandle`s), then uploads each handle to a device buffer. This is the same "load CPU-side, then upload" pattern `VulkanMamba3TransformerModel.LoadFromSafetensors` already uses (`Mamba3WeightLoader.Load(config, file)` → `VulkanMamba3Weights.Upload(...)`), and gets `Mamba3WeightLoadReport` diagnostics for free instead of duplicating tensor-name/shape validation in CUDA-specific code.
- **Reuse generic F32 kernels for non-SSM-specific ops.** Embedding lookup (`embedding_lookup_f32_f32out` via `CudaKernels.LaunchEmbeddingLookupF32`), RMSNorm (`rmsnorm_f32` via `CudaKernels.LaunchRmsNormF32`), and residual add (`add_f32` via `CudaKernels.LaunchAddF32`) already exist and are architecture-agnostic. No new kernel for any of these three ops.
- **Cross-backend bug rule (CLAUDE.md).** If implementing/testing any of these tasks surfaces a correctness bug in the CPU or Vulkan Mamba3 reference (e.g. a formula mismatch against `Mamba3CanonicalSsd`/`Mamba3Block`/`Mamba3DataRoPE`), stop and fix it in CPU + Vulkan + CUDA together before proceeding — do not silently "match the buggy behavior" in the new CUDA kernel.
- **No LoRA path.** CPU's `Mamba3TransformerModel.ForwardBatch` explicitly rejects LoRA adapters ("no Mamba-3 LoRA path today"). `CudaMamba3TransformerModel` follows the same restriction — no LoRA-adapter parameter threading anywhere in this plan.

---

### Task 1: `mamba3_data_rope_f32` CUDA kernel

**Files:**
- Create: `native/kernels/mamba3_data_rope_f32.cu`
- Modify: `native/build_ptx.bat`

**Interfaces:**
- Produces: PTX symbol `mamba3_data_rope_f32` in `native/ptx/mamba3_data_rope_f32.ptx`, consumed by Task 2's launcher.

This is a direct GLSL→CUDA-C translation of `native/vulkan/shaders/mamba3_data_rope_f32.comp`, whose kernel body (already verified against the CPU-authoritative `Mamba3DataRoPE.ExecuteCanonical` in `src/DotLLM.Cpu/Kernels/Mamba3DataRoPE.cs`) is:

```glsl
for (uint t = 0u; t < pc.seqLen; t++) {
    float dtHere = dt[t * pc.nHead + h];
    for (uint k = tid; k < nra; k += WG_SIZE) {
        float raw = anglesRaw[t * nra + k];
        float tanhPi = tanh(raw) * PI_F;
        float v = sharedCum[k] + dtHere * tanhPi;
        float floored = floor(v * INV_TWO_PI);
        v = v - TWO_PI * floored;
        sharedCum[k] = v;
        sharedCos[k] = cos(v);
        sharedSin[k] = sin(v);
    }
    barrier();
    if (tid < nra) {
        float co = sharedCos[tid]; float si = sharedSin[tid];
        uint tokenBase = t * bcTokenStride;
        for (uint r = 0u; r < pc.nRank; r++) {
            uint bcBase = tokenBase + r * bcRankStride + h * bcHeadStride;
            if (pc.mode == 0u) {
                uint i0 = bcBase + 2u * tid; uint i1 = i0 + 1u;
                float be = bArr[i0]; float bo = bArr[i1];
                float ce = cArr[i0]; float co2 = cArr[i1];
                bArr[i0] = co * be - si * bo;  bArr[i1] = si * be + co * bo;
                cArr[i0] = co * ce - si * co2; cArr[i1] = si * ce + co * co2;
            } else {
                uint i0 = bcBase + tid; uint i1 = bcBase + halfDState + tid;
                float be = bArr[i0]; float bo = bArr[i1];
                float ce = cArr[i0]; float co2 = cArr[i1];
                bArr[i0] = co * be - si * bo;  bArr[i1] = si * be + co * bo;
                cArr[i0] = co * ce - si * co2; cArr[i1] = si * ce + co * co2;
            }
        }
    }
    barrier();
}
if (pc.writeCumOut != 0u) {
    for (uint k = tid; k < nra; k += WG_SIZE) cumOut[h * nra + k] = sharedCum[k];
}
```
Bindings: 0=`b` RW `[T,R,H,N]`, 1=`c` RW `[T,R,H,N]`, 2=`anglesRaw` ro `[T,numRopeAngles]`, 3=`dt` ro `[T,H]`, 4=`cumPrev` ro `[H,numRopeAngles]`, 5=`cumOut` wo `[H,numRopeAngles]`. Push constants: `seqLen, nRank, nHead, dState, numRopeAngles, mode (0=Pairwise,1=Halved), hasCumPrev, writeCumOut`. Dispatch: `local_size_x=64`, one workgroup per head. Shared arrays capped at `MAX_ROPE_ANGLES=256` (Vulkan's own cap, enforced C#-side in Task 2's launcher).

The CPU-authoritative angle formula (`Mamba3DataRoPE.ExecuteCanonicalInto`, `src/DotLLM.Cpu/Kernels/Mamba3DataRoPE.cs:637-661`) confirms this GLSL body exactly: `tanhPiVec = tanh(angRow) * π`, `cumRow += dt[h] * tanhPiVec` (mod 2π via `floor`), `cos`/`sin` of the wrapped angle, then pairwise (`ApplyPairRotation`, rotary channels `[0, 2*numRopeAngles)`) or halved (`ApplyHalvedRotation`, pairs `(k, k+halfDState)`) rotation of `b`/`c`.

- [ ] **Step 1: Write the kernel**

Create `native/kernels/mamba3_data_rope_f32.cu`:

```c
// mamba3_data_rope_f32.cu — Mamba-3 canonical data-dependent RoPE (issue #346).
//
// Direct CUDA-C port of native/vulkan/shaders/mamba3_data_rope_f32.comp, itself
// validated against the CPU-authoritative DotLLM.Cpu.Kernels.Mamba3DataRoPE.ExecuteCanonical
// (src/DotLLM.Cpu/Kernels/Mamba3DataRoPE.cs). One CUDA block per head; sequential loop
// over t inside the block (barrier()->__syncthreads()); NO_FMA (see build_ptx.bat) to
// match the CPU reference's plain MathF operations bit-for-bit-modulo-fp-reduction-order.
//
// Layout: b, c are [T, nRank, nHead, dState] row-major, mutated in place.
// anglesRaw is [T, numRopeAngles] (shared across rank & head). dt is [T, nHead].
// cumPrev/cumOut are [nHead, numRopeAngles].
//
// mode: 0 = Pairwise (SISO — pairs (v[2k], v[2k+1]) over the first 2*numRopeAngles
//            channels; tail passes through unchanged).
//       1 = Halved (MIMO — pairs (v[k], v[k+dState/2]) for k in [0, numRopeAngles);
//            remaining lanes of each half pass through unchanged).

#define WG_SIZE 64
#define MAX_ROPE_ANGLES 256

extern "C" __global__ void __launch_bounds__(WG_SIZE) mamba3_data_rope_f32(
    float* __restrict__ b, float* __restrict__ c,
    const float* __restrict__ anglesRaw, const float* __restrict__ dt,
    const float* __restrict__ cumPrev, float* __restrict__ cumOut,
    const int seqLen, const int nRank, const int nHead, const int dState,
    const int numRopeAngles, const int mode, const int hasCumPrev, const int writeCumOut)
{
    __shared__ float sharedCum[MAX_ROPE_ANGLES];
    __shared__ float sharedCos[MAX_ROPE_ANGLES];
    __shared__ float sharedSin[MAX_ROPE_ANGLES];

    const int h = blockIdx.x;
    if (h >= nHead) return;
    const int tid = threadIdx.x;
    const int nra = numRopeAngles;
    const int halfDState = dState >> 1;
    const int bcHeadStride = dState;
    const int bcRankStride = nHead * dState;
    const int bcTokenStride = nRank * bcRankStride;
    const float PI_F = 3.14159265358979323846f;
    const float TWO_PI = 2.0f * PI_F;
    const float INV_TWO_PI = 1.0f / TWO_PI;

    for (int k = tid; k < nra; k += WG_SIZE)
        sharedCum[k] = hasCumPrev ? cumPrev[h * nra + k] : 0.0f;
    __syncthreads();

    for (int t = 0; t < seqLen; t++)
    {
        float dtHere = dt[t * nHead + h];

        for (int k = tid; k < nra; k += WG_SIZE)
        {
            float raw = anglesRaw[t * nra + k];
            float tanhPi = tanhf(raw) * PI_F;
            float v = sharedCum[k] + dtHere * tanhPi;
            float floored = floorf(v * INV_TWO_PI);
            v = v - TWO_PI * floored;
            sharedCum[k] = v;
            sharedCos[k] = cosf(v);
            sharedSin[k] = sinf(v);
        }
        __syncthreads();

        if (tid < nra)
        {
            float co = sharedCos[tid];
            float si = sharedSin[tid];
            int tokenBase = t * bcTokenStride;
            for (int r = 0; r < nRank; r++)
            {
                int bcBase = tokenBase + r * bcRankStride + h * bcHeadStride;
                int i0, i1;
                if (mode == 0)
                {
                    i0 = bcBase + 2 * tid; i1 = i0 + 1;
                }
                else
                {
                    i0 = bcBase + tid; i1 = bcBase + halfDState + tid;
                }
                float be = b[i0]; float bo = b[i1];
                float ce = c[i0]; float co2 = c[i1];
                b[i0] = co * be - si * bo;   b[i1] = si * be + co * bo;
                c[i0] = co * ce - si * co2;  c[i1] = si * ce + co * co2;
            }
        }
        __syncthreads();
    }

    if (writeCumOut)
    {
        for (int k = tid; k < nra; k += WG_SIZE)
            cumOut[h * nra + k] = sharedCum[k];
    }
}
```

- [ ] **Step 2: Register in `build_ptx.bat`'s `NO_FMA` list**

In `native/build_ptx.bat`, find the line:
```
set "NO_FMA=conv1d_causal gated_delta_net_scan elementwise_f32 turboquant"
```
Change to:
```
set "NO_FMA=conv1d_causal gated_delta_net_scan elementwise_f32 turboquant mamba3_data_rope_f32 mamba3_chunk_boundary_f32 mamba3_ssd_scan_siso_f32 mamba3_ssd_scan_mimo_f32"
```
(Adding all four Mamba3 kernel base names now, even though Tasks 3/5/13 create the other three files later — this line only needs touching once.)

- [ ] **Step 3: Build PTX**

Run: `native\build_ptx.bat` (repo root; requires `CUDA_PATH` set to a CUDA 13.x toolkit).
Expected: `native\ptx\mamba3_data_rope_f32.ptx` generated with no compile errors.

- [ ] **Step 4: Commit**

```bash
git add native/kernels/mamba3_data_rope_f32.cu native/ptx/mamba3_data_rope_f32.ptx native/build_ptx.bat
git commit -m "feat(cuda): add mamba3_data_rope_f32 kernel (#346)"
```

---

### Task 2: `CudaKernels.cs` loader + launcher for the RoPE kernel, unit-tested against the CPU oracle

**Files:**
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3DataRopeF32Tests.cs`

**Interfaces:**
- Consumes: PTX symbol `mamba3_data_rope_f32` (Task 1).
- Produces: `public bool HasMamba3DataRope { get; }` and
  `public void LaunchMamba3DataRopeF32(nint b, nint c, nint anglesRaw, nint dt, nint cumPrev, nint cumOut, int seqLen, int nRank, int nHead, int dState, int numRopeAngles, int mode, bool hasCumPrev, bool writeCumOut, nint stream)` — consumed by Task 9 (model forward) and this task's own unit test.

- [ ] **Step 1: Declare the fields**

In `src/DotLLM.Cuda/CudaKernels.cs`, near the GDN scan module fields (after `_gdnScanF32Module`/`_gdnScanStepF32Func` block, around line 360), add:

```csharp
    // Issue #346 (Mamba3 CUDA host): canonical data-dependent RoPE on B/C.
    private CudaModule? _mamba3DataRopeF32Module;
    private nint _mamba3DataRopeF32Func;
```

- [ ] **Step 2: Load it (Tier-2 optional pattern, mirrors the GDN scan block at `CudaKernels.cs:931-948`)**

In the `CudaKernels(string ptxDir)` constructor, after the GDN scan module block, add:

```csharp
        string mamba3DataRopeF32Path = Path.Combine(ptxDir, "mamba3_data_rope_f32.ptx");
        if (File.Exists(mamba3DataRopeF32Path))
        {
            _mamba3DataRopeF32Module = CudaModule.LoadFromFile(mamba3DataRopeF32Path);
            _mamba3DataRopeF32Func = _mamba3DataRopeF32Module.TryGetFunction("mamba3_data_rope_f32");
        }
```

- [ ] **Step 3: Add the capability flag**

Near `HasGdnKernels`, add:

```csharp
    /// <summary>
    /// True when the Mamba-3 canonical data-RoPE kernel (issue #346,
    /// <see cref="LaunchMamba3DataRopeF32"/>) is loaded. Optional — a stale PTX build
    /// without this symbol still loads; <see cref="Architectures.CudaMamba3TransformerModel"/>
    /// throws a descriptive error only when it actually needs this kernel.
    /// </summary>
    public bool HasMamba3DataRope => _mamba3DataRopeF32Func != 0;
```

- [ ] **Step 4: Add the launcher**

```csharp
    /// <summary>
    /// Mamba-3 canonical data-dependent RoPE on B/C (issue #346). One CUDA block per
    /// head; sequential loop over t inside the kernel (mirrors
    /// native/vulkan/shaders/mamba3_data_rope_f32.comp's one-workgroup-per-head design,
    /// itself validated against <c>DotLLM.Cpu.Kernels.Mamba3DataRoPE.ExecuteCanonical</c>).
    /// <paramref name="b"/>/<paramref name="c"/> are mutated in place, shape
    /// <c>[seqLen, nRank, nHead, dState]</c>. <paramref name="mode"/>: 0=Pairwise (SISO),
    /// 1=Halved (MIMO). Pass <paramref name="hasCumPrev"/>=false to start the cumulative
    /// angle from zero (fresh sequence); pass <paramref name="writeCumOut"/>=false if the
    /// caller does not need the final angle (rare — decode continuity needs it every call).
    /// </summary>
    public void LaunchMamba3DataRopeF32(nint b, nint c, nint anglesRaw, nint dt,
        nint cumPrev, nint cumOut, int seqLen, int nRank, int nHead, int dState,
        int numRopeAngles, int mode, bool hasCumPrev, bool writeCumOut, nint stream)
    {
        if (_mamba3DataRopeF32Func == 0)
            throw new InvalidOperationException(
                "mamba3_data_rope_f32 kernel not available. Recompile native/kernels/mamba3_data_rope_f32.cu to PTX.");
        if (numRopeAngles > 256)
            throw new ArgumentOutOfRangeException(nameof(numRopeAngles),
                $"numRopeAngles={numRopeAngles}; mamba3_data_rope_f32 is compiled with MAX_ROPE_ANGLES=256 shared-memory arrays.");

        nint bArg = b, cArg = c, anglesArg = anglesRaw, dtArg = dt, cumPrevArg = cumPrev, cumOutArg = cumOut;
        int seqArg = seqLen, rankArg = nRank, headArg = nHead, stateArg = dState, nraArg = numRopeAngles, modeArg = mode;
        int hasCumArg = hasCumPrev ? 1 : 0, writeCumArg = writeCumOut ? 1 : 0;

        void** args = stackalloc void*[] {
            &bArg, &cArg, &anglesArg, &dtArg, &cumPrevArg, &cumOutArg,
            &seqArg, &rankArg, &headArg, &stateArg, &nraArg, &modeArg, &hasCumArg, &writeCumArg };

        CudaDriverApi.cuLaunchKernel(_mamba3DataRopeF32Func,
                (uint)nHead, 1, 1, 64, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
```

- [ ] **Step 5: Free the module in `Dispose()`**

In `CudaKernels.Dispose()`, alongside the other optional-module frees, add:
```csharp
        _mamba3DataRopeF32Module?.Dispose();
```

- [ ] **Step 6: Write the unit test against the CPU oracle**

Create `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3DataRopeF32Tests.cs`, mirroring `CudaGdnScanStepF32Tests.cs`'s structure (`FindPtxDir`, `IsCudaDriverPresent`, tiny + larger shape `[InlineData]`, upload random inputs to both CPU and GPU, assert bit-exact equality):

```csharp
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3DataRopeF32"/> (the
/// <c>mamba3_data_rope_f32</c> CUDA kernel, native/kernels/mamba3_data_rope_f32.cu)
/// against its CPU oracle, <see cref="Mamba3DataRoPE.ExecuteCanonical"/>. Issue #346.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMamba3DataRopeF32Tests
{
    private readonly ITestOutputHelper _out;
    public CudaMamba3DataRopeF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

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
    [InlineData(1, 4, 8, 2, 3)]     // SISO-shaped: nRank=1, mode=Pairwise
    [InlineData(1, 32, 128, 32, 5)] // ib-ssm/mamba3-370M-10BT shape (nHead=32, dState=128, numRopeAngles=32)
    public void Mamba3DataRopeF32_MatchesCpuReference_Pairwise(
        int nRank, int nHead, int dState, int numRopeAngles, int seqLen)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3DataRope, "mamba3_data_rope_f32 PTX symbol not found (stale build)");

        var rng = new Random(0xA3CE ^ nHead ^ (dState << 8) ^ (seqLen << 16));
        int bcLen = seqLen * nRank * nHead * dState;
        int dtLen = seqLen * nHead;
        int angLen = seqLen * numRopeAngles;
        int cumLen = nHead * numRopeAngles;

        float[] bCpu = RandomArray(rng, bcLen), cCpu = RandomArray(rng, bcLen);
        float[] bGpu = (float[])bCpu.Clone(), cGpu = (float[])cCpu.Clone();
        float[] anglesRaw = RandomArray(rng, angLen);
        float[] dt = new float[dtLen];
        for (int i = 0; i < dtLen; i++) dt[i] = (float)rng.NextDouble() * 0.1f; // dt > 0, post-softplus range

        float[] cumOutCpu = new float[cumLen];
        Mamba3DataRoPE.ExecuteCanonical(
            bCpu, cCpu, anglesRaw, dt,
            cumAnglePrev: ReadOnlySpan<float>.Empty, cumAngleOut: cumOutCpu,
            seqLen, nRank, nHead, dState, numRopeAngles, Mamba3RoPEMode.Pairwise);

        nint dB = 0, dC = 0, dAng = 0, dDt = 0, dCumPrev = 0, dCumOut = 0;
        try
        {
            long bcBytes = (long)bcLen * sizeof(float);
            long dtBytes = (long)dtLen * sizeof(float);
            long angBytes = (long)angLen * sizeof(float);
            long cumBytes = (long)cumLen * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dC, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAng, (nuint)angBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)dtBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dCumPrev, (nuint)Math.Max(cumBytes, 4)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dCumOut, (nuint)cumBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = bGpu) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cGpu) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = anglesRaw) CudaDriverApi.cuMemcpyHtoD_v2(dAng, (nint)p, (nuint)angBytes).ThrowOnError();
                fixed (float* p = dt) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)dtBytes).ThrowOnError();
            }

            kernels.LaunchMamba3DataRopeF32(dB, dC, dAng, dDt, dCumPrev, dCumOut,
                seqLen, nRank, nHead, dState, numRopeAngles, mode: 0,
                hasCumPrev: false, writeCumOut: true, stream.Handle);
            stream.Synchronize();

            float[] bGpuOut = new float[bcLen], cGpuOut = new float[bcLen], cumOutGpu = new float[cumLen];
            unsafe
            {
                fixed (float* p = bGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dB, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dC, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cumOutGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dCumOut, (nuint)cumBytes).ThrowOnError();
            }

            Assert.True(bCpu.AsSpan().SequenceEqual(bGpuOut), "B rotation mismatch.");
            Assert.True(cCpu.AsSpan().SequenceEqual(cGpuOut), "C rotation mismatch.");
            Assert.True(cumOutCpu.AsSpan().SequenceEqual(cumOutGpu), "cum_angle output mismatch.");
            _out.WriteLine($"nRank={nRank} nHead={nHead} dState={dState} numRopeAngles={numRopeAngles} seqLen={seqLen}: exact match.");
        }
        finally
        {
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dC != 0) CudaDriverApi.cuMemFree_v2(dC);
            if (dAng != 0) CudaDriverApi.cuMemFree_v2(dAng);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dCumPrev != 0) CudaDriverApi.cuMemFree_v2(dCumPrev);
            if (dCumOut != 0) CudaDriverApi.cuMemFree_v2(dCumOut);
        }
    }

    private static float[] RandomArray(Random rng, int len)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
```

- [ ] **Step 7: Run the test**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --filter FullyQualifiedName~CudaMamba3DataRopeF32Tests`
Expected: PASS on a CUDA-equipped host (skips gracefully otherwise).

- [ ] **Step 8: Commit**

```bash
git add src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaMamba3DataRopeF32Tests.cs
git commit -m "feat(cuda): wire mamba3_data_rope_f32 launcher + CPU-parity test (#346)"
```

---

### Task 3: `mamba3_chunk_boundary_f32` CUDA kernel

**Files:**
- Create: `native/kernels/mamba3_chunk_boundary_f32.cu`

**Interfaces:**
- Produces: PTX symbol `mamba3_chunk_boundary_f32` in `native/ptx/mamba3_chunk_boundary_f32.ptx`, consumed by Task 4's launcher.

Direct CUDA-C port of `native/vulkan/shaders/mamba3_chunk_boundary_f32.comp`, which implements the streaming-decode boundary correction documented in `Mamba3Block.ApplyChunkBoundaryAdjustment` (`src/DotLLM.Models/Architectures/Mamba3Block.cs:767-796`, SISO form) and `Mamba3CanonicalSsd.ExecuteMimoStreaming` (rank-summed MIMO form, `src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs:508-545`):

```
ssm_state[h, p, n] += v_state[h, p] · (Σ_r k_state[r, h, n]) · coef[h]
where coef[h] = dt[0, h] · (1 - trap[0, h])
```
(`nRank=1` collapses the sum to a single term for SISO.) GLSL body (`mamba3_chunk_boundary_f32.comp:53-75`):
```glsl
void main() {
    uint h = gl_WorkGroupID.z;
    uint p = gl_GlobalInvocationID.y;
    uint n = gl_GlobalInvocationID.x;
    if (h >= pc.nHead || p >= pc.headDim || n >= pc.dState) return;
    float c = coef[h];
    if (c == 0.0) return;
    uint kRankStride = pc.nHead * pc.dState;
    uint kHeadOff    = h * pc.dState + n;
    float kSum = 0.0;
    for (uint r = 0u; r < pc.nRank; r++) kSum += kState[r * kRankStride + kHeadOff];
    float v = vState[h * pc.headDim + p];
    uint stateIdx = h * pc.headDim * pc.dState + p * pc.dState + n;
    state[stateIdx] += v * kSum * c;
}
```
Bindings: 0=`state` RW `[H,P,N]`, 1=`vState` ro `[H,P]`, 2=`kState` ro `[R,H,N]`, 3=`coef` ro `[H]`. Dispatch: `local_size_x=16, local_size_y=16`, 3D grid `(ceil(dState/16), ceil(headDim/16), nHead)` — one thread per `(h,p,n)` triple, fully parallel (no time recurrence, unlike the scan kernels).

- [ ] **Step 1: Write the kernel**

Create `native/kernels/mamba3_chunk_boundary_f32.cu`:

```c
// mamba3_chunk_boundary_f32.cu — Mamba-3 streaming-decode chunk-boundary state
// correction (issue #346). Direct CUDA-C port of
// native/vulkan/shaders/mamba3_chunk_boundary_f32.comp. Fully parallel over
// (head, headDim, dState) — no time recurrence, unlike the SSD scan kernels.
// NO_FMA (see build_ptx.bat) for CPU-bit-parity with
// Mamba3Block.ApplyChunkBoundaryAdjustment / Mamba3CanonicalSsd.ExecuteMimoStreaming's
// boundary block.
//
// state:  [nHead, headDim, dState] F32, mutated in place (+=).
// vState: [nHead, headDim] F32, previous chunk's last-token V.
// kState: [nRank, nHead, dState] F32, previous chunk's last-token post-RoPE K
//         (nRank=1 for SISO).
// coef:   [nHead] F32, precomputed dt[0,h]*(1-trap[0,h]) — computed host-side
//         during the per-token preprocessing step (Task 9), NOT inside this kernel.
//
// ssm_state[h,p,n] += vState[h,p] * (sum_r kState[r,h,n]) * coef[h]

#define BOUNDARY_WG_X 16
#define BOUNDARY_WG_Y 16

extern "C" __global__ void __launch_bounds__(BOUNDARY_WG_X * BOUNDARY_WG_Y) mamba3_chunk_boundary_f32(
    float* __restrict__ state,
    const float* __restrict__ vState,
    const float* __restrict__ kState,
    const float* __restrict__ coef,
    const int nHead, const int headDim, const int dState, const int nRank)
{
    const int n = blockIdx.x * BOUNDARY_WG_X + threadIdx.x;
    const int p = blockIdx.y * BOUNDARY_WG_Y + threadIdx.y;
    const int h = blockIdx.z;
    if (h >= nHead || p >= headDim || n >= dState) return;

    float c = coef[h];
    if (c == 0.0f) return;

    int kRankStride = nHead * dState;
    int kHeadOff = h * dState + n;
    float kSum = 0.0f;
    for (int r = 0; r < nRank; r++)
        kSum += kState[r * kRankStride + kHeadOff];

    float v = vState[h * headDim + p];
    int stateIdx = h * headDim * dState + p * dState + n;
    state[stateIdx] += v * kSum * c;
}
```

- [ ] **Step 2: Build PTX**

Run: `native\build_ptx.bat` (the `NO_FMA` entry for `mamba3_chunk_boundary_f32` was already added in Task 1 Step 2 — no `build_ptx.bat` edit needed here).
Expected: `native\ptx\mamba3_chunk_boundary_f32.ptx` generated, no errors.

- [ ] **Step 3: Commit**

```bash
git add native/kernels/mamba3_chunk_boundary_f32.cu native/ptx/mamba3_chunk_boundary_f32.ptx
git commit -m "feat(cuda): add mamba3_chunk_boundary_f32 kernel (#346)"
```

---

### Task 4: `CudaKernels.cs` loader + launcher for the chunk-boundary kernel, unit-tested against the documented formula

**Files:**
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3ChunkBoundaryF32Tests.cs`

**Interfaces:**
- Consumes: PTX symbol `mamba3_chunk_boundary_f32` (Task 3).
- Produces: `public bool HasMamba3ChunkBoundary { get; }` and
  `public void LaunchMamba3ChunkBoundaryF32(nint state, nint vState, nint kState, nint coef, int nHead, int headDim, int dState, int nRank, nint stream)` — consumed by Task 9 and this task's test.

- [ ] **Step 1: Declare the fields**

```csharp
    // Issue #346 (Mamba3 CUDA host): streaming-decode chunk-boundary state correction.
    private CudaModule? _mamba3ChunkBoundaryF32Module;
    private nint _mamba3ChunkBoundaryF32Func;
```

- [ ] **Step 2: Load it (Tier-2 optional pattern)**

```csharp
        string mamba3ChunkBoundaryF32Path = Path.Combine(ptxDir, "mamba3_chunk_boundary_f32.ptx");
        if (File.Exists(mamba3ChunkBoundaryF32Path))
        {
            _mamba3ChunkBoundaryF32Module = CudaModule.LoadFromFile(mamba3ChunkBoundaryF32Path);
            _mamba3ChunkBoundaryF32Func = _mamba3ChunkBoundaryF32Module.TryGetFunction("mamba3_chunk_boundary_f32");
        }
```

- [ ] **Step 3: Add the capability flag**

```csharp
    /// <summary>
    /// True when the Mamba-3 streaming-decode chunk-boundary kernel (issue #346,
    /// <see cref="LaunchMamba3ChunkBoundaryF32"/>) is loaded.
    /// </summary>
    public bool HasMamba3ChunkBoundary => _mamba3ChunkBoundaryF32Func != 0;
```

- [ ] **Step 4: Add the launcher**

```csharp
    /// <summary>
    /// Mamba-3 streaming-decode chunk-boundary state correction (issue #346):
    /// <c>state[h,p,n] += vState[h,p] * (sum_r kState[r,h,n]) * coef[h]</c>. Fully
    /// parallel 3D dispatch over (dState, headDim, nHead) — no time recurrence. Call
    /// BEFORE the SSD scan kernel for a chunk that carries a non-empty prior
    /// (kState, vState) pair (see Mamba3Block.Forward's Step 5.5 doc comment for why
    /// this must run before, not after, the scan).
    /// </summary>
    public void LaunchMamba3ChunkBoundaryF32(nint state, nint vState, nint kState, nint coef,
        int nHead, int headDim, int dState, int nRank, nint stream)
    {
        if (_mamba3ChunkBoundaryF32Func == 0)
            throw new InvalidOperationException(
                "mamba3_chunk_boundary_f32 kernel not available. Recompile native/kernels/mamba3_chunk_boundary_f32.cu to PTX.");

        nint stateArg = state, vArg = vState, kArg = kState, coefArg = coef;
        int hArg = nHead, pArg = headDim, nArg = dState, rArg = nRank;

        void** args = stackalloc void*[] { &stateArg, &vArg, &kArg, &coefArg, &hArg, &pArg, &nArg, &rArg };

        uint gridX = (uint)((dState + 15) / 16);
        uint gridY = (uint)((headDim + 15) / 16);
        uint gridZ = (uint)nHead;

        CudaDriverApi.cuLaunchKernel(_mamba3ChunkBoundaryF32Func,
                gridX, gridY, gridZ, 16, 16, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
```

- [ ] **Step 5: Free the module in `Dispose()`**

```csharp
        _mamba3ChunkBoundaryF32Module?.Dispose();
```

- [ ] **Step 6: Write the unit test**

Create `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3ChunkBoundaryF32Tests.cs`. The CPU oracle here is the closed-form formula itself (documented identically in `Mamba3Block.ApplyChunkBoundaryAdjustment` and `Mamba3CanonicalSsd.ExecuteMimoStreaming`'s boundary block) — computed inline rather than via a public CPU entry point, since `Mamba3Block.ApplyChunkBoundaryAdjustment` is `private`:

```csharp
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3ChunkBoundaryF32"/> against
/// the documented closed-form formula (<c>Mamba3Block.ApplyChunkBoundaryAdjustment</c> /
/// <c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>'s boundary block):
/// <c>state[h,p,n] += vState[h,p] * (sum_r kState[r,h,n]) * coef[h]</c>. Issue #346.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMamba3ChunkBoundaryF32Tests
{
    private readonly ITestOutputHelper _out;
    public CudaMamba3ChunkBoundaryF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

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
    [InlineData(1, 4, 4, 8)]   // SISO: nRank=1
    [InlineData(3, 4, 4, 8)]   // MIMO: nRank=3
    [InlineData(1, 32, 64, 128)] // ib-ssm/mamba3-370M-10BT shape
    public void Mamba3ChunkBoundaryF32_MatchesClosedFormFormula(int nRank, int nHead, int headDim, int dState)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3ChunkBoundary, "mamba3_chunk_boundary_f32 PTX symbol not found (stale build)");

        var rng = new Random(0xB0DA ^ nRank ^ nHead ^ (headDim << 8) ^ (dState << 16));
        int stateLen = nHead * headDim * dState;
        int vLen = nHead * headDim;
        int kLen = nRank * nHead * dState;

        float[] stateCpu = RandomArray(rng, stateLen);
        float[] stateGpu = (float[])stateCpu.Clone();
        float[] vState = RandomArray(rng, vLen);
        float[] kState = RandomArray(rng, kLen);
        float[] coef = new float[nHead];
        for (int h = 0; h < nHead; h++)
            coef[h] = h % 3 == 0 ? 0f : (float)(rng.NextDouble() * 0.5); // exercise the coef==0 early-out too

        // CPU oracle: closed-form formula, matching Mamba3Block.ApplyChunkBoundaryAdjustment /
        // Mamba3CanonicalSsd.ExecuteMimoStreaming's boundary block exactly.
        for (int h = 0; h < nHead; h++)
        {
            float c = coef[h];
            if (c == 0f) continue;
            for (int p = 0; p < headDim; p++)
            {
                float v = vState[h * headDim + p];
                for (int n = 0; n < dState; n++)
                {
                    float kSum = 0f;
                    for (int r = 0; r < nRank; r++)
                        kSum += kState[r * nHead * dState + h * dState + n];
                    stateCpu[h * headDim * dState + p * dState + n] += v * kSum * c;
                }
            }
        }

        nint dState_ = 0, dV = 0, dK = 0, dCoef = 0;
        try
        {
            long stateBytes = (long)stateLen * sizeof(float);
            long vBytes = (long)vLen * sizeof(float);
            long kBytes = (long)kLen * sizeof(float);
            long coefBytes = (long)nHead * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dCoef, (nuint)coefBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = vState) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = kState) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kBytes).ThrowOnError();
                fixed (float* p = coef) CudaDriverApi.cuMemcpyHtoD_v2(dCoef, (nint)p, (nuint)coefBytes).ThrowOnError();
            }

            kernels.LaunchMamba3ChunkBoundaryF32(dState_, dV, dK, dCoef, nHead, headDim, dState, nRank, stream.Handle);
            stream.Synchronize();

            float[] stateGpuOut = new float[stateLen];
            unsafe
            {
                fixed (float* p = stateGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            Assert.True(stateCpu.AsSpan().SequenceEqual(stateGpuOut), "chunk-boundary state mismatch.");
            _out.WriteLine($"nRank={nRank} nHead={nHead} headDim={headDim} dState={dState}: exact match.");
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dCoef != 0) CudaDriverApi.cuMemFree_v2(dCoef);
        }
    }

    private static float[] RandomArray(Random rng, int len)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
```

- [ ] **Step 7: Run the test**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --filter FullyQualifiedName~CudaMamba3ChunkBoundaryF32Tests`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaMamba3ChunkBoundaryF32Tests.cs
git commit -m "feat(cuda): wire mamba3_chunk_boundary_f32 launcher + formula test (#346)"
```

---

### Task 5: `mamba3_ssd_scan_siso_f32` CUDA kernel

**Files:**
- Create: `native/kernels/mamba3_ssd_scan_siso_f32.cu`

**Interfaces:**
- Produces: PTX symbol `mamba3_ssd_scan_siso_f32` in `native/ptx/mamba3_ssd_scan_siso_f32.ptx`, consumed by Task 6's launcher.

This is the core recurrence kernel — a direct CUDA-C port of `native/vulkan/shaders/mamba3_canonical_ssd_siso_f32.comp`, itself a proven GPU port of the CPU-authoritative `Mamba3CanonicalSsd.ExecuteSiso` (`src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs:110-203`). One CUDA **block** per head (matching Vulkan's one-**workgroup**-per-head, NOT the GDN scan kernel's one-launch-per-token host loop — Mamba3's recurrence fits entirely inside a single kernel launch because the state row for `(h,p)` is thread-private across the whole sequential `t` loop, so `__syncthreads()` inside the kernel replaces GDN's inter-launch host synchronization). `blockDim.x = 256` (matches Vulkan's `WG_SIZE=256`, and this codebase's `CudaKernels.BlockSize` constant). Threads `p = tid; p < headDim; p += 256` each own the full `dState`-wide state row for their `(h,p)`, iterated serially over `n` — this exactly mirrors the CPU reference's `for (p) { for (n) {...} }` nesting, so the per-thread accumulation order is bit-identical to the CPU scalar reference by construction (no cross-thread reduction anywhere in the state update itself).

GLSL body (`mamba3_canonical_ssd_siso_f32.comp:64-126`), already verified numerically equal to `Mamba3CanonicalSsd.ExecuteSiso`:
```glsl
void main() {
    uint h = gl_WorkGroupID.x;
    if (h >= pc.nHead) return;
    uint tid = gl_LocalInvocationID.x;
    uint stateHeadBase = h * pc.headDim * pc.dState;
    float dh = d[h];

    for (uint t = 0u; t < pc.seqLen; t++) {
        if (tid == 0u) {
            uint hdrIdx = t * pc.nHead + h;
            float decay = exp(adt[hdrIdx]);
            float scl_  = scl[hdrIdx];
            float gm_   = gm[hdrIdx];
            float qkp_  = qkp[hdrIdx];
            float skip_ = dh + gm_ * qkp_;
            decayShared = decay; sclShared = scl_; skipShared = skip_;
        }
        barrier();
        float decay = decayShared; float scaleT = sclShared; float skipT = skipShared;

        uint vRowBase  = t * pc.nHead * pc.headDim + h * pc.headDim;
        uint bcRowBase = t * pc.nHead * pc.dState  + h * pc.dState;
        uint yRowBase  = vRowBase; uint zRowBase = vRowBase;

        for (uint p = tid; p < pc.headDim; p += WG_SIZE) {
            float vp = v[vRowBase + p];
            uint stateRowBase = stateHeadBase + p * pc.dState;
            float yScan = 0.0;
            for (uint n = 0u; n < pc.dState; n++) {
                float kScaled = kR[bcRowBase + n] * scaleT;
                float s = decay * st[stateRowBase + n] + vp * kScaled;
                st[stateRowBase + n] = s;
                yScan += qR[bcRowBase + n] * s;
            }
            float yOut = yScan + skipT * vp;
            if (pc.hasZ != 0u) {
                float zv = z[zRowBase + p];
                float siluZ = zv / (1.0 + exp(-zv));
                yOut *= siluZ;
            }
            y[yRowBase + p] = yOut;
        }
        barrier();
    }
}
```
Bindings: 0=`state` RW `[H,P,N]`, 1=`v` ro `[T,H,P]`, 2=`qRoped` ro `[T,H,N]`, 3=`kRoped` ro `[T,H,N]`, 4=`qkPreDot` ro `[T,H]`, 5=`scale` ro `[T,H]`, 6=`gamma` ro `[T,H]`, 7=`adt` ro `[T,H]`, 8=`d` ro `[H]`, 9=`z` ro `[T,H,P]` (optional), 10=`y` wo `[T,H,P]`.

- [ ] **Step 1: Write the kernel**

Create `native/kernels/mamba3_ssd_scan_siso_f32.cu`:

```c
// mamba3_ssd_scan_siso_f32.cu — Mamba-3 canonical SISO SSD scan (issue #346).
// Direct CUDA-C port of native/vulkan/shaders/mamba3_canonical_ssd_siso_f32.comp,
// itself validated against DotLLM.Cpu.Kernels.Mamba3CanonicalSsd.ExecuteSiso
// (src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs). One CUDA block per head; each
// thread strides over headDim (p) and owns the full dState-wide (p, :) state row
// across the entire sequential t loop, iterating n serially inside its own thread —
// this reproduces the CPU reference's `for(p){for(n){...}}` nesting exactly, so no
// cross-thread reduction exists in the recurrence itself and per-thread FP
// accumulation order matches the CPU scalar reference bit-for-bit. NO_FMA (see
// build_ptx.bat) for the same reason.
//
// state:  [nHead, headDim, dState] F32, mutated in place across the whole call
//         (this is a single kernel launch covering ALL seqLen tokens, unlike the
//         GDN scan kernel's one-launch-per-token host loop).
// v:      [seqLen, nHead, headDim] F32 (the SSM "x" / value input).
// qRoped/kRoped: [seqLen, nHead, dState] F32 (post-RoPE C / B).
// qkPreDot, scale, gamma, adt: [seqLen, nHead] F32 (host-precomputed, see Task 9).
// d:      [nHead] F32 skip coefficient.
// z:      [seqLen, nHead, headDim] F32 gate input, or nullptr when hasZ==0.
// y:      [seqLen, nHead, headDim] F32 output.

#define SSD_WG_SIZE 256

extern "C" __global__ void __launch_bounds__(SSD_WG_SIZE) mamba3_ssd_scan_siso_f32(
    float* __restrict__ state,
    const float* __restrict__ v,
    const float* __restrict__ qRoped,
    const float* __restrict__ kRoped,
    const float* __restrict__ qkPreDot,
    const float* __restrict__ scale,
    const float* __restrict__ gamma,
    const float* __restrict__ adt,
    const float* __restrict__ d,
    const float* __restrict__ z,
    float* __restrict__ y,
    const int seqLen, const int nHead, const int headDim, const int dState, const int hasZ)
{
    __shared__ float decayShared, sclShared, skipShared;

    const int h = blockIdx.x;
    if (h >= nHead) return;
    const int tid = threadIdx.x;
    const int stateHeadBase = h * headDim * dState;
    const float dh = d[h];

    for (int t = 0; t < seqLen; t++)
    {
        if (tid == 0)
        {
            int hdrIdx = t * nHead + h;
            float decay = expf(adt[hdrIdx]);
            float scl_ = scale[hdrIdx];
            float gm_ = gamma[hdrIdx];
            float qkp_ = qkPreDot[hdrIdx];
            decayShared = decay;
            sclShared = scl_;
            skipShared = dh + gm_ * qkp_;
        }
        __syncthreads();

        float decay = decayShared;
        float scaleT = sclShared;
        float skipT = skipShared;

        int vRowBase = t * nHead * headDim + h * headDim;
        int bcRowBase = t * nHead * dState + h * dState;
        int yRowBase = vRowBase;
        int zRowBase = vRowBase;

        for (int p = tid; p < headDim; p += SSD_WG_SIZE)
        {
            float vp = v[vRowBase + p];
            int stateRowBase = stateHeadBase + p * dState;
            float yScan = 0.0f;
            for (int n = 0; n < dState; n++)
            {
                float kScaled = kRoped[bcRowBase + n] * scaleT;
                float s = decay * state[stateRowBase + n] + vp * kScaled;
                state[stateRowBase + n] = s;
                yScan += qRoped[bcRowBase + n] * s;
            }

            float yOut = yScan + skipT * vp;
            if (hasZ)
            {
                float zv = z[zRowBase + p];
                float siluZ = zv / (1.0f + expf(-zv));
                yOut *= siluZ;
            }
            y[yRowBase + p] = yOut;
        }
        __syncthreads();
    }
}
```

- [ ] **Step 2: Build PTX**

Run: `native\build_ptx.bat` (the `NO_FMA` entry was already added in Task 1 Step 2).
Expected: `native\ptx\mamba3_ssd_scan_siso_f32.ptx` generated, no errors.

- [ ] **Step 3: Commit**

```bash
git add native/kernels/mamba3_ssd_scan_siso_f32.cu native/ptx/mamba3_ssd_scan_siso_f32.ptx
git commit -m "feat(cuda): add mamba3_ssd_scan_siso_f32 kernel (#346)"
```

---

### Task 6: `CudaKernels.cs` loader + launcher for the SISO scan kernel, unit-tested against `Mamba3CanonicalSsd.ExecuteSiso`

**Files:**
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3SsdScanSisoF32Tests.cs`

**Interfaces:**
- Consumes: PTX symbol `mamba3_ssd_scan_siso_f32` (Task 5).
- Produces: `public bool HasMamba3SsdScanSiso { get; }` and
  `public void LaunchMamba3SsdScanSisoF32(nint state, nint v, nint qRoped, nint kRoped, nint qkPreDot, nint scale, nint gamma, nint adt, nint d, nint z, nint y, int seqLen, int nHead, int headDim, int dState, bool hasZ, nint stream)` — consumed by Task 9 and this task's test.

- [ ] **Step 1: Declare the fields**

```csharp
    // Issue #346 (Mamba3 CUDA host): canonical SISO SSD scan.
    private CudaModule? _mamba3SsdScanSisoF32Module;
    private nint _mamba3SsdScanSisoF32Func;
```

- [ ] **Step 2: Load it (Tier-2 optional pattern)**

```csharp
        string mamba3SsdScanSisoF32Path = Path.Combine(ptxDir, "mamba3_ssd_scan_siso_f32.ptx");
        if (File.Exists(mamba3SsdScanSisoF32Path))
        {
            _mamba3SsdScanSisoF32Module = CudaModule.LoadFromFile(mamba3SsdScanSisoF32Path);
            _mamba3SsdScanSisoF32Func = _mamba3SsdScanSisoF32Module.TryGetFunction("mamba3_ssd_scan_siso_f32");
        }
```

- [ ] **Step 3: Add the capability flag**

```csharp
    /// <summary>
    /// True when the Mamba-3 canonical SISO SSD scan kernel (issue #346,
    /// <see cref="LaunchMamba3SsdScanSisoF32"/>) is loaded.
    /// </summary>
    public bool HasMamba3SsdScanSiso => _mamba3SsdScanSisoF32Func != 0;
```

- [ ] **Step 4: Add the launcher**

```csharp
    /// <summary>
    /// Mamba-3 canonical SISO SSD scan (issue #346). One CUDA block per head, block
    /// size 256 (matches <see cref="BlockSize"/> and Vulkan's WG_SIZE); covers ALL
    /// <paramref name="seqLen"/> tokens in a single kernel launch (unlike the GDN scan
    /// kernel's per-token host loop) via a sequential internal loop with
    /// <c>__syncthreads()</c> barriers. <paramref name="state"/> is mutated in place —
    /// pass the persistent <c>[nHead, headDim, dState]</c> SSM state buffer from
    /// <see cref="Architectures.CudaMamba3StateCache"/>. Pass <paramref name="z"/>=0
    /// and <paramref name="hasZ"/>=false to skip the silu(z) gate (never done in
    /// practice — every known checkpoint has a gate — but the CPU/Vulkan kernels both
    /// support it, so this launcher mirrors that for interface parity).
    /// </summary>
    public void LaunchMamba3SsdScanSisoF32(nint state, nint v, nint qRoped, nint kRoped,
        nint qkPreDot, nint scale, nint gamma, nint adt, nint d, nint z, nint y,
        int seqLen, int nHead, int headDim, int dState, bool hasZ, nint stream)
    {
        if (_mamba3SsdScanSisoF32Func == 0)
            throw new InvalidOperationException(
                "mamba3_ssd_scan_siso_f32 kernel not available. Recompile native/kernels/mamba3_ssd_scan_siso_f32.cu to PTX.");

        nint stateArg = state, vArg = v, qArg = qRoped, kArg = kRoped;
        nint qkpArg = qkPreDot, sclArg = scale, gmArg = gamma, adtArg = adt, dArg = d, zArg = z, yArg = y;
        int seqArg = seqLen, headArg = nHead, hdArg = headDim, dsArg = dState, hasZArg = hasZ ? 1 : 0;

        void** args = stackalloc void*[] {
            &stateArg, &vArg, &qArg, &kArg, &qkpArg, &sclArg, &gmArg, &adtArg, &dArg, &zArg, &yArg,
            &seqArg, &headArg, &hdArg, &dsArg, &hasZArg };

        CudaDriverApi.cuLaunchKernel(_mamba3SsdScanSisoF32Func,
                (uint)nHead, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
```

- [ ] **Step 5: Free the module in `Dispose()`**

```csharp
        _mamba3SsdScanSisoF32Module?.Dispose();
```

- [ ] **Step 6: Write the unit test against `Mamba3CanonicalSsd.ExecuteSiso`**

Create `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3SsdScanSisoF32Tests.cs`, mirroring `CudaGdnScanStepF32Tests.cs`'s structure but running the WHOLE sequence through one kernel launch (matching this kernel's single-launch-covers-all-`t` design, unlike GDN's per-token launch loop):

```csharp
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3SsdScanSisoF32"/> against
/// its CPU oracle, <see cref="Mamba3CanonicalSsd.ExecuteSiso"/>. Issue #346.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMamba3SsdScanSisoF32Tests
{
    private readonly ITestOutputHelper _out;
    public CudaMamba3SsdScanSisoF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

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
    [InlineData(4, 4, 8, 5, true)]      // tiny shape, with gate
    [InlineData(4, 4, 8, 5, false)]     // tiny shape, no gate
    [InlineData(32, 64, 128, 6, true)]  // ib-ssm/mamba3-370M-10BT shape (nHead=32, headDim=64, dState=128)
    public void Mamba3SsdScanSisoF32_MatchesCpuReference(
        int nHead, int headDim, int dState, int seqLen, bool hasZ)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3SsdScanSiso, "mamba3_ssd_scan_siso_f32 PTX symbol not found (stale build)");

        var rng = new Random(0x53D0 ^ nHead ^ (headDim << 4) ^ (dState << 12) ^ (seqLen << 20));

        int stateLen = nHead * headDim * dState;
        int vLen = seqLen * nHead * headDim;
        int bcLen = seqLen * nHead * dState;
        int hdrLen = seqLen * nHead;

        float[] stateCpu = RandomArray(rng, stateLen);
        float[] stateGpu = (float[])stateCpu.Clone();
        float[] v = RandomArray(rng, vLen);
        float[] qRoped = RandomArray(rng, bcLen);
        float[] kRoped = RandomArray(rng, bcLen);
        float[] qkPreDot = RandomArray(rng, hdrLen);
        float[] scale = RandomArray(rng, hdrLen);
        float[] gamma = RandomArray(rng, hdrLen);
        float[] adt = new float[hdrLen];
        for (int i = 0; i < hdrLen; i++) adt[i] = -(float)rng.NextDouble() * 2f; // A*DT <= 0 (decay)
        float[] d = RandomArray(rng, nHead);
        float[] z = hasZ ? RandomArray(rng, vLen) : Array.Empty<float>();
        float[] yCpu = new float[vLen];

        Mamba3CanonicalSsd.ExecuteSiso(
            stateCpu, v, qRoped, kRoped, qkPreDot, scale, gamma, adt, d, z, yCpu,
            seqLen, nHead, headDim, dState);

        nint dState_ = 0, dV = 0, dQ = 0, dK = 0, dQkp = 0, dScl = 0, dGm = 0, dAdt = 0, dD = 0, dZ = 0, dY = 0;
        try
        {
            long stateBytes = (long)stateLen * sizeof(float);
            long vBytes = (long)vLen * sizeof(float);
            long bcBytes = (long)bcLen * sizeof(float);
            long hdrBytes = (long)hdrLen * sizeof(float);
            long dBytes = (long)nHead * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQkp, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dScl, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGm, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAdt, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)dBytes).ThrowOnError();
            if (hasZ) CudaDriverApi.cuMemAlloc_v2(out dZ, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)vBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = qRoped) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = kRoped) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = qkPreDot) CudaDriverApi.cuMemcpyHtoD_v2(dQkp, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = scale) CudaDriverApi.cuMemcpyHtoD_v2(dScl, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = gamma) CudaDriverApi.cuMemcpyHtoD_v2(dGm, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = adt) CudaDriverApi.cuMemcpyHtoD_v2(dAdt, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)dBytes).ThrowOnError();
                if (hasZ) fixed (float* p = z) CudaDriverApi.cuMemcpyHtoD_v2(dZ, (nint)p, (nuint)vBytes).ThrowOnError();
            }

            kernels.LaunchMamba3SsdScanSisoF32(dState_, dV, dQ, dK, dQkp, dScl, dGm, dAdt, dD, dZ, dY,
                seqLen, nHead, headDim, dState, hasZ, stream.Handle);
            stream.Synchronize();

            float[] yGpu = new float[vLen];
            float[] stateGpuOut = new float[stateLen];
            unsafe
            {
                fixed (float* p = yGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)vBytes).ThrowOnError();
                fixed (float* p = stateGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            Assert.True(yCpu.AsSpan().SequenceEqual(yGpu), "y output mismatch.");
            Assert.True(stateCpu.AsSpan().SequenceEqual(stateGpuOut), "final ssm_state mismatch.");
            _out.WriteLine($"nHead={nHead} headDim={headDim} dState={dState} seqLen={seqLen} hasZ={hasZ}: exact match.");
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dQkp != 0) CudaDriverApi.cuMemFree_v2(dQkp);
            if (dScl != 0) CudaDriverApi.cuMemFree_v2(dScl);
            if (dGm != 0) CudaDriverApi.cuMemFree_v2(dGm);
            if (dAdt != 0) CudaDriverApi.cuMemFree_v2(dAdt);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dZ != 0) CudaDriverApi.cuMemFree_v2(dZ);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
        }
    }

    private static float[] RandomArray(Random rng, int len)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
```

- [ ] **Step 7: Run the test**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --filter FullyQualifiedName~CudaMamba3SsdScanSisoF32Tests`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaMamba3SsdScanSisoF32Tests.cs
git commit -m "feat(cuda): wire mamba3_ssd_scan_siso_f32 launcher + CPU-parity test (#346)"
```

---

### Task 7: `CudaMamba3StateCache` — device-side per-sequence recurrent state

**Files:**
- Create: `src/DotLLM.Cuda/Architectures/CudaMamba3StateCache.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3StateCacheTests.cs`

**Interfaces:**
- Consumes: `DotLLM.Core.Models.Mamba3Config` (existing, `src/DotLLM.Core/Models/Mamba3Config.cs`), `DotLLM.Core.Models.IMambaState` (existing, `src/DotLLM.Core/Models/IMambaState.cs`).
- Produces: `public sealed unsafe class CudaMamba3StateCache : IMambaState` with `nint GetSsmStatePtr(int layerIndex)`, `nint GetCumAnglePtr(int layerIndex)`, `nint GetKStatePtr(int layerIndex)`, `nint GetVStatePtr(int layerIndex)`, `int NumLayers { get; }`, `void Reset()`, `CudaMamba3StateCache Clone()`, `void CopyTo(CudaMamba3StateCache destination)` — consumed by Task 8/9 (model class) and Task 12 (real-weight decode parity test).

**Public, not internal — unlike `CudaGdnStateCache`.** `CudaGdnStateCache` can stay `internal` because no public API ever takes one as a parameter (the GDN hybrid models' checkpoint API is the opaque `object? CheckpointRecurrentState()`). This class is different: Task 9's `Forward(..., CudaMamba3StateCache state)` overload exposes it directly as a public parameter type — mirroring CPU's own `public sealed unsafe class Mamba3State : IMambaState` (`src/DotLLM.Models/Architectures/Mamba3State.cs:76`, confirmed `public` while planning) — and Task 12/14's parity tests construct it directly from a different assembly (`DotLLM.Tests.Integration`). An `internal` declaration would not compile from those call sites.

This mirrors `CudaGdnStateCache` (`src/DotLLM.Cuda/Architectures/CudaGdnStateCache.cs`, read in full) — same `cuMemAlloc_v2`/`cuMemsetD8_v2`/`cuMemFree_v2`/`cuMemcpyDtoD_v2` idiom, same contiguous-per-layer-block pointer arithmetic, same `Clone()`/`CopyTo()` D2D-copy pattern for speculative-decoding checkpoint/rollback parity (issue #287) — but with **four** buffers per layer (matching CPU `Mamba3State`, `src/DotLLM.Models/Architectures/Mamba3State.cs`) instead of GDN's two, and no conv-state buffer (Global Constraints — Mamba3 has no conv1d).

| buffer | shape | meaning (matches CPU `Mamba3State`) |
|---|---|---|
| `ssm_state` | `[nHead, headDim, dState]` | canonical SSM hidden state |
| `cum_angle` | `[nHead, numRopeAngles]` | running cumulative data-dependent RoPE angle |
| `k_state` | SISO `[nHead, dState]`; MIMO `[mimoRank, nHead, dState]` | previous chunk's last post-RoPE K |
| `v_state` | `[nHead, headDim]` | previous chunk's last raw V |

- [ ] **Step 1: Write the class**

Create `src/DotLLM.Cuda/Architectures/CudaMamba3StateCache.cs`:

```csharp
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side per-sequence recurrent state cache for a Mamba-3 model's SSM layers.
/// Mirror of <see cref="DotLLM.Models.Architectures.Mamba3State"/> but allocating GPU
/// memory via <c>cuMemAlloc_v2</c> — same allocation/dispose/reset idiom as the
/// structurally-analogous <see cref="CudaGdnStateCache"/>, but with FOUR buffers per
/// layer (matching Mamba3State) instead of GDN's two, and no conv-state buffer
/// (Mamba-3 has no causal-conv1d step — see the plan's Global Constraints).
/// </summary>
/// <remarks>
/// <para>
/// Per Mamba-3 layer the cache stores, all zero-initialised at construction:
/// </para>
/// <list type="bullet">
///   <item><description><c>ssm_state</c> — <c>[nHead, headDim, dState]</c> canonical SSM hidden state.</description></item>
///   <item><description><c>cum_angle</c> — <c>[nHead, numRopeAngles]</c> running cumulative RoPE angle.</description></item>
///   <item><description><c>k_state</c> — SISO <c>[nHead, dState]</c>; MIMO <c>[mimoRank, nHead, dState]</c> — previous chunk's last-token post-RoPE K.</description></item>
///   <item><description><c>v_state</c> — <c>[nHead, headDim]</c> — previous chunk's last-token V.</description></item>
/// </list>
/// <para>
/// Buffers are F32 on device. <c>mamba3_ssd_scan_siso_f32</c> /
/// <c>mamba3_ssd_scan_mimo_f32</c> / <c>mamba3_data_rope_f32</c> /
/// <c>mamba3_chunk_boundary_f32</c> consume and mutate these pointers directly via
/// the <see cref="CudaKernels"/> launchers.
/// </para>
/// </remarks>
public sealed unsafe class CudaMamba3StateCache : IMambaState
{
    private readonly int _numLayers;
    private readonly int _ssmStateElementsPerLayer;
    private readonly int _cumAngleElementsPerLayer;
    private readonly int _kStateElementsPerLayer;
    private readonly int _vStateElementsPerLayer;

    // Contiguous per-layer blocks, same pointer-arithmetic pattern as CudaGdnStateCache.
    private nint _ssmState;
    private nint _cumAngle;
    private nint _kState;
    private nint _vState;

    private bool _disposed;

    /// <inheritdoc/>
    public int NumLayers => _numLayers;

    /// <summary>Elements per layer in the SSM hidden state (<c>nHead * headDim * dState</c>).</summary>
    public int SsmStateElementsPerLayer => _ssmStateElementsPerLayer;

    /// <summary>Elements per layer in the cumulative RoPE angle buffer (<c>nHead * numRopeAngles</c>).</summary>
    public int CumAngleElementsPerLayer => _cumAngleElementsPerLayer;

    /// <summary>Elements per layer in the K state (rank-aware — see class doc).</summary>
    public int KStateElementsPerLayer => _kStateElementsPerLayer;

    /// <summary>Elements per layer in the V state (<c>nHead * headDim</c>).</summary>
    public int VStateElementsPerLayer => _vStateElementsPerLayer;

    /// <summary>Total bytes allocated across all four state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * (_ssmStateElementsPerLayer + _cumAngleElementsPerLayer
                            + _kStateElementsPerLayer + _vStateElementsPerLayer) * sizeof(float);

    /// <summary>
    /// Creates a new Mamba-3 state cache for the given config and layer count. All
    /// buffers are zero-initialised (zero state = start of sequence) via
    /// <c>cuMemsetD8_v2</c>.
    /// </summary>
    /// <param name="m3">Mamba-3 hyperparameters (<see cref="ModelConfig.Mamba3Config"/>).</param>
    /// <param name="numLayers">Number of Mamba-3 layers covered by this cache.</param>
    public CudaMamba3StateCache(Mamba3Config m3, int numLayers)
    {
        ArgumentNullException.ThrowIfNull(m3);
        if (numLayers < 0) throw new ArgumentOutOfRangeException(nameof(numLayers));

        _numLayers = numLayers;
        // k_state carries a rank axis in MIMO — mirrors Mamba3State's kRank logic exactly.
        int kRank = m3.IsMimo ? m3.MimoRank : 1;
        _ssmStateElementsPerLayer = m3.NumHeads * m3.HeadDim * m3.StateSize;
        _cumAngleElementsPerLayer = m3.NumHeads * m3.NumRopeAngles;
        _kStateElementsPerLayer = kRank * m3.NumHeads * m3.StateSize;
        _vStateElementsPerLayer = m3.NumHeads * m3.HeadDim;

        if (numLayers == 0)
        {
            _ssmState = 0; _cumAngle = 0; _kState = 0; _vState = 0;
            return;
        }

        if (_ssmStateElementsPerLayer <= 0 || _cumAngleElementsPerLayer <= 0
            || _kStateElementsPerLayer <= 0 || _vStateElementsPerLayer <= 0)
            throw new ArgumentException(
                "CudaMamba3StateCache requires positive ssm/cum_angle/k_state/v_state element counts; check Mamba3Config dims.",
                nameof(m3));

        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out _ssmState, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _cumAngle, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _kState, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vState, (nuint)vBytes).ThrowOnError();

        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_cumAngle, 0, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_kState, 0, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_vState, 0, (nuint)vBytes).ThrowOnError();
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s ssm_state, length <see cref="SsmStateElementsPerLayer"/> floats.</summary>
    public nint GetSsmStatePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _ssmState + (nint)((long)layerIndex * _ssmStateElementsPerLayer * sizeof(float));
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s cum_angle, length <see cref="CumAngleElementsPerLayer"/> floats.</summary>
    public nint GetCumAnglePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _cumAngle + (nint)((long)layerIndex * _cumAngleElementsPerLayer * sizeof(float));
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s k_state, length <see cref="KStateElementsPerLayer"/> floats.</summary>
    public nint GetKStatePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _kState + (nint)((long)layerIndex * _kStateElementsPerLayer * sizeof(float));
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s v_state, length <see cref="VStateElementsPerLayer"/> floats.</summary>
    public nint GetVStatePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _vState + (nint)((long)layerIndex * _vStateElementsPerLayer * sizeof(float));
    }

    /// <summary>
    /// Device-to-device deep-copies this cache into a freshly-allocated
    /// <see cref="CudaMamba3StateCache"/> of the same shape — mirrors
    /// <see cref="CudaGdnStateCache.Clone"/>'s speculative-decoding checkpoint role
    /// (issue #287), extended to Mamba-3's four-buffer state.
    /// </summary>
    public CudaMamba3StateCache Clone(Mamba3Config m3)
    {
        ThrowIfDisposed();
        var clone = new CudaMamba3StateCache(m3, _numLayers);
        CopyTo(clone);
        return clone;
    }

    /// <summary>Device-to-device overwrites <paramref name="destination"/>'s buffers with this cache's current contents.</summary>
    public void CopyTo(CudaMamba3StateCache destination)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(destination);
        destination.ThrowIfDisposed();
        if (destination._numLayers != _numLayers
            || destination._ssmStateElementsPerLayer != _ssmStateElementsPerLayer
            || destination._cumAngleElementsPerLayer != _cumAngleElementsPerLayer
            || destination._kStateElementsPerLayer != _kStateElementsPerLayer
            || destination._vStateElementsPerLayer != _vStateElementsPerLayer)
        {
            throw new ArgumentException("Destination CudaMamba3StateCache shape does not match this cache's shape.", nameof(destination));
        }

        if (_numLayers == 0) return;

        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);

        CudaDriverApi.cuMemcpyDtoD_v2(destination._ssmState, _ssmState, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(destination._cumAngle, _cumAngle, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(destination._kState, _kState, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(destination._vState, _vState, (nuint)vBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numLayers == 0) return;
        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);
        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_cumAngle, 0, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_kState, 0, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_vState, 0, (nuint)vBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_ssmState != 0) { CudaDriverApi.cuMemFree_v2(_ssmState); _ssmState = 0; }
        if (_cumAngle != 0) { CudaDriverApi.cuMemFree_v2(_cumAngle); _cumAngle = 0; }
        if (_kState != 0) { CudaDriverApi.cuMemFree_v2(_kState); _kState = 0; }
        if (_vState != 0) { CudaDriverApi.cuMemFree_v2(_vState); _vState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaMamba3StateCache));
    }

    /// <summary>Finalizer — last-ditch free if not disposed.</summary>
    ~CudaMamba3StateCache()
    {
        if (_disposed) return;
        if (_ssmState != 0) CudaDriverApi.cuMemFree_v2(_ssmState);
        if (_cumAngle != 0) CudaDriverApi.cuMemFree_v2(_cumAngle);
        if (_kState != 0) CudaDriverApi.cuMemFree_v2(_kState);
        if (_vState != 0) CudaDriverApi.cuMemFree_v2(_vState);
    }
}
```

- [ ] **Step 2: Write the unit test**

Create `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3StateCacheTests.cs`:

```csharp
using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public class CudaMamba3StateCacheTests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static Mamba3Config TinySisoConfig() => new()
    {
        StateSize = 8, NumHeads = 4, HeadDim = 4, Expand = 2, NumGroups = 1,
        ChunkSize = 64, IsMimo = false, MimoRank = 4, AFloor = 1e-4f,
        DtInitFloor = 1e-4f, DtMin = 1e-3f, DtMax = 0.1f, UseL2Warp = false,
        RopeFraction = 0.5f, IsOutProjNorm = false, RescalePrenormResidual = true,
        ResidualInFp32 = true,
    };

    private static Mamba3Config TinyMimoConfig() => TinySisoConfig() with { IsMimo = true, MimoRank = 3 };

    [SkippableFact]
    public void Constructor_ZeroInitializes_AllFourBuffers()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var cache = new CudaMamba3StateCache(m3, numLayers: 2);

        Assert.Equal(2, cache.NumLayers);
        Assert.Equal(m3.NumHeads * m3.HeadDim * m3.StateSize, cache.SsmStateElementsPerLayer);
        Assert.Equal(m3.NumHeads * m3.NumRopeAngles, cache.CumAngleElementsPerLayer);
        Assert.Equal(m3.NumHeads * m3.StateSize, cache.KStateElementsPerLayer); // SISO: kRank=1
        Assert.Equal(m3.NumHeads * m3.HeadDim, cache.VStateElementsPerLayer);

        AssertAllZero(cache.GetSsmStatePtr(0), cache.SsmStateElementsPerLayer);
        AssertAllZero(cache.GetSsmStatePtr(1), cache.SsmStateElementsPerLayer);
        AssertAllZero(cache.GetKStatePtr(0), cache.KStateElementsPerLayer);
        AssertAllZero(cache.GetVStatePtr(0), cache.VStateElementsPerLayer);
    }

    [SkippableFact]
    public void MimoConfig_KStateElementsPerLayer_IncludesRankAxis()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinyMimoConfig();
        using var cache = new CudaMamba3StateCache(m3, numLayers: 1);
        Assert.Equal(m3.MimoRank * m3.NumHeads * m3.StateSize, cache.KStateElementsPerLayer);
    }

    [SkippableFact]
    public void Reset_ZeroesNonZeroState()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var cache = new CudaMamba3StateCache(m3, numLayers: 1);

        float[] ones = new float[cache.SsmStateElementsPerLayer];
        Array.Fill(ones, 1.0f);
        unsafe
        {
            fixed (float* p = ones)
                CudaDriverApi.cuMemcpyHtoD_v2(cache.GetSsmStatePtr(0), (nint)p, (nuint)(ones.Length * sizeof(float))).ThrowOnError();
        }

        cache.Reset();
        AssertAllZero(cache.GetSsmStatePtr(0), cache.SsmStateElementsPerLayer);
    }

    [SkippableFact]
    public void CopyTo_DeepCopiesState_IndependentOfSource()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var src = new CudaMamba3StateCache(m3, numLayers: 1);
        using var dst = new CudaMamba3StateCache(m3, numLayers: 1);

        float[] vals = new float[src.SsmStateElementsPerLayer];
        for (int i = 0; i < vals.Length; i++) vals[i] = i + 1;
        unsafe
        {
            fixed (float* p = vals)
                CudaDriverApi.cuMemcpyHtoD_v2(src.GetSsmStatePtr(0), (nint)p, (nuint)(vals.Length * sizeof(float))).ThrowOnError();
        }

        src.CopyTo(dst);

        float[] dstOut = new float[dst.SsmStateElementsPerLayer];
        unsafe
        {
            fixed (float* p = dstOut)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dst.GetSsmStatePtr(0), (nuint)(dstOut.Length * sizeof(float))).ThrowOnError();
        }
        Assert.Equal(vals, dstOut);

        // Mutate source after copy; destination must be unaffected (independent allocation).
        src.Reset();
        float[] dstAfterSrcReset = new float[dst.SsmStateElementsPerLayer];
        unsafe
        {
            fixed (float* p = dstAfterSrcReset)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dst.GetSsmStatePtr(0), (nuint)(dstAfterSrcReset.Length * sizeof(float))).ThrowOnError();
        }
        Assert.Equal(vals, dstAfterSrcReset);
    }

    private static unsafe void AssertAllZero(nint devicePtr, int elementCount)
    {
        float[] host = new float[elementCount];
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        foreach (float v in host) Assert.Equal(0f, v);
    }
}
```

- [ ] **Step 3: Run the tests**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --filter FullyQualifiedName~CudaMamba3StateCacheTests`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaMamba3StateCache.cs tests/DotLLM.Tests.Unit/Cuda/CudaMamba3StateCacheTests.cs
git commit -m "feat(cuda): add CudaMamba3StateCache device recurrent state (#346)"
```

---

### Task 8: `CudaMamba3TransformerModel` skeleton — fields, constructor, weight upload, `LoadFromSafetensors`, `Dispose`

**Files:**
- Create: `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`

**Interfaces:**
- Consumes: `Mamba3WeightLoader.Load(ModelConfig, ISafetensorsTensorSource)` → `Mamba3Weights` (existing, `src/DotLLM.Models/Architectures/Mamba3WeightLoader.cs`), `CudaMamba3StateCache` (Task 7), `CudaKernels`/`CudaGemm`/`CudaContext`/`CudaStream`/`CudaCublasHandle` (existing infra).
- Produces: `public sealed unsafe class CudaMamba3TransformerModel : IModel` with
  `public static CudaMamba3TransformerModel LoadFromSafetensors(ISafetensorsTensorSource file, ModelConfig config, int deviceId = 0, string? ptxDir = null)` — consumed by Task 10 (`CudaModelLoader.LoadMamba3FromSafetensors`). `Forward`/`ForwardBatch` land in Task 9.

This class is deliberately **not** based on `CudaQwen3HybridDenseTransformerModel` — that class carries GDN/attention/MoE/MTP/LoRA/KV-cache machinery Mamba3 has none of (Mamba3 is homogeneous: every layer is the same SSM block, no attention layers, no MoE, no LoRA path — CPU's own `ForwardBatch` explicitly rejects LoRA adapters). It borrows only what's structurally relevant: the `CudaContext`/`CudaStream`/`CudaCublasHandle`/`CudaKernels` construction idiom (`CudaTransformerModel.cs:545-551`, `CudaQwen3HybridDenseTransformerModel.cs:354-374`), `CudaGemm.LinearF32` for GEMMs (same F32-for-CPU-parity rationale as the MLA Phase 1 path), and the generic `LaunchEmbeddingLookupF32`/`LaunchRmsNormF32`/`LaunchAddF32` kernels already used across every other CUDA model.

- [ ] **Step 1: Write the skeleton, weight upload, and `LoadFromSafetensors`**

Create `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`:

```csharp
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the Mamba-3 pure-SSM architecture (issue #346). F32
/// activations and weights throughout, mirroring
/// <see cref="DotLLM.Models.Architectures.Mamba3TransformerModel"/> (CPU) and
/// <see cref="DotLLM.Vulkan.VulkanMamba3TransformerModel"/> on the GPU. Per-token
/// preprocessing (softplus/sigmoid/RMSNorm+bias/qk_pre_dot/scale) runs host-side,
/// mirroring the Vulkan port's proven design decision (see that class's own remarks
/// for the rationale) rather than fusing it into a device kernel on day one.
/// </summary>
/// <remarks>
/// <para>
/// <b>Homogeneous, not hybrid.</b> Every layer is <c>{RMSNorm, Mamba3Block, residual
/// add}</c> — no attention layers, no MoE, no LoRA path (CPU's own
/// <c>Mamba3TransformerModel.ForwardBatch</c> rejects LoRA adapters outright). This
/// class is therefore structurally simpler than
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>: no <c>HybridLayerLayout</c>
/// dispatch, no KV-cache, no MTP head.
/// </para>
/// <para>
/// <b>Safetensors only.</b> Mamba-3 has no GGUF tensor-naming convention on any
/// dotLLM backend — see this plan's "Deviations from the issue text" section. There is
/// no <c>LoadFromGguf</c> on this class.
/// </para>
/// <para>
/// <b>Weight loading strategy.</b> Reuses the existing CPU
/// <see cref="Mamba3WeightLoader.Load"/> to resolve tensor names/shapes/diagnostics
/// against a <see cref="Mamba3Weights"/> (host-side, mmap-backed handles), then
/// uploads each populated handle to a device buffer. This avoids duplicating
/// tensor-name/shape validation in CUDA-specific code — the same "load CPU-side, then
/// upload" strategy <c>VulkanMamba3TransformerModel.LoadFromSafetensors</c> already
/// uses.
/// </para>
/// </remarks>
public sealed unsafe class CudaMamba3TransformerModel : IModel
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaKernels _kernels;
    private readonly Mamba3Config _m3;
    private readonly int _numLayers;

    private readonly DeviceLayer[] _layers;
    private readonly nint _tokenEmbedDevice;   // [vocab, hidden]
    private readonly nint _finalNormDevice;    // [hidden]
    private readonly nint _lmHeadDevice;       // [vocab, hidden] — aliases _tokenEmbedDevice when tied
    private readonly bool _lmHeadOwnsDevice;

    // Forward scratch — device buffers sized to the widest seqLen seen so far,
    // grown power-of-two on demand by EnsureScratchCapacity (Task 9). Mirrors
    // Mamba3ForwardScratch (CPU) / VulkanMamba3ForwardScratch (Vulkan).
    private int _scratchCapacity;
    private nint _hidden, _residual, _normOut, _blockOut;      // [cap, hidden]
    private nint _projDevice;                                   // [cap, dInProj] (in_proj GEMM output)
    private nint _xDevice, _zDevice, _yScanDevice;               // [cap, dInner]
    private nint _dtDevice, _adtDevice, _trapDevice, _gammaDevice, _scaleDevice, _qkPreDotDevice; // [cap, nHead]
    private nint _anglesRawDevice;                               // [cap, numRopeAngles]
    private nint _bDevice, _cDevice;                             // [cap, effRank, nHead, dState]
    private nint _coefDevice;                                    // [nHead] — chunk-boundary coefficients
    private nint _logitsDevice;                                  // [vocab] — last-token logits (allocated once)

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => ScratchAllocatedBytes();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => new CudaMamba3StateCache(_m3, _numLayers);

    /// <inheritdoc/>
    /// <remarks>
    /// Deliberately a no-op, matching CPU's <c>Mamba3TransformerModel.ResetSequenceState</c>:
    /// this model owns no persistent recurrent state of its own — every forward that is
    /// not given a caller-supplied <see cref="CudaMamba3StateCache"/> allocates and
    /// disposes a fresh ephemeral one for that call (see Task 9), so consecutive
    /// uncached forwards are already independent sequences.
    /// </remarks>
    public void ResetSequenceState() { }

    private readonly record struct DeviceLayer(
        nint Norm, nint InProj, nint OutProj, nint BNorm, nint CNorm,
        nint BBias, nint CBias, nint D, nint DtBias,
        nint MimoZ, nint MimoO);

    private CudaMamba3TransformerModel(
        ModelConfig config, CudaContext context, CudaStream stream, CudaCublasHandle cublas,
        CudaKernels kernels, DeviceLayer[] layers, nint tokenEmbedDevice, nint finalNormDevice,
        nint lmHeadDevice, bool lmHeadOwnsDevice)
    {
        Config = config;
        _m3 = config.Mamba3Config!;
        _numLayers = config.NumLayers;
        _context = context;
        _stream = stream;
        _cublas = cublas;
        _kernels = kernels;
        _layers = layers;
        _tokenEmbedDevice = tokenEmbedDevice;
        _finalNormDevice = finalNormDevice;
        _lmHeadDevice = lmHeadDevice;
        _lmHeadOwnsDevice = lmHeadOwnsDevice;

        cublas.SetStream(stream);

        long vocabBytes = (long)config.VocabSize * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _logitsDevice, (nuint)vocabBytes).ThrowOnError();
    }

    /// <summary>
    /// Loads a Mamba-3 model from an opened HF-convention safetensors source onto the
    /// specified GPU. Mirrors
    /// <see cref="DotLLM.Models.Architectures.Mamba3TransformerModel.LoadFromSafetensors"/>
    /// (CPU) and <c>VulkanMamba3TransformerModel.LoadFromSafetensors</c>'s "resolve on
    /// CPU, upload to device" strategy.
    /// </summary>
    /// <param name="file">An opened safetensors source positioned at a Mamba-3 checkpoint. Must outlive the returned model's load (not required to outlive the model itself — weights are uploaded, then the CPU-side handles are released).</param>
    /// <param name="config">Model config with <see cref="ModelConfig.Mamba3Config"/> populated and <see cref="ModelConfig.Architecture"/> == <see cref="Architecture.Mamba3"/>.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect (<c>AppContext.BaseDirectory/ptx</c>).</param>
    /// <exception cref="InvalidDataException">One or more required Mamba-3 tensors are missing/malformed — see <see cref="Mamba3Weights.Report"/> via the thrown message.</exception>
    public static CudaMamba3TransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Mamba3)
            throw new ArgumentException(
                $"CudaMamba3TransformerModel requires Architecture.Mamba3, got {config.Architecture}.",
                nameof(config));
        if (config.Mamba3Config is null)
            throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated for CudaMamba3TransformerModel.",
                nameof(config));

        // Reuse the CPU loader for tensor resolution/shape validation/diagnostics —
        // see the class doc for why this is preferable to a CUDA-specific re-implementation.
        Mamba3Weights weights = Mamba3WeightLoader.Load(config, file);
        try
        {
            if (weights.Report.HasMissingRequired)
                throw new InvalidDataException(
                    $"Mamba-3 weights are incomplete ({weights.Report.MissingRequiredCount} required tensors "
                    + "missing). Inspect Mamba3Weights.Report.Problems before attempting a CUDA load.");

            var context = CudaContext.Create(deviceId);
            var stream = CudaStream.Create();
            var cublas = CudaCublasHandle.Create();
            ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
            var kernels = new CudaKernels(ptxDir);

            var m3 = config.Mamba3Config;
            int hidden = config.HiddenSize;
            int vocab = config.VocabSize;
            int dInner = m3.DInner;
            int nHead = m3.NumHeads;
            int dState = m3.StateSize;
            int effRank = m3.IsMimo ? m3.MimoRank : 1;
            int bcBiasElems = nHead * effRank * dState;
            int mimoElems = m3.IsMimo ? nHead * m3.MimoRank * m3.HeadDim : 0;

            nint tokenEmbedDevice = UploadF32(weights.TokenEmbedding, (long)vocab * hidden, stream.Handle);
            nint finalNormDevice = UploadF32(weights.FinalNorm, hidden, stream.Handle);

            bool tied = weights.LmHead.Pointer == weights.TokenEmbedding.Pointer;
            nint lmHeadDevice = tied ? tokenEmbedDevice : UploadF32(weights.LmHead, (long)vocab * hidden, stream.Handle);

            var layers = new DeviceLayer[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
            {
                ref readonly var lw = ref weights.Layers[i];
                layers[i] = new DeviceLayer(
                    Norm: UploadF32(lw.Norm, hidden, stream.Handle),
                    InProj: UploadF32(lw.InProj, (long)m3.InputProjectionDim * hidden, stream.Handle),
                    OutProj: UploadF32(lw.OutProj, (long)hidden * dInner, stream.Handle),
                    BNorm: UploadF32(lw.BNorm, dState, stream.Handle),
                    CNorm: UploadF32(lw.CNorm, dState, stream.Handle),
                    BBias: UploadF32(lw.BBias, bcBiasElems, stream.Handle),
                    CBias: UploadF32(lw.CBias, bcBiasElems, stream.Handle),
                    D: UploadF32(lw.D, nHead, stream.Handle),
                    DtBias: UploadF32(lw.DtBias, nHead, stream.Handle),
                    MimoZ: m3.IsMimo ? UploadF32(lw.MimoZ, mimoElems, stream.Handle) : 0,
                    MimoO: m3.IsMimo ? UploadF32(lw.MimoO, mimoElems, stream.Handle) : 0);
            }

            // All H2D copies above were issued async on `stream` — synchronize before
            // releasing the CPU-side (mmap-backed) weight handles.
            stream.Synchronize();

            return new CudaMamba3TransformerModel(
                config, context, stream, cublas, kernels, layers,
                tokenEmbedDevice, finalNormDevice, lmHeadDevice, lmHeadOwnsDevice: !tied);
        }
        finally
        {
            // weights.Dispose() is a no-op for mmap-backed (OwnsMemory=false) handles —
            // the safetensors file itself is the lifetime anchor, and this method does
            // not retain a reference to it after the H2D copies above complete.
            weights.Dispose();
        }
    }

    /// <summary>
    /// Uploads a populated F32 <see cref="Mamba3TensorHandle"/> to a freshly-allocated
    /// device buffer via an async H2D copy on <paramref name="stream"/>. Returns 0 (no
    /// allocation) for an unpopulated handle — e.g. <c>MimoZ</c>/<c>MimoO</c> on a SISO
    /// checkpoint.
    /// </summary>
    private static nint UploadF32(Mamba3TensorHandle handle, long expectedElements, nint stream)
    {
        if (!handle.IsPopulated) return 0;
        if (handle.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"CudaMamba3TransformerModel requires F32 tensors; got {handle.SourceDType}. "
                + "Quantized/F16 Mamba-3 weights are not yet supported on CUDA (CPU-parity scope, issue #346).");

        long bytes = expectedElements * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoDAsync_v2(devPtr, handle.Pointer, (nuint)bytes, stream).ThrowOnError();
        return devPtr;
    }

    private long ScratchAllocatedBytes()
    {
        if (_scratchCapacity == 0) return 0;
        long cap = _scratchCapacity;
        int hidden = Config.HiddenSize, dInner = _m3.DInner, nHead = _m3.NumHeads;
        int dState = _m3.StateSize, numRopeAngles = _m3.NumRopeAngles;
        int effRank = _m3.IsMimo ? _m3.MimoRank : 1;
        long floats = cap * hidden * 4L                       // hidden/residual/normOut/blockOut
                    + cap * _m3.InputProjectionDim             // proj
                    + cap * dInner * 3L                        // x/z/yScan
                    + cap * nHead * 6L                         // dt/adt/trap/gamma/scale/qkPreDot
                    + cap * numRopeAngles                      // anglesRaw
                    + cap * effRank * nHead * dState * 2L;     // b/c
        return floats * sizeof(float);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var l in _layers)
        {
            FreeIfNonZero(l.Norm); FreeIfNonZero(l.InProj); FreeIfNonZero(l.OutProj);
            FreeIfNonZero(l.BNorm); FreeIfNonZero(l.CNorm); FreeIfNonZero(l.BBias); FreeIfNonZero(l.CBias);
            FreeIfNonZero(l.D); FreeIfNonZero(l.DtBias); FreeIfNonZero(l.MimoZ); FreeIfNonZero(l.MimoO);
        }
        FreeIfNonZero(_tokenEmbedDevice);
        FreeIfNonZero(_finalNormDevice);
        if (_lmHeadOwnsDevice) FreeIfNonZero(_lmHeadDevice);
        FreeIfNonZero(_logitsDevice);
        FreeScratch();

        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
        GC.SuppressFinalize(this);
    }

    private static void FreeIfNonZero(nint ptr)
    {
        if (ptr != 0) CudaDriverApi.cuMemFree_v2(ptr);
    }

    private void FreeScratch()
    {
        FreeIfNonZero(_hidden); FreeIfNonZero(_residual); FreeIfNonZero(_normOut); FreeIfNonZero(_blockOut);
        FreeIfNonZero(_projDevice); FreeIfNonZero(_xDevice); FreeIfNonZero(_zDevice); FreeIfNonZero(_yScanDevice);
        FreeIfNonZero(_dtDevice); FreeIfNonZero(_adtDevice); FreeIfNonZero(_trapDevice);
        FreeIfNonZero(_gammaDevice); FreeIfNonZero(_scaleDevice); FreeIfNonZero(_qkPreDotDevice);
        FreeIfNonZero(_anglesRawDevice); FreeIfNonZero(_bDevice); FreeIfNonZero(_cDevice); FreeIfNonZero(_coefDevice);
        _scratchCapacity = 0;
    }
}
```

- [ ] **Step 2: Build and verify it compiles**

Run: `dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj`
Expected: builds clean (Task 9 adds the `Forward`/`EnsureScratchCapacity` bodies the fields above are declared for — this step just confirms the skeleton itself is syntactically complete and every field/type reference resolves).

- [ ] **Step 3: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs
git commit -m "feat(cuda): add CudaMamba3TransformerModel skeleton + LoadFromSafetensors (#346)"
```

---

### Task 9: `CudaMamba3TransformerModel.Forward` (SISO) — scratch allocation, per-layer dispatch, host prep

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`

**Interfaces:**
- Consumes: `CudaKernels.LaunchEmbeddingLookupF32`/`LaunchRmsNormF32`/`LaunchAddF32` (existing, generic), `CudaGemm.LinearF32` (existing), `LaunchMamba3DataRopeF32`/`LaunchMamba3ChunkBoundaryF32`/`LaunchMamba3SsdScanSisoF32` (Tasks 2/4/6), `CudaMamba3StateCache` (Task 7).
- Produces: `ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)`, `ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)`, `ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, CudaMamba3StateCache state)`, `IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)` — satisfies `IModel`, consumed by Task 11/12's parity tests.

**Design note — chunk-boundary is unconditional, not flag-gated.** CPU's `Mamba3Block.Forward` calls `ApplyChunkBoundaryAdjustment` whenever the caller passes non-empty `kState`/`vState` spans, full stop — no "is this the first chunk" flag. It works because a fresh `Mamba3State` zero-initializes `v_state`, and the adjustment's inner term is `vState[h,p] * coef[h]` — zero `vState` makes the whole contribution zero regardless of `coef`, so calling it on a fresh sequence is a mathematically exact no-op. This plan follows the same policy: **the state-threaded `Forward(..., CudaMamba3StateCache)` overload always dispatches `LaunchMamba3ChunkBoundaryF32`; the ephemeral (no persistent state) overload never does** — mirroring CPU's short (no `kState`/`vState` params) vs. long `Mamba3Block.Forward` overloads exactly. (Vulkan's `HasBoundary` flag is a *dispatch-skip* perf optimization on top of the same underlying no-op-when-fresh property, not a correctness requirement — this plan does not need to reproduce that optimization to be correct, and skipping it keeps the state-cache class simpler.)

- [ ] **Step 1: Add scratch capacity management + embedding lookup, in a new region of the same file**

Append to `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`, inside the class body (after the `Dispose`/`FreeScratch` region from Task 8):

```csharp
    // ────────────────────────────────────────────────────────────────────────
    // Forward
    // ────────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        _ = kvCache; // Mamba-3 uses SSM state, not KV-cache — matches CPU's Forward(..., IKvCache?) contract.
        using var ephemeral = new CudaMamba3StateCache(_m3, _numLayers);
        return ForwardCore(tokenIds, positions, deviceId, ephemeral, runChunkBoundary: false);
    }

    /// <summary>
    /// Runs a forward pass that reads and writes a persistent
    /// <see cref="CudaMamba3StateCache"/>, enabling prefill-then-decode sequences.
    /// Mirrors CPU <c>Mamba3TransformerModel.Forward(..., Mamba3State)</c>.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
        CudaMamba3StateCache state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.NumLayers != _numLayers)
            throw new ArgumentException(
                $"CudaMamba3StateCache has {state.NumLayers} layers but model has {_numLayers}.", nameof(state));
        return ForwardCore(tokenIds, positions, deviceId, state, runChunkBoundary: true);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Mirrors CPU <c>Mamba3TransformerModel.ForwardBatch</c>: rejects LoRA adapters
    /// (no Mamba-3 LoRA path), requires every request to carry a per-seq
    /// <see cref="CudaMamba3StateCache"/> (via <see cref="SequenceForwardRequest.MambaState"/>)
    /// once 2+ requests are batched together, and otherwise loops per request — no
    /// fused-GEMM batching in this v1 (see this plan's biggest-risk note for why).
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "CudaMamba3TransformerModel.ForwardBatch does not support LoRA adapters "
                    + "(no Mamba-3 LoRA path today, matching the CPU host).");
        }

        if (requests.Count >= 2)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                if (requests[i].MambaState is null)
                    throw new ArgumentException(
                        $"CudaMamba3TransformerModel.ForwardBatch with {requests.Count} requests requires "
                        + $"every request to supply a per-seq MambaState; request[{i}] has none.",
                        nameof(requests));
            }
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            if (r.MambaState is null)
            {
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache);
            }
            else if (r.MambaState is CudaMamba3StateCache cudaState)
            {
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, cudaState);
            }
            else
            {
                throw new ArgumentException(
                    $"CudaMamba3TransformerModel requires a CudaMamba3StateCache; got {r.MambaState.GetType().Name}.",
                    nameof(requests));
            }
        }
        return results;
    }

    /// <summary>
    /// Grows every device scratch buffer so at least <paramref name="seqLen"/> tokens
    /// can be served without further allocation. Power-of-two growth, mirroring
    /// <c>Mamba3ForwardScratch.EnsureCapacity</c> (CPU) /
    /// <c>VulkanMamba3ForwardScratch.EnsureCapacity</c> (Vulkan).
    /// </summary>
    private void EnsureScratchCapacity(int seqLen)
    {
        if (seqLen <= _scratchCapacity) return;

        int cap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeScratch();
        _scratchCapacity = 0; // FreeScratch already resets this; explicit for clarity before re-set below.

        int hidden = Config.HiddenSize;
        int dInProj = _m3.InputProjectionDim;
        int dInner = _m3.DInner;
        int nHead = _m3.NumHeads;
        int dState = _m3.StateSize;
        int numRopeAngles = _m3.NumRopeAngles;
        int effRank = _m3.IsMimo ? _m3.MimoRank : 1;

        _hidden = AllocF32((long)cap * hidden);
        _residual = AllocF32((long)cap * hidden);
        _normOut = AllocF32((long)cap * hidden);
        _blockOut = AllocF32((long)cap * hidden);
        _projDevice = AllocF32((long)cap * dInProj);
        _xDevice = AllocF32((long)cap * dInner);
        _zDevice = AllocF32((long)cap * dInner);
        _yScanDevice = AllocF32((long)cap * dInner);
        _dtDevice = AllocF32((long)cap * nHead);
        _adtDevice = AllocF32((long)cap * nHead);
        _trapDevice = AllocF32((long)cap * nHead);
        _gammaDevice = AllocF32((long)cap * nHead);
        _scaleDevice = AllocF32((long)cap * nHead);
        _qkPreDotDevice = AllocF32((long)cap * nHead);
        _anglesRawDevice = AllocF32((long)cap * numRopeAngles);
        _bDevice = AllocF32((long)cap * effRank * nHead * dState);
        _cDevice = AllocF32((long)cap * effRank * nHead * dState);
        _coefDevice = AllocF32(nHead);

        _scratchCapacity = cap;
    }

    private static nint AllocF32(long elementCount)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        return ptr;
    }
```

- [ ] **Step 2: Run `dotnet build` to check the additions compile against Task 8's field declarations**

Run: `dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj`
Expected: builds with only "unused private member" warnings for `ForwardCore` (added in Step 3) — no type errors.

- [ ] **Step 3: Add `ForwardCore` — the per-layer loop**

Append `ForwardCore` and its host-prep helper to the same class:

```csharp
    [System.Runtime.CompilerServices.SkipLocalsInit]
    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
        CudaMamba3StateCache state, bool runChunkBoundary)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        // deviceId controls the RETURNED tensor's placement, not where compute runs (compute
        // always runs on the GPU this model was loaded onto via LoadFromSafetensors' own
        // deviceId). This v1 only implements host-resident output (mirrors every CPU/Vulkan
        // Forward call site, which always passes -1) — matches CPU's
        // Mamba3TransformerModel.ForwardCore threading `deviceId` into UnmanagedTensor.Allocate,
        // but a device-resident (deviceId >= 0) result would need a D2D copy instead of the D2H
        // copy below, which is out of scope here.
        if (deviceId >= 0)
            throw new NotSupportedException(
                "CudaMamba3TransformerModel.Forward only supports deviceId=-1 (host-resident output "
                + "tensor) today. Device-resident output tensors are a future optimization.");
        if (_m3.IsMimo)
            throw new NotSupportedException(
                "CudaMamba3TransformerModel.Forward (SISO path) does not support IsMimo=true "
                + "checkpoints yet — see ForwardMimo (issue #346 Task 14).");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int nHead = _m3.NumHeads;
        int headDim = _m3.HeadDim;
        int dState = _m3.StateSize;
        int dInner = _m3.DInner;
        int dInProj = _m3.InputProjectionDim;
        int numRopeAngles = _m3.NumRopeAngles;
        float aFloor = _m3.AFloor;
        float eps = Config.NormEpsilon;
        nint s = _stream.Handle;

        EnsureScratchCapacity(seqLen);

        // 1. Token upload + embedding lookup (device).
        int[] tokenIdsArr = tokenIds.ToArray();
        nint tokenIdsDevice = 0;
        try
        {
            long tokenBytes = (long)seqLen * sizeof(int);
            CudaDriverApi.cuMemAlloc_v2(out tokenIdsDevice, (nuint)tokenBytes).ThrowOnError();
            fixed (int* p = tokenIdsArr)
                CudaDriverApi.cuMemcpyHtoD_v2(tokenIdsDevice, (nint)p, (nuint)tokenBytes).ThrowOnError();

            _kernels.LaunchEmbeddingLookupF32(_tokenEmbedDevice, QuantizationType.F32,
                tokenIdsDevice, _hidden, seqLen, hiddenSize, s);

            // 2. Layers.
            for (int layer = 0; layer < _numLayers; layer++)
            {
                var lw = _layers[layer];

                // Snapshot residual (D2D) + pre-norm (device).
                CudaDriverApi.cuMemcpyDtoDAsync_v2(_residual, _hidden, (nuint)((long)seqLen * hiddenSize * sizeof(float)), s).ThrowOnError();
                _kernels.LaunchRmsNormF32(_hidden, lw.Norm, _normOut, hiddenSize, eps, seqLen, s);

                // in_proj GEMM (device): proj[seqLen, dInProj] = normOut[seqLen, hidden] @ inProj[dInProj, hidden]^T.
                CudaGemm.LinearF32(_cublas.Handle, _normOut, lw.InProj, _projDevice, seqLen, hiddenSize, dInProj, s);
                _stream.Synchronize();

                // 3. Host prep — D2H the in_proj output, run the per-token
                // softplus/sigmoid/RMSNorm+bias/qk_pre_dot/scale math on CPU (mirrors
                // Mamba3Block.Forward Steps 2-4 exactly), H2D the results back.
                HostPrepareSiso(seqLen, dInProj, dInner, nHead, dState, numRopeAngles, aFloor, eps,
                    lw.DtBias, lw.BNorm, lw.CNorm, lw.BBias, lw.CBias, s);

                // 4. Data-RoPE (device) — mutates _bDevice/_cDevice in place, threads cum_angle.
                _kernels.LaunchMamba3DataRopeF32(_bDevice, _cDevice, _anglesRawDevice, _dtDevice,
                    state.GetCumAnglePtr(layer), state.GetCumAnglePtr(layer),
                    seqLen, nRank: 1, nHead, dState, numRopeAngles, mode: 0,
                    hasCumPrev: true, writeCumOut: true, s);

                // 5. Chunk-boundary correction (device) — BEFORE the scan, only for the
                // state-threaded overload (see this task's design note above).
                if (runChunkBoundary)
                {
                    _kernels.LaunchMamba3ChunkBoundaryF32(
                        state.GetSsmStatePtr(layer), state.GetVStatePtr(layer), state.GetKStatePtr(layer),
                        _coefDevice, nHead, headDim, dState, nRank: 1, s);
                }

                // 6. SISO SSD scan (device) — mutates ssm_state in place, writes _yScanDevice.
                _kernels.LaunchMamba3SsdScanSisoF32(
                    state.GetSsmStatePtr(layer), _xDevice, _cDevice, _bDevice,
                    _qkPreDotDevice, _scaleDevice, _gammaDevice, _adtDevice, lw.D, _zDevice, _yScanDevice,
                    seqLen, nHead, headDim, dState, hasZ: true, s);

                // 6.5. Persist this chunk's last-token post-RoPE K / raw V for the NEXT
                // call's chunk-boundary correction (D2D — matches CPU's bHRN/xBuf slice
                // copy at Mamba3Block.cs Step 6.5).
                if (runChunkBoundary)
                {
                    long kBytes = (long)nHead * dState * sizeof(float);
                    long vBytes = (long)nHead * headDim * sizeof(float);
                    nint lastKSrc = _cDevice + (nint)((long)(seqLen - 1) * nHead * dState * sizeof(float));
                    nint lastVSrc = _xDevice + (nint)((long)(seqLen - 1) * nHead * headDim * sizeof(float));
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetKStatePtr(layer), lastKSrc, (nuint)kBytes, s).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetVStatePtr(layer), lastVSrc, (nuint)vBytes, s).ThrowOnError();
                }

                // 7. out_proj GEMM (device): blockOut[seqLen, hidden] = yScan[seqLen, dInner] @ outProj[hidden, dInner]^T.
                CudaGemm.LinearF32(_cublas.Handle, _yScanDevice, lw.OutProj, _blockOut, seqLen, dInner, hiddenSize, s);

                // Residual add (device): hidden = residual + blockOut.
                _kernels.LaunchAddF32(_residual, _blockOut, _hidden, seqLen * hiddenSize, s);
            }

            // 8. Final RMSNorm (device, in place) + lm_head GEMM (device, last token only).
            _kernels.LaunchRmsNormF32(_hidden, _finalNormDevice, _hidden, hiddenSize, eps, seqLen, s);

            nint lastHidden = _hidden + (nint)((long)(seqLen - 1) * hiddenSize * sizeof(float));
            CudaGemm.GemvF32(_cublas.Handle, _lmHeadDevice, lastHidden, _logitsDevice, vocabSize, hiddenSize, s);

            var shape = new TensorShape(1, vocabSize);
            var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
            _stream.Synchronize();
            CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _logitsDevice, (nuint)((long)vocabSize * sizeof(float))).ThrowOnError();
            return result;
        }
        finally
        {
            if (tokenIdsDevice != 0) CudaDriverApi.cuMemFree_v2(tokenIdsDevice);
        }
    }
```

- [ ] **Step 4: Add `HostPrepareSiso` — line-for-line port of `Mamba3Block.Forward` Steps 2-4**

Append the host-prep helper (D2H the in-proj output, replicate the CPU split/softplus/sigmoid/RMSNorm+bias/qk_pre_dot/shifted-γ math exactly as `Mamba3Block.Forward`'s Steps 2-4 do at `src/DotLLM.Models/Architectures/Mamba3Block.cs:234-357`, then H2D the results):

```csharp
    /// <summary>
    /// Host-side per-token preprocessing for one SISO layer: downloads the in_proj GEMM
    /// output, replicates <c>Mamba3Block.Forward</c>'s Steps 2-4 (split, softplus/sigmoid
    /// DT/A/trap/gamma, B/C RMSNorm+bias, qk_pre_dot, shifted-gamma/scale) on the CPU
    /// exactly as written there, then uploads the per-token tables the device kernels
    /// need. Mirrors Vulkan's <c>ComputeHostTables</c> design decision (see this class's
    /// doc comment) — a fused on-device version is a documented future optimization, not
    /// attempted in this plan (see the biggest-risk note).
    /// </summary>
    private void HostPrepareSiso(int seqLen, int dInProj, int dInner, int nHead, int dState,
        int numRopeAngles, float aFloor, float normEps,
        nint dtBiasDevice, nint bNormDevice, nint cNormDevice, nint bBiasDevice, nint cBiasDevice,
        nint stream)
    {
        float[] proj = new float[seqLen * dInProj];
        fixed (float* p = proj)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, _projDevice, (nuint)(proj.Length * sizeof(float))).ThrowOnError();

        float[] dtBias = DownloadF32(dtBiasDevice, nHead);
        float[] bNormW = DownloadF32(bNormDevice, dState);
        float[] cNormW = DownloadF32(cNormDevice, dState);
        float[] bBias = DownloadF32(bBiasDevice, nHead * dState);   // numBcHeads=1 SISO, so [nHead, dState]
        float[] cBias = DownloadF32(cBiasDevice, nHead * dState);

        int ofsZ = 0, ofsX = dInner, ofsB = 2 * dInner, ofsC = ofsB + dState;
        int ofsDdDt = ofsC + dState, ofsDdA = ofsDdDt + nHead, ofsTrap = ofsDdA + nHead, ofsAngles = ofsTrap + nHead;

        var x = new float[seqLen * dInner];
        var z = new float[seqLen * dInner];
        var dt = new float[seqLen * nHead];
        var adt = new float[seqLen * nHead];
        var trap = new float[seqLen * nHead];
        var gamma = new float[seqLen * nHead];
        var scale = new float[seqLen * nHead];
        var anglesRaw = new float[seqLen * numRopeAngles];
        var bHRN = new float[seqLen * nHead * dState];
        var cHRN = new float[seqLen * nHead * dState];
        var qkPreDot = new float[seqLen * nHead];

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;
            Array.Copy(proj, src + ofsZ, z, t * dInner, dInner);
            Array.Copy(proj, src + ofsX, x, t * dInner, dInner);

            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];

                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;

                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;
            }

            Array.Copy(proj, src + ofsAngles, anglesRaw, t * numRopeAngles, numRopeAngles);

            // B/C RMSNorm + bias (numBcHeads=1 broadcasts to every head — matches every
            // real checkpoint's n_groups=1; multi-group is a Mamba3Block-documented
            // future extension, not implemented on any backend today).
            int bSrcBase = src + ofsB, cSrcBase = src + ofsC;
            RmsNormFactor(proj, bSrcBase, dState, normEps, out float bInvRms);
            RmsNormFactor(proj, cSrcBase, dState, normEps, out float cInvRms);
            for (int h = 0; h < nHead; h++)
            {
                int biasBase = h * dState;
                int dstBase = (t * nHead + h) * dState;
                for (int n = 0; n < dState; n++)
                {
                    bHRN[dstBase + n] = proj[bSrcBase + n] * bInvRms * bNormW[n] + bBias[biasBase + n];
                    cHRN[dstBase + n] = proj[cSrcBase + n] * cInvRms * cNormW[n] + cBias[biasBase + n];
                }
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            int baseT = t * nHead * dState;
            for (int h = 0; h < nHead; h++)
            {
                float dot = 0f;
                int off = baseT + h * dState;
                for (int n = 0; n < dState; n++) dot += cHRN[off + n] * bHRN[off + n];
                qkPreDot[t * nHead + h] = dot;
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }

        // Chunk-boundary coefficient: coef[h] = dt[0,h] * (1 - trap[0,h]) — only the
        // FIRST token's dt/trap matter (Mamba3Block.ApplyChunkBoundaryAdjustment reads
        // dt[0,:]/trap[0,:] only).
        var coef = new float[nHead];
        for (int h = 0; h < nHead; h++) coef[h] = dt[h] * (1f - trap[h]);

        UploadF32Array(x, _xDevice, stream);
        UploadF32Array(z, _zDevice, stream);
        UploadF32Array(dt, _dtDevice, stream);
        UploadF32Array(adt, _adtDevice, stream);
        UploadF32Array(trap, _trapDevice, stream);
        UploadF32Array(gamma, _gammaDevice, stream);
        UploadF32Array(scale, _scaleDevice, stream);
        UploadF32Array(anglesRaw, _anglesRawDevice, stream);
        UploadF32Array(bHRN, _bDevice, stream);
        UploadF32Array(cHRN, _cDevice, stream);
        UploadF32Array(qkPreDot, _qkPreDotDevice, stream);
        UploadF32Array(coef, _coefDevice, stream);
        _stream.Synchronize();
    }

    private static float[] DownloadF32(nint devicePtr, int elementCount)
    {
        var host = new float[elementCount];
        if (devicePtr == 0) return host;
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        return host;
    }

    private static void UploadF32Array(float[] host, nint devicePtr, nint stream)
    {
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyHtoDAsync_v2(devicePtr, (nint)p, (nuint)(host.Length * sizeof(float)), stream).ThrowOnError();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RmsNormFactor(float[] proj, int offset, int n, float normEps, out float invRms)
    {
        // F32 accumulator upcast to double — matches Mamba3Block.RmsNormInto's accumulator
        // precision exactly (bit-parity requirement, not just "close enough").
        double acc = 0.0;
        for (int i = 0; i < n; i++) { double v = proj[offset + i]; acc += v * v; }
        float mean = (float)(acc / n);
        invRms = 1f / MathF.Sqrt(mean + normEps);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftPlus(float x)
    {
        if (x > 20f) return x;
        if (x < -20f) return MathF.Exp(x);
        return MathF.Log(1f + MathF.Exp(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
```

- [ ] **Step 5: Add the missing `using` for `MethodImpl`/`MethodImplOptions`**

At the top of `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`, add:
```csharp
using System.Runtime.CompilerServices;
```

- [ ] **Step 6: Build**

Run: `dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj`
Expected: builds clean, no errors.

- [ ] **Step 7: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs
git commit -m "feat(cuda): implement CudaMamba3TransformerModel.Forward (SISO) (#346)"
```

---

### Task 10: `CudaModelLoader` wiring — `LoadMamba3FromSafetensors` + guard rewording

**Files:**
- Modify: `src/DotLLM.Cuda/CudaModelLoader.cs`

**Interfaces:**
- Consumes: `CudaMamba3TransformerModel.LoadFromSafetensors` (Task 8).
- Produces: `public static (CudaMamba3TransformerModel Model, ISafetensorsTensorSource Source, ModelConfig Config) LoadMamba3FromSafetensors(string path, int deviceId = 0, string? ptxDir = null)` — the new real entry point CLI/server code (or a follow-up ticket) wires up.

**Why not just widen `LoadFromSafetensors`'s return type:** `CudaModelLoader.LoadFromSafetensors` returns `(CudaTransformerModel Model, ...)` — a **concrete type**, not `IModel` (unlike `CreateFromGguf`, which already returns `(IModel Model, ...)` and is therefore polymorphism-safe). Widening that signature to `IModel` would be a breaking API change to every existing caller that pattern-matches on `CudaTransformerModel`, which is out of scope for this ticket. A new dedicated method is the same shape the codebase already uses for `Qwen3HybridDense`/`Qwen3MoeHybrid` on the GGUF side (their own `LoadFromGguf` static methods, dispatched from `CreateFromGguf`'s switch) — this plan applies the identical pattern to the safetensors side.

- [ ] **Step 1: Add the `Mamba3` real-dispatch method**

In `src/DotLLM.Cuda/CudaModelLoader.cs`, add a new public static method after `LoadFromSafetensors` (currently the last method in the file, ending at line 154):

```csharp

    /// <summary>
    /// Loads a Mamba-3 model from an HF safetensors checkpoint onto the specified GPU.
    /// This is the CUDA safetensors entry point for <see cref="Architecture.Mamba3"/> —
    /// a dedicated method rather than a branch inside <see cref="LoadFromSafetensors"/>
    /// because that method's return type is pinned to the concrete
    /// <see cref="CudaTransformerModel"/>, which <see cref="Architectures.CudaMamba3TransformerModel"/>
    /// is not (Mamba-3's layer shape — no attention, no standard FFN — is not
    /// <see cref="CudaTransformerModel"/>-compatible). Mirrors
    /// <see cref="ModelLoader.LoadFromSafetensors(string, ThreadingConfig?)"/> (CPU) and
    /// <c>VulkanMamba3TransformerModel.LoadFromSafetensors</c>'s dedicated-entry-point
    /// pattern.
    /// </summary>
    /// <param name="path">A <c>*.safetensors</c> file or a directory containing one, plus a sibling <c>config.json</c>.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    public static (Architectures.CudaMamba3TransformerModel Model, ISafetensorsTensorSource Source, ModelConfig Config)
        LoadMamba3FromSafetensors(string path, int deviceId = 0, string? ptxDir = null)
    {
        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
        try
        {
            if (config.Architecture != Architecture.Mamba3)
                throw new ArgumentException(
                    $"CudaModelLoader.LoadMamba3FromSafetensors requires a Mamba3 checkpoint, "
                    + $"got Architecture.{config.Architecture}. Use CudaModelLoader.LoadFromSafetensors instead.",
                    nameof(path));

            var model = Architectures.CudaMamba3TransformerModel.LoadFromSafetensors(source, config, deviceId, ptxDir);
            return (model, source, config);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }
```

- [ ] **Step 2: Reword the `LoadFromSafetensors` guard to point at the new method**

Replace:
```csharp
            if (config.Architecture == Architecture.Mamba3)
                throw new NotSupportedException(
                    "CUDA loader does not yet support Mamba3. Use the CPU safetensors loader "
                    + "(ModelLoader.LoadFromSafetensors) or the GGUF path.");
```
with:
```csharp
            if (config.Architecture == Architecture.Mamba3)
                throw new NotSupportedException(
                    "Mamba3 is not loadable via CudaModelLoader.LoadFromSafetensors — its layer "
                    + "shape (no attention, no standard FFN) is not CudaTransformerModel-compatible. "
                    + "Use CudaModelLoader.LoadMamba3FromSafetensors(path, deviceId, ptxDir) instead.");
```

- [ ] **Step 3: Reword the `CreateFromGguf` guard to reflect that no backend has GGUF support (not just CUDA)**

Replace:
```csharp
            case Architecture.Mamba3:
                throw new NotSupportedException(
                    "CUDA has no dedicated loader for Mamba3 yet (tracked separately from the "
                    + "Mamba3 guard on CudaModelLoader.LoadFromSafetensors — that one covers the "
                    + "safetensors path only, this is the GGUF path). Use the CPU or Vulkan "
                    + "backend for Mamba3 checkpoints.");
```
with:
```csharp
            case Architecture.Mamba3:
                throw new NotSupportedException(
                    "Mamba3 has no GGUF tensor-naming convention on ANY dotLLM backend (CPU, Vulkan, "
                    + "or CUDA) — see docs/SUPPORTED_MODELS.md's 'No upstream GGUF mapping' note. "
                    + "Load Mamba3 checkpoints via CudaModelLoader.LoadMamba3FromSafetensors (CUDA), "
                    + "ModelLoader.LoadFromSafetensors (CPU), or VulkanMamba3TransformerModel.LoadFromSafetensors "
                    + "(Vulkan) instead.");
```
(This still throws `NotSupportedException` and still contains the literal string `"Mamba3"` — `CudaUnsupportedArchitectureGuardTests`'s existing `[InlineData(Architecture.Mamba3)]` case (Task 14 verifies, does not need to change) keeps passing unmodified.)

- [ ] **Step 4: Build**

Run: `dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj`
Expected: builds clean.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cuda/CudaModelLoader.cs
git commit -m "feat(cuda): add CudaModelLoader.LoadMamba3FromSafetensors, reword guards (#346)"
```

---

### Task 11: Synthetic CPU-vs-CUDA parity test (always-on CI, SISO)

**Files:**
- Create: `tests/DotLLM.Tests.Integration/Cuda/CudaMamba3ParitySyntheticTests.cs`

**Interfaces:**
- Consumes: `ModelLoader.LoadFromSafetensors` (CPU, existing), `CudaModelLoader.LoadMamba3FromSafetensors` (Task 10).

This is the first parity test that does not need an external checkpoint download — it synthesizes a deterministic tiny Mamba-3 checkpoint on disk, exactly like `TinyMamba3SafetensorsLoadTests.WriteSyntheticMamba3Checkpoint` (`tests/DotLLM.Tests.Integration/Models/Loaders/TinyMamba3SafetensorsLoadTests.cs:170-315`, read in full), then loads it into both the CPU and CUDA models and compares logits. Only `[Trait("Category", "GPU")]`-gated (skips without a CUDA device) — no `DOTLLM_*` env var needed, matching the codebase's existing convention of duplicating small fixture-writer helpers per test file rather than extracting a shared one (see `TinyMamba3SafetensorsLoadTests` vs `Mamba3TransformerModelTests`, which use the same shape tuple via independent code).

- [ ] **Step 1: Write the test**

Create `tests/DotLLM.Tests.Integration/Cuda/CudaMamba3ParitySyntheticTests.cs`:

```csharp
using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// CPU-vs-CUDA forward parity for <see cref="CudaMamba3TransformerModel"/> (issue #346)
/// on a deterministic synthetic checkpoint — no external download required, runs in
/// every CI pass with a CUDA device. Mirrors
/// <c>IbSsmMamba3VulkanGenerationTests.Mamba3_VulkanForward_MatchesCpuReference_OnPromptPrefill</c>'s
/// comparison methodology (single one-shot prefill, last-token-focused tolerance —
/// see that class's remarks for why growing-context reprefill is NOT used for Mamba-3
/// parity tests: there is no public state-reset API this test needs to work around
/// here since both sides load fresh models per test).
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaMamba3ParitySyntheticTests : IDisposable
{
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize;
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    // Absolute logit tolerance, matching IbSsmMamba3VulkanGenerationTests' calibration
    // for an F32-on-both-sides forward (pure reduction-order noise, no quantization).
    private const float LogitsAbsTol = 1e-2f;

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaMamba3ParitySyntheticTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mamba3-parity-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void CudaForward_MatchesCpuReference_OnSyntheticCheckpoint()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMamba3Checkpoint(modelPath, configPath);
        _output.WriteLine($"Synthesised tiny Mamba-3 checkpoint at: {modelPath}");

        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        try
        {
            int[] tokenIds = [0, 1, 2, 3, 5];
            int[] positions = [0, 1, 2, 3, 4];

            using ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1);
            using ITensor cudaLogits = cudaModel.Forward(tokenIds, positions, deviceId: -1);

            // CPU returns [seqLen, vocab] (per-position); CUDA returns [1, vocab]
            // (last token only) — same shape contract Vulkan uses. Compare the last row.
            Assert.Equal(2, cpuLogits.Shape.Rank);
            Assert.Equal(2, cudaLogits.Shape.Rank);
            Assert.Equal(tokenIds.Length, cpuLogits.Shape[0]);
            Assert.Equal(1, cudaLogits.Shape[0]);
            Assert.Equal(VocabSize, cpuLogits.Shape[1]);
            Assert.Equal(VocabSize, cudaLogits.Shape[1]);

            float[] cpuLast = ExtractRow(cpuLogits, tokenIds.Length - 1, VocabSize);
            float[] cudaLast = ExtractRow(cudaLogits, 0, VocabSize);

            float maxAbs = 0f;
            int worstIdx = 0;
            for (int i = 0; i < VocabSize; i++)
            {
                float diff = MathF.Abs(cpuLast[i] - cudaLast[i]);
                if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
            }
            _output.WriteLine(
                $"Last-token logit drift: max_abs={maxAbs:E3} at idx {worstIdx} "
                + $"(cpu={cpuLast[worstIdx]:G6} cuda={cudaLast[worstIdx]:G6})");

            int cpuArg = ArgMax(cpuLast);
            int cudaArg = ArgMax(cudaLast);
            _output.WriteLine($"Argmax: cpu={cpuArg} cuda={cudaArg}");

            Assert.True(maxAbs <= LogitsAbsTol,
                $"Last-token logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} at idx {worstIdx}.");
            Assert.Equal(cpuArg, cudaArg);
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    private static unsafe float[] ExtractRow(ITensor logits, int row, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, logits.Shape[0] * vocabSize);
        float[] result = new float[vocabSize];
        span.Slice(row * vocabSize, vocabSize).CopyTo(result);
        return result;
    }

    private static int ArgMax(float[] values)
    {
        int idx = 0;
        float best = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > best) { best = values[i]; idx = i; }
        return idx;
    }

    // ------------------------------------------------------------------
    // Fixture synthesis — same shape tuple and write strategy as
    // TinyMamba3SafetensorsLoadTests.WriteSyntheticMamba3Checkpoint.
    // ------------------------------------------------------------------

    private static void WriteSyntheticMamba3Checkpoint(string safetensorsPath, string configPath)
    {
        WriteConfigJson(configPath);
        WriteSafetensorsFixture(safetensorsPath);
    }

    private static void WriteConfigJson(string path)
    {
        using var fs = File.Create(path);
        using var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteString("model_type", "mamba3");
        writer.WriteNumber("hidden_size", HiddenSize);
        writer.WriteNumber("vocab_size", VocabSize);
        writer.WriteNumber("num_hidden_layers", NumLayers);
        writer.WriteNumber("num_heads", NumHeads);
        writer.WriteNumber("head_dim", HeadDim);
        writer.WriteNumber("expand", Expand);
        writer.WriteNumber("n_groups", 1);
        writer.WriteNumber("state_size", StateSize);
        writer.WriteNumber("chunk_size", 2);
        writer.WriteNumber("mimo_rank", 1);
        writer.WriteBoolean("is_mimo", false);
        writer.WriteBoolean("is_outproj_norm", false);
        writer.WriteBoolean("use_l2warp", false);
        writer.WriteBoolean("tie_word_embeddings", false);
        writer.WriteBoolean("rescale_prenorm_residual", true);
        writer.WriteBoolean("residual_in_fp32", true);
        writer.WriteNumber("A_floor", 1e-4);
        writer.WriteNumber("dt_init_floor", 1e-4);
        writer.WriteNumber("dt_min", 1e-3);
        writer.WriteNumber("dt_max", 0.1);
        writer.WriteNumber("norm_eps", 1e-5);
        writer.WriteNumber("rope_fraction", 0.5);
        writer.WriteNumber("max_position_embeddings", 32);
        writer.WriteEndObject();
    }

    private static void WriteSafetensorsFixture(string path)
    {
        var tensors = new List<(string Name, int[] Shape, float[] Values)>();

        AddSmall(tensors, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], amplitude: 0.05f, seed: 0);
        AddSmall(tensors, Mamba3TensorMapping.FinalNorm, [HiddenSize], amplitude: 0.5f, seed: 1);
        AddSmall(tensors, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], amplitude: 0.05f, seed: 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            AddSmall(tensors, Mamba3TensorMapping.LayerNorm(i), [HiddenSize], amplitude: 0.5f, seed: sBase + 0);
            AddSmall(tensors, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize], amplitude: 0.02f, seed: sBase + 1);
            AddSmall(tensors, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], amplitude: 0.05f, seed: sBase + 2);
            AddSmall(tensors, Mamba3TensorMapping.BNorm(i), [StateSize], amplitude: 0.5f, seed: sBase + 3);
            AddSmall(tensors, Mamba3TensorMapping.CNorm(i), [StateSize], amplitude: 0.5f, seed: sBase + 4);
            AddSmall(tensors, Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize], amplitude: 0.02f, seed: sBase + 5);
            AddSmall(tensors, Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize], amplitude: 0.02f, seed: sBase + 6);
            AddSmall(tensors, Mamba3TensorMapping.D(i), [NumHeads], amplitude: 0.1f, seed: sBase + 7);
            AddSmall(tensors, Mamba3TensorMapping.DtBias(i), [NumHeads], amplitude: 0.02f, seed: sBase + 8);
        }

        WriteSafetensorsFile(path, tensors);
    }

    private static void AddSmall(List<(string, int[], float[])> sink, string name, int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        sink.Add((name, shape, values));
    }

    private static void WriteSafetensorsFile(string path, List<(string Name, int[] Shape, float[] Values)> tensors)
    {
        using var headerMs = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerMs, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var (name, shape, values) in tensors)
            {
                long byteLen = values.Length * sizeof(float);
                w.WriteStartObject(name);
                w.WriteString("dtype", "F32");
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (int d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + byteLen);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += byteLen;
            }
            w.WriteEndObject();
        }
        byte[] headerBytes = headerMs.ToArray();

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerBytes.Length);
        fs.Write(prefix);
        fs.Write(headerBytes);

        foreach (var (_, _, values) in tensors)
        {
            byte[] bytes = new byte[values.Length * sizeof(float)];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
            fs.Write(bytes);
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj --filter FullyQualifiedName~CudaMamba3ParitySyntheticTests`
Expected: PASS on a CUDA-equipped host.

- [ ] **Step 3: Commit**

```bash
git add tests/DotLLM.Tests.Integration/Cuda/CudaMamba3ParitySyntheticTests.cs
git commit -m "test(cuda): add synthetic CPU-vs-CUDA Mamba3 parity test (#346)"
```

---

### Task 12: Gated real-weight parity test — `ib-ssm/mamba3-370M-10BT`, prefill + decode

**Files:**
- Create: `tests/DotLLM.Tests.Integration/Cuda/IbSsmMamba3CudaParityTests.cs`

**Interfaces:**
- Consumes: `CudaMamba3TransformerModel.Forward(..., CudaMamba3StateCache)` (Task 9), `CudaMamba3StateCache` (Task 7).

This is the acceptance-criteria-mandated "real-model CPU vs CUDA parity test (prefill + decode)". It combines the checkpoint-resolution pattern from `IbSsmMamba3RealWeightsLoadTests.cs` (`DOTLLM_IBSSM_CHECKPOINT_PATH` env var / conventional paths / `Skip.If`) with the state-threaded prefill+decode methodology from that same file's `DecodeMatchesPrefillOnRealCheckpoint` test AND the CPU-vs-backend comparison structure from `IbSsmMamba3VulkanGenerationTests.cs` — both read in full during planning. Per this plan's "SISO before MIMO" constraint, this only exercises the real (SISO) checkpoint; MIMO gets its own synthetic-only test in Task 14.

- [ ] **Step 1: Write the test**

Create `tests/DotLLM.Tests.Integration/Cuda/IbSsmMamba3CudaParityTests.cs`:

```csharp
using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Real-weight CPU-vs-CUDA forward parity for <see cref="CudaMamba3TransformerModel"/>
/// on <c>ib-ssm/mamba3-370M-10BT</c> (issue #346's acceptance criterion: "Real-model
/// CPU vs CUDA parity test (prefill + decode)"). Combines
/// <c>IbSsmMamba3RealWeightsLoadTests</c>'s checkpoint-resolution + state-threaded
/// prefill/decode methodology with <c>IbSsmMamba3VulkanGenerationTests</c>'s
/// cross-backend comparison structure — both read in full while planning this test.
/// </summary>
/// <remarks>
/// <b>Gating.</b> Same as the CPU/Vulkan siblings: <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c>
/// env var, then <c>C:/temp/dotllm-ibssm/model.safetensors</c>, then
/// <c>%USERPROFILE%/dotllm-ibssm-370m/model.safetensors</c>. Skips gracefully if none
/// resolve. Additionally requires a CUDA device.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class IbSsmMamba3CudaParityTests
{
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string SafetensorsName = "model.safetensors";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    private const float LogitsAbsTol = 3.0f; // matches IbSsmMamba3VulkanGenerationTests

    private readonly ITestOutputHelper _output;
    public IbSsmMamba3CudaParityTests(ITestOutputHelper output) => _output = output;

    private static string? ResolveCheckpointPath()
    {
        string? env = Environment.GetEnvironmentVariable(CheckpointPathEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (File.Exists(env)) return env;
            if (Directory.Exists(env))
            {
                string candidate = Path.Combine(env, SafetensorsName);
                if (File.Exists(candidate)) return candidate;
            }
        }

        string conventional = Path.Combine(ConventionalDir, SafetensorsName);
        if (File.Exists(conventional)) return conventional;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrWhiteSpace(home))
        {
            string fallback = Path.Combine(home, UserProfileFallbackDir, SafetensorsName);
            if (File.Exists(fallback)) return fallback;
        }
        return null;
    }

    [SkippableFact]
    public void CudaForward_MatchesCpuReference_OnPromptPrefill()
    {
        string? checkpointPath = ResolveCheckpointPath();
        Skip.If(checkpointPath is null,
            $"ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
            + $"to the safetensors file or its directory, or place it at {ConventionalDir}/{SafetensorsName}.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        var cpuLoadWatch = Stopwatch.StartNew();
        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(checkpointPath!);
        cpuLoadWatch.Stop();
        Assert.Equal(Architecture.Mamba3, config.Architecture);
        _output.WriteLine($"CPU load: {cpuLoadWatch.Elapsed.TotalSeconds:F1} s");

        var cudaLoadWatch = Stopwatch.StartNew();
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(checkpointPath!);
        cudaLoadWatch.Stop();
        _output.WriteLine($"CUDA load: {cudaLoadWatch.Elapsed.TotalSeconds:F1} s");

        try
        {
            int[] tokenIds = [0, 100, 1000, 10000, 31999];
            int[] positions = [0, 1, 2, 3, 4];

            using ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1);
            using ITensor cudaLogits = cudaModel.Forward(tokenIds, positions, deviceId: -1);

            float[] cpuLast = LastRow(cpuLogits, config.VocabSize);
            float[] cudaLast = LastRow(cudaLogits, config.VocabSize);

            AssertLogitsMatch(cpuLast, cudaLast, "prefill");
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    [SkippableFact]
    public void CudaDecode_MatchesCpuReference_PrefillThenDecode()
    {
        string? checkpointPath = ResolveCheckpointPath();
        Skip.If(checkpointPath is null,
            $"ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar}.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(checkpointPath!);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(checkpointPath!);
        try
        {
            var cpuM3 = Assert.IsType<Mamba3TransformerModel>(cpuModel);

            int[] tokenIds = [0, 100, 1000];
            int[] positions = [0, 1, 2];
            int vocabSize = config.VocabSize;

            // CPU: prefill 2 + decode 1, state-threaded (mirrors DecodeMatchesPrefillOnRealCheckpoint).
            using var cpuState = new Mamba3State(config);
            using ITensor cpuSplitPrefill = cpuM3.Forward(tokenIds.AsSpan(0, 2), positions.AsSpan(0, 2), deviceId: -1, cpuState);
            using ITensor cpuDecodeTail = cpuM3.Forward(tokenIds.AsSpan(2, 1), positions.AsSpan(2, 1), deviceId: -1, cpuState);
            float[] cpuLast = LastRow(cpuDecodeTail, vocabSize);

            // CUDA: same split, state-threaded via CudaMamba3StateCache. deviceId=-1 here means
            // "host-resident output tensor" (see ForwardCore's doc comment, Task 9) — it does NOT
            // select which GPU runs the compute; that was fixed at LoadMamba3FromSafetensors time.
            using var cudaState = new CudaMamba3StateCache(config.Mamba3Config!, config.NumLayers);
            using ITensor cudaSplitPrefill = cudaModel.Forward(tokenIds.AsSpan(0, 2), positions.AsSpan(0, 2), deviceId: -1, cudaState);
            using ITensor cudaDecodeTail = cudaModel.Forward(tokenIds.AsSpan(2, 1), positions.AsSpan(2, 1), deviceId: -1, cudaState);
            float[] cudaLast = LastRow(cudaDecodeTail, vocabSize);

            AssertLogitsMatch(cpuLast, cudaLast, "prefill+decode");
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    private void AssertLogitsMatch(float[] cpuLast, float[] cudaLast, string label)
    {
        foreach (float v in cpuLast) Assert.True(float.IsFinite(v), $"[{label}] CPU logits contain NaN/Inf.");
        foreach (float v in cudaLast) Assert.True(float.IsFinite(v), $"[{label}] CUDA logits contain NaN/Inf.");

        float maxAbs = 0f;
        int worstIdx = 0;
        for (int i = 0; i < cpuLast.Length; i++)
        {
            float diff = MathF.Abs(cpuLast[i] - cudaLast[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }

        int cpuArg = ArgMax(cpuLast);
        int cudaArg = ArgMax(cudaLast);
        _output.WriteLine(
            $"[{label}] max_abs={maxAbs:E3} at idx {worstIdx} "
            + $"(cpu={cpuLast[worstIdx]:G6} cuda={cudaLast[worstIdx]:G6}); "
            + $"argmax: cpu={cpuArg} cuda={cudaArg}");

        Assert.True(maxAbs <= LogitsAbsTol,
            $"[{label}] L-inf logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} at idx {worstIdx}.");
        Assert.Equal(cpuArg, cudaArg);
    }

    private static unsafe float[] LastRow(ITensor logits, int vocabSize)
    {
        int seqLen = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocabSize);
        float[] result = new float[vocabSize];
        span.Slice((seqLen - 1) * vocabSize, vocabSize).CopyTo(result);
        return result;
    }

    private static int ArgMax(float[] values)
    {
        int idx = 0;
        float best = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > best) { best = values[i]; idx = i; }
        return idx;
    }
}
```

- [ ] **Step 2: Run locally (requires the real checkpoint)**

```powershell
$env:DOTLLM_IBSSM_CHECKPOINT_PATH = "C:/temp/dotllm-ibssm/model.safetensors"
dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj --filter FullyQualifiedName~IbSsmMamba3CudaParityTests
```
Expected: PASS if the checkpoint is present on a CUDA host; SKIP (not FAIL) otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/DotLLM.Tests.Integration/Cuda/IbSsmMamba3CudaParityTests.cs
git commit -m "test(cuda): add gated real-weight ib-ssm Mamba3 prefill+decode parity test (#346)"
```

---

### Task 13: `mamba3_ssd_scan_mimo_f32` CUDA kernel + launcher, unit-tested against `Mamba3CanonicalSsd.ExecuteMimo`

**Files:**
- Create: `native/kernels/mamba3_ssd_scan_mimo_f32.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Create: `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3SsdScanMimoF32Tests.cs`

**Interfaces:**
- Produces: PTX symbol `mamba3_ssd_scan_mimo_f32`; `public bool HasMamba3SsdScanMimo { get; }` and
  `public void LaunchMamba3SsdScanMimoF32(nint state, nint v, nint qRoped, nint kRoped, nint qkPreDotSum, nint scale, nint gamma, nint adt, nint d, nint z, nint mimoZ, nint mimoO, nint y, int seqLen, int nRank, int nHead, int headDim, int dState, bool hasZ, nint stream)` — consumed by Task 14 and this task's test.

Based directly on the CPU-authoritative `Mamba3CanonicalSsd.ExecuteMimo` (`src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs:242-389`, read in full during planning), NOT reconstructed from a paraphrase of the Vulkan MIMO shader. Same one-block-per-head, thread-strides-over-`headDim`(`p`) structure as the SISO kernel (Task 5). CPU nests two separate loops over `p` per `(t,h)` — a state-update loop then a readout/gate/contraction loop — but since each thread owns an exclusive `p` throughout, and no thread ever reads another thread's `state[p,:]` row, this plan fuses them into a single per-thread `p` loop (state-update-then-readout for that `p`) without changing the math: within one thread, sequential execution already guarantees every `n` of that thread's state row is updated before the readout reads it, which is the only ordering the CPU's two-loop split actually enforces.

- [ ] **Step 1: Write the kernel**

Create `native/kernels/mamba3_ssd_scan_mimo_f32.cu`:

```c
// mamba3_ssd_scan_mimo_f32.cu — Mamba-3 canonical MIMO SSD scan (issue #346).
// Direct CUDA-C translation of the CPU-authoritative
// DotLLM.Cpu.Kernels.Mamba3CanonicalSsd.ExecuteMimo (src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs).
// One CUDA block per head, threads stride over headDim (p); each thread's
// state-update-then-readout for its own p is fused into one loop (see this
// task's plan note for why this preserves the CPU's ordering guarantee).
// NO_FMA (see build_ptx.bat) for CPU bit-parity.
//
// state:  [nHead, headDim, dState] F32, mutated in place, rank-free (K is
//         summed over rank INSIDE the state update — matches ExecuteMimo's
//         "h is shape [nHead,headDim,dState], no rank dim" contract).
// v:      [seqLen, nHead, headDim] F32.
// qRoped/kRoped: [seqLen, nRank, nHead, dState] F32 (post-RoPE C / B per rank).
// qkPreDotSum, scale, gamma, adt: [seqLen, nHead] F32 (host-precomputed).
// d: [nHead] F32. z: [seqLen, nHead, headDim] F32 or nullptr. mimoZ/mimoO: [nHead, nRank, headDim] F32.
// y: [seqLen, nHead, headDim] F32 output (rank-contracted).

#define SSD_MIMO_WG_SIZE 256

extern "C" __global__ void __launch_bounds__(SSD_MIMO_WG_SIZE) mamba3_ssd_scan_mimo_f32(
    float* __restrict__ state,
    const float* __restrict__ v,
    const float* __restrict__ qRoped,
    const float* __restrict__ kRoped,
    const float* __restrict__ qkPreDotSum,
    const float* __restrict__ scale,
    const float* __restrict__ gamma,
    const float* __restrict__ adt,
    const float* __restrict__ d,
    const float* __restrict__ z,
    const float* __restrict__ mimoZ,
    const float* __restrict__ mimoO,
    float* __restrict__ y,
    const int seqLen, const int nRank, const int nHead, const int headDim, const int dState, const int hasZ)
{
    __shared__ float decayShared, sclShared, gmShared, qkpShared, skipShared;

    const int h = blockIdx.x;
    if (h >= nHead) return;
    const int tid = threadIdx.x;
    const int stateHeadBase = h * headDim * dState;
    const float dh = d[h];
    const float invRank = 1.0f / (float)nRank;

    const int bcHeadStride = dState;
    const int bcRankStride = nHead * dState;
    const int bcTokStride = nRank * bcRankStride;
    const int mimoHeadStride = nRank * headDim;
    const int mimoRankStride = headDim;

    for (int t = 0; t < seqLen; t++)
    {
        if (tid == 0)
        {
            int hdrIdx = t * nHead + h;
            decayShared = expf(adt[hdrIdx]);
            sclShared = scale[hdrIdx];
            gmShared = gamma[hdrIdx];
            qkpShared = qkPreDotSum[hdrIdx];
            skipShared = dh + gmShared * qkpShared;
        }
        __syncthreads();

        float decay = decayShared, scl = sclShared, skip = skipShared;
        int vTokBase = t * nHead * headDim;
        int bcTokBase = t * bcTokStride;

        for (int p = tid; p < headDim; p += SSD_MIMO_WG_SIZE)
        {
            int vIdx = vTokBase + h * headDim + p;
            float vp = v[vIdx];
            int stateRowBase = stateHeadBase + p * dState;

            // State update: h_new[p,n] = decay*h_old[p,n] + vp*(sum_r kRoped[t,r,h,n])*scl.
            for (int n = 0; n < dState; n++)
            {
                float kSum = 0.0f;
                for (int r = 0; r < nRank; r++)
                {
                    int kIdx = bcTokBase + r * bcRankStride + h * bcHeadStride + n;
                    kSum += kRoped[kIdx];
                }
                float newState = decay * state[stateRowBase + n] + vp * (kSum * scl);
                state[stateRowBase + n] = newState;
            }

            // Per-rank readout + gate + rank contraction (same thread, same p — the
            // state row this thread just finished writing is the one it reads here).
            float contracted = 0.0f;
            for (int r = 0; r < nRank; r++)
            {
                int qBase = bcTokBase + r * bcRankStride + h * bcHeadStride;
                float yScanR = 0.0f;
                for (int n = 0; n < dState; n++)
                    yScanR += qRoped[qBase + n] * state[stateRowBase + n];

                float yR = yScanR + skip * invRank * vp;

                if (hasZ)
                {
                    int mimoZIdx = h * mimoHeadStride + r * mimoRankStride + p;
                    float zGated = z[vIdx] * mimoZ[mimoZIdx];
                    float silu = zGated / (1.0f + expf(-zGated));
                    yR *= silu;
                }

                int mimoOIdx = h * mimoHeadStride + r * mimoRankStride + p;
                contracted += yR * mimoO[mimoOIdx];
            }
            y[vIdx] = contracted;
        }
        __syncthreads();
    }
}
```

- [ ] **Step 2: Build PTX**

Run: `native\build_ptx.bat` (`NO_FMA` entry already added in Task 1 Step 2).
Expected: `native\ptx\mamba3_ssd_scan_mimo_f32.ptx` generated, no errors.

- [ ] **Step 3: Add the `CudaKernels.cs` loader + launcher (Tier-2 optional pattern)**

Fields:
```csharp
    // Issue #346 (Mamba3 CUDA host): canonical MIMO SSD scan.
    private CudaModule? _mamba3SsdScanMimoF32Module;
    private nint _mamba3SsdScanMimoF32Func;
```

Constructor load block:
```csharp
        string mamba3SsdScanMimoF32Path = Path.Combine(ptxDir, "mamba3_ssd_scan_mimo_f32.ptx");
        if (File.Exists(mamba3SsdScanMimoF32Path))
        {
            _mamba3SsdScanMimoF32Module = CudaModule.LoadFromFile(mamba3SsdScanMimoF32Path);
            _mamba3SsdScanMimoF32Func = _mamba3SsdScanMimoF32Module.TryGetFunction("mamba3_ssd_scan_mimo_f32");
        }
```

Capability flag:
```csharp
    /// <summary>True when the Mamba-3 canonical MIMO SSD scan kernel (issue #346) is loaded.</summary>
    public bool HasMamba3SsdScanMimo => _mamba3SsdScanMimoF32Func != 0;
```

Launcher:
```csharp
    /// <summary>
    /// Mamba-3 canonical MIMO SSD scan (issue #346). Same one-block-per-head,
    /// covers-all-seqLen-in-one-launch structure as
    /// <see cref="LaunchMamba3SsdScanSisoF32"/>, extended with a rank axis
    /// <paramref name="nRank"/> (state stays rank-free; K is rank-summed inside the
    /// state update, matching <c>Mamba3CanonicalSsd.ExecuteMimo</c>).
    /// </summary>
    public void LaunchMamba3SsdScanMimoF32(nint state, nint v, nint qRoped, nint kRoped,
        nint qkPreDotSum, nint scale, nint gamma, nint adt, nint d, nint z, nint mimoZ, nint mimoO, nint y,
        int seqLen, int nRank, int nHead, int headDim, int dState, bool hasZ, nint stream)
    {
        if (_mamba3SsdScanMimoF32Func == 0)
            throw new InvalidOperationException(
                "mamba3_ssd_scan_mimo_f32 kernel not available. Recompile native/kernels/mamba3_ssd_scan_mimo_f32.cu to PTX.");

        nint stateArg = state, vArg = v, qArg = qRoped, kArg = kRoped, qkpArg = qkPreDotSum;
        nint sclArg = scale, gmArg = gamma, adtArg = adt, dArg = d, zArg = z, mzArg = mimoZ, moArg = mimoO, yArg = y;
        int seqArg = seqLen, rankArg = nRank, headArg = nHead, hdArg = headDim, dsArg = dState, hasZArg = hasZ ? 1 : 0;

        void** args = stackalloc void*[] {
            &stateArg, &vArg, &qArg, &kArg, &qkpArg, &sclArg, &gmArg, &adtArg, &dArg, &zArg, &mzArg, &moArg, &yArg,
            &seqArg, &rankArg, &headArg, &hdArg, &dsArg, &hasZArg };

        CudaDriverApi.cuLaunchKernel(_mamba3SsdScanMimoF32Func,
                (uint)nHead, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
```

`Dispose()` addition: `_mamba3SsdScanMimoF32Module?.Dispose();`

- [ ] **Step 4: Write the unit test against `Mamba3CanonicalSsd.ExecuteMimo`**

Create `tests/DotLLM.Tests.Unit/Cuda/CudaMamba3SsdScanMimoF32Tests.cs`, structurally identical to `CudaMamba3SsdScanSisoF32Tests.cs` (Task 6) but calling `Mamba3CanonicalSsd.ExecuteMimo` as the CPU oracle and allocating the rank-extended `qRoped`/`kRoped`/`mimoZ`/`mimoO` buffers:

```csharp
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3SsdScanMimoF32"/> against
/// its CPU oracle, <see cref="Mamba3CanonicalSsd.ExecuteMimo"/>. Issue #346.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMamba3SsdScanMimoF32Tests
{
    private readonly ITestOutputHelper _out;
    public CudaMamba3SsdScanMimoF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

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
    [InlineData(3, 4, 4, 8, 5, true)]   // tiny MIMO shape, rank=3, with gate
    [InlineData(2, 4, 4, 8, 5, false)]  // tiny MIMO shape, rank=2, no gate
    public void Mamba3SsdScanMimoF32_MatchesCpuReference(
        int nRank, int nHead, int headDim, int dState, int seqLen, bool hasZ)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3SsdScanMimo, "mamba3_ssd_scan_mimo_f32 PTX symbol not found (stale build)");

        var rng = new Random(0x3310 ^ nRank ^ nHead ^ (headDim << 4) ^ (dState << 12) ^ (seqLen << 20));

        int stateLen = nHead * headDim * dState;
        int vLen = seqLen * nHead * headDim;
        int bcLen = seqLen * nRank * nHead * dState;
        int hdrLen = seqLen * nHead;
        int mimoLen = nHead * nRank * headDim;

        float[] stateCpu = RandomArray(rng, stateLen);
        float[] stateGpu = (float[])stateCpu.Clone();
        float[] v = RandomArray(rng, vLen);
        float[] qRoped = RandomArray(rng, bcLen);
        float[] kRoped = RandomArray(rng, bcLen);
        float[] qkPreDotSum = RandomArray(rng, hdrLen);
        float[] scale = RandomArray(rng, hdrLen);
        float[] gamma = RandomArray(rng, hdrLen);
        float[] adt = new float[hdrLen];
        for (int i = 0; i < hdrLen; i++) adt[i] = -(float)rng.NextDouble() * 2f;
        float[] d = RandomArray(rng, nHead);
        float[] z = hasZ ? RandomArray(rng, vLen) : Array.Empty<float>();
        float[] mimoZ = RandomArray(rng, mimoLen);
        float[] mimoO = RandomArray(rng, mimoLen);
        float[] yCpu = new float[vLen];

        Mamba3CanonicalSsd.ExecuteMimo(
            stateCpu, v, qRoped, kRoped, qkPreDotSum, scale, gamma, adt, d, z, mimoZ, mimoO,
            yCpu, yPerRank: Span<float>.Empty, seqLen, nRank, nHead, headDim, dState);

        nint dSt = 0, dV = 0, dQ = 0, dK = 0, dQkp = 0, dScl = 0, dGm = 0, dAdt = 0, dD = 0, dZ = 0, dMz = 0, dMo = 0, dY = 0;
        try
        {
            long stateBytes = (long)stateLen * sizeof(float);
            long vBytes = (long)vLen * sizeof(float);
            long bcBytes = (long)bcLen * sizeof(float);
            long hdrBytes = (long)hdrLen * sizeof(float);
            long dBytes = (long)nHead * sizeof(float);
            long mimoBytes = (long)mimoLen * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dSt, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQkp, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dScl, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGm, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAdt, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)dBytes).ThrowOnError();
            if (hasZ) CudaDriverApi.cuMemAlloc_v2(out dZ, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dMz, (nuint)mimoBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dMo, (nuint)mimoBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)vBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyHtoD_v2(dSt, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = qRoped) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = kRoped) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = qkPreDotSum) CudaDriverApi.cuMemcpyHtoD_v2(dQkp, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = scale) CudaDriverApi.cuMemcpyHtoD_v2(dScl, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = gamma) CudaDriverApi.cuMemcpyHtoD_v2(dGm, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = adt) CudaDriverApi.cuMemcpyHtoD_v2(dAdt, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)dBytes).ThrowOnError();
                if (hasZ) fixed (float* p = z) CudaDriverApi.cuMemcpyHtoD_v2(dZ, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = mimoZ) CudaDriverApi.cuMemcpyHtoD_v2(dMz, (nint)p, (nuint)mimoBytes).ThrowOnError();
                fixed (float* p = mimoO) CudaDriverApi.cuMemcpyHtoD_v2(dMo, (nint)p, (nuint)mimoBytes).ThrowOnError();
            }

            kernels.LaunchMamba3SsdScanMimoF32(dSt, dV, dQ, dK, dQkp, dScl, dGm, dAdt, dD, dZ, dMz, dMo, dY,
                seqLen, nRank, nHead, headDim, dState, hasZ, stream.Handle);
            stream.Synchronize();

            float[] yGpu = new float[vLen];
            float[] stateGpuOut = new float[stateLen];
            unsafe
            {
                fixed (float* p = yGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)vBytes).ThrowOnError();
                fixed (float* p = stateGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dSt, (nuint)stateBytes).ThrowOnError();
            }

            Assert.True(yCpu.AsSpan().SequenceEqual(yGpu), "y output mismatch.");
            Assert.True(stateCpu.AsSpan().SequenceEqual(stateGpuOut), "final ssm_state mismatch.");
            _out.WriteLine($"nRank={nRank} nHead={nHead} headDim={headDim} dState={dState} seqLen={seqLen} hasZ={hasZ}: exact match.");
        }
        finally
        {
            if (dSt != 0) CudaDriverApi.cuMemFree_v2(dSt);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dQkp != 0) CudaDriverApi.cuMemFree_v2(dQkp);
            if (dScl != 0) CudaDriverApi.cuMemFree_v2(dScl);
            if (dGm != 0) CudaDriverApi.cuMemFree_v2(dGm);
            if (dAdt != 0) CudaDriverApi.cuMemFree_v2(dAdt);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dZ != 0) CudaDriverApi.cuMemFree_v2(dZ);
            if (dMz != 0) CudaDriverApi.cuMemFree_v2(dMz);
            if (dMo != 0) CudaDriverApi.cuMemFree_v2(dMo);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
        }
    }

    private static float[] RandomArray(Random rng, int len)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
```

- [ ] **Step 5: Run the tests**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --filter FullyQualifiedName~CudaMamba3SsdScanMimoF32Tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add native/kernels/mamba3_ssd_scan_mimo_f32.cu native/ptx/mamba3_ssd_scan_mimo_f32.ptx src/DotLLM.Cuda/CudaKernels.cs tests/DotLLM.Tests.Unit/Cuda/CudaMamba3SsdScanMimoF32Tests.cs
git commit -m "feat(cuda): add mamba3_ssd_scan_mimo_f32 kernel + launcher + CPU-parity test (#346)"
```

---

### Task 14: `CudaMamba3TransformerModel` MIMO forward wiring + synthetic MIMO parity test

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`
- Create: `tests/DotLLM.Tests.Integration/Cuda/CudaMamba3MimoParitySyntheticTests.cs`

**Interfaces:**
- Consumes: `LaunchMamba3SsdScanMimoF32` (Task 13), `LaunchMamba3DataRopeF32` with `mode=1` (Halved, Task 2 — already supports this, no change needed), `LaunchMamba3ChunkBoundaryF32` with `nRank>1` (Task 4 — already rank-parameterized, no change needed).

**No public checkpoint exists for MIMO** — mirrors CPU/Vulkan's own coverage exactly (`docs/ROADMAP.md` step 60f: "No public MIMO checkpoint — real-weight MIMO verification deferred P4.3"). This task's only correctness evidence is the Task 13 kernel-level unit test (against `Mamba3CanonicalSsd.ExecuteMimo`) plus this task's synthetic end-to-end fixture test — there is no gated real-weight MIMO equivalent of Task 12 to add.

- [ ] **Step 1: Replace `ForwardCore`'s IsMimo guard with a per-layer branch**

In `src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs`'s `ForwardCore` (Task 9, Step 3), remove:
```csharp
        if (_m3.IsMimo)
            throw new NotSupportedException(
                "CudaMamba3TransformerModel.Forward (SISO path) does not support IsMimo=true "
                + "checkpoints yet — see ForwardMimo (issue #346 Task 14).");
```
and replace the per-layer block's Steps 3-6.5 (RoPE / chunk-boundary / scan / persist) with a branch mirroring CPU `Mamba3TransformerModel.ForwardCore`'s `if (isMimo) {...} else {...}` structure (`src/DotLLM.Models/Architectures/Mamba3TransformerModel.cs:414-488`). The `int effRank = _m3.IsMimo ? _m3.MimoRank : 1;` local (added at the top of `ForwardCore`, alongside the other dimension locals from Task 9 Step 3) threads through both branches so `EnsureScratchCapacity`'s already-`effRank`-aware `_bDevice`/`_cDevice` sizing (Task 9 Step 1) needs no change:

```csharp
                int effRank = _m3.IsMimo ? _m3.MimoRank : 1;

                if (_m3.IsMimo)
                {
                    HostPrepareMimo(seqLen, dInProj, dInner, nHead, dState, numRopeAngles, effRank, aFloor, eps,
                        lw.DtBias, lw.BNorm, lw.CNorm, lw.BBias, lw.CBias, s);

                    _kernels.LaunchMamba3DataRopeF32(_bDevice, _cDevice, _anglesRawDevice, _dtDevice,
                        state.GetCumAnglePtr(layer), state.GetCumAnglePtr(layer),
                        seqLen, effRank, nHead, dState, numRopeAngles, mode: 1 /* Halved */,
                        hasCumPrev: true, writeCumOut: true, s);

                    if (runChunkBoundary)
                    {
                        _kernels.LaunchMamba3ChunkBoundaryF32(
                            state.GetSsmStatePtr(layer), state.GetVStatePtr(layer), state.GetKStatePtr(layer),
                            _coefDevice, nHead, headDim, dState, effRank, s);
                    }

                    _kernels.LaunchMamba3SsdScanMimoF32(
                        state.GetSsmStatePtr(layer), _xDevice, _cDevice, _bDevice,
                        _qkPreDotDevice, _scaleDevice, _gammaDevice, _adtDevice, lw.D, _zDevice,
                        lw.MimoZ, lw.MimoO, _yScanDevice,
                        seqLen, effRank, nHead, headDim, dState, hasZ: true, s);

                    if (runChunkBoundary)
                    {
                        long kBytes = (long)effRank * nHead * dState * sizeof(float);
                        long vBytes = (long)nHead * headDim * sizeof(float);
                        // kRoped layout [T, R, H, N] — the whole last-token [R, H, N]
                        // slice (all ranks) is contiguous, matching Mamba3CanonicalSsd
                        // .ExecuteMimoStreaming's per-rank k_state persist.
                        nint lastKSrc = _cDevice + (nint)((long)(seqLen - 1) * effRank * nHead * dState * sizeof(float));
                        nint lastVSrc = _xDevice + (nint)((long)(seqLen - 1) * nHead * headDim * sizeof(float));
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetKStatePtr(layer), lastKSrc, (nuint)kBytes, s).ThrowOnError();
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetVStatePtr(layer), lastVSrc, (nuint)vBytes, s).ThrowOnError();
                    }
                }
                else
                {
                    HostPrepareSiso(seqLen, dInProj, dInner, nHead, dState, numRopeAngles, aFloor, eps,
                        lw.DtBias, lw.BNorm, lw.CNorm, lw.BBias, lw.CBias, s);

                    _kernels.LaunchMamba3DataRopeF32(_bDevice, _cDevice, _anglesRawDevice, _dtDevice,
                        state.GetCumAnglePtr(layer), state.GetCumAnglePtr(layer),
                        seqLen, nRank: 1, nHead, dState, numRopeAngles, mode: 0 /* Pairwise */,
                        hasCumPrev: true, writeCumOut: true, s);

                    if (runChunkBoundary)
                    {
                        _kernels.LaunchMamba3ChunkBoundaryF32(
                            state.GetSsmStatePtr(layer), state.GetVStatePtr(layer), state.GetKStatePtr(layer),
                            _coefDevice, nHead, headDim, dState, nRank: 1, s);
                    }

                    _kernels.LaunchMamba3SsdScanSisoF32(
                        state.GetSsmStatePtr(layer), _xDevice, _cDevice, _bDevice,
                        _qkPreDotDevice, _scaleDevice, _gammaDevice, _adtDevice, lw.D, _zDevice, _yScanDevice,
                        seqLen, nHead, headDim, dState, hasZ: true, s);

                    if (runChunkBoundary)
                    {
                        long kBytes = (long)nHead * dState * sizeof(float);
                        long vBytes = (long)nHead * headDim * sizeof(float);
                        nint lastKSrc = _cDevice + (nint)((long)(seqLen - 1) * nHead * dState * sizeof(float));
                        nint lastVSrc = _xDevice + (nint)((long)(seqLen - 1) * nHead * headDim * sizeof(float));
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetKStatePtr(layer), lastKSrc, (nuint)kBytes, s).ThrowOnError();
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetVStatePtr(layer), lastVSrc, (nuint)vBytes, s).ThrowOnError();
                    }
                }
```
(This is Task 9 Step 3's SISO-only block, unchanged, now under the `else`.) The `_stream.Synchronize()` call between the in-proj GEMM and `HostPrepare*` (Task 9 Step 3) stays exactly where it was — both branches need the downloaded `proj` regardless of SISO/MIMO.

- [ ] **Step 2: Add `HostPrepareMimo` — port of `Mamba3Block.ForwardMimo` Steps 2-4**

Add alongside `HostPrepareSiso` (same class), following `Mamba3Block.ForwardMimo`'s split/softplus/sigmoid/RMSNorm+bias/qk_pre_dot-sum math exactly (`src/DotLLM.Models/Architectures/Mamba3Block.cs:572-665`):

```csharp
    /// <summary>
    /// MIMO analog of <see cref="HostPrepareSiso"/> — line-for-line port of
    /// <c>Mamba3Block.ForwardMimo</c>'s Steps 2-4. Differs from SISO in: <c>bcPerToken</c>
    /// includes the rank factor, B/C are laid out <c>[T, R, H, N]</c> (RmsNorm+bias applied
    /// per <c>(r, g)</c> slice with bias shape <c>[H, R, N]</c>), and <c>qkPreDot</c> sums
    /// the pre-rotation dot over rank (<c>qkPreDotSum</c>).
    /// </summary>
    private void HostPrepareMimo(int seqLen, int dInProj, int dInner, int nHead, int dState,
        int numRopeAngles, int mimoRank, float aFloor, float normEps,
        nint dtBiasDevice, nint bNormDevice, nint cNormDevice, nint bBiasDevice, nint cBiasDevice,
        nint stream)
    {
        int r_ = mimoRank;
        int bcPerToken = dState * r_; // numBcHeads=1 on every known checkpoint

        float[] proj = new float[seqLen * dInProj];
        fixed (float* p = proj)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, _projDevice, (nuint)(proj.Length * sizeof(float))).ThrowOnError();

        float[] dtBias = DownloadF32(dtBiasDevice, nHead);
        float[] bNormW = DownloadF32(bNormDevice, dState);
        float[] cNormW = DownloadF32(cNormDevice, dState);
        float[] bBias = DownloadF32(bBiasDevice, nHead * r_ * dState);   // [H, R, N]
        float[] cBias = DownloadF32(cBiasDevice, nHead * r_ * dState);

        int ofsZ = 0, ofsX = dInner, ofsB = 2 * dInner, ofsC = ofsB + bcPerToken;
        int ofsDdDt = ofsC + bcPerToken, ofsDdA = ofsDdDt + nHead, ofsTrap = ofsDdA + nHead, ofsAngles = ofsTrap + nHead;

        var x = new float[seqLen * dInner];
        var z = new float[seqLen * dInner];
        var dt = new float[seqLen * nHead];
        var adt = new float[seqLen * nHead];
        var trap = new float[seqLen * nHead];
        var gamma = new float[seqLen * nHead];
        var scale = new float[seqLen * nHead];
        var anglesRaw = new float[seqLen * numRopeAngles];
        var bRHN = new float[seqLen * r_ * nHead * dState];
        var cRHN = new float[seqLen * r_ * nHead * dState];
        var qkPreDotSum = new float[seqLen * nHead];

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;
            Array.Copy(proj, src + ofsZ, z, t * dInner, dInner);
            Array.Copy(proj, src + ofsX, x, t * dInner, dInner);

            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];
                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;
                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;
            }

            Array.Copy(proj, src + ofsAngles, anglesRaw, t * numRopeAngles, numRopeAngles);

            for (int rr = 0; rr < r_; rr++)
            {
                int bSrcBase = src + ofsB + rr * dState;
                int cSrcBase = src + ofsC + rr * dState;
                RmsNormFactor(proj, bSrcBase, dState, normEps, out float bInvRms);
                RmsNormFactor(proj, cSrcBase, dState, normEps, out float cInvRms);
                for (int h = 0; h < nHead; h++)
                {
                    int biasBase = (h * r_ + rr) * dState;
                    int dstBase = ((t * r_ + rr) * nHead + h) * dState;
                    for (int n = 0; n < dState; n++)
                    {
                        bRHN[dstBase + n] = proj[bSrcBase + n] * bInvRms * bNormW[n] + bBias[biasBase + n];
                        cRHN[dstBase + n] = proj[cSrcBase + n] * cInvRms * cNormW[n] + cBias[biasBase + n];
                    }
                }
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sum = 0f;
                for (int rr = 0; rr < r_; rr++)
                {
                    int baseIdx = ((t * r_ + rr) * nHead + h) * dState;
                    for (int n = 0; n < dState; n++) sum += cRHN[baseIdx + n] * bRHN[baseIdx + n];
                }
                qkPreDotSum[t * nHead + h] = sum;
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }

        var coef = new float[nHead];
        for (int h = 0; h < nHead; h++) coef[h] = dt[h] * (1f - trap[h]);

        UploadF32Array(x, _xDevice, stream);
        UploadF32Array(z, _zDevice, stream);
        UploadF32Array(dt, _dtDevice, stream);
        UploadF32Array(adt, _adtDevice, stream);
        UploadF32Array(trap, _trapDevice, stream);
        UploadF32Array(gamma, _gammaDevice, stream);
        UploadF32Array(scale, _scaleDevice, stream);
        UploadF32Array(anglesRaw, _anglesRawDevice, stream);
        UploadF32Array(bRHN, _bDevice, stream);
        UploadF32Array(cRHN, _cDevice, stream);
        UploadF32Array(qkPreDotSum, _qkPreDotDevice, stream);
        UploadF32Array(coef, _coefDevice, stream);
        _stream.Synchronize();
    }
```

- [ ] **Step 3: Build**

Run: `dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj`
Expected: builds clean.

- [ ] **Step 4: Write the synthetic MIMO parity test**

Create `tests/DotLLM.Tests.Integration/Cuda/CudaMamba3MimoParitySyntheticTests.cs`, adapting Task 11's `CudaMamba3ParitySyntheticTests` fixture writer: set `is_mimo=true`, `mimo_rank=3` in the config JSON, extend `BBias`/`CBias` to `[NumHeads, MimoRank, StateSize]`, and add `MimoX`/`MimoZ`/`MimoO` tensors of shape `[NumHeads, MimoRank, HeadDim]` (canonical init values 1/R, 1, 1/R respectively, matching `Mamba3TensorMapping`'s doc comments for those three tensors):

```csharp
using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// CPU-vs-CUDA forward parity for <c>CudaMamba3TransformerModel</c>'s MIMO path
/// (issue #346, Task 14) on a deterministic synthetic checkpoint — synthetic-only,
/// matching CPU/Vulkan's own MIMO coverage (no public MIMO checkpoint exists
/// anywhere in the codebase; see docs/ROADMAP.md step 60f).
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaMamba3MimoParitySyntheticTests : IDisposable
{
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int MimoRank = 3;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize * MimoRank;
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    private const float LogitsAbsTol = 1e-2f;

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaMamba3MimoParitySyntheticTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mamba3-mimo-parity-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void CudaForward_MatchesCpuReference_OnSyntheticMimoCheckpoint()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMimoCheckpoint(modelPath, configPath);

        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        try
        {
            Assert.True(config.Mamba3Config!.IsMimo);

            int[] tokenIds = [0, 1, 2, 3, 5];
            int[] positions = [0, 1, 2, 3, 4];

            using ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1);
            using ITensor cudaLogits = cudaModel.Forward(tokenIds, positions, deviceId: -1);

            float[] cpuLast = ExtractRow(cpuLogits, tokenIds.Length - 1, VocabSize);
            float[] cudaLast = ExtractRow(cudaLogits, 0, VocabSize);

            float maxAbs = 0f;
            int worstIdx = 0;
            for (int i = 0; i < VocabSize; i++)
            {
                float diff = MathF.Abs(cpuLast[i] - cudaLast[i]);
                if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
            }
            _output.WriteLine($"MIMO last-token drift: max_abs={maxAbs:E3} at idx {worstIdx}");
            Assert.True(maxAbs <= LogitsAbsTol, $"MIMO logit divergence {maxAbs:G6} > {LogitsAbsTol:G4}.");
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    private static unsafe float[] ExtractRow(ITensor logits, int row, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, logits.Shape[0] * vocabSize);
        float[] result = new float[vocabSize];
        span.Slice(row * vocabSize, vocabSize).CopyTo(result);
        return result;
    }

    private static void WriteSyntheticMimoCheckpoint(string safetensorsPath, string configPath)
    {
        using (var fs = File.Create(configPath))
        using (var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteString("model_type", "mamba3");
            writer.WriteNumber("hidden_size", HiddenSize);
            writer.WriteNumber("vocab_size", VocabSize);
            writer.WriteNumber("num_hidden_layers", NumLayers);
            writer.WriteNumber("num_heads", NumHeads);
            writer.WriteNumber("head_dim", HeadDim);
            writer.WriteNumber("expand", Expand);
            writer.WriteNumber("n_groups", 1);
            writer.WriteNumber("state_size", StateSize);
            writer.WriteNumber("chunk_size", 2);
            writer.WriteNumber("mimo_rank", MimoRank);
            writer.WriteBoolean("is_mimo", true);
            writer.WriteBoolean("is_outproj_norm", false);
            writer.WriteBoolean("use_l2warp", false);
            writer.WriteBoolean("tie_word_embeddings", false);
            writer.WriteBoolean("rescale_prenorm_residual", true);
            writer.WriteBoolean("residual_in_fp32", true);
            writer.WriteNumber("A_floor", 1e-4);
            writer.WriteNumber("dt_init_floor", 1e-4);
            writer.WriteNumber("dt_min", 1e-3);
            writer.WriteNumber("dt_max", 0.1);
            writer.WriteNumber("norm_eps", 1e-5);
            writer.WriteNumber("rope_fraction", 0.5);
            writer.WriteNumber("max_position_embeddings", 32);
            writer.WriteEndObject();
        }

        var tensors = new List<(string Name, int[] Shape, float[] Values)>();
        AddSmall(tensors, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], 0.05f, 0);
        AddSmall(tensors, Mamba3TensorMapping.FinalNorm, [HiddenSize], 0.5f, 1);
        AddSmall(tensors, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], 0.05f, 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            AddSmall(tensors, Mamba3TensorMapping.LayerNorm(i), [HiddenSize], 0.5f, sBase + 0);
            AddSmall(tensors, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize], 0.02f, sBase + 1);
            AddSmall(tensors, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], 0.05f, sBase + 2);
            AddSmall(tensors, Mamba3TensorMapping.BNorm(i), [StateSize], 0.5f, sBase + 3);
            AddSmall(tensors, Mamba3TensorMapping.CNorm(i), [StateSize], 0.5f, sBase + 4);
            AddSmall(tensors, Mamba3TensorMapping.BBias(i), [NumHeads, MimoRank, StateSize], 0.02f, sBase + 5);
            AddSmall(tensors, Mamba3TensorMapping.CBias(i), [NumHeads, MimoRank, StateSize], 0.02f, sBase + 6);
            AddSmall(tensors, Mamba3TensorMapping.D(i), [NumHeads], 0.1f, sBase + 7);
            AddSmall(tensors, Mamba3TensorMapping.DtBias(i), [NumHeads], 0.02f, sBase + 8);
            // Canonical init values: mimo_x ~ 1/R, mimo_z ~ 1, mimo_o ~ 1/R (Mamba3TensorMapping doc comments).
            AddConstant(tensors, Mamba3TensorMapping.MimoX(i), [NumHeads, MimoRank, HeadDim], 1f / MimoRank);
            AddConstant(tensors, Mamba3TensorMapping.MimoZ(i), [NumHeads, MimoRank, HeadDim], 1f);
            AddConstant(tensors, Mamba3TensorMapping.MimoO(i), [NumHeads, MimoRank, HeadDim], 1f / MimoRank);
        }

        WriteSafetensorsFile(safetensorsPath, tensors);
    }

    private static void AddSmall(List<(string, int[], float[])> sink, string name, int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        sink.Add((name, shape, values));
    }

    private static void AddConstant(List<(string, int[], float[])> sink, string name, int[] shape, float value)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var values = new float[n];
        Array.Fill(values, value);
        sink.Add((name, shape, values));
    }

    private static void WriteSafetensorsFile(string path, List<(string Name, int[] Shape, float[] Values)> tensors)
    {
        using var headerMs = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerMs, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var (name, shape, values) in tensors)
            {
                long byteLen = values.Length * sizeof(float);
                w.WriteStartObject(name);
                w.WriteString("dtype", "F32");
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (int d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + byteLen);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += byteLen;
            }
            w.WriteEndObject();
        }
        byte[] headerBytes = headerMs.ToArray();

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerBytes.Length);
        fs.Write(prefix);
        fs.Write(headerBytes);

        foreach (var (_, _, values) in tensors)
        {
            byte[] bytes = new byte[values.Length * sizeof(float)];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
            fs.Write(bytes);
        }
    }
}
```

- [ ] **Step 5: Run the test**

Run: `dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj --filter FullyQualifiedName~CudaMamba3MimoParitySyntheticTests`
Expected: PASS on a CUDA-equipped host.

- [ ] **Step 6: Commit**

```bash
git add src/DotLLM.Cuda/Architectures/CudaMamba3TransformerModel.cs tests/DotLLM.Tests.Integration/Cuda/CudaMamba3MimoParitySyntheticTests.cs
git commit -m "feat(cuda): wire CudaMamba3TransformerModel MIMO forward + synthetic parity test (#346)"
```

---

### Task 15: Guard-test verification, new safetensors-guard regression test, and documentation updates

**Files:**
- Verify (no change expected): `tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs`
- Modify: `tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs` (add one new test)
- Modify: `docs/ROADMAP.md`
- Modify: `docs/SUPPORTED_MODELS.md`

- [ ] **Step 1: Verify the existing `CreateFromGguf` guard test still passes unmodified**

Run: `dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj --filter FullyQualifiedName~CudaUnsupportedArchitectureGuardTests`
Expected: PASS, including the `[InlineData(Architecture.Mamba3)]` case — Task 10 Step 3's reworded `CreateFromGguf` guard message still throws `NotSupportedException` and still contains the literal substring `"Mamba3"` (case-insensitive), which is all `CreateFromGguf_UnsupportedArchitecture_ThrowsNotSupportedInsteadOfSilentFallthrough`'s `Assert.Contains`/`Assert.DoesNotContain` checks require (`tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs:43-49`, read in full during planning). **No code change needed for this test to keep passing** — this step is a verification-only checkpoint, not an edit.

- [ ] **Step 2: Add a regression test for the `LoadFromSafetensors` guard's new message**

Append a new test method to `CudaUnsupportedArchitectureGuardTests` (same file, same class) asserting the *safetensors* guard (Task 10 Step 2) now redirects to `LoadMamba3FromSafetensors` by name, using the Task 11 synthetic-fixture writer pattern (a minimal one-tensor-set inline fixture is enough — this test only needs to reach the `config.Architecture == Architecture.Mamba3` check inside `LoadFromSafetensors`, it never runs a forward pass):

```csharp
    [SkippableFact]
    public void LoadFromSafetensors_Mamba3Checkpoint_ThrowsNotSupportedPointingAtDedicatedLoader()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string scratch = Path.Combine(Path.GetTempPath(), $"dotllm-mamba3-guard-{Guid.NewGuid():N}");
        Directory.CreateDirectory(scratch);
        try
        {
            string modelPath = Path.Combine(scratch, "model.safetensors");
            string configPath = Path.Combine(scratch, "config.json");
            // Reuses the exact synthetic-checkpoint writer introduced in
            // CudaMamba3ParitySyntheticTests (Task 11) — smallest fixture that
            // resolves to Architecture.Mamba3 via Mamba3ConfigExtractor.
            DotLLM.Tests.Integration.Cuda.CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest(
                modelPath, configPath);

            var ex = Assert.Throws<NotSupportedException>(
                () => CudaModelLoader.LoadFromSafetensors(modelPath));

            Assert.Contains("LoadMamba3FromSafetensors", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            try { Directory.Delete(scratch, recursive: true); } catch { /* best-effort */ }
        }
    }
```

This references a small internal helper that Task 11's fixture writer needs to expose for reuse — add it there rather than duplicating the whole synthetic-checkpoint writer a third time:

- [ ] **Step 3: Expose a reusable minimal-fixture helper from Task 11's test class**

In `tests/DotLLM.Tests.Integration/Cuda/CudaMamba3ParitySyntheticTests.cs` (Task 11), change the existing `private static void WriteSyntheticMamba3Checkpoint(...)` method's accessibility from `private` to `internal`, and rename it to `WriteMinimalMamba3CheckpointForGuardTest` for call-site clarity at the new guard-test call site (Step 2 above) — no behavior change, the method body is untouched:

```csharp
    internal static void WriteMinimalMamba3CheckpointForGuardTest(string safetensorsPath, string configPath)
        => WriteSyntheticMamba3Checkpoint(safetensorsPath, configPath);
```
(Add this as a thin `internal` wrapper alongside the existing `private static void WriteSyntheticMamba3Checkpoint` rather than renaming it in place — keeps Task 11's own test method calls (`WriteSyntheticMamba3Checkpoint(modelPath, configPath);` in `CudaForward_MatchesCpuReference_OnSyntheticCheckpoint`) unchanged.)

- [ ] **Step 4: Run both guard tests**

Run: `dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj --filter FullyQualifiedName~CudaUnsupportedArchitectureGuardTests`
Expected: PASS (3 `CreateFromGguf` cases + 1 new `LoadFromSafetensors` case).

- [ ] **Step 5: Update `docs/ROADMAP.md`**

Add a new row after the existing `60f` row (`docs/ROADMAP.md:154`), following the established `60<letter>` numbering convention for Mamba-3 follow-up steps:

```
| 60g | **Mamba-3 CUDA host (`CudaMamba3TransformerModel`)** :white_check_mark: | Dedicated CUDA host for the pure-SSM Mamba-3 architecture, closing the CPU/Vulkan-only gap (issue #346). Four new F32 CUDA kernels (`mamba3_data_rope_f32`, `mamba3_chunk_boundary_f32`, `mamba3_ssd_scan_siso_f32`, `mamba3_ssd_scan_mimo_f32`) port the already-validated Vulkan GLSL compute shaders to CUDA C — one CUDA block per head, sequential loop over t inside the kernel with `__syncthreads()` (not a per-token host-loop like the GDN scan kernel), `NO_FMA` for CPU bit-parity. Per-token preprocessing (softplus/sigmoid/RMSNorm+bias/qk_pre_dot/scale) stays host-side C#, mirroring Vulkan's design decision. `CudaMamba3StateCache` mirrors `CudaGdnStateCache`'s allocation idiom with Mamba-3's four-buffer state (`ssm_state`/`cum_angle`/`k_state`/`v_state`) instead of GDN's two. `CudaMamba3TransformerModel.LoadFromSafetensors` reuses the CPU `Mamba3WeightLoader` for tensor resolution, then uploads to device. **No GGUF path** — confirmed no backend (CPU/Vulkan/CUDA) has a Mamba-3 GGUF tensor-naming convention; `CudaModelLoader.CreateFromGguf`'s guard is reworded (not removed) to say so, and a new `CudaModelLoader.LoadMamba3FromSafetensors` is the real CUDA entry point (the existing `LoadFromSafetensors`'s Mamba3 guard now redirects there instead of throwing "not supported"). Verified: kernel-level unit tests against the CPU oracles (`Mamba3CanonicalSsd.ExecuteSiso`/`ExecuteMimo`, `Mamba3DataRoPE.ExecuteCanonical`) with bit-exact (`SequenceEqual`) assertions; synthetic end-to-end CPU-vs-CUDA parity (SISO + MIMO); gated real-weight prefill+decode parity on `ib-ssm/mamba3-370M-10BT` (`DOTLLM_IBSSM_CHECKPOINT_PATH`). MIMO is synthetic-fixture-only, matching CPU/Vulkan's own coverage (no public MIMO checkpoint exists anywhere). F32 only — quantized/F16 Mamba-3 weights (Vulkan's Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16 overlay support) are not ported to CUDA in this step. A dotLLM-invented GGUF convention for Mamba-3 is out of scope, tracked as a separate follow-up issue. | 60, 60e, 60f, Phase 4 CUDA backend |
```

- [ ] **Step 6: Update `docs/SUPPORTED_MODELS.md`**

In the Mamba-3 row of the top summary table (`docs/SUPPORTED_MODELS.md:39`), the "verified" cell currently reads `` `verified: real weights` (CPU + Vulkan) — ... `` — append `` + CUDA`` and extend the parenthetical with the CUDA parity test names, e.g. change:
```
`verified: real weights` (CPU + Vulkan) — `ib-ssm/mamba3-370M-10BT` (1.55 GB, CPU `IbSsmMamba3RealWeightsLoadTests`, Vulkan `Mamba3_VulkanForward_MatchesCpuReference_OnPromptPrefill`);
```
to:
```
`verified: real weights` (CPU + Vulkan + CUDA) — `ib-ssm/mamba3-370M-10BT` (1.55 GB, CPU `IbSsmMamba3RealWeightsLoadTests`, Vulkan `Mamba3_VulkanForward_MatchesCpuReference_OnPromptPrefill`, CUDA `IbSsmMamba3CudaParityTests`);
```

In the `### Mamba3 (`Architecture.Mamba3`)` prose section (`docs/SUPPORTED_MODELS.md:110-119`), append a paragraph documenting the CUDA host, its safetensors-only load path, and the F32-only / SISO-real-weight-MIMO-synthetic-only scope — mirroring the existing paragraph's style for the Vulkan port (`docs/SUPPORTED_MODELS.md:112-119`, e.g. `` **Vulkan**: full forward path landed (`VulkanMamba3TransformerModel` at `e40ada4` SISO + `7142f31` MIMO + `dfc9759` streaming-chunk + `effd8fc` Q8_0). `` — add a parallel `` **CUDA**: `` sentence naming `CudaMamba3TransformerModel` and this issue (#346).

- [ ] **Step 7: Check whether `README.md`'s Roadmap phase-count table needs a bump**

Per `CLAUDE.md`'s workflow rule 7 ("When a PR completes a roadmap step, update `docs/ROADMAP.md` ... and `README.md`"): open `README.md`'s Roadmap table (`README.md:789+`) and check whether step 60g's `Depends on` phase (`Phase 4 CUDA backend`, per Step 5's new row) has an incomplete step counter that this step should increment. If Phase 4's row already shows a completed/closed count that doesn't enumerate individual `60<letter>` sub-steps (the existing 60a-60f rows did not each bump a phase counter either, per inspection of the current table), leave `README.md` unchanged — do not invent a counter bump the existing sibling steps didn't make either. If in doubt, match whatever the closest analogous prior Mamba-3 sub-step PR did (check its diff via `git log --oneline --all -- README.md | grep -i mamba` for precedent) rather than guessing.

- [ ] **Step 8: Commit**

```bash
git add tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs tests/DotLLM.Tests.Integration/Cuda/CudaMamba3ParitySyntheticTests.cs docs/ROADMAP.md docs/SUPPORTED_MODELS.md
git commit -m "docs(cuda): update ROADMAP/SUPPORTED_MODELS + guard regression test for Mamba3 CUDA host (#346)"
```

---
