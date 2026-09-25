# gpt-oss CUDA MoE Support (issue #348) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gpt-oss GGUF checkpoints load and run correctly on the CUDA backend by adding the three pieces of MoE math the CUDA kernels currently lack — additive router/expert bias, the OAI-clamped SwiGLU activation, and softmax-after-top-k gating — then removing the `NotSupportedException` guard in `CudaModelLoader.CreateFromGguf` that currently blocks gpt-oss on CUDA.

**Architecture:** gpt-oss already runs correctly on CPU via `DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.Execute`. On CUDA, gpt-oss's MoE layers already flow through the **generic** `CudaTransformerModel` → `CudaMoeWeightsLoader.LoadLayerQuant` → `CudaMoeFfn.Forward` pipeline (the same one Mixtral/DeepSeek-V2 use) because `TransformerLayerWeights.Moe.HasRawQuantView` is true for gpt-oss's on-disk MXFP4 expert tensors. The gap is narrow: that pipeline never reads `Moe.RouterBias`/`GateExpsBias`/`UpExpsBias`/`DownExpsBias`/`UseSwiGluOai`, so it silently drops the bias and runs plain SiLU-gated SwiGLU instead of the clamped OAI variant — wrong output on every layer, not a crash (gpt-oss is all-MoE). Router-bias wiring and the softmax-after-top-k gating already have a working, math-equivalent implementation via existing kernels (`moe_gate_bias_add_f32` + `moe_softmax_topk_f32` + `moe_renorm_topk_f32` with `NormTopKProb=true` — proven below); only per-expert bias-add (reusing the same generic kernel with different shape params) and one genuinely new kernel (`swiglu_oai_f32`) are new code.

**Tech Stack:** C# (.NET 10, `DotLLM.Cuda`), CUDA C (`native/kernels/*.cu` → PTX via `native/build_ptx.bat`), xUnit (`DotLLM.Tests.Unit`).

## Global Constraints

- **Math equivalence (load-bearing — read before touching gating code):** gpt-oss's `softmaxAfterTopK` gating (`top-k(raw logits)` → `softmax(selected k logits)`) is mathematically identical to the existing kernel pipeline's `softmax(all logits)` → `top-k` → `renormalize-to-sum-1`, PROVIDED renormalization is applied. Proof: let `p_i = exp(l_i)/Σ_all exp(l_j)` (full softmax). Selecting top-k by `p_i` picks the same set as selecting by raw `l_i` (softmax is monotonic). Renormalizing the selected set: `p_i / Σ_topk p_j = [exp(l_i)/Z] / [Σ_topk exp(l_j)/Z] = exp(l_i) / Σ_topk exp(l_j)` — exactly `softmax` restricted to the top-k raw logits. **No new gating kernel is needed.** This equivalence REQUIRES the renormalization step to actually run, i.e. `CudaMoeLayerWeights.NormTopKProb` must be `true` whenever `moe.SoftmaxAfterTopK` is `true`. `DotLLM.Core.Models.MoeConfig.NormTopKProb` defaults to `true` and gpt-oss's `GgufModelConfigExtractor.ExtractGptOssMoeConfig` does not override it, so this holds today — Task 4 adds a defensive throw if it ever doesn't.
- Every new/modified CUDA kernel must be F32, `extern "C"`, and follow the existing file's fast-math policy: `native/kernels/moe_ffn.cu` is **not** in `build_ptx.bat`'s `FAST_MATH` list, so kernels added there compile with precise `expf`, matching `MoeQuantSwiGluMlp.SwiGluOai`'s `MathF.Exp` on the CPU side.
- `gpt-oss` constants: `alpha = 1.702f`, `limit = 7.0f` (`DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiAlpha` / `SwiGluOaiLimit`). Reuse these literal values verbatim in the CUDA kernel — do not re-derive them.
- New/changed kernel loads follow the existing optional-kernel pattern (`TryGetFunction` + `Has*` capability flag), NOT the required-kernel pattern — a stale PTX build must still load without gpt-oss support rather than crash unrelated models.
- Do not touch `ForwardBitNetI2S`, `CudaMoeWeightsLoader.LoadLayer`, or `CudaMoeWeightsLoader.LoadLayerBitNetI2S` — this ticket is scoped to the `MoePrecision.Quantized`/`F32` path only (BitNet-MoE issue #246 is a separate, already-shipped, unrelated code path).
- Per CLAUDE.md's cross-backend rule: this bug (missing bias/activation) is CUDA-only — CPU and Vulkan are unaffected (CPU already implements it via `MoeQuantSwiGluMlp`; Vulkan has no gpt-oss MoE path at all today and is out of scope for this ticket).

---

### Task 1: `swiglu_oai_f32` CUDA kernel

**Files:**
- Modify: `native/kernels/moe_ffn.cu`

**Interfaces:**
- Produces: `extern "C" __global__ void swiglu_oai_f32(const float* gate, const float* up, float* output, int n, int seq_len, float alpha, float limit)` — elementwise over `n * seq_len` elements, `output[idx] = glu(gate[idx]) * (clamp(up[idx]) + 1)`. Safe to call with `output` aliasing `gate` (matches `swiglu_f32`'s existing in-place-safe contract, reused by Task 5).

- [ ] **Step 1: Add the kernel**

Append to `native/kernels/moe_ffn.cu` (after the existing `moe_gate_bias_add_f32` kernel, before `moe_softmax_topk_f32`, keeping the file's existing kernel-inventory comment block at the top in sync by adding one line):

```c
// ── swiglu_oai_f32 ───────────────────────────────────────────────────────
//
// gpt-oss clamped SwiGLU activation (issue #348, llama.cpp ggml_swiglu_oai):
//   x = min(gate, limit)
//   y = clamp(up, -limit, limit)
//   out = x / (1 + exp(-alpha * x)) * (y + 1)
// Matches DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOai exactly (alpha=1.702f,
// limit=7.0f). This file is not in build_ptx.bat's FAST_MATH list, so expf()
// here is precise — matching the CPU oracle's MathF.Exp, unlike swiglu_f32.cu's
// fast-math sigmoid (see that file's header comment).
extern "C" __global__ void __launch_bounds__(256) swiglu_oai_f32(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, const int n, const int seq_len,
    const float alpha, const float limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * seq_len)
    {
        float g = gate[idx], u = up[idx];
        float x = fminf(g, limit);
        float y = fmaxf(fminf(u, limit), -limit);
        float glu = x / (1.0f + expf(-alpha * x));
        output[idx] = glu * (y + 1.0f);
    }
}
```

Also update the file's top-of-file kernel inventory comment (lines 7-26) by adding one entry after `moe_gate_bias_add_f32`:
```
//   swiglu_oai_f32                — gpt-oss clamped SwiGLU activation (issue #348):
//                                    x=min(gate,limit); y=clamp(up,-limit,limit);
//                                    out = x/(1+exp(-alpha*x)) * (y+1).
```

- [ ] **Step 2: Rebuild PTX**

Run: `native\build_ptx.bat` (from the repo root, or `cd native && build_ptx.bat` — requires `CUDA_PATH` set to a CUDA 13.x toolkit; see the script's own header comment for MSVC discovery requirements).
Expected: `native\ptx\moe_ffn.ptx` is regenerated with no compile errors; console output includes `moe_ffn.cu -> moe_ffn.ptx` (or equivalent per-file success line) with no `FAIL` marker.

- [ ] **Step 3: Commit**

```bash
git add native/kernels/moe_ffn.cu native/ptx/moe_ffn.ptx
git commit -m "feat(cuda): add swiglu_oai_f32 kernel for gpt-oss MoE (#348)"
```

---

### Task 2: `CudaKernels.cs` loader + launcher for `swiglu_oai_f32`

**Files:**
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`

**Interfaces:**
- Consumes: PTX symbol `swiglu_oai_f32` from `moe_ffn.ptx` (Task 1).
- Produces: `public bool HasSwiGluOai { get; }` and `public void LaunchSwiGLUOaiF32(nint gate, nint up, nint output, int n, int seqLen, nint stream)` — later tasks call this.

- [ ] **Step 1: Declare the field**

In the "MoE (Mixture-of-Experts) helper kernels (F32)" field block (near line 372, right after `_moeGateBiasAddF32Func`), add:

```csharp
    // Issue #348 (gpt-oss MoE): OAI-clamped SwiGLU activation.
    private readonly nint _swigluOaiF32Func;
```

- [ ] **Step 2: Load it (optional pattern)**

In the `moeFfnPath` loading block (near line 886-899), add one line after the existing `_moeGateBiasAddF32Func` load:

```csharp
            _moeGateBiasAddF32Func = _moeFfnModule.TryGetFunction("moe_gate_bias_add_f32");
            _swigluOaiF32Func = _moeFfnModule.TryGetFunction("swiglu_oai_f32");
```

- [ ] **Step 3: Add the capability flag**

Near `HasMoeGateBiasAdd` (line 1119), add:

```csharp
    /// <summary>
    /// True when the gpt-oss OAI-clamped-SwiGLU activation kernel (issue #348,
    /// <see cref="LaunchSwiGLUOaiF32"/>) is loaded. Optional — a stale PTX build
    /// without this symbol still loads; <see cref="CudaMoeFfn.Forward"/> throws
    /// a descriptive error only if a model actually needs it
    /// (<c>CudaMoeLayerWeights.UseSwiGluOai == true</c>).
    /// </summary>
    public bool HasSwiGluOai => _swigluOaiF32Func != 0;
```

- [ ] **Step 4: Add the launcher**

Near `LaunchMoeGateBiasAddF32` (line 5459-5471), add a sibling launcher using the exact same argument-marshalling pattern as `LaunchSwiGLUF32` (existing launcher — read it first to match its grid/block sizing) plus two extra scalar args:

```csharp
    /// <summary>
    /// gpt-oss clamped SwiGLU activation (issue #348): <c>out = x/(1+exp(-alpha*x)) * (y+1)</c>
    /// where <c>x=min(gate,limit)</c>, <c>y=clamp(up,-limit,limit)</c>. Safe to call with
    /// <paramref name="output"/> aliasing <paramref name="gate"/>.
    /// </summary>
    public void LaunchSwiGLUOaiF32(nint gate, nint up, nint output, int n, int seqLen,
        float alpha, float limit, nint stream)
    {
        if (_swigluOaiF32Func == 0)
            throw new InvalidOperationException(
                "swiglu_oai_f32 kernel not available. Recompile native/kernels/moe_ffn.cu to PTX.");

        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;
        float alphaArg = alpha, limitArg = limit;
        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg, &alphaArg, &limitArg};
        uint total = (uint)(n * seqLen);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_swigluOaiF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
```

(If `LaunchSwiGLUF32`'s actual grid/block sizing differs from the `moe_gate_bias_add_f32`-style flat grid-stride shown above — check it first via `Grep "LaunchSwiGLUF32" src/DotLLM.Cuda/CudaKernels.cs` — match ITS pattern exactly instead, since `swiglu_oai_f32` shares `swiglu_f32`'s exact `(n, seq_len)` grid shape, not `moe_gate_bias_add_f32`'s.)

- [ ] **Step 5: Build and verify no compile errors**

Run: `dotnet build src/DotLLM.Cuda -c Release`
Expected: builds clean (0 errors).

- [ ] **Step 6: Commit**

```bash
git add src/DotLLM.Cuda/CudaKernels.cs
git commit -m "feat(cuda): load swiglu_oai_f32 kernel, add LaunchSwiGLUOaiF32 (#348)"
```

---

### Task 3: `CudaMoeLayerWeights` — bias + activation fields

**Files:**
- Modify: `src/DotLLM.Cuda/CudaMoeWeights.cs`

**Interfaces:**
- Consumes: nothing new (pure data-holder change).
- Produces: `CudaMoeLayerWeights.RouterBiasF32`, `.GateExpsBiasF32`, `.UpExpsBiasF32`, `.DownExpsBiasF32` (all `nint`, 0 = absent), `.UseSwiGluOai` (`bool`) — Task 4 populates them, Task 5 reads them.

- [ ] **Step 1: Add the fields**

In `CudaMoeLayerWeights`, after the existing `GateBiasF32` property (line 120), add:

```csharp
    /// <summary>
    /// Optional F32 router-bias device pointer <c>[numExperts]</c> for the quantized-expert
    /// gating path (gpt-oss <c>ffn_gate_inp.bias</c>, issue #348). 0 when absent. Distinct
    /// from <see cref="GateBiasF32"/> (the identity-MoTE / <see cref="MoePrecision.BitNetI2S"/>
    /// router bias) so the two features can evolve independently even though both add to the
    /// same router logits before softmax/top-k.
    /// </summary>
    public nint RouterBiasF32 { get; }

    /// <summary>
    /// Optional per-expert gate-projection bias, flat F32 device pointer
    /// <c>[numExperts × moeIntermediateSize]</c> (gpt-oss <c>ffn_gate_exps.bias</c>). 0 when
    /// absent. Expert <c>e</c>'s slice starts at byte offset
    /// <c>e * moeIntermediateSize * sizeof(float)</c>.
    /// </summary>
    public nint GateExpsBiasF32 { get; }

    /// <summary>Optional per-expert up-projection bias. Same layout as <see cref="GateExpsBiasF32"/>.</summary>
    public nint UpExpsBiasF32 { get; }

    /// <summary>
    /// Optional per-expert down-projection bias, flat F32 device pointer
    /// <c>[numExperts × hiddenSize]</c>. 0 when absent. Expert <c>e</c>'s slice starts at byte
    /// offset <c>e * hiddenSize * sizeof(float)</c>.
    /// </summary>
    public nint DownExpsBiasF32 { get; }

    /// <summary>
    /// True = gpt-oss clamped swiglu_oai activation (<see cref="CudaKernels.LaunchSwiGLUOaiF32"/>);
    /// false = plain SwiGLU (<see cref="CudaKernels.LaunchSwiGLUF32"/>). Default false.
    /// </summary>
    public bool UseSwiGluOai { get; }
```

- [ ] **Step 2: Extend the full ctor**

Add four new optional parameters to the "Full ctor with explicit precision + per-projection quant types" (after the existing `rmsEps = 0f` parameter, so all existing call sites — `CudaMoeWeightsLoader.LoadLayerBitNetI2S`, test helpers — keep compiling unchanged):

```csharp
    public CudaMoeLayerWeights(
        int numExperts, int numExpertsPerTok, int hiddenSize, int moeIntermediateSize,
        bool normTopKProb,
        nint router,
        nint[] gateProj, nint[] upProj, nint[] downProj,
        int numSharedExperts, int sharedIntermediateSize,
        nint[]? sharedGateProj, nint[]? sharedUpProj, nint[]? sharedDownProj,
        nint sharedExpertGate,
        MoePrecision precision,
        QuantizationType gateProjQuantType,
        QuantizationType upProjQuantType,
        QuantizationType downProjQuantType,
        QuantizationType sharedGateProjQuantType,
        QuantizationType sharedUpProjQuantType,
        QuantizationType sharedDownProjQuantType,
        nint[]? expertFfnSubNormF32 = null,
        nint gateBiasF32 = 0,
        float rmsEps = 0f,
        nint routerBiasF32 = 0,
        nint gateExpsBiasF32 = 0,
        nint upExpsBiasF32 = 0,
        nint downExpsBiasF32 = 0,
        bool useSwiGluOai = false)
    {
        // ... existing body unchanged ...
        RouterBiasF32 = routerBiasF32;
        GateExpsBiasF32 = gateExpsBiasF32;
        UpExpsBiasF32 = upExpsBiasF32;
        DownExpsBiasF32 = downExpsBiasF32;
        UseSwiGluOai = useSwiGluOai;
    }
```

(Add the five new assignment lines at the end of the existing constructor body, after the existing `RmsEps = rmsEps;` line — do not reorder existing assignments.)

- [ ] **Step 3: Build**

Run: `dotnet build src/DotLLM.Cuda -c Release`
Expected: builds clean. All existing callers of the full ctor (positional-through-`rmsEps`, or named args) still compile since the new params are optional and appended last.

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cuda/CudaMoeWeights.cs
git commit -m "feat(cuda): add gpt-oss bias/activation fields to CudaMoeLayerWeights (#348)"
```

---

### Task 4: `CudaMoeWeightsLoader.LoadLayerQuant` — upload bias arrays

**Files:**
- Modify: `src/DotLLM.Cuda/CudaMoeWeightsLoader.cs`

**Interfaces:**
- Consumes: `DotLLM.Models.Architectures.MoeLayerWeights.RouterBias`/`GateExpsBias`/`UpExpsBias`/`DownExpsBias` (all `float[]?`), `.UseSwiGluOai` (`bool`), `.SoftmaxAfterTopK` (`bool`), `.NormTopKProb` (`bool`) — already populated by the shared GGUF loader for gpt-oss, read-only here.
- Produces: a `CudaMoeLayerWeights` with the Task 3 fields populated, passed to `CudaMoeFfn.Forward` (Task 5) by the existing `CudaWeights.cs:493` call site (unchanged — no signature change to `LoadLayerQuant` itself).

- [ ] **Step 1: Add the equivalence guard**

At the top of `LoadLayerQuant`, right after the existing `HasRawQuantView` check (after line 130), add:

```csharp
        if (moe.SoftmaxAfterTopK && !moe.NormTopKProb)
            throw new NotSupportedException(
                "CudaMoeWeightsLoader.LoadLayerQuant: SoftmaxAfterTopK gating requires " +
                "NormTopKProb=true on CUDA — the GPU forward path implements softmax-after-top-k " +
                "as softmax-over-all-experts + top-k + renormalize (mathematically equivalent only " +
                "when the top-k weights are renormalized to sum to 1). A model with " +
                "SoftmaxAfterTopK=true and NormTopKProb=false is not supported on CUDA.");
```

- [ ] **Step 2: Upload the bias arrays**

Right before the `return new CudaMoeLayerWeights(...)` statement (line 191), add:

```csharp
        nint routerBiasF32 = moe.RouterBias is float[] rb ? UploadF32Array(rb, allocs) : (nint)0;
        nint gateExpsBiasF32 = moe.GateExpsBias is float[] gb ? UploadF32Array(gb, allocs) : (nint)0;
        nint upExpsBiasF32 = moe.UpExpsBias is float[] ub ? UploadF32Array(ub, allocs) : (nint)0;
        nint downExpsBiasF32 = moe.DownExpsBias is float[] db ? UploadF32Array(db, allocs) : (nint)0;
```

- [ ] **Step 3: Pass them into the ctor call**

Change the existing `return new CudaMoeLayerWeights(...)` call (lines 191-205) to add the five new named args:

```csharp
        return new CudaMoeLayerWeights(
            numExperts, moe.NumExpertsPerTok, hidden, moeIntermediate,
            moe.NormTopKProb,
            router,
            gateProj, upProj, downProj,
            numSharedExperts, sharedIntermediate,
            sharedGate, sharedUp, sharedDown,
            sharedExpertGate,
            precision: MoePrecision.Quantized,
            gateProjQuantType: moe.GateExpsRawQt,
            upProjQuantType: moe.UpExpsRawQt,
            downProjQuantType: moe.DownExpsRawQt,
            sharedGateProjQuantType: sgQt,
            sharedUpProjQuantType: suQt,
            sharedDownProjQuantType: sdQt,
            routerBiasF32: routerBiasF32,
            gateExpsBiasF32: gateExpsBiasF32,
            upExpsBiasF32: upExpsBiasF32,
            downExpsBiasF32: downExpsBiasF32,
            useSwiGluOai: moe.UseSwiGluOai);
```

- [ ] **Step 4: Build**

Run: `dotnet build src/DotLLM.Cuda -c Release`
Expected: builds clean.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cuda/CudaMoeWeightsLoader.cs
git commit -m "feat(cuda): upload gpt-oss router/expert bias in LoadLayerQuant (#348)"
```

---

### Task 5: `CudaMoeFfn.Forward` — wire bias-add + OAI activation

**Files:**
- Modify: `src/DotLLM.Cuda/CudaMoeFfn.cs`

**Interfaces:**
- Consumes: `CudaMoeLayerWeights.RouterBiasF32`/`GateExpsBiasF32`/`UpExpsBiasF32`/`DownExpsBiasF32`/`UseSwiGluOai` (Task 3), `CudaKernels.LaunchMoeGateBiasAddF32` (existing, generic row-broadcast-add — reused as-is, NOT a new kernel), `CudaKernels.LaunchSwiGLUOaiF32`/`HasSwiGluOai` (Task 2).
- Produces: correct MoE forward output for gpt-oss-shaped `CudaMoeLayerWeights` (no change to existing Mixtral/DeepSeek/Qwen behavior — all new fields default to 0/false).

- [ ] **Step 1: Router bias — add before softmax/top-k**

In `Forward` (not `ForwardBitNetI2S`), between Step 2 (router GEMV, ends at line 124) and Step 3 (`LaunchMoeSoftmaxTopk`, line 127), insert:

```csharp
        // ── Step 2b: additive router bias (gpt-oss, issue #348) ──
        // MUST run before softmax/top-k — same ordering as ForwardBitNetI2S's identity-MoTE
        // router bias (bias shifts both the top-k argmax AND the softmax probabilities).
        if (weights.RouterBiasF32 != 0)
        {
            if (!kernels.HasMoeGateBiasAdd)
                throw new InvalidOperationException(
                    "MoE layer has a router bias but moe_gate_bias_add_f32 is not available. " +
                    "Recompile native/kernels/moe_ffn.cu to PTX.");
            kernels.LaunchMoeGateBiasAddF32(scratch.Logits, weights.RouterBiasF32, seqLen, E, stream);
        }
```

- [ ] **Step 2: Guard the grouped fast-path off when a per-expert bias is present**

The grouped-GEMV fast path (`useGrouped`, computed just before the `if (useGrouped && kActive > 0)` block) skips the per-expert gate/up `ProjectF32OrQuant` calls entirely, so it would silently skip the new bias-add too if ever combined with a K-quant grouped-GEMV-eligible quant type. Today this can't happen (gpt-oss uses MXFP4, which has no `HasMoeGroupedGemv` kernel — see `CudaKernels.cs`'s grouped-GEMV quant list), but make the exclusion explicit rather than relying on that coincidence. Change the `useGrouped` computation (around line 221-227):

```csharp
        bool useGrouped = weights.Precision == MoePrecision.Quantized
            && seqLen == 1
            && (hidden % 256) == 0
            && kernels.HasMoeGroupedGemv(weights.GateProjQuantType)
            && kernels.HasMoeGroupedGemv(weights.UpProjQuantType)
            && weights.GateProjQuantType == weights.UpProjQuantType
            && weights.GateExpsBiasF32 == 0 && weights.UpExpsBiasF32 == 0
            && activeExperts > 0;
```

- [ ] **Step 3: Gate/up per-expert bias-add — reuse the generic row-broadcast-add kernel**

Immediately after the two `ProjectF32OrQuant` calls for gate and up inside the `if (!useGrouped)` block (lines 291-304), add bias-adds. `LaunchMoeGateBiasAddF32(logits, bias, seqLen, numExperts, stream)` computes `logits[t*numExperts + e] += bias[e % numExperts]` for a `[seqLen, numExperts]` buffer — passing `rows=batch` and `width=I` makes it add a `[I]`-wide bias to every row of a `[batch, I]` buffer, which is exactly what a per-expert bias broadcast across this expert's routed-token batch needs (no new kernel required):

```csharp
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.GatheredInput, batch, K: hidden, M: I,
                    weightF32: weights.GateProj[e],
                    weightQuant: weights.GateProj[e], weightQt: weights.GateProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                    gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.GateBatch);
                if (weights.GateExpsBiasF32 != 0)
                    kernels.LaunchMoeGateBiasAddF32(
                        scratch.GateBatch, weights.GateExpsBiasF32 + (nint)((long)e * I * sizeof(float)),
                        seqLen: batch, numExperts: I, stream);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.GatheredInput, batch, K: hidden, M: I,
                    weightF32: weights.UpProj[e],
                    weightQuant: weights.UpProj[e], weightQt: weights.UpProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                    gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.UpBatch);
                if (weights.UpExpsBiasF32 != 0)
                    kernels.LaunchMoeGateBiasAddF32(
                        scratch.UpBatch, weights.UpExpsBiasF32 + (nint)((long)e * I * sizeof(float)),
                        seqLen: batch, numExperts: I, stream);
```

- [ ] **Step 4: Down per-expert bias-add**

Immediately after the down-projection `ProjectF32OrQuant` call (lines 334-340), add:

```csharp
            ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                scratch.SiluBatch, batch, K: I, M: hidden,
                weightF32: weights.DownProj[e],
                weightQuant: weights.DownProj[e], weightQt: weights.DownProjQuantType,
                dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                outputF32: scratch.DownBatch);
            if (weights.DownExpsBiasF32 != 0)
                kernels.LaunchMoeGateBiasAddF32(
                    scratch.DownBatch, weights.DownExpsBiasF32 + (nint)((long)e * hidden * sizeof(float)),
                    seqLen: batch, numExperts: hidden, stream);
```

- [ ] **Step 5: Swap the activation kernel**

Replace the single `LaunchSwiGLUF32` call in the per-expert loop (lines 324-328) with a branch:

```csharp
            if (weights.UseSwiGluOai)
            {
                if (!kernels.HasSwiGluOai)
                    throw new InvalidOperationException(
                        "MoE layer requires the gpt-oss clamped SwiGLU activation but " +
                        "swiglu_oai_f32 is not available. Recompile native/kernels/moe_ffn.cu to PTX.");
                kernels.LaunchSwiGLUOaiF32(
                    scratch.GateBatch + (nint)gateOff,
                    scratch.UpBatch + (nint)upOff,
                    scratch.SiluBatch,
                    I, batch,
                    DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiAlpha,
                    DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiLimit,
                    stream);
            }
            else
            {
                kernels.LaunchSwiGLUF32(
                    scratch.GateBatch + (nint)gateOff,
                    scratch.UpBatch + (nint)upOff,
                    scratch.SiluBatch,
                    I, batch, stream);
            }
```

(`DotLLM.Cuda` does not currently reference `DotLLM.Cpu` — check `src/DotLLM.Cuda/DotLLM.Cuda.csproj` for an existing `<ProjectReference>` to `DotLLM.Cpu`. If none exists, hardcode the two literals directly instead of referencing the CPU constants: `alpha: 1.702f, limit: 7.0f`, with a comment `// DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiAlpha/SwiGluOaiLimit — literal copy, no cross-project ref` — verify which applies before writing this step's code, since `CudaMoeWeightsLoader.cs` already references `DotLLM.Cpu.Kernels.Dequantize` directly (line 146-148), which suggests the reference DOES exist and the direct-constant-reference form above is correct.)

- [ ] **Step 6: Leave the shared-expert loop's activation unconditional**

Do NOT touch the shared-expert path's `LaunchSwiGLUF32` call (lines 380-382) — gpt-oss has no shared experts (`MoeQuantSwiGluMlp.Execute` has no shared-expert parameters at all), so `weights.NumSharedExperts == 0` for every gpt-oss layer and this code never executes for it. Wiring `UseSwiGluOai` into the shared-expert branch would be speculative generality for a combination no current architecture produces.

- [ ] **Step 7: Build**

Run: `dotnet build src/DotLLM.Cuda -c Release`
Expected: builds clean.

- [ ] **Step 8: Commit**

```bash
git add src/DotLLM.Cuda/CudaMoeFfn.cs
git commit -m "feat(cuda): wire gpt-oss router/expert bias + OAI SwiGLU into CudaMoeFfn.Forward (#348)"
```

---

### Task 6: Remove the `CudaModelLoader` guard

**Files:**
- Modify: `src/DotLLM.Cuda/CudaModelLoader.cs`
- Modify: `tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs`

**Interfaces:**
- Consumes: nothing new.
- Produces: `CudaModelLoader.CreateFromGguf` no longer throws for `Architecture.GptOss`; falls through to the `default:` case (`CudaTransformerModel.LoadFromGguf`), which now produces correct output per Tasks 1-5.

- [ ] **Step 1: Remove the throw**

In `CreateFromGguf`'s switch (lines 97-110), delete the entire `case Architecture.GptOss:` block:

```csharp
            // gpt-oss's MoE experts carry a per-expert bias (GateExpsBias/UpExpsBias/DownExpsBias)
            // and use an OAI-clamped-SwiGLU activation (UseSwiGluOai), neither of which
            // CudaMoeWeightsLoader/CudaMoeFfn reference (confirmed: zero call sites outside
            // generated XML docs). Falling through to the generic MoE path would silently drop
            // the bias and run the wrong activation on every layer (gpt-oss is all-MoE) rather
            // than crash or warn — worse than a clean failure. Fail loudly until CudaMoeFfn
            // actually implements both.
            case Architecture.GptOss:
                throw new NotSupportedException(
                    "CUDA does not yet implement gpt-oss's per-expert MoE bias or OAI-clamped-"
                    + "SwiGLU activation (CudaMoeFfn has no support for UseQuantExperts/"
                    + "*ExpsBias/UseSwiGluOai) — falling through to the generic MoE path would "
                    + "silently produce wrong output rather than fail. Use the CPU or Vulkan "
                    + "backend for gpt-oss checkpoints.");

```

(Leave the `Mamba3` and `NemotronH` cases untouched — those are separate tickets, #346 and #347.)

- [ ] **Step 2: Remove the now-invalid test case**

In `CudaUnsupportedArchitectureGuardTests.cs`, the test asserts `CreateFromGguf` throws for `Architecture.GptOss` — that's no longer true after Step 1. Remove the `[InlineData(Architecture.GptOss)]` attribute (line 28) and the special-cased message-check logic in the test body that only existed for it:

```csharp
    [SkippableTheory]
    [InlineData(Architecture.Mamba3)]
    [InlineData(Architecture.NemotronH)]
    public void CreateFromGguf_UnsupportedArchitecture_ThrowsNotSupportedInsteadOfSilentFallthrough(
        Architecture architecture)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF (content unused by this test)"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var realConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var config = realConfig with { Architecture = architecture };

        var ex = Assert.Throws<NotSupportedException>(
            () => CudaModelLoader.CreateFromGguf(gguf, config));

        Assert.DoesNotContain("attn_output.weight", ex.Message, StringComparison.Ordinal);
        Assert.Contains(architecture.ToString(), ex.Message, StringComparison.OrdinalIgnoreCase);
    }
```

(This simplifies away the `expectedNameInMessage` ternary since both remaining cases use the enum name verbatim in their message.)

- [ ] **Step 3: Build**

Run: `dotnet build src/DotLLM.Cuda tests/DotLLM.Tests.Integration -c Release`
Expected: builds clean.

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cuda/CudaModelLoader.cs tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs
git commit -m "feat(cuda): allow gpt-oss through CudaModelLoader now that MoE bias/activation is wired (#348)"
```

---

### Task 7: Discriminating CUDA parity test

**Files:**
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaMoeFfnTests.cs`

**Interfaces:**
- Consumes: `MoeQuantSwiGluMlp.Execute` (CPU oracle, `DotLLM.Cpu.Kernels`), `CudaMoeLayerWeights`'s full ctor (Task 3), `CudaMoeFfn.Forward` (Task 5).
- Produces: a test that FAILS on `dev` (pre-Task-1-6 code) and PASSES after this plan's changes — proves the fix, not just "doesn't crash."

This test follows `GptOssKernelTests.MoeQuant_SoftmaxAfterTopK_F32Experts_MatchesManualReference`'s precedent of using `QuantizationType.F32` experts to isolate the routing/bias/activation semantics under test from quantization-format concerns (MXFP4 dequant correctness is already covered by the existing `Mxfp4Tests.cs` CPU suite and CUDA's generic `LaunchDequantToF16` MXFP4 path — out of scope here). It also follows `CudaMoeFfnTests.Run()`'s existing structure (build CPU oracle, build device buffers by hand via the ctor — NOT `CudaMoeWeightsLoader`, since that requires a real GGUF-shaped `TransformerLayerWeights`).

- [ ] **Step 1: Write the failing test**

Add to `CudaMoeFfnTests.cs`, after `MoeFfn_DeepSeekV2LiteShape_MatchesCpuOracle` (after line 183, before the `Run` helper at line 185):

```csharp
    /// <summary>
    /// gpt-oss-shaped fixture (issue #348): router + per-expert bias, softmax-after-top-k
    /// gating, OAI-clamped SwiGLU activation. Oracle is
    /// <see cref="MoeQuantSwiGluMlp.Execute"/> (the same kernel gpt-oss uses on CPU) with F32
    /// experts — isolating bias/gating/activation correctness from MXFP4 dequant correctness
    /// (covered separately by Mxfp4Tests.cs). No shared experts (gpt-oss has none).
    /// </summary>
    [SkippableFact]
    public unsafe void MoeFfn_GptOssShape_MatchesQuantSwiGluOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");
        Skip.IfNot(_kernels!.HasSwiGluOai, "swiglu_oai_f32 kernel not available (stale PTX)");

        const int seqLen = 3, numExperts = 8, topK = 2, hidden = 32, intermediate = 48;
        var rng = new Random(348);

        float[] hiddenIn = RandomArr(rng, seqLen * hidden, 0.3f);
        float[] router = RandomArr(rng, numExperts * hidden, 0.05f);
        float[] routerBias = RandomArr(rng, numExperts, 0.05f);
        float[] gateBias = RandomArr(rng, numExperts * intermediate, 0.05f);
        float[] upBias = RandomArr(rng, numExperts * intermediate, 0.05f);
        float[] downBias = RandomArr(rng, numExperts * hidden, 0.05f);

        float[][] w1 = new float[numExperts][];
        float[][] w2 = new float[numExperts][];
        float[][] w3 = new float[numExperts][];
        for (int e = 0; e < numExperts; e++)
        {
            w1[e] = RandomArr(rng, intermediate * hidden, 0.05f);
            w3[e] = RandomArr(rng, intermediate * hidden, 0.05f);
            w2[e] = RandomArr(rng, hidden * intermediate, 0.05f);
        }

        // ── CPU oracle: MoeQuantSwiGluMlp.Execute, F32 experts, one token at a time
        // (Execute's seqLen loop already handles multi-token, but flattening per-expert
        // pointer arrays for the fixed-block-based CPU API is simplest done once). ──
        float[] cpuOut = new float[seqLen * hidden];
        var pins = new List<System.Runtime.InteropServices.GCHandle>();
        try
        {
            var w1Handles = new System.Runtime.InteropServices.GCHandle[numExperts];
            var w2Handles = new System.Runtime.InteropServices.GCHandle[numExperts];
            var w3Handles = new System.Runtime.InteropServices.GCHandle[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                w1Handles[e] = System.Runtime.InteropServices.GCHandle.Alloc(w1[e], System.Runtime.InteropServices.GCHandleType.Pinned);
                w2Handles[e] = System.Runtime.InteropServices.GCHandle.Alloc(w2[e], System.Runtime.InteropServices.GCHandleType.Pinned);
                w3Handles[e] = System.Runtime.InteropServices.GCHandle.Alloc(w3[e], System.Runtime.InteropServices.GCHandleType.Pinned);
                pins.Add(w1Handles[e]); pins.Add(w2Handles[e]); pins.Add(w3Handles[e]);
            }

            // MoeQuantSwiGluMlp.Execute expects one contiguous 3D-stacked bank per projection
            // (expert e at byte offset e * expertBytes), matching the GGUF on-disk layout. Build
            // that layout on the host by concatenating the per-expert F32 arrays.
            float[] gateExpsFlat = new float[numExperts * intermediate * hidden];
            float[] upExpsFlat = new float[numExperts * intermediate * hidden];
            float[] downExpsFlat = new float[numExperts * hidden * intermediate];
            for (int e = 0; e < numExperts; e++)
            {
                w1[e].CopyTo(gateExpsFlat, e * intermediate * hidden);
                w3[e].CopyTo(upExpsFlat, e * intermediate * hidden);
                w2[e].CopyTo(downExpsFlat, e * hidden * intermediate);
            }

            fixed (float* hp = hiddenIn)
            fixed (float* op = cpuOut)
            fixed (float* gp = gateExpsFlat)
            fixed (float* up = upExpsFlat)
            fixed (float* dp = downExpsFlat)
            {
                MoeQuantSwiGluMlp.Execute(
                    hidden: hp, output: op, seqLen: seqLen,
                    routerWeight: router, routerBias: routerBias,
                    gateExpsBase: (nint)gp, gateQt: DotLLM.Core.Configuration.QuantizationType.F32,
                    upExpsBase: (nint)up, upQt: DotLLM.Core.Configuration.QuantizationType.F32,
                    downExpsBase: (nint)dp, downQt: DotLLM.Core.Configuration.QuantizationType.F32,
                    gateBias: gateBias, upBias: upBias, downBias: downBias,
                    numExperts: numExperts, numExpertsPerTok: topK,
                    hiddenSize: hidden, intermediateSize: intermediate,
                    softmaxAfterTopK: true, useSwiGluOai: true,
                    pool: null);
            }
        }
        finally
        {
            foreach (var h in pins) h.Free();
        }

        // ── GPU forward ──
        var allocs = new List<nint>();
        try
        {
            nint dHidden = AllocAndUploadF32(hiddenIn, allocs);
            nint dRouter = AllocAndUploadF32(router, allocs);
            nint dRouterBias = AllocAndUploadF32(routerBias, allocs);
            nint dGateBias = AllocAndUploadF32(gateBias, allocs);
            nint dUpBias = AllocAndUploadF32(upBias, allocs);
            nint dDownBias = AllocAndUploadF32(downBias, allocs);

            nint[] dW1 = new nint[numExperts];
            nint[] dW2 = new nint[numExperts];
            nint[] dW3 = new nint[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                dW1[e] = AllocAndUploadF32(w1[e], allocs);
                dW2[e] = AllocAndUploadF32(w2[e], allocs);
                dW3[e] = AllocAndUploadF32(w3[e], allocs);
            }
            nint dOut = AllocF32(seqLen * hidden, allocs);

            var weights = new CudaMoeLayerWeights(
                numExperts: numExperts, numExpertsPerTok: topK, hiddenSize: hidden,
                moeIntermediateSize: intermediate,
                normTopKProb: true,
                router: dRouter,
                gateProj: dW1, upProj: dW3, downProj: dW2,
                numSharedExperts: 0, sharedIntermediateSize: 0,
                sharedGateProj: null, sharedUpProj: null, sharedDownProj: null,
                sharedExpertGate: 0,
                precision: MoePrecision.F32,
                gateProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                upProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                downProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                sharedGateProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                sharedUpProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                sharedDownProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                routerBiasF32: dRouterBias,
                gateExpsBiasF32: dGateBias,
                upExpsBiasF32: dUpBias,
                downExpsBiasF32: dDownBias,
                useSwiGluOai: true);

            using var scratch = new CudaMoeScratch();

            CudaMoeFfn.Forward(
                hiddenF32: dHidden, outputF32: dOut,
                seqLen: seqLen,
                weights: weights,
                scratch: scratch, cublasHandle: _cublas!.Handle,
                kernels: _kernels!, stream: _stream!.Handle);
            _stream.Synchronize();

            float[] gpuOut = new float[seqLen * hidden];
            fixed (float* p = gpuOut)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut,
                    (nuint)(gpuOut.Length * sizeof(float))).ThrowOnError();

            int mismatches = 0;
            float maxDiff = 0f;
            int maxDiffIdx = -1;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > DefaultTolerance)
                {
                    mismatches++;
                    if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
                }
            }
            Assert.True(mismatches == 0,
                $"gpt-oss MoE forward: {mismatches}/{cpuOut.Length} elements outside tolerance "
              + $"{DefaultTolerance} (max diff {maxDiff} at idx {maxDiffIdx}: "
              + $"cpu={(maxDiffIdx >= 0 ? cpuOut[maxDiffIdx] : 0)} "
              + $"gpu={(maxDiffIdx >= 0 ? gpuOut[maxDiffIdx] : 0)}).");
        }
        finally
        {
            foreach (var p in allocs)
                CudaDriverApi.cuMemFree_v2(p);
        }
    }
```

- [ ] **Step 2: Run it to verify it FAILS before Tasks 1-6**

This step is retroactive verification, not a literal pre-implementation run (this plan is written after Tasks 1-6's design is already fixed) — but the *implementer executing this plan* should run this test immediately after writing it and BEFORE Task 1-6's code exists (i.e., stash or check out `dev` at the pre-plan commit, or simply run it before touching any other file) to confirm it genuinely discriminates:

Run: `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~MoeFfn_GptOssShape_MatchesQuantSwiGluOracle"`
Expected (against unmodified `CudaMoeFfn.Forward`/`CudaMoeLayerWeights`): **compile error** — the new named ctor args (`routerBiasF32`, etc.) don't exist yet. This is an acceptable discriminating signal (proves the test exercises the new surface) since Task 3 must land before this test can even compile; do not skip ahead by writing a version that compiles against old code, since that version would not test the real bug (missing bias/activation application), only its absence would silently pass with near-random results going undetected by count.

If executing tasks in order (Task 1→7, the plan's natural sequence), skip literal pre-implementation failure verification for this task — Tasks 1-6 already landed by the time this test is written, so instead verify the test FAILS by temporarily commenting out one of Task 5's bias-add calls (e.g. Step 3's gate bias-add) and confirming the test then fails with a nonzero mismatch count, then restore it. This is the practical substitute for "run it to see it fail" when the fix is already in place.

- [ ] **Step 3: Run it to verify it PASSES**

Run: `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~MoeFfn_GptOssShape_MatchesQuantSwiGluOracle"`
Expected: PASS (skips cleanly with `Skip.IfNot` if no CUDA GPU is present in the CI/build environment).

- [ ] **Step 4: Run the full existing MoE test suite for regressions**

Run: `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~CudaMoeFfnTests"`
Expected: all tests in the file PASS (existing Mixtral/DeepSeek/Qwen fixtures unaffected — all new `CudaMoeLayerWeights` fields default to 0/false for them).

- [ ] **Step 5: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Cuda/CudaMoeFfnTests.cs
git commit -m "test(cuda): add discriminating gpt-oss MoE bias/activation parity test (#348)"
```

---

## Self-Review Notes (for the implementer, not a task)

- **Spec coverage:** router bias (Task 5 Step 1), per-expert gate/up/down bias (Task 5 Steps 3-4), OAI-clamped activation (Task 1, 2, Task 5 Step 5), softmax-after-top-k gating (proven equivalent to existing kernels in Global Constraints — no code task needed beyond the `NormTopKProb` guard in Task 4 Step 1), loader guard removal (Task 6), regression test (Task 7). All five gaps named in the original issue (#348: "per-expert bias / OAI-clamped-SwiGLU activation / UseQuantExperts / *ExpsBias / UseSwiGluOai") are covered.
- **Cross-backend rule (CLAUDE.md):** this is a CUDA-only gap — CPU already has the correct implementation (`MoeQuantSwiGluMlp`, unmodified by this plan) and Vulkan has no gpt-oss MoE path to fix (out of scope, not a regression introduced here).
- **Out of scope, deliberately:** a real end-to-end gpt-oss GGUF fixture test (`RealGgufCudaParityTests`-style) is not included — no gpt-oss GGUF fixture convention (`DOTLLM_*_GGUF` env var) exists in this repo yet. Once one is cached locally, add a fixture-gated parity test mirroring `RealGgufCudaParityTests.cs`'s pattern as a fast follow-up; this plan's Task 7 synthetic test is what actually proves the fix.
