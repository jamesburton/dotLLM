# gpt-oss CUDA Alternating Sliding-Window Attention (issue #366) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CUDA transformer path apply gpt-oss's alternating sliding-window/dense attention pattern per layer (window on even layers, dense on odd), mirroring the CPU reference exactly, and resolve the RoPE-scaling investigation the issue bundles with it.

**Architecture:** Host-side-only change for the SWA half: the CUDA attention kernels (`attention_f32`, `attention_f32_split_kv`, flash variants) ALREADY accept a `sliding_window` int parameter with correct masking (`pos_q - tkv >= sliding_window` at `native/kernels/attention_f32.cu:88`) — passing `0` disables the window. Today `CudaTransformerModel` computes one `int slidingWindow = Config.SlidingWindowSize ?? 0` per forward (lines 949 and 1962) and passes it to every layer. The fix ports the CPU's `GetLayerSlidingWindow` (`src/DotLLM.Models/Architectures/TransformerModel.cs:706-715`) and evaluates it per layer inside the layer loop. NO kernel, PTX, or launcher changes.

**Sequencing:** This plan executes BEFORE the sibling sinks plan (`2026-08-15-gptoss-cuda-attention-sinks.md`, issue #365). Rationale: this plan is host-only and independently testable on existing architectures (Gemma-3-style patterns test the same code path); the sinks plan changes kernel signatures and, as the second lander, owns the final `Architecture.GptOss` guard removal in `CudaModelLoader.CreateFromGguf` (per both issues' acceptance criteria: the guard stays until BOTH are done). The two plans touch disjoint lines except the shared attention call sites — landing this one first keeps the sinks diff clean.

## Global Constraints

- Branch `issue/366-cuda-swa-alternating` from `dev`. Commits reference `(#366)`.
- NO changes to `native/kernels/*.cu`, `native/ptx/*.ptx`, or `CudaKernels.cs` launchers — this plan is pure C# host logic + tests. (If you believe a kernel change is needed, STOP and report; that means the research premise broke.)
- The CPU reference is authoritative: `TransformerModel.GetLayerSlidingWindow` semantics — `PerLayerSlidingWindow` list wins if set; else `SlidingWindowPattern` N>0 applies the window only to layers where `layer % N < N - 1` (llama.cpp `set_swa_pattern(N, dense_first=false)`; gpt-oss N=2 → even layers windowed, odd dense); else uniform `SlidingWindowSize`.
- MakeCurrent/stream conventions untouched (#368); no new allocations (no ledger concerns, #383).
- Tests: documented-tolerance parity (never `SequenceEqual`); discriminating shapes — the SWA test MUST use a sequence longer than the window so a window-applied-to-dense-layer bug produces O(1) logit divergence (per CLAUDE.md's discriminating-test rule).
- Do NOT remove the `Architecture.GptOss` guard in `CudaModelLoader.CreateFromGguf` — that happens at the end of the #365 plan.

---

### Task 1: Per-layer window resolver in `CudaTransformerModel`

**Files:**
- Modify: `src/DotLLM.Cuda/CudaTransformerModel.cs`
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaPerLayerSlidingWindowTests.cs` (create)

**Interfaces:**
- Produces: `private int GetLayerSlidingWindowCuda(int layer)` returning the kernel-ready int (0 = dense), used by Tasks 2-3. Kernel convention is `0 = no window` (see `attention_f32.cu:88`), whereas CPU uses `int?` — the resolver converts.

- [ ] **Step 1: Write the failing test.** The resolver is private; test it through a tiny internal hook or `InternalsVisibleTo` (already granted to the test assemblies — check `DotLLM.Cuda.csproj`; if only `DotLLM.Tests.Unit` lacks it, add the resolver as `internal` instead of `private`). Test cases (pure logic, no GPU):

```csharp
// tests/DotLLM.Tests.Unit/Cuda/CudaPerLayerSlidingWindowTests.cs
using DotLLM.Core.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Verifies CudaTransformerModel's per-layer sliding-window resolution matches the CPU
/// reference semantics (TransformerModel.GetLayerSlidingWindow, TransformerModel.cs:706-715):
/// PerLayerSlidingWindow list wins; else pattern N>0 windows layers where layer%N &lt; N-1
/// (gpt-oss N=2: even windowed, odd dense); else uniform. Kernel convention: 0 = dense.
/// </summary>
public class CudaPerLayerSlidingWindowTests
{
    [Theory]
    // gpt-oss shape: window=128, pattern=2 → layer0 windowed, layer1 dense, layer2 windowed...
    [InlineData(128, 2, 0, 128)]
    [InlineData(128, 2, 1, 0)]
    [InlineData(128, 2, 2, 128)]
    [InlineData(128, 2, 3, 0)]
    // pattern<=0 → uniform
    [InlineData(4096, 0, 0, 4096)]
    [InlineData(4096, 0, 5, 4096)]
    // no window at all
    [InlineData(0, 2, 0, 0)]
    public void ResolveLayerWindow_PatternSemanticsMatchCpu(
        int windowSize, int pattern, int layer, int expected)
    {
        int actual = CudaSlidingWindowResolver.Resolve(
            windowSize == 0 ? null : windowSize, pattern, perLayer: null, layer);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ResolveLayerWindow_PerLayerListWins()
    {
        var perLayer = new int?[] { null, 256, null, 64 };
        Assert.Equal(0,   CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 0));
        Assert.Equal(256, CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 1));
        Assert.Equal(0,   CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 2));
        Assert.Equal(64,  CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 3));
    }
}
```

- [ ] **Step 2: Run to verify it fails** (`CudaSlidingWindowResolver` doesn't exist): `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~CudaPerLayerSlidingWindowTests"` → build error CS0103/CS0246.

- [ ] **Step 3: Implement.** Add a small static resolver class (static + testable, avoids exposing model internals) in `src/DotLLM.Cuda/CudaSlidingWindowResolver.cs`:

```csharp
namespace DotLLM.Cuda;

/// <summary>
/// Per-layer sliding-window resolution for CUDA attention dispatch. Mirrors the CPU
/// reference <c>TransformerModel.GetLayerSlidingWindow</c> (TransformerModel.cs:706-715)
/// exactly, but returns the CUDA kernel convention: 0 = dense/full attention, positive =
/// window length (attention_f32.cu masks <c>pos_q - tkv &gt;= sliding_window</c> only when
/// the value is &gt; 0). Pattern N&gt;0 windows layers where <c>layer % N &lt; N - 1</c>
/// (llama.cpp <c>set_swa_pattern(N, dense_first=false)</c>; gpt-oss N=2 → even layers
/// windowed, odd dense; Gemma-3 uses N=6).
/// </summary>
internal static class CudaSlidingWindowResolver
{
    internal static int Resolve(int? slidingWindowSize, int pattern,
        System.Collections.Generic.IReadOnlyList<int?>? perLayer, int layer)
    {
        if (perLayer is not null && (uint)layer < (uint)perLayer.Count)
            return perLayer[layer] ?? 0;
        if (slidingWindowSize is null) return 0;
        if (pattern <= 0) return slidingWindowSize.Value;
        return (layer % pattern) < pattern - 1 ? slidingWindowSize.Value : 0;
    }
}
```

- [ ] **Step 4: Run test to verify pass**: same filter → all green.
- [ ] **Step 5: Commit**: `git add src/DotLLM.Cuda/CudaSlidingWindowResolver.cs tests/DotLLM.Tests.Unit/Cuda/CudaPerLayerSlidingWindowTests.cs && git commit -m "feat(cuda): per-layer sliding-window resolver matching CPU semantics (#366)"`

### Task 2: Thread the per-layer window through `CudaTransformerModel`'s attention dispatch

**Files:**
- Modify: `src/DotLLM.Cuda/CudaTransformerModel.cs` (the `int slidingWindow = Config.SlidingWindowSize ?? 0;` sites at ~lines 949 and 1962, and every consumer in the layer loops — attention calls at ~1197/1217/1221/1227/1250/1282 for the first body; find the equivalents after 1962 for the second)

**Interfaces:**
- Consumes: `CudaSlidingWindowResolver.Resolve` (Task 1).

- [ ] **Step 1:** Read both forward bodies fully. Replace each hoisted `int slidingWindow = Config.SlidingWindowSize ?? 0;` with a per-layer computation at the TOP of the layer loop body:

```csharp
// Per-layer window: gpt-oss alternates window/dense (pattern=2), Gemma-3 uses
// pattern=6; uniform-window and no-window models resolve identically to the old
// hoisted value. 0 = dense (kernel convention). Mirrors CPU GetLayerSlidingWindow.
int slidingWindow = CudaSlidingWindowResolver.Resolve(
    Config.SlidingWindowSize, Config.SlidingWindowPattern,
    Config.PerLayerSlidingWindow, layer);
```

Keep the local name `slidingWindow` so every downstream consumer (`LaunchAttentionF32`, `_flashAttention.CanUse`, `_g3Attention.CanUse`, split-KV path) is textually unchanged. Verify by diff that ONLY the declaration moved/changed and no call site was edited. IMPORTANT: check whether any consumer sits OUTSIDE the layer loop (e.g. a KV-cache sizing or a scratch decision computed once per forward using the old hoisted value) — if so, that consumer must use the MAXIMUM window across layers (`Config.SlidingWindowSize ?? 0` is exactly that; keep a separately-named hoisted `int maxSlidingWindow` for it and document why).

- [ ] **Step 2:** Repeat for the other CUDA model files the issue names, IF they can ever see a patterned config: `CudaPipelineTransformerModel.cs:436` and `HybridTransformerModel.cs:408`. Both wrap the same generic transformer architectures, so apply the same per-layer resolution there (same code shape). The Qwen3-hybrid/Mamba3/NemotronH models do NOT use `SlidingWindowPattern` (their layer-kind dispatch is separate) — leave untouched.
- [ ] **Step 3:** Build 0/0: `dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj -c Release --no-incremental`.
- [ ] **Step 4:** Regression: `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~Cuda"` — profile must match the pre-change baseline (capture baseline BEFORE editing; the uniform-window models must be bit-identical since the resolver returns the same value for pattern<=0).
- [ ] **Step 5: Commit**: `git commit -am "feat(cuda): apply sliding window per layer in attention dispatch (#366)"`

### Task 3: Discriminating CPU-vs-CUDA parity test for the alternating pattern

**Files:**
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaAlternatingSwaParityTests.cs` (create)

- [ ] **Step 1:** Build a synthetic fixture the way the existing CUDA synthetic parity tests do (find the existing synthetic Llama-shaped fixture builder used by `CudaTransformerModel` unit tests — grep `tests/DotLLM.Tests.Unit/Cuda` for the smallest existing CPU-vs-CUDA `CudaTransformerModel` parity test and mirror its fixture mechanism). Config: 4 layers, `SlidingWindowSize=8`, `SlidingWindowPattern=2`, hidden/heads small but pairwise-distinct dims (per CLAUDE.md: e.g. heads=2, headDim=6, hidden=12), **seqLen=24 — three times the window** so dense layers' receptive field genuinely exceeds the window.
- [ ] **Step 2:** Assert CPU-vs-CUDA last-token logits parity with a documented tolerance calibrated from observed (expect F32 reduction noise ~1e-6; set ~100x margin; state the observed number in the comment — the established pattern).
- [ ] **Step 3:** **Mutation check (required, per the #384/#385 precedent):** temporarily force the resolver to return the window on ALL layers (one-line local mutation), verify the test FAILS with O(0.1+) divergence (dense layers truncated to 8 tokens of context), revert, verify clean `git status`, record both numbers in the test's remarks.
- [ ] **Step 4:** Run on real GPU — must PASS. Commit: `git commit -am "test(cuda): discriminating alternating-SWA CPU-vs-CUDA parity (#366)"`

### Task 4: RoPE-scaling investigation (the issue's bundled second scope item)

**Files:**
- Create: `.docs/366-rope-scaling-findings.md` (git-ignored working notes) and EITHER code changes OR a filed follow-up issue, per the decision gate below.

- [ ] **Step 1:** Establish the facts: (a) read `src/DotLLM.Cuda/CudaTransformerModel.cs:643-650` (RoPE kernel args today: Theta/DimensionCount/Type only); (b) grep `src/DotLLM.Cuda` for `ScalingType|ScalingFactor|AttnFactor|BetaFast|BetaSlow` (issue's claim: zero references — confirm); (c) read the CPU YaRN implementation (`RoPE.PrecomputeFrequencyTableYarn` — grep `src/DotLLM.Cpu` for it) and determine what gpt-oss's config actually ships (`rope_scaling` in its GGUF metadata — check `GgufModelConfigExtractor`'s gpt-oss branch); (d) determine whether YaRN's `attn_factor`/mscale changes SHORT-context numerics for gpt-oss (if mscale multiplies attention scores at all positions, it does — cite the CPU code line that proves it either way).
- [ ] **Step 2: Decision gate.** IF (d) shows short-context numerics are affected → gpt-oss CUDA cannot be numerically correct without it, and the fix belongs HERE: implement the frequency-table port — precompute the YaRN-scaled frequency/mscale table on host with the EXISTING CPU code (call the same `RoPE.PrecomputeFrequencyTableYarn`, no reimplementation), upload once at load (a new device buffer in `CudaWeights` — REMEMBER the #383 allocation ledger), and extend the RoPE launcher/kernel to consume a table pointer when present (this DOES touch a kernel — the Global Constraint "no kernel changes" is scoped to Tasks 1-3; here it's the explicit deliverable; follow the pinned-toolkit PTX rules: `E:\CUDA_v12.8.1\bin\nvcc.exe`, `.version 8.7`, mtime-newer-than-.cu, both fmad lists if NO_FMA is warranted — RoPE is position-encoding, use the same fmad treatment as the existing rope kernel). Add a CPU-vs-CUDA parity case with YaRN active. ELSE (long-context-only) → file a follow-up issue with the findings (title: "CUDA RoPE ignores scaling config (YaRN) — long-context correctness gap"), citing every file:line from Step 1, and paste its number into the PR description.
- [ ] **Step 3:** Whichever branch: commit with the evidence in the message. `git commit -m "feat(cuda)|docs: RoPE-scaling investigation outcome (#366)"`

### Task 5: Docs + wrap

- [ ] **Step 1:** Update `docs/SUPPORTED_MODELS.md`'s gpt-oss row: alternating SWA now supported on CUDA (do NOT claim gpt-oss loads on CUDA — the loader guard stays until #365 lands; say exactly that).
- [ ] **Step 2:** Full CUDA test filter regression + commit docs. Merge via superpowers:finishing-a-development-branch.

## Self-review checklist (author ran per the writing-plans skill)
- Spec coverage: SWA per-layer ✓ (Tasks 1-3), RoPE investigation with both outcomes specified ✓ (Task 4), discriminating test ✓ (Task 3 incl. mutation check), guard explicitly NOT removed ✓ (constraint + Task 5).
- No placeholders: resolver code, test code, and dispatch snippet are complete; Task 4's two branches are each fully specified with concrete steps.
- Type consistency: `Resolve(int?, int, IReadOnlyList<int?>?, int) → int` used identically in Tasks 1-2. `ModelConfig.PerLayerSlidingWindow`'s actual type must be confirmed in Task 1 Step 3 (the CPU code shows `perLayer.Count` + indexer, i.e. IReadOnlyList-compatible; adjust the parameter type to the real property type if it differs — this is the one deliberately-flagged verification point).
