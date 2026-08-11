# Vulkan Quant Residency — Unit 0 + Q5_0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Vulkan F32 weight expansion observable, then eliminate it for Q5_0 by giving that type real device-resident kernels.

**Architecture:** `VulkanWeights.DeviceQuantTypeFor` currently falls through to `QuantizationType.F32` for any type lacking a Vulkan kernel, silently expanding weights 5–7×. Unit 0 adds a load-time residency report (the instrument every later unit uses to prove its kernel is actually routed to) and makes the routed-MoE F32 skip per-bank instead of model-global. Unit 1 then adds the Q5_0 dequant/GEMV/GEMM shaders plus the `KeepQ5_0OnDevice` predicate that makes them reachable.

**Tech Stack:** C# / .NET 10, GLSL compute shaders compiled to SPIR-V with `glslc` (Vulkan 1.2 target), xUnit + `SkippableFact`.

## Global Constraints

- Base branch is `dev`, never `main`. One issue → one branch → one PR. Commit messages carry `(#<n>)`.
- **Never use `git stash`** — the stash stack is shared across worktrees and parallel agents collide on it. Use file copies.
- **Acquire `scripts/gpu-lock.sh` before any GPU run**, release after. Other agents share this GPU; queuing is expected.
- **Full builds only. Never pass `--no-build`.** `.spv` shaders load from the repo tree at runtime while CPU kernels are compiled into `DotLLM.Cpu.dll`, so a stale build yields a half-updated system and a convincing but fictitious divergence (issue #341).
- Shaders are rebuilt with `bash native/vulkan/build.sh`; the resulting `.spv` **is committed** (tracked build artifact).
- `--filter ~Quant` drags in the whole Vulkan GPU suite (46+ min). Use `~Quantiz` for CPU-only runs.
- Model weights never enter the repo. Fixtures resolve via `TestFixtureResolver`.
- **A self-authored fixture may never be the sole oracle** for a layout. Real GGUF bytes are primary.
- **Every new parity test ships a proven negative control** — demonstrate it fails against a deliberately broken variant before trusting it.
- When looping over tensors on the GPU, **allocate device buffers once at maximum size and reuse them**. `DescriptorSetCache` is keyed on raw Vulkan buffer handles and Vulkan recycles handles, so per-iteration alloc/free can bind a dead smaller buffer — writes past its extent become zeros, mimicking kernel truncation.

## File Structure

| file | responsibility |
|---|---|
| `src/DotLLM.Vulkan/VulkanResidencyReport.cs` | **new** — value types describing what was uploaded packed vs expanded |
| `src/DotLLM.Vulkan/VulkanWeights.cs` | residency decisions (`DeviceQuantTypeFor`, `Keep*OnDevice`), upload accounting |
| `native/vulkan/shaders/q5_0_dequant_f32.comp` | **new** — Q5_0 → F32 |
| `native/vulkan/shaders/matmul_q5_0_f32_gemv.comp` | **new** — decode |
| `native/vulkan/shaders/matmul_q5_0_f32_gemm.comp` | **new** — prefill |
| `src/DotLLM.Vulkan/Kernels/Q5_0DequantF32Kernel.cs` | **new** — pipeline/dispatch wrapper |
| `src/DotLLM.Vulkan/Kernels/MatMulQ5_0GemvF32Kernel.cs` | **new** |
| `src/DotLLM.Vulkan/Kernels/MatMulQ5_0GemmF32Kernel.cs` | **new** |
| `tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeQ5_0AnchorTests.cs` | **new** — llama.cpp transcription anchor |
| `tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs` | **new** — real-bytes Vulkan parity |
| `tests/DotLLM.Tests.Unit/Vulkan/VulkanResidencyReportTests.cs` | **new** — accounting correctness |

---

### Task 1: Residency report — make expansion observable

**Files:**
- Create: `src/DotLLM.Vulkan/VulkanResidencyReport.cs`
- Modify: `src/DotLLM.Vulkan/VulkanWeights.cs` (`DeviceQuantTypeFor` ~line 1172, `Upload` ~line 533)
- Test: `tests/DotLLM.Tests.Unit/Vulkan/VulkanResidencyReportTests.cs`

**Interfaces:**
- Produces: `VulkanResidencyReport` with `IReadOnlyList<VulkanTensorResidency> Entries`, `long PackedBytes`, `long UploadedBytes`, `long ExpandedTensorCount`, and `string Describe()`. `VulkanWeights.LastResidencyReport` (static, set by `Upload`) exposes it to tests and callers.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```csharp
[Fact]
public void Report_CountsExpansion_WhenTypeHasNoVulkanKernel()
{
    var report = new VulkanResidencyReport();
    // Q5_0: 22 bytes per 32 elements; expanded to F32 = 128 bytes per 32.
    report.Add("blk.0.ffn_down.weight", QuantizationType.Q5_0, QuantizationType.F32,
        packedBytes: 22 * 64, uploadedBytes: 128 * 64);
    report.Add("blk.0.attn_q.weight", QuantizationType.Q8_0, QuantizationType.Q8_0,
        packedBytes: 34 * 64, uploadedBytes: 34 * 64);

    Assert.Equal(1, report.ExpandedTensorCount);
    Assert.Equal(22 * 64 + 34 * 64, report.PackedBytes);
    Assert.Equal(128 * 64 + 34 * 64, report.UploadedBytes);
    Assert.Contains("blk.0.ffn_down.weight", report.Describe());
    Assert.Contains("Q5_0", report.Describe());
    // The tensor that stayed packed must NOT be listed as expanded.
    Assert.DoesNotContain("blk.0.attn_q.weight", report.Describe());
}

[Fact]
public void Report_IsClean_WhenNothingExpanded()
{
    var report = new VulkanResidencyReport();
    report.Add("blk.0.attn_q.weight", QuantizationType.Q4_K, QuantizationType.Q4_K,
        packedBytes: 144 * 16, uploadedBytes: 144 * 16);

    Assert.Equal(0, report.ExpandedTensorCount);
    Assert.Equal(report.PackedBytes, report.UploadedBytes);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~VulkanResidencyReportTests" --nologo`
Expected: FAIL — `VulkanResidencyReport` does not exist.

- [ ] **Step 3: Implement `VulkanResidencyReport`**

```csharp
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan;

/// <summary>One tensor's on-device residency outcome.</summary>
public readonly record struct VulkanTensorResidency(
    string Name, QuantizationType Source, QuantizationType Device,
    long PackedBytes, long UploadedBytes)
{
    /// <summary>True when the tensor could not be kept packed and was widened on upload.</summary>
    public bool Expanded => Device != Source;
}

/// <summary>
/// Accounting for what Vulkan kept packed and what it widened to F32.
/// <para>
/// This exists because <c>VulkanWeights.DeviceQuantTypeFor</c> ends in an unconditional
/// <c>return QuantizationType.F32</c>: a type with no Vulkan kernel expands 5-7x on upload
/// with no diagnostic at all. It is also how a new kernel is proven to be genuinely routed
/// to — a capability flag says a path exists, not that it ran.
/// </para>
/// </summary>
public sealed class VulkanResidencyReport
{
    private readonly List<VulkanTensorResidency> _entries = new();

    public IReadOnlyList<VulkanTensorResidency> Entries => _entries;
    public long PackedBytes { get; private set; }
    public long UploadedBytes { get; private set; }
    public int ExpandedTensorCount { get; private set; }

    public void Add(string name, QuantizationType source, QuantizationType device,
        long packedBytes, long uploadedBytes)
    {
        var entry = new VulkanTensorResidency(name, source, device, packedBytes, uploadedBytes);
        _entries.Add(entry);
        PackedBytes += packedBytes;
        UploadedBytes += uploadedBytes;
        if (entry.Expanded) ExpandedTensorCount++;
    }

    /// <summary>Human-readable summary; lists only the tensors that were widened.</summary>
    public string Describe()
    {
        if (ExpandedTensorCount == 0)
            return $"Vulkan residency: all {_entries.Count} tensors kept packed ({PackedBytes:N0} bytes).";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine(
            $"Vulkan residency: {ExpandedTensorCount} of {_entries.Count} tensors widened on upload "
            + $"({PackedBytes:N0} packed -> {UploadedBytes:N0} uploaded bytes, "
            + $"{(double)UploadedBytes / Math.Max(PackedBytes, 1):F1}x).");
        foreach (var group in _entries.Where(e => e.Expanded)
                     .GroupBy(e => (e.Source, e.Device))
                     .OrderByDescending(g => g.Sum(e => e.UploadedBytes - e.PackedBytes)))
        {
            long extra = group.Sum(e => e.UploadedBytes - e.PackedBytes);
            sb.AppendLine(
                $"  {group.Key.Source} -> {group.Key.Device}: {group.Count()} tensor(s), "
                + $"+{extra:N0} bytes. First: {group.First().Name}");
        }
        return sb.ToString().TrimEnd();
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~VulkanResidencyReportTests" --nologo`
Expected: PASS, 2 tests.

- [ ] **Step 5: Wire it into `VulkanWeights.Upload`**

In `VulkanWeights.cs`, add a static `public static VulkanResidencyReport? LastResidencyReport { get; private set; }`. In `Upload`, create a fresh report at the start (next to the existing `ResetUploadCounters()` call) and assign it to `LastResidencyReport` before returning. In `UploadMatrix`, after `DeviceQuantTypeFor` resolves the device type, record the entry using the already-computed source type, `Dequantize.RowByteSize(inputDim, srcQt) * outputDim` for packed bytes, and the existing `uploadedBytes` out-parameter for uploaded bytes.

- [ ] **Step 6: Verify on a real model**

Run, with the GPU lock held:
```bash
bash scripts/gpu-lock.sh acquire residency "residency report check" 1800
dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj -c Release \
  --filter "FullyQualifiedName~RealGgufVulkanParityTests.Llama32_1B_Q8_0" --nologo
bash scripts/gpu-lock.sh release residency
```
Expected: PASS, and `LastResidencyReport.ExpandedTensorCount` is 0 for this all-Q8_0 model. Add a one-line assertion of that to the test if it is not already observable.

- [ ] **Step 7: Commit**

```bash
git add src/DotLLM.Vulkan/VulkanResidencyReport.cs src/DotLLM.Vulkan/VulkanWeights.cs \
        tests/DotLLM.Tests.Unit/Vulkan/VulkanResidencyReportTests.cs
git commit -m "feat(vulkan): report weight expansion instead of widening silently (#344)"
```

---

### Task 2: Per-bank routed-MoE residency

**Files:**
- Modify: `src/DotLLM.Vulkan/VulkanWeights.cs` (`CanSkipMoeF32HostDequant`, ~line 1965)
- Modify: `src/DotLLM.Vulkan/VulkanTransformerModel.cs:1155` and `:1184` (call sites)
- Test: `tests/DotLLM.Tests.Unit/Vulkan/VulkanResidencyReportTests.cs`

**Interfaces:**
- Consumes: `VulkanResidencyReport` from Task 1.
- Produces: `CanSkipMoeF32HostDequant` resolved **per expert bank** rather than per model.

**Context:** today one unsupported sibling forces every routed bank to host F32. In DeepSeek-V2-Lite Q4_K_M, 64 of 78 banks are already resident-capable and are widened only because 14 are Q5_0 — so per-bank resolution alone cuts the fallback ~5× with no new kernels. Tracked as issue #327.

- [ ] **Step 1: Write the failing test**

```csharp
[Fact]
public void MoeSkip_IsPerBank_NotModelGlobal()
{
    // 3 banks: two Q4_K (resident-capable), one Q5_0 (not, before Unit 1).
    var banks = new[]
    {
        (Name: "blk.0.ffn_gate_exps.weight", Qt: QuantizationType.Q4_K),
        (Name: "blk.0.ffn_up_exps.weight",   Qt: QuantizationType.Q4_K),
        (Name: "blk.0.ffn_down_exps.weight", Qt: QuantizationType.Q5_0),
    };

    var resident = banks
        .Where(b => VulkanWeights.CanKeepBankResident(b.Qt, inputDim: 1408))
        .Select(b => b.Name)
        .ToArray();

    Assert.Equal(2, resident.Length);
    Assert.DoesNotContain("blk.0.ffn_down_exps.weight", resident);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~MoeSkip_IsPerBank" --nologo`
Expected: FAIL — `CanKeepBankResident` does not exist.

- [ ] **Step 3: Implement**

Add to `VulkanWeights`:

```csharp
/// <summary>
/// True when a single routed-expert bank can be held on device in packed form.
/// <para>
/// Resolved per bank on purpose. Deciding this model-globally means one unsupported
/// sibling widens every bank: in DeepSeek-V2-Lite Q4_K_M, 64 of 78 banks are
/// resident-capable and were widened only because 14 are Q5_0 (#327).
/// </para>
/// </summary>
public static bool CanKeepBankResident(QuantizationType qt, int inputDim)
    => DeviceQuantTypeFor(qt, inputDim, dequantToFp32: false) == qt;
```

Then change `CanSkipMoeF32HostDequant` to evaluate `CanKeepBankResident` per bank and return the per-bank decision to callers rather than a single model-wide bool.

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~MoeSkip_IsPerBank" --nologo`
Expected: PASS.

- [ ] **Step 5: Verify no numeric change on a real MoE model**

Run, with the GPU lock held, the Gemma-4 MoE Vulkan decode test (the existing routed-MoE coverage). Expected: unchanged pass — this task changes *how much* is widened, never the values.

- [ ] **Step 6: Commit**

```bash
git add src/DotLLM.Vulkan/VulkanWeights.cs src/DotLLM.Vulkan/VulkanTransformerModel.cs \
        tests/DotLLM.Tests.Unit/Vulkan/VulkanResidencyReportTests.cs
git commit -m "fix(vulkan): resolve routed-MoE residency per bank, not per model (#327, #344)"
```

---

### Task 3: llama.cpp anchor for CPU Q5_0 dequant

**Files:**
- Create: `tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeQ5_0AnchorTests.cs`

**Interfaces:**
- Consumes: `DotLLM.Cpu.Kernels.Dequantize.DequantizeQ5_0Scalar(nint src, long elementCount, Span<float> dest)` (internal; visible via `InternalsVisibleTo`).
- Produces: the trusted CPU oracle every later Q5_0 task compares against.

**Context:** `Q3_K` is currently the *only* dequant type with a literal llama.cpp transcription test, and Q3_K is exactly the one that shipped transposed. This must merge **before** the Q5_0 kernel PR merges.

- [ ] **Step 1: Write the failing test**

```csharp
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Anchors <see cref="Dequantize.DequantizeQ5_0Scalar"/> to a literal transcription of
/// llama.cpp's <c>dequantize_row_q5_0</c> over dense random blocks.
/// <para>
/// Structurally unlike the production indexing on purpose: the reference below keeps
/// llama.cpp's own control flow and its <c>xh_0 / xh_1</c> shift trick rather than
/// re-deriving bit positions. Q5_0 hides the 5th bit of each weight in a separate
/// <c>qh</c> word, which is the same class of indexing that Q3_K got transposed.
/// </para>
/// </summary>
public unsafe class DequantizeQ5_0AnchorTests
{
    private const int Qk = 32;
    private const int BlockBytes = 22;

    [Fact]
    public void Q5_0_DenseRandomBlocks_MatchLlamaCppReference()
    {
        const int blocks = 64;
        var rng = new Random(0x5C0DE);
        var raw = new byte[blocks * BlockBytes];
        rng.NextBytes(raw);
        // Keep d in a sane finite range: overwrite each block's fp16 delta.
        for (int b = 0; b < blocks; b++)
        {
            Half d = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            Unsafe.WriteUnaligned(ref raw[b * BlockBytes], d);
        }

        var actual = new float[blocks * Qk];
        var expected = new float[blocks * Qk];
        fixed (byte* p = raw)
        {
            Dequantize.DequantizeQ5_0Scalar((nint)p, blocks * Qk, actual);
            LlamaCppDequantizeRowQ5_0(p, blocks, expected);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i],
                $"element {i}: llama.cpp={expected[i]:R} dotLLM={actual[i]:R}");
    }

    /// <summary>
    /// Literal transcription of llama.cpp <c>ggml-quants.c dequantize_row_q5_0</c>,
    /// kept in its original control-flow shape on purpose.
    /// </summary>
    private static void LlamaCppDequantizeRowQ5_0(byte* x, int nb, Span<float> y)
    {
        for (int i = 0; i < nb; i++)
        {
            byte* blk = x + i * BlockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(blk);
            uint qh = Unsafe.ReadUnaligned<uint>(blk + 2);
            byte* qs = blk + 6;

            for (int j = 0; j < Qk / 2; j++)
            {
                byte xh_0 = (byte)(((qh >> (j + 0)) << 4) & 0x10);
                byte xh_1 = (byte)((qh >> (j + 12)) & 0x10);

                int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
                int x1 = ((qs[j] >> 4) | xh_1) - 16;

                y[i * Qk + j + 0] = x0 * d;
                y[i * Qk + j + Qk / 2] = x1 * d;
            }
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~DequantizeQ5_0AnchorTests" --nologo`
Expected: PASS. If it FAILS, **stop** — the CPU oracle is wrong, and that is a bigger finding than this plan. File it and fix CPU before continuing.

- [ ] **Step 3: Prove the anchor discriminates (negative control)**

Temporarily change `xh_1` in the reference to `(byte)((qh >> (j + 13)) & 0x10)` (a one-bit shift error) and re-run. Expected: FAIL. Revert the change and re-run. Expected: PASS. Do not commit the broken variant.

- [ ] **Step 4: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeQ5_0AnchorTests.cs
git commit -m "test(quant): anchor CPU Q5_0 dequant to a llama.cpp transcription (#344)"
```

---

### Task 4: Q5_0 dequant shader + kernel

**Files:**
- Create: `native/vulkan/shaders/q5_0_dequant_f32.comp`
- Create: `src/DotLLM.Vulkan/Kernels/Q5_0DequantF32Kernel.cs`
- Create: `tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs`

**Interfaces:**
- Consumes: the CPU oracle anchored in Task 3.
- Produces: `Q5_0DequantF32Kernel.Create(VulkanDevice device, string spvDir)`, `Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalBlocks)`, `Record(nint cmdBuf, ...)` — same shape as `Q3KDequantF32Kernel`.

- [ ] **Step 1: Write the shader**

```glsl
#version 450
// Q5_0 dequantization to FP32.
//
// Block layout (22 bytes, 32 elements), matching llama.cpp's block_q5_0 and
// DotLLM.Cpu.Kernels.Dequantize.DequantizeQ5_0Scalar byte-for-byte:
//   bytes [0,1]   = fp16 d
//   bytes [2..5]  = qh (uint32) — the 5th bit of every element
//   bytes [6..21] = qs[16]      — two 4-bit nibbles per byte
//
// Per-element decode, j in [0,16):
//   lo = qs[j] & 0xF          high bit = (qh >> j)        & 1  -> element j
//   hi = (qs[j] >> 4) & 0xF   high bit = (qh >> (j + 16)) & 1  -> element j + 16
//   value = d * ((bit << 4 | nibble) - 16)
//
// The 5th bit living in a separate word is the same class of indexing Q3_K got
// transposed, so this mapping is asserted against real GGUF bytes, not a fixture.
//
// Dispatch: one workgroup per 8 blocks (256 elements), 256 threads, one element each.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufW { uint  src[]; };
layout(set = 0, binding = 1, std430) writeonly buffer BufY { float dst[]; };

layout(push_constant) uniform PushConstants {
    uint totalBlocks;
    uint srcUints;
} pc;

const uint Q5_0_BLOCK_BYTES = 22u;
const uint Q5_0_GROUP_SIZE  = 32u;
const uint BLOCKS_PER_WG    = 8u;

uint readByte(uint absByteOff) {
    uint u = src[absByteOff >> 2u];
    uint shift = (absByteOff & 3u) * 8u;
    return (u >> shift) & 0xFFu;
}

float readHalf(uint absByteOff) {
    uint alignedIdx = absByteOff >> 2u;
    uint byteInWord = absByteOff & 3u;
    uint u = src[alignedIdx];
    uint half16;
    if (byteInWord <= 2u) {
        half16 = (u >> (byteInWord * 8u)) & 0xFFFFu;
    } else {
        uint uNext = src[alignedIdx + 1u];
        half16 = ((u >> 24) & 0xFFu) | ((uNext & 0xFFu) << 8);
    }
    return unpackHalf2x16(half16).x;
}

uint readUint(uint absByteOff) {
    return readByte(absByteOff)
         | (readByte(absByteOff + 1u) << 8)
         | (readByte(absByteOff + 2u) << 16)
         | (readByte(absByteOff + 3u) << 24);
}

void main() {
    uint local = gl_LocalInvocationID.x;          // 0..255
    uint blockInWg = local / Q5_0_GROUP_SIZE;     // 0..7
    uint t = local % Q5_0_GROUP_SIZE;             // 0..31
    uint blockIdx = gl_WorkGroupID.x * BLOCKS_PER_WG + blockInWg;
    if (blockIdx >= pc.totalBlocks) return;

    uint base = blockIdx * Q5_0_BLOCK_BYTES;
    float d = readHalf(base);
    uint qh = readUint(base + 2u);

    uint j = t & 15u;                             // nibble index 0..15
    uint qsByte = readByte(base + 6u + j);
    uint nibble = (t < 16u) ? (qsByte & 0xFu) : ((qsByte >> 4) & 0xFu);
    uint bitPos = (t < 16u) ? j : (j + 16u);
    uint bit5 = (qh >> bitPos) & 1u;

    int q = int(nibble | (bit5 << 4)) - 16;
    dst[blockIdx * Q5_0_GROUP_SIZE + t] = d * float(q);
}
```

- [ ] **Step 2: Compile the shader**

Run: `bash native/vulkan/build.sh`
Expected: `q5_0_dequant_f32.comp -> q5_0_dequant_f32.spv`, no errors.

- [ ] **Step 3: Add the kernel wrapper**

Create `src/DotLLM.Vulkan/Kernels/Q5_0DequantF32Kernel.cs` as a copy of `Q3KDequantF32Kernel.cs` with these changes: `Q5_0BlockBytes = 22`, `Q5_0GroupSize = 32`, shader file `q5_0_dequant_f32.spv`, and the dispatch group count `(totalBlocks + 7) / 8` instead of `totalBlocks` (8 blocks per workgroup). Keep the 2 bindings, the `PushConstantBytes = 2 * sizeof(uint)` push constants, the buffer-size guards and the `DescriptorSetCache` usage unchanged.

- [ ] **Step 4: Write the real-bytes parity test**

Create `tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs` modelled on `RealGgufQ3KDequantParityTests.cs`. It must:
- resolve a real Q5_0 GGUF via `TestFixtureResolver`;
- `Assert` the file actually contains `QuantizationType.Q5_0` tensors, listing the types present if not;
- **allocate `bufSrc`/`bufDst` once at the maximum size and reuse them** across tensors;
- compare Vulkan output against `Dequantize.DequantizeQ5_0` exactly (0 ULP — both do one multiply per element, no reduction);
- report tensors and blocks checked via `ITestOutputHelper`.

- [ ] **Step 5: Run it**

```bash
bash scripts/gpu-lock.sh acquire q5_0 "Q5_0 real-bytes dequant parity" 1800
dotnet build tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj -c Release
dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj -c Release \
  --filter "FullyQualifiedName~RealGgufQ5_0ParityTests" --nologo
bash scripts/gpu-lock.sh release q5_0
```
Expected: PASS.

- [ ] **Step 6: Prove it discriminates (negative control)**

Temporarily change the shader's `bitPos` for the high half from `j + 16u` to `j + 15u`, rebuild with `native/vulkan/build.sh`, re-run the test. Expected: FAIL. Revert, rebuild, re-run. Expected: PASS. Record both outcomes in the PR. Do not commit the broken shader or its `.spv`.

- [ ] **Step 7: Commit**

```bash
git add native/vulkan/shaders/q5_0_dequant_f32.comp native/vulkan/spv/q5_0_dequant_f32.spv \
        src/DotLLM.Vulkan/Kernels/Q5_0DequantF32Kernel.cs \
        tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs
git commit -m "feat(vulkan): Q5_0 dequant kernel, verified on real GGUF bytes (#344)"
```

---

### Task 5: Q5_0 F32 GEMV (decode)

**Files:**
- Create: `native/vulkan/shaders/matmul_q5_0_f32_gemv.comp`
- Create: `src/DotLLM.Vulkan/Kernels/MatMulQ5_0GemvF32Kernel.cs`
- Modify: `tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs`

**Interfaces:**
- Consumes: the block decode proven in Task 4.
- Produces: `MatMulQ5_0GemvF32Kernel.Create(device, spvDir)` and `Launch(bufW, bufX, bufY, int m, int k)` — same signature as `MatMulQ3KGemvF32Kernel`.

- [ ] **Step 1: Write the shader**

Model it on `native/vulkan/shaders/matmul_q3_k_f32_gemv.comp`. One row per workgroup; each thread walks a strided subset of the row's blocks, decoding with the exact expression proven in Task 4 (`nibble | (bit5 << 4)) - 16`, `bit` at `j` for the low half and `j + 16` for the high half), multiplying by the matching `x[]` element and accumulating into a shared-memory reduction. **Follow the coalesced access pattern** used by the current decode GEMVs — consecutive threads must read consecutive `qs` bytes; that coalescing was the largest decode win recorded in this backend (#338/#339).

- [ ] **Step 2: Compile**

Run: `bash native/vulkan/build.sh`
Expected: `matmul_q5_0_f32_gemv.comp -> matmul_q5_0_f32_gemv.spv`.

- [ ] **Step 3: Add the kernel wrapper**

Copy `src/DotLLM.Vulkan/Kernels/MatMulQ3KGemvF32Kernel.cs`, changing the shader name to `matmul_q5_0_f32_gemv.spv` and the block constants to 22 bytes / 32 elements. Keep the 3-binding layout (`w`, `x`, `y`) and push constants `(m, k)`.

- [ ] **Step 4: Add a real-bytes GEMV case**

Extend `RealGgufQ5_0ParityTests` with a GEMV method mirroring `Bielik15B_Q3_K_RealGgufBytes_VulkanGemv_MatchesCpuReference`: dequantise the tensor with the CPU oracle, take a random `x`, compute the reference dot per row in `double`, then compare against the Vulkan kernel relative to row magnitude (bound `2e-2`). **Allocate `bufW`/`bufX`/`bufY` once at the maximum size across tensors and reuse them.**

- [ ] **Step 5: Run**

```bash
bash scripts/gpu-lock.sh acquire q5_0 "Q5_0 GEMV real-bytes parity" 1800
dotnet build tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj -c Release
dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj -c Release \
  --filter "FullyQualifiedName~RealGgufQ5_0ParityTests" --nologo
bash scripts/gpu-lock.sh release q5_0
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add native/vulkan/shaders/matmul_q5_0_f32_gemv.comp native/vulkan/spv/matmul_q5_0_f32_gemv.spv \
        src/DotLLM.Vulkan/Kernels/MatMulQ5_0GemvF32Kernel.cs \
        tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs
git commit -m "feat(vulkan): Q5_0 F32 GEMV decode kernel (#344)"
```

---

### Task 6: Q5_0 F32 GEMM (prefill)

**Files:**
- Create: `native/vulkan/shaders/matmul_q5_0_f32_gemm.comp`
- Create: `src/DotLLM.Vulkan/Kernels/MatMulQ5_0GemmF32Kernel.cs`
- Modify: `tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs`

**Interfaces:**
- Produces: `MatMulQ5_0GemmF32Kernel.Create(device, spvDir)` and `Launch(bufW, bufX, bufY, int n, int m, int k)` — same signature as `MatMulQ3KGemmF32Kernel`.

- [ ] **Step 1: Write the shader**

Model it on `native/vulkan/shaders/matmul_q3_k_f32_gemm.comp`, substituting the Q5_0 block decode from Task 4. Keep that file's tiling and shared-memory staging unchanged — do not introduce a new tile shape here. Tile-shape experiments on this backend are a closed, four-times-refuted line (#384–#387); this task is coverage, not a tuning exercise.

- [ ] **Step 2: Compile**

Run: `bash native/vulkan/build.sh`
Expected: `matmul_q5_0_f32_gemm.comp -> matmul_q5_0_f32_gemm.spv`.

- [ ] **Step 3: Add the kernel wrapper**

Copy `src/DotLLM.Vulkan/Kernels/MatMulQ3KGemmF32Kernel.cs`, changing the shader name and block constants as in Task 5.

- [ ] **Step 4: Add real-bytes GEMM cases**

Extend `RealGgufQ5_0ParityTests` with a GEMM method covering **two token counts, `n = 6` and `n = 68`** (a partial tile and a multi-tile case, matching the shapes PR #340 used for Q3_K). Reference is the same dequantise-then-F32-matmul path. Reuse buffers across tensors.

- [ ] **Step 5: Run**

Same commands as Task 5 Step 5. Expected: PASS on both `n` values.

- [ ] **Step 6: Commit**

```bash
git add native/vulkan/shaders/matmul_q5_0_f32_gemm.comp native/vulkan/spv/matmul_q5_0_f32_gemm.spv \
        src/DotLLM.Vulkan/Kernels/MatMulQ5_0GemmF32Kernel.cs \
        tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs
git commit -m "feat(vulkan): Q5_0 F32 GEMM prefill kernel (#344)"
```

---

### Task 7: Make Q5_0 device-resident and prove it

**Files:**
- Modify: `src/DotLLM.Vulkan/VulkanWeights.cs` (`DeviceQuantTypeFor` ~line 1172, new `KeepQ5_0OnDevice`)
- Modify: `src/DotLLM.Vulkan/VulkanTransformerModel.cs` (matmul dispatch for the new device type)
- Test: `tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs`

**Interfaces:**
- Consumes: Task 1's `VulkanResidencyReport`; Tasks 4–6's kernels.
- Produces: `Q5_0` returned from `DeviceQuantTypeFor` for Q5_0 sources.

**Context:** without this task the kernels exist but are unreachable — `DeviceQuantTypeFor` still falls through to F32.

- [ ] **Step 1: Write the failing test**

```csharp
[SkippableFact]
public void Q5_0_Model_KeepsWeightsPacked_NoF32Expansion()
{
    FixtureLocation fixture = TestFixtureResolver.ResolveFile(/* the Q5_0 fixture */);
    Skip.If(!fixture.Found, fixture.SkipMessage("Q5_0 GGUF"));

    // load the model on Vulkan here, then:
    VulkanResidencyReport report = VulkanWeights.LastResidencyReport!;
    var q5Expanded = report.Entries
        .Where(e => e.Source == QuantizationType.Q5_0 && e.Expanded)
        .ToArray();

    Assert.True(q5Expanded.Length == 0,
        $"Q5_0 tensors still widened on upload:\n{report.Describe()}");
}
```

- [ ] **Step 2: Run to verify it fails**

Run with the GPU lock held. Expected: FAIL, listing Q5_0 → F32 expansions. **This failure is the proof that the report works and that the type is not yet routed.**

- [ ] **Step 3: Add the predicate and wire it**

```csharp
/// <summary>Returns true when the matrix will be kept on device as Q5_0 blocks
/// (22 bytes per 32 elements). Q5_0 blocks are 32 elements, so unlike the K-quants
/// this needs no 256-alignment gate.</summary>
private static bool KeepQ5_0OnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
    => !dequantToFp32 && qt == QuantizationType.Q5_0 && (inputDim % 32) == 0;
```

Add `if (KeepQ5_0OnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q5_0;` to `DeviceQuantTypeFor`, and register the Q5_0 GEMV/GEMM kernels in `VulkanTransformerModel`'s matmul dispatch next to the existing Q3_K/Q4_K entries.

- [ ] **Step 4: Run to verify it passes**

Run with the GPU lock held. Expected: PASS, `report.Describe()` shows all tensors kept packed.

- [ ] **Step 5: Run the full Vulkan suite**

```bash
bash scripts/gpu-lock.sh acquire q5_0 "full Vulkan suite" 3600
dotnet build tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release \
  --filter "Category=GPU" --nologo
bash scripts/gpu-lock.sh release q5_0
```
Expected: no new failures. Six `VulkanMatMulI2SGemvF32KernelTests.MultiRow` ULP failures are pre-existing on `dev` (issue #332) — confirm the count matches the baseline rather than assuming.

- [ ] **Step 6: Commit and open the PR**

```bash
git add src/DotLLM.Vulkan/VulkanWeights.cs src/DotLLM.Vulkan/VulkanTransformerModel.cs \
        tests/DotLLM.Tests.Integration/Vulkan/RealGgufQ5_0ParityTests.cs
git commit -m "feat(vulkan): keep Q5_0 weights device-resident instead of widening to F32 (#344)"
git push -u origin issue/344-vulkan-q5_0-residency
```

PR body must record: the residency report before/after (bytes packed vs uploaded), the negative-control evidence from Task 4 Step 6, and confirmation that Task 3's anchor is merged.

---

## Out of scope for this plan

MMVQ, MMQ and `moe_indexed_matmul_q5_0_*` are deliberately excluded. MMVQ and the MoE-indexed path get their own plan once Task 1's report can quantify what they buy; MMQ is measurement-gated per the spec, because #384–#391 refuted four MMQ tiling variants on this hardware. DeepSeek-V2-Lite's OOM is only fully resolved once the MoE-indexed path lands — Task 2 reduces it, and the acceptance criterion for it belongs to that later plan.

Types Q4_0, Q4_1, Q5_1 and MXFP4 follow this same task shape and are separate units per the spec.
