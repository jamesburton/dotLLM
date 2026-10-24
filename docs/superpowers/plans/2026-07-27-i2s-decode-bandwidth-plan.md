# I2_S Decode Bandwidth Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether CPU BitNet (I2_S) GEMV decode, after the #128 unpack-SIMD fix already on `dev`, is now memory-bandwidth-bound or still compute-bound on this Strix Halo box, to gate whether the AVX-512 activation-LUT dot kernel (suggested upstream in issue #334) is worth building next.

**Architecture:** Measurement-only. Adds one small kernel-side helper (`BenchStreamingReadOnly`, a vectorized touch-every-byte loop with no unpack/decode) next to the existing `BenchUnpackRowI8Only` helper from #128/#131. Follows this repo's **existing convention** for this exact kind of investigation — `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SUnpackProfileBench.cs` is a `TEMPORARY` xUnit `[Theory]` using `Stopwatch` + `ITestOutputHelper` to print wall-clock numbers, not a `BenchmarkDotNet` class — so this plan adds a sibling profiling test in the same style rather than introducing a new benchmark harness. (This deviates from the design doc's original "commit a BenchmarkDotNet class" framing in favor of matching the pattern the codebase already uses for this precise scenario; still measurement-only, same shapes, same comparison.) Two access patterns are measured per shape: **hot** (same weight buffer reused every iteration, matching #131's original methodology and mirroring how a resident weight tensor behaves across consecutive decode steps) and **cold** (round-robin across enough distinct buffers to exceed L3, forcing genuine DRAM traffic) — comparing both against full-decode GB/s tells us whether we're cache-bound, DRAM-bound, or compute-bound.

**Tech Stack:** C# / .NET 10, `System.Runtime.Intrinsics.X86.Avx2`, xUnit v3, `ITestOutputHelper`, `System.Diagnostics.Stopwatch`.

**Spec reference:** `docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-design.md`.

**Effort estimate:** ~1-2 hours, no new production logic, 1 PR.

## Global Constraints

- Existing shapes from #131, for continuity with prior data: `attn_qproj` (m=2560, k=2560), `ffn_gate` (m=6912, k=2560), `ffn_down` (m=2560, k=6912).
- Single-thread only (`threadPool: null` on `GemvI2_S`) — matches #131's methodology; multi-thread dispatch is orthogonal to the per-core bandwidth question and would confound it.
- No changes to `src/DotLLM.Cpu/Kernels/MatMul.I2S.cs` production kernels — this plan is purely additive profiling infra.
- The new kernel-side helper lives in `src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs` (the existing home for #128/#131's `BenchUnpackRowI8Only`), same `unsafe partial class MatMul` pattern.
- Every new/modified file must build under the project's existing `<Nullable>enable</Nullable>` + `[SkipLocalsInit]`/`[MethodImpl(AggressiveOptimization)]` conventions already used in this file.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs` | Add `BenchStreamingReadOnly(byte* weights, int m, int k)` — vectorized (AVX2) touch-every-packed-byte loop, no unpack, returns a checksum byte to defeat dead-code elimination. |
| Modify | `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SUnpackProfileBench.cs` | Add a smoke assertion that `BenchStreamingReadOnly` and a scalar reference agree on the XOR checksum (cheap correctness guard for the new helper; not a performance test). |
| Create | `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SDecodeBandwidthProfileBench.cs` | TEMPORARY profiling `[Theory]` test (same style as `I2SUnpackProfileBench`): for each of the 3 shapes, times hot-streaming, cold-streaming, unpack-only, and full `GemvI2_S` decode; prints ns/row, GB/s, and ratio-to-ceiling via `ITestOutputHelper`. |
| Create | `docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-RESULTS.md` | Captured benchmark output + go/no-go verdict, written after running the profiling test on this box. |

No changes to production kernels. No new abstractions.

---

### Task 0: File the tracking issue and create the branch

Per this repo's issue-driven workflow (every task starts with a GitHub issue; dedicated branch
per issue): this is a small, already-scoped task, so file the issue with the acceptance criteria
below rather than iterating on scope in the issue itself.

- [ ] **Step 1: File the tracking issue**

`gh issue create --repo jamesburton/dotLLM --title "bench(cpu/i2s): profile GEMV decode bandwidth vs compute" --body "Measure whether I2_S GEMV decode (post-#128 unpack fix) is memory-bandwidth-bound or compute-bound on Strix Halo, to gate the AVX-512 activation-LUT dot kernel proposed upstream in #334. Design: docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-design.md.\n\nAcceptance criteria:\n- [ ] BenchStreamingReadOnly helper added and smoke-tested\n- [ ] Profiling test measures hot/cold streaming ceiling vs unpack-only vs full decode for the 3 #131 shapes\n- [ ] Results recorded with a go/no-go verdict\n- [ ] Verdict posted to upstream #334; follow-up issue filed per the verdict"`

Note the returned issue number — it's `{N}` in the branch name and PR below.

- [ ] **Step 2: Create the branch**

```bash
git checkout main
git pull
git checkout -b issue/{N}-i2s-decode-bandwidth-profiling
```

(Branch from `main` per the workflow rule; this is standalone profiling work, not part of the
`dev` WIP-integration flow described in CLAUDE.md's "large pre-existing feature branches" section.)

---

### Task 1: Add `BenchStreamingReadOnly` kernel-side helper

**Files:**
- Modify: `src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs`
- Test: `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SUnpackProfileBench.cs`

**Interfaces:**
- Produces: `public static byte MatMul.BenchStreamingReadOnly(byte* weights, int m, int k)` — reads exactly `m * (k/4)` packed bytes (the same footprint `UnpackRowI8`/`GemvI2_S` touch per call), does no unpack/decode, returns an XOR checksum of every byte read (forces the JIT to keep the reads live).

- [ ] **Step 1: Write the failing smoke test**

Add to `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SUnpackProfileBench.cs` (new `[Fact]` in the existing `I2SUnpackProfileBench` class):

```csharp
    [Fact]
    public void BenchStreamingReadOnly_MatchesScalarChecksum()
    {
        var rng = new Random(7);
        const int m = 64, k = 512;
        int rowBytes = k / 4;
        byte[] buf = new byte[m * rowBytes];
        rng.NextBytes(buf);

        byte expected = 0;
        foreach (byte b in buf) expected ^= b;

        fixed (byte* p = buf)
        {
            byte actual = MatMul.BenchStreamingReadOnly(p, m, k);
            Assert.Equal(expected, actual);
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~I2SUnpackProfileBench.BenchStreamingReadOnly_MatchesScalarChecksum"`
Expected: FAIL — `CS0117` / build error, `MatMul` has no `BenchStreamingReadOnly`.

- [ ] **Step 3: Implement the helper**

Add to `src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs` (inside the existing `partial class MatMul`, after `BenchUnpackRowI8Only`):

```csharp
    /// <summary>
    /// Reads every packed byte a full <see cref="GemvI2_S(byte*, float*, float*, int, int, ComputeThreadPool?)"/>
    /// call would touch (<c>m·k/4</c> bytes), doing no unpack/decode — an empirical
    /// achievable-bandwidth probe for this exact access pattern. Returns an XOR checksum so the
    /// JIT cannot eliminate the reads as dead code.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static byte BenchStreamingReadOnly(byte* weights, int m, int k)
    {
        long totalBytes = (long)m * (k / 4);
        Vector256<byte> acc = Vector256<byte>.Zero;
        long i = 0;

        if (Avx2.IsSupported)
        {
            long vecEnd = totalBytes - (totalBytes % 32);
            for (; i < vecEnd; i += 32)
                acc ^= Unsafe.ReadUnaligned<Vector256<byte>>(weights + i);
        }

        Span<byte> lanes = stackalloc byte[32];
        acc.CopyTo(lanes);
        byte result = 0;
        foreach (byte lane in lanes) result ^= lane;

        for (; i < totalBytes; i++)
            result ^= weights[i];

        return result;
    }
```

Add the required usings at the top of the file if not already present (`System.Runtime.CompilerServices`, `System.Runtime.Intrinsics`, `System.Runtime.Intrinsics.X86`) — check the existing usings first since `MatMul.I2S.cs` (the partial sibling) already has them; `MatMul.I2S.Bench.cs` currently only has `using System.Buffers;` so add:

```csharp
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~I2SUnpackProfileBench.BenchStreamingReadOnly_MatchesScalarChecksum"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SUnpackProfileBench.cs
git commit -m "bench(cpu/i2s): add streaming-read-only bandwidth probe helper"
```

---

### Task 2: Add the decode-bandwidth profiling test

**Files:**
- Create: `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SDecodeBandwidthProfileBench.cs`

**Interfaces:**
- Consumes: `MatMul.BenchStreamingReadOnly(byte*, int, int)` (Task 1), `MatMul.BenchUnpackRowI8Only(byte*, int, int)` (existing, `MatMul.I2S.Bench.cs`), `MatMul.GemvI2_S(byte*, float*, float*, int, int, float, ComputeThreadPool?)` (existing explicit-scale overload — avoids needing a tail-scale float appended to the buffer).
- Produces: test-output-only (`ITestOutputHelper`); no new public surface for later tasks to consume.

- [ ] **Step 1: Write the profiling test**

Create `tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SDecodeBandwidthProfileBench.cs`:

```csharp
using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// TEMPORARY profiling harness — measures whether I2_S GEMV decode (post-#128 unpack-SIMD fix) is
/// memory-bandwidth-bound or still compute-bound on this box, gating the AVX-512 activation-LUT
/// dot kernel proposed upstream in issue #334. Not a correctness test; prints wall-clock numbers
/// via test output. Delete once the go/no-go verdict is recorded.
/// </summary>
public sealed unsafe class I2SDecodeBandwidthProfileBench
{
    private readonly ITestOutputHelper _output;

    public I2SDecodeBandwidthProfileBench(ITestOutputHelper output) => _output = output;

    /// <summary>Number of distinct weight-buffer copies for the "cold" (>L3) measurement.</summary>
    private const int ColdBufferCount = 32;

    [Theory]
    [InlineData(2560, 2560, "attn_qproj")]
    [InlineData(6912, 2560, "ffn_gate")]
    [InlineData(2560, 6912, "ffn_down")]
    public void ProfileGemvI2SDecode_BandwidthVsCompute(int m, int k, string label)
    {
        var rng = new Random(42);
        int rowBytes = k / 4;
        long weightBytes = (long)m * rowBytes;
        const float scale = 0.02f;

        // Hot buffer: single allocation, reused every iteration (matches #131's methodology and a
        // resident weight tensor reused across consecutive decode steps).
        byte* hotWeights = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);

        // Cold buffers: enough distinct copies to exceed typical L3, round-robined across
        // iterations so each touch is a fresh cache line from DRAM.
        byte*[] coldWeights = new byte*[ColdBufferCount];

        float* x = (float*)NativeMemory.AllocZeroed((nuint)(k * sizeof(float)));
        float* result = (float*)NativeMemory.AllocZeroed((nuint)(m * sizeof(float)));

        try
        {
            byte[] randRow = new byte[weightBytes];
            rng.NextBytes(randRow);
            Marshal.Copy(randRow, 0, (nint)hotWeights, (int)weightBytes);

            for (int c = 0; c < ColdBufferCount; c++)
            {
                coldWeights[c] = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);
                rng.NextBytes(randRow);
                Marshal.Copy(randRow, 0, (nint)coldWeights[c], (int)weightBytes);
            }

            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            const int iters = 20;

            // Warm up (JIT).
            MatMul.GemvI2_S(hotWeights, x, result, m, k, scale, null);
            MatMul.BenchUnpackRowI8Only(hotWeights, m, k);
            MatMul.BenchStreamingReadOnly(hotWeights, m, k);

            double hotStreamMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.BenchStreamingReadOnly(hotWeights, m, k);
            }, iters);

            double coldStreamMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.BenchStreamingReadOnly(coldWeights[i % ColdBufferCount], m, k);
            }, iters);

            double unpackMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.BenchUnpackRowI8Only(hotWeights, m, k);
            }, iters);

            double decodeMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.GemvI2_S(hotWeights, x, result, m, k, scale, null);
            }, iters);

            double hotStreamGBs = GBs(weightBytes, hotStreamMs);
            double coldStreamGBs = GBs(weightBytes, coldStreamMs);
            double decodeGBs = GBs(weightBytes, decodeMs);

            _output.WriteLine($"[{label}] m={m} k={k} weightBytes={weightBytes} " +
                               $"AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
                               $"AvxVnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported}");
            _output.WriteLine($"  hot streaming-only:  {hotStreamMs:F4} ms/call   {hotStreamGBs:F2} GB/s   (cache-resident ceiling)");
            _output.WriteLine($"  cold streaming-only: {coldStreamMs:F4} ms/call   {coldStreamGBs:F2} GB/s   (DRAM-forced ceiling)");
            _output.WriteLine($"  unpack-only:         {unpackMs:F4} ms/call");
            _output.WriteLine($"  full GemvI2_S decode:{decodeMs:F4} ms/call   {decodeGBs:F2} GB/s");
            _output.WriteLine($"  decode / hot-ceiling ratio:  {decodeGBs / hotStreamGBs:P1}");
            _output.WriteLine($"  decode / cold-ceiling ratio: {decodeGBs / coldStreamGBs:P1}");
        }
        finally
        {
            NativeMemory.Free(hotWeights);
            foreach (byte* p in coldWeights)
                if (p is not null) NativeMemory.Free(p);
            NativeMemory.Free(x);
            NativeMemory.Free(result);
        }
    }

    private static double Time(Action body, int iters)
    {
        var sw = Stopwatch.StartNew();
        body();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static double GBs(long bytes, double ms) => bytes / (ms / 1000.0) / 1e9;
}
```

- [ ] **Step 2: Run it to verify it compiles and executes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~I2SDecodeBandwidthProfileBench" --logger "console;verbosity=detailed"`
Expected: PASS (3 theory cases, one per shape) — this is a profiling harness, not a correctness assertion, so "pass" just means it ran to completion without exceptions. Confirm the printed output shows plausible non-zero GB/s numbers for all three shapes and all four measurements.

- [ ] **Step 3: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Cpu/Kernels/I2SDecodeBandwidthProfileBench.cs
git commit -m "bench(cpu/i2s): profile GEMV decode vs hot/cold streaming bandwidth ceiling"
```

---

### Task 3: Run the profiling test on this box and record results

**Files:**
- Create: `docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-RESULTS.md`

**Interfaces:**
- Consumes: printed `ITestOutputHelper` output from Task 2's test run.
- Produces: the go/no-go verdict that Task 4 acts on.

- [ ] **Step 1: Run the profiling test in Release with output captured**

Run: `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~I2SDecodeBandwidthProfileBench" --logger "console;verbosity=detailed" > /tmp/i2s-bandwidth-results.txt 2>&1`

(Use the project's scratchpad path instead of `/tmp` if running on Windows PowerShell — redirect to a file under the session scratchpad directory.)

Release build matters here — Debug JIT will understate the SIMD paths.

- [ ] **Step 2: Extract the three shapes' numbers into a results table**

Write `docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-RESULTS.md`:

```markdown
# I2_S Decode Bandwidth Profiling — Results

Ran on: Strix Halo (Ryzen AI Max 395, Zen5, AVX2+AVX-512+VNNI), .NET 10, Release build,
`dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~I2SDecodeBandwidthProfileBench"`.

| Shape | Hot ceiling (GB/s) | Cold/DRAM ceiling (GB/s) | Full decode (GB/s) | decode/hot | decode/cold |
|---|---|---|---|---|---|
| attn_qproj (2560×2560) | <fill from run> | <fill from run> | <fill from run> | <fill from run> | <fill from run> |
| ffn_gate (6912×2560) | <fill from run> | <fill from run> | <fill from run> | <fill from run> | <fill from run> |
| ffn_down (2560×6912) | <fill from run> | <fill from run> | <fill from run> | <fill from run> | <fill from run> |

## Verdict

<Write one paragraph: is decode close to either ceiling (bandwidth-bound) or well under both
(compute-bound, headroom for the LUT kernel)? State which follow-up issue to file per the design
doc's decision rule.>
```

Fill in the actual numbers from the Task 3 Step 1 output — no placeholders left in the committed file.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-RESULTS.md
git commit -m "docs: record I2_S decode bandwidth profiling results and verdict"
```

---

### Task 4: Close the loop upstream and file the follow-up issue (requires user go-ahead)

This task performs actions visible to others (posting to a GitHub issue not owned by this repo,
filing a new issue) — confirm with the user before running these steps; do not execute
automatically even if Tasks 1-3 were run non-interactively.

**Files:** none (GitHub actions only).

- [ ] **Step 1: Post the results table to upstream issue #334**

`gh issue comment 334 --repo kkokosa/dotLLM --body "<results table + one-paragraph verdict from Task 3>"`

- [ ] **Step 2: File the follow-up issue on the fork per the verdict**

If bandwidth-bound → file "Packing density improvements for I2_S BitNet weights" (fallback per
shifulegend's advice in #334).
If headroom exists → file "AVX-512 activation-LUT dot kernel for I2_S GEMV decode", referencing
shifulegend's N/K-groups-axis = our M/output-rows-axis mapping from their 2026-07-18 comment, and
noting the `dev-dotnet11` preview-track question is moot here since `Avx512BW`/`Avx512F` are
already mainstream-shipping intrinsics.

`gh issue create --repo jamesburton/dotLLM --title "<title per verdict>" --body "<scoped acceptance criteria, referencing this profiling issue/PR and the RESULTS.md table>"`

- [ ] **Step 3: Open the PR for this profiling work**

`gh pr create --repo jamesburton/dotLLM --title "bench(cpu/i2s): profile GEMV decode bandwidth vs compute" --body "Closes #{N} (Task 0). Results: docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-RESULTS.md"`

---

## Self-Review Notes

- **Spec coverage:** the design's three probes (streaming ceiling, full decode, ratio) are covered by Task 2; the design's "commit as reusable benchmark" intent is satisfied by the profiling test being a checked-in file (deviated to the xUnit/Stopwatch style already used for this exact scenario in `I2SUnpackProfileBench.cs`, per "follow existing patterns"); the design's "post to #334 + file follow-up" output step is Task 4.
- **Placeholder scan:** the only bracketed placeholders left are inside the RESULTS.md template in Task 3 Step 2, which Task 3 Step 2 itself instructs to fill in before committing — no placeholders remain in any task's *committed* deliverable.
- **Type consistency:** `MatMul.BenchStreamingReadOnly(byte*, int, int)` (Task 1) is the exact signature Task 2's test calls; `MatMul.GemvI2_S(byte*, float*, float*, int, int, float, ComputeThreadPool?)` is the existing explicit-scale overload (confirmed in `src/DotLLM.Cpu/Kernels/MatMul.I2S.cs:128`), used instead of the tail-scale-reading overload so the synthetic buffer doesn't need a trailing scale float appended.
