using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// The CPU R4 repack is a second committed copy of the weights; <see cref="RepackBudget"/> decides how much of
/// it a host can afford (#792). The decision is a pure function, so these tests pin it on any machine.
/// </summary>
public sealed unsafe class RepackBudgetTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void AmpleMemory_RepacksEverything()
    {
        // 8B Q4_K_M: ~5 GiB weights + ~4.7 GiB repack on a 64 GiB box
        long allowed = RepackBudget.AllowedRepackBytes(weightBytes: 5 * GiB, candidateBytes: 5 * GiB, availablePhysicalBytes: 64 * GiB);
        Assert.Equal(5 * GiB, allowed);
    }

    [Fact]
    public void Gemma4_31B_On32GiBHost_SkipsRepack()
    {
        // The #792 case: 17.7 GiB mapped + ~17 GiB repack, ~28 GiB available -> 0.8 * 28 = 22.4 < 17.7 + 17.
        long allowed = RepackBudget.AllowedRepackBytes(
            weightBytes: (long)(17.7 * GiB), candidateBytes: 17 * GiB, availablePhysicalBytes: 28 * GiB);
        Assert.True(allowed < 17 * GiB, "the full repack must not be allowed");
        // headroom is 22.4 - 17.7 = 4.7 GiB -> partial, never the full copy
        Assert.InRange(allowed, 4 * GiB, 5 * GiB);
    }

    [Fact]
    public void WeightsAloneExceedBudget_SkipsEntirely()
    {
        long allowed = RepackBudget.AllowedRepackBytes(weightBytes: 20 * GiB, candidateBytes: 10 * GiB, availablePhysicalBytes: 24 * GiB);
        Assert.Equal(0, allowed);   // 0.8 * 24 = 19.2 < 20
    }

    [Fact]
    public void UnknownAvailableMemory_KeepsDefaultBehaviour()
    {
        Assert.Equal(7 * GiB, RepackBudget.AllowedRepackBytes(5 * GiB, 7 * GiB, availablePhysicalBytes: 0));
        Assert.Equal(7 * GiB, RepackBudget.AllowedRepackBytes(5 * GiB, 7 * GiB, availablePhysicalBytes: -1));
    }

    [Theory]
    [InlineData(10, 10, 25, true)]    // 0.8*25 = 20 == 10 + 10 -> fits exactly
    [InlineData(10, 10, 24, false)]   // 0.8*24 = 19.2 < 20 -> partial
    public void Boundary_IsInclusive(long weightGiB, long repackGiB, long availGiB, bool full)
    {
        long allowed = RepackBudget.AllowedRepackBytes(weightGiB * GiB, repackGiB * GiB, availGiB * GiB);
        Assert.Equal(full, allowed == repackGiB * GiB);
    }

    [Fact]
    public void NothingToRepack_IsZero() =>
        Assert.Equal(0, RepackBudget.AllowedRepackBytes(5 * GiB, 0, 64 * GiB));

    [Theory]
    [InlineData(null, "Auto")]
    [InlineData("", "Auto")]
    [InlineData("auto", "Auto")]
    [InlineData("garbage", "Auto")]
    [InlineData("always", "Always")]
    [InlineData("1", "Always")]
    [InlineData("NEVER", "Never")]
    [InlineData("0", "Never")]
    public void ParseMode(string? value, string expected) =>
        Assert.Equal(expected, RepackBudget.ParseMode(value).ToString());

    [Fact]
    public void Resolve_OverridesBeatTheMemoryProbe()
    {
        Assert.Equal(17 * GiB, RepackBudget.Resolve(18 * GiB, 17 * GiB, RepackBudget.RepackMode.Always, 8 * GiB));
        Assert.Equal(0, RepackBudget.Resolve(1 * GiB, 1 * GiB, RepackBudget.RepackMode.Never, 64 * GiB));
        Assert.Equal(0, RepackBudget.Resolve(18 * GiB, 17 * GiB, RepackBudget.RepackMode.Auto, 8 * GiB));
    }

    [Fact]
    public void Describe_IsSilentWhenFullyRepacked_AndNamesTheReasonOtherwise()
    {
        Assert.Null(RepackBudget.Describe(RepackBudget.RepackMode.Auto, 5 * GiB, 5 * GiB, 5 * GiB, 5 * GiB, 32, 32, 64 * GiB));
        string skipped = RepackBudget.Describe(RepackBudget.RepackMode.Auto, 18 * GiB, 17 * GiB, 0, 0, 0, 60, 28 * GiB)!;
        Assert.Contains("skipped", skipped);
        Assert.Contains("DOTLLM_CPU_REPACK=always", skipped);
        Assert.DoesNotContain('\n', skipped);
        string partial = RepackBudget.Describe(RepackBudget.RepackMode.Auto, 18 * GiB, 17 * GiB, 5 * GiB, 4 * GiB, 14, 60, 28 * GiB)!;
        Assert.Contains("14/60", partial);
    }

    [Fact]
    public void QueryAvailablePhysicalBytes_IsPlausibleOnThisHost()
    {
        long avail = RepackBudget.QueryAvailablePhysicalBytes();
        // 0 is "unknown" on platforms without a probe; when known it must be positive and below an absurd bound.
        Assert.InRange(avail, 0, 64L * 1024 * GiB);
    }

    [Theory]
    [InlineData(QuantizationType.Q4_K, 4096, 4096)]
    [InlineData(QuantizationType.Q6_K, 262144, 5376)]
    [InlineData(QuantizationType.Q8_0, 10, 64)]     // M % 4 != 0
    [InlineData(QuantizationType.Q5_0, 8, 64)]
    public void RepackedSize_MatchesWhatRepackR4Allocates(QuantizationType qt, int m, int k)
    {
        long predicted = WeightRepacking.RepackedSize(qt, m, k);
        var (blockBytes, groupSize) = WeightRepacking.GetBlockInfo(qt);
        // Source is never read for the size, but RepackR4 does read it: use a small shape for the real call.
        if ((long)m * (k / groupSize) * blockBytes <= 64L * 1024 * 1024)
        {
            long srcBytes = predicted;
            byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)srcBytes, 64);
            try
            {
                new Span<byte>(src, (int)srcBytes).Clear();
                var rw = WeightRepacking.RepackR4((nint)src, qt, m, k);
                try { Assert.Equal(rw.AllocatedBytes, predicted); }
                finally { rw.Dispose(); }
            }
            finally { NativeMemory.AlignedFree(src); }
        }
        Assert.Equal((long)m * (k / groupSize) * blockBytes, predicted);
    }

    [Fact]
    public void RepackedSize_IsZeroForUnrepackableTypes() =>
        Assert.Equal(0, WeightRepacking.RepackedSize(QuantizationType.F32, 64, 64));
}
