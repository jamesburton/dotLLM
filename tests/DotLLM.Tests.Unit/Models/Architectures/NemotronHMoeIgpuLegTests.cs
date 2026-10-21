using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Cuda;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1083, findings of the first quiet window on the dev box (RTX 3060 + Arc iGPU), pinned in pure CPU code (no GPU is
/// touched, so these run in the CPU-safe suite):
/// <list type="bullet">
///   <item><b>Row capacity.</b> The iGPU service is created for a fixed number of rows per submission. A layer call of n tokens
///   at top-k k carries up to n·k rows, and top-k differs per layer, so the capacity must be the block's tokens × the
///   <b>largest</b> top-k (<see cref="VulkanNemotronHMoeExpertService.MaxRowsFor"/>, which production uses). A fixed 64 failed on
///   17 tokens of the fixture's top-k 4 layer (68 rows).</item>
///   <item><b>Fence before read.</b> <see cref="FencedHostRows"/> (the service's <c>End</c>) must observe the fence before it
///   copies a result byte. The fake device here writes its results <b>only</b> when the host waits on the fence, the latest moment
///   a real device may finish, so a missing or reordered wait reads the stale rows deterministically, with no timing race.</item>
///   <item><b>Import range.</b> The alignment and size arithmetic of the zero-copy host import (<see cref="HostImportRange"/>).</item>
///   <item><b>Staged banks.</b> The load-log line of a refused import names its RAM cost and, for a read-only GGUF, the cause.</item>
/// </list>
/// </summary>
public sealed unsafe class NemotronHMoeIgpuLegTests : IDisposable
{
    private readonly List<nint> _allocs = [];

    public void Dispose()
    {
        foreach (nint p in _allocs) NativeMemory.Free((void*)p);
    }

    // ───────────────────────── row capacity ─────────────────────────

    /// <summary>The synthetic Bolt fixture's MoE layers (PR A), in model order.</summary>
    private static readonly int[] FixtureMoeLayers = [1, 2, 4, 5];

    [Fact]
    public void MaxRowsFor_TheFixtureLayers_Is17TokensTimesTheLargestTopK_Not64()
    {
        int[] topKs = FixtureMoeLayers.Select(CudaNemotronHMoeParityTests.TopKOfLayer).ToArray();
        Assert.Equal([2, 3, 1, 4], topKs);

        int rows = VulkanNemotronHMoeExpertService.MaxRowsFor(17, topKs.Select(Weights));
        Assert.Equal(68, rows);
        Assert.True(rows > 64, "the first quiet window's fixed capacity of 64 is short of this fixture's 17-token block");
    }

    [Theory]
    [InlineData(new[] { 2, 3, 1, 4 })]
    [InlineData(new[] { 4, 1, 3, 2 })]
    [InlineData(new[] { 1, 4, 1 })]
    [InlineData(new[] { 6 })]
    public void MaxRowsFor_IsTheLargestTopK_WhateverTheLayerOrder(int[] topKs)
    {
        foreach (int tokens in new[] { 1, 3, 17, 512 })
            Assert.Equal(tokens * topKs.Max(), VulkanNemotronHMoeExpertService.MaxRowsFor(tokens, topKs.Select(Weights)));
    }

    [Fact]
    public void MaxRowsFor_NoLayers_IsTheTokens_AndRefusesAnEmptyBlock()
    {
        Assert.Equal(5, VulkanNemotronHMoeExpertService.MaxRowsFor(5, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => VulkanNemotronHMoeExpertService.MaxRowsFor(0, [Weights(2)]));
    }

    /// <summary>
    /// The production invariant: every layer call is a block of at most <c>blockTokens</c> tokens, and its iGPU rows (from
    /// <see cref="VulkanNemotronHMoeExpertService.BuildRows"/>) are a subset of its tokens × top-k slots, so a service created with
    /// <see cref="VulkanNemotronHMoeExpertService.MaxRowsFor"/> takes any call, even one that puts every expert of the call on the
    /// iGPU, and the bound is tight for the largest top-k.
    /// </summary>
    [Theory]
    [InlineData(17)]
    [InlineData(4)]
    [InlineData(1)]
    public void BuildRows_OfAFullBlockAllOnTheIgpu_FitsMaxRowsFor_ForEveryFixtureLayer(int blockTokens)
    {
        int[] topKs = FixtureMoeLayers.Select(CudaNemotronHMoeParityTests.TopKOfLayer).ToArray();
        int capacity = VulkanNemotronHMoeExpertService.MaxRowsFor(blockTokens, topKs.Select(Weights));
        var rng = new Random(1083 + blockTokens);
        foreach (int topK in topKs)
        {
            const int Experts = 8;
            int slots = blockTokens * topK;

            // Each token routes to topK distinct experts; the schedule groups the slots by expert (ascending slot ids).
            var byExpert = new List<int>[Experts];
            for (int e = 0; e < Experts; e++) byExpert[e] = [];
            for (int t = 0; t < blockTokens; t++)
            {
                int[] picks = Enumerable.Range(0, Experts).OrderBy(_ => rng.Next()).Take(topK).ToArray();
                for (int j = 0; j < topK; j++) byExpert[picks[j]].Add(t * topK + j);
            }
            int[] groupExperts = Enumerable.Range(0, Experts).Where(e => byExpert[e].Count > 0).ToArray();
            int[] groupStart = new int[groupExperts.Length + 1];
            int[] slotIds = new int[slots];
            for (int g = 0, pos = 0; g < groupExperts.Length; g++)
            {
                groupStart[g] = pos;
                foreach (int s in byExpert[groupExperts[g]].Order()) slotIds[pos++] = s;
                groupStart[g + 1] = pos;
            }

            int[] rowExperts = new int[capacity], rowTokens = new int[capacity];
            int rows;
            fixed (int* gs = groupStart, ids = slotIds)
                rows = VulkanNemotronHMoeExpertService.BuildRows(groupExperts, 0, groupExperts.Length, gs, ids, topK, rowExperts, rowTokens);
            Assert.Equal(slots, rows);
            Assert.InRange(rows, 1, capacity);
            Assert.All(rowTokens[..rows], t => Assert.InRange(t, 0, blockTokens - 1));
            if (topK == topKs.Max()) Assert.Equal(capacity, rows);
        }
    }

    private static NemotronHMoeWeights Weights(int topK) => new()
    {
        GateInpWeight = 0, GateInpQuantType = QuantizationType.F32,
        SelectionBias = new float[8],
        UpExpsWeight = 0, UpExpsQuantType = QuantizationType.Q4_K, UpPerExpertBytes = 0,
        DownExpsWeight = 0, DownExpsQuantType = QuantizationType.Q4_K, DownPerExpertBytes = 0,
        UpShexpWeight = 0, UpShexpQuantType = QuantizationType.F32,
        DownShexpWeight = 0, DownShexpQuantType = QuantizationType.F32,
        NumExperts = 8, NumExpertsPerTok = topK, MoeIntermediateSize = 256, SharedIntermediateSize = 1,
        NormalizeWeights = true, WeightsScale = 1f, LatentSize = 256,
    };

    // ───────────────────────── fence before read ─────────────────────────

    /// <summary>
    /// A stand-in device whose results land in its host-visible output only when the host waits on the fence. Until then the
    /// output holds the previous submission's rows (<see cref="Stale"/>), as a real one does.
    /// </summary>
    private sealed class FakeDevice
    {
        public const float Stale = -7.25f;
        public float* Output;
        public int Floats;
        public float Base;
        public int FenceWaits;
        public Exception? Lost;

        public void WaitFence()
        {
            FenceWaits++;
            if (Lost is not null) throw Lost;
            for (int i = 0; i < Floats; i++) Output[i] = Base + i;
        }
    }

    [Fact]
    public void Collect_WaitsOnTheFenceBeforeReadingTheRows()
    {
        const int Rows = 3, RowFloats = 5;
        var dev = new FakeDevice { Output = FloatsFilled(Rows * RowFloats, FakeDevice.Stale), Floats = Rows * RowFloats, Base = 100 };
        var fenced = new FencedHostRows(dev.WaitFence);

        fenced.Submitted(Rows, RowFloats, Stopwatch.GetTimestamp());
        Assert.True(fenced.InFlight);
        float* outRows = FloatsFilled(Rows * RowFloats, float.NaN);
        Assert.Equal(Rows, fenced.Collect(dev.Output, outRows));

        // A missing wait copies the stale rows; a wait after the copy too. Either reads -7.25 here, not 100 + i.
        Assert.Equal(1, dev.FenceWaits);
        for (int i = 0; i < Rows * RowFloats; i++)
            Assert.True(outRows[i] == 100 + i, $"element {i}: {outRows[i]} (stale {FakeDevice.Stale}: read before the fence)");
        Assert.False(fenced.InFlight);
        Assert.True(fenced.LastComputeUs >= fenced.LastWaitUs && fenced.LastWaitUs >= 0);
    }

    [Fact]
    public void Collect_EachSubmissionReadsItsOwnRows_NotThePreviousOnes()
    {
        const int RowFloats = 4;
        var dev = new FakeDevice { Output = FloatsFilled(8 * RowFloats, FakeDevice.Stale) };
        var fenced = new FencedHostRows(dev.WaitFence);
        float* outRows = FloatsFilled(8 * RowFloats, float.NaN);
        foreach (var (rows, b) in new[] { (2, 10f), (8, 1000f), (1, -50f) })
        {
            dev.Floats = rows * RowFloats;
            dev.Base = b;
            fenced.Submitted(rows, RowFloats, Stopwatch.GetTimestamp());
            Assert.Equal(rows, fenced.Collect(dev.Output, outRows));
            for (int i = 0; i < rows * RowFloats; i++)
                Assert.True(outRows[i] == b + i, $"rows={rows} element {i}: {outRows[i]}, want {b + i}");
        }
        Assert.Equal(3, dev.FenceWaits);
    }

    [Fact]
    public void Collect_AFailedWait_EndsTheSubmission_CopiesNothing_AndRethrows()
    {
        var lost = new InvalidOperationException("device lost");
        var dev = new FakeDevice { Output = FloatsFilled(4, 1f), Floats = 4, Lost = lost };
        var fenced = new FencedHostRows(dev.WaitFence);
        fenced.Submitted(1, 4, Stopwatch.GetTimestamp());
        float* outRows = FloatsFilled(4, float.NaN);
        Assert.Same(lost, Assert.Throws<InvalidOperationException>(() => fenced.Collect(dev.Output, outRows)));
        Assert.False(fenced.InFlight);
        for (int i = 0; i < 4; i++) Assert.True(float.IsNaN(outRows[i]));

        // The next submission is not refused as "in flight".
        dev.Lost = null;
        fenced.Submitted(1, 4, Stopwatch.GetTimestamp());
        Assert.Equal(1, fenced.Collect(dev.Output, outRows));
    }

    [Fact]
    public void Submitted_RefusesASecondSubmission_AndCollect_RefusesWhenNothingIsInFlight()
    {
        var dev = new FakeDevice { Output = FloatsFilled(4, 0f), Floats = 4 };
        var fenced = new FencedHostRows(dev.WaitFence);
        float* outRows = FloatsFilled(4, 0f);
        Assert.Throws<InvalidOperationException>(() => fenced.Collect(dev.Output, outRows));
        fenced.ThrowIfInFlight();
        fenced.Submitted(1, 4, Stopwatch.GetTimestamp());
        Assert.Throws<InvalidOperationException>(() => fenced.Submitted(1, 4, Stopwatch.GetTimestamp()));
        Assert.Throws<InvalidOperationException>(fenced.ThrowIfInFlight);
        Assert.Throws<ArgumentOutOfRangeException>(() => new FencedHostRows(dev.WaitFence).Submitted(0, 4, 0));
        Assert.Equal(0, dev.FenceWaits);
    }

    [Fact]
    public void DrainForDispose_WaitsOnlyWhenInFlight_AndSwallowsOnlyALostDevice()
    {
        var dev = new FakeDevice { Output = FloatsFilled(4, 0f), Floats = 4 };
        var fenced = new FencedHostRows(dev.WaitFence);
        fenced.DrainForDispose<TimeoutException>();
        Assert.Equal(0, dev.FenceWaits);

        fenced.Submitted(1, 4, Stopwatch.GetTimestamp());
        dev.Lost = new TimeoutException("lost");
        fenced.DrainForDispose<TimeoutException>();
        Assert.Equal(1, dev.FenceWaits);
        Assert.False(fenced.InFlight);

        fenced.Submitted(1, 4, Stopwatch.GetTimestamp());
        dev.Lost = new InvalidOperationException("not a lost device");
        Assert.Throws<InvalidOperationException>(fenced.DrainForDispose<TimeoutException>);
    }

    // ───────────────────────── import range ─────────────────────────

    [Theory]
    [InlineData(0x10000UL, 0x2000L, 0x1000UL, 0x10000UL, 0L, 0x2000UL)]           // aligned start and end: the range itself
    [InlineData(0x10123UL, 0x2000L, 0x1000UL, 0x10000UL, 0x123L, 0x3000UL)]       // a GGUF tensor mid-page: down, then up
    [InlineData(0x10F00UL, 0x100L, 0x1000UL, 0x10000UL, 0xF00L, 0x1000UL)]        // ends exactly on the next boundary
    [InlineData(0x10F00UL, 0x101L, 0x1000UL, 0x10000UL, 0xF00L, 0x2000UL)]        // one byte past it: one more granule
    [InlineData(0x123456UL, 1L, 0x10000UL, 0x120000UL, 0x3456L, 0x10000UL)]       // a 64 KiB granule
    [InlineData(0x7FFF_0000_0040UL, 0x40L, 0x40UL, 0x7FFF_0000_0040UL, 0L, 0x40UL)] // a small granule, aligned
    public void ImportRange_RoundsTheStartDownAndTheEndUp(ulong address, long size, ulong alignment, ulong wantBase, long wantBind,
        ulong wantSize)
    {
        Assert.True(HostImportRange.TryCompute(address, size, alignment, out var r));
        Assert.Equal(new HostImportRange(wantBase, wantBind, wantSize), r);
    }

    [Fact]
    public void ImportRange_Invariants_OnRandomRanges()
    {
        var rng = new Random(1083);
        foreach (ulong alignment in new ulong[] { 1, 64, 4096, 65536, 1UL << 21 })
            for (int i = 0; i < 2000; i++)
            {
                ulong address = (ulong)rng.NextInt64(1, 1L << 46);
                long size = rng.NextInt64(1, 1L << 30);
                Assert.True(HostImportRange.TryCompute(address, size, alignment, out var r));
                Assert.Equal(0UL, r.Base % alignment);                                  // pHostPointer: a multiple of the alignment
                Assert.Equal(0UL, r.Size % alignment);                                  // allocationSize: a multiple of it
                Assert.Equal(address, r.Base + (ulong)r.BindOffset);                    // bound at the logical start
                Assert.InRange((ulong)r.BindOffset, 0UL, alignment - 1);
                Assert.True(r.Base + r.Size >= address + (ulong)size);                  // covers the whole logical range
                Assert.True(r.Size - ((ulong)r.BindOffset + (ulong)size) < alignment); // and not a granule more
            }
    }

    [Theory]
    [InlineData(0UL)]
    [InlineData(3000UL)]
    [InlineData(4097UL)]
    public void ImportRange_RefusesAnAlignmentThatIsNotAPowerOfTwo(ulong alignment) =>
        Assert.False(HostImportRange.TryCompute(0x10000, 16, alignment, out _));

    [Fact]
    public void ImportRange_RefusesAnOverflowingRange_AndANonPositiveSize()
    {
        Assert.False(HostImportRange.TryCompute(ulong.MaxValue - 100, 4096, 4096, out _));
        Assert.False(HostImportRange.TryCompute(0x1000, long.MaxValue, 4096, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() => HostImportRange.TryCompute(0x1000, 0, 4096, out _));
    }

    [Fact]
    public void ImportRange_AllocationSize_IsTheRange_OrTheBuffersRequirementRoundedUp()
    {
        var r = new HostImportRange(0x10000, 0x123, 0x3000);
        Assert.Equal(0x3000UL, r.AllocationSize(0x2123, 0x1000));
        Assert.Equal(0x3000UL, r.AllocationSize(0x3000, 0x1000));
        Assert.Equal(0x4000UL, r.AllocationSize(0x3001, 0x1000));
        Assert.Equal(4096UL, HostImportRange.RoundUp(1, 4096));
        Assert.Equal(4096UL, HostImportRange.RoundUp(4096, 4096));
        Assert.Equal(8192UL, HostImportRange.RoundUp(4097, 4096));
    }

    // ───────────────────────── staged banks ─────────────────────────

    [Fact]
    public void DescribeStagedBanks_NamesTheRamCostAndBudget_AndForAReadOnlyGguf_TheCauseAndRemedy()
    {
        const long MiB = 1024 * 1024;
        string ro = VulkanNemotronHMoeExpertService.DescribeStagedBanks(9, "import_rejected(vkAllocateMemory:-2)", 3 * MiB / 2,
            700 * MiB, 9700 * MiB, ggufReadOnly: true);
        Assert.StartsWith("MoE iGPU: blk.9 expert banks: the zero-copy import was refused (import_rejected(vkAllocateMemory:-2)); " +
            "staged 1.5 MiB into device memory instead, a second copy in RAM on a UMA device (700 of 9700 MiB staging budget used; " +
            "past it, layers stay on the CPU).", ro, StringComparison.Ordinal);
        Assert.Contains("mapped read-only", ro, StringComparison.Ordinal);
        Assert.Contains("DOTLLM_GGUF_MAP_COW=1", ro, StringComparison.Ordinal);

        string cow = VulkanNemotronHMoeExpertService.DescribeStagedBanks(9, "import_rejected(vkAllocateMemory:-2)", MiB, MiB, 10 * MiB,
            ggufReadOnly: false);
        Assert.DoesNotContain("DOTLLM_GGUF_MAP_COW", cow, StringComparison.Ordinal);
        Assert.Contains("a second copy in RAM", cow, StringComparison.Ordinal);
    }

    private float* FloatsFilled(int count, float value)
    {
        float* p = (float*)NativeMemory.Alloc((nuint)Math.Max(count, 1), sizeof(float));
        _allocs.Add((nint)p);
        new Span<float>(p, count).Fill(value);
        return p;
    }
}
