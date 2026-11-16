using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// GPU-free logic behind CUDA dense-KV prefix reuse (issue #1594): the peak-bytes arithmetic a growth is admitted with, and which
/// cuts a mixed-precision-window quantized cache can take without losing rows.
/// </summary>
[Collection(CudaCollection.Name)] // Admit_* sets the process-wide admission override the GPU reuse tests also use
public sealed class CudaKvGrowthLogicTests
{
    [Fact]
    public void PeakGrowthBytes_IsAddedRowsPlusLargestOldLayerInTransit()
    {
        // Two layers of different row widths (K+V bytes per position): 100 and 300.
        long[] rows = [100, 300];
        // 1000 -> 1500 positions: added = 500 * (100 + 300) = 200,000; largest old layer = 1000 * 300 = 300,000.
        Assert.Equal(500_000, CudaKvGrowth.PeakGrowthBytes(rows, 1000, 1500));
        Assert.Equal(500_000 + 64, CudaKvGrowth.PeakGrowthBytes(rows, 1000, 1500, fixedExtraBytes: 64));
        Assert.Equal(0, CudaKvGrowth.PeakGrowthBytes(rows, 1500, 1500));
        Assert.Equal(0, CudaKvGrowth.PeakGrowthBytes(rows, 1500, 1000));
    }

    [Fact]
    public void PeakGrowthBytes_TwilEightBGeometry_F16()
    {
        // Llama-3-8B-shaped (TwIL-8B): 32 layers, 8 KV heads x 128, FP16 K and V = 4 KiB per layer-position.
        long[] rows = Enumerable.Repeat(2L * 8 * 128 * 2, 32).ToArray();
        // 3,000 -> 3,600 positions: 600 x 128 KiB added + one old layer (3,000 x 4 KiB).
        Assert.Equal(600L * 32 * 4096 + 3000L * 4096, CudaKvGrowth.PeakGrowthBytes(rows, 3000, 3600));
    }

    [Theory]
    // Pure quantized (window 0): any cut within the live rows.
    [InlineData(0, 100, 100, 0, true)]
    [InlineData(0, 100, 100, 37, true)]
    [InlineData(0, 100, 100, 100, true)]
    [InlineData(0, 100, 100, 101, false)]
    // Window 32, 100 rows, 68 already evicted: no cut at all (a cut to 90 then a short append lowers the quantized length below 68
    // and reads stale ring slots; the graph path re-evicts row pos - 32 from a slot now holding a cut row). A no-op cut is fine.
    [InlineData(32, 100, 68, 99, false)]
    [InlineData(32, 100, 68, 90, false)]
    [InlineData(32, 100, 68, 68, false)]
    [InlineData(32, 100, 68, 67, false)]
    [InlineData(32, 100, 68, 10, false)]
    [InlineData(32, 100, 68, 100, true)]
    // Window not yet full: nothing evicted, every row is still in its own ring slot, any cut.
    [InlineData(32, 20, 0, 5, true)]
    [InlineData(32, 32, 0, 31, true)]
    public void QuantizedCanTruncate_RefusesCutsOnceAWindowHasEvicted(int window, int current, int quantized, int length, bool expected)
        => Assert.Equal(expected, CudaQuantizedKvCache.CanTruncate(window, current, quantized, length));

    [Fact]
    public void Admit_UsesTheOverride_AndCountsRefusals()
    {
        long refusals = CudaKvGrowth.RefusalCount;
        CudaKvGrowth.AdmissionOverride = bytes => bytes < 1000;
        try
        {
            Assert.True(CudaKvGrowth.Admit(999));
            Assert.False(CudaKvGrowth.Admit(1000));
        }
        finally
        {
            CudaKvGrowth.AdmissionOverride = null;
        }
        Assert.Equal(refusals + 1, CudaKvGrowth.RefusalCount);
    }
}
