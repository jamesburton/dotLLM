using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.Configuration;

/// <summary>
/// Issue #1068: the admission arithmetic every context-sized CUDA allocation goes through. A growth that does not leave the
/// reserve free must be refused, because WDDM would otherwise back it with shared system memory and the request slows from
/// minutes to hours. Pure, so it runs without a GPU.
/// </summary>
public sealed class DeviceMemoryBudgetTests
{
    private const long MiB = 1L << 20;

    [Theory]
    [InlineData(1000, 400, 500, true)]    // 1000 - 500 = 500 >= 400
    [InlineData(1000, 500, 500, true)]    // exactly fits
    [InlineData(1000, 501, 500, false)]   // one byte over
    [InlineData(400, 1, 500, false)]      // free below the reserve: nothing fits
    [InlineData(0, 0, 500, true)]         // a zero-byte growth is always admitted
    public void Fits_LeavesTheReserveFree(long free, long required, long reserve, bool expected)
        => Assert.Equal(expected, DeviceMemoryBudget.Fits(free * MiB, required * MiB, reserve * MiB));

    [Theory]
    [InlineData(null, 256)]
    [InlineData("", 256)]
    [InlineData("abc", 256)]
    [InlineData("-1", 256)]
    [InlineData("0", 0)]
    [InlineData("1024", 1024)]
    public void ResolveReserveBytes_ParsesMiB_AndFallsBackToTheDefault(string? env, long expectedMiB)
        => Assert.Equal(expectedMiB * MiB, DeviceMemoryBudget.ResolveReserveBytes(env));

    [Fact]
    public void MaxFittingContext_DividesTheBudget_RoundsDownToTheGranule_AndCapsAtTheModel()
    {
        // 3 GiB free, 512 MiB reserve, 500 MiB fixed, 34 KiB/token (Bonsai 2 q8_0 KV): (3072-512-500) MiB / 34 KiB = 62042 -> 61440.
        int ctx = DeviceMemoryBudget.MaxFittingContext(3072 * MiB, 512 * MiB, 500 * MiB, 34 * 1024, modelMaxContext: 262144);
        Assert.Equal(61440, ctx);
        Assert.Equal(0, ctx % DeviceMemoryBudget.ContextGranularity);

        Assert.Equal(4096, DeviceMemoryBudget.MaxFittingContext(64L << 30, 0, 0, 1024, modelMaxContext: 4096));
    }

    [Fact]
    public void MaxFittingContext_IsZero_WhenNothingFits()
    {
        Assert.Equal(0, DeviceMemoryBudget.MaxFittingContext(400 * MiB, 512 * MiB, 0, 1024, 262144));
        Assert.Equal(0, DeviceMemoryBudget.MaxFittingContext(600 * MiB, 512 * MiB, 0, 1 * MiB, 262144));   // 88 tokens < one granule
    }

    [Fact]
    public void DescribeRefusal_NamesTheAllocation_TheNumbers_AndTheRemedies()
    {
        string m = DeviceMemoryBudget.DescribeRefusal("KV cache (q8_0) for 98304 tokens", 3300 * MiB, 2000 * MiB, 512 * MiB);
        Assert.Contains("KV cache (q8_0) for 98304 tokens", m);
        Assert.Contains("3300 MiB", m);
        Assert.Contains("2000 MiB", m);
        Assert.Contains("512 MiB", m);
        Assert.Contains("--ctx-size", m);
        Assert.Contains(DeviceMemoryBudget.ReserveEnvVar, m);
    }
}
