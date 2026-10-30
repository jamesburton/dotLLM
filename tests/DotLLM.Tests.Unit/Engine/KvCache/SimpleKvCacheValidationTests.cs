using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// <see cref="SimpleKvCache"/> rejects non-positive dimensions and overflowing byte sizes before allocating
/// (#1273, upstream kkokosa/dotLLM#444).
/// </summary>
public sealed class SimpleKvCacheValidationTests
{
    [Theory]
    [InlineData(0, 2, 32, 16, "numLayers")]
    [InlineData(-1, 2, 32, 16, "numLayers")]
    [InlineData(1, 0, 32, 16, "numKvHeads")]
    [InlineData(1, -1, 32, 16, "numKvHeads")]
    [InlineData(1, 2, 0, 16, "headDim")]
    [InlineData(1, 2, -1, 16, "headDim")]
    [InlineData(1, 2, 32, 0, "maxSeqLen")]
    [InlineData(1, 2, 32, -1, "maxSeqLen")]
    public void Constructor_RejectsNonPositiveDimensions(
        int layers, int heads, int dimension, int length, string parameter)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(
            () => new SimpleKvCache(layers, heads, dimension, length).Dispose());
        Assert.Equal(parameter, error.ParamName);
    }

    [Fact]
    public void Constructor_RejectsOverflowingStrideBeforeAllocation()
    {
        Assert.Throws<OverflowException>(() => new SimpleKvCache(1, int.MaxValue, 2, 1).Dispose());
    }

    [Fact]
    public void Constructor_RejectsOverflowingLayerBytesBeforeAllocation()
    {
        Assert.Throws<OverflowException>(
            () => new SimpleKvCache(1, 1, int.MaxValue, int.MaxValue).Dispose());
    }

    [Fact]
    public void Constructor_RejectsOverflowingLaterLayerBeforeAllocation()
    {
        // Only the second layer overflows: validation must cover every layer up front,
        // not fail midway through the allocation loop.
        Assert.Throws<OverflowException>(
            () => new SimpleKvCache(KvGeometry.PerLayer([1, int.MaxValue]), int.MaxValue).Dispose());
    }
}
