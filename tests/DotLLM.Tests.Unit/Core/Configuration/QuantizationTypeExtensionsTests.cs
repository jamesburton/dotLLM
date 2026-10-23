using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.Configuration;

public sealed class QuantizationTypeExtensionsTests
{
    [Theory]
    [InlineData(QuantizationType.Q2_K, 256, 84)]
    [InlineData(QuantizationType.IQ4_NL, 32, 18)]
    [InlineData(QuantizationType.IQ4_XS, 256, 136)]
    public void ComputeByteCount_Phase2QuantTypes_MatchGgufBlockSizes(
        QuantizationType quantType,
        long elementCount,
        long expectedBytes)
    {
        Assert.Equal(expectedBytes, quantType.ComputeByteCount(elementCount));
        Assert.Equal(expectedBytes * 4, quantType.ComputeByteCount(elementCount * 4));
    }
}
