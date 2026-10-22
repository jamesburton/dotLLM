using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The hybrid (Qwen3.5 / Qwen3-MoE-hybrid) weight loader widened any quantization missing from its keep-packed list to F32 at load.
/// Q2_K, Q3_K, IQ4_XS, IQ1_S and IQ4_NL were missing although the models' dispatch tables serve them packed (issue #627): a
/// requantised Tev1-4B IQ4_XS streamed ~17 GB per decoded token (6 tok/s) and needed ~7x the device memory. This pins that the
/// upload size of each such type is its packed size, and that an unsupported type still takes the F32 fallback.
/// </summary>
public sealed class VulkanHybridKeepPackedQuantTests
{
    private const int M = 512, K = 2560;

    [Theory]
    [InlineData(QuantizationType.Q2_K)]
    [InlineData(QuantizationType.Q3_K)]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.IQ1_S)]
    [InlineData(QuantizationType.IQ2_XXS)]
    [InlineData(QuantizationType.IQ2_XS)]
    [InlineData(QuantizationType.IQ2_S)]
    [InlineData(QuantizationType.IQ3_XXS)]
    [InlineData(QuantizationType.IQ3_S)]
    [InlineData(QuantizationType.IQ4_XS)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void SupportedQuantizations_AreUploadedPacked(QuantizationType qt)
    {
        long packed = Dequantize.RowByteSize(K, qt) * M;
        Assert.Equal(packed, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(M, K, qt));
        Assert.True(packed < (long)M * K * sizeof(float) / 3, $"{qt} should be well under a third of the F32 size");
    }

    [Theory]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q5_1)]
    public void UnsupportedQuantizations_StillFallBackToF32(QuantizationType qt)
        => Assert.Equal((long)M * K * sizeof(float), VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(M, K, qt));

    [Fact]
    public void NonMultipleOfTheBlockSize_FallsBackToF32()
        => Assert.Equal((long)M * 2000 * sizeof(float),
            VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(M, 2000, QuantizationType.IQ4_XS));   // 2000 % 256 != 0
}
