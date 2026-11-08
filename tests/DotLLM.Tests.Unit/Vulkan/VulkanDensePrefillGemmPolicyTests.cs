using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The shared dense-prefill GEMM selection table (#1149): the dense host and the hybrid-dense host pick from the same crossovers, so a
/// change here moves both. Pins the measured crossovers the dense host shipped with (K-quants 128, IQ4_XS 256, Q2_K / Q5_0 any n &gt; 1,
/// IQ4_NL never coopmat) and the fallback order (MMQ when coopmat does not win, coopmat when there is no MMQ, scalar last). Needs no GPU.
/// </summary>
public sealed class VulkanDensePrefillGemmPolicyTests
{
    private static int T => VulkanDensePrefillGemmPolicy.KQuantCoopmatMinTokens;

    [Theory]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    [InlineData(QuantizationType.Q3_K)]
    public void KQuants_CoopmatWinsFromTheCrossover(QuantizationType qt)
    {
        Assert.False(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, 1));
        Assert.False(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, T - 1));
        Assert.True(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, T));
        Assert.True(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, 2048));
    }

    [Fact]
    public void Iq4Xs_CoopmatWinsFrom256()
    {
        int t = VulkanDensePrefillGemmPolicy.Iq4XsCoopmatMinTokens;
        Assert.True(t >= 256);
        Assert.False(VulkanDensePrefillGemmPolicy.CoopmatWins(QuantizationType.IQ4_XS, t - 1));
        Assert.True(VulkanDensePrefillGemmPolicy.CoopmatWins(QuantizationType.IQ4_XS, t));
    }

    [Theory]
    [InlineData(QuantizationType.Q2_K)]
    [InlineData(QuantizationType.Q5_0)]
    public void Q2K_Q5_0_CoopmatWinsAtAnyPrefill(QuantizationType qt)
    {
        Assert.False(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, 1));
        Assert.True(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, 2));
    }

    [Theory]
    [InlineData(QuantizationType.IQ4_NL)]
    [InlineData(QuantizationType.Q8_0)]
    public void Iq4Nl_Q8_0_NeverPreferCoopmatOverMmq(QuantizationType qt)
    {
        Assert.False(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, 16));
        Assert.False(VulkanDensePrefillGemmPolicy.CoopmatWins(qt, 4096));
    }

    [Fact]
    public void Select_PrefersMmqBelowTheCrossover_CoopmatAbove()
    {
        Assert.Equal(DensePrefillGemmPath.Mmq, VulkanDensePrefillGemmPolicy.Select(QuantizationType.Q4_K, T - 1, hasCoopmat: true, hasMmq: true));
        Assert.Equal(DensePrefillGemmPath.Coopmat, VulkanDensePrefillGemmPolicy.Select(QuantizationType.Q4_K, T, hasCoopmat: true, hasMmq: true));
        Assert.Equal(DensePrefillGemmPath.Mmq, VulkanDensePrefillGemmPolicy.Select(QuantizationType.IQ4_NL, 4096, hasCoopmat: false, hasMmq: true));
    }

    [Fact]
    public void Select_FallsBackInOrder()
    {
        // No MMQ (no integer dot): coopmat at any n, even below the crossover.
        Assert.Equal(DensePrefillGemmPath.Coopmat, VulkanDensePrefillGemmPolicy.Select(QuantizationType.Q4_K, 8, hasCoopmat: true, hasMmq: false));
        // No coopmat (Arc): MMQ at any n, even above the crossover.
        Assert.Equal(DensePrefillGemmPath.Mmq, VulkanDensePrefillGemmPolicy.Select(QuantizationType.Q4_K, 2048, hasCoopmat: false, hasMmq: true));
        // Neither: the scalar F32 GEMM.
        Assert.Equal(DensePrefillGemmPath.Scalar, VulkanDensePrefillGemmPolicy.Select(QuantizationType.Q4_K, 512, hasCoopmat: false, hasMmq: false));
    }

    /// <summary>The dense host's own threshold field must be the shared one, or the two hosts drift apart again.</summary>
    [Fact]
    public void DenseHostThreshold_IsTheSharedOne()
        => Assert.Equal(VulkanDensePrefillGemmPolicy.KQuantCoopmatMinTokens, VulkanTransformerModel.Q4KCoopmatPreferSeqLen);
}
