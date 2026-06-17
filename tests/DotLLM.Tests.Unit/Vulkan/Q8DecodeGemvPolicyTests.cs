using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Unit tests for <see cref="Q8DecodeGemvPolicy"/> — the per-shape DP4a-vs-
/// workgroup decision for the Q8_0 decode GEMV. Pure logic; no GPU required.
/// </summary>
public class Q8DecodeGemvPolicyTests
{
    private const uint VendorIntel = 0x8086;
    private const uint VendorNvidia = 0x10DE;
    private const uint VendorAmd = 0x1002;
    private const string MinKEnv = "DOTLLM_VULKAN_DP4A_MIN_K";

    [Fact]
    public void Intel_PrefersDp4a_AtEveryMeasuredK()
    {
        ClearEnv();
        // Arc: DP4a wins every measured shape (K 192..8192) — threshold 0.
        Assert.Equal(0, Q8DecodeGemvPolicy.Dp4aMinK(VendorIntel));
        foreach (int k in new[] { 192, 576, 1536, 2048, 8192 })
            Assert.True(Q8DecodeGemvPolicy.UseDp4a(VendorIntel, k), $"K={k}");
    }

    [Theory]
    // NVIDIA crossover is K=2048: wg64 wins the short SmolLM K's, DP4a wins K>=2048.
    [InlineData(576, false)]
    [InlineData(1536, false)]
    [InlineData(2048, true)]
    [InlineData(8192, true)]
    public void Nvidia_UsesDp4a_OnlyForLongContractions(int k, bool expectDp4a)
    {
        ClearEnv();
        Assert.Equal(2048, Q8DecodeGemvPolicy.Dp4aMinK(VendorNvidia));
        Assert.Equal(expectDp4a, Q8DecodeGemvPolicy.UseDp4a(VendorNvidia, k));
    }

    [Fact]
    public void UnknownVendor_DefaultsToNoVeto()
    {
        ClearEnv();
        // Threshold 0 → preserves prior "DP4a wherever it was chosen" behaviour.
        Assert.Equal(0, Q8DecodeGemvPolicy.Dp4aMinK(VendorAmd));
        Assert.True(Q8DecodeGemvPolicy.UseDp4a(VendorAmd, 576));
    }

    [Fact]
    public void EnvOverride_ForcesThreshold_AllVendors()
    {
        try
        {
            Environment.SetEnvironmentVariable(MinKEnv, "4096");
            Assert.Equal(4096, Q8DecodeGemvPolicy.Dp4aMinK(VendorIntel));
            Assert.Equal(4096, Q8DecodeGemvPolicy.Dp4aMinK(VendorNvidia));
            Assert.False(Q8DecodeGemvPolicy.UseDp4a(VendorNvidia, 2048));
            Assert.True(Q8DecodeGemvPolicy.UseDp4a(VendorNvidia, 4096));
        }
        finally
        {
            ClearEnv();
        }
    }

    [Fact]
    public void EnvOverride_IgnoresInvalidValues()
    {
        try
        {
            Environment.SetEnvironmentVariable(MinKEnv, "not-a-number");
            // Falls back to the per-vendor default.
            Assert.Equal(2048, Q8DecodeGemvPolicy.Dp4aMinK(VendorNvidia));
        }
        finally
        {
            ClearEnv();
        }
    }

    private static void ClearEnv() => Environment.SetEnvironmentVariable(MinKEnv, null);
}
