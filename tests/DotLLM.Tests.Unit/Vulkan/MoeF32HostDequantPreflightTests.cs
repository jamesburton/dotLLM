using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #326. The routed-MoE host F32 dequant preflight must turn an unactionable
/// <c>OutOfMemoryException</c> (thrown ~112 s into a load, from inside
/// <c>TransformerWeights.SliceExpertsToF32</c>) into an explanation produced before the first
/// allocation: how much host RAM the fallback needs, and which expert banks forced it.
/// </summary>
/// <remarks>
/// These exercise the decision and the message only — no Vulkan device, no GGUF file, no GPU.
/// Device-backed coverage of the underlying <c>CanSkipMoeF32HostDequant</c> predicate already
/// lives in <see cref="VulkanTransformerModelMoeKQuantRoutedForwardTests"/>.
/// <para>
/// The numbers below are the real DeepSeek-V2-Lite ones: 26 MoE layers x 64 experts x 3 banks,
/// hidden 2048, moe-intermediate 1408 (not a multiple of QK_K=256, which is precisely why
/// llama.cpp is forced to store every <c>ffn_down_exps</c> as Q5_0 or IQ4_NL rather than a
/// K-quant).
/// </para>
/// </remarks>
public sealed class MoeF32HostDequantPreflightTests
{
    private const long DeepSeekV2LiteF32Bytes = 26L * 64 * 4 * 3 * 1408 * 2048; // ~57 GiB

    private static VulkanWeights.MoeF32HostDequantPlan DeepSeekV2LiteQ4KMPlan()
    {
        // The Q4_K_M build's actual census: gate/up are Q4_K everywhere (device-resident-capable),
        // but 14 of the 26 ffn_down_exps banks are Q5_0, which is not covered — and one uncovered
        // bank refuses the skip for the whole model, so all 78 banks get F32'd.
        var fallbacks = new List<VulkanWeights.MoeRoutedBankFallback>();
        foreach (int layer in new[] { 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22 })
        {
            fallbacks.Add(new VulkanWeights.MoeRoutedBankFallback(
                layer, "ffn_down_exps.weight", QuantizationType.Q5_0, ContractionDim: 1408));
        }

        return new VulkanWeights.MoeF32HostDequantPlan(
            CanSkip: false, DeepSeekV2LiteF32Bytes, fallbacks, TotalBanks: 78);
    }

    [Fact]
    public void UnaffordableFallback_ThrowsNamingFootprintQuantAndBankCount()
    {
        var plan = DeepSeekV2LiteQ4KMPlan();

        // 128 GiB box with a ~57 GiB CPU reference model already resident: ~64 GiB free is
        // enough for the second copy on paper but not once the model itself is mapped — the
        // case here is the harsher one where the budget is clearly short.
        var ex = Assert.Throws<InsufficientMemoryException>(
            () => VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, 20L * 1024 * 1024 * 1024));

        Assert.Contains("53.6 GiB of host F32", ex.Message, StringComparison.Ordinal);
        Assert.Contains("14 of 78 routed expert banks", ex.Message, StringComparison.Ordinal);
        Assert.Contains("ffn_down_exps.weight", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Q5_0", ex.Message, StringComparison.Ordinal);
        Assert.Contains("K=1408", ex.Message, StringComparison.Ordinal);
        // And a way forward, not just a diagnosis.
        Assert.Contains("#327", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Describe_CollapsesRepeatedOffendersIntoOneLine()
    {
        // 14 identical offenders must not produce 14 lines; a 60-layer model would otherwise
        // bury the useful content.
        string described = DeepSeekV2LiteQ4KMPlan().Describe();
        Assert.Equal(1, described.Split("ffn_down_exps.weight").Length - 1);
        Assert.Contains("14x ffn_down_exps.weight stored as Q5_0", described, StringComparison.Ordinal);
    }

    [Fact]
    public void AffordableFallback_DoesNotThrow()
    {
        // A small MoE model whose fallback genuinely fits must keep loading exactly as before —
        // the preflight replaces an OOM, it must not manufacture one.
        var plan = DeepSeekV2LiteQ4KMPlan();
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, DeepSeekV2LiteF32Bytes + 1);
    }

    [Fact]
    public void SkippablePlan_NeverThrows()
    {
        var plan = new VulkanWeights.MoeF32HostDequantPlan(
            CanSkip: true, HostF32Bytes: 0, [], TotalBanks: 78);
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, availableBytes: 1);
    }

    [Fact]
    public void UnknownMemoryBudget_NeverThrows()
    {
        // AvailableHostMemoryBytes returns 0 when the runtime cannot report the machine's
        // memory. That must degrade to the old behaviour (attempt the load), never to a
        // refusal on a host that would have coped.
        var plan = DeepSeekV2LiteQ4KMPlan();
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, availableBytes: 0);
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, availableBytes: -1);
    }

    [Fact]
    public void AvailableHostMemoryBytes_IsNonNegativeAndBelowTotal()
    {
        long available = VulkanWeights.AvailableHostMemoryBytes();
        Assert.True(available >= 0, $"negative available memory: {available}");
        Assert.True(available <= GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
            "free memory must not exceed the machine's total");
    }
}
