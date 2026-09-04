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
/// <para>
/// <b>Historical note (#407).</b> Those Q5_0 down banks no longer fall back at all: Q5_0 and
/// IQ4_NL gained routed (<c>moe_indexed</c>) Vulkan kernels and resolver wiring, and 1408 is
/// 32-aligned, so the real DeepSeek-V2-Lite Q4_K_M plan is now fully resident. The plans below
/// are constructed by hand rather than resolved from a GGUF precisely so this file keeps
/// testing the accounting / refusal MESSAGE independently of which types happen to have kernels
/// on any given day — the scenario is still reachable for any type without one (#344 Units 2-4).
/// </para>
/// </remarks>
public sealed class MoeF32HostDequantPreflightTests
{
    // One routed expert bank, F32: numExperts * sizeof(float) * moeIntermediate * hidden.
    private const long OneBankF32Bytes = 64L * 4 * 1408 * 2048;

    // #327 made the skip PER BANK, so only the 14 offending ffn_down_exps banks are dequantised
    // — their Q4_K gate/up siblings stay device-resident and cost nothing. Before #327 this was
    // all 78 banks (~57 GiB); charging that now would over-report by 5.6x and let
    // ThrowIfMoeF32HostDequantUnaffordable refuse a load the per-bank path made fit.
    private const long DeepSeekV2LiteF32Bytes = 14 * OneBankF32Bytes; // ~9.6 GiB

    private static VulkanWeights.MoeF32HostDequantPlan DeepSeekV2LiteQ4KMPlan()
    {
        // The Q4_K_M build's actual census: gate/up are Q4_K everywhere (device-resident-capable),
        // but 14 of the 26 ffn_down_exps banks are Q5_0. Those 14 are what falls back.
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

        // An 8 GiB box: the ~9.6 GiB per-bank fallback cannot fit in RAM there at all, so the
        // preflight must refuse rather than let the load reach OutOfMemoryException.
        var ex = Assert.Throws<InsufficientMemoryException>(
            () => VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, 8L * 1024 * 1024 * 1024));

        Assert.Contains("9.6 GiB of host F32", ex.Message, StringComparison.Ordinal);
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
    public void FallbackThatFitsInPhysicalMemory_DoesNotThrow()
    {
        // A model whose fallback fits in RAM must keep loading exactly as before — the preflight
        // replaces an OOM, it must not manufacture one. This includes the case where the machine
        // is currently busy: the bound is TOTAL physical memory, not free memory, because
        // Windows backs commit with the pagefile and a load over the free-memory line can still
        // complete. See ThrowIfMoeF32HostDequantUnaffordable's remarks.
        var plan = DeepSeekV2LiteQ4KMPlan();
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, DeepSeekV2LiteF32Bytes);
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, DeepSeekV2LiteF32Bytes + 1);
    }

    /// <summary>
    /// The concrete consequence of #327's per-bank accounting, and the regression this guards:
    /// on the 31.6 GiB Strix Halo box that triaged #326, DeepSeek-V2-Lite Q4_K_M's fallback now
    /// FITS (~9.6 GiB for the 14 offending down banks). Under the pre-#327 all-or-nothing
    /// figure (~57 GiB for all 78 banks) this same call threw, refusing a load that is in fact
    /// affordable. If the planner ever reverts to charging resident siblings, this fails.
    /// </summary>
    [Fact]
    public void PerBankFootprint_FitsOnTheBoxWhereTheAllOrNothingFigureDidNot()
    {
        var plan = DeepSeekV2LiteQ4KMPlan();

        // 33.9e9 bytes is what GC.GetGCMemoryInfo().TotalAvailableMemoryBytes actually reports
        // on that box (~31.6 GiB — the rest is the BIOS UMA carve-out).
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, 33_900_000_000L);

        // ...and the pre-#327 figure genuinely did not fit, so the two answers really do differ
        // in outcome, not merely in magnitude.
        var allOrNothing = new VulkanWeights.MoeF32HostDequantPlan(
            CanSkip: false, 78 * OneBankF32Bytes, plan.Fallbacks, TotalBanks: 78);
        Assert.Throws<InsufficientMemoryException>(
            () => VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(allOrNothing, 33_900_000_000L));
    }

    [Fact]
    public void SkippablePlan_NeverThrows()
    {
        var plan = new VulkanWeights.MoeF32HostDequantPlan(
            CanSkip: true, HostF32Bytes: 0, [], TotalBanks: 78);
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, physicalMemoryBytes: 1);
    }

    [Fact]
    public void UnknownMemoryBudget_NeverThrows()
    {
        // HostPhysicalMemoryBytes returns 0 when the runtime cannot report the machine's memory.
        // That must degrade to the old behaviour (attempt the load), never to a refusal on a
        // host that would have coped.
        var plan = DeepSeekV2LiteQ4KMPlan();
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, physicalMemoryBytes: 0);
        VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable(plan, physicalMemoryBytes: -1);
    }

    [Fact]
    public void HostPhysicalMemoryBytes_ReportsAPlausibleAmount()
    {
        // If this ever returned 0 on a normal host the preflight would be silently inert, which
        // is exactly the failure mode #326 is about — assert it reports something real.
        long physical = VulkanWeights.HostPhysicalMemoryBytes();
        Assert.True(physical >= 1L << 30, $"implausible physical memory report: {physical} bytes");
        Assert.Equal(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, physical);
    }
}
