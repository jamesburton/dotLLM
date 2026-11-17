using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.Configuration;

/// <summary>
/// Issue #1851: <see cref="LayerSplitPlanner"/> picks the largest dGPU layer prefix that fits, then the iGPU, then the CPU.
/// The fixture is chosen so that every term moves the split point on its own: layer bytes are non-uniform and the budget sits
/// strictly between two prefix boundaries (600 MiB used at N=4, 700 MiB at N=5, 650 MiB usable). Dropping any one of the
/// reserve, the KV, the activation scratch or the fixed bytes lets a fifth layer in.
/// </summary>
public sealed class LayerSplitPlannerTests
{
    private const long M = 1L << 20;

    // Per layer: weights + 10 MiB KV (1 KiB/token/layer x 10,240 tokens). Prefix used (with the 110 MiB once-cost):
    // N=1 220, N=2 350, N=3 440, N=4 600, N=5 700, N=6 820, N=7 960.
    private static readonly long[] Layers = [100 * M, 120 * M, 80 * M, 150 * M, 90 * M, 110 * M, 130 * M, 70 * M];

    private static LayerSplitInput Input(long freeMib = 750, long reserveMib = 100, long fixedMib = 50, long kvPerTokenPerLayer = 1024,
        int ctx = 10240, long activationPerToken = 2048, int activationTokens = 30720, long headMib = 40, IgpuTier? igpu = null)
        => new(freeMib * M, reserveMib * M, fixedMib * M, Layers, kvPerTokenPerLayer, ctx, activationPerToken, activationTokens,
            headMib * M, igpu);

    [Fact]
    public void NonDegenerateSplit_PicksLargestFittingPrefix()
    {
        var plan = LayerSplitPlanner.Plan(Input());
        Assert.Equal((4, 0, 4), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
        Assert.Equal("4 cuda + 4 cpu layers", plan.Describe());
        Assert.Contains("no iGPU", plan.Reason);
    }

    [Fact]
    public void ExactBoundary_IsInclusive()
    {
        // 600 MiB used at N=4 == 600 MiB usable.
        var plan = LayerSplitPlanner.Plan(Input(freeMib: 700));
        Assert.Equal(4, plan.DgpuLayers);
        // One byte less and the fourth layer no longer fits.
        var tight = LayerSplitPlanner.Plan(Input(freeMib: 700) with { DgpuFreeBytes = 700 * M - 1 });
        Assert.Equal(3, tight.DgpuLayers);
    }

    [Theory]
    [InlineData("reserve")]
    [InlineData("kv")]
    [InlineData("activation")]
    [InlineData("fixed")]
    public void EveryCostTerm_MovesTheSplit(string term)
    {
        // Each term alone is what keeps layer 5 off the dGPU: without it, N becomes 5.
        var input = term switch
        {
            "reserve" => Input(reserveMib: 0),
            "kv" => Input(kvPerTokenPerLayer: 0),
            "activation" => Input(activationPerToken: 0),
            _ => Input(fixedMib: 0),
        };
        Assert.Equal(5, LayerSplitPlanner.Plan(input).DgpuLayers);
        Assert.Equal(4, LayerSplitPlanner.Plan(Input()).DgpuLayers);
    }

    [Fact]
    public void KvScalesWithContext()
    {
        // Half the context: 110 + 105 + 125 + 85 + 155 = 580, + 95 = 675 > 650, so still 4. Ten times the context (100 MiB of KV
        // per layer): 110 + 200 + 220 = 530, + 180 = 710 > 650, so 2.
        Assert.Equal(4, LayerSplitPlanner.Plan(Input(ctx: 5120)).DgpuLayers);
        Assert.Equal(2, LayerSplitPlanner.Plan(Input(ctx: 102400)).DgpuLayers);
    }

    [Fact]
    public void Head_IsChargedOnlyWhenEveryLayerIsOnTheDgpu()
    {
        // All 8 layers without the head need 1040 MiB; with a 200 MiB head, 1240. 1100 usable: the head keeps one layer off.
        var plan = LayerSplitPlanner.Plan(Input(freeMib: 1200, headMib: 200));
        Assert.Equal((7, 1), (plan.DgpuLayers, plan.CpuLayers));

        // In a split the head lives on the tail, so it must not shrink the prefix: 7 layers use 960 of 1000 usable MiB.
        var split = LayerSplitPlanner.Plan(Input(freeMib: 1100, headMib: 200));
        Assert.Equal(7, split.DgpuLayers);
    }

    [Fact]
    public void EverythingFits_AllOnDgpu()
    {
        var plan = LayerSplitPlanner.Plan(Input(freeMib: 1180));   // 1080 needed, 1080 usable
        Assert.True(plan.AllOnDgpu);
        Assert.Equal((8, 0, 0), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
        Assert.Contains("every layer fits", plan.Reason);
        Assert.Equal(7, LayerSplitPlanner.Plan(Input(freeMib: 1179)).DgpuLayers);
    }

    [Fact]
    public void NothingFits_ZeroDgpuLayers()
    {
        var plan = LayerSplitPlanner.Plan(Input(freeMib: 300));   // 200 usable < 220 for the first layer
        Assert.Equal((0, 0, 8), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
        Assert.Contains("not even one layer", plan.Reason);
    }

    [Fact]
    public void Igpu_PresentAndFitting_TakesTheTail_AfterTheDgpu()
    {
        // Tail = head 40 + layers 4..7 (400) + 4 x 10 KV = 480 MiB; 500 usable on the iGPU.
        var igpu = new IgpuTier("vulkan:1", "Arc", 600 * M, 100 * M, Supported: true);
        var plan = LayerSplitPlanner.Plan(Input(igpu: igpu));
        Assert.Equal((4, 4, 0), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
        Assert.Equal("vulkan:1", plan.IgpuDevice);
        Assert.Equal("4 cuda + 4 vulkan(iGPU vulkan:1) layers", plan.Describe());
    }

    [Fact]
    public void Igpu_LargeEnoughForEverything_StillComesAfterTheDgpu()
    {
        // Tier order: the dGPU is filled first even when the iGPU could hold the whole model.
        var igpu = new IgpuTier("vulkan:1", "Arc", 64_000 * M, 100 * M, Supported: true);
        var plan = LayerSplitPlanner.Plan(Input(igpu: igpu));
        Assert.Equal((4, 4, 0), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
    }

    [Theory]
    [InlineData(580, true)]    // 480 usable == 480 needed
    [InlineData(579, false)]   // one MiB short
    [InlineData(560, false)]   // a tail without its KV (440) or without the head (440) would wrongly fit here
    public void Igpu_FitCountsLayersKvAndHead(long igpuFreeMib, bool fits)
    {
        var igpu = new IgpuTier("vulkan:1", "Arc", igpuFreeMib * M, 100 * M, Supported: true);
        var plan = LayerSplitPlanner.Plan(Input(igpu: igpu));
        Assert.Equal(4, plan.DgpuLayers);
        Assert.Equal(fits ? (4, 0) : (0, 4), (plan.IgpuLayers, plan.CpuLayers));
        if (!fits) Assert.Contains("cannot hold", plan.Reason);
    }

    [Fact]
    public void Igpu_FittingButUnsupported_FallsBackToCpu_AndSaysSo()
    {
        var igpu = new IgpuTier("vulkan:1", "Arc", 600 * M, 100 * M, Supported: false);
        var plan = LayerSplitPlanner.Plan(Input(igpu: igpu));
        Assert.Equal((4, 0, 4), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
        Assert.Contains("not available yet", plan.Reason);
        Assert.Contains(LayerSplitPlanner.IgpuTierIssueUrl, plan.Reason);
    }

    [Fact]
    public void Igpu_NotUsed_WhenNoLayerFitsTheDgpu()
    {
        var igpu = new IgpuTier("vulkan:1", "Arc", 64_000 * M, 100 * M, Supported: true);
        var plan = LayerSplitPlanner.Plan(Input(freeMib: 300, igpu: igpu));
        Assert.Equal((0, 0, 8), (plan.DgpuLayers, plan.IgpuLayers, plan.CpuLayers));
    }
}
