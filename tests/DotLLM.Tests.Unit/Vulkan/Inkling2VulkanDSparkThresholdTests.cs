using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>The context threshold of the lazily attached Vulkan DSpark drafter (issue #2046, James 2026-11-28). Own fixture: its model must not have an explicit drafter attached.</summary>
public sealed class Inkling2VulkanDSparkThresholdTests : IClassFixture<VkInklingFixture>
{
    private readonly VkInklingFixture _fx;
    public Inkling2VulkanDSparkThresholdTests(VkInklingFixture fx) => _fx = fx;

    // ───────────── context threshold (lazy attach) ─────────────

    /// <summary>Pure rules: below the threshold nothing attaches and nothing speculates; at or above, both; an explicit attach forces it on. A policy that ignores the threshold fails every row.</summary>
    [Theory]
    [InlineData(100, 4096, false, false, false)]
    [InlineData(4095, 4096, false, false, false)]
    [InlineData(4096, 4096, true, true, true)]
    [InlineData(20000, 4096, true, true, true)]
    [InlineData(100, 0, true, true, true)]
    [InlineData(100, int.MaxValue, false, false, false)]
    public void AttachPolicy_FollowsTheThreshold(int ctx, int min, bool attach, bool mayUseUnforced, bool unused)
    {
        _ = unused;
        Assert.Equal(attach, DSparkAttachPolicy.ShouldAttach(true, ctx, min));
        Assert.Equal(mayUseUnforced, DSparkAttachPolicy.MayUse(false, ctx, min));
        Assert.True(DSparkAttachPolicy.MayUse(true, ctx, min));          // an explicit --dspark forces it on at every length
        Assert.False(DSparkAttachPolicy.ShouldAttach(false, ctx, min));  // nothing deferred, nothing to attach
    }

    [Theory]
    [InlineData(null, 8192, 8192)]
    [InlineData("", 8192, 8192)]
    [InlineData("0", 8192, 0)]
    [InlineData("4096", 8192, 4096)]
    [InlineData("off", 8192, int.MaxValue)]
    [InlineData("OFF", 8192, int.MaxValue)]
    public void MinContextEnv_Parses(string? text, int dflt, int want) => Assert.Equal(want, DSparkAttachPolicy.ParseMinContext(text, dflt));

    [Theory]
    [InlineData("-1")]
    [InlineData("many")]
    public void MinContextEnv_RejectsGarbage(string text) => Assert.Throws<ArgumentException>(() => DSparkAttachPolicy.ParseMinContext(text, 1));

    /// <summary>
    /// Real model: with an auto-detected drafter DEFERRED, a request below the threshold allocates nothing and does not speculate; one at the threshold attaches it (device bytes jump by the drafter's weights);
    /// an explicit attach forces it on at every length. Mutant: a threshold of 0 (the threshold ignored) attaches at 500 tokens and fails the first assertion.
    /// </summary>
    [SkippableFact]
    public void DeferredDrafter_AttachesOnlyAtTheThreshold_AndExplicitForcesIt()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        Skip.If(_fx.TargetPath is null || _fx.DraftPath is null, "model or drafter missing");
        var m = _fx.Get();
        m.DeferAutoAttach(_fx.DraftPath!);
        m.DSparkMinContext = 1000;
        long before = m.ComputeMemoryBytes;
        Assert.False(m.PrepareMtpFor(500));
        Assert.False(m.HasDFlashDrafter);
        Assert.False(m.SupportsMtp);
        Assert.Equal(before, m.ComputeMemoryBytes);                 // nothing allocated below the threshold
        Assert.True(m.PrepareMtpFor(1000));                         // at the threshold: attached, may speculate
        Assert.True(m.HasDFlashDrafter);
        Assert.True(m.ComputeMemoryBytes - before > 400L * 1024 * 1024);
        Assert.False(m.PrepareMtpFor(999));                         // attached, but a short request does not speculate
        Assert.True(m.PrepareMtpFor(5000));
    }
}
