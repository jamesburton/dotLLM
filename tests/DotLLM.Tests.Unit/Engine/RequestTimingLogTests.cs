using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// #1288: the per-request timing line rates computed prompt tokens only (prefix-cache hits excluded), reports cache
/// reuse and draft acceptance + mean accepted length when they apply, and the prefill pulse reports completed work.
/// </summary>
public sealed class RequestTimingLogTests
{
    [Fact]
    public void Format_RatesComputedTokensOnly_AndReportsCacheReuse()
    {
        var t = new InferenceTimings
        {
            PrefillTimeMs = 100,
            PrefillTokenCount = 300,
            CachedTokenCount = 100, // 200 computed -> 0.50 ms/tok, 2000 tok/s
            DecodeTimeMs = 50,
            DecodeTokenCount = 10,
        };

        string line = RequestTimingLog.Format(t);

        Assert.StartsWith("timing: prompt eval 100.0 ms / 200 tokens (0.50 ms/tok, 2000.0 tok/s)", line);
        Assert.Contains("eval 50.0 ms / 10 tokens (5.00 ms/tok, 200.0 tok/s)", line);
        Assert.Contains("prompt cache 100/300 tokens reused", line);
        Assert.DoesNotContain("draft", line);
    }

    [Fact]
    public void Format_ReportsDraftAcceptanceAndMeanLength_WhenSpeculating()
    {
        var t = new InferenceTimings
        {
            PrefillTimeMs = 10,
            PrefillTokenCount = 10,
            DecodeTimeMs = 10,
            DecodeTokenCount = 20,
            SpeculativeDraftTokens = 40,
            SpeculativeAcceptedTokens = 20,
            SpeculativeSteps = 8,
        };

        string line = RequestTimingLog.Format(t);

        Assert.Contains("draft acceptance 0.500 (20/40), mean len 3.50", line);
        Assert.DoesNotContain("prompt cache", line);
    }

    /// <summary>#1723: a single-pass prefill cannot report partial progress, so it must not print a misleading 0/N (0.0%).</summary>
    [Fact]
    public void Pulse_SinglePass_PrintsInProgressNotZeroPercent()
    {
        var lines = new List<string>();
        using var pulse = new PrefillPulse(lines.Add, total: 9200, Timeout.InfiniteTimeSpan) { SinglePass = true };

        pulse.Pulse();

        string line = Assert.Single(lines);
        Assert.StartsWith("prefill in progress (single pass), 9200 tokens", line);
        Assert.DoesNotContain("0.0%", line);
    }

    /// <summary>A chunked prefill (not single-pass) keeps the counted form even at zero completed.</summary>
    [Fact]
    public void Pulse_Chunked_StillPrintsCounts()
    {
        var lines = new List<string>();
        using var pulse = new PrefillPulse(lines.Add, total: 400, Timeout.InfiniteTimeSpan);

        pulse.Pulse();

        Assert.StartsWith("prefill: 0/400 tokens (0.0%)", Assert.Single(lines));
    }

    [Theory]
    [InlineData(1, null, DotLLM.Vulkan.VulkanPrefillChunk.IntegratedDefault)]
    [InlineData(2, null, DotLLM.Vulkan.VulkanPrefillChunk.DiscreteDefault)]
    [InlineData(1, "0", 0)]          // explicit single pass
    [InlineData(2, "1024", 1024)]    // explicit override wins over the device default
    [InlineData(1, "-3", DotLLM.Vulkan.VulkanPrefillChunk.IntegratedDefault)]   // malformed -> default
    [InlineData(1, "abc", DotLLM.Vulkan.VulkanPrefillChunk.IntegratedDefault)]
    public void VulkanPrefillChunk_ResolvesEnvThenDeviceDefault(int deviceType, string? env, int expected)
        => Assert.Equal(expected, DotLLM.Vulkan.VulkanPrefillChunk.Resolve(deviceType, env));

    [Fact]
    public void VulkanPrefillChunk_DefaultsAreConservative()
    {
        Assert.InRange(DotLLM.Vulkan.VulkanPrefillChunk.IntegratedDefault, 64, 1024);
        Assert.InRange(DotLLM.Vulkan.VulkanPrefillChunk.DiscreteDefault, 64, 1024);
    }

    [Fact]
    public void Pulse_ReportsCompletedOverTotal()
    {
        var lines = new List<string>();
        using var pulse = new PrefillPulse(lines.Add, total: 400, Timeout.InfiniteTimeSpan);

        pulse.Advance(100);
        pulse.Pulse();

        Assert.Equal(100, pulse.Completed);
        Assert.StartsWith("prefill: 100/400 tokens (25.0%)", Assert.Single(lines));
    }
}
