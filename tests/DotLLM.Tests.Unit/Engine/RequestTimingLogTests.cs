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

        Assert.Contains("draft acceptance 0.500 (20/40), mean len 2.50", line);
        Assert.DoesNotContain("prompt cache", line);
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
