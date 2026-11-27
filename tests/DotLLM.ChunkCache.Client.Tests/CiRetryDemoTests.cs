using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// TEMPORARY demonstration of the CI retry policy (#2052). Inert unless DOTLLM_CI_RETRY_DEMO is set by the workflow.
/// Removed before merge.
/// </summary>
public class CiRetryDemoTests
{
    private static string? Mode => Environment.GetEnvironmentVariable("DOTLLM_CI_RETRY_DEMO");

    [Fact]
    public void Demo_FlakyOnce_FailsFirstAttemptOnly()
    {
        if (Mode is not ("once" or "always")) return;
        var marker = Path.Combine(Path.GetTempPath(), "dotllm-ci-retry-demo-once.marker");
        if (!File.Exists(marker))
        {
            File.WriteAllText(marker, "attempt1");
            Assert.Fail("deliberate first-attempt failure (flaky-once demo, #2052)");
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Demo_AlwaysFails(int n)
    {
        if (!string.Equals(Mode, "always", StringComparison.Ordinal)) return;
        Assert.Fail($"deliberate permanent failure {n} (always-fail demo, #2052)");
    }
}
