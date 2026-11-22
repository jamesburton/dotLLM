using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1892: the prompt-prefill chunk is clamped to a model's hard per-forward limit by one rule every prefill path shares.</summary>
public class PrefillChunkBoundTests
{
    [Theory]
    [InlineData(0, 0, 0, 0, false)]          // no limit, nothing configured: one pass
    [InlineData(512, 0, 0, 512, false)]      // no limit: the configured size stands
    [InlineData(0, 300, 0, 300, false)]      // no limit: the preference stands
    [InlineData(0, 0, 2047, 2047, false)]    // a limit turns "one pass" into the limit (nothing was clamped: nothing was asked)
    [InlineData(0, 2047, 2047, 2047, false)]
    [InlineData(100, 2047, 2047, 100, false)]// below the limit: stands
    [InlineData(2047, 0, 2047, 2047, false)] // at the limit: stands
    [InlineData(2048, 2047, 2047, 2047, true)]
    [InlineData(100000, 0, 2047, 2047, true)]// an explicit size above the limit is clamped, and reported once by the caller
    public void Resolve(int configured, int preferred, int hardMax, int expected, bool expectClamped)
    {
        Assert.Equal(expected, PrefillChunkBound.Resolve(configured, preferred, hardMax, out bool clamped));
        Assert.Equal(expectClamped, clamped);
    }

    [Fact]
    public void ClampMessage_NamesBothNumbers()
    {
        string m = PrefillChunkBound.ClampMessage(4096, 2047);
        Assert.Contains("4096", m);
        Assert.Contains("2047", m);
    }
}
