using DotLLM.Cli.Benchmarking;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

public sealed class BenchStatsTests
{
    [Fact]
    public void Median_Odd_Count_Returns_Middle()
    {
        Assert.Equal(3.0, BenchStats.Median([5.0, 1.0, 3.0]));
    }

    [Fact]
    public void Median_Even_Count_Averages_Middle_Two()
    {
        Assert.Equal(2.5, BenchStats.Median([4.0, 1.0, 2.0, 3.0]));
    }

    [Fact]
    public void Median_Single_Value_Is_That_Value()
    {
        Assert.Equal(7.0, BenchStats.Median([7.0]));
    }

    [Fact]
    public void Median_Empty_Throws()
    {
        Assert.Throws<ArgumentException>(() => BenchStats.Median(Array.Empty<double>()));
    }

    [Fact]
    public void Min_Returns_Smallest()
    {
        Assert.Equal(1.5, BenchStats.Min([3.0, 1.5, 9.0]));
    }

    [Fact]
    public void Min_Empty_Throws()
    {
        Assert.Throws<ArgumentException>(() => BenchStats.Min(Array.Empty<double>()));
    }

    [Fact]
    public void DiscardWarmup_Drops_First_Rep()
    {
        var reps = new[] { 100.0, 10.0, 11.0, 12.0 };
        var measured = BenchStats.DiscardWarmup(reps);
        Assert.Equal([10.0, 11.0, 12.0], measured);
    }

    [Fact]
    public void DiscardWarmup_Requires_More_Than_Warmup_Count()
    {
        Assert.Throws<ArgumentException>(() => BenchStats.DiscardWarmup(new[] { 1.0 }));
    }

    [Fact]
    public void TilePrompt_Tiles_And_Truncates_To_Exact_Length()
    {
        int[] seed = [10, 20, 30];
        int[] prompt = BenchStats.TilePrompt(seed, 7);
        Assert.Equal([10, 20, 30, 10, 20, 30, 10], prompt);
    }

    [Fact]
    public void TilePrompt_Shorter_Than_Seed_Truncates()
    {
        int[] prompt = BenchStats.TilePrompt([10, 20, 30], 2);
        Assert.Equal([10, 20], prompt);
    }

    [Fact]
    public void TilePrompt_Empty_Seed_Throws()
    {
        Assert.Throws<ArgumentException>(() => BenchStats.TilePrompt(ReadOnlySpan<int>.Empty, 4));
    }

    [Theory]
    [InlineData(2485.4, "2485")]
    [InlineData(99.06, "99.1")]
    [InlineData(56.71, "56.7")]
    [InlineData(0.52, "0.52")]
    [InlineData(0.014, "0.014")]
    public void FormatTokS_Scales_Precision_With_Magnitude(double value, string expected)
    {
        Assert.Equal(expected, BenchStats.FormatTokS(value));
    }

    [Fact]
    public void BenchRep_Computes_Throughput()
    {
        var rep = new BenchRep(PrefillMs: 500, DecodeMs: 2000, PromptTokens: 512, DecodeTokens: 128);
        Assert.Equal(1024.0, rep.PrefillTokS, 3);
        Assert.Equal(64.0, rep.DecodeTokS, 3);
    }

    [Fact]
    public void BenchResult_Summaries_Exclude_Warmup()
    {
        var warmup = new BenchRep(1000, 4000, 100, 50); // pathologically slow warm-up
        var reps = new[]
        {
            new BenchRep(100, 1000, 100, 50), // pp 1000, tg 50
            new BenchRep(200, 2000, 100, 50), // pp 500,  tg 25
            new BenchRep(400, 500, 100, 50),  // pp 250,  tg 100
        };
        var result = new BenchResult
        {
            Warmup = warmup,
            Reps = reps,
            LoadMs = 1234,
            PromptTokens = 100,
            DecodeTokens = 50,
            Depth = 0,
        };

        Assert.Equal(500.0, result.PrefillTokSMedian, 3);
        Assert.Equal(1000.0, result.PrefillTokSBest, 3);
        Assert.Equal(50.0, result.DecodeTokSMedian, 3);
        Assert.Equal(100.0, result.DecodeTokSBest, 3);
        Assert.Equal(100.0, result.PrefillMsMin, 3);
        Assert.Equal(200.0, result.PrefillMsMedian, 3);
        Assert.Equal(500.0, result.DecodeMsMin, 3);
        Assert.Equal(100, result.DecodeCtxDepth);
    }

    [Fact]
    public void BenchResult_Depth_Adds_To_Ctx_Depth()
    {
        var rep = new BenchRep(1, 1, 32, 8);
        var result = new BenchResult
        {
            Warmup = rep,
            Reps = new[] { rep },
            LoadMs = 0,
            PromptTokens = 32,
            DecodeTokens = 8,
            Depth = 96,
        };
        Assert.Equal(128, result.DecodeCtxDepth);
    }
}
