using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class TopNSigmaSamplerTests
{
    private readonly TopNSigmaSampler _sampler = new();

    [Fact]
    public void Apply_NegativeNSigma_Skips()
    {
        float[] logits = [1.0f, 5.0f, 3.0f];
        float[] original = [1.0f, 5.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null, TopNSigma: -1f);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_MasksTokensBelowMeanMinusNSigma()
    {
        // A small set of outliers among mostly-uniform low logits.
        float[] logits = [0.0f, 0.0f, 0.0f, 0.0f, 100.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null, TopNSigma: 0.5f);

        _sampler.Apply(logits, context);

        // The dominant outlier must always survive (it defines the max).
        Assert.False(float.IsNegativeInfinity(logits[4]));
        // The low uniform tokens, far below max - 0.5*sigma, should be masked.
        Assert.True(float.IsNegativeInfinity(logits[0]));
    }

    [Fact]
    public void Apply_UniformDistribution_ZeroStdDev_KeepsAll()
    {
        float[] logits = [2.0f, 2.0f, 2.0f, 2.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null, TopNSigma: 1.0f);

        _sampler.Apply(logits, context);

        for (int i = 0; i < logits.Length; i++)
            Assert.False(float.IsNegativeInfinity(logits[i]));
    }

    [Fact]
    public void Apply_LargeNSigma_KeepsAll()
    {
        float[] logits = [1.0f, 50.0f, -20.0f, 5.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null, TopNSigma: 100.0f);

        _sampler.Apply(logits, context);

        for (int i = 0; i < logits.Length; i++)
            Assert.False(float.IsNegativeInfinity(logits[i]));
    }

    [Fact]
    public void Apply_IgnoresAlreadyMaskedTokensInStatistics()
    {
        // Pre-mask one token to -inf (as an earlier pipeline step, e.g. a constraint mask, would).
        // The mean/stddev computation must skip it rather than being skewed to -infinity.
        float[] logits = [float.NegativeInfinity, 1.0f, 1.0f, 1.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null, TopNSigma: 1.0f);

        _sampler.Apply(logits, context);

        Assert.True(float.IsNegativeInfinity(logits[0]));
        Assert.False(float.IsNegativeInfinity(logits[1]));
        Assert.False(float.IsNegativeInfinity(logits[2]));
        Assert.False(float.IsNegativeInfinity(logits[3]));
    }
}
