using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

/// <summary>#1433: <see cref="EntropyBoundSampler"/> must draw Gumbel-max tokens at the real sequence index, not an empty-history index 0.</summary>
public class EntropyBoundSamplerGumbelTests
{
    private const int Vocab = 64, Rows = 12;

    private static SamplerPipeline Gumbel(int seed = 7)
        => new(new InferenceOptions { Temperature = 1.0f, Seed = seed, GumbelSampling = true });

    // Identical (uniform) rows at every position: only the noise can tell positions apart.
    private static UnmaskDecision Run(EntropyBoundSampler s, int tokenIndexBase, int[] positions)
        => s.SelectAndSample(new float[positions.Length * Vocab], positions, Vocab, positions.Length,
            new DiffusionStepContext(1.0f, 100, 1e9f, 0f, tokenIndexBase));

    [Fact]
    public void GumbelPipeline_DrawsAtTheRealSequenceIndex_NotAnEmptyHistoryIndex()
    {
        int[] positions = Enumerable.Range(0, Rows).ToArray();
        var pipeline = Gumbel();
        var d = Run(new EntropyBoundSampler(pipeline), tokenIndexBase: 100, positions);

        Assert.Equal(Rows, d.TokenIds.Count);
        for (int i = 0; i < Rows; i++)
            Assert.Equal(GumbelNoise.Sample(new float[Vocab], pipeline.GumbelSeed, 100 + d.UnmaskedPositions[i]), d.TokenIds[i]);
        // The empty-history bug drew every position with the SAME noise vector -> one repeated token.
        Assert.True(d.TokenIds.Distinct().Count() > 1, "all positions drew the same token: shared noise index");
    }

    [Fact]
    public void GumbelPipeline_IsReproducible_AndKeyedOnTheCanvasBase()
    {
        int[] positions = Enumerable.Range(0, Rows).ToArray();
        var a = Run(new EntropyBoundSampler(Gumbel()), 0, positions);
        var b = Run(new EntropyBoundSampler(Gumbel()), 0, positions);
        Assert.Equal(a.TokenIds, b.TokenIds);                           // same seed, same text
        var next = Run(new EntropyBoundSampler(Gumbel()), 1000, positions);
        Assert.NotEqual(a.TokenIds, next.TokenIds);                     // a later canvas does not replay the first canvas' noise
        var other = Run(new EntropyBoundSampler(Gumbel(8)), 0, positions);
        Assert.NotEqual(a.TokenIds, other.TokenIds);
    }

    [Fact]
    public void NonGumbelPipeline_IsUnchanged()
    {
        var s = new EntropyBoundSampler(new SamplerPipeline(new InferenceOptions { Temperature = 0f }));
        var d = Run(s, 0, [0, 1, 2]);
        Assert.All(d.TokenIds, t => Assert.Equal(0, t));
    }
}
