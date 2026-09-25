using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class TopPSamplerTests
{
    private readonly TopPSampler _sampler = new();

    [Fact]
    public void Apply_CumulativeProbabilityThreshold()
    {
        // Logits that produce a peaked distribution
        float[] logits = [10.0f, 1.0f, 0.0f, -1.0f, -10.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.95f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // The highest logit should always survive
        Assert.False(float.IsNegativeInfinity(logits[0]));
        // The very low logit should be masked
        Assert.True(float.IsNegativeInfinity(logits[4]));
    }

    [Fact]
    public void Apply_P1_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_VeryLowP_KeepsOnlyTopToken()
    {
        // Very peaked: only one dominant token
        float[] logits = [10.0f, 0.0f, 0.0f, 0.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.01f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // The top token (index 0) should survive, rest masked
        Assert.False(float.IsNegativeInfinity(logits[0]));
        Assert.True(float.IsNegativeInfinity(logits[1]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.True(float.IsNegativeInfinity(logits[3]));
    }

    [Fact]
    public void Apply_UniformDistribution_KeepsSmallestPrefixOverThreshold()
    {
        float[] logits = [0.0f, 0.0f, 0.0f, 0.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.50f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        int kept = logits.Count(v => !float.IsNegativeInfinity(v));
        Assert.Equal(2, kept);
    }

    // ---------------------------------------------------------------------------------------
    // Deterministic tie-breaking (#423).
    //
    // At large vocabularies distinct logits round to the same float probability after softmax.
    // Array.Sort is an unstable IntroSort whose strategy varies with array size, so without an
    // explicit secondary key the surviving member of a tie is unspecified. TopPSampler orders
    // candidates by descending probability and, within a tie, ascending token id.
    //
    // Both tests below use a 32K vocabulary on purpose: they must exercise the introsort /
    // heapsort paths, not the small-array insertion sort, to be discriminating.
    // ---------------------------------------------------------------------------------------

    private const int LargeVocab = 32768;

    [Fact]
    public void Apply_DistinctLogitsRoundingToEqualProbability_KeepsLowestTokenId()
    {
        // Two candidates with *distinct* logits that are indistinguishable after softmax:
        // exp(-1e-8f) rounds to 1.0f, so both tokens receive the identical float probability.
        // Every other token is far enough down to contribute nothing.
        float[] logits = new float[LargeVocab];
        Array.Fill(logits, -30.0f);
        const int lowId = 1000;
        const int highId = 2000;
        logits[lowId] = 1e-8f;
        logits[highId] = 0.0f;
        Assert.NotEqual(logits[lowId], logits[highId]); // genuinely distinct logits

        // Sanity: the two probabilities really are the same float.
        float[] probeProbs = new float[LargeVocab];
        System.Numerics.Tensors.TensorPrimitives.SoftMax(logits, probeProbs);
        Assert.Equal(probeProbs[lowId], probeProbs[highId]);

        // ~0.5 each, so the first candidate alone clears the threshold: exactly one survives.
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.30f, MinP: 0f, Seed: null);
        _sampler.Apply(logits, context);

        int[] survivors = [.. Enumerable.Range(0, LargeVocab).Where(i => !float.IsNegativeInfinity(logits[i]))];
        Assert.Equal([lowId], survivors);
    }

    [Fact]
    public void Apply_FullyTiedVocabulary_KeepsLowestTokenIds()
    {
        // Every token ties with every other: 1/32768 is exactly representable, so the whole
        // vocabulary is one tie group and the entire kept set is decided by tie-breaking alone.
        float[] logits = new float[LargeVocab];
        var context = new SamplerContext(
            Temperature: 1.0f, TopK: 0, TopP: 3.0f / LargeVocab, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        int[] survivors = [.. Enumerable.Range(0, LargeVocab).Where(i => !float.IsNegativeInfinity(logits[i]))];
        Assert.Equal([0, 1, 2], survivors);
    }

    [Fact]
    public void Apply_IsReproducibleAcrossRepeatedCalls()
    {
        static float[] Run()
        {
            float[] logits = new float[LargeVocab];
            Array.Fill(logits, 0.0f);
            for (int i = 0; i < LargeVocab; i += 3)
                logits[i] = 1e-8f; // sub-ULP variation: many exact probability ties

            new TopPSampler().Apply(
                logits,
                new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.5f, MinP: 0f, Seed: null));
            return logits;
        }

        Assert.Equal(Run(), Run());
    }
}
