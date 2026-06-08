using System.Numerics.Tensors;
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

    /// <summary>
    /// Reference implementation that mirrors the pre-cutoff algorithm: full softmax,
    /// full O(V log V) sort, descending cumulative-mass walk. Used to verify that the
    /// production sampler (which prunes via Karpathy's `(1 - topP) / (n - 1)` cutoff
    /// before sorting) produces bit-identical masked logits for non-tie distributions.
    /// </summary>
    private static void ApplyReference(Span<float> logits, float topP)
    {
        if (topP >= 1.0f) return;

        int vocab = logits.Length;
        var probs = new float[vocab];
        var indices = new int[vocab];
        TensorPrimitives.SoftMax(logits, probs);
        for (int i = 0; i < vocab; i++) indices[i] = i;
        Array.Sort(probs, indices, 0, vocab);

        float cumulative = 0f;
        int cutoffCount = vocab;
        for (int i = vocab - 1; i >= 0; i--)
        {
            cumulative += probs[i];
            if (cumulative >= topP)
            {
                cutoffCount = vocab - i;
                break;
            }
        }

        var keep = new bool[vocab];
        int keepStart = vocab - cutoffCount;
        for (int i = keepStart; i < vocab; i++)
            keep[indices[i]] = true;

        for (int i = 0; i < vocab; i++)
            if (!keep[i])
                logits[i] = float.NegativeInfinity;
    }

    /// <summary>
    /// Bit-exact parity test: the pre-cutoff optimization must produce identical masked
    /// logits to the original full-sort algorithm for a realistically large vocab with
    /// random (non-tied) probabilities. This is the key correctness guarantee.
    /// </summary>
    [Theory]
    [InlineData(32_000, 0.9f, 1)]
    [InlineData(32_000, 0.95f, 2)]
    [InlineData(128_000, 0.9f, 3)]
    [InlineData(128_000, 0.5f, 4)]
    [InlineData(128_000, 0.99f, 5)]
    public void Apply_BitExactParityWithFullSort(int vocabSize, float topP, int seed)
    {
        var rng = new Random(seed);
        var refLogits = new float[vocabSize];
        var optLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            // Random in roughly [-10, 10) — produces no engineered ties.
            float v = (float)(rng.NextDouble() * 20.0 - 10.0);
            refLogits[i] = v;
            optLogits[i] = v;
        }

        var context = new SamplerContext(
            Temperature: 1.0f, TopK: 0, TopP: topP, MinP: 0f, Seed: null);

        ApplyReference(refLogits, topP);
        _sampler.Apply(optLogits, context);

        for (int i = 0; i < vocabSize; i++)
        {
            // Either both masked or both surviving with the exact same logit value.
            bool refMasked = float.IsNegativeInfinity(refLogits[i]);
            bool optMasked = float.IsNegativeInfinity(optLogits[i]);
            Assert.Equal(refMasked, optMasked);
            if (!refMasked)
                Assert.Equal(refLogits[i], optLogits[i]);
        }
    }
}
