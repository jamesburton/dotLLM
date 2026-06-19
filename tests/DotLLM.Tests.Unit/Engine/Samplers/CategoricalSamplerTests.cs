using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class CategoricalSamplerTests
{
    [Fact]
    public void Sample_ReturnsValidIndex()
    {
        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
        var rng = new Random(42);

        int result = CategoricalSampler.Sample(logits, rng);

        Assert.InRange(result, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_SeededDeterminism()
    {
        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result1 = CategoricalSampler.Sample(logits, new Random(42));
        int result2 = CategoricalSampler.Sample(logits, new Random(42));

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void Sample_PeakedDistribution_FavorsMax()
    {
        // One very high logit, rest are -inf → softmax gives prob 1.0 at index 0
        float[] logits = [10.0f, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];

        // Should always pick index 0
        for (int i = 0; i < 10; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            Assert.Equal(0, result);
        }
    }

    [Fact]
    public void Sample_NegativeInfinityNeverSelected()
    {
        // Only index 2 has a valid logit
        float[] logits = [float.NegativeInfinity, float.NegativeInfinity, 1.0f, float.NegativeInfinity];
        var rng = new Random(42);

        for (int i = 0; i < 10; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            Assert.Equal(2, result);
        }
    }

    [Fact]
    public void Sample_UniformDistribution_ProducesVariety()
    {
        float[] logits = [0f, 0f, 0f, 0f];
        var seen = new HashSet<int>();

        // With enough samples, should see multiple indices
        for (int i = 0; i < 100; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            seen.Add(result);
        }

        Assert.True(seen.Count > 1, "Uniform distribution should produce variety over 100 samples.");
    }

    [Fact]
    public void Sample_FallbackReturnsHighestProbToken()
    {
        // Construct logits where index 2 has the highest logit.
        // The fallback should return index 2, not vocabSize-1 (index 4).
        float[] logits = [-10f, -10f, 10f, -10f, -10f];

        // With such peaked logits, softmax gives ~1.0 to index 2.
        // All seeds should pick index 2 (either through normal sampling or fallback).
        for (int i = 0; i < 20; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            Assert.Equal(2, result);
        }
    }

    [Fact]
    public void SampleTopK_ReturnsOnlyTopKToken()
    {
        float[] logits = [0.1f, 9.0f, 0.2f, 8.0f, 0.3f, 7.0f];
        var allowed = new HashSet<int> { 1, 3, 5 };

        for (int i = 0; i < 100; i++)
        {
            int result = CategoricalSampler.SampleTopK(logits, topK: 3, temperature: 1.0f, new Random(i));
            Assert.Contains(result, allowed);
        }
    }

    [Fact]
    public void SampleTopK_K1_ReturnsArgMax()
    {
        float[] logits = [0.1f, 9.0f, 0.2f, 8.0f, 0.3f];

        int result = CategoricalSampler.SampleTopK(logits, topK: 1, temperature: 0.8f, new Random(42));

        Assert.Equal(1, result);
    }

    [Fact]
    public void SampleTopK_UsesVocabularyOrderForThresholdTies()
    {
        float[] logits = [1.0f, 2.0f, 2.0f, 0.5f];

        for (int i = 0; i < 20; i++)
        {
            int result = CategoricalSampler.SampleTopK(logits, topK: 2, temperature: 1.0f, new Random(i));
            Assert.True(result is 1 or 2, $"Expected token 1 or 2, got {result}.");
        }
    }

    [Fact]
    public void SampleTopK_TemperatureKeepsSelectionWithinTopK()
    {
        float[] logits = [0.1f, 3.0f, 0.2f, 2.9f, 0.3f];

        for (int i = 0; i < 50; i++)
        {
            int result = CategoricalSampler.SampleTopK(logits, topK: 2, temperature: 0.7f, new Random(i));
            Assert.True(result is 1 or 3, $"Expected token 1 or 3, got {result}.");
        }
    }

    [Fact]
    public void ArgMax_MatchesReference_AcrossSizes()
    {
        var rng = new Random(123);
        foreach (int n in new[] { 1, 2, 3, 7, 31, 64, 127, 128, 1000, 131072 })
        {
            var logits = new float[n];
            for (int i = 0; i < n; i++)
                logits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

            int expected = ReferenceArgMax(logits);
            int actual = CategoricalSampler.ArgMax(logits);

            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void ArgMax_ReturnsFirstIndexOnTies()
    {
        float[] logits = [1.0f, 5.0f, 5.0f, 2.0f, 5.0f];

        Assert.Equal(1, CategoricalSampler.ArgMax(logits));
    }

    // Mirrors TensorPrimitives.IndexOfMax semantics for finite inputs: first index of the maximum.
    private static int ReferenceArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }
}
