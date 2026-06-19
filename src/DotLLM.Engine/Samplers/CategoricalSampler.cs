using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Numerics.Tensors;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Samples a token ID from a logit distribution using categorical (multinomial) sampling.
/// Converts logits to probabilities via softmax, then samples from the cumulative distribution.
/// </summary>
public static class CategoricalSampler
{
    private const int StackTopKThreshold = 512;

    private struct TopKCandidate
    {
        public int Id;
        public float Value;
    }

    /// <summary>
    /// Samples a single token index from the given logits using the provided RNG.
    /// </summary>
    /// <param name="logits">Logit values (will be converted to probabilities internally).</param>
    /// <param name="rng">Random number generator for sampling.</param>
    /// <returns>The sampled token index.</returns>
    public static int Sample(ReadOnlySpan<float> logits, Random rng)
    {
        int vocabSize = logits.Length;
        float[]? rented = null;
        Span<float> probs = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : (rented = ArrayPool<float>.Shared.Rent(vocabSize)).AsSpan(0, vocabSize);

        try
        {
            TensorPrimitives.SoftMax(logits, probs);

            double r = rng.NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < vocabSize; i++)
            {
                cumulative += probs[i];
                if (r < cumulative)
                    return i;
            }

            // Floating-point edge case: return highest-probability token
            return TensorPrimitives.IndexOfMax(probs);
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Samples from the top-K logits without first masking and softmaxing the whole vocabulary.
    /// </summary>
    /// <param name="logits">Full vocabulary logits.</param>
    /// <param name="topK">Number of top tokens to sample from.</param>
    /// <param name="temperature">Sampling temperature. Values at or below zero are treated as 1.</param>
    /// <param name="rng">Random number generator for sampling.</param>
    /// <returns>The sampled token index.</returns>
    public static int SampleTopK(ReadOnlySpan<float> logits, int topK, float temperature, Random rng)
    {
        if (topK == 1)
            return TensorPrimitives.IndexOfMax(logits);

        if (topK <= 0 || topK >= logits.Length)
        {
            if (temperature > 0f && temperature != 1.0f)
                return SampleWithTemperature(logits, temperature, rng);

            return Sample(logits, rng);
        }

        TopKCandidate[]? rentedCandidates = null;

        Span<TopKCandidate> candidates = topK <= StackTopKThreshold
            ? stackalloc TopKCandidate[topK]
            : (rentedCandidates = ArrayPool<TopKCandidate>.Shared.Rent(topK)).AsSpan(0, topK);

        try
        {
            CollectTopK(logits, candidates);
            candidates.Sort(static (left, right) => left.Id.CompareTo(right.Id));
            return SampleCandidates(candidates, temperature, rng);
        }
        finally
        {
            if (rentedCandidates is not null)
                ArrayPool<TopKCandidate>.Shared.Return(rentedCandidates);
        }
    }

    private static int SampleWithTemperature(ReadOnlySpan<float> logits, float temperature, Random rng)
    {
        int vocabSize = logits.Length;
        float[]? rented = null;
        Span<float> scaled = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : (rented = ArrayPool<float>.Shared.Rent(vocabSize)).AsSpan(0, vocabSize);

        try
        {
            float invTemp = 1f / temperature;
            for (int i = 0; i < vocabSize; i++)
                scaled[i] = logits[i] * invTemp;

            return Sample(scaled, rng);
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static int SampleCandidates(ReadOnlySpan<TopKCandidate> candidates, float temperature, Random rng)
    {
        Debug.Assert(candidates.Length > 0);

        float invTemp = temperature > 0f && temperature != 1.0f ? 1f / temperature : 1f;

        float max = float.NegativeInfinity;
        for (int i = 0; i < candidates.Length; i++)
        {
            float scaled = candidates[i].Value * invTemp;
            if (scaled > max)
                max = scaled;
        }

        double sum = 0.0;
        for (int i = 0; i < candidates.Length; i++)
            sum += Math.Exp((candidates[i].Value * invTemp) - max);

        double threshold = rng.NextDouble() * sum;
        double cumulative = 0.0;
        for (int i = 0; i < candidates.Length; i++)
        {
            cumulative += Math.Exp((candidates[i].Value * invTemp) - max);
            if (threshold < cumulative)
                return candidates[i].Id;
        }

        int maxIndex = 0;
        for (int i = 1; i < candidates.Length; i++)
        {
            if (candidates[i].Value > candidates[maxIndex].Value)
                maxIndex = i;
        }

        return candidates[maxIndex].Id;
    }

    private static void CollectTopK(ReadOnlySpan<float> logits, Span<TopKCandidate> heap)
    {
        int k = heap.Length;
        for (int i = 0; i < k; i++)
        {
            heap[i] = new TopKCandidate { Id = i, Value = logits[i] };
        }

        for (int i = (k >> 1) - 1; i >= 0; i--)
            SiftDown(heap, i);

        for (int i = k; i < logits.Length; i++)
        {
            float v = logits[i];
            Debug.Assert(!float.IsNaN(v), "CategoricalSampler: logits must not contain NaN");
            if (v > heap[0].Value)
            {
                heap[0] = new TopKCandidate { Id = i, Value = v };
                SiftDown(heap, 0);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiftDown(Span<TopKCandidate> heap, int i)
    {
        int n = heap.Length;
        TopKCandidate x = heap[i];
        while (true)
        {
            int left = (i << 1) + 1;
            if (left >= n)
                break;
            int right = left + 1;
            int smaller = (right < n && heap[right].Value < heap[left].Value) ? right : left;
            if (heap[smaller].Value >= x.Value)
                break;
            heap[i] = heap[smaller];
            i = smaller;
        }

        heap[i] = x;
    }
}
