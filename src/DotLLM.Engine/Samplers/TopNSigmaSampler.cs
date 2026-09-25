using System.Numerics.Tensors;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Top-nσ sampling (llama.cpp <c>--top-nsigma</c>): masks tokens whose logit falls more than
/// <c>n</c> standard deviations below the maximum logit, computed over the raw (pre-temperature)
/// logit distribution. Unlike top-p/min-p, the threshold is derived from the distribution's shape
/// rather than cumulative probability mass, which the "Top-nσ" paper argues is more robust to
/// temperature scaling — the mean/stddev computation ignores tokens already masked to -∞ by an
/// earlier pipeline step (e.g. a constraint mask).
/// </summary>
public sealed class TopNSigmaSampler : ISamplerStep
{
    private readonly float? _nSigma;

    /// <summary>Creates a top-nσ step that reads from <see cref="SamplerContext"/>.</summary>
    public TopNSigmaSampler() { }

    /// <summary>Creates a self-configured top-nσ step.</summary>
    /// <param name="nSigma">Number of standard deviations below the max logit to keep (ignores context).</param>
    public TopNSigmaSampler(float nSigma) => _nSigma = nSigma;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        float n = _nSigma ?? context.TopNSigma;
        if (n < 0f)
            return;

        float maxLogit = TensorPrimitives.Max(logits);

        double sum = 0;
        int count = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (float.IsNegativeInfinity(logits[i]))
                continue;
            sum += logits[i];
            count++;
        }
        if (count == 0)
            return;

        double mean = sum / count;
        double sumSquares = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (float.IsNegativeInfinity(logits[i]))
                continue;
            double delta = logits[i] - mean;
            sumSquares += delta * delta;
        }
        double stdDev = Math.Sqrt(sumSquares / count);

        float threshold = maxLogit - n * (float)stdDev;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] < threshold)
                logits[i] = float.NegativeInfinity;
        }
    }
}
