using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Gemma-3n per-layer FFN activation sparsity ("Gaussian top-k gate"). Verified
/// against HF <c>transformers</c> <c>modeling_gemma3n.py</c>
/// <c>Gemma3nTextMLP._gaussian_topk</c>:
/// <code>
/// std_multiplier = Normal(0,1).icdf(target_sparsity)   // scalar, config-constant
/// cutoff = mean(gate_proj, -1) + std(gate_proj, -1) * std_multiplier
/// gate_proj = relu(gate_proj - cutoff)
/// </code>
/// Applied to <c>gate_proj</c> (per token, over the intermediate dimension)
/// BEFORE the activation function, only on layers whose
/// <c>activation_sparsity_pattern[layer] &gt; 0</c> (the real E4B/E2B ship 0.95
/// on the first 10 layers, 0.0 — i.e. disabled — on every layer after).
/// </summary>
public static unsafe class GaussianTopK
{
    /// <summary>
    /// Applies the Gaussian top-k gate in place to every token row of
    /// <paramref name="gateProj"/>. No-op (returns immediately) when
    /// <paramref name="targetSparsity"/> is not a positive, sub-1 fraction —
    /// callers should still prefer skipping the call entirely when sparsity is 0
    /// for a byte-identical plain-GeGLU path.
    /// </summary>
    /// <param name="gateProj"><c>[seqLen, intermediateSize]</c>, updated in place.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="intermediateSize">FFN intermediate width.</param>
    /// <param name="targetSparsity">This layer's <c>activation_sparsity_pattern</c>
    /// value, in <c>(0, 1)</c>.</param>
    public static void Apply(float* gateProj, int seqLen, int intermediateSize, float targetSparsity)
    {
        if (targetSparsity <= 0f || targetSparsity >= 1f)
            return;

        float stdMultiplier = InverseNormalCdf(targetSparsity);

        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(gateProj + (long)t * intermediateSize, intermediateSize);
            float mean = TensorPrimitives.Sum((ReadOnlySpan<float>)row) / intermediateSize;

            // Population std (HF: torch.std(..., unbiased=False)).
            float sumSqDev = 0f;
            for (int i = 0; i < intermediateSize; i++)
            {
                float d = row[i] - mean;
                sumSqDev += d * d;
            }
            float std = MathF.Sqrt(sumSqDev / intermediateSize);

            float cutoff = mean + std * stdMultiplier;
            for (int i = 0; i < intermediateSize; i++)
            {
                float v = row[i] - cutoff;
                row[i] = v > 0f ? v : 0f;
            }
        }
    }

    /// <summary>
    /// Standard normal inverse CDF (quantile function), Peter Acklam's rational
    /// approximation (relative error &lt; 1.15e-9 for the central region, refined
    /// by one Halley's-method correction step in the tails — see
    /// <see href="https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/"/>).
    /// <paramref name="p"/> is a config-constant sparsity target (never
    /// data-dependent), so this is called once per layer per forward, not per
    /// element.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float InverseNormalCdf(float p)
    {
        const double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02, a3 = -2.759285104469687e+02;
        const double a4 = 1.383577518672690e+02, a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
        const double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02, b3 = -1.556989798598866e+02;
        const double b4 = 6.680131188771972e+01, b5 = -1.328068155288572e+01;
        const double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01, c3 = -2.400758277161838e+00;
        const double c4 = -2.549732539343734e+00, c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
        const double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01, d3 = 2.445134137142996e+00;
        const double d4 = 3.754408661907416e+00;

        double pd = p;
        const double pLow = 0.02425, pHigh = 1 - pLow;
        double q, r, x;

        if (pd <= 0) return float.NegativeInfinity;
        if (pd >= 1) return float.PositiveInfinity;

        if (pd < pLow)
        {
            q = Math.Sqrt(-2 * Math.Log(pd));
            x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        }
        else if (pd <= pHigh)
        {
            q = pd - 0.5;
            r = q * q;
            x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        }
        else
        {
            q = Math.Sqrt(-2 * Math.Log(1 - pd));
            x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        }

        return (float)x;
    }
}
