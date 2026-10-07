using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #784 — Gemma-4 26B-A4B / 31B full-attention layers rotate the leading 128 dims of a 512-dim head with
/// <c>rope_freqs.weight</c> factors. ggml runs that rope with <c>n_rot = 512</c>, so the per-pair exponent is
/// <c>theta^(-2i/512)</c>. The CPU table used to be built with the ROTATED count (128) as the exponent denominator
/// whenever factors were present, which made every global layer rotate ~4x too fast (garbage text on CPU).
/// Expected values are the analytic ggml formula, never another dotLLM path. Shapes are the real 31B ones
/// (global head 512, 64 live pairs, 192 frozen pairs), not a degenerate rotated == full head.
/// </summary>
public sealed class Gemma4GlobalRopeFactorsTests
{
    private const int FullHeadDim = 512;
    private const int RotatedDim = 128;
    private const float GlobalTheta = 1_000_000f;

    private static float[] ProportionalFactors()
    {
        var f = new float[FullHeadDim / 2];
        for (int i = 0; i < f.Length; i++) f[i] = i < RotatedDim / 2 ? 1f : 1e30f;
        return f;
    }

    [Fact]
    public void GlobalTable_PartialRotaryWithFactors_UsesFullHeadDimExponent()
    {
        const int maxSeq = 300;
        using var state = new TransformerForwardState(
            hiddenSize: 256, numHeads: 4, numKvHeads: 2, headDim: 256,
            intermediateSize: 512, vocabSize: 1000, maxSeqLen: maxSeq, ropeDim: 256,
            ropeTheta: 10_000f,
            globalRopeDim: RotatedDim, globalRopeTheta: GlobalTheta,
            globalFullHeadDim: FullHeadDim, globalFreqFactors: ProportionalFactors());

        int half = RotatedDim / 2;
        Assert.Equal(maxSeq * half, state.GlobalCosTable!.Length);
        for (int pos = 1; pos < maxSeq; pos += 11)
        for (int i = 0; i < half; i++)
        {
            double angle = pos * Math.Pow(GlobalTheta, -2.0 * i / FullHeadDim);
            Assert.InRange(state.GlobalCosTable![pos * half + i], Math.Cos(angle) - 2e-4, Math.Cos(angle) + 2e-4);
            Assert.InRange(state.GlobalSinTable![pos * half + i], Math.Sin(angle) - 2e-4, Math.Sin(angle) + 2e-4);
        }

        // Discriminator: the buggy exponent (rotated count) gives a measurably different angle at the mid pairs.
        int probe = half / 2;
        double buggy = 100 * Math.Pow(GlobalTheta, -2.0 * probe / RotatedDim);
        double right = 100 * Math.Pow(GlobalTheta, -2.0 * probe / FullHeadDim);
        Assert.True(Math.Abs(Math.Sin(buggy) - Math.Sin(right)) > 0.05, "probe pair must tell the two exponents apart");
    }
}
