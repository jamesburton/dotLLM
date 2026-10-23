using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Pins the <em>accuracy</em> of <see cref="FastMath"/>'s default path (#501).
/// </summary>
/// <remarks>
/// <para>The sibling <see cref="FastMathTests"/> allow 2-5% relative error, which is exactly the
/// Schraudolph bit trick's error budget — they pass whether the default is the approximation or a
/// correctly-rounded <c>exp</c>, so they cannot detect a regression back to it. These bounds are
/// 1e-5, which the bit trick misses by ~1000x.</para>
/// <para>Why it matters: a ~1-2% <em>relative</em> error in the attention softmax does not cancel
/// when dividing by the sum — it reweights the mixture per element. Measured cost on
/// <c>Llama-3.2-1B-pure</c> Q3_K, wikitext-2, ctx 512, 40 chunks, paired per-chunk:
/// <b>+0.01724 +/- 0.00185 nats</b> (t = 9.3), i.e. +1.71% perplexity, and it was ~85% of the
/// gap to llama.cpp reported in #501. On Q8_0 it costs nothing, which is why it went unnoticed.</para>
/// </remarks>
public sealed class FastMathAccuracyTests
{
    /// <summary>The attention-softmax domain: scores are max-subtracted, so always &lt;= 0.</summary>
    [Fact]
    public void FastExp_IsAccurate_OverAttentionRange()
    {
        var rng = new Random(20501);
        for (int i = 0; i < 20_000; i++)
        {
            float x = rng.NextSingle() * -80f;
            float expected = MathF.Exp(x);
            float actual = FastMath.FastExp(x);

            float relError = MathF.Abs(actual - expected) / expected;
            Assert.True(relError < 1e-5f,
                $"FastExp({x}) = {actual}, MathF.Exp = {expected}, relative error = {relError:E3}. "
                + "The Schraudolph approximation is ~1-2% here and must not be the default (#501).");
        }
    }

    [Fact]
    public void ExpSumAndStore_IsAccurate_ElementwiseAndInTheSum()
    {
        var rng = new Random(20501);
        const int n = 1024;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.NextSingle() * 20f - 20f;

        const float offset = -1.25f;

        float[] actual = new float[n];
        float actualSum = FastMath.ExpSumAndStore(input, actual, offset);

        double referenceSum = 0;
        for (int i = 0; i < n; i++)
        {
            float expected = MathF.Exp(input[i] + offset);
            referenceSum += expected;

            float relError = MathF.Abs(actual[i] - expected) / expected;
            Assert.True(relError < 1e-5f,
                $"Element {i}: actual = {actual[i]}, expected = {expected}, relative error = {relError:E3}.");
        }

        float sumRelError = (float)(Math.Abs(actualSum - referenceSum) / referenceSum);
        Assert.True(sumRelError < 1e-5f,
            $"Sum: actual = {actualSum}, expected = {referenceSum}, relative error = {sumRelError:E3}.");
    }

    /// <summary>
    /// The online softmax in <c>Attention</c> seeds its running max with
    /// <see cref="float.NegativeInfinity"/> and takes <c>FastExp(maxSoFar - newMax)</c> on the
    /// first tile. That must be a finite ~0, not a NaN.
    /// </summary>
    [Fact]
    public void ExpSumAndStore_NegativeInfinityOffset_ProducesFiniteZero()
    {
        float[] input = [0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] output = new float[input.Length];

        float sum = FastMath.ExpSumAndStore(input, output, float.NegativeInfinity);

        Assert.True(float.IsFinite(sum), $"Sum was {sum}.");
        Assert.All(output, v => Assert.True(float.IsFinite(v) && v >= 0f && v < 1e-30f, $"Got {v}."));
        Assert.True(float.IsFinite(FastMath.FastExp(float.NegativeInfinity)));
    }
}
