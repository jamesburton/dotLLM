using ChunkKvLab;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>Quality metrics of the E1 harness (<c>benchmarks/ChunkKvLab</c>): KL, top-1 agreement, greedy exact match.</summary>
public sealed class MetricsTests
{
    [Fact]
    public void Metrics_KlOfIdenticalLogits_IsZero()
    {
        float[] logits = [2.0f, -1.0f, 0.5f, 3.25f];
        Assert.Equal(0.0, Metrics.KlDivergence(logits, logits), 12);

        // Shift invariance: softmax ignores a constant offset, so KL of (x, x + c) is zero too.
        float[] shifted = [12.0f, 9.0f, 10.5f, 13.25f];
        Assert.Equal(0.0, Metrics.KlDivergence(logits, shifted), 9);

        // A hand-computed non-zero case, so "return 0", a missing log-softmax or a reversed direction cannot pass.
        // P = softmax([0, ln 3]) = [1/4, 3/4]; Q = softmax([0, 0]) = [1/2, 1/2].
        // KL(P || Q) = 1/4 ln(1/2) + 3/4 ln(3/2) = 0.130812035...; KL(Q || P) = 1/2 ln 2 + 1/2 ln(2/3) = 0.143841036...
        float[] p = [0f, MathF.Log(3f)];
        float[] q = [0f, 0f];
        double expected = 0.25 * Math.Log(0.5) + 0.75 * Math.Log(1.5);
        Assert.Equal(expected, Metrics.KlDivergence(p, q), 6);
        Assert.Equal(0.5 * Math.Log(2) + 0.5 * Math.Log(2.0 / 3.0), Metrics.KlDivergence(q, p), 6);

        // MeanKl averages the per-row KL over rows of a flattened [rows, vocab] buffer.
        float[] refRows = [.. p, .. q, .. logits.AsSpan(0, 2)];
        float[] rows = [.. q, .. q, .. logits.AsSpan(0, 2)];
        Assert.Equal(expected / 3.0, Metrics.MeanKl(refRows, rows, vocab: 2), 6);
    }

    [Fact]
    public void Metrics_Top1Agreement_CountsMatchingArgmax()
    {
        // vocab 3, four rows; argmax ref = [2, 0, 1, 1], candidate = [2, 1, 1, 0] -> rows 0 and 2 agree -> 0.5.
        float[] reference =
        [
            0.1f, 0.2f, 0.9f,
            5.0f, 1.0f, 4.0f,
            -1f, 3.0f, 2.9f,
            0.0f, 0.5f, 0.4f,
        ];
        float[] candidate =
        [
            1.0f, 2.0f, 9.0f, // argmax 2 (agrees; values differ from the reference)
            1.0f, 5.0f, 4.0f, // argmax 1 (reference 0)
            0f, 7.0f, 6.0f,   // argmax 1 (agrees)
            0.6f, 0.5f, 0.4f, // argmax 0 (reference 1)
        ];
        Assert.Equal(0.5, Metrics.Top1Agreement(reference, candidate, vocab: 3), 12);
        Assert.Equal(1.0, Metrics.Top1Agreement(reference, reference, vocab: 3), 12);
        Assert.Equal(2, Metrics.ArgMax([0.1f, 0.2f, 0.9f]));
        Assert.Equal(0, Metrics.ArgMax([0.9f, 0.2f, 0.9f])); // ties go to the lowest index
    }

    [Fact]
    public void Metrics_ExactMatch_IsMatchingPrefixFraction()
    {
        Assert.Equal(1.0, Metrics.ExactMatch([1, 2, 3, 4], [1, 2, 3, 4]), 12);
        Assert.Equal(0.5, Metrics.ExactMatch([1, 2, 3, 4], [1, 2, 9, 4]), 12); // stops at the first divergence
        Assert.Equal(0.0, Metrics.ExactMatch([1, 2, 3, 4], [7, 2, 3, 4]), 12);
    }
}
