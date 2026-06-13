using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Evaluation;

/// <summary>
/// Unit tests for <see cref="PerplexityEvaluator.ComputeWindowNll"/> — the pure NLL math
/// that both the CLI <c>eval perplexity</c> command and the integration anchor test rely on.
/// </summary>
public sealed class PerplexityEvaluatorTests
{
    /// <summary>
    /// Hand-constructed closed-form case. Three asymmetric rows (vocab=4) with targets chosen
    /// so they are NOT the per-row argmax — this discriminates the shift direction: scoring
    /// <c>tokenIds[t+1]</c> from <c>logits[t]</c> (correct) gives meanNLL ≈ 0.2998 / ppl ≈ 1.3495,
    /// whereas the off-by-one "score tokenIds[t] from logits[t]" bug yields meanNLL ≈ 3.0498 — a
    /// completely different number. Expected values computed in double precision via numpy.
    /// </summary>
    [Fact]
    public void ComputeWindowNll_AsymmetricRows_MatchesClosedForm()
    {
        // 3 rows × 4 vocab, row-major.
        float[] logits =
        [
            2.0f, 1.0f, 0.1f, -1.0f,
            0.5f, 3.0f, -0.5f, 0.0f,
            1.0f, 1.0f, 2.5f, 0.2f,
        ];
        int[] tokenIds = [3, 0, 1];

        var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(logits, rows: 3, vocabSize: 4, tokenIds);

        // 3 rows → 2 scored targets (tokenIds[1]=0 from row0, tokenIds[2]=1 from row1).
        Assert.Equal(2, scored);
        Assert.Equal(0.5995154270, sumNll, precision: 6);
        Assert.Equal(0.2997577135, sumNll / scored, precision: 6);
    }

    /// <summary>
    /// A perfectly-confident model (one logit ≫ rest, and the target IS that token) yields
    /// NLL ≈ 0 and perplexity ≈ 1 — the lower bound. Confirms the log-softmax is oriented
    /// correctly (probability of the target, not its complement).
    /// </summary>
    [Fact]
    public void ComputeWindowNll_ConfidentCorrect_NearZero()
    {
        // Row 0 is overwhelmingly confident in token 1; tokenIds[1] == 1 → near-zero NLL.
        float[] logits =
        [
            0f, 100f, 0f,   // row 0 predicts token 1
            100f, 0f, 0f,   // row 1 (unscored — last row has no following target)
        ];
        int[] tokenIds = [0, 1];

        var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(logits, rows: 2, vocabSize: 3, tokenIds);

        Assert.Equal(1, scored);
        Assert.True(sumNll < 1e-6, $"Expected near-zero NLL, got {sumNll}.");
    }

    /// <summary>
    /// Uniform logits over <c>V</c> classes give per-token NLL = ln(V) regardless of target,
    /// so perplexity = V exactly. This is the canonical sanity check for cross-entropy.
    /// </summary>
    [Fact]
    public void ComputeWindowNll_UniformLogits_NllEqualsLnVocab()
    {
        const int v = 8;
        float[] logits = new float[2 * v]; // all zero → uniform
        int[] tokenIds = [3, 5]; // arbitrary; uniform NLL is target-independent

        var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(logits, rows: 2, vocabSize: v, tokenIds);

        Assert.Equal(1, scored);
        Assert.Equal(Math.Log(v), sumNll, precision: 6);
    }

    /// <summary>A single-token window scores nothing (no preceding context for any target).</summary>
    [Fact]
    public void ComputeWindowNll_SingleToken_ScoresNothing()
    {
        float[] logits = [1f, 2f, 3f];
        int[] tokenIds = [2];

        var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(logits, rows: 1, vocabSize: 3, tokenIds);

        Assert.Equal(0, scored);
        Assert.Equal(0.0, sumNll);
    }

    [Fact]
    public void ComputeWindowNll_MismatchedTokenCount_Throws()
    {
        float[] logits = new float[8];
        int[] tokenIds = [0, 1, 2]; // 3 ids but rows=2

        Assert.Throws<ArgumentException>(() =>
            PerplexityEvaluator.ComputeWindowNll(logits, rows: 2, vocabSize: 4, tokenIds));
    }

    [Fact]
    public void ComputeWindowNll_TargetOutOfRange_Throws()
    {
        float[] logits = new float[8];
        int[] tokenIds = [0, 9]; // target 9 >= vocab 4

        Assert.Throws<ArgumentException>(() =>
            PerplexityEvaluator.ComputeWindowNll(logits, rows: 2, vocabSize: 4, tokenIds));
    }
}
