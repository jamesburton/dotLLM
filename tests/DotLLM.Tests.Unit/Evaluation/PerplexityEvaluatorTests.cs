using DotLLM.Core.Evaluation;
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class PerplexityEvaluatorTests
{
    // Vocab must exceed every token id used as a scoring target.
    private const int Vocab = 64;
    private static readonly int[] Tokens = Enumerable.Range(0, 32).ToArray();

    [Fact]
    public void TeacherForced_AllRowsBackend_UniformLogitsGivesVocabSizePerplexity()
    {
        // Uniform logits => P(any target) = 1/vocab => perplexity == vocab, exactly.
        using var model = new FakePerplexityModel(
            vocabSize: Vocab, maxContextLength: 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: 32, Stride: 32));

        Assert.Equal(Vocab, result.Perplexity, 9);
        Assert.Equal(31, result.ScoredTokens);   // n-1 targets from one pass
    }

    [Fact]
    public void TeacherForced_AllRowsBackend_UsesASingleForwardPass()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        // The whole point of ReturnsAllRows: one pass scores every target.
        Assert.Single(model.ForwardCalls);
        Assert.Equal(32, model.ForwardCalls[0].Length);
    }

    [Fact]
    public void MeanNll_AndPerplexity_AreConsistent()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        Assert.Equal(result.Perplexity, Math.Exp(result.MeanNegativeLogLikelihood), 9);
    }

    [Fact]
    public void TeacherForced_LastRowOnlyBackend_MatchesAllRowsBackendExactly()
    {
        // Position-dependent but deterministic logits, so a wrong row/position mapping shows up.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            for (int j = 0; j < vocab; j++) row[j] = (float)Math.Sin((position + 1) * (j + 1) * 0.37);
            return row;
        }

        using var allRows = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, Rows);
        using var lastRow = new FakePerplexityModel(Vocab, 64, returnsAllRows: false, Rows);
        var options = new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32);

        var a = PerplexityEvaluator.Evaluate(allRows, Tokens, options);
        var b = PerplexityEvaluator.Evaluate(lastRow, Tokens, options);

        Assert.Equal(a.Perplexity, b.Perplexity, 9);
        Assert.Equal(a.ScoredTokens, b.ScoredTokens);
    }

    [Fact]
    public void TeacherForced_LastRowOnlyBackend_ReprefixesGrowingWindows()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: false, FakePerplexityModel.Uniform);

        PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        // One forward per scored target, each one token longer than the last.
        Assert.Equal(31, model.ForwardCalls.Count);
        for (int i = 0; i < model.ForwardCalls.Count; i++)
            Assert.Equal(i + 1, model.ForwardCalls[i].Length);
    }

    [Fact]
    public void SlidingWindow_TilesScoredTokensWithoutGapsOrOverlap()
    {
        var tokens = Enumerable.Range(0, 40).ToArray();
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        // L=16, S=8 => windows start at 0, 8, 16, 24; each scores its last 8 targets.
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, ContextLength: 16, Stride: 8));

        Assert.Equal(4, result.WindowCount);
        Assert.Equal(32, result.ScoredTokens);        // 4 windows x 8 targets
        Assert.Equal(Vocab, result.Perplexity, 9);    // uniform logits

        Assert.Equal(4, model.ForwardCalls.Count);
        Assert.All(model.ForwardCalls, w => Assert.Equal(16, w.Length));
        Assert.Equal(0, model.ForwardCalls[0][0]);
        Assert.Equal(8, model.ForwardCalls[1][0]);
        Assert.Equal(16, model.ForwardCalls[2][0]);
        Assert.Equal(24, model.ForwardCalls[3][0]);
    }

    [Fact]
    public void SlidingWindow_ScoresEachTargetAtItsTrueAbsolutePosition()
    {
        // Logits keyed to absolute position: a window that restarts positions at zero scores
        // different values and fails this.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            row[position % vocab] = 10f;
            return row;
        }

        // Row at absolute position p predicts token p+1, and argmax(row_p) == p % Vocab.
        // So make tokens[p+1] == p % Vocab, i.e. tokens[i] == (i-1) mod Vocab.
        var tokens = new int[40];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = (i + Vocab - 1) % Vocab;

        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, Rows);
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, 16, 8));

        Assert.True(result.MeanNegativeLogLikelihood < 0.01,
            $"expected confident predictions, got mean NLL {result.MeanNegativeLogLikelihood}");
    }

    [Fact]
    public void SlidingWindow_RejectsStrideNotSmallerThanContext()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);
        Assert.Throws<ArgumentException>(() => PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, ContextLength: 16, Stride: 16)));
    }
}
