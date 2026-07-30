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
}
