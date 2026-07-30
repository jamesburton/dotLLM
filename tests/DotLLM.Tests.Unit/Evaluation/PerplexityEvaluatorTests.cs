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
}
