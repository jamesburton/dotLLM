using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Configuration;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Evaluation;

/// <summary>
/// <c>--docs-file</c> (#1706) on the real cached SmolLM-135M Q8_0: a single-document file must reproduce the existing
/// teacher-forced mean NLL, and the output must not depend on the CPU thread count.
/// </summary>
[Collection("SmallModel")]
public sealed class DocsFileRealModelTests
{
    private static readonly string[] Texts =
    [
        "The capital of France is Paris, and the capital of Germany is Berlin.",
        "In 1969 humans first walked on the Moon, a feat that took years of preparation.",
    ];

    private readonly SmallModelFixture _fixture;

    public DocsFileRealModelTests(SmallModelFixture fixture) => _fixture = fixture;

    private (IModel Model, GgufFile Gguf, BpeTokenizer Tok) Load(int threads)
    {
        var (model, gguf, _) = ModelLoader.LoadFromGguf(_fixture.FilePath, new ThreadingConfig(threads));
        return (model, gguf, GgufBpeTokenizerFactory.Load(gguf.Metadata));
    }

    [Fact]
    public void SingleDocument_NllMatchesTheExistingTeacherForcedMean()
    {
        var (model, gguf, tok) = Load(threads: 1);
        using (gguf)
        using (model)
        {
            int[] ids = tok.Encode(Texts[0]);
            Assert.True(ids.Length > 8);
            var pm = new BackendPerplexityModel(model, -1, returnsAllRows: true);

            PerplexityResult existing = PerplexityEvaluator.Evaluate(
                pm, ids, new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: ids.Length, Stride: ids.Length));

            var docs = DocumentEvaluator.Run(pm, [ids], default, new StringWriter());

            Assert.Equal(existing.ScoredTokens, docs[0].Score.TokensScored);
            Assert.Equal(ids.Length - 1, docs[0].Score.TokensScored);
            // The old path forms (logit - max) in float, the new one in double: agreement is to float rounding, not bit-exact.
            Assert.Equal(existing.MeanNegativeLogLikelihood, docs[0].Score.NllMean!.Value, 5);
            Assert.InRange(docs[0].Score.NllMean!.Value, 0.5, 12.0);   // a trained model, not noise
        }
    }

    [Fact]
    public void Output_IsIdenticalAcrossCpuThreadCounts()
    {
        string Score(int threads)
        {
            var (model, gguf, tok) = Load(threads);
            using (gguf)
            using (model)
            {
                var pm = new BackendPerplexityModel(model, -1, returnsAllRows: true);
                var sw = new StringWriter();
                DocumentEvaluator.Run(pm, Texts.Select(t => tok.Encode(t)), default, sw);
                return sw.ToString();
            }
        }

        string one = Score(1);
        Assert.Equal(2, one.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length);
        Assert.Equal(one, Score(4));
    }
}
