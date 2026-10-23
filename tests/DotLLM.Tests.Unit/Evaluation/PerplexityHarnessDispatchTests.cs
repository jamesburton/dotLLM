using DotLLM.Core.Evaluation;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Regression coverage for issue #259: the <c>perplexity</c> command loaded its CPU model with
/// <see cref="TransformerModel.LoadFromGguf(GgufFile, DotLLM.Core.Models.ModelConfig, DotLLM.Core.Configuration.ThreadingConfig)"/>
/// directly instead of the shared per-architecture dispatch
/// (<see cref="ModelLoader.CreateCpuModelFromGguf"/>) that <c>run</c>/<c>chat</c>/<c>bench</c> and
/// the server already used. On a Gated-DeltaNet hybrid — whose GDN layers have no
/// <c>attn_output.weight</c> — that threw
/// <c>KeyNotFoundException: The given key 'blk.0.attn_output.weight' was not present</c>, so the
/// model could be generated from but never scored.
/// </summary>
/// <remarks>
/// The test reproduces the harness composition the command performs — shared dispatch →
/// <see cref="BackendPerplexityModel"/> → <see cref="PerplexityEvaluator"/> — against the tiny
/// synthetic <c>qwen35moe</c> fixture whose layer 0 is GDN. That layout is what discriminates
/// broken from fixed dispatch: routed to the plain <see cref="TransformerModel"/> loader it throws
/// before a single logit is produced.
/// </remarks>
public sealed class PerplexityHarnessDispatchTests : IDisposable
{
    private readonly string _scratch;

    public PerplexityHarnessDispatchTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-ppl-dispatch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void PerplexityHarness_GdnHybridGguf_ScoresInsteadOfThrowingOnAttnOutput()
    {
        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "qwen35moe-tiny.gguf"));

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);

        // Exactly what PerplexityCommand's CPU branch does: THE shared per-architecture dispatch.
        // Pre-fix this line was TransformerModel.LoadFromGguf and threw KeyNotFoundException.
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.IsType<Qwen3MoeHybridTransformerModel>(model);

        bool returnsAllRows = BackendPerplexityModel.Probe(model, deviceId: -1);
        var perplexityModel = new BackendPerplexityModel(model, deviceId: -1, returnsAllRows);

        // Token ids stay inside the fixture's 8-entry vocabulary; the window fits its 8-token context.
        int[] tokens = [0, 1, 2, 3, 4, 5, 6, 7];
        var result = PerplexityEvaluator.Evaluate(
            perplexityModel,
            tokens,
            new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: 8, Stride: 8));

        Assert.True(result.ScoredTokens > 0, "no targets were scored");
        Assert.True(double.IsFinite(result.Perplexity), $"non-finite perplexity: {result.Perplexity}");
        Assert.True(result.Perplexity > 0.0, $"non-positive perplexity: {result.Perplexity}");
    }
}
