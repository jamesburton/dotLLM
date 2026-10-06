using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Attribution arms for the residual difference to llama.cpp (issue #739). Each arm flips ONE
/// implementation choice and reports the worst cosine against the llama.cpp reference, so the
/// remaining gap is attributed by measurement rather than guessed.
/// </summary>
/// <remarks>
/// The tests are serial (static seams) and restore the seams in <c>finally</c>.
/// </remarks>
[Collection("BertSeams")]
public sealed class BertEmbeddingDiagnosticArmsTests(ITestOutputHelper output)
{
    private static unsafe double Worst(string key, out string pooling)
    {
        var spec = BertEmbeddingLlamaCppParityTests.SpecFor(key);
        var loc = TestFixtureResolver.ResolveFile(spec.Env, spec.Org, spec.Repo, spec.File);
        Skip.If(!loc.Found, loc.SkipMessage(spec.Description));

        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "Embeddings", spec.Fixture)));
        using var gguf = GgufFile.Open(loc.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = BertEncoderModel.LoadFromGguf(gguf, config);
        var pool = EmbeddingPooler.Resolve(null, model.DeclaredPoolingType);
        pooling = pool.ToString();

        double worst = 1;
        var inputs = doc.RootElement.GetProperty("inputs").EnumerateArray().ToArray();
        var embs = doc.RootElement.GetProperty("embeddings").EnumerateArray().ToArray();
        for (int i = 0; i < inputs.Length; i++)
        {
            int[] tokens = inputs[i].GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            float[] expected = embs[i].EnumerateArray().Select(x => x.GetSingle()).ToArray();
            using var h = model.ForwardHidden(tokens, Enumerable.Range(0, tokens.Length).ToArray(), 0);
            var v = new float[config.HiddenSize];
            EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)h.DataPointer, tokens.Length * config.HiddenSize),
                tokens.Length, config.HiddenSize, pool, v);
            EmbeddingPooler.L2Normalize(v);
            worst = Math.Min(worst, BertEmbeddingLlamaCppParityTests.Cosine(v, expected));
        }
        return worst;
    }

    [SkippableTheory]
    [InlineData("minilm-f16")]
    [InlineData("nomic-f16")]
    [InlineData("nomic-f32")]
    public void Gelu_tanh_vs_erf_against_llamacpp(string key)
    {
        try
        {
            BertEncoderModel.UseErfGelu = false;
            double tanh = Worst(key, out _);
            BertEncoderModel.UseErfGelu = true;
            double erf = Worst(key, out _);
            output.WriteLine($"{key}: worst cos vs llama.cpp  tanh-GELU={tanh:F9}  erf-GELU={erf:F9}");
            Assert.True(tanh >= 0.9999 && erf >= 0.9999);
        }
        finally { BertEncoderModel.UseErfGelu = false; }
    }

    [SkippableTheory]
    [InlineData("minilm-q8")]
    [InlineData("nomic-q8")]
    [InlineData("nomic-q4km")]
    [InlineData("mxbai-q8")]
    public void Quantized_weights_Q8_activations_vs_F32_activations(string key)
    {
        try
        {
            BertEncoderModel.ForceF32Activations = false;
            double q8act = Worst(key, out string pooling);
            BertEncoderModel.ForceF32Activations = true;
            double f32act = Worst(key, out _);
            output.WriteLine($"{key} ({pooling}): worst cos vs llama.cpp  Q8-activations={q8act:F9}  F32-activations={f32act:F9}");
        }
        finally { BertEncoderModel.ForceF32Activations = false; }
    }
}

[CollectionDefinition("BertSeams", DisableParallelization = true)]
public sealed class BertSeamsCollection;
