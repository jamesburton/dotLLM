using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// #1398: Qwen3-Embedding (decoder, last-token pooling) must have the tokenizer EOS appended before pooling.
/// Reference = sentence-transformers fp32 CPU (<c>Fixtures/Embeddings/st-qwen3-embedding-0.6b.json</c>, which
/// appends <c>&lt;|endoftext|&gt;</c> itself). Measured in #1369: with EOS cosine 1.000000, without it
/// mean 0.804 / min 0.466 — the two arms are separated by a wide margin, so the gate discriminates.
/// </summary>
public sealed class Qwen3EmbeddingEosParityTests
{
    private const string Description = "Qwen3-Embedding-0.6B F16 GGUF";
    private static FixtureLocation ResolveModel() => TestFixtureResolver.ResolveFile(
        "DOTLLM_QWEN3_EMBEDDING_06B_F16_GGUF", "Qwen", "Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-f16.gguf");

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    [SkippableFact]
    public void Default_path_appends_eos_and_matches_sentence_transformers()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        using var refDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            AppContext.BaseDirectory, "Fixtures", "Embeddings", "st-qwen3-embedding-0.6b.json")));
        var texts = refDoc.RootElement.GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var refs = refDoc.RootElement.GetProperty("vectors").EnumerateArray()
            .Select(v => v.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => TransformerModel.LoadFromGguf(gguf, config));
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        var emb = (IEmbeddingModel)model;
        var pooling = EmbeddingPooler.Resolve(null, emb.DeclaredPoolingType);
        Assert.Equal(PoolingType.Last, pooling);

        int hidden = config.HiddenSize;
        for (int i = 0; i < texts.Length; i++)
        {
            int[] raw = tokenizer.Encode(texts[i]).ToArray();
            int[] withEos = EmbeddingPooler.AppendEos(raw, emb.DeclaredPoolingType, emb.AppendsEosForLastPooling, tokenizer.EosTokenId, null);
            Assert.Equal(raw.Length + 1, withEos.Length);

            double cosWith = Cosine(Run(emb, withEos, hidden, pooling), refs[i]);
            double cosWithout = Cosine(Run(emb, raw, hidden, pooling), refs[i]);
            Assert.True(cosWith >= 0.9999, $"text[{i}] with EOS: cosine {cosWith:F6} < 0.9999");
            Assert.True(cosWithout < 0.99, $"text[{i}] without EOS: cosine {cosWithout:F6} should be clearly wrong (control arm inert)");
        }
    }

    private static unsafe float[] Run(IEmbeddingModel emb, int[] tokens, int hidden, PoolingType pooling)
    {
        var positions = Enumerable.Range(0, tokens.Length).ToArray();
        var vec = new float[hidden];
        using (var h = emb.ForwardHidden(tokens, positions, 0))
            EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)h.DataPointer, tokens.Length * hidden), tokens.Length, hidden, pooling, vec);
        EmbeddingPooler.L2Normalize(vec);
        return vec;
    }
}
