using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// EmbeddingGemma-300M (GGUF arch <c>gemma-embedding</c>, issue #1399) against sentence-transformers fp32
/// (<c>Fixtures/Embeddings/st-embeddinggemma-300m.json</c>: Transformer, mean Pooling, Dense 768-3072, Dense 3072-768, Normalize).
/// Two fixture texts (384 and 752 tokens) exceed the 257-token span of the symmetric sliding window, so the window
/// rule is exercised; with the window disabled the same texts fall to 0.9886 (measured, 9 texts &gt; 257 tokens in the
/// 103-text set).
/// </summary>
/// <remarks>
/// The GGUF is the Q8_0 <c>ggml-org/embeddinggemma-300M-GGUF</c> file (it carries the dense heads). Q8_0 weights and
/// Q8_0 activations cost cosine: llama.cpp b9016 on the same file measures 0.99881 min / 0.99961 mean against
/// sentence-transformers over 103 texts and dotLLM 0.99885 / 0.99971, so the gate is the quantisation floor, not 0.9999.
/// The 0.9999 gate was met on an F32 GGUF (heads added from the sentence-transformers modules): min and mean 1.0000000,
/// 103 texts, NN-1 agreement 1.00 (see docs/EMBEDDINGS.md).
/// </remarks>
public sealed class EmbeddingGemmaParityTests(ITestOutputHelper output)
{
    private const string Description = "EmbeddingGemma-300M Q8_0 GGUF (gemma-embedding)";
    private const double Q8Gate = 0.998;

    private static FixtureLocation ResolveModel() => TestFixtureResolver.ResolveFile(
        "DOTLLM_EMBEDDINGGEMMA_Q8_GGUF", "ggml-org", "embeddinggemma-300M-GGUF", "embeddinggemma-300M-Q8_0.gguf");

    private static (string[] Texts, int[][] Ids, float[][] Vectors) LoadReference()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            AppContext.BaseDirectory, "Fixtures", "Embeddings", "st-embeddinggemma-300m.json")));
        var r = doc.RootElement;
        return (
            r.GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray(),
            r.GetProperty("ids").EnumerateArray().Select(a => a.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray(),
            r.GetProperty("vectors").EnumerateArray().Select(v => v.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray());
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    [SkippableFact]
    public void Config_is_a_bidirectional_gemma3_tower_with_mean_pooling()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        using var gguf = GgufFile.Open(loc.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(DotLLM.Core.Configuration.Architecture.GemmaEmbedding, config.Architecture);
        Assert.True(config.BidirectionalAttention);
        Assert.Equal(PoolingType.Mean, config.PoolingType);
        Assert.Equal(24, config.NumLayers);
        Assert.Equal(768, config.HiddenSize);
        // 5 windowed layers then 1 global, repeating (sliding_window_pattern default 6), window 512.
        Assert.Equal(512, config.SlidingWindowOf(0));
        Assert.Null(config.SlidingWindowOf(5));
        Assert.Equal(512, config.SlidingWindowOf(6));
    }

    [SkippableFact]
    public void Tokens_and_embeddings_match_sentence_transformers()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        var (texts, ids, refs) = LoadReference();

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => TransformerModel.LoadFromGguf(gguf, config));
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        var emb = (IEmbeddingModel)model;
        var pooling = EmbeddingPooler.Resolve(null, emb.DeclaredPoolingType);
        Assert.Equal(PoolingType.Mean, pooling);

        double worst = 1;
        for (int i = 0; i < texts.Length; i++)
        {
            int[] tokens = EmbeddingPooler.AppendEos(tokenizer.Encode(texts[i]), emb.DeclaredPoolingType, emb.AppendsEosForLastPooling, tokenizer.EosTokenId, null);
            Assert.True(ids[i].SequenceEqual(tokens), $"text[{i}] token ids differ from sentence-transformers (<bos> ... <eos> framing)");
            double cos = Cosine(Run(emb, tokens, config.HiddenSize, pooling), refs[i]);
            worst = Math.Min(worst, cos);
            output.WriteLine($"text[{i}] ({tokens.Length} tok): cosine {cos:F6}");
            Assert.True(cos >= Q8Gate, $"text[{i}] ({tokens.Length} tokens): cosine {cos:F6} < {Q8Gate}");
        }
        output.WriteLine($"worst cosine {worst:F6}");
    }

    [SkippableFact]
    public void Causal_forward_is_refused()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => TransformerModel.LoadFromGguf(gguf, config));
        Assert.Throws<NotSupportedException>(() => model.Forward([2, 3, 4], [0, 1, 2], 0, null));
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
