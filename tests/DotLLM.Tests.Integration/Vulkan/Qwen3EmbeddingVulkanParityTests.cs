using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1406: the Vulkan decoder embedder (<see cref="VulkanTransformerModel"/> as <see cref="IEmbeddingModel"/>) must produce the
/// same vectors as the CPU <see cref="TransformerModel"/> path: cosine &gt;= 0.9999 per text, for the one-row last-token readback
/// (<see cref="IEmbeddingModel.ForwardHiddenForPooling"/>), the all-row <see cref="IEmbeddingModel.ForwardHidden"/> and mean pooling.
/// The sentence-transformers fp32 fixture (committed) is the external reference for the first four texts.
/// Control arms: a different text must be clearly far (the comparison can disagree), and the single-row readback must be
/// exactly the last row of the full readback (it is the same norm kernel on the same row).
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Qwen3EmbeddingVulkanParityTests
{
    private const string Description = "Qwen3-Embedding-0.6B F16 GGUF";
    private static FixtureLocation ResolveModel() => TestFixtureResolver.ResolveFile(
        "DOTLLM_QWEN3_EMBEDDING_06B_F16_GGUF", "Qwen", "Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-f16.gguf");

    private static string[] ExtraTexts() =>
    [
        "x",
        "Привет, мир! 你好，世界。 ¿Qué tal?",
        "dotLLM is a high-performance LLM inference engine written natively in C#/.NET with SIMD CPU kernels and a Vulkan backend.",
        string.Join(' ', Enumerable.Repeat("Retrieval augmented generation embeds document chunks and ranks them by cosine similarity.", 14)),
        string.Join(' ', Enumerable.Repeat("The quick brown fox jumps over the lazy dog while the model streams tokens.", 30)),
    ];

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    private static float[] Embed(IEmbeddingModel emb, int[] tokens, int hidden, PoolingType pooling)
    {
        var vec = new float[hidden];
        emb.ForwardHiddenForPooling([tokens], pooling, (_, rows) =>
        {
            Assert.Equal(0, rows.Length % hidden);
            EmbeddingPooler.Pool(rows, rows.Length / hidden, hidden, pooling, vec);
        }, 0);
        EmbeddingPooler.L2Normalize(vec);
        return vec;
    }

    [SkippableFact]
    public unsafe void Vulkan_hidden_state_matches_cpu_and_sentence_transformers()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");

        using var refDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            AppContext.BaseDirectory, "Fixtures", "Embeddings", "st-qwen3-embedding-0.6b.json")));
        var fixtureTexts = refDoc.RootElement.GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var refs = refDoc.RootElement.GetProperty("vectors").EnumerateArray()
            .Select(v => v.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();
        var texts = fixtureTexts.Concat(ExtraTexts()).ToArray();

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cpu = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => TransformerModel.LoadFromGguf(gguf, config));
        using var vk = CheckpointGuard.LoadOrSkip(loc.Path!, Description,
            () => VulkanTransformerModel.LoadFromGguf(gguf, config, VulkanModelLoader.ResolveSpvDir()));
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        IEmbeddingModel cpuEmb = cpu, vkEmb = vk;
        Assert.Equal(PoolingType.Last, vkEmb.DeclaredPoolingType);
        int hidden = config.HiddenSize;

        var vkVecs = new float[texts.Length][];
        for (int i = 0; i < texts.Length; i++)
        {
            int[] tokens = EmbeddingPooler.AppendEos(tokenizer.Encode(texts[i]).ToArray(),
                vkEmb.DeclaredPoolingType, vkEmb.AppendsEosForLastPooling, tokenizer.EosTokenId, null);
            var c = Embed(cpuEmb, tokens, hidden, PoolingType.Last);
            var v = Embed(vkEmb, tokens, hidden, PoolingType.Last);
            vkVecs[i] = v;
            double cos = Cosine(c, v);
            Assert.True(cos >= 0.9999, $"text[{i}] ({tokens.Length} tokens): Vulkan vs CPU cosine {cos:F6} < 0.9999");
            if (i < refs.Length)
            {
                double cosSt = Cosine(refs[i], v);
                Assert.True(cosSt >= 0.9999, $"text[{i}]: Vulkan vs sentence-transformers cosine {cosSt:F6} < 0.9999");
            }

            // Single-row readback == last row of the all-row readback (same norm kernel, same row).
            var positions = Enumerable.Range(0, tokens.Length).ToArray();
            using var all = vkEmb.ForwardHidden(tokens, positions, 0);
            var lastRow = new ReadOnlySpan<float>((void*)all.DataPointer, tokens.Length * hidden).Slice((tokens.Length - 1) * hidden, hidden).ToArray();
            EmbeddingPooler.L2Normalize(lastRow);
            Assert.True(Cosine(lastRow, v) >= 0.999999, $"text[{i}]: one-row readback differs from the all-row readback");

            // Mean pooling (every row comes back) must agree with the CPU too.
            double cosMean = Cosine(Embed(cpuEmb, tokens, hidden, PoolingType.Mean), Embed(vkEmb, tokens, hidden, PoolingType.Mean));
            Assert.True(cosMean >= 0.9999, $"text[{i}]: mean-pooled Vulkan vs CPU cosine {cosMean:F6} < 0.9999");
        }

        // Control arm: the metric can disagree. Different texts are clearly separated.
        Assert.True(Cosine(vkVecs[0], vkVecs[3]) < 0.95, "control: unrelated texts should not look identical");
        // Control arm 2: dropping the EOS changes the Vulkan vector (the same gate as the CPU test, #1398).
        int[] noEos = tokenizer.Encode(texts[0]).ToArray();
        Assert.True(Cosine(Embed(vkEmb, noEos, hidden, PoolingType.Last), refs[0]) < 0.99, "control: no-EOS Vulkan vector must differ from the reference");
    }

    [SkippableFact]
    public void Vulkan_embedding_rejects_unsupported_positions()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var vk = CheckpointGuard.LoadOrSkip(loc.Path!, Description,
            () => VulkanTransformerModel.LoadFromGguf(gguf, config, VulkanModelLoader.ResolveSpvDir()));
        Assert.Throws<NotSupportedException>(() => ((IEmbeddingModel)vk).ForwardHidden([1, 2, 3], [5, 6, 7], 0));
    }
}
