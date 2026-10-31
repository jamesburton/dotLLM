using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1437: several short documents packed into one Vulkan forward (padding-free varlen; one causal attention dispatch per document,
/// positions restarting per document, one pooled row per document) must give the vectors the one-document-at-a-time forward gives
/// (cosine &gt;= 0.9999, the #1406 gate), independent of the neighbours and the order. The control arm turns the attention mask
/// into one causal dispatch over the concatenation (cross-document leak) and requires the comparison to FAIL, so a wrong mask is detectable.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Qwen3EmbeddingPackedTests
{
    private const string Description = "Qwen3-Embedding-0.6B F16 GGUF";
    private static FixtureLocation ResolveModel() => TestFixtureResolver.ResolveFile(
        "DOTLLM_QWEN3_EMBEDDING_06B_F16_GGUF", "Qwen", "Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-f16.gguf");

    private static readonly string[] Texts =
    [
        "x",
        "Hello",
        "The capital of France is Paris.",
        "Привет, мир! 你好，世界。 ¿Qué tal?",
        "dotLLM is a high-performance LLM inference engine written natively in C#/.NET with SIMD CPU kernels and a Vulkan backend.",
        "Retrieval augmented generation embeds document chunks and ranks them by cosine similarity.",
        "A short query about GPUs",
        string.Join(' ', Enumerable.Repeat("The quick brown fox jumps over the lazy dog while the model streams tokens.", 6)),
        "Another document, unrelated to the previous ones, about cooking pasta al dente.",
        "Vulkan cooperative matrices",
    ];

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    private static float[][] Embed(IEmbeddingModel emb, IReadOnlyList<int[]> docs, int hidden, PoolingType pooling)
    {
        var vecs = new float[docs.Count][];
        emb.ForwardHiddenForPooling(docs, pooling, (i, rows) =>
        {
            Assert.Equal(0, rows.Length % hidden);
            var v = new float[hidden];
            EmbeddingPooler.Pool(rows, rows.Length / hidden, hidden, pooling, v);
            EmbeddingPooler.L2Normalize(v);
            vecs[i] = v;
        }, 0);
        return vecs;
    }

    private static float[][] EmbedOneByOne(IEmbeddingModel emb, IReadOnlyList<int[]> docs, int hidden, PoolingType pooling)
    {
        var vecs = new float[docs.Count][];
        for (int i = 0; i < docs.Count; i++) vecs[i] = Embed(emb, [docs[i]], hidden, pooling)[0];
        return vecs;
    }

    [SkippableFact]
    public void Packed_forward_matches_one_document_at_a_time_and_a_cross_document_leak_is_detected()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cpu = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => TransformerModel.LoadFromGguf(gguf, config));
        using var vk = CheckpointGuard.LoadOrSkip(loc.Path!, Description,
            () => VulkanTransformerModel.LoadFromGguf(gguf, config, VulkanModelLoader.ResolveSpvDir()));
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        IEmbeddingModel vkEmb = vk, cpuEmb = cpu;
        int hidden = config.HiddenSize;

        var docs = Texts.Select(t => EmbeddingPooler.AppendEos(tokenizer.Encode(t).ToArray(),
            vkEmb.DeclaredPoolingType, vkEmb.AppendsEosForLastPooling, tokenizer.EosTokenId, null)).ToList();

        // Proof the packed path is the one under test: with packing off the results are the one-at-a-time ones by construction.
        var single = EmbedOneByOne(vkEmb, docs, hidden, PoolingType.Last);
        var cpuVecs = EmbedOneByOne(cpuEmb, docs, hidden, PoolingType.Last);

        // 1. Packed (all ten in one forward) == one at a time == CPU.
        var packed = Embed(vkEmb, docs, hidden, PoolingType.Last);
        for (int i = 0; i < docs.Count; i++)
        {
            double cs = Cosine(single[i], packed[i]), cc = Cosine(cpuVecs[i], packed[i]);
            Assert.True(cs >= 0.9999, $"doc[{i}] ({docs[i].Length} tokens): packed vs single cosine {cs:F6} < 0.9999");
            Assert.True(cc >= 0.9999, $"doc[{i}]: packed vs CPU cosine {cc:F6} < 0.9999");
        }

        // 2. Independent of neighbours and order: reversed packing gives the same per-document vectors.
        var rev = Embed(vkEmb, docs.AsEnumerable().Reverse().ToList(), hidden, PoolingType.Last);
        for (int i = 0; i < docs.Count; i++)
            Assert.True(Cosine(packed[i], rev[docs.Count - 1 - i]) >= 0.9999, $"doc[{i}]: vector depends on its neighbours / position in the pack");

        // 3. Mean pooling (every row of every document comes back).
        var meanSingle = EmbedOneByOne(vkEmb, docs, hidden, PoolingType.Mean);
        var meanPacked = Embed(vkEmb, docs, hidden, PoolingType.Mean);
        for (int i = 0; i < docs.Count; i++)
            Assert.True(Cosine(meanSingle[i], meanPacked[i]) >= 0.9999, $"doc[{i}]: mean-pooled packed vs single below 0.9999");

        // 4. A pack split by the token budget (several packs of 2-3 documents) gives the same vectors.
        Environment.SetEnvironmentVariable("DOTLLM_EMBED_PACK_TOKENS", "40");
        try
        {
            var split = Embed(vkEmb, docs, hidden, PoolingType.Last);
            for (int i = 0; i < docs.Count; i++)
                Assert.True(Cosine(single[i], split[i]) >= 0.9999, $"doc[{i}]: budget-split packs differ from single");
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_EMBED_PACK_TOKENS", null); }

        // 5. Control: a cross-document leak (one causal dispatch over the concatenation) must be caught by the same comparison.
        // Document 0 attends only itself either way; every later document sees the earlier ones and must move.
        vk.EmbedPackLeakMutant = true;
        float[][] leaked;
        try { leaked = Embed(vkEmb, docs, hidden, PoolingType.Last); }
        finally { vk.EmbedPackLeakMutant = false; }
        Assert.True(Cosine(single[0], leaked[0]) >= 0.9999, "control: the first document cannot see a leak");
        int detected = 0;
        for (int i = 1; i < docs.Count; i++) if (Cosine(single[i], leaked[i]) < 0.9999) detected++;
        Assert.True(detected >= docs.Count - 2, $"control: the leak mutant moved only {detected}/{docs.Count - 1} later documents; the comparison cannot see a wrong mask");

        // Control 2: unrelated documents are clearly separated (the cosine metric can disagree).
        Assert.True(Cosine(packed[2], packed[8]) < 0.95, "control: unrelated documents should not look identical");
    }
}
