using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #741: the Vulkan BERT encoder (<see cref="VulkanBertEncoderModel"/>) must produce the vectors of the CPU
/// <see cref="BertEncoderModel"/> (itself validated against llama.cpp and sentence-transformers, docs/EMBEDDINGS.md):
/// cosine &gt;= 0.9999 per text for the whole packed batch, for one-at-a-time forwards, for CLS and mean pooling, and for the all-row
/// hidden state. Control arms: unrelated texts must be clearly separated, a causal-instead-of-bidirectional mask and a
/// cross-document attention leak must each be DETECTED (the same comparison must fail), so the gate can disagree.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class BertEncoderVulkanParityTests
{
    public static IEnumerable<object[]> Models() =>
    [
        ["bge-small", "bge-small-en-v1.5-gguf", "bge-small-en-v1.5_fp16.gguf"],
        ["e5-small-v2", "e5-small-v2-gguf", "e5-small-v2_fp16.gguf"],
        ["bge-base", "bge-base-en-v1.5-gguf", "bge-base-en-v1.5_fp16.gguf"],
    ];

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
        string.Join(' ', Enumerable.Repeat("Retrieval augmented generation embeds document chunks and ranks them by cosine similarity.", 40)),   // truncated to 512 below
        "Mixed-length neighbours exercise the per-document attention slices, the position restart and the pooled-row gather.",
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

    private sealed class Ctx : IDisposable
    {
        public required GgufFile Gguf;
        public required ModelConfig Config;
        public required BertEncoderModel Cpu;
        public required VulkanBertEncoderModel Vk;
        public required int[][] Docs;
        public required PoolingType Pooling;
        public void Dispose() { Vk.Dispose(); Cpu.Dispose(); Gguf.Dispose(); }
    }

    private static Ctx? Load(string repo, string file, string description)
    {
        var loc = TestFixtureResolver.ResolveFile($"DOTLLM_{repo.Replace('-', '_').ToUpperInvariant()}_F16_GGUF", "ChristianAzinn", repo, file);
        Skip.If(!loc.Found, loc.SkipMessage(description));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var cpu = CheckpointGuard.LoadOrSkip(loc.Path!, description, () => BertEncoderModel.LoadFromGguf(gguf, config, new ThreadingConfig(8, 8)));
        var vk = CheckpointGuard.LoadOrSkip(loc.Path!, description, () => VulkanBertEncoderModel.LoadFromGguf(gguf, config, VulkanModelLoader.ResolveSpvDir()));
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        var docs = Texts.Select(t =>
        {
            var ids = tokenizer.Encode(t).ToArray();
            if (ids.Length > config.MaxSequenceLength) { var cut = ids[..config.MaxSequenceLength]; cut[^1] = ids[^1]; ids = cut; }
            return ids;
        }).ToArray();
        var pooling = EmbeddingPooler.Resolve(null, ((IEmbeddingModel)vk).DeclaredPoolingType);
        return new Ctx { Gguf = gguf, Config = config, Cpu = cpu, Vk = vk, Docs = docs, Pooling = pooling };
    }

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void Packed_vulkan_encoder_matches_cpu_encoder(string name, string repo, string file)
    {
        using var c = Load(repo, file, $"{name} F16 GGUF")!;
        IEmbeddingModel cpu = c.Cpu, vk = c.Vk;
        int h = c.Config.HiddenSize;
        Assert.True(c.Docs.Any(d => d.Length > 400), "the fixture must include a long document");

        foreach (var pooling in new[] { PoolingType.Cls, PoolingType.Mean })
        {
            var want = Embed(cpu, c.Docs, h, pooling);
            var packed = Embed(vk, c.Docs, h, pooling);                                   // one packed forward of all texts
            for (int i = 0; i < c.Docs.Length; i++)
            {
                double cos = Cosine(want[i], packed[i]);
                Assert.True(cos >= 0.9999, $"{name} {pooling} text[{i}] ({c.Docs[i].Length} tokens): packed Vulkan vs CPU cosine {cos:F6} < 0.9999");
                var single = Embed(vk, [c.Docs[i]], h, pooling)[0];                       // a pack of one
                double cs = Cosine(want[i], single);
                Assert.True(cs >= 0.9999, $"{name} {pooling} text[{i}]: single Vulkan vs CPU cosine {cs:F6} < 0.9999");
                Assert.True(Cosine(single, packed[i]) >= 0.99999, $"{name} {pooling} text[{i}]: the vector depends on the packed neighbours");
            }
            // Reversed order -> different neighbours and offsets, same vectors.
            var rev = Embed(vk, c.Docs.Reverse().ToArray(), h, pooling);
            for (int i = 0; i < c.Docs.Length; i++)
                Assert.True(Cosine(packed[i], rev[c.Docs.Length - 1 - i]) >= 0.99999, $"{name} {pooling} text[{i}]: reversed pack order changes the vector");
            // Control arm: the comparison can disagree.
            Assert.True(Cosine(packed[1], packed[8]) < 0.95, $"{name} {pooling}: unrelated texts must not look identical");
        }
    }

    [SkippableFact]
    public void Activation_scratch_is_f16_not_f32()
    {
        // #1474 / docs/PRECISION.md rule 1: activations are stored half-width. 7 [cap,h] + 1 [cap,ff] + host staging, 2 bytes each.
        using var c = Load("bge-small-en-v1.5-gguf", "bge-small-en-v1.5_fp16.gguf", "bge-small F16 GGUF")!;
        long cap = Math.Max(VulkanBertEncoderModel.PackTokenBudget, c.Config.MaxSequenceLength);
        long f16 = (8 * cap * c.Config.HiddenSize + cap * c.Config.IntermediateSize) * sizeof(ushort);
        long f32 = (8 * cap * c.Config.HiddenSize + cap * c.Config.IntermediateSize) * sizeof(float);
        Assert.Equal(f16, c.Vk.ComputeMemoryBytes);
        Assert.True(c.Vk.ComputeMemoryBytes * 2 == f32, "F16 scratch must be exactly half of the F32 layout it replaced");
    }

    [SkippableTheory]
    [InlineData("bge-small", "bge-small-en-v1.5-gguf", "bge-small-en-v1.5_fp16.gguf")]
    public unsafe void All_row_hidden_state_matches_cpu(string name, string repo, string file)
    {
        using var c = Load(repo, file, $"{name} F16 GGUF")!;
        int h = c.Config.HiddenSize;
        foreach (int i in new[] { 1, 4, 9, 10 })
        {
            var ids = c.Docs[i];
            var pos = Enumerable.Range(0, ids.Length).ToArray();
            using var a = ((IEmbeddingModel)c.Cpu).ForwardHidden(ids, pos, 0);
            using var b = ((IEmbeddingModel)c.Vk).ForwardHidden(ids, pos, 0);
            var ha = new ReadOnlySpan<float>((void*)a.DataPointer, ids.Length * h);
            var hb = new ReadOnlySpan<float>((void*)b.DataPointer, ids.Length * h);
            double worst = 1;
            for (int t = 0; t < ids.Length; t++)
                worst = Math.Min(worst, Cosine(ha.Slice(t * h, h).ToArray(), hb.Slice(t * h, h).ToArray()));
            Assert.True(worst >= 0.9999, $"{name} text[{i}] ({ids.Length} tokens): worst per-token hidden cosine {worst:F6} < 0.9999");
        }
    }

    [SkippableTheory]
    [InlineData("bge-small", "bge-small-en-v1.5-gguf", "bge-small-en-v1.5_fp16.gguf")]
    public void Wrong_masks_are_detected(string name, string repo, string file)
    {
        using var c = Load(repo, file, $"{name} F16 GGUF")!;
        int h = c.Config.HiddenSize;
        var pooling = PoolingType.Mean;
        var want = Embed(c.Cpu, c.Docs, h, pooling);

        // Causal attention instead of bidirectional: every document of more than one token must fall below the gate.
        c.Vk.CausalMutant = true;
        var causal = Embed(c.Vk, c.Docs, h, pooling);
        c.Vk.CausalMutant = false;
        int below = 0, multi = 0;
        for (int i = 0; i < c.Docs.Length; i++)
        {
            if (c.Docs[i].Length < 4) continue;
            multi++;
            if (Cosine(want[i], causal[i]) < 0.9999) below++;
        }
        Assert.True(below >= multi - 1, $"control: causal mask was detected on only {below}/{multi} documents");

        // Cross-document leak: one dispatch over the whole concatenation. Document 0 cannot see a leak when bidirectional... it can (it attends
        // everything), so every document must change, not only the later ones.
        c.Vk.PackLeakMutant = true;
        var leak = Embed(c.Vk, c.Docs, h, pooling);
        c.Vk.PackLeakMutant = false;
        int changed = 0;
        for (int i = 0; i < c.Docs.Length; i++)
            if (Cosine(want[i], leak[i]) < 0.9999) changed++;
        Assert.True(changed >= c.Docs.Length - 1, $"control: a cross-document leak was detected on only {changed}/{c.Docs.Length} documents");
    }
}
