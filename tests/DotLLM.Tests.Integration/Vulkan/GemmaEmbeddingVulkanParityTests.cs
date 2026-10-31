using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1472: EmbeddingGemma-300M on the Vulkan encoder (<see cref="VulkanGemmaEmbeddingModel"/>: F16 activations, BF16 residual stream,
/// Q8_0 weights read in place, symmetric-window flash attention, dense heads on the device) against the CPU encoder
/// (<see cref="TransformerModel"/> bidirectional path, itself validated against sentence-transformers) on the 103-text #1399 set.
/// The CPU quantises its activations to Q8_1, so the gate is the Q8_0 class (0.998), not 0.9999. Control arms: the window ignored
/// (the nine &gt;257-token texts fall to ~0.9886 on the CPU) and a causal mask must each be DETECTED by the same comparison.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class GemmaEmbeddingVulkanParityTests(ITestOutputHelper output)
{
    private const string Description = "EmbeddingGemma-300M Q8_0 GGUF (gemma-embedding)";
    private const double Gate = 0.998;

    private static FixtureLocation ResolveModel() => TestFixtureResolver.ResolveFile(
        "DOTLLM_EMBEDDINGGEMMA_Q8_GGUF", "ggml-org", "embeddinggemma-300M-GGUF", "embeddinggemma-300M-Q8_0.gguf");

    private static (int[][] Ids, float[][] Vectors) LoadReference()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            AppContext.BaseDirectory, "Fixtures", "Embeddings", "st-embeddinggemma-300m.json")));
        var r = doc.RootElement;
        return (
            r.GetProperty("ids").EnumerateArray().Select(a => a.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray(),
            r.GetProperty("vectors").EnumerateArray().Select(v => v.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray());
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    private static float[][] Embed(IEmbeddingModel emb, IReadOnlyList<int[]> docs, int hidden)
    {
        var vecs = new float[docs.Count][];
        emb.ForwardHiddenForPooling(docs, PoolingType.Mean, (i, rows) =>
        {
            Assert.Equal(hidden, rows.Length);
            var v = rows.ToArray();
            EmbeddingPooler.L2Normalize(v);
            vecs[i] = v;
        }, 0);
        return vecs;
    }

    private static unsafe float[][] EmbedCpu(IEmbeddingModel emb, IReadOnlyList<int[]> docs, int hidden)
    {
        var vecs = new float[docs.Count][];
        for (int i = 0; i < docs.Count; i++)
        {
            var tokens = docs[i];
            var vec = new float[hidden];
            using (var h = emb.ForwardHidden(tokens, Enumerable.Range(0, tokens.Length).ToArray(), 0))
                EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)h.DataPointer, tokens.Length * hidden), tokens.Length, hidden, PoolingType.Mean, vec);
            EmbeddingPooler.L2Normalize(vec);
            vecs[i] = vec;
        }
        return vecs;
    }

    private sealed class Ctx : IDisposable
    {
        public required GgufFile Gguf;
        public required ModelConfig Config;
        public required TransformerModel Cpu;
        public required VulkanGemmaEmbeddingModel Vk;
        public required int[][] Docs;
        public required float[][] St;
        public void Dispose() { Vk.Dispose(); Cpu.Dispose(); Gguf.Dispose(); }
    }

    private static Ctx Load()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        var (ids, st) = LoadReference();
        var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var cpu = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => TransformerModel.LoadFromGguf(gguf, config));
        var vk = CheckpointGuard.LoadOrSkip(loc.Path!, Description, () => VulkanGemmaEmbeddingModel.LoadFromGguf(gguf, config, VulkanModelLoader.ResolveSpvDir()));
        return new Ctx { Gguf = gguf, Config = config, Cpu = cpu, Vk = vk, Docs = ids, St = st };
    }

    private (double Min, double Mean, double MinLong) Compare(float[][] got, float[][] want, int[][] docs, string label)
    {
        double min = 1, sum = 0, minLong = 1;
        for (int i = 0; i < got.Length; i++)
        {
            double c = Cosine(got[i], want[i]);
            min = Math.Min(min, c); sum += c;
            if (docs[i].Length > 257) minLong = Math.Min(minLong, c);
        }
        output.WriteLine($"{label}: min {min:F6} mean {sum / got.Length:F6} min over >257-token texts {minLong:F6}");
        return (min, sum / got.Length, minLong);
    }

    [SkippableFact]
    public void Packed_vulkan_gemma_matches_cpu_and_sentence_transformers()
    {
        using var c = Load();
        int h = c.Config.HiddenSize;
        Assert.True(c.Docs.Count(d => d.Length > 257) >= 1, "the fixture must include >257-token texts that exercise the symmetric window");

        var cpu = EmbedCpu(c.Cpu, c.Docs, h);
        var packed = Embed(c.Vk, c.Docs, h);
        var vsCpu = Compare(packed, cpu, c.Docs, "Vulkan packed vs CPU");
        Compare(packed, c.St, c.Docs, "Vulkan packed vs sentence-transformers");
        Compare(cpu, c.St, c.Docs, "CPU vs sentence-transformers (floor)");
        Assert.True(vsCpu.Min >= Gate, $"min Vulkan-vs-CPU cosine {vsCpu.Min:F6} < {Gate}");

        // A pack of one and a reversed pack (different neighbours/offsets) give the same vectors.
        for (int i = 0; i < c.Docs.Length; i += 7)
        {
            var single = Embed(c.Vk, [c.Docs[i]], h)[0];
            Assert.True(Cosine(single, packed[i]) >= 0.9999, $"text[{i}] ({c.Docs[i].Length} tokens): the vector depends on the packed neighbours");
        }
        var rev = Embed(c.Vk, c.Docs.Reverse().ToArray(), h);
        for (int i = 0; i < c.Docs.Length; i++)
            Assert.True(Cosine(packed[i], rev[c.Docs.Length - 1 - i]) >= 0.9999, $"text[{i}]: reversed pack order changes the vector");
    }

    [SkippableFact]
    public void Control_arms_window_ignored_causal_and_leak_are_detected()
    {
        using var c = Load();
        int h = c.Config.HiddenSize;
        var cpu = EmbedCpu(c.Cpu, c.Docs, h);

        var ok = Compare(Embed(c.Vk, c.Docs, h), cpu, c.Docs, "baseline");
        Assert.True(ok.Min >= Gate);

        c.Vk.WindowIgnoredMutant = true;
        var noWindow = Compare(Embed(c.Vk, c.Docs, h), cpu, c.Docs, "window ignored (mutant)");
        c.Vk.WindowIgnoredMutant = false;
        Assert.True(noWindow.MinLong < ok.MinLong - 0.002 && noWindow.MinLong < 0.996,
            $"the >257-token texts must expose a missing window: mutant {noWindow.MinLong:F6} vs baseline {ok.MinLong:F6}");

        c.Vk.CausalMutant = true;
        var causal = Compare(Embed(c.Vk, c.Docs, h), cpu, c.Docs, "causal mask (mutant)");
        c.Vk.CausalMutant = false;
        Assert.True(causal.Min < 0.98, $"a causal mask must be detected (min {causal.Min:F6})");

        c.Vk.PackLeakMutant = true;
        var leak = Compare(Embed(c.Vk, c.Docs, h), cpu, c.Docs, "cross-document leak (mutant)");
        c.Vk.PackLeakMutant = false;
        Assert.True(leak.Min < 0.99, $"cross-document attention must be detected (min {leak.Min:F6})");

        c.Vk.NoQkNormMutant = true;
        var noQk = Compare(Embed(c.Vk, c.Docs, h), cpu, c.Docs, "no QK-norm (mutant)");
        c.Vk.NoQkNormMutant = false;
        Assert.True(noQk.Min < 0.99, $"a missing QK-norm must be detected (min {noQk.Min:F6})");
    }

    [SkippableFact]
    public void Non_mean_pooling_is_refused()
    {
        using var c = Load();
        Assert.Throws<NotSupportedException>(() => c.Vk.ForwardHiddenForPooling([[2, 3, 4]], PoolingType.Cls, (_, _) => { }, 0));
        Assert.Throws<NotSupportedException>(() => c.Vk.ForwardHidden([2, 3, 4], [0, 1, 2], 0));
    }
}
