using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// Issue #1851: the CUDA prefix + CPU tail split (<see cref="HybridTransformerModel"/>), which <c>--gpu-layers auto</c> now
/// chooses on its own, must match both references through a real <see cref="HybridKvCache"/> (prefill, then single-token
/// decode) at N = 1, L/2 and L-1:
/// <list type="bullet">
/// <item>the all-CPU model with the same Q8_0 KV the tail now uses (the tail was an FP32 cache, against rule 1);</item>
/// <item>the all-GPU <see cref="CudaTransformerModel"/>.</item>
/// </list>
/// The existing hybrid tests pass <c>kvCache: null</c>, so neither the tail KV nor its multi-token quantized prefill ran there.
/// Red arm: the same decode with the history dropped (a fresh cache per step) must fail the bar, so the bar can see a tail
/// that loses its cache. Also covers TwIL-8B's Granite shape (attention multiplier folded into RoPE on the GPU, Q pre-scaled
/// on the CPU tail), which the split host refused before.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class HybridSplitParityTests
{
    private readonly ITestOutputHelper _output;

    public HybridSplitParityTests(ITestOutputHelper output) => _output = output;

    private static readonly int[] Tokens =
        [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 40, 6, 33, 78, 14, 90, 27, 3, 52, 11, 67, 35];

    private const int PrefillLen = 18;   // then 6 single-token decode steps

    /// <summary>Cosine every step must reach against each reference.</summary>
    private const double MinCosine = 0.999;

    private static string? SmolLmPath =>
        DotLLM.Tests.Integration.Fixtures.TestFixtureResolver.ResolveFile(
            [], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path;

    private static unsafe float[] Last(ITensor logits, int vocab)
    {
        long total = logits.ElementCount;
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (total - vocab), vocab).CopyTo(row);
        return row;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>Logits after the prefill and after each decode step, through <paramref name="newCache"/>.</summary>
    private static List<float[]> Run(IModel model, Func<Core.Attention.IKvCache> newCache, int vocab, bool dropHistory = false)
    {
        var rows = new List<float[]>();
        var cache = newCache();
        try
        {
            using (ITensor l = model.Forward(Tokens.AsSpan(0, PrefillLen).ToArray(), Enumerable.Range(0, PrefillLen).ToArray(), 0, cache))
                rows.Add(Last(l, vocab));
            for (int t = PrefillLen; t < Tokens.Length; t++)
            {
                if (dropHistory)
                {
                    cache.Dispose();
                    cache = newCache();
                }
                using ITensor l = model.Forward([Tokens[t]], [dropHistory ? 0 : t], 0, cache);
                rows.Add(Last(l, vocab));
            }
        }
        finally
        {
            cache.Dispose();
        }
        return rows;
    }

    private void AssertClose(string label, List<float[]> reference, List<float[]> actual, out double worst)
    {
        worst = 1;
        Assert.Equal(reference.Count, actual.Count);
        for (int s = 0; s < reference.Count; s++)
        {
            double cos = Cosine(reference[s], actual[s]);
            worst = Math.Min(worst, cos);
            Assert.True(cos >= MinCosine, $"{label} step {s}: cosine {cos:F6} < {MinCosine}");
            Assert.Equal(ArgMax(reference[s]), ArgMax(actual[s]));
        }
        _output.WriteLine($"{label}: worst cosine {worst:F6} over prefill + {reference.Count - 1} decode steps");
    }

    private void CheckModel(string path, string name)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int vocab = config.VocabSize;
        int maxLen = Tokens.Length + 1;
        int layers = config.NumLayers;

        List<float[]> cpu;
        using (var cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.SingleThreaded))
            cpu = Run(cpuModel, () => new QuantizedKvCache(layers, config.NumKvHeads, config.HeadDim, maxLen,
                HybridTransformerModel.CpuTailKvDType, HybridTransformerModel.CpuTailKvDType, windowSize: 0), vocab);

        List<float[]> gpu;
        var (gpuModel, gpuKv) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0);
        using (gpuModel)
            gpu = Run(gpuModel, () => gpuKv(maxLen), vocab);

        foreach (int n in new[] { 1, layers / 2, layers - 1 }.Distinct())
        {
            using var hybrid = HybridTransformerModel.LoadFromGguf(gguf, config, n, deviceId: 0, ThreadingConfig.SingleThreaded);
            using (var probe = hybrid.CreateKvCache(maxLen))
            {
                // Rule 1 (#1851): the tail KV is Q8_0 blocks, not an FP32 SimpleKvCache.
                var tail = Assert.IsType<QuantizedKvCache>(probe.CpuCache);
                Assert.Equal((KvCacheDType.Q8_0, KvCacheDType.Q8_0), (tail.KeyDType, tail.ValueDType));
            }
            var split = Run(hybrid, () => hybrid.CreateKvCache(maxLen), vocab);
            Assert.Equal(0, hybrid.CheckSpill($"after the N={n} parity run"));   // WDDM: no device buffer backed by system RAM
            AssertClose($"{name} N={n}/{layers} vs CPU (q8_0 KV)", cpu, split, out _);
            AssertClose($"{name} N={n}/{layers} vs all-GPU", gpu, split, out _);

            // Red arm: the same bar must reject a tail that has lost its history.
            var amnesic = Run(hybrid, () => hybrid.CreateKvCache(maxLen), vocab, dropHistory: true);
            double worstRed = Enumerable.Range(1, cpu.Count - 1).Min(s => Cosine(cpu[s], amnesic[s]));
            _output.WriteLine($"{name} N={n}: red arm (history dropped) worst cosine {worstRed:F6}");
            Assert.True(worstRed < MinCosine, $"{name} N={n}: a cacheless decode still reached cosine {worstRed:F6}; the bar cannot see a lost KV");
        }
    }

    [SkippableFact]
    public void SmolLm_SplitMatchesCpuAndGpu_ThroughTheHybridKvCache()
    {
        Skip.If(!File.Exists(SmolLmPath), "SmolLM-135M Q8_0 GGUF not found.");
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        CheckModel(SmolLmPath!, "SmolLM-135M");
    }

    [SkippableFact]
    public void GpuLayersBeyondFreeMemory_AreRefusedBeforeUpload_NotSpilled()
    {
        Skip.If(!File.Exists(SmolLmPath), "SmolLM-135M Q8_0 GGUF not found.");
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        using var gguf = GgufFile.Open(SmolLmPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        long refusals = CudaAdmission.RefusalCount;
        // Free memory as admission sees it: below the reserve, so no layer's weights fit. Before #1851 the hybrid uploaded
        // without asking, and WDDM backed what did not fit with shared system memory.
        CudaAdmission.FreeBytesOverride = () => 64L << 20;
        try
        {
            var ex = Assert.Throws<DeviceMemoryAdmissionException>(() =>
                HybridTransformerModel.LoadFromGguf(gguf, config, config.NumLayers - 1, deviceId: 0, ThreadingConfig.SingleThreaded));
            Assert.Contains("GPU layers", ex.Message);
            Assert.Contains("--gpu-layers auto", ex.Message);
            Assert.True(CudaAdmission.RefusalCount > refusals);
        }
        finally
        {
            CudaAdmission.FreeBytesOverride = null;
        }
    }

    [SkippableFact]
    public void FoldedGraniteScalars_TwIlShape_SplitMatchesCpuAndGpu()
    {
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        string path = Path.Combine(Path.GetTempPath(), $"syn_granite_split_{Guid.NewGuid():N}.gguf");
        // TwIL-8B / Granite-4.2: only attention.scale is set (1/head_dim instead of 1/sqrt(head_dim)).
        SyntheticGraniteGguf.Write(path, new SyntheticGraniteConfig
            { Hidden = 256, Heads = 4, KvHeads = 2, Layers = 3, EmbeddingScale = 1f, ResidualScale = 1f, LogitScale = 1f, AttentionScale = 1f / 64, QkWeightScale = 1f });
        try
        {
            using (var gguf = GgufFile.Open(path))
                Assert.True(GgufModelConfigExtractor.Extract(gguf.Metadata).GraniteScalarsFoldIntoRope);
            CheckModel(path, "granite-folded");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }
}
