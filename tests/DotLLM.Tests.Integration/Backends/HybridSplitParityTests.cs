using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
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

    private static int[] TopK(float[] v, int k) =>
        Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(k).ToArray();

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

    /// <summary>
    /// The bar for one step: the split may be no further from a reference than <see cref="GapMultiple"/> times the distance between
    /// the two references themselves (all-CPU and all-GPU disagree by their F16-activation and KV-format differences, and a split is
    /// a mix of the two), with a floor of <see cref="MinGap"/>.
    /// </summary>
    private static double Allowed(double referenceGap) => Math.Max(GapMultiple * referenceGap, MinGap);

    private const double GapMultiple = 3.0;
    private const double MinGap = 2e-4;

    private void CheckModel(string path, string name, ThreadingConfig? threading = null, int[]? splits = null)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int vocab = config.VocabSize;
        int maxLen = Tokens.Length + 1;
        int layers = config.NumLayers;

        List<float[]> cpu, cpuF32;
        threading ??= ThreadingConfig.SingleThreaded;
        using (var cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, config, threading))
        {
            // The reference KV matches the tail's format: Q8_0, or (head_dim % 32 != 0, #1870) the F32 reference cache.
            cpu = HybridTransformerModel.TailUsesQ8_0(config.HeadDim)
                ? Run(cpuModel, () => new QuantizedKvCache(layers, config.NumKvHeads, config.HeadDim, maxLen,
                    HybridTransformerModel.CpuTailKvDType, HybridTransformerModel.CpuTailKvDType, windowSize: 0), vocab)
                : Run(cpuModel, () => new SimpleKvCache(layers, config.NumKvHeads, config.HeadDim, maxLen), vocab);
            cpuF32 = Run(cpuModel, () => new SimpleKvCache(layers, config.NumKvHeads, config.HeadDim, maxLen), vocab);
        }

        List<float[]> gpu;
        var (gpuModel, gpuKv) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0);
        using (gpuModel)
            gpu = Run(gpuModel, () => gpuKv(maxLen), vocab);

        int steps = cpu.Count;
        var gap = new double[steps];
        for (int s = 0; s < steps; s++)
        {
            gap[s] = 1 - Cosine(cpu[s], gpu[s]);
            _output.WriteLine($"{name} step {s}: 1-cos cpu(q8_0 KV) vs gpu {gap[s]:E3}, cpu(f32 KV) vs gpu {1 - Cosine(cpuF32[s], gpu[s]):E3}, " +
                              $"cpu q8_0 vs f32 KV {1 - Cosine(cpu[s], cpuF32[s]):E3}");
        }

        var failures = new List<string>();
        foreach (int n in (splits ?? [1, layers / 2, layers - 1]).Distinct())
        {
            using var hybrid = HybridTransformerModel.LoadFromGguf(gguf, config, n, deviceId: 0, threading.Value);
            using (var probe = hybrid.CreateKvCache(maxLen))
            {
                // Rule 1 (#1851): the tail KV is Q8_0 blocks, not an FP32 SimpleKvCache.
                if (HybridTransformerModel.TailUsesQ8_0(config.HeadDim))
                {
                    var tail = Assert.IsType<QuantizedKvCache>(probe.CpuCache);
                    Assert.Equal((KvCacheDType.Q8_0, KvCacheDType.Q8_0), (tail.KeyDType, tail.ValueDType));
                }
                else
                {
                    Assert.IsType<HybridF16TailKvCache>(probe.CpuCache);   // TODO(#1870): temporary F16 fallback
                }
                // The engine sees the GPU half's device bytes (#1851 review): retained split caches are evictable for admission.
                Assert.Equal(probe.GpuCache.DeviceBytes, ((IDeviceResidentKvCache)probe).DeviceBytes);
                Assert.True(probe.GpuCache.DeviceBytes > 0);
            }
            var split = Run(hybrid, () => hybrid.CreateKvCache(maxLen), vocab);
            var splitF32 = Run(hybrid, () => hybrid.CreateKvCacheWithF32TailForTests(maxLen), vocab);
            Assert.Equal(0, hybrid.CheckSpill($"after the N={n} parity run"));   // WDDM: no device buffer backed by system RAM

            // Red arm: the same decode with its history dropped.
            var amnesic = Run(hybrid, () => hybrid.CreateKvCache(maxLen), vocab, dropHistory: true);
            bool redCaught = false;
            for (int s = 0; s < steps; s++)
            {
                double dCpu = 1 - Cosine(cpu[s], split[s]), dGpu = 1 - Cosine(gpu[s], split[s]);
                double dF32 = 1 - Cosine(gpu[s], splitF32[s]), dRed = 1 - Cosine(cpu[s], amnesic[s]);
                double bar = Allowed(gap[s]);
                _output.WriteLine($"{name} N={n}/{layers} step {s}: 1-cos split vs cpu {dCpu:E3}, vs gpu {dGpu:E3} " +
                                  $"(f32-tail split vs gpu {dF32:E3}); bar {bar:E3}; red arm {dRed:E3}");
                if (dCpu > bar) failures.Add($"{name} N={n} step {s}: split vs CPU 1-cos {dCpu:E3} > {bar:E3}");
                if (dGpu > bar) failures.Add($"{name} N={n} step {s}: split vs GPU 1-cos {dGpu:E3} > {bar:E3}");
                // Top-1 must be among both references' top 5: a near tie may pick a different winner (the references themselves can
                // disagree there), but a broken tail does not land in both top-5 lists.
                int top = ArgMax(split[s]);
                if (!TopK(gpu[s], 5).Contains(top) || !TopK(cpu[s], 5).Contains(top))
                    failures.Add($"{name} N={n} step {s}: split argmax {top} is outside a reference's top 5");
                if (s > 0 && dRed > bar) redCaught = true;
            }
            if (!redCaught) failures.Add($"{name} N={n}: the red arm (history dropped) passed the bar, so the bar cannot see a lost KV");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [SkippableFact]
    public void SmolLm_SplitMatchesCpuAndGpu_ThroughTheHybridKvCache()
    {
        Skip.If(!File.Exists(SmolLmPath), "SmolLM-135M Q8_0 GGUF not found.");
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        CheckModel(SmolLmPath!, "SmolLM-135M");
    }

    /// <summary>
    /// Opt-in: the same parity on a real model named by <c>DOTLLM_SPLIT_PARITY_GGUF</c> (e.g. TwIL-8B), threaded, at the split points
    /// in <c>DOTLLM_SPLIT_PARITY_LAYERS</c> (comma-separated; default 1, L/2, L-1).
    /// </summary>
    [SkippableFact]
    public void RealModel_FromEnvironment_SplitMatchesCpuAndGpu()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_SPLIT_PARITY_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_SPLIT_PARITY_GGUF not set.");
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        int[]? splits = Environment.GetEnvironmentVariable("DOTLLM_SPLIT_PARITY_LAYERS")?.Split(',').Select(int.Parse).ToArray();
        CheckModel(path!, Path.GetFileNameWithoutExtension(path!), new ThreadingConfig(0, 0), splits);
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

    /// <summary>
    /// Review blocker 2 (#1851): head_dim 80 (Phi-2 / some Granite and Qwen shapes) is not a multiple of the Q8_0 block, and the Q8_0
    /// tail's head-at-a-time attention threw. The tail falls back to F16 (TODO #1870) and must match both references.
    /// </summary>
    [SkippableFact]
    public void HeadDim80_TailFallsBackToF16_AndMatchesCpuAndGpu()
    {
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        string path = Path.Combine(Path.GetTempPath(), $"syn_granite_hd80_{Guid.NewGuid():N}.gguf");
        SyntheticGraniteGguf.Write(path, new SyntheticGraniteConfig
            { Hidden = 320, Heads = 4, KvHeads = 2, Layers = 3, EmbeddingScale = 1f, ResidualScale = 1f, LogitScale = 1f, AttentionScale = 1f / 80, QkWeightScale = 1f });
        try
        {
            using (var gguf = GgufFile.Open(path))
            {
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                Assert.Equal(80, config.HeadDim);
                Assert.False(HybridTransformerModel.TailUsesQ8_0(config.HeadDim));
            }
            CheckModel(path, "hd80");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    /// <summary>
    /// Review blocker 1 (#1851): two unrelated prompts in turn through a prefix-cached generator, on a simulated card with room for
    /// one request's cache beside the split. The second request must not be refused because of the first one's cache.
    /// </summary>
    [SkippableFact]
    public unsafe void TwoUnrelatedPrompts_UnderABudgetForOneCache_BothRun()
    {
        Skip.If(!File.Exists(SmolLmPath), "SmolLM-135M Q8_0 GGUF not found.");
        Skip.If(!CudaDevice.IsAvailable(), "No CUDA device available.");
        using var gguf = GgufFile.Open(SmolLmPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var hybrid = HybridTransformerModel.LoadFromGguf(gguf, config, config.NumLayers / 2, deviceId: 0, ThreadingConfig.SingleThreaded);
        var opts = new InferenceOptions { MaxTokens = 48, Temperature = 0 };
        const string first = "The capital of France is", second = "def fibonacci(n):";

        // Warm the forward scratch to the prompt length first, so the budget below only has the KV caches to admit.
        new TextGenerator(hybrid, tokenizer, (c, size) => hybrid.CreateKvCache(size), prefillChunkSize: 1024).Generate(first, opts);

        long cacheBytes = 0;
        var prefix = new PrefixCache(4);
        var gen = new TextGenerator(hybrid, tokenizer, (c, size) =>
        {
            var kv = hybrid.CreateKvCache(size);
            cacheBytes = Math.Max(cacheBytes, kv.GpuCache.DeviceBytes);
            return kv;
        }, prefixCache: prefix, prefillChunkSize: 1024);

        // Simulated free memory: what is really free now, minus everything except the reserve plus 1.5 request caches. A second
        // cache allocated beside a retained first one does not fit; after evicting it, it does.
        Assert.Equal(0, DotLLM.Cuda.Interop.CudaDriverApi.cuMemGetInfoRaw(out nuint baseFree, out _));
        long probeBytes;
        using (var probe = hybrid.CreateKvCache(Tokens.Length + opts.MaxTokens + 16)) probeBytes = probe.GpuCache.DeviceBytes;
        long slack = CudaAdmission.ReserveBytes + probeBytes * 3 / 2;
        CudaAdmission.FreeBytesOverride = () =>
            DotLLM.Cuda.Interop.CudaDriverApi.cuMemGetInfoRaw(out nuint free, out _) == 0 ? slack - ((long)baseFree - (long)free) : null;
        try
        {
            var a = gen.Generate(first, opts);
            var b = gen.Generate(second, opts);
            Assert.NotEmpty(a.GeneratedTokenIds);
            Assert.NotEmpty(b.GeneratedTokenIds);
            _output.WriteLine($"cache {cacheBytes} B, probe {probeBytes} B; prefix entries {prefix.EntryCount}, admission evictions {prefix.AdmissionEvictions}");
        }
        finally
        {
            CudaAdmission.FreeBytesOverride = null;
            prefix.Clear();
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
