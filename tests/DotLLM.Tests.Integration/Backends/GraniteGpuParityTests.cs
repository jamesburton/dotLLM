using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// GGUF <c>granite</c> / <c>granitemoe</c> GPU parity (#764, #313): each GPU backend must reproduce the CPU backend's
/// per-position logits on a synthetic fixture whose four Granite scalars are != 1 and mutually distinct (embedding 3.0,
/// attention 0.07, residual 0.4, logit 5.0) — through prefill, single-token KV-cache decode, the cacheless scoring
/// path and the scheduler's <c>ForwardBatch</c>. The CPU side is proven against an independent reference in
/// <c>GraniteGgufTests</c> (with ablation arms), so a GPU backend that dropped or misplaced any scalar diverges here by
/// far more than the tolerance. NB the logit scale never changes the argmax: tests must compare logit VALUES.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class GraniteGpuParityTests
{
    private readonly ITestOutputHelper _output;

    public GraniteGpuParityTests(ITestOutputHelper output) => _output = output;

    private static readonly int[] Tokens =
        [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 40, 6, 33, 78, 14, 90, 27, 3, 52, 11, 67, 35];

    private const int PrefillLen = 18;   // then 6 single-token decode steps

    public static IEnumerable<object[]> DenseFixtures()
    {
        yield return new object[] { "granite-3L-gqa", new SyntheticGraniteConfig() };
        yield return new object[] { "granite-tied-hd64", new SyntheticGraniteConfig { Hidden = 256, Heads = 4, KvHeads = 4, Tied = true, Layers = 2 } };
    }

    /// <summary>
    /// Dense fixtures only for the CPU-vs-GPU parity tests. GraniteMoe is covered against the exact double-precision
    /// reference in <c>GraniteVulkanReferenceTests</c> instead: the CPU routed-MoE path multiplies Q8_0 expert weights by
    /// Q8-QUANTIZED activations (as llama.cpp does) while Vulkan keeps F32 activations, so on a Q8_0 MoE fixture the two
    /// backends legitimately differ by ~1% and a CPU oracle would mask real GPU bugs.
    /// </summary>
    public static IEnumerable<object[]> AllFixtures() => DenseFixtures();

    private static string WriteFixture(SyntheticGraniteConfig cfg)
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_granite_gpu_{Guid.NewGuid():N}.gguf");
        SyntheticGraniteGguf.Write(path, cfg);
        return path;
    }

    private static unsafe float[] Rows(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var all = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(all);
        return all;
    }

    private static float[] Last(ITensor logits, int vocab)
    {
        float[] all = Rows(logits);
        return all.AsSpan(all.Length - vocab, vocab).ToArray();
    }

    private static float[][] CpuRows(string path, int[] tokens, out int vocab)
    {
        var (model, gguf, cfg) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            vocab = cfg.VocabSize;
            int[] pos = Enumerable.Range(0, tokens.Length).ToArray();
            using ITensor logits = model.Forward(tokens, pos, -1, kvCache: null);
            float[] all = Rows(logits);
            var rows = new float[tokens.Length][];
            for (int t = 0; t < rows.Length; t++) rows[t] = all.AsSpan(t * vocab, vocab).ToArray();
            return rows;
        }
    }

    private static void Compare(string label, float[] cpu, float[] gpu, double absTol, ref double worst)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            double d = Math.Abs(cpu[i] - gpu[i]);
            worst = Math.Max(worst, d);
            double bar = absTol + 5e-3 * Math.Abs(cpu[i]);
            Assert.True(d <= bar, $"{label}: col {i}: cpu={cpu[i]:F5} gpu={gpu[i]:F5} |diff|={d:E3} > {bar:E3}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(AllFixtures))]
    public void Vulkan_MatchesCpu_PrefillAndKvCacheDecode(string name, SyntheticGraniteConfig cfg)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        string path = WriteFixture(cfg);
        try
        {
            float[][] cpu = CpuRows(path, Tokens, out int vocab);
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(cfg.IsMoe ? Architecture.GraniteMoe : Architecture.Granite, config.Architecture);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, ResolveSpvDir());

            double worst = 0;
            using (var cache = model.CreateKvCache(maxSeqLen: Tokens.Length + 1))
            {
                using (ITensor l = model.Forward(Tokens.AsSpan(0, PrefillLen).ToArray(),
                           Enumerable.Range(0, PrefillLen).ToArray(), -1, cache))
                    Compare($"{name} prefill", cpu[PrefillLen - 1], Last(l, vocab), 0.02, ref worst);
                for (int t = PrefillLen; t < Tokens.Length; t++)
                {
                    using ITensor l = model.Forward([Tokens[t]], [t], -1, cache);
                    Compare($"{name} decode@{t}", cpu[t], Last(l, vocab), 0.02, ref worst);
                }
            }
            using (ITensor l = model.Forward(Tokens, Enumerable.Range(0, Tokens.Length).ToArray(), -1, kvCache: null))
                Compare($"{name} cacheless", cpu[Tokens.Length - 1], Last(l, vocab), 0.02, ref worst);
            _output.WriteLine($"{name}: Vulkan vs CPU worst |diff| = {worst:E3} over prefill + 6 decode steps + cacheless.");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    /// <summary>
    /// The fused batched decode (<c>ForwardBatch</c>, at least 2 "simple" sequences) runs a dense Llama-style layer loop
    /// with none of the Granite scalars: Granite sequences must take the per-sequence path.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(AllFixtures))]
    public void Vulkan_ForwardBatch_MatchesCpu(string name, SyntheticGraniteConfig cfg)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        int[] seqA = Tokens;
        int[] seqB = Tokens.Select((t, i) => i == 0 ? 1 : (t * 7 + 3) % 90 + 4).ToArray();
        string path = WriteFixture(cfg);
        try
        {
            float[][] cpuA = CpuRows(path, seqA, out int vocab);
            float[][] cpuB = CpuRows(path, seqB, out _);
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, ResolveSpvDir());
            using var cacheA = model.CreateKvCache(maxSeqLen: seqA.Length + 1);
            using var cacheB = model.CreateKvCache(maxSeqLen: seqB.Length + 1);
            foreach (var (seq, cache) in new[] { (seqA, cacheA), (seqB, cacheB) })
            {
                using ITensor l = model.Forward(seq.AsSpan(0, PrefillLen).ToArray(),
                    Enumerable.Range(0, PrefillLen).ToArray(), -1, cache);
            }
            double worst = 0;
            for (int t = PrefillLen; t < seqA.Length; t++)
            {
                var requests = new[]
                {
                    new SequenceForwardRequest { TokenIds = new[] { seqA[t] }, Positions = new[] { t }, KvCache = cacheA },
                    new SequenceForwardRequest { TokenIds = new[] { seqB[t] }, Positions = new[] { t }, KvCache = cacheB },
                };
                IReadOnlyList<ITensor> results = model.ForwardBatch(requests, deviceId: -1);
                Compare($"{name} batch A@{t}", cpuA[t], Last(results[0], vocab), 0.02, ref worst);
                Compare($"{name} batch B@{t}", cpuB[t], Last(results[1], vocab), 0.02, ref worst);
                foreach (var r in results) r.Dispose();
            }
            _output.WriteLine($"{name}: Vulkan ForwardBatch vs CPU worst |diff| = {worst:E3}.");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    /// <summary>CUDA dense Granite through the FP32 dense forward: prefill, decode, cacheless and the batch entry point.</summary>
    [SkippableTheory]
    [MemberData(nameof(DenseFixtures))]
    public void Cuda_MatchesCpu_PrefillDecodeCachelessAndBatch(string name, SyntheticGraniteConfig cfg)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA device/driver on this host.");

        string path = WriteFixture(cfg);
        try
        {
            int[] seqB = Tokens.Select((t, i) => i == 0 ? 1 : (t * 7 + 3) % 90 + 4).ToArray();
            float[][] cpuA = CpuRows(path, Tokens, out int vocab);
            float[][] cpuB = CpuRows(path, seqB, out _);

            var (model, gguf, config) = CudaModelLoader.LoadFromGguf(path, deviceId: 0);
            using (gguf)
            using (model)
            {
                Assert.Equal(Architecture.Granite, config.Architecture);
                double worst = 0;
                const double abs = 0.05;   // FP16 weights + FP16 logits head + FP16 KV store

                using (var cache = model.CreateKvCache(maxSeqLen: Tokens.Length + 1))
                {
                    using (ITensor l = model.Forward(Tokens.AsSpan(0, PrefillLen).ToArray(),
                               Enumerable.Range(0, PrefillLen).ToArray(), -1, cache))
                        Compare($"{name} prefill", cpuA[PrefillLen - 1], Last(l, vocab), abs, ref worst);
                    for (int t = PrefillLen; t < Tokens.Length; t++)
                    {
                        using ITensor l = model.Forward([Tokens[t]], [t], -1, cache);
                        Compare($"{name} decode@{t}", cpuA[t], Last(l, vocab), abs, ref worst);
                    }
                }
                using (ITensor l = model.Forward(Tokens, Enumerable.Range(0, Tokens.Length).ToArray(), -1, kvCache: null))
                    Compare($"{name} cacheless", cpuA[Tokens.Length - 1], Last(l, vocab), abs, ref worst);

                using var cacheA = model.CreateKvCache(maxSeqLen: Tokens.Length + 1);
                using var cacheB = model.CreateKvCache(maxSeqLen: seqB.Length + 1);
                foreach (var (seq, cache) in new[] { (Tokens, cacheA), (seqB, cacheB) })
                {
                    using ITensor l = model.Forward(seq.AsSpan(0, PrefillLen).ToArray(),
                        Enumerable.Range(0, PrefillLen).ToArray(), -1, cache);
                }
                for (int t = PrefillLen; t < Tokens.Length; t++)
                {
                    var requests = new[]
                    {
                        new SequenceForwardRequest { TokenIds = new[] { Tokens[t] }, Positions = new[] { t }, KvCache = cacheA },
                        new SequenceForwardRequest { TokenIds = new[] { seqB[t] }, Positions = new[] { t }, KvCache = cacheB },
                    };
                    IReadOnlyList<ITensor> results = model.ForwardBatch(requests, deviceId: -1);
                    Compare($"{name} batch A@{t}", cpuA[t], Last(results[0], vocab), abs, ref worst);
                    Compare($"{name} batch B@{t}", cpuB[t], Last(results[1], vocab), abs, ref worst);
                    foreach (var r in results) r.Dispose();
                }
                _output.WriteLine($"{name}: CUDA vs CPU worst |diff| = {worst:E3} (prefill + 6 decode + cacheless + batch).");
            }
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    /// <summary>
    /// CUDA cannot honour GraniteMoe (no scalars in the routed-MoE layer loop): it must be REFUSED at load, with an
    /// actionable message — never a silent wrong path or CPU fallback. Needs no GPU: the guard runs before any CUDA resource.
    /// </summary>
    [Fact]
    public void Cuda_GraniteMoe_IsRejectedAtLoad_AndCompositeHostsRejectGranite()
    {
        string path = WriteFixture(new SyntheticGraniteConfig { Arch = "granitemoe" });
        string densePath = WriteFixture(new SyntheticGraniteConfig());
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            var ex = Assert.Throws<NotSupportedException>(() => CudaModelLoader.LoadFromGguf(path, deviceId: 0));
            Assert.Contains("GraniteMoe", ex.Message);
            Assert.Contains("issues/764", ex.Message);
            Assert.Throws<NotSupportedException>(() => CudaTransformerModel.LoadFromGguf(gguf, config));

            // Composite hosts drive a generic FP16 layer loop with no Granite scalars: dense Granite is refused too.
            using var denseGguf = GgufFile.Open(densePath);
            var denseConfig = GgufModelConfigExtractor.Extract(denseGguf.Metadata);
            Assert.Throws<NotSupportedException>(() => CudaPipelineTransformerModel.LoadFromGguf(denseGguf, denseConfig, 2, 0, 1));
            Assert.Throws<NotSupportedException>(() => HybridTransformerModel.LoadFromGguf(denseGguf, denseConfig, 2, 0, ThreadingConfig.SingleThreaded));
            Assert.Throws<NotSupportedException>(() => HybridVulkanCudaTransformerModel.LoadFromGguf(denseGguf, denseConfig, 2));
        }
        finally
        {
            try { File.Delete(path); File.Delete(densePath); } catch { /* best-effort */ }
        }
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
