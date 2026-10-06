using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// GGUF <c>gemma</c> / <c>gemma2</c> GPU parity (#736): each GPU backend must reproduce the CPU
/// backend's per-position logits on the synthetic fixtures — through BOTH the prefill path and
/// the single-token KV-cache decode path (split-KV flash-decoding engages from ~17 tokens of
/// context, so the sequence is deliberately longer than that).
/// </summary>
/// <remarks>
/// The fixtures are chosen so each Gemma-2 feature <i>bites</i> (soft-caps of 2.0 / 3.0 against
/// pre-cap scores of several units, a window of 3 against a 24-token sequence, GQA with 2 KV
/// heads, a 46-layer config so <c>query_pre_attn_scalar</c> != head_dim); the CPU side of that is
/// proven against an independent reference in <c>Gemma2GgufTests</c> (ablation arms prove the
/// fixture is sensitive). A GPU backend that silently dropped any feature therefore diverges here
/// by far more than the tolerance.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Gemma2GpuParityTests
{
    private readonly ITestOutputHelper _output;

    public Gemma2GpuParityTests(ITestOutputHelper output) => _output = output;

    // 24 tokens; first one is BOS (2).
    private static readonly int[] Tokens =
        [2, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 40, 6, 33, 78, 14, 90, 27, 3, 52, 11, 67, 35];

    private const int PrefillLen = 18;   // then 6 single-token decode steps

    public static IEnumerable<object[]> Fixtures()
    {
        yield return new object[] { "gemma2-46L-hd8-win3", new SyntheticGemma2Config() };
        // Real Gemma-2 head dim (256) — the size at which f16 attention inputs broke Qwen3.5 (#NLL regression).
        yield return new object[] { "gemma2-8L-hd256-win5", new SyntheticGemma2Config
            { Layers = 8, HeadDim = 256, SlidingWindow = 5, AttnSoftcap = 4.0f, FinalSoftcap = 3.0f } };
        // Gemma 1 / CodeGemma shape: two-norm, MQA (1 KV head), no caps, no window.
        yield return new object[] { "gemma1-4L-mqa", new SyntheticGemma2Config
            { Arch = "gemma", Layers = 4, KvHeads = 1, HeadDim = 32 } };
    }

    private static string WriteFixture(SyntheticGemma2Config cfg)
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma2_gpu_{Guid.NewGuid():N}.gguf");
        SyntheticGemma2Gguf.Write(path, cfg);
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

    /// <summary>CPU oracle: one cacheless forward returns logits for every position.</summary>
    private static float[][] CpuRows(string path, out int vocab)
    {
        var (model, gguf, cfg) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            vocab = cfg.VocabSize;
            int[] pos = Enumerable.Range(0, Tokens.Length).ToArray();
            using ITensor logits = model.Forward(Tokens, pos, -1, kvCache: null);
            float[] all = Rows(logits);
            Assert.Equal(Tokens.Length * vocab, all.Length);
            var rows = new float[Tokens.Length][];
            for (int t = 0; t < rows.Length; t++) rows[t] = all.AsSpan(t * vocab, vocab).ToArray();
            return rows;
        }
    }

    private void Compare(string label, float[] cpu, float[] gpu, double absTol, ref double worst)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            double d = Math.Abs(cpu[i] - gpu[i]);
            worst = Math.Max(worst, d);
            Assert.True(d <= absTol, $"{label}: col {i}: cpu={cpu[i]:F5} gpu={gpu[i]:F5} |diff|={d:E3} > {absTol:E3}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Fixtures))]
    public void Vulkan_MatchesCpu_PrefillAndKvCacheDecode(string name, SyntheticGemma2Config cfg)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        string path = WriteFixture(cfg);
        try
        {
            float[][] cpu = CpuRows(path, out int vocab);

            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(cfg.Arch == "gemma2" ? Architecture.Gemma2 : Architecture.Gemma, config.Architecture);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, ResolveSpvDir());

            double worst = 0;
            // Prefill with a KV cache (fills it), compare the last prefill row.
            using (var cache = model.CreateKvCache(maxSeqLen: Tokens.Length + 1))
            {
                int[] ids = Tokens.AsSpan(0, PrefillLen).ToArray();
                int[] pos = Enumerable.Range(0, PrefillLen).ToArray();
                using (ITensor logits = model.Forward(ids, pos, -1, cache))
                {
                    float[] all = Rows(logits);
                    Compare($"{name} prefill", cpu[PrefillLen - 1], all.AsSpan(all.Length - vocab, vocab).ToArray(), 0.03, ref worst);
                }
                // Single-token decode steps — the S==1 path (split-KV from ~17 ctx, sliding window per layer).
                for (int t = PrefillLen; t < Tokens.Length; t++)
                {
                    using ITensor logits = model.Forward([Tokens[t]], [t], -1, cache);
                    float[] all = Rows(logits);
                    Compare($"{name} decode@{t}", cpu[t], all.AsSpan(all.Length - vocab, vocab).ToArray(), 0.03, ref worst);
                }
            }

            // Cacheless full-sequence forward too (the perplexity / scoring path).
            using (ITensor logits = model.Forward(Tokens, Enumerable.Range(0, Tokens.Length).ToArray(), -1, kvCache: null))
            {
                float[] all = Rows(logits);
                Compare($"{name} cacheless", cpu[Tokens.Length - 1], all.AsSpan(all.Length - vocab, vocab).ToArray(), 0.03, ref worst);
            }
            _output.WriteLine($"{name}: Vulkan vs CPU worst |diff| = {worst:E3} over prefill + 6 decode steps + cacheless.");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
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
