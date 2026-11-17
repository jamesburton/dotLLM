using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1850, model level: the dense q8_0 KV cache reading its rows in place against the pre-#1850 path (a full-history FP16
/// dequant scratch + the F16 attention kernels, kept as a test-only oracle behind <see cref="CudaQuantizedKvCache.ReferenceDequantScratch"/>),
/// on Qwen2.5-0.5B (head_dim 64, 14/2 heads: group 7, 24 layers).
/// <list type="bullet">
/// <item><b>Exact arm.</b> With G3, G-flash and the kvq tensor-core prefill off, the legacy path is dequant + <c>attention_f16</c> and
/// the new one is <c>attention_kvq</c>: the prefill logits and every teacher-forced decode step must be BIT-identical, pure
/// quantized and with an FP16 window (the windowed decode replays the CUDA graph through <c>attention_kvq_dyn</c>, the legacy
/// arm decodes eagerly).</item>
/// <item><b>Production arm.</b> Default dispatch: the new prefill is the kvq tensor-core kernel, the legacy one G3 / G-flash over the
/// scratch (not bit-comparable). Greedy tokens must be identical over the run, the per-step KL(legacy || new) small, and the new
/// path no further from an F16 KV cache (the adjudicator) than the legacy path is.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQuantizedKvInPlaceParityTests
{
    private const int Steps = 24;
    private readonly ITestOutputHelper _out;
    public CudaQuantizedKvInPlaceParityTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;

    private const string Filler =
        "The quick brown fox jumps over the lazy dog while the committee reviews the quarterly figures in detail. ";

    private static string Prompt() =>
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + string.Concat(Enumerable.Repeat(Filler, 28)) + "Summarise that in one sentence.<|im_end|>\n<|im_start|>assistant\n";

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(32)]
    public void ScalarArm_IsBitIdenticalToTheLegacyScratchPath(int window)
    {
        using var env = Load(out var model, out int[] prompt);
        bool g3 = CudaG3Attention.Enabled, flash = CudaFlashAttention.Enabled, kvqFlash = CudaTransformerModel.EnableKvqFlashPrefill;
        int depth = CudaTransformerModel.GraphCaptureMaxDepth;
        CudaG3Attention.Enabled = false;
        CudaFlashAttention.Enabled = false;
        CudaTransformerModel.EnableKvqFlashPrefill = false;
        CudaTransformerModel.GraphCaptureMaxDepth = 8192;   // let the windowed decode replay the quantized graph
        try
        {
            var kv = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0, window);
            long kvq0 = CudaKernels.AttentionKvqLaunches;
            var legacy = Run(model, kv, prompt, reference: true, forced: null);
            Assert.Equal(kvq0, CudaKernels.AttentionKvqLaunches);
            int replays0 = model.GraphReplayCount;
            var fresh = Run(model, kv, prompt, reference: false, forced: legacy.Tokens);
            long kvq = CudaKernels.AttentionKvqLaunches - kvq0;
            _out.WriteLine($"window {window}: attention_kvq eager launches {kvq}, graph replays +{model.GraphReplayCount - replays0}");
            Assert.True(kvq >= model.Config.NumLayers, "the new arm did not run attention_kvq");
            if (window > 0 && model.UseGraphCapture)
                Assert.True(model.GraphReplayCount > replays0, "the windowed decode did not replay the quantized graph");

            for (int s = 0; s < legacy.Logits.Count; s++)
            {
                int d = FirstDiff(legacy.Logits[s], fresh.Logits[s]);
                Assert.True(d < 0, $"window {window} step {s}: logit {d} differs: {fresh.Logits[s][Math.Max(d, 0)]:R} vs " +
                                   $"{legacy.Logits[s][Math.Max(d, 0)]:R}");
            }
        }
        finally
        {
            CudaG3Attention.Enabled = g3;
            CudaFlashAttention.Enabled = flash;
            CudaTransformerModel.EnableKvqFlashPrefill = kvqFlash;
            CudaTransformerModel.GraphCaptureMaxDepth = depth;
        }
    }

    [SkippableFact]
    public void ProductionArm_MatchesTheLegacyPath_GreedyIdentical_SmallKl()
    {
        using var env = Load(out var model, out int[] prompt);
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);

        var legacy = Run(model, q8, prompt, reference: true, forced: null);
        long flash0 = CudaKernels.AttentionFlashPrefillKvqLaunches;
        var fresh = Run(model, q8, prompt, reference: false, forced: legacy.Tokens);
        long flash = CudaKernels.AttentionFlashPrefillKvqLaunches - flash0;
        var f16 = Run(model, new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32), prompt, reference: false, forced: legacy.Tokens);

        double[] klNew = Kl(legacy, fresh), klF16Legacy = Kl(f16, legacy), klF16New = Kl(f16, fresh);
        _out.WriteLine($"prompt {prompt.Length} tokens, kvq flash prefill launches {flash}");
        _out.WriteLine($"first-token logprob: legacy {LogProb(legacy.Logits[0], legacy.Tokens[0]):R}, new " +
                       $"{LogProb(fresh.Logits[0], legacy.Tokens[0]):R}, f16 KV {LogProb(f16.Logits[0], legacy.Tokens[0]):R}");
        _out.WriteLine($"KL(legacy||new) mean {klNew.Average():E3} max {klNew.Max():E3}; KL(f16||legacy) mean " +
                       $"{klF16Legacy.Average():E3}; KL(f16||new) mean {klF16New.Average():E3}");

        if (model.Kernels.HasAttentionFlashPrefillKvq(model.Config.HeadDim, KvCacheDType.Q8_0, KvCacheDType.Q8_0))
            Assert.True(flash >= model.Config.NumLayers, "the new prefill did not run the kvq tensor-core kernel");
        for (int s = 0; s < legacy.Tokens.Length; s++)
            Assert.True(Argmax(fresh.Logits[s]) == legacy.Tokens[s], $"greedy token {s} differs from the legacy path");
        Assert.True(Math.Abs(LogProb(fresh.Logits[0], legacy.Tokens[0]) - LogProb(legacy.Logits[0], legacy.Tokens[0])) < 0.02,
            "first-token logprob moved");
        Assert.True(klNew.Average() < 2e-3, $"mean KL(legacy||new) {klNew.Average()}");
        Assert.True(klF16New.Average() <= 1.5 * klF16Legacy.Average() + 1e-3,
            $"the new path is further from the F16 cache ({klF16New.Average()}) than the legacy path ({klF16Legacy.Average()})");
    }

    /// <summary>
    /// A cuMemGetInfo delta proves the history scratch is gone: a one-layer cache of 65,536 positions at stride 1,024 (TwIL-8B's)
    /// costs its q8_0 rows (2 x 68 MiB) and not the 256 MiB the FP16 K+V scratch added before #1850.
    /// </summary>
    [SkippableFact]
    public void CacheAllocation_MeasuresOnlyTheQuantizedRows()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var geom = KvGeometry.Uniform(1, 8, 128);
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        const int positions = 65536;
        long expected = CudaQuantizedKvCache.BytesFor(geom, positions, q8);
        long legacyScratch = 2L * positions * 1024 * sizeof(ushort);
        Assert.Equal(2L * positions * 1088, expected);

        CudaDriverApi.cuMemGetInfoRaw(out nuint before, out _).ThrowOnError();
        using var cache = new CudaQuantizedKvCache(geom, positions, q8);
        CudaDriverApi.cuMemGetInfoRaw(out nuint after, out _).ThrowOnError();
        long delta = (long)before - (long)after;
        _out.WriteLine($"cache delta {delta >> 20} MiB, rows {expected >> 20} MiB, pre-#1850 scratch would add {legacyScratch >> 20} MiB");
        Assert.Equal(expected, cache.AllocatedBytes);
        Assert.True(delta < expected + legacyScratch / 4, $"cache took {delta} B, more than its {expected} B of rows allows");
    }

    private sealed record Arm(List<float[]> Logits, int[] Tokens);

    /// <summary>Prefill the prompt, then <see cref="Steps"/> decode steps (greedy, or forced to <paramref name="forced"/>).</summary>
    private static Arm Run(CudaTransformerModel model, KvCacheConfig kv, int[] prompt, bool reference, int[]? forced)
    {
        CudaQuantizedKvCache.ReferenceDequantScratch = reference;
        try
        {
            using var cache = (IKvCache)model.CreateKvCache(prompt.Length + Steps + 8, kv);
            int vocab = model.Config.VocabSize;
            var logits = new List<float[]>();
            var tokens = new int[Steps];
            using (ITensor t = model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), 0, cache))
                logits.Add(LastRow(t, vocab));
            for (int s = 0; s < Steps; s++)
            {
                tokens[s] = forced?[s] ?? Argmax(logits[^1]);
                if (s == Steps - 1) break;
                using ITensor t = model.Forward([tokens[s]], [prompt.Length + s], 0, cache);
                logits.Add(LastRow(t, vocab));
            }
            return new Arm(logits, tokens);
        }
        finally { CudaQuantizedKvCache.ReferenceDequantScratch = false; }
    }

    private static unsafe float[] LastRow(ITensor t, int vocab)
        => new ReadOnlySpan<float>((float*)t.DataPointer + (t.ElementCount - vocab), vocab).ToArray();

    private static int Argmax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private static double[] LogSoftmax(float[] a)
    {
        double m = a.Max(), sum = 0;
        foreach (float x in a) sum += Math.Exp(x - m);
        double lse = m + Math.Log(sum);
        return a.Select(x => x - lse).ToArray();
    }

    private static double LogProb(float[] logits, int token) => LogSoftmax(logits)[token];

    private static double[] Kl(Arm p, Arm q)
    {
        var kl = new double[p.Logits.Count];
        for (int s = 0; s < kl.Length; s++)
        {
            double[] lp = LogSoftmax(p.Logits[s]), lq = LogSoftmax(q.Logits[s]);
            double acc = 0;
            for (int i = 0; i < lp.Length; i++) acc += Math.Exp(lp[i]) * (lp[i] - lq[i]);
            kl[s] = acc;
        }
        return kl;
    }

    private static int FirstDiff(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) return i;
        return -1;
    }

    private IDisposable Load(out CudaTransformerModel model, out int[] prompt)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir!);
        prompt = tokenizer.Encode(Prompt());
        return new Owned(model, gguf);
    }

    private sealed class Owned(IDisposable model, IDisposable gguf) : IDisposable
    {
        public void Dispose()
        {
            model.Dispose();
            gguf.Dispose();
        }
    }
}
