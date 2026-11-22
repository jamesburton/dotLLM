using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1923, model level: the dense F16 KV cache's prefill on the kvq flash kernel's F16 entry, on Qwen3-4B (head_dim 128,
/// 32/8 heads: group 4, 36 layers; the model of the llama.cpp gap measurement).
/// <list type="bullet">
/// <item><b>Rows.</b> The dense CUDA forward returns the last row's logits only, so the last <see cref="Rows"/> prompt rows are
/// produced one each: the prefix is prefilled in chunks of 1000 (off the 16-row grid, partial last chunk), then for each row r the
/// cache is rolled back to r - 31 and the 32-row window ending at r is prefilled at its offset. Every arm takes its own kernel for
/// every window (32 rows is G3's offset minimum, and above the flash entry's 16).</item>
/// <item><b>Proof of fire.</b> The F16 flash entry runs exactly once per layer per forward; G3's offset path and the scalar
/// <c>attention_f16</c> prefill never run. The G3 arm must really run G3's offset path (a reference that cannot differ proves
/// nothing).</item>
/// <item><b>Gate (James 2026-11-21)</b>, per-row KL against the exact arm (G3, G-flash and the F16 flash prefill off:
/// <c>attention_f16</c>, F32 softmax) over the last 256 prompt rows: median at most 1e-5, and max at most
/// max(1e-3, 1.25 x the G3 arm's max). A relative max bar: G3, the path this replaces, does not meet an absolute 1e-3 either
/// (framework 2026-11-20 over 21 rows: G3 max 7.1e-3, flash 4.3e-3; both round P to F16 for a tensor-core P.V). First-token
/// logprob within 0.01 nats; top-1 agreement reported, not gated (a near-tie may flip on a quantized model).</item>
/// <item><b>Sensitivity (review M1, framework 2026-11-22).</b> A subtle mutant, the F16 entry's softmax scale x (1 + 1/1024), raises
/// the max KL from 5.2e-3 to 9.8e-3 against a bound of 7.2e-3 (1.25 x G3's 5.7e-3): killed by the max bar. The median only moves
/// 2.25e-9 -> 2.40e-9, so the median bar is a coarse guard (no tightening can see this mutant with margin); the max bar is the
/// sensitive one.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaF16FlashPrefillTests
{
    private const int Rows = 256;
    private const int Window = 32;
    private const int Chunk = 1000;
    private readonly ITestOutputHelper _out;
    public CudaF16FlashPrefillTests(ITestOutputHelper output) => _out = output;

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
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF", "*Q4_K_M.gguf").Path;

    // Varied text (not one sentence repeated): a repeated sentence makes attention near-uniform over copies, which hides offset bugs.
    private static string Prompt()
    {
        var sb = new System.Text.StringBuilder("<|im_start|>user\n");
        var rng = new Random(1923);
        string[] words = ["river", "ledger", "copper", "falcon", "quartz", "harbor", "violet", "engine", "marble", "signal",
            "cobalt", "meadow", "lantern", "orbit", "saddle", "thistle", "vector", "willow", "zephyr", "anchor"];
        for (int i = 0; i < 280; i++)
        {
            sb.Append($"Item {i}: the {words[rng.Next(words.Length)]} near the {words[rng.Next(words.Length)]} weighs {rng.Next(1, 999)} units. ");
        }
        sb.Append("What does item 17 weigh?<|im_end|>\n<|im_start|>assistant\n");
        return sb.ToString();
    }

    [SkippableFact]
    public void ChunkedF16Prefill_RunsTheFlashEntryEveryChunk_AndMatchesTheExactArm()
    {
        using var env = Load(out var model, out int[] prompt);
        Skip.IfNot(model.Kernels.HasAttentionFlashPrefillKvq(model.Config.HeadDim, null, null), "F16 flash prefill entry not loaded");
        Skip.IfNot(CudaG3Attention.Enabled, "G3 (the relative gate's reference arm) is off on this device");
        // Trim so the length is off the 16-row grid and the prefix's partial last chunk still has >= 32 rows (a chunk under 16 rows
        // would rightly skip the flash entry and break the exact launch count).
        int len = prompt.Length;
        while (len % 16 == 0 || (len - Rows - (Window - 1)) % Chunk < 32) len--;
        prompt = prompt[..len];
        int prefix = prompt.Length - Rows - (Window - 1);
        Assert.True(prefix > 3 * Chunk, $"prompt {prompt.Length} must leave a prefix of 3+ chunks");
        int forwards = (prefix + Chunk - 1) / Chunk + Rows;
        long layers = model.Config.NumLayers;
        var kv = new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);   // the dense F16 CudaKvCache

        bool g3 = CudaG3Attention.Enabled, flash = CudaFlashAttention.Enabled, f16Flash = CudaTransformerModel.EnableF16FlashPrefill;
        List<float[]> exact, fresh, g3Arm;
        long flashLaunches, offsetRuns, scalarPrefills, g3OffsetRuns, exactScalar;
        try
        {
            CudaG3Attention.Enabled = false;
            CudaFlashAttention.Enabled = false;
            CudaTransformerModel.EnableF16FlashPrefill = false;
            long e0 = CudaKernels.AttentionF16PrefillLaunches;
            exact = Run(model, kv, prompt);
            exactScalar = CudaKernels.AttentionF16PrefillLaunches - e0;

            CudaG3Attention.Enabled = g3;
            CudaFlashAttention.Enabled = flash;
            long g0 = CudaG3Attention.OffsetRunCount;
            g3Arm = Run(model, kv, prompt);
            g3OffsetRuns = CudaG3Attention.OffsetRunCount - g0;

            CudaTransformerModel.EnableF16FlashPrefill = true;
            long f0 = CudaKernels.AttentionFlashPrefillF16Launches, o0 = CudaG3Attention.OffsetRunCount;
            long s0 = CudaKernels.AttentionF16PrefillLaunches;
            fresh = Run(model, kv, prompt);
            flashLaunches = CudaKernels.AttentionFlashPrefillF16Launches - f0;
            offsetRuns = CudaG3Attention.OffsetRunCount - o0;
            scalarPrefills = CudaKernels.AttentionF16PrefillLaunches - s0;
        }
        finally
        {
            CudaG3Attention.Enabled = g3;
            CudaFlashAttention.Enabled = flash;
            CudaTransformerModel.EnableF16FlashPrefill = f16Flash;
        }

        double[] kl = Kl(exact, fresh), klG3 = Kl(exact, g3Arm);
        int next = Argmax(exact[^1]);   // the exact arm's first generated token
        double lpExact = LogProb(exact[^1], next), lpFresh = LogProb(fresh[^1], next);
        int top1 = Enumerable.Range(0, Rows).Count(r => Argmax(fresh[r]) == Argmax(exact[r]));
        int top1G3 = Enumerable.Range(0, Rows).Count(r => Argmax(g3Arm[r]) == Argmax(exact[r]));
        _out.WriteLine($"prompt {prompt.Length} tokens: prefix {prefix} in chunks of {Chunk}, then {Rows} windows of {Window}; " +
                       $"{forwards} forwards per arm");
        _out.WriteLine($"F16 flash launches {flashLaunches}, G3 offset runs {offsetRuns}, attention_f16 prefill launches {scalarPrefills}; " +
                       $"G3 arm offset runs {g3OffsetRuns}; exact arm attention_f16 prefill launches {exactScalar}");
        _out.WriteLine($"KL(exact||flash) over the last {Rows} rows: median {Median(kl):E3} max {kl.Max():E3}; " +
                       $"KL(exact||G3) median {Median(klG3):E3} max {klG3.Max():E3}");
        _out.WriteLine($"first-token logprob: exact {lpExact:R}, flash {lpFresh:R}, G3 {LogProb(g3Arm[^1], next):R}; " +
                       $"top-1 agreement flash {top1}/{Rows}, G3 {top1G3}/{Rows}");

        Assert.Equal(layers * forwards, flashLaunches);
        Assert.Equal(0, offsetRuns);
        Assert.Equal(0, scalarPrefills);
        Assert.True(g3OffsetRuns >= layers * Rows, $"the G3 reference arm ran its offset path only {g3OffsetRuns} times");
        Assert.True(exactScalar >= layers * forwards, $"the exact arm ran attention_f16 only {exactScalar} times");
        Assert.True(Math.Abs(lpFresh - lpExact) < 0.01, $"first-token logprob moved {lpFresh - lpExact:R}");
        // James 2026-11-21: relative gate, median <= 1e-5 and max <= max(1e-3, 1.25 x G3 max), over the last 256 rows.
        Assert.True(Median(kl) <= 1e-5, $"median KL(exact||flash) {Median(kl)}");
        Assert.True(kl.Max() <= Math.Max(1e-3, 1.25 * klG3.Max()), $"max KL(exact||flash) {kl.Max()} vs G3 {klG3.Max()}");
    }

    /// <summary>Logits of the last <see cref="Rows"/> prompt rows, row r from the <see cref="Window"/>-row forward ending at r.</summary>
    private static List<float[]> Run(CudaTransformerModel model, KvCacheConfig kv, int[] prompt)
    {
        using var cache = (IKvCache)model.CreateKvCache(prompt.Length + 8, kv);
        Assert.IsType<CudaKvCache>(cache);
        int vocab = model.Config.VocabSize;
        int prefix = prompt.Length - Rows - (Window - 1);
        for (int start = 0; start < prefix; start += Chunk)
        {
            int n = Math.Min(Chunk, prefix - start);
            using ITensor t = model.Forward(prompt.AsSpan(start, n).ToArray(), Enumerable.Range(start, n).ToArray(), 0, cache);
        }
        var logits = new List<float[]>(Rows);
        for (int r = prompt.Length - Rows; r < prompt.Length; r++)
        {
            int start = r - (Window - 1);
            cache.Rollback(start);
            using ITensor t = model.Forward(prompt.AsSpan(start, Window).ToArray(), Enumerable.Range(start, Window).ToArray(), 0, cache);
            logits.Add(LastRow(t, vocab));
        }
        return logits;
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

    private static double Median(double[] a)
    {
        var s = a.OrderBy(x => x).ToArray();
        return s.Length % 2 == 1 ? s[s.Length / 2] : 0.5 * (s[s.Length / 2 - 1] + s[s.Length / 2]);
    }

    private static double[] Kl(List<float[]> p, List<float[]> q)
    {
        var kl = new double[p.Count];
        for (int s = 0; s < kl.Length; s++)
        {
            double[] lp = LogSoftmax(p[s]), lq = LogSoftmax(q[s]);
            double acc = 0;
            for (int i = 0; i < lp.Length; i++) acc += Math.Exp(lp[i]) * (lp[i] - lq[i]);
            kl[s] = acc;
        }
        return kl;
    }

    private IDisposable Load(out CudaTransformerModel model, out int[] prompt)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = FindGguf();
        Skip.If(path is null, "Qwen3-4B Q4_K_M GGUF not found (DOTLLM_QWEN3_4B_GGUF).");
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
