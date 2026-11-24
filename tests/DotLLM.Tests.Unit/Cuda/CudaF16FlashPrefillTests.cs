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
/// (framework 2026-11-20 over 21 rows: G3 max 7.1e-3, flash 4.3e-3; both round P to F16 for a tensor-core P.V). Since #1953 the
/// G3 arm is the FIXED G3 (P x 2^15 in its FP16 plane; James 2026-11-24). First-token logprob (James 2026-11-24, as #1928):
/// within K = 2 x the larger movement of two same-run control arms, the flash arm with its softmax scale one F32 ulp up and down
/// (<c>CudaKernels.FlashPrefillKvqScaleUlpNudge</c>); a control that does not move fails the test. It replaces a fixed 0.01,
/// which sat inside this Q4_K_M model's digitisation noise (G3 itself moved 0.016). Top-1 agreement reported, not gated (a
/// near-tie may flip on a quantized model).</item>
/// <item><b>Sensitivity (review M1, framework 2026-11-22).</b> A subtle mutant, the F16 entry's softmax scale x (1 + 1/1024), raises
/// the max KL from 5.2e-3 to 9.8e-3 against a bound of 7.2e-3 (1.25 x G3's 5.7e-3): killed by the max bar. The median only moves
/// 2.25e-9 -> 2.40e-9, so the median bar is a coarse guard (no tightening can see this mutant with margin); the max bar is the
/// sensitive one.</item>
/// <item><b>Depth (#1943).</b> <c>DOTLLM_F16_FLASH_KL_ITEMS</c> sets the prompt's item count (default 280, ~5.1K tokens; 1,780 gives
/// ~32K). <c>DOTLLM_F16_FLASH_KL_CACHE</c> names a directory where the exact and G3 arms (which do not depend on the flash kernel) are
/// stored and reused, keyed by the prompt token hash. <c>DOTLLM_F16_FLASH_KL_REF_PTX</c> adds a reported (not gated) arm: the flash
/// entry from that <c>attention_flash_mma_prefill_kvq.ptx</c> (e.g. the pre-E4 build), and KL(ref flash || flash).</item>
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
        int items = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_F16_FLASH_KL_ITEMS"), out int n) ? n : 280;
        for (int i = 0; i < items; i++)
        {
            sb.Append($"Item {i}: the {words[rng.Next(words.Length)]} near the {words[rng.Next(words.Length)]} weighs {rng.Next(1, 999)} units. ");
        }
        sb.Append($"What does item {Math.Min(17, items - 1)} weigh?<|im_end|>\n<|im_start|>assistant\n");
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
        List<float[]> exact, fresh, g3Arm, ctrlUp, ctrlDown;
        long flashLaunches, offsetRuns, scalarPrefills;
        long? g3OffsetRuns, exactScalar;   // null = that arm came from the cache, so its launch counts are not re-checked
        try
        {
            CudaG3Attention.Enabled = false;
            CudaFlashAttention.Enabled = false;
            CudaTransformerModel.EnableF16FlashPrefill = false;
            long e0 = CudaKernels.AttentionF16PrefillLaunches;
            exact = Cached("exact", prompt, () => Run(model, kv, prompt), out bool exactCached);
            exactScalar = exactCached ? null : CudaKernels.AttentionF16PrefillLaunches - e0;

            CudaG3Attention.Enabled = g3;
            CudaFlashAttention.Enabled = flash;
            long g0 = CudaG3Attention.OffsetRunCount;
            g3Arm = Cached("g3", prompt, () => Run(model, kv, prompt), out bool g3Cached);
            g3OffsetRuns = g3Cached ? null : CudaG3Attention.OffsetRunCount - g0;
            if (exactCached || g3Cached) _out.WriteLine($"exact arm cached: {exactCached}, G3 arm cached: {g3Cached} (launch counts not re-checked)");

            CudaTransformerModel.EnableF16FlashPrefill = true;
            long f0 = CudaKernels.AttentionFlashPrefillF16Launches, o0 = CudaG3Attention.OffsetRunCount;
            long s0 = CudaKernels.AttentionF16PrefillLaunches;
            fresh = Run(model, kv, prompt);
            flashLaunches = CudaKernels.AttentionFlashPrefillF16Launches - f0;
            offsetRuns = CudaG3Attention.OffsetRunCount - o0;
            scalarPrefills = CudaKernels.AttentionF16PrefillLaunches - s0;

            // #1953 sensitivity control (James 2026-11-24, as #1928): the same flash arm with the softmax scale one F32 ulp up and
            // down. Their first-token movement is what a single rounding does on this instrument; the bar is derived from it.
            try
            {
                CudaKernels.FlashPrefillKvqScaleUlpNudge = 1;
                ctrlUp = Run(model, kv, prompt);
                CudaKernels.FlashPrefillKvqScaleUlpNudge = -1;
                ctrlDown = Run(model, kv, prompt);
            }
            finally { CudaKernels.FlashPrefillKvqScaleUlpNudge = 0; }
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
        // First-token bar (James 2026-11-24): K x the larger first-token movement of the two 1-ulp control arms, measured against the
        // flash arm itself (a control that cannot move is rejected below). K = 2, as #1928's option B.
        const double K = 2.0;
        double ctrlShift = Math.Max(Math.Abs(LogProb(ctrlUp[^1], next) - lpFresh), Math.Abs(LogProb(ctrlDown[^1], next) - lpFresh));
        double lpBar = K * ctrlShift;
        int top1 = Enumerable.Range(0, Rows).Count(r => Argmax(fresh[r]) == Argmax(exact[r]));
        int top1G3 = Enumerable.Range(0, Rows).Count(r => Argmax(g3Arm[r]) == Argmax(exact[r]));
        _out.WriteLine($"prompt {prompt.Length} tokens: prefix {prefix} in chunks of {Chunk}, then {Rows} windows of {Window}; " +
                       $"{forwards} forwards per arm");
        _out.WriteLine($"F16 flash launches {flashLaunches}, G3 offset runs {offsetRuns}, attention_f16 prefill launches {scalarPrefills}; " +
                       $"G3 arm offset runs {g3OffsetRuns?.ToString() ?? "cached"}; exact arm attention_f16 prefill launches " +
                       $"{exactScalar?.ToString() ?? "cached"}");
        _out.WriteLine($"KL(exact||flash) over the last {Rows} rows: median {Median(kl):E3} max {kl.Max():E3}; " +
                       $"KL(exact||G3) median {Median(klG3):E3} max {klG3.Max():E3}");
        _out.WriteLine($"first-token logprob: exact {lpExact:R}, flash {lpFresh:R}, G3 {LogProb(g3Arm[^1], next):R}; " +
                       $"top-1 agreement flash {top1}/{Rows}, G3 {top1G3}/{Rows}");
        _out.WriteLine($"1-ulp control arms: first-token logprob +1 ulp {LogProb(ctrlUp[^1], next):R}, -1 ulp {LogProb(ctrlDown[^1], next):R}; " +
                       $"control shift {ctrlShift:E3}; bar = {K} x control = {lpBar:E3}; flash |shift| {Math.Abs(lpFresh - lpExact):E3}");

        if (Environment.GetEnvironmentVariable("DOTLLM_F16_FLASH_KL_REF_PTX") is { Length: > 0 } refPtx)
        {
            // #1943: the same flash arm from another build of the kvq kernel (reported, not gated).
            model.Dispose();
            string dir = Path.Combine(Path.GetTempPath(), $"dotllm-klref-{Environment.ProcessId}");
            Directory.CreateDirectory(dir);
            foreach (string f in Directory.GetFiles(FindPtxDir()!, "*.ptx")) File.Copy(f, Path.Combine(dir, Path.GetFileName(f)), true);
            File.Copy(refPtx, Path.Combine(dir, "attention_flash_mma_prefill_kvq.ptx"), true);
            using var gguf = GgufFile.Open(FindGguf()!);
            using var refModel = CudaTransformerModel.LoadFromGguf(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), 0, dir);
            CudaTransformerModel.EnableF16FlashPrefill = true;
            var refArm = Run(refModel, kv, prompt);
            CudaTransformerModel.EnableF16FlashPrefill = f16Flash;
            double[] klRef = Kl(exact, refArm), klRefNew = Kl(refArm, fresh);
            double lpRef = LogProb(refArm[^1], next);
            _out.WriteLine($"ref build {refPtx}: KL(exact||ref flash) median {Median(klRef):E3} max {klRef.Max():E3}; " +
                           $"KL(ref flash||flash) median {Median(klRefNew):E3} max {klRefNew.Max():E3}; top-1 ref " +
                           $"{Enumerable.Range(0, Rows).Count(r => Argmax(refArm[r]) == Argmax(exact[r]))}/{Rows}; first-token |shift| " +
                           $"{Math.Abs(lpRef - lpExact):E3} vs bar {lpBar:E3}: {(Math.Abs(lpRef - lpExact) <= lpBar ? "WITHIN" : "OUTSIDE")}; " +
                           $"max KL vs bound {Math.Max(1e-3, 1.25 * klG3.Max()):E3}: {(klRef.Max() <= Math.Max(1e-3, 1.25 * klG3.Max()) ? "WITHIN" : "OUTSIDE")}");
        }

        Assert.Equal(layers * forwards, flashLaunches);
        Assert.Equal(0, offsetRuns);
        Assert.Equal(0, scalarPrefills);
        if (g3OffsetRuns is { } g3Runs)
            Assert.True(g3Runs >= layers * Rows, $"the G3 reference arm ran its offset path only {g3Runs} times");
        if (exactScalar is { } exactRuns)
            Assert.True(exactRuns >= layers * forwards, $"the exact arm ran attention_f16 only {exactRuns} times");
        Assert.True(ctrlShift > 0, "the 1-ulp control arms did not move the first-token logprob: no bar can be derived from them");
        Assert.True(Math.Abs(lpFresh - lpExact) <= lpBar,
            $"first-token logprob moved {lpFresh - lpExact:R}, above {K} x the 1-ulp control's {ctrlShift:E3} = {lpBar:E3}");
        // James 2026-11-21: relative gate, median <= 1e-5 and max <= max(1e-3, 1.25 x G3 max), over the last 256 rows.
        Assert.True(Median(kl) <= 1e-5, $"median KL(exact||flash) {Median(kl)}");
        Assert.True(kl.Max() <= Math.Max(1e-3, 1.25 * klG3.Max()), $"max KL(exact||flash) {kl.Max()} vs G3 {klG3.Max()}");
    }

    /// <summary>
    /// Stable cache key (#1943 review I3): SHA-256 over the arm, the prompt token bytes, the GGUF's full path, size and last-write
    /// time, and the row / window shape. (System.HashCode is seeded per process, so it never hit across runs.)
    /// </summary>
    private static string CacheKey(string arm, int[] prompt)
    {
        var gguf = new FileInfo(FindGguf()!);
        using var sha = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        sha.AppendData(System.Text.Encoding.UTF8.GetBytes($"{arm}|{gguf.FullName}|{gguf.Length}|{gguf.LastWriteTimeUtc.Ticks}|r{Rows}|w{Window}|"));
        sha.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(prompt.AsSpan()));
        return Convert.ToHexString(sha.GetHashAndReset(), 0, 8).ToLowerInvariant();
    }

    /// <summary>An arm that does not depend on the flash kernel, stored under DOTLLM_F16_FLASH_KL_CACHE (keyed by <see cref="CacheKey"/>).</summary>
    private static List<float[]> Cached(string arm, int[] prompt, Func<List<float[]>> run, out bool hit)
    {
        hit = false;
        string? dir = Environment.GetEnvironmentVariable("DOTLLM_F16_FLASH_KL_CACHE");
        if (string.IsNullOrEmpty(dir)) return run();
        string path = Path.Combine(dir, $"qwen3-4b-{arm}-{prompt.Length}-{CacheKey(arm, prompt)}.bin");
        if (File.Exists(path))
        {
            using var br = new BinaryReader(File.OpenRead(path));
            int rows = br.ReadInt32(), vocab = br.ReadInt32();
            var l = new List<float[]>(rows);
            for (int r = 0; r < rows; r++)
            {
                var a = new float[vocab];
                br.BaseStream.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()));
                l.Add(a);
            }
            hit = true;
            return l;
        }
        var res = run();
        Directory.CreateDirectory(dir);
        using (var bw = new BinaryWriter(File.Create(path + ".tmp")))
        {
            bw.Write(res.Count);
            bw.Write(res[0].Length);
            foreach (var a in res) bw.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()));
        }
        File.Move(path + ".tmp", path, overwrite: true);
        return res;
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
