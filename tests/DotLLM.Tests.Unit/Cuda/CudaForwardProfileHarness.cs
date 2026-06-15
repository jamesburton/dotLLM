using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Opt-in CUDA per-stage profiling harness (G0): runs a long prefill and a decode burst with
/// <see cref="CudaTransformerModel.ProfilingEnabled"/> on, and dumps the per-category GPU ms split
/// (GEMMs vs attention vs norms vs lm-head) so the prefill-GEMM optimisation prize can be sized.
/// Opt-in via <c>DOTLLM_CUDA_PROFILE=1</c>; model via <c>DOTLLM_CUDA_PERF_GGUF</c> (defaults to the
/// cached Llama-3.2-1B Q8_0). Prefill length via <c>DOTLLM_CUDA_PROFILE_PREFILL</c> (default 256).
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaForwardProfileHarness
{
    private readonly ITestOutputHelper _output;

    public CudaForwardProfileHarness(ITestOutputHelper output) => _output = output;

    private static readonly string[] CatNames =
    {
        "Embed", "QkvProj", "Rope/extras", "KvUpdate", "Attention", "OProj",
        "Norm", "MlpUp(gate+up)", "Swiglu", "MlpDown", "LmHead", "Convert",
    };

    [SkippableFact]
    public void ProfileForwardSplit()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_CUDA_PROFILE") == "1", "DOTLLM_CUDA_PROFILE=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string model = Environment.GetEnvironmentVariable("DOTLLM_CUDA_PERF_GGUF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.IfNot(File.Exists(model), $"Model not found: {model}");

        int prefillLen = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_PROFILE_PREFILL"), out int p) ? p : 256;
        int decodeSteps = 24;
        string ptxDir = ResolvePtxDir();

        using var gguf = GgufFile.Open(model);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cuda = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        cuda.ProfilingEnabled = true;

        var rng = new Random(7);
        int vocab = config.VocabSize;
        int[] prompt = new int[prefillLen];
        int[] pos = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prompt[i] = rng.Next(1, Math.Min(vocab, 32000)); pos[i] = i; }

        _output.WriteLine($"model={Path.GetFileName(model)} prefillLen={prefillLen} decodeSteps={decodeSteps}");

        // ---- Prefill (warm up first: the FIRST forward pays one-time cuBLAS/PTX-JIT/workspace init,
        // which otherwise lands entirely in the first profiled category. Fresh cache per iteration; time the last) ----
        double prefillWall = 0;
        for (int it = 0; it < 3; it++)
        {
            using var warmCache = cuda.CreateKvCache(maxSeqLen: prefillLen + 8);
            var sw = Stopwatch.StartNew();
            using (var _ = cuda.Forward(prompt, pos, deviceId: 0, warmCache)) { }
            sw.Stop();
            prefillWall = sw.Elapsed.TotalMilliseconds;
        }
        DumpSplit($"PREFILL pp{prefillLen} (warmed)", cuda, prefillWall, prefillLen);

        using var cache = cuda.CreateKvCache(maxSeqLen: prefillLen + decodeSteps + 8);
        using (var _ = cuda.Forward(prompt, pos, deviceId: 0, cache)) { }

        // ---- Decode (steady state: average the category split over the burst) ----
        var acc = new double[CatNames.Length];
        double gpuSum = 0, wallSum = 0;
        int[] one = new int[1];
        int[] opos = new int[1];
        int tok = prompt[^1];
        for (int s = 0; s < decodeSteps; s++)
        {
            one[0] = tok; opos[0] = prefillLen + s;
            var dsw = Stopwatch.StartNew();
            using (var logits = cuda.Forward(one, opos, deviceId: 0, cache)) { tok = Argmax(logits); }
            dsw.Stop();
            if (s >= 4) // skip warmup
            {
                var cm = cuda.LastCategoryMs;
                for (int c = 0; c < acc.Length && c < cm.Length; c++) acc[c] += cm[c];
                gpuSum += cuda.LastGpuLaunchMs;
                wallSum += dsw.Elapsed.TotalMilliseconds;
            }
        }
        int n = decodeSteps - 4;
        for (int c = 0; c < acc.Length; c++) acc[c] /= n;
        DumpSplitArr($"DECODE (avg of {n})", acc, gpuSum / n, wallSum / n);
    }

    /// <summary>
    /// G1 A/B: prefill GEMM compute type CUBLAS_COMPUTE_32F vs _16F, INTERLEAVED within one warmed
    /// process so consumer-GPU clock drift cancels (a fresh-process A/B is confounded by thermal state).
    /// Reports min/median wall ms per type over interleaved reps. Opt-in <c>DOTLLM_CUDA_GEMM_AB=1</c>.
    /// </summary>
    [SkippableFact]
    public void CompareGemmComputeType()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_CUDA_GEMM_AB") == "1", "DOTLLM_CUDA_GEMM_AB=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string model = Environment.GetEnvironmentVariable("DOTLLM_CUDA_PERF_GGUF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.IfNot(File.Exists(model), $"Model not found: {model}");
        int prefillLen = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_PROFILE_PREFILL"), out int p) ? p : 256;
        int reps = 10;
        string ptxDir = ResolvePtxDir();

        using var gguf = GgufFile.Open(model);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cuda = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        var rng = new Random(7);
        int vocab = config.VocabSize;
        int[] prompt = new int[prefillLen];
        int[] pos = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prompt[i] = rng.Next(1, Math.Min(vocab, 32000)); pos[i] = i; }

        double PrefillOnce()
        {
            using var cache = cuda.CreateKvCache(maxSeqLen: prefillLen + 8);
            var sw = Stopwatch.StartNew();
            using var logits = cuda.Forward(prompt, pos, deviceId: 0, cache);
            _ = Argmax(logits); // forces host read -> sync
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        }

        // Warmup both paths.
        for (int w = 0; w < 4; w++) { CudaGemm.Use16FCompute = false; PrefillOnce(); CudaGemm.Use16FCompute = true; PrefillOnce(); }

        var t32 = new List<double>(); var t16 = new List<double>();
        for (int r = 0; r < reps; r++)
        {
            CudaGemm.Use16FCompute = false; t32.Add(PrefillOnce());
            CudaGemm.Use16FCompute = true; t16.Add(PrefillOnce());
        }
        CudaGemm.Use16FCompute = false;
        t32.Sort(); t16.Sort();
        double Med(List<double> l) => l[l.Count / 2];
        _output.WriteLine($"model={Path.GetFileName(model)} pp{prefillLen} reps={reps} (interleaved, wall ms)");
        _output.WriteLine($"  COMPUTE_32F  min={t32[0]:F2}  median={Med(t32):F2}  max={t32[^1]:F2}");
        _output.WriteLine($"  COMPUTE_16F  min={t16[0]:F2}  median={Med(t16):F2}  max={t16[^1]:F2}");
        _output.WriteLine($"  speedup(16F vs 32F) min={t32[0] / t16[0]:F3}x  median={Med(t32) / Med(t16):F3}x");
        _output.WriteLine($"  (prefill includes attention ~40% which is unaffected by the GEMM compute type)");
    }

    private void DumpSplit(string label, CudaTransformerModel m, double wallMs, int tokens)
        => DumpSplitArr(label, m.LastCategoryMs.Select(x => (double)x).ToArray(), m.LastGpuLaunchMs, wallMs);

    private void DumpSplitArr(string label, double[] cat, double gpuMs, double wallMs)
    {
        double sum = 0; for (int c = 0; c < cat.Length; c++) sum += cat[c];
        _output.WriteLine($"== {label} ==  gpu={gpuMs:F2}ms wall={wallMs:F2}ms catSum={sum:F2}ms");
        var idx = Enumerable.Range(0, Math.Min(cat.Length, CatNames.Length)).OrderByDescending(i => cat[i]);
        foreach (int i in idx)
        {
            if (cat[i] <= 0) continue;
            _output.WriteLine($"   {CatNames[i],-16} {cat[i],8:F3} ms  {(sum > 0 ? cat[i] / sum * 100 : 0),5:F1}%");
        }
        // GEMM share = QkvProj+OProj+MlpUp+MlpDown (the cuBLAS/dequant-GEMM categories)
        double gemm = cat[1] + cat[5] + cat[7] + cat[9];
        _output.WriteLine($"   --> GEMM(QKV+O+Up+Down) = {gemm:F3} ms ({(sum > 0 ? gemm / sum * 100 : 0):F1}% of catSum)  attn={cat[4]:F3} ms");
    }

    private static unsafe int Argmax(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int best = 0; float bv = span[0];
        for (int i = 1; i < n; i++) if (span[i] > bv) { bv = span[i]; best = i; }
        return best;
    }

    private static string ResolvePtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            string cand = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(cand)) return cand;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return Path.Combine(AppContext.BaseDirectory, "native", "ptx");
    }
}
