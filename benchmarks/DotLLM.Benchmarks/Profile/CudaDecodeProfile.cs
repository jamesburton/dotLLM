using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.HuggingFace;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Standalone profiler for the CUDA decode forward pass. Measures
///
///   wall_ms   — Stopwatch around <c>CudaTransformerModel.Forward</c> (includes
///               host dispatch + stream sync + final D2H memcpy of FP32 logits).
///   gpu_ms    — <c>cuEventElapsedTime</c> between the first and last kernel of
///               the launch sequence (pure GPU wallclock; excludes the cost of
///               <c>cuStreamSynchronize</c> returning to host).
///   overhead  — wall_ms − gpu_ms; bounds the host-side dispatch + sync round-trip.
///
/// The ratio <c>gpu_ms / wall_ms</c> is the single decision input for whether
/// CUDA Graphs (collapses host dispatch into one packet submission) is the right
/// next step. Below ~70 % means launches dominate; above ~85 % means kernels
/// themselves are the bottleneck and fusion / better algorithms matter more.
/// </summary>
internal static class CudaDecodeProfile
{
    private const string DefaultRepoId = "QuantFactory/SmolLM-135M-GGUF";
    private const string DefaultFilename = "SmolLM-135M.Q4_K_M.gguf";
    private const string DefaultPrompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is";
    private const int DefaultPrefillTokens = 96;
    private const int DefaultDecodeTokens = 200;
    private const int DefaultWarmupTokens = 16;

    public static int Run(string[] args)
    {
        if (!CudaDevice.IsAvailable())
        {
            Console.Error.WriteLine("CUDA device not available — install driver/toolkit and retry.");
            return 2;
        }

        // --graph        : (legacy, now redundant) force CUDA-Graphs decode path
        // --no-graph     : force eager decode (overrides the new default-on)
        // --compare      : run BOTH eager and graph back-to-back, side-by-side report
        // --no-profiling : measure eager wall WITHOUT cuEventRecord overhead so the
        //                  comparison vs graph is true-apples-to-apples
        // (default)      : graph capture (mirrors CudaTransformerModel default-on; flipped
        //                  in commit "CUDA: graph capture default-ON for all k").
        bool noGraph = args.Contains("--no-graph") ||
                       Environment.GetEnvironmentVariable("DOTLLM_DISABLE_GRAPH_CAPTURE") == "1";
        bool useGraph = !noGraph;
        bool compare = args.Contains("--compare");
        bool noProfiling = args.Contains("--no-profiling");
        // --kv-quant : run with the mixed-precision quantized KV cache
        //              (Q8_0 stored region + 16-row FP16 window). Validates the
        //              quantized-cache CUDA Graphs decode path lands the same
        //              ~2× speedup as the FP16 cache.
        bool kvQuant = args.Contains("--kv-quant");
        KvCacheConfig kvCfg = kvQuant
            ? new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0, MixedPrecisionWindowSize: 16)
            : KvCacheConfig.Default;

        string modelPath = ResolveModelPath();
        Console.WriteLine($"Model: {modelPath}");
        Console.WriteLine($"Device: {CudaDevice.GetDevice(0)}");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] promptTokens = tokenizer.Encode(DefaultPrompt).ToArray();
        Console.WriteLine($"Prompt tokens: {promptTokens.Length} (using up to {DefaultPrefillTokens})");

        if (compare)
        {
            string cacheLabel = kvQuant ? "KV-quant Q8_0 + W16 window" : "FP16 KV";
            Console.WriteLine($"Cache config: {cacheLabel}");
            Console.WriteLine();
            Console.WriteLine("════════ EAGER (with per-category profiling) ════════");
            var eagerResult = RunOne(gguf, config, promptTokens, useGraphCapture: false, disableProfiling: false, kvCfg: kvCfg);
            Console.WriteLine();
            Console.WriteLine("════════ EAGER (no profiling — true wall) ════════");
            var eagerCleanResult = RunOne(gguf, config, promptTokens, useGraphCapture: false, disableProfiling: true, kvCfg: kvCfg);
            Console.WriteLine();
            Console.WriteLine("════════ CUDA GRAPH (capture+replay) ════════");
            var graphResult = RunOne(gguf, config, promptTokens, useGraphCapture: true, disableProfiling: true, kvCfg: kvCfg);
            Console.WriteLine();
            Console.WriteLine("════════ SUMMARY ════════");
            Console.WriteLine($"  cache config                = {cacheLabel}");
            Console.WriteLine($"  eager+profile  median tok/s = {1000.0 / eagerResult.MedianWallMs,7:F1}  (wall {eagerResult.MedianWallMs:F2} ms)");
            Console.WriteLine($"  eager (clean)  median tok/s = {1000.0 / eagerCleanResult.MedianWallMs,7:F1}  (wall {eagerCleanResult.MedianWallMs:F2} ms)");
            Console.WriteLine($"  graph          median tok/s = {1000.0 / graphResult.MedianWallMs,7:F1}  (wall {graphResult.MedianWallMs:F2} ms)");
            Console.WriteLine($"  speedup vs clean eager      = {eagerCleanResult.MedianWallMs / graphResult.MedianWallMs,7:F2}×");
            return 0;
        }

        var single = RunOne(gguf, config, promptTokens, useGraphCapture: useGraph, disableProfiling: noProfiling, kvCfg: kvCfg);
        return 0;
    }

    private readonly struct DecodeStats
    {
        public required double MedianWallMs { get; init; }
        public required double MedianGpuMs { get; init; }
    }

    private static DecodeStats RunOne(GgufFile gguf, DotLLM.Core.Models.ModelConfig config,
                                       int[] promptTokens, bool useGraphCapture,
                                       bool disableProfiling = false,
                                       KvCacheConfig kvCfg = default)
    {
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0);
        model.UseGraphCapture = useGraphCapture;

        int prefillLen = Math.Min(promptTokens.Length, DefaultPrefillTokens);
        int[] prefill = promptTokens[..prefillLen];

        int kvCapacity = prefillLen + DefaultWarmupTokens + DefaultDecodeTokens + 8;
        using var kv = kvCfg.IsQuantized
            ? (IKvCache)model.CreateKvCache(kvCapacity, kvCfg)
            : (IKvCache)model.CreateKvCache(kvCapacity);

        int[] prefillPositions = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) prefillPositions[i] = i;
        using (var _ = model.Forward(prefill, prefillPositions, deviceId: 0, kv))
        { }

        int nextPos = prefillLen;
        int currentToken = promptTokens[prefillLen - 1];

        int[] tokBuf = new int[1];
        int[] posBuf = new int[1];

        for (int i = 0; i < DefaultWarmupTokens; i++)
        {
            tokBuf[0] = currentToken;
            posBuf[0] = nextPos;
            using var t = model.Forward(tokBuf, posBuf, deviceId: 0, kv);
            currentToken = ArgmaxFirstRow(t);
            nextPos++;
        }

        // Per-category profiling is disabled on the graph path (event-record between
        // launches breaks stream capture). For the graph path we still get wall-clock
        // and a single GPU bracket via the cuEvent on entry/exit of the graph launch.
        int categoryCount = CudaTransformerModel.ProfileCategoryCount;
        var wallTimes = new double[DefaultDecodeTokens];
        var gpuTimes = new double[DefaultDecodeTokens];
        var categoryTimes = new double[categoryCount, DefaultDecodeTokens];
        var sw = new Stopwatch();

        if (!useGraphCapture && !disableProfiling)
        {
            // Eager: full per-category profiling.
            model.ProfilingEnabled = true;
            for (int i = 0; i < DefaultDecodeTokens; i++)
            {
                tokBuf[0] = currentToken;
                posBuf[0] = nextPos;
                sw.Restart();
                using var t = model.Forward(tokBuf, posBuf, deviceId: 0, kv);
                sw.Stop();
                wallTimes[i] = sw.Elapsed.TotalMilliseconds;
                gpuTimes[i] = model.LastGpuLaunchMs;
                for (int c = 0; c < categoryCount; c++)
                    categoryTimes[c, i] = model.LastCategoryMs[c];
                currentToken = ArgmaxFirstRow(t);
                nextPos++;
            }
        }
        else
        {
            // Graph, or eager with profiling disabled: wall only.
            model.ProfilingEnabled = false;
            for (int i = 0; i < DefaultDecodeTokens; i++)
            {
                tokBuf[0] = currentToken;
                posBuf[0] = nextPos;
                sw.Restart();
                using var t = model.Forward(tokBuf, posBuf, deviceId: 0, kv);
                sw.Stop();
                wallTimes[i] = sw.Elapsed.TotalMilliseconds;
                gpuTimes[i] = double.NaN;
                currentToken = ArgmaxFirstRow(t);
                nextPos++;
            }
        }

        // Reporting: show per-category breakdown only when we captured it.
        // hideCategoryBreakdown is semantically what Report's last arg means in practice.
        bool hadProfiling = !useGraphCapture && !disableProfiling;
        Report(wallTimes, gpuTimes, categoryTimes, prefillLen, kvCapacity,
               config.NumLayers, config.HiddenSize,
               useGraphCapture: useGraphCapture, hideCategory: !hadProfiling);

        var sortedWall = (double[])wallTimes.Clone();
        Array.Sort(sortedWall);
        var sortedGpu = (double[])gpuTimes.Clone();
        Array.Sort(sortedGpu);
        return new DecodeStats { MedianWallMs = Median(sortedWall), MedianGpuMs = Median(sortedGpu) };
    }

    private static unsafe int ArgmaxFirstRow(DotLLM.Core.Tensors.ITensor logits)
    {
        // logits shape [1, vocab]; FP32; already on host after Forward()'s D2H
        int n = checked((int)logits.Shape.ElementCount);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int best = 0;
        float v = span[0];
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > v) { v = span[i]; best = i; }
        }
        return best;
    }

    private static readonly string[] CategoryNames =
    {
        "Embed", "QkvProj", "Rope+Bias+QkNorm", "KvUpdate", "Attention",
        "OProj", "Norm (rmsnorm/fused-add)", "MlpUp (gate+up)", "Swiglu",
        "MlpDown", "LmHead", "Convert+ResidAdd"
    };

    private static void Report(double[] wall, double[] gpu, double[,] categoryTimes,
                                int prefillLen, int kvCapacity, int layers, int hidden,
                                bool useGraphCapture = false, bool hideCategory = false)
    {
        Array.Sort(wall);
        var gpuSorted = (double[])gpu.Clone();
        Array.Sort(gpuSorted);

        double wallMedian = Median(wall);
        double gpuMedian = Median(gpuSorted);
        double wallP10 = Percentile(wall, 10);
        double wallP90 = Percentile(wall, 90);
        double gpuP10 = Percentile(gpuSorted, 10);
        double gpuP90 = Percentile(gpuSorted, 90);
        double wallMin = wall[0];
        double gpuMin = gpuSorted[0];
        double overhead = wallMedian - gpuMedian;
        double gpuFraction = gpuMedian / wallMedian;
        double tokPerSec = 1000.0 / wallMedian;

        Console.WriteLine();
        Console.WriteLine("──────── CUDA decode profile ────────");
        string pathLabel = useGraphCapture ? "GRAPH" : (hideCategory ? "EAGER (no profiling)" : "EAGER");
        Console.WriteLine($"Layers={layers}  Hidden={hidden}  Prefill={prefillLen}  KvCapacity={kvCapacity}  Path={pathLabel}");
        Console.WriteLine($"Iterations: {wall.Length} timed (after {DefaultWarmupTokens} warmup)");
        Console.WriteLine();
        Console.WriteLine($"             {"min",8} {"p10",8} {"p50",8} {"p90",8}");
        Console.WriteLine($"  wall ms   {wallMin,8:F3} {wallP10,8:F3} {wallMedian,8:F3} {wallP90,8:F3}");
        if (!hideCategory)
        {
            Console.WriteLine($"  gpu  ms   {gpuMin,8:F3} {gpuP10,8:F3} {gpuMedian,8:F3} {gpuP90,8:F3}");
        }
        Console.WriteLine();
        if (!hideCategory)
        {
            Console.WriteLine($"  median wall − gpu  = {overhead,7:F3} ms  (host dispatch + sync + D2H)");
            Console.WriteLine($"  median gpu / wall  = {gpuFraction,7:P1}");
        }
        Console.WriteLine($"  median tok/s       = {tokPerSec,7:F1}");
        Console.WriteLine();

        if (hideCategory)
        {
            if (useGraphCapture)
                Console.WriteLine("Per-category breakdown disabled in graph mode (event-record between launches breaks stream capture).");
            else
                Console.WriteLine("Per-category breakdown disabled (run without --no-profiling for the full breakdown).");
            Console.WriteLine("─────────────────────────────────────");
            return;
        }

        // Per-category breakdown — sort by median time descending.
        int categoryCount = categoryTimes.GetLength(0);
        int n = categoryTimes.GetLength(1);
        var medians = new double[categoryCount];
        for (int c = 0; c < categoryCount; c++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++) col[i] = categoryTimes[c, i];
            Array.Sort(col);
            medians[c] = Median(col);
        }
        var order = Enumerable.Range(0, categoryCount).OrderByDescending(c => medians[c]).ToArray();

        Console.WriteLine("Per-category GPU time (median ms / token):");
        Console.WriteLine($"  {"category",-26} {"ms",8} {"%gpu",8}");
        double sumCat = 0;
        for (int k = 0; k < order.Length; k++)
        {
            int c = order[k];
            double pct = medians[c] / gpuMedian * 100;
            sumCat += medians[c];
            Console.WriteLine($"  {CategoryNames[c],-26} {medians[c],8:F3} {pct,7:F1}%");
        }
        double accountedPct = sumCat / gpuMedian * 100;
        Console.WriteLine($"  {"-- accounted --",-26} {sumCat,8:F3} {accountedPct,7:F1}%");
        Console.WriteLine($"  (unaccounted = event-record overhead between marks)");
        Console.WriteLine();

        if (gpuFraction < 0.70)
        {
            Console.WriteLine("VERDICT: launch-bound. CUDA Graphs is the highest-ROI fix");
            Console.WriteLine("         (collapses ~400 launches/token into one packet submission).");
        }
        else if (gpuFraction > 0.85)
        {
            Console.WriteLine("VERDICT: kernel-bound. Top-3 categories above are the targets.");
        }
        else
        {
            Console.WriteLine("VERDICT: mixed. Both CUDA Graphs and kernel fusion would help.");
        }
        Console.WriteLine("─────────────────────────────────────");
    }

    private static double Median(double[] sorted)
    {
        int n = sorted.Length;
        if (n == 0) return 0;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double Percentile(double[] sorted, double p)
    {
        if (sorted.Length == 0) return 0;
        double rank = (p / 100.0) * (sorted.Length - 1);
        int lo = (int)Math.Floor(rank);
        int hi = (int)Math.Ceiling(rank);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (rank - lo) * (sorted[hi] - sorted[lo]);
    }

    private static string ResolveModelPath()
    {
        string envPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH") ?? "";
        if (!string.IsNullOrEmpty(envPath) && File.Exists(envPath)) return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string cli = Path.Combine(home, ".dotllm", "models",
            DefaultRepoId.Replace('/', Path.DirectorySeparatorChar), DefaultFilename);
        if (File.Exists(cli)) return cli;

        string benchCacheDir = Path.Combine(home, ".dotllm", "test-cache");
        string benchCached = Path.Combine(benchCacheDir,
            DefaultRepoId.Replace('/', Path.DirectorySeparatorChar), DefaultFilename);
        if (File.Exists(benchCached)) return benchCached;

        Console.WriteLine($"Downloading {DefaultRepoId}/{DefaultFilename}...");
        using var dl = new HuggingFaceDownloader();
        return dl.DownloadFileAsync(DefaultRepoId, DefaultFilename, benchCacheDir).GetAwaiter().GetResult();
    }
}
