using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Benchmarks.Columns;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.HuggingFace;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>
/// CUDA variant of the end-to-end inference benchmark. Mirrors
/// <see cref="InferenceBenchmarks"/> but always loads the model on GPU 0 via
/// <c>CudaTransformerModel.LoadFromGguf</c>. Skips with an informative error
/// if no CUDA driver/device is present, so CI-on-CPU does not break.
///
/// Runs the same prompt + max-tokens defaults as the CPU variant; the metrics
/// key is prefixed <c>Cuda_</c> to avoid collision with CPU runs.
/// </summary>
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class CudaInferenceBenchmarks
{
    private static readonly Dictionary<BenchmarkModel, (string RepoId, string Filename, int ApproxSizeMB)> s_models = new()
    {
        // Q4_K_M is the end-to-end-verified quantization on CUDA (see
        // CudaLogitComparisonTest.CompareLogits_PrefillAndDecode_Q4KM).
        [BenchmarkModel.SmolLM_135M] = ("QuantFactory/SmolLM-135M-GGUF", "SmolLM-135M.Q4_K_M.gguf", 84),
        [BenchmarkModel.Llama32_1B] = ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 800),
        [BenchmarkModel.Llama32_3B] = ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2100),
    };

    private const string DefaultPrompt = "The capital of France is";
    private const int DefaultMaxTokens = 20;

    /// <summary>Model to benchmark. Use <c>--filter *SmolLM*</c> etc. to select one.</summary>
    [Params(BenchmarkModel.SmolLM_135M)]
    public BenchmarkModel Model { get; set; }

    private GgufFile _gguf = null!;
    private IModel _model = null!;
    private BpeTokenizer _tokenizer = null!;
    private TextGenerator _generator = null!;
    private string _modelPath = null!;
    private string _prompt = DefaultPrompt;
    private int _maxTokens = DefaultMaxTokens;
    private string _metricsKey = null!;

    private readonly List<InferenceTimings> _timings = new();

    [GlobalSetup]
    public void Setup()
    {
        // Gate-check: CUDA must be available. Throw a clear message rather than
        // letting a later P/Invoke fail with DllNotFoundException.
        if (!ProbeCudaAvailable())
        {
            throw new InvalidOperationException(
                "CUDA not available on this system (no driver or no GPU). " +
                "Skip CudaInferenceBenchmarks by filtering, e.g. --filter '*InferenceBenchmarks*'.");
        }

        var envModelPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (!string.IsNullOrEmpty(envModelPath) && File.Exists(envModelPath))
        {
            _modelPath = envModelPath;
            _metricsKey = "Cuda_" + Path.GetFileNameWithoutExtension(envModelPath);
            Console.WriteLine($"*** Model override active: {_metricsKey} ***");
            Console.WriteLine($"    Path: {envModelPath}");
        }
        else
        {
            var (repoId, filename, approxMB) = s_models[Model];
            // Prefer local model cache (CLI convention: ~/.dotllm/models/<repo>/<file>),
            // fall back to benchmark test-cache and finally HF download.
            _modelPath = ResolveModelPath(repoId, filename, approxMB);
            _metricsKey = "Cuda_" + Model.ToString();
        }

        var envPrompt = Environment.GetEnvironmentVariable("DOTLLM_BENCH_PROMPT");
        if (!string.IsNullOrEmpty(envPrompt))
            _prompt = envPrompt;

        var envMaxTokens = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MAX_TOKENS");
        if (!string.IsNullOrEmpty(envMaxTokens) && int.TryParse(envMaxTokens, out var parsedTokens))
            _maxTokens = parsedTokens;

        var promptPreview = _prompt.Length > 60 ? _prompt[..60] + "..." : _prompt;
        Console.WriteLine($"Prompt: \"{promptPreview}\", MaxTokens: {_maxTokens}");

        _gguf = GgufFile.Open(_modelPath);
        var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        _tokenizer = GgufBpeTokenizerFactory.Load(_gguf.Metadata);

        int gpuId = 0;
        var envGpuId = Environment.GetEnvironmentVariable("DOTLLM_BENCH_GPU_ID");
        if (!string.IsNullOrEmpty(envGpuId))
            int.TryParse(envGpuId, out gpuId);

        var kvFactory = LoadGpuModel(_gguf, config, gpuId);
        _generator = new TextGenerator(_model, _tokenizer, kvFactory);
    }

    /// <summary>
    /// Separated into its own method with <see cref="MethodImplOptions.NoInlining"/>
    /// so the JIT only resolves <c>DotLLM.Cuda</c> types (and their native cublas/nvcuda
    /// dependencies) when actually called — not when <c>Setup()</c> is compiled.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private Func<ModelConfig, int, IKvCache> LoadGpuModel(GgufFile gguf, ModelConfig config, int gpuId)
    {
        try
        {
            var cudaModel = Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
            _model = cudaModel;

            var device = Cuda.CudaDevice.GetDevice(gpuId);
            Console.WriteLine($"Device: GPU ({device})");
            if (!string.IsNullOrEmpty(cudaModel.VramWarning))
                Console.WriteLine($"VRAM: {cudaModel.VramWarning}");

            return (_, size) => cudaModel.CreateKvCache(size);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException(
                $"CUDA libraries not found. Install CUDA Toolkit/driver. ({ex.Message})", ex);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool ProbeCudaAvailable()
    {
        try { return Cuda.CudaDevice.IsAvailable(); }
        catch { return false; }
    }

    [Benchmark(Description = "CUDA E2E inference (prefill + decode)")]
    public InferenceResponse Inference()
    {
        var options = new InferenceOptions
        {
            Temperature = 0f, // greedy
            MaxTokens = _maxTokens
        };

        var response = _generator.Generate(_prompt, options);
        _timings.Add(response.Timings);
        return response;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_timings.Count > 0)
        {
            var prefillTokPerSecAll = _timings.Select(t => t.PrefillTokensPerSec).ToArray();
            var decodeTokPerSecAll = _timings.Select(t => t.DecodeTokensPerSec).ToArray();
            var prefillMsAll = _timings.Select(t => t.PrefillTimeMs).ToArray();
            var decodeMsAll = _timings.Select(t => t.DecodeTimeMs).ToArray();

            var prefillSorted = prefillTokPerSecAll.OrderBy(v => v).ToList();
            var decodeSorted = decodeTokPerSecAll.OrderBy(v => v).ToList();
            var prefillMsSorted = prefillMsAll.OrderBy(v => v).ToList();
            var decodeMsSorted = decodeMsAll.OrderBy(v => v).ToList();

            var metrics = new InferenceMetricsFile(
                MedianPrefillTokPerSec: Median(prefillSorted),
                MedianDecodeTokPerSec: Median(decodeSorted),
                MedianPrefillMs: Median(prefillMsSorted),
                MedianDecodeMs: Median(decodeMsSorted),
                PrefillTokenCount: _timings[0].PrefillTokenCount,
                DecodeTokenCount: _timings[0].DecodeTokenCount,
                Iterations: _timings.Count,
                BestPrefillTokPerSec: prefillTokPerSecAll.Max(),
                BestDecodeTokPerSec: decodeTokPerSecAll.Max(),
                BestPrefillMs: prefillMsAll.Min(),
                BestDecodeMs: decodeMsAll.Min(),
                DecodeCv: Cv(decodeTokPerSecAll),
                PrefillCv: Cv(prefillTokPerSecAll),
                AllDecodeTokPerSec: decodeTokPerSecAll,
                AllPrefillTokPerSec: prefillTokPerSecAll,
                AllDecodeMs: decodeMsAll,
                AllPrefillMs: prefillMsAll);

            InferenceMetricsFile.Write(_metricsKey, metrics);

            Console.WriteLine($"[{_metricsKey}] prefill={Median(prefillSorted):F1} tok/s ({metrics.PrefillTokenCount} tok), " +
                              $"decode={Median(decodeSorted):F1} tok/s ({metrics.DecodeTokenCount} tok), n={_timings.Count}");
        }

        _model?.Dispose();
        _gguf?.Dispose();
    }

    private static double Median(List<double> sorted)
    {
        int n = sorted.Count;
        if (n == 0) return 0;
        if (n % 2 == 1) return sorted[n / 2];
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double StdDev(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / (values.Length - 1));
    }

    private static double Cv(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = values.Average();
        if (mean == 0) return 0;
        return StdDev(values) / mean;
    }

    /// <summary>
    /// Looks in the CLI model cache (<c>~/.dotllm/models/&lt;repo&gt;/&lt;file&gt;</c>)
    /// before falling back to the benchmark test-cache + HF download.
    /// </summary>
    private static string ResolveModelPath(string repoId, string filename, int approxMB)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        string cliCached = Path.Combine(
            home, ".dotllm", "models",
            repoId.Replace('/', Path.DirectorySeparatorChar), filename);
        if (File.Exists(cliCached))
            return cliCached;

        string benchCacheDir = Path.Combine(home, ".dotllm", "test-cache");
        string benchCached = Path.Combine(benchCacheDir,
            repoId.Replace('/', Path.DirectorySeparatorChar), filename);
        if (File.Exists(benchCached))
            return benchCached;

        Console.WriteLine($"Downloading {repoId}/{filename} (~{approxMB} MB)...");
        using var downloader = new HuggingFaceDownloader();
        return downloader.DownloadFileAsync(repoId, filename, benchCacheDir).GetAwaiter().GetResult();
    }
}
