using BenchmarkDotNet.Attributes;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.HuggingFace;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Benchmarks;

/// <summary>
/// Pure-prefill benchmark (llama.cpp <c>pp512</c> equivalent). Measures a single
/// <see cref="TransformerModel.Forward"/> call on an N-token synthetic sequence
/// with a fresh KV-cache, isolating the prefill kernel path from tokenizer,
/// sampling, and HTTP overhead.
/// </summary>
/// <remarks>
/// Env var <c>DOTLLM_BENCH_MODEL_PATH</c> overrides the default SmolLM-135M model.
/// Env var <c>DOTLLM_BENCH_PREFILL_TOKENS</c> overrides the prompt length (default 512).
/// </remarks>
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public unsafe class PrefillBenchmarks
{
    private const string DefaultModelRepo = "QuantFactory/SmolLM-135M-GGUF";
    private const string DefaultModelFile = "SmolLM-135M.Q8_0.gguf";
    private const int DefaultModelSizeMB = 145;

    private GgufFile _gguf = null!;
    private IModel _model = null!;
    private IKvCache _kvCache = null!;
    private int[] _tokenIds = null!;
    private int[] _positions = null!;
    private int _prefillTokens;

    [GlobalSetup]
    public void Setup()
    {
        var envTokens = Environment.GetEnvironmentVariable("DOTLLM_BENCH_PREFILL_TOKENS");
        _prefillTokens = !string.IsNullOrEmpty(envTokens) && int.TryParse(envTokens, out var parsed)
            ? parsed
            : 512;

        var envModelPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        string modelPath = !string.IsNullOrEmpty(envModelPath) && File.Exists(envModelPath)
            ? envModelPath
            : DownloadModel(DefaultModelRepo, DefaultModelFile, DefaultModelSizeMB);

        _gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        _model = TransformerModel.LoadFromGguf(_gguf, config, ThreadingConfig.Auto);

        _tokenIds = new int[_prefillTokens];
        _positions = new int[_prefillTokens];
        var rng = new Random(42);
        int vocab = config.VocabSize;
        for (int i = 0; i < _prefillTokens; i++)
        {
            _tokenIds[i] = rng.Next(1, Math.Min(vocab, 32000));
            _positions[i] = i;
        }

        _kvCache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, _prefillTokens);

        Console.WriteLine($"Model: {Path.GetFileName(modelPath)} ({config.Architecture})");
        Console.WriteLine($"Prefill tokens: {_prefillTokens}");
        Console.WriteLine($"Device: CPU ({ThreadingConfig.Auto.EffectiveThreadCount} threads)");
    }

    [Benchmark(Description = "Prefill forward pass (fresh KV-cache)")]
    public void Prefill()
    {
        // Rollback to 0 so every iteration measures a true prefill from position 0.
        _kvCache.Rollback(0);
        using var logits = _model.Forward(_tokenIds, _positions, deviceId: -1, _kvCache);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _kvCache?.Dispose();
        _model?.Dispose();
        _gguf?.Dispose();
    }

    private static string DownloadModel(string repoId, string filename, int approxMB)
    {
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache");
        string cachedPath = Path.Combine(cacheDir, repoId.Replace('/', Path.DirectorySeparatorChar), filename);
        if (File.Exists(cachedPath))
            return cachedPath;

        Console.WriteLine($"Downloading {repoId}/{filename} (~{approxMB} MB)...");
        using var downloader = new HuggingFaceDownloader();
        return downloader.DownloadFileAsync(repoId, filename, cacheDir).GetAwaiter().GetResult();
    }
}
