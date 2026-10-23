using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Strategies;
using DotLLM.HuggingFace;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>
/// Hybrid CPU-prefill / Vulkan-decode benchmark. Measures end-to-end latency
/// for the first <c>MaxTokens</c> generated tokens across a range of prompt
/// lengths, with the strategy enabled vs disabled. Validates the
/// recommendation H4 acceptance criterion: for short prompts, hybrid mode
/// should reduce first-32-tokens latency by ≥ 10 % vs pure-Vulkan baseline;
/// for long prompts, the crossover threshold gates correctly and hybrid
/// mode does not regress.
/// </summary>
/// <remarks>
/// <para>
/// Requires a host with both AVX-512 (or AVX2) CPU and a Vulkan-capable iGPU
/// in the same address space — the load-bearing assumption being that both
/// backends can mmap the same GGUF file. On Strix Halo (Ryzen AI Max+ 395 /
/// Radeon 8060S) this is the default.
/// </para>
/// <para>
/// Mode selection: <see cref="Mode"/> param chooses between <c>PureVulkan</c>
/// (existing TextGenerator path, decode-only on Vulkan, prefill also on
/// Vulkan) and <c>Hybrid</c> (CPU prefill, then handoff to Vulkan for
/// decode). <see cref="PromptTokens"/> picks one of 16 / 64 / 256 / 1024.
/// Tokens beyond the tokenised prompt are padded with a constant filler
/// token; this slightly disadvantages CPU prefill because the model still
/// has to process those tokens, so the measured win is conservative.
/// </para>
/// <para>
/// Env overrides:
/// <list type="bullet">
///   <item><c>DOTLLM_BENCH_MODEL_PATH</c> — explicit GGUF path. Defaults to TinyLlama-1.1B Q8_0.</item>
///   <item><c>DOTLLM_BENCH_HYBRID_MAX_TOKENS</c> — generated tokens per run (default 32, matching H4).</item>
///   <item><c>DOTLLM_HYBRID_PREFILL_CROSSOVER</c> — strategy crossover; not used by this bench since the mode is forced.</item>
/// </list>
/// </para>
/// </remarks>
[SimpleJob(warmupCount: 1, iterationCount: 3)]
// NOTE: Cannot be `sealed` — BenchmarkDotNet's BenchmarkConverter silently filters out
// sealed classes during discovery because its runtime toolchains (in-process Emit
// included) generate a subclass per benchmark case via Reflection.Emit. With `sealed`
// the type is unsubclassable and BDN drops it without diagnostic. See
// `.planning/notes/bdn-discovery-skip-finding.md`.
public class HybridPrefillDecodeBenchmarks
{
    private const string DefaultModelRepo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
    private const string DefaultModelFile = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    private const int DefaultModelSizeMB = 1170;
    private const string FallbackModelRepo = "QuantFactory/SmolLM-135M-GGUF";
    private const string FallbackModelFile = "SmolLM-135M.Q8_0.gguf";
    private const int FallbackModelSizeMB = 145;

    /// <summary>Generation mode.</summary>
    public enum BenchmarkMode
    {
        /// <summary>Pure Vulkan iGPU prefill + decode (baseline).</summary>
        PureVulkan,
        /// <summary>CPU prefill + Vulkan decode hybrid.</summary>
        Hybrid,
    }

    /// <summary>Prompt lengths exercised by the benchmark.</summary>
    [Params(16, 64, 256, 1024)]
    public int PromptTokens { get; set; }

    /// <summary>Mode under test.</summary>
    [Params(BenchmarkMode.PureVulkan, BenchmarkMode.Hybrid)]
    public BenchmarkMode Mode { get; set; }

    private GgufFile _cpuGguf = null!;
    private GgufFile _vkGguf = null!;
    private IModel _cpuModel = null!;
    private IModel _vkModel = null!;
    private BpeTokenizer _tokenizer = null!;
    private TextGenerator _generator = null!;
    private int _maxTokens;
    private int[] _promptIds = null!;
    private string _prompt = null!;
    private Func<ModelConfig, int, IKvCache> _kvFactory = null!;

    [GlobalSetup]
    public void Setup()
    {
        _maxTokens = 32;
        var envMax = Environment.GetEnvironmentVariable("DOTLLM_BENCH_HYBRID_MAX_TOKENS");
        if (!string.IsNullOrEmpty(envMax) && int.TryParse(envMax, out int parsed) && parsed > 0)
            _maxTokens = parsed;

        string modelPath = ResolveModelPath();

        // Load BOTH backends from the same GGUF (the OS page cache shares the
        // mmap'd pages between them — H4 mechanism). One GgufFile per backend
        // since each takes ownership of the mmap handle internally.
        _cpuGguf = GgufFile.Open(modelPath);
        var cpuConfig = GgufModelConfigExtractor.Extract(_cpuGguf.Metadata);
        _cpuModel = TransformerModel.LoadFromGguf(_cpuGguf, cpuConfig, ThreadingConfig.Auto);

        _vkGguf = GgufFile.Open(modelPath);
        var vkConfig = GgufModelConfigExtractor.Extract(_vkGguf.Metadata);
        _vkModel = LoadVulkan(_vkGguf, vkConfig);

        _tokenizer = GgufBpeTokenizerFactory.Load(_cpuGguf.Metadata);

        // Build a prompt of the desired length by repeating a base sentence
        // and tokenising once. We then truncate/pad in token space to hit the
        // exact PromptTokens target so the bench is comparable across runs.
        string basePrompt = "The capital of France is Paris and it is a beautiful city full of history. ";
        var sb = new System.Text.StringBuilder();
        while (sb.Length < PromptTokens * 8) sb.Append(basePrompt);
        int[] all = _tokenizer.Encode(sb.ToString()).ToArray();
        if (all.Length >= PromptTokens)
        {
            _promptIds = new int[PromptTokens];
            Array.Copy(all, _promptIds, PromptTokens);
        }
        else
        {
            // Pad with the last token to hit the target length.
            _promptIds = new int[PromptTokens];
            Array.Copy(all, _promptIds, all.Length);
            int pad = all.Length > 0 ? all[^1] : _tokenizer.EosTokenId;
            for (int i = all.Length; i < PromptTokens; i++) _promptIds[i] = pad;
        }
        _prompt = _tokenizer.Decode(_promptIds, stripBosSpace: false);

        _kvFactory = BuildVulkanKvFactory();

        HybridPrefillDecodeStrategy? strategy = null;
        if (Mode == BenchmarkMode.Hybrid)
        {
            strategy = BuildStrategy(_cpuModel, _vkModel);
        }

        _generator = new TextGenerator(_vkModel, _tokenizer, _kvFactory, hybridStrategy: strategy);

        Console.WriteLine($"Mode={Mode} PromptTokens={PromptTokens} (encoded={_promptIds.Length}) MaxTokens={_maxTokens}");
    }

    [Benchmark(Description = "First-32-tokens latency (prefill + decode)")]
    public InferenceResponse RunGeneration()
    {
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = _maxTokens };
        // Tokenise via our existing _prompt would re-tokenise; bypass that by
        // re-using _promptIds via a custom path is awkward here. Pay the
        // tokenizer cost once per Generate (same on both Mode arms, so the
        // delta is the metric of interest).
        return _generator.Generate(_prompt, options);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cpuModel?.Dispose();
        _vkModel?.Dispose();
        _cpuGguf?.Dispose();
        _vkGguf?.Dispose();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static IModel LoadVulkan(GgufFile gguf, ModelConfig config)
    {
        try
        {
            return Vulkan.VulkanTransformerModel.LoadFromGguf(gguf, config);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException(
                "Vulkan loader not available. Install the Vulkan SDK / runtime.", ex);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private Func<ModelConfig, int, IKvCache> BuildVulkanKvFactory()
    {
        var vk = (Vulkan.VulkanTransformerModel)_vkModel;
        return (_, size) => vk.CreateKvCache(size);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static HybridPrefillDecodeStrategy BuildStrategy(IModel cpu, IModel vk)
    {
        return new HybridPrefillDecodeStrategy(
            prefillModel: cpu,
            decodeModel: vk,
            handoff: (host, dec) =>
            {
                var vkCache = (Vulkan.VulkanKvCache)dec;
                int length = host.CurrentLength;
                for (int layer = 0; layer < host.NumLayers; layer++)
                {
                    vkCache.IngestFromHost(layer, length,
                        host.KeysSpan(layer), host.ValuesSpan(layer));
                }
                vkCache.SetCurrentLength(length);
            });
    }

    private static string ResolveModelPath()
    {
        var envModelPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (!string.IsNullOrEmpty(envModelPath) && File.Exists(envModelPath))
            return envModelPath;

        // Prefer TinyLlama (the H4 acceptance target). If the host has not yet
        // downloaded it, fall back to the smaller SmolLM-135M fixture used by
        // the integration tests so the bench still runs.
        try
        {
            return DownloadModel(DefaultModelRepo, DefaultModelFile, DefaultModelSizeMB);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HybridPrefillDecodeBenchmarks] TinyLlama download failed ({ex.Message}); falling back to SmolLM-135M.");
            return DownloadModel(FallbackModelRepo, FallbackModelFile, FallbackModelSizeMB);
        }
    }

    private static string DownloadModel(string repoId, string filename, int approxMB)
    {
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache");
        string cachedPath = Path.Combine(cacheDir,
            repoId.Replace('/', Path.DirectorySeparatorChar), filename);
        if (File.Exists(cachedPath)) return cachedPath;

        Console.WriteLine($"Downloading {repoId}/{filename} (~{approxMB} MB)...");
        using var downloader = new HuggingFaceDownloader();
        return downloader.DownloadFileAsync(repoId, filename, cacheDir).GetAwaiter().GetResult();
    }
}
