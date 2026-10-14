using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Benchmarks decode throughput at one or more context depths in a single process.
/// Loads the model once, progressively fills the KV-cache to each checkpoint,
/// and measures decode tok/s at each depth.
/// </summary>
internal sealed class BenchCommand : AsyncCommand<BenchCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID.")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--depth-list")]
        [Description("Comma-separated context depths to measure decode at (e.g., 512,2048,8192). " +
                     "The model is loaded once and context is filled progressively.")]
        public string? DepthList { get; set; }

        [CommandOption("--decode-tokens")]
        [Description("Number of decode tokens to generate at each depth checkpoint.")]
        [DefaultValue(128)]
        public int DecodeTokens { get; set; } = 128;

        [CommandOption("--warmup")]
        [Description("Number of warmup iterations before measurement (0 = skip). " +
                     "Warmup triggers JIT compilation so measured runs use optimized code.")]
        [DefaultValue(1)]
        public int Warmup { get; set; } = 1;

        [CommandOption("--prompt|-p")]
        [Description("Prompt text used to fill context. Repeated to reach each depth. " +
                     "If omitted, uses a built-in corpus.")]
        public string? Prompt { get; set; }

        [CommandOption("--prompt-file")]
        [Description("Read fill corpus from a file instead of --prompt.")]
        public string? PromptFile { get; set; }

        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference.")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode.")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        [CommandOption("--numa-pin")]
        [Description("Pin workers to NUMA-local cores.")]
        [DefaultValue(false)]
        public bool NumaPin { get; set; }

        [CommandOption("--pcore-only")]
        [Description("Pin workers to P-cores only (Intel hybrid).")]
        [DefaultValue(false)]
        public bool PCoreOnly { get; set; }

        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'gpu', 'gpu:0', 'gpu:1'.")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        [CommandOption("--gpu-layers")]
        [Description("Number of transformer layers to offload to GPU.")]
        public int? GpuLayers { get; set; }

        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist.")]
        public string? Quant { get; set; }

        [CommandOption("--cache-type-k")]
        [Description("KV-cache key quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeK { get; set; } = "f32";

        [CommandOption("--cache-type-v")]
        [Description("KV-cache value quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeV { get; set; } = "f32";

        [CommandOption("--cache-window")]
        [Description("Mixed-precision window for KV-cache quantization.")]
        [DefaultValue(0)]
        public int CacheWindow { get; set; }
    }

    /// <summary>Default fill corpus when no --prompt is given.</summary>
    private const string DefaultCorpus =
        "The transformer architecture, introduced in 2017 by Vaswani et al., " +
        "revolutionized natural language processing by replacing recurrent neural networks " +
        "with a self-attention mechanism. Unlike RNNs, which process tokens sequentially, " +
        "transformers can attend to all positions in a sequence simultaneously, enabling " +
        "massive parallelism during training. The architecture consists of an encoder and " +
        "decoder, each built from stacked layers of multi-head self-attention and " +
        "position-wise feed-forward networks. Layer normalization and residual connections " +
        "stabilize training. Positional encodings inject sequence order information. " +
        "Modern variants like GPT use only the decoder stack with causal masking. " +
        "Rotary Position Embeddings have largely replaced sinusoidal encodings. " +
        "Grouped-Query Attention reduces the KV-cache memory footprint by sharing " +
        "key-value heads across multiple query heads. ";

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        int[] depths = ParseDepthList(settings.DepthList);
        if (depths.Length == 0)
        {
            AnsiConsole.MarkupLine("[red]--depth-list is required (e.g., --depth-list 512,2048,8192)[/]");
            return 1;
        }

        string corpus;
        if (!string.IsNullOrEmpty(settings.Prompt))
            corpus = settings.Prompt;
        else if (!string.IsNullOrEmpty(settings.PromptFile))
            corpus = File.ReadAllText(settings.PromptFile);
        else
            corpus = DefaultCorpus;

        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        GgufFile gguf = null!;
        ModelConfig config = null!;
        Tokenizers.Bpe.BpeTokenizer tokenizer = null!;
        IModel model = null!;

        var loadSw = Stopwatch.StartNew();
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading model...", _ =>
            {
                gguf = GgufFile.Open(resolvedPath);
                config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
                model = LoadModel(settings, config, gguf);
            });
        loadSw.Stop();

        var threadingInfo = new ThreadingConfig(settings.Threads, settings.DecodeThreads,
            settings.NumaPin, settings.PCoreOnly);
        var kvConfig = new KvCacheConfig(
            KvCacheConfig.ParseDType(settings.CacheTypeK),
            KvCacheConfig.ParseDType(settings.CacheTypeV),
            settings.CacheWindow);

        Func<ModelConfig, int, DotLLM.Core.Attention.IKvCache>? kvFactory =
            BuildKvFactory(model, settings, kvConfig);

        int maxDepth = depths[^1];

        // Encode enough corpus tokens to fill the deepest checkpoint
        int[] corpusTokens = EncodeCorpusToLength(tokenizer, corpus, maxDepth);

        // Print config banner
        var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
        AnsiConsole.MarkupLine($"[dim]Model: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel}[/]");
        AnsiConsole.MarkupLine($"[dim]Depths: {string.Join(", ", depths)} | Decode: {settings.DecodeTokens} tokens | Warmup: {settings.Warmup}[/]");
        AnsiConsole.MarkupLine($"[dim]Load: {loadSw.Elapsed.TotalMilliseconds:N0} ms[/]");
        AnsiConsole.WriteLine();

        try
        {
            // Warmup: run a short generation to trigger JIT
            if (settings.Warmup > 0)
            {
                AnsiConsole.MarkupLine("[dim]Running warmup...[/]");
                for (int w = 0; w < settings.Warmup; w++)
                {
                    var warmupGen = new TextGenerator(model, tokenizer, kvFactory);
                    var warmupOpts = new InferenceOptions
                    {
                        MaxTokens = 16,
                        Threading = threadingInfo,
                    };
                    await foreach (var _ in warmupGen.GenerateStreamingTokensAsync(
                        "The capital of France is", warmupOpts))
                    { }
                }
            }

            var results = new List<BenchDepthResult>();

            foreach (int depth in depths)
            {
                AnsiConsole.MarkupLine($"[bold]Depth {depth}[/]");

                // For each depth, create a fresh generator and KV cache so measurements are independent.
                // Slice the corpus to exactly 'depth' tokens, decode to text, then generate from it.
                int promptLen = Math.Min(depth, corpusTokens.Length);
                int[] promptTokens = corpusTokens[..promptLen];
                string promptText = tokenizer.Decode(promptTokens);

                var generator = new TextGenerator(model, tokenizer, kvFactory);
                var opts = new InferenceOptions
                {
                    MaxTokens = settings.DecodeTokens,
                    Threading = threadingInfo,
                };

                var sw = Stopwatch.StartNew();
                InferenceTimings timings = default;
                int generated = 0;

                await foreach (var token in generator.GenerateStreamingTokensAsync(promptText, opts))
                {
                    if (token.FinishReason is null || token.Text.Length > 0)
                        generated++;
                    if (token.FinishReason.HasValue)
                        timings = token.Timings ?? default;
                }
                sw.Stop();

                double prefillMs = timings.PrefillTimeMs;
                double decodeMs = timings.DecodeTimeMs;
                int prefillTokens = timings.PrefillTokenCount;
                int decodeTokens = timings.DecodeTokenCount;
                double prefillTokS = prefillTokens > 0 && prefillMs > 0
                    ? prefillTokens / (prefillMs / 1000.0) : 0;
                double decodeTokS = decodeTokens > 0 && decodeMs > 0
                    ? decodeTokens / (decodeMs / 1000.0) : 0;

                var result = new BenchDepthResult
                {
                    Depth = depth,
                    Model = Path.GetFileName(resolvedPath),
                    Architecture = config.Architecture.ToString(),
                    Usage = new RunUsageDto
                    {
                        PromptTokens = prefillTokens,
                        GeneratedTokens = generated,
                    },
                    Timings = new RunTimingsDto
                    {
                        LoadMs = Math.Round(loadSw.Elapsed.TotalMilliseconds, 1),
                        PrefillMs = Math.Round(prefillMs, 1),
                        DecodeMs = Math.Round(decodeMs, 1),
                        SamplingMs = Math.Round(timings.SamplingTimeMs, 1),
                        TotalMs = Math.Round(sw.Elapsed.TotalMilliseconds, 1),
                        PrefillTokS = Math.Round(prefillTokS, 2),
                        DecodeTokS = Math.Round(decodeTokS, 2),
                        GeneratedTokens = generated,
                    },
                };

                results.Add(result);

                AnsiConsole.MarkupLine(
                    $"  pp={prefillTokens} ({prefillTokS:F1} tok/s, {prefillMs:F0} ms)  " +
                    $"tg={decodeTokens} ([green]{decodeTokS:F2} tok/s[/], {decodeMs:F0} ms)");
            }

            // JSON output
            AnsiConsole.WriteLine();
            string json = JsonSerializer.Serialize(results, CliJsonContext.Default.ListBenchDepthResult);
            Console.WriteLine(json);

            return 0;
        }
        finally
        {
            model.Dispose();
            gguf.Dispose();
        }
    }

    private static int[] ParseDepthList(string? depthList)
    {
        if (string.IsNullOrWhiteSpace(depthList))
            return [];

        var depths = depthList.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(s => int.TryParse(s, out int v) ? v : -1)
            .Where(v => v > 0)
            .Distinct()
            .OrderBy(v => v)
            .ToArray();
        return depths;
    }

    private static int[] EncodeCorpusToLength(ITokenizer tokenizer, string corpus, int targetLength)
    {
        int[] baseTokens = tokenizer.Encode(corpus);
        if (baseTokens.Length >= targetLength)
            return baseTokens[..targetLength];

        var result = new int[targetLength];
        for (int i = 0; i < targetLength; i++)
            result[i] = baseTokens[i % baseTokens.Length];
        return result;
    }

    private static IModel LoadModel(Settings settings, ModelConfig config, GgufFile gguf)
    {
        int gpuLayers = ResolveGpuLayers(settings, config);
        if (gpuLayers <= 0)
        {
            return TransformerModel.LoadFromGguf(gguf, config,
                new ThreadingConfig(settings.Threads, settings.DecodeThreads,
                    settings.NumaPin, settings.PCoreOnly));
        }
        if (gpuLayers >= config.NumLayers)
        {
            int gpuId = ParseGpuId(settings.Device);
            return DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
        }
        int gid = ParseGpuId(settings.Device);
        return DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gid,
            new ThreadingConfig(settings.Threads, settings.DecodeThreads,
                settings.NumaPin, settings.PCoreOnly));
    }

    private static Func<ModelConfig, int, DotLLM.Core.Attention.IKvCache>? BuildKvFactory(
        IModel model, Settings settings, KvCacheConfig kvConfig)
    {
        if (model is DotLLM.Cuda.CudaTransformerModel cudaModel)
        {
            return kvConfig.IsQuantized
                ? (cfg, size) => cudaModel.CreateKvCache(size, kvConfig)
                : (cfg, size) => cudaModel.CreateKvCache(size);
        }
        if (model is DotLLM.Cuda.HybridTransformerModel hybridModel)
            return (cfg, size) => hybridModel.CreateKvCache(size);
        if (kvConfig.IsQuantized)
        {
            return (cfg, size) => new DotLLM.Engine.KvCache.QuantizedKvCache(
                cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }
        return null;
    }

    private static int ResolveGpuLayers(Settings settings, ModelConfig config)
    {
        if (settings.GpuLayers.HasValue)
            return Math.Clamp(settings.GpuLayers.Value, 0, config.NumLayers);
        return settings.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase)
            ? config.NumLayers : 0;
    }

    private static int ParseGpuId(string device)
    {
        int colonIdx = device.IndexOf(':');
        if (colonIdx < 0) return 0;
        return int.Parse(device.AsSpan(colonIdx + 1));
    }

    private static string InferQuantLabel(string resolvedPath, string? quantFlag)
    {
        if (!string.IsNullOrEmpty(quantFlag))
            return quantFlag;
        var match = Regex.Match(Path.GetFileName(resolvedPath),
            @"\.(Q[\w]+)\.gguf$", RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value : "unknown";
    }
}
