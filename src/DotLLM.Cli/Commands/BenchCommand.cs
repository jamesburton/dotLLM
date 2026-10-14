using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
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
/// Benchmark command: loads the model once and measures decode throughput at one or more
/// context depths (checkpoint sweep). Avoids the ~70s cold-reload penalty per depth point
/// that afflicts separate <c>dotllm run</c> invocations.
/// </summary>
internal sealed class BenchCommand : AsyncCommand<BenchCommand.Settings>
{
    /// <summary>
    /// Default text used to build prompts at arbitrary token depths. A single passage is
    /// repeated until the target depth is reached, then truncated to exact token count.
    /// </summary>
    private const string FillText =
        "The transformer architecture, introduced in the 2017 paper Attention Is All You Need "
        + "by Vaswani et al., revolutionized natural language processing by replacing recurrent "
        + "neural networks with a self-attention mechanism. Unlike RNNs, which process tokens "
        + "sequentially, transformers can attend to all positions in a sequence simultaneously, "
        + "enabling massive parallelism during training. The architecture consists of an encoder "
        + "and decoder, each built from stacked layers of multi-head self-attention and "
        + "position-wise feed-forward networks. Layer normalization and residual connections "
        + "stabilize training. Positional encodings, originally sinusoidal, inject sequence order "
        + "information since the model has no inherent notion of position. Modern variants like "
        + "GPT use only the decoder stack with causal masking, while BERT uses only the encoder "
        + "with bidirectional attention. Rotary Position Embeddings (RoPE) have largely replaced "
        + "sinusoidal encodings, enabling better length generalization. Grouped-Query Attention "
        + "(GQA) reduces the KV-cache memory footprint by sharing key-value heads across multiple "
        + "query heads, which is critical for efficient inference at long context lengths. "
        + "Large language models have become a cornerstone of modern artificial intelligence, "
        + "demonstrating remarkable capabilities in text generation, reasoning, and code synthesis. "
        + "Quantization reduces the precision of model weights from 16-bit floating point to lower "
        + "formats, dramatically reducing memory requirements and improving throughput. ";

    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID.")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--depth")]
        [Description("Single context depth (number of prefill tokens). Mutually exclusive with --depth-list.")]
        public int? Depth { get; set; }

        [CommandOption("--depth-list")]
        [Description("Comma-separated context depths for a multi-point sweep in one process. " +
                     "E.g. '512,1024,2048,4096'. Model loads once; KV-cache is reset between checkpoints.")]
        public string? DepthList { get; set; }

        [CommandOption("--max-tokens|-n")]
        [Description("Maximum decode tokens per checkpoint.")]
        [DefaultValue(20)]
        public int MaxTokens { get; set; } = 20;

        [CommandOption("--warmup")]
        [Description("Number of warmup repetitions (discarded). 0 skips warmup entirely.")]
        [DefaultValue(1)]
        public int Warmup { get; set; } = 1;

        [CommandOption("--threads")]
        [Description("Number of CPU threads. 0 = auto/all cores (default).")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode. 0 = auto.")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        [CommandOption("--numa-pin")]
        [Description("Pin workers to NUMA-local cores.")]
        [DefaultValue(false)]
        public bool NumaPin { get; set; }

        [CommandOption("--pcore-only")]
        [Description("Pin workers to P-cores only (Intel hybrid architectures).")]
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
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }

        [CommandOption("--prompt|-p")]
        [Description("Custom prompt text. If shorter than max depth, it is repeated. " +
                     "If omitted, a built-in filler passage is used.")]
        public string? Prompt { get; set; }

        [CommandOption("--json")]
        [Description("Output results as JSON (one envelope with an array of per-checkpoint rows).")]
        [DefaultValue(false)]
        public bool Json { get; set; }

        /// <summary>KV-cache key quantization type.</summary>
        [CommandOption("--cache-type-k")]
        [Description("KV-cache key quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeK { get; set; } = "f32";

        /// <summary>KV-cache value quantization type.</summary>
        [CommandOption("--cache-type-v")]
        [Description("KV-cache value quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeV { get; set; } = "f32";

        /// <summary>Mixed-precision window size for KV-cache quantization.</summary>
        [CommandOption("--cache-window")]
        [Description("Mixed-precision window: recent N tokens in full precision (0 = all quantized).")]
        [DefaultValue(0)]
        public int CacheWindow { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        // ── Parse depths ──────────────────────────────────────────────────
        int[] depths = ParseDepths(settings);
        if (depths.Length == 0)
        {
            WriteError(settings, "Specify --depth or --depth-list.");
            return 1;
        }

        if (settings.Depth.HasValue && !string.IsNullOrEmpty(settings.DepthList))
        {
            WriteError(settings, "--depth and --depth-list are mutually exclusive.");
            return 1;
        }

        // Sort ascending so we can share the prefilled KV-cache prefix across checkpoints.
        Array.Sort(depths);

        // ── Resolve model ─────────────────────────────────────────────────
        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        // ── Load model (once) ─────────────────────────────────────────────
        GgufFile gguf = null!;
        ModelConfig config = null!;
        Tokenizers.Bpe.BpeTokenizer tokenizer = null!;
        IModel model = null!;

        void LoadModel()
        {
            gguf = GgufFile.Open(resolvedPath);
            config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

            int gpuLayers = ResolveGpuLayers(settings, config);
            if (gpuLayers <= 0)
            {
                model = TransformerModel.LoadFromGguf(gguf, config,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly));
            }
            else if (gpuLayers >= config.NumLayers)
            {
                int gpuId = ParseGpuId(settings.Device);
                model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
            }
            else
            {
                int gpuId = ParseGpuId(settings.Device);
                model = DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly));
            }
        }

        var loadSw = Stopwatch.StartNew();
        if (settings.Json)
        {
            LoadModel();
        }
        else
        {
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Loading model...", _ => LoadModel());
        }
        loadSw.Stop();
        double loadMs = loadSw.Elapsed.TotalMilliseconds;

        // ── Build prompt tokens ───────────────────────────────────────────
        int maxDepth = depths[^1];
        int[] promptIds = BuildPromptIds(tokenizer, maxDepth, settings.Prompt);
        if (promptIds.Length < maxDepth)
        {
            WriteError(settings, $"Could not build a prompt of {maxDepth} tokens (got {promptIds.Length}). " +
                                 "Provide a longer --prompt or reduce --depth-list.");
            model.Dispose();
            gguf.Dispose();
            return 1;
        }

        // ── KV-cache factory ──────────────────────────────────────────────
        var kvConfig = new KvCacheConfig(
            KvCacheConfig.ParseDType(settings.CacheTypeK),
            KvCacheConfig.ParseDType(settings.CacheTypeV),
            settings.CacheWindow);

        Func<ModelConfig, int, Core.Attention.IKvCache>? kvFactory = BuildKvFactory(model, settings, config, kvConfig);

        // ── Run benchmark ─────────────────────────────────────────────────
        long fileSize = new FileInfo(resolvedPath).Length;
        long modelWeightsBytes = fileSize - gguf.DataSectionOffset;

        var results = new List<BenchJsonResult>();

        if (!settings.Json)
        {
            var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
            var threadingInfo = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);
            var deviceLabel = model switch
            {
                DotLLM.Cuda.CudaTransformerModel => DotLLM.Cuda.CudaDevice.GetDevice(ParseGpuId(settings.Device)).ToString(),
                DotLLM.Cuda.HybridTransformerModel h => $"hybrid {h.NumGpuLayers}gpu/{config.NumLayers - h.NumGpuLayers}cpu",
                _ => $"{threadingInfo.EffectiveThreadCount} threads"
            };
            var segments = $"{config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {deviceLabel}";
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule($"[grey]dotllm bench | {Markup.Escape(segments)} | loaded in {loadMs:N0} ms[/]").LeftJustified());
            AnsiConsole.MarkupLine($"[dim]Depths: {string.Join(", ", depths)} | warmup: {settings.Warmup} | decode: {settings.MaxTokens} tok[/]");
            AnsiConsole.WriteLine();
        }

        try
        {
            for (int di = 0; di < depths.Length; di++)
            {
                int depth = depths[di];

                if (!settings.Json)
                    AnsiConsole.MarkupLine($"[bold]Checkpoint {di + 1}/{depths.Length}: depth={depth}[/]");

                // Warmup runs (discarded)
                for (int w = 0; w < settings.Warmup; w++)
                {
                    if (!settings.Json)
                        AnsiConsole.MarkupLine($"  [dim]warmup {w + 1}/{settings.Warmup}...[/]");
                    RunOneCheckpoint(model, tokenizer, config, kvFactory, promptIds, depth,
                        settings.MaxTokens, kvConfig, out _, out _, out _);
                }

                // Measured run
                if (!settings.Json)
                    AnsiConsole.MarkupLine("  [dim]measured run...[/]");

                RunOneCheckpoint(model, tokenizer, config, kvFactory, promptIds, depth,
                    settings.MaxTokens, kvConfig,
                    out InferenceTimings timings, out int generated, out long kvCacheBytes);

                double prefillMs = timings.PrefillTimeMs;
                double decodeMs = timings.DecodeTimeMs;
                double samplingMs = timings.SamplingTimeMs;
                int prefillTokens = timings.PrefillTokenCount;
                int decodeSteps = timings.DecodeTokenCount;
                double prefillTokS = prefillTokens > 0 ? prefillTokens / (prefillMs / 1000.0) : 0;
                double decodeTokS = decodeSteps > 0 ? decodeSteps / (decodeMs / 1000.0) : 0;

                long computeBytes = model.ComputeMemoryBytes;
                long repackedBytes = model.RepackedWeightBytes;
                long totalMemory = modelWeightsBytes + repackedBytes + computeBytes + kvCacheBytes;

                var result = new BenchJsonResult
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
                        LoadMs = Math.Round(loadMs, 1),
                        PrefillMs = Math.Round(prefillMs, 1),
                        DecodeMs = Math.Round(decodeMs, 1),
                        SamplingMs = Math.Round(samplingMs, 1),
                        TotalMs = Math.Round(prefillMs + decodeMs + samplingMs, 1),
                        PrefillTokS = Math.Round(prefillTokS, 2),
                        DecodeTokS = Math.Round(decodeTokS, 2),
                        GeneratedTokens = generated,
                    },
                    Memory = new RunMemoryDto
                    {
                        WeightsBytes = modelWeightsBytes,
                        RepackedBytes = repackedBytes,
                        ComputeBytes = computeBytes,
                        KvCacheBytes = kvCacheBytes,
                        TotalBytes = totalMemory,
                    },
                };
                results.Add(result);

                if (!settings.Json)
                {
                    AnsiConsole.MarkupLine(
                        $"  prefill [green]{prefillTokS:F2} tok/s[/] ({prefillMs:N1} ms, {prefillTokens} tok) | " +
                        $"decode [green]{decodeTokS:F2} tok/s[/] ({decodeMs:N1} ms, {decodeSteps} tok)");
                    AnsiConsole.WriteLine();
                }
            }

            // ── Output ────────────────────────────────────────────────────────
            if (settings.Json)
            {
                var envelope = new BenchJsonEnvelope
                {
                    Model = Path.GetFileName(resolvedPath),
                    Architecture = config.Architecture.ToString(),
                    MaxTokens = settings.MaxTokens,
                    Warmup = settings.Warmup,
                    Depths = depths,
                    Results = results.ToArray(),
                };
                Console.WriteLine(JsonSerializer.Serialize(envelope, CliJsonContext.Default.BenchJsonEnvelope));
            }
            else
            {
                // Summary table
                var table = new Table()
                    .Border(TableBorder.Rounded)
                    .AddColumn("Depth")
                    .AddColumn(new TableColumn("Prefill tok/s").RightAligned())
                    .AddColumn(new TableColumn("Decode tok/s").RightAligned())
                    .AddColumn(new TableColumn("Prefill ms").RightAligned())
                    .AddColumn(new TableColumn("Decode ms").RightAligned())
                    .AddColumn(new TableColumn("KV Cache").RightAligned());

                foreach (var r in results)
                {
                    table.AddRow(
                        r.Depth.ToString("N0"),
                        r.Timings.PrefillTokS.ToString("F2"),
                        r.Timings.DecodeTokS.ToString("F2"),
                        r.Timings.PrefillMs.ToString("N1"),
                        r.Timings.DecodeMs.ToString("N1"),
                        FormatHelpers.FormatMiB(r.Memory.KvCacheBytes));
                }

                AnsiConsole.Write(table);
            }
        }
        finally
        {
            model.Dispose();
            gguf.Dispose();
        }

        return 0;
    }

    /// <summary>
    /// Runs a single checkpoint: prefill up to <paramref name="depth"/> tokens, then decode
    /// <paramref name="maxTokens"/> tokens. Each call allocates a fresh KV-cache, so state
    /// does not leak between checkpoints.
    /// </summary>
    private static void RunOneCheckpoint(
        IModel model, ITokenizer tokenizer, ModelConfig config,
        Func<ModelConfig, int, Core.Attention.IKvCache>? kvFactory,
        int[] fullPromptIds, int depth, int maxTokens,
        KvCacheConfig kvConfig,
        out InferenceTimings timings, out int generated, out long kvCacheBytes)
    {
        // Slice prompt to the requested depth
        var promptSlice = fullPromptIds.AsSpan(0, depth);

        var generator = new TextGenerator(model, tokenizer, kvFactory);
        var options = new InferenceOptions
        {
            Temperature = 0f, // greedy for reproducibility
            MaxTokens = maxTokens,
            Threading = new ThreadingConfig(0, 0),
        };

        // Use the sync Generate overload — benchmarks don't need streaming.
        // We encode from token IDs by decoding back to text. This preserves the exact depth.
        string promptText = tokenizer.Decode(promptSlice);
        var response = generator.Generate(promptText, options);

        timings = response.Timings;
        generated = response.GeneratedTokenCount;

        // Estimate KV-cache bytes
        if (timings.KvCacheBytes > 0)
        {
            kvCacheBytes = timings.KvCacheBytes;
        }
        else
        {
            int cacheSize = Math.Min(depth + maxTokens, config.MaxSequenceLength);
            kvCacheBytes = (long)config.NumLayers * 2 * cacheSize
                * config.NumKvHeads * config.HeadDim * sizeof(float);
        }
    }

    /// <summary>Parse --depth / --depth-list into a sorted array of depth values.</summary>
    private static int[] ParseDepths(Settings settings)
    {
        if (settings.Depth.HasValue && string.IsNullOrEmpty(settings.DepthList))
            return [settings.Depth.Value];

        if (!string.IsNullOrEmpty(settings.DepthList))
        {
            var parts = settings.DepthList.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var list = new List<int>(parts.Length);
            foreach (var part in parts)
            {
                if (int.TryParse(part, out int d) && d > 0)
                    list.Add(d);
            }
            return list.Distinct().Order().ToArray();
        }

        return [];
    }

    /// <summary>
    /// Builds an array of token IDs that is at least <paramref name="targetLen"/> tokens long
    /// by repeating a fill text (or custom prompt).
    /// </summary>
    private static int[] BuildPromptIds(ITokenizer tokenizer, int targetLen, string? customPrompt)
    {
        string baseText = customPrompt ?? FillText;

        // Encode the base text once to estimate tokens-per-char ratio
        int[] baseIds = tokenizer.Encode(baseText);
        if (baseIds.Length >= targetLen)
            return baseIds[..targetLen];

        // Repeat until we exceed the target
        int reps = (targetLen / baseIds.Length) + 2;
        var sb = new System.Text.StringBuilder(baseText.Length * reps);
        for (int i = 0; i < reps; i++)
            sb.Append(baseText);

        int[] ids = tokenizer.Encode(sb.ToString());
        return ids.Length >= targetLen ? ids[..targetLen] : ids;
    }

    /// <summary>Build a KV-cache factory matching the RunCommand pattern.</summary>
    private static Func<ModelConfig, int, Core.Attention.IKvCache>? BuildKvFactory(
        IModel model, Settings settings, ModelConfig config, KvCacheConfig kvConfig)
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
            return (cfg, size) => new Engine.KvCache.QuantizedKvCache(
                cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }

        return null; // default SimpleKvCache
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
        var match = Regex.Match(Path.GetFileName(resolvedPath), @"\.(Q[\w]+)\.gguf$", RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value : "unknown";
    }

    private static void WriteError(Settings settings, string message)
    {
        if (settings.Json)
            Console.Error.WriteLine($"Error: {message}");
        else
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(message)}[/]");
    }
}
