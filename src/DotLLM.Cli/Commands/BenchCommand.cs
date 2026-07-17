using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using DotLLM.Cli.Benchmarking;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// llama-bench-equivalent benchmark: synthetic prefill of <c>-p</c> tokens plus
/// <c>-n</c> greedy decode steps, repeated <c>-r</c> times (plus one discarded
/// warm-up rep), reporting median + best prefill / decode tokens-per-second.
/// Load time is excluded from every repetition and reported separately. Also
/// prints a ready-to-paste <c>benchmarks/perf-matrix/results.csv</c> row.
/// </summary>
internal sealed class BenchCommand : Command<BenchCommand.Settings>
{
    /// <summary>Fixed seed prompt tiled to synthesize the benchmark prompt.</summary>
    internal const string DefaultSeedPrompt = "The quick brown fox jumps over the lazy dog. ";

    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'vulkan', 'cuda' / 'cuda:1' ('gpu' is an alias for cuda).")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        [CommandOption("--prompt-tokens|-p")]
        [Description("Synthetic prompt length in tokens (tiled from the seed prompt).")]
        [DefaultValue(512)]
        public int PromptTokens { get; set; } = 512;

        [CommandOption("--gen-tokens|-n")]
        [Description("Greedy decode steps to time per repetition.")]
        [DefaultValue(128)]
        public int GenTokens { get; set; } = 128;

        [CommandOption("--reps|-r")]
        [Description("Measured repetitions (one extra warm-up rep runs first and is discarded).")]
        [DefaultValue(5)]
        public int Reps { get; set; } = 5;

        [CommandOption("--depth")]
        [Description("Extra synthetic context tokens fed (untimed) after prefill, before the timed decode. " +
                     "Decode runs at context depth p + depth.")]
        [DefaultValue(0)]
        public int Depth { get; set; }

        [CommandOption("--seed-prompt")]
        [Description("Seed text tiled to synthesize the prompt.")]
        public string SeedPrompt { get; set; } = DefaultSeedPrompt;

        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }

        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto/all cores (default).")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode. 0 = auto (caps at memory channel count).")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        [CommandOption("--json")]
        [Description("Output result as a single JSON object (suppresses all formatted output).")]
        [DefaultValue(false)]
        public bool Json { get; set; }

        [CommandOption("--n-cpu-moe")]
        [Description(
            "MoE CPU/GPU expert offload (#370, llama.cpp --n-cpu-moe equivalent, Vulkan " +
            "Qwen3MoeHybrid only): put the first N MoE layers' routed experts on the CPU " +
            "instead of the GPU, trading decode/prefill throughput for reduced device " +
            "memory. 0 = fully GPU-resident/streaming (default). No-op on cpu/cuda backends " +
            "or non-Qwen3MoeHybrid Vulkan models.")]
        [DefaultValue(0)]
        public int NCpuMoeLayers { get; set; }

        public override ValidationResult Validate()
        {
            if (PromptTokens <= 0) return ValidationResult.Error("--prompt-tokens|-p must be positive.");
            if (GenTokens <= 0) return ValidationResult.Error("--gen-tokens|-n must be positive.");
            if (Reps <= 0) return ValidationResult.Error("--reps|-r must be positive.");
            if (Depth < 0) return ValidationResult.Error("--depth must be >= 0.");
            if (NCpuMoeLayers < 0) return ValidationResult.Error("--n-cpu-moe must be >= 0.");
            string dev = Device.Split(':')[0].ToLowerInvariant();
            if (dev is not ("cpu" or "vulkan" or "cuda" or "gpu"))
                return ValidationResult.Error($"Unknown --device '{Device}'. Expected cpu, vulkan, or cuda[:N].");
            return ValidationResult.Success();
        }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var ggufPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (ggufPath is null)
            return 1;

        string backend = settings.Device.Split(':')[0].ToLowerInvariant();
        if (backend == "gpu") backend = "cuda";
        int gpuId = ParseDeviceOrdinal(settings.Device);

        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] seedTokens = tokenizer.Encode(settings.SeedPrompt).ToArray();
        if (seedTokens.Length == 0)
        {
            WriteError(settings, "Seed prompt produced no tokens.");
            return 1;
        }

        // Clamp the prompt so prompt + depth + decode (+ margin) fits the model context.
        int promptLen = settings.PromptTokens;
        int budget = config.MaxSequenceLength - settings.Depth - settings.GenTokens - 8;
        if (budget <= 0)
        {
            WriteError(settings,
                $"--depth {settings.Depth} + -n {settings.GenTokens} exceeds the model's max sequence " +
                $"length ({config.MaxSequenceLength}).");
            return 1;
        }
        if (promptLen > budget)
        {
            promptLen = budget;
            if (!settings.Json)
                AnsiConsole.MarkupLine(
                    $"[yellow]WARNING: prompt clamped to {promptLen} tokens to fit the model context " +
                    $"({config.MaxSequenceLength}).[/]");
            else
                Console.Error.WriteLine($"WARNING: prompt clamped to {promptLen} tokens.");
        }
        int[] promptTokens = BenchStats.TilePrompt(seedTokens, promptLen);

        var threading = new ThreadingConfig(settings.Threads, settings.DecodeThreads);

        IModel model = null!;
        Func<int, IKvCache> kvFactory = null!;
        DotLLM.Vulkan.VulkanDevice? vulkanDevice = null;
        string deviceLabel = "";

        void LoadModel()
        {
            switch (backend)
            {
                case "cpu":
                    // THE shared per-architecture CPU dispatch (#135) — routes hybrid
                    // architectures (Nemotron-H, Qwen3MoeHybrid) to their dedicated loaders.
                    model = ModelLoader.CreateCpuModelFromGguf(gguf, config, threading);
                    kvFactory = size => new SimpleKvCache(KvGeometry.FromConfig(config), size);
                    deviceLabel = $"cpu-{threading.EffectiveThreadCount}t";
                    break;

                case "cuda":
                {
                    var cudaModel = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
                    model = cudaModel;
                    kvFactory = size => cudaModel.CreateKvCache(size);
                    deviceLabel = DotLLM.Cuda.CudaDevice.GetDevice(gpuId).Name;
                    break;
                }

                case "vulkan":
                    switch (config.Architecture)
                    {
                        case Architecture.Qwen3MoeHybrid:
                        {
                            vulkanDevice = DotLLM.Vulkan.VulkanDevice.Create();
                            var vkMoe = DotLLM.Vulkan.VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                                vulkanDevice, gguf, config, ResolveSpvDir(), settings.NCpuMoeLayers);
                            model = vkMoe;
                            kvFactory = size => vkMoe.CreateKvCache(size);
                            deviceLabel = vulkanDevice.DeviceName;
                            if (vkMoe.NCpuMoeLayers > 0 && !settings.Json)
                                AnsiConsole.MarkupLine(
                                    $"[grey]MoE CPU offload: {vkMoe.NCpuMoeLayers} layer(s), " +
                                    $"~{vkMoe.EstimatedCpuOffloadVramSavedBytes / (1024.0 * 1024.0):F0} MiB " +
                                    "GPU expert-bank upload avoided.[/]");
                            break;
                        }
                        case Architecture.NemotronH:
                            throw new NotSupportedException(
                                "Nemotron-H has no Vulkan GGUF loader yet — bench it with --device cpu.");
                        default:
                        {
                            vulkanDevice = DotLLM.Vulkan.VulkanDevice.Create();
                            var vkModel = DotLLM.Vulkan.VulkanTransformerModel.LoadFromGguf(
                                vulkanDevice, gguf, config, ResolveSpvDir());
                            model = vkModel;
                            kvFactory = size => vkModel.CreateKvCache(size);
                            deviceLabel = vulkanDevice.DeviceName;
                            break;
                        }
                    }
                    break;

                default:
                    throw new InvalidOperationException($"Unhandled backend '{backend}'.");
            }
        }

        var loadSw = Stopwatch.StartNew();
        try
        {
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
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or DllNotFoundException)
        {
            WriteError(settings, e.Message);
            return 1;
        }
        loadSw.Stop();

        try
        {
            var result = BenchRunner.Run(
                model, kvFactory, promptTokens,
                decodeTokens: settings.GenTokens,
                reps: settings.Reps,
                depth: settings.Depth,
                loadMs: loadSw.Elapsed.TotalMilliseconds);

            string quant = BenchEnvironment.InferQuantLabel(ggufPath, settings.Quant);
            string modelName = BenchEnvironment.InferModelName(ggufPath, quant);
            string runtimeVersion = BenchEnvironment.ResolveRuntimeVersion();
            string host = Environment.MachineName.ToLowerInvariant();

            string settingsSummary = BuildSettingsSummary(settings, result, backend);
            string notes = $"pp{result.PromptTokens}/tg{result.DecodeTokens} r{settings.Reps} warmup-discarded; " +
                           $"tg best {BenchStats.FormatTokS(result.DecodeTokSBest)}; load {result.LoadMs:F0} ms";
            string csvRow = BenchCsv.FormatRow(
                DateOnly.FromDateTime(DateTime.Now), host, SanitizeLabel(deviceLabel), backend,
                runtimeVersion, modelName, quant,
                result.PrefillTokSMedian, result.DecodeTokSMedian, result.DecodeCtxDepth,
                settingsSummary, notes);

            if (settings.Json)
            {
                var json = new BenchJsonResult
                {
                    Model = modelName,
                    ModelPath = ggufPath,
                    Quant = quant,
                    Backend = backend,
                    Device = deviceLabel,
                    Host = host,
                    Commit = runtimeVersion,
                    LoadMs = Math.Round(result.LoadMs, 1),
                    PromptTokens = result.PromptTokens,
                    DecodeTokens = result.DecodeTokens,
                    Depth = result.Depth,
                    DecodeCtxDepth = result.DecodeCtxDepth,
                    Warmup = BenchJsonRep.From(result.Warmup),
                    Reps = result.Reps.Select(BenchJsonRep.From).ToArray(),
                    PrefillTokSMedian = Math.Round(result.PrefillTokSMedian, 2),
                    PrefillTokSBest = Math.Round(result.PrefillTokSBest, 2),
                    DecodeTokSMedian = Math.Round(result.DecodeTokSMedian, 2),
                    DecodeTokSBest = Math.Round(result.DecodeTokSBest, 2),
                    PrefillMsMedian = Math.Round(result.PrefillMsMedian, 2),
                    PrefillMsMin = Math.Round(result.PrefillMsMin, 2),
                    DecodeMsMedian = Math.Round(result.DecodeMsMedian, 2),
                    DecodeMsMin = Math.Round(result.DecodeMsMin, 2),
                    CsvRow = csvRow,
                };
                Console.WriteLine(JsonSerializer.Serialize(json, CliJsonContext.Default.BenchJsonResult));
            }
            else
            {
                RenderTable(settings, config, result, modelName, quant, backend, deviceLabel, runtimeVersion, csvRow);
            }

            return 0;
        }
        finally
        {
            model.Dispose();
            vulkanDevice?.Dispose();
        }
    }

    private static void RenderTable(
        Settings settings, ModelConfig config, BenchResult result,
        string modelName, string quant, string backend, string deviceLabel,
        string runtimeVersion, string csvRow)
    {
        var rule = $"dotllm bench | {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | " +
                   $"{quant} | {backend} ({deviceLabel}) | {runtimeVersion}";
        AnsiConsole.Write(new Rule($"[grey]{Markup.Escape(rule)}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("rep");
        table.AddColumn(new TableColumn("prefill ms").RightAligned());
        table.AddColumn(new TableColumn("pp tok/s").RightAligned());
        table.AddColumn(new TableColumn("decode ms").RightAligned());
        table.AddColumn(new TableColumn("tg tok/s").RightAligned());

        table.AddRow(
            "[dim]warmup[/]",
            $"[dim]{result.Warmup.PrefillMs:F1}[/]",
            $"[dim]{result.Warmup.PrefillTokS:F2}[/]",
            $"[dim]{result.Warmup.DecodeMs:F1}[/]",
            $"[dim]{result.Warmup.DecodeTokS:F2}[/]");
        for (int i = 0; i < result.Reps.Count; i++)
        {
            var r = result.Reps[i];
            table.AddRow(
                (i + 1).ToString(),
                $"{r.PrefillMs:F1}", $"{r.PrefillTokS:F2}",
                $"{r.DecodeMs:F1}", $"{r.DecodeTokS:F2}");
        }
        table.AddRow(
            "[bold]median[/]",
            $"[bold]{result.PrefillMsMedian:F1}[/]", $"[bold green]{result.PrefillTokSMedian:F2}[/]",
            $"[bold]{result.DecodeMsMedian:F1}[/]", $"[bold green]{result.DecodeTokSMedian:F2}[/]");
        table.AddRow(
            "[dim]best[/]",
            $"[dim]{result.PrefillMsMin:F1}[/]", $"[dim]{result.PrefillTokSBest:F2}[/]",
            $"[dim]{result.DecodeMsMin:F1}[/]", $"[dim]{result.DecodeTokSBest:F2}[/]");

        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine(
            $"  [dim]prompt {result.PromptTokens} tok (tiled seed), decode {result.DecodeTokens} tok at ctx depth " +
            $"{result.DecodeCtxDepth}{(result.Depth > 0 ? $" (+{result.Depth} untimed depth tokens)" : "")}, " +
            $"{result.Reps.Count} reps, load {result.LoadMs:F0} ms (excluded)[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]results.csv row[/] [dim](benchmarks/perf-matrix/results.csv)[/]:");
        Console.WriteLine(csvRow);
    }

    private static string BuildSettingsSummary(Settings settings, BenchResult result, string backend)
    {
        var parts = new List<string> { $"bench --device {backend} -p {result.PromptTokens} -n {result.DecodeTokens} -r {settings.Reps}" };
        if (result.Depth > 0) parts.Add($"--depth {result.Depth}");
        if (settings.Threads > 0) parts.Add($"--threads {settings.Threads}");
        if (settings.DecodeThreads > 0) parts.Add($"--decode-threads {settings.DecodeThreads}");
        return string.Join(" ", parts);
    }

    private static void WriteError(Settings settings, string message)
    {
        if (settings.Json)
            Console.Error.WriteLine($"Error: {message}");
        else
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(message)}[/]");
    }

    /// <summary>Replaces whitespace with '-' so device names stay single CSV tokens.</summary>
    private static string SanitizeLabel(string label) =>
        string.Join("-", label.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    private static int ParseDeviceOrdinal(string device)
    {
        int colonIdx = device.IndexOf(':');
        if (colonIdx < 0) return 0;
        return int.TryParse(device.AsSpan(colonIdx + 1), out int id) ? id : 0;
    }

    /// <summary>
    /// Resolves the SPIR-V blob directory for Vulkan: <c>spv/</c> next to the running
    /// assembly (the MSBuild content-copy pattern), falling back to the in-repo
    /// <c>native/vulkan/spv</c> for `dotnet run` from the source tree.
    /// </summary>
    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found (looked for spv/ beside the binary and native/vulkan/spv). " +
            "Build them with native/vulkan/build.ps1 (requires the Vulkan SDK).");
    }
}
