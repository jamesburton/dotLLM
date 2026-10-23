using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using DotLLM.Cli.Benchmarking;
using DotLLM.Cli.Diagnostics;
using DotLLM.Cli.Helpers;
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

    public sealed class Settings : DotLLM.Cli.Helpers.KvQuantSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--device|-d")]
        [Description(DeviceCli.OptionHelp)]
        [DefaultValue(DeviceCli.DefaultDevice)]
        public string Device { get; set; } = DeviceCli.DefaultDevice;

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

        [CommandOption(MoeOffloadCli.NCpuMoeOption)]
        [Description(
            "MoE CPU/GPU expert offload (llama.cpp --n-cpu-moe equivalent): put the routed experts of the first N layers on " +
            "the CPU, trading throughput for device memory. Vulkan Qwen3MoeHybrid (#370): CPU expert compute for those layers. " +
            "CUDA nemotron_h_moe (#1071): those layers' experts are never cached and always computed on the CPU. Unset = none " +
            "(env fallback DOTLLM_N_CPU_MOE). No-op on the cpu backend.")]
        public int? NCpuMoeLayers { get; set; }

        /// <summary>Routed-expert placement on CUDA (issue #1070).</summary>
        [CommandOption(MoeOffloadCli.ModeOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.ModeHelp)]
        public string? MoeOffload { get; set; }

        /// <summary>VRAM expert-cache size in MiB (issue #1070).</summary>
        [CommandOption(MoeOffloadCli.CacheMibOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.CacheMibHelp)]
        public string? MoeCacheMib { get; set; }

        /// <summary>Expert-cache replacement policy (issue #1070).</summary>
        [CommandOption(MoeOffloadCli.PolicyOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.PolicyHelp)]
        public string? MoeCachePolicy { get; set; }

        /// <summary>Where uncached offloaded experts run (issue #1071).</summary>
        [CommandOption(MoeOffloadCli.ColdOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.ColdHelp)]
        public string? MoeCold { get; set; }

        /// <summary>CPU cold-expert worker threads (issue #1071).</summary>
        [CommandOption(MoeOffloadCli.CpuThreadsOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.CpuThreadsHelp)]
        public int? MoeCpuThreads { get; set; }

        /// <summary>llama.cpp <c>--cpu-moe</c>: every MoE layer's experts on the CPU (issue #1071).</summary>
        [CommandOption(MoeOffloadCli.CpuMoeOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.CpuMoeHelp)]
        [DefaultValue(false)]
        public bool CpuMoe { get; set; }

        /// <summary>Re-run the MoE offload cost calibration (issue #1071).</summary>
        [CommandOption(MoeOffloadCli.CalibrateOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.CalibrateHelp)]
        [DefaultValue(false)]
        public bool MoeCalibrate { get; set; }

        /// <summary>Context the MoE offload VRAM budget reserves KV cache for (issue #1071).</summary>
        [CommandOption(MoeOffloadCli.ContextOption)]
        [Description(DotLLM.Core.Moe.MoeOffloadOptions.ContextHelp)]
        public int? CtxSize { get; set; }

        /// <summary>The raw MoE offload option values (issues #1070, #1071).</summary>
        public DotLLM.Core.Moe.MoeCliValues MoeValues => new(MoeOffload, MoeCacheMib, MoeCachePolicy, MoeCold, MoeCpuThreads,
            CpuMoe, NCpuMoeLayers, MoeCalibrate, CtxSize);

        [CommandOption("--depth-list")]
        [Description("Comma-separated context depths for multi-point sweep (e.g., 512,2048,8192). " +
                     "Measures decode at each checkpoint in one process without reloading the model. " +
                     "Mutually exclusive with --depth.")]
        public string? DepthList { get; set; }

        [CommandOption("--warmup")]
        [Description("Number of warm-up repetitions to discard per checkpoint (default: 1). " +
                     "Use 0 to skip warm-up entirely.")]
        [DefaultValue(-1)]
        public int Warmup { get; set; } = -1;

        /// <summary>Effective warm-up rep count: <c>--warmup N</c> or 1 when unset.</summary>
        public int EffectiveWarmup => Warmup < 0 ? 1 : Warmup;

        public override ValidationResult Validate()
        {
            if (PromptTokens <= 0) return ValidationResult.Error("--prompt-tokens|-p must be positive.");
            if (GenTokens <= 0) return ValidationResult.Error("--gen-tokens|-n must be positive.");
            if (Reps <= 0) return ValidationResult.Error("--reps|-r must be positive.");
            if (Depth < 0) return ValidationResult.Error("--depth must be >= 0.");
            if (NCpuMoeLayers < 0) return ValidationResult.Error("--n-cpu-moe must be >= 0.");
            if (Warmup < -1) return ValidationResult.Error("--warmup must be >= 0.");
            if (DepthList is not null && Depth > 0)
                return ValidationResult.Error("--depth-list and --depth are mutually exclusive.");
            if (DepthList is not null && !TryParseDepthList(DepthList, out _, out string? parseError))
                return ValidationResult.Error(parseError!);
            if (DeviceCli.Validate(Device) is { } deviceError)
                return ValidationResult.Error(deviceError);
            if (MoeOffloadCli.Validate(MoeValues) is { } moeError)
                return ValidationResult.Error(moeError);
            return ValidationResult.Success();
        }

        internal static bool TryParseDepthList(string input, out int[] depths, out string? error)
        {
            error = null;
            depths = [];
            var parts = input.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length == 0)
            {
                error = "--depth-list requires at least one depth value.";
                return false;
            }
            var parsed = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                if (!int.TryParse(parts[i], out int val) || val < 0)
                {
                    error = $"--depth-list: '{parts[i]}' is not a valid non-negative integer.";
                    return false;
                }
                parsed[i] = val;
            }
            Array.Sort(parsed);
            depths = parsed;
            return true;
        }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var ggufPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (ggufPath is null)
            return 1;

        // One shared resolution (#790): auto -> CUDA / Vulkan / CPU with a visible line (and a warning on CPU); an explicit GPU that
        // cannot be honoured is an error, never a CPU measurement.
        if (!DeviceCli.TryResolveForTool(settings.Device, ggufPath, settings.Json, out string backend, out int gpuId))
            return 1;

        if (!DotLLM.Cli.Helpers.KvStorageCli.TryBuild(null, null, settings.KvQuantK, settings.KvQuantV, 0, out var kvRequest, out var kvFlagError))
        {
            WriteError(settings, kvFlagError!);
            return 1;
        }

        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] seedTokens = tokenizer.Encode(settings.SeedPrompt).ToArray();
        if (seedTokens.Length == 0)
        {
            WriteError(settings, "Seed prompt produced no tokens.");
            return 1;
        }

        // Parse depth-list if provided; otherwise single-depth mode.
        int[]? depthList = null;
        if (settings.DepthList is not null)
            Settings.TryParseDepthList(settings.DepthList, out depthList, out _);

        int maxDepth = depthList is not null ? depthList[^1] : settings.Depth;

        // Clamp the prompt so prompt + maxDepth + decode (+ margin) fits the model context.
        int promptLen = settings.PromptTokens;
        int budget = config.MaxSequenceLength - maxDepth - settings.GenTokens - 8;
        if (budget <= 0)
        {
            WriteError(settings,
                $"--depth {maxDepth} + -n {settings.GenTokens} exceeds the model's max sequence " +
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
        var moeNotes = new List<string>();   // expert-offload load notes, printed after the spinner (#1070)

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
                    // THE shared per-architecture CUDA dispatch (#259) — routes hybrid
                    // architectures (Qwen3MoeHybrid, Qwen3HybridDense) to their dedicated
                    // loaders; their GDN layers have no attn_output.weight.
                    // Without --ctx-size the offload budget reserves KV for what this run uses, not the model's maximum (#1071).
                    (model, kvFactory) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(gguf, config, gpuId,
                        moeOffload: MoeOffloadCli.Build(settings.MoeValues with
                        {
                            ContextTokens = settings.CtxSize ?? Math.Min(config.MaxSequenceLength,
                                promptTokens.Length + maxDepth + settings.GenTokens + 8),
                        }) with { Log = moeNotes.Add });
                    deviceLabel = DotLLM.Cuda.CudaDevice.GetDevice(gpuId).Name;
                    break;

                case "vulkan":
                {
                    vulkanDevice = DotLLM.Vulkan.VulkanDevice.Create();
                    // THE shared per-architecture Vulkan dispatch (#259).
                    (model, kvFactory) = DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(
                        vulkanDevice, gguf, config, ResolveSpvDir(),
                        DotLLM.Core.Moe.MoeOffloadOptions.ResolveCpuMoeLayers(settings.CpuMoe, settings.NCpuMoeLayers));
                    deviceLabel = vulkanDevice.DeviceName;
                    if (model is DotLLM.Vulkan.VulkanQwen3MoeHybridTransformerModel vkMoe
                        && vkMoe.NCpuMoeLayers > 0 && !settings.Json)
                    {
                        AnsiConsole.MarkupLine(
                            $"[grey]MoE CPU offload: {vkMoe.NCpuMoeLayers} layer(s), " +
                            $"~{vkMoe.EstimatedCpuOffloadVramSavedBytes / (1024.0 * 1024.0):F0} MiB " +
                            "GPU expert-bank upload avoided.[/]");
                    }
                    break;
                }

                default:
                    throw new InvalidOperationException($"Unhandled backend '{backend}'.");
            }
        }

        // Issue #438 — opt-in residency probe. The mapped tensor-data region is
        // [DataBasePointer, EOF); its pages are the ones every device upload reads from.
        // A split GGUF has one mapping per shard and no single base: the probe is skipped there (#756).
        nint mapBase = gguf.IsSplit ? 0 : gguf.DataBasePointer;
        // The mapped length comes from the already-open GgufFile, never from a second handle:
        // FileInfo.Length on a symlink reports the reparse point (every HF-cache model is reached
        // through one), and File.OpenRead throws a sharing violation once DOTLLM_GGUF_MAP_COW=1,
        // because a copy-on-write mapping needs write access and so holds the file FileShare.None.
        long mapLength = HostResidencyProbe.Enabled && !gguf.IsSplit ? gguf.DataSectionLength : 0;
        if (HostResidencyProbe.Enabled)
            HostResidencyProbe.Report("before-load", mapBase, mapLength, deviceSnapshot: null);

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
        foreach (string note in moeNotes)
        {
            if (settings.Json) Console.Error.WriteLine(note);
            else AnsiConsole.MarkupLine($"[grey]{Markup.Escape(note)}[/]");
        }

        // #967: apply --kv-quant-k/-v to a CUDA hybrid-dense model; warn (never silently ignore) elsewhere.
        if (kvRequest.IsSpecified)
        {
            var kvTarget = DotLLM.Cli.Helpers.KvStorageCli.Classify(model, vulkanDevice is not null);
            foreach (var w in DotLLM.Cli.Helpers.KvStorageCli.Warnings(kvRequest, kvTarget, config.Architecture.ToString(), backend, engineCacheHonoured: false))
                Console.Error.WriteLine($"WARNING: {w}");
            try
            {
                DotLLM.Cli.Helpers.KvStorageCli.Apply(model, kvRequest);
            }
            catch (NotSupportedException e)
            {
                WriteError(settings, e.Message);
                model.Dispose();
                vulkanDevice?.Dispose();
                return 1;
            }
        }

        if (HostResidencyProbe.Enabled)
            HostResidencyProbe.Report("after-load", mapBase, mapLength, vulkanDevice?.MemorySnapshot());

        // Issue #438 experiment arm — DOTLLM_BENCH_RELEASE_GGUF=1 unmaps the GGUF right
        // after upload. Every device-resident weight was copied into a device buffer by
        // then; nothing in the Vulkan forward path reads a host pointer. If that premise
        // is wrong this faults, which is itself the answer.
        // Offloaded experts (#1070) without a pinned copy are read from the mapping for the whole run: never unmap then.
        bool readsMapping = model is DotLLM.Cuda.Architectures.CudaNemotronHTransformerModel { MoeOffload.PinnedStore: false };
        if (Environment.GetEnvironmentVariable("DOTLLM_BENCH_RELEASE_GGUF") == "1" && !readsMapping)
        {
            gguf.Dispose();
            if (HostResidencyProbe.Enabled)
            {
                // mapBase is dangling now — pass 0 so the page census is skipped rather
                // than querying an unmapped range.
                HostResidencyProbe.Report("after-gguf-dispose", 0, 0, vulkanDevice?.MemorySnapshot());
            }
            mapBase = 0;
            mapLength = 0;
        }

        try
        {
            string quant = BenchEnvironment.InferQuantLabel(ggufPath, settings.Quant);
            string modelName = BenchEnvironment.InferModelName(ggufPath, quant);
            string runtimeVersion = BenchEnvironment.ResolveRuntimeVersion();
            string host = Environment.MachineName.ToLowerInvariant();
            double loadMs = loadSw.Elapsed.TotalMilliseconds;
            int warmupReps = settings.EffectiveWarmup;

            var moeStart = DotLLM.Core.Moe.MoeTelemetry.Snapshot();
            if (depthList is not null)
            {
                int rc = ExecuteDepthList(settings, depthList, model, kvFactory, promptTokens, warmupReps,
                    loadMs, mapBase, mapLength, vulkanDevice,
                    ggufPath, quant, modelName, runtimeVersion, host, backend, deviceLabel, config);
                MoeOffloadCli.PrintSummary(moeStart, settings.Json);
                return rc;
            }

            var result = BenchRunner.Run(
                model, kvFactory, promptTokens,
                decodeTokens: settings.GenTokens,
                reps: settings.Reps,
                depth: settings.Depth,
                loadMs: loadMs,
                warmupReps: warmupReps);

            if (HostResidencyProbe.Enabled)
                HostResidencyProbe.Report("after-decode", mapBase, mapLength, vulkanDevice?.MemorySnapshot());

            string settingsSummary = BuildSettingsSummary(settings, result, backend);
            string notes = $"pp{result.PromptTokens}/tg{result.DecodeTokens} r{settings.Reps} " +
                           $"warmup={warmupReps}; " +
                           $"tg best {BenchStats.FormatTokS(result.DecodeTokSBest)}; load {result.LoadMs:F0} ms";
            string csvRow = BenchCsv.FormatRow(
                DateOnly.FromDateTime(DateTime.Now), host, SanitizeLabel(deviceLabel), backend,
                runtimeVersion, modelName, quant,
                result.PrefillTokSMedian, result.DecodeTokSMedian, result.DecodeCtxDepth,
                settingsSummary, notes);

            if (settings.Json)
            {
                Console.WriteLine(JsonSerializer.Serialize(
                    BuildJsonResult(result, modelName, ggufPath, quant, backend, deviceLabel, host, runtimeVersion, csvRow),
                    CliJsonContext.Default.BenchJsonResult));
            }
            else
            {
                RenderTable(settings, config, result, modelName, quant, backend, deviceLabel, runtimeVersion, csvRow);
                if (DotLLM.Cli.Helpers.KvStorageCli.Describe(model, result.PromptTokens + result.Depth + result.DecodeTokens) is { } kvInfo)
                    AnsiConsole.MarkupLine($"  [dim]{Markup.Escape(kvInfo.Text)}[/]");
            }
            MoeOffloadCli.PrintSummary(moeStart, settings.Json);

            return 0;
        }
        finally
        {
            model.Dispose();
            vulkanDevice?.Dispose();
        }
    }

    private static int ExecuteDepthList(
        Settings settings, int[] depthList, IModel model, Func<int, IKvCache> kvFactory,
        int[] promptTokens, int warmupReps, double loadMs,
        nint mapBase, long mapLength, DotLLM.Vulkan.VulkanDevice? vulkanDevice,
        string ggufPath, string quant, string modelName, string runtimeVersion,
        string host, string backend, string deviceLabel, ModelConfig config)
    {
        var jsonResults = new BenchJsonResult[depthList.Length];

        for (int di = 0; di < depthList.Length; di++)
        {
            int depth = depthList[di];
            if (!settings.Json)
                AnsiConsole.MarkupLine($"[grey]depth-list checkpoint {di + 1}/{depthList.Length}: depth={depth}[/]");

            var result = BenchRunner.Run(
                model, kvFactory, promptTokens,
                decodeTokens: settings.GenTokens,
                reps: settings.Reps,
                depth: depth,
                loadMs: loadMs,
                warmupReps: warmupReps);

            string settingsSummary = BuildSettingsSummary(settings, result, backend);
            string notes = $"pp{result.PromptTokens}/tg{result.DecodeTokens} r{settings.Reps} " +
                           $"warmup={warmupReps} depth-list[{di}]; " +
                           $"tg best {BenchStats.FormatTokS(result.DecodeTokSBest)}; load {result.LoadMs:F0} ms";
            string csvRow = BenchCsv.FormatRow(
                DateOnly.FromDateTime(DateTime.Now), host, SanitizeLabel(deviceLabel), backend,
                runtimeVersion, modelName, quant,
                result.PrefillTokSMedian, result.DecodeTokSMedian, result.DecodeCtxDepth,
                settingsSummary, notes);

            jsonResults[di] = BuildJsonResult(result, modelName, ggufPath, quant, backend, deviceLabel, host, runtimeVersion, csvRow);

            if (!settings.Json)
                RenderTable(settings, config, result, modelName, quant, backend, deviceLabel, runtimeVersion, csvRow);
        }

        if (HostResidencyProbe.Enabled)
            HostResidencyProbe.Report("after-decode", mapBase, mapLength, vulkanDevice?.MemorySnapshot());

        if (settings.Json)
        {
            Console.WriteLine(JsonSerializer.Serialize(jsonResults, CliJsonContext.Default.BenchJsonResultArray));
        }
        else
        {
            RenderSweepSummary(depthList, jsonResults, modelName, quant, backend, deviceLabel);
        }

        return 0;
    }

    private static BenchJsonResult BuildJsonResult(
        BenchResult result, string modelName, string modelPath,
        string quant, string backend, string device, string host,
        string commit, string csvRow) => new()
    {
        Model = modelName,
        ModelPath = modelPath,
        Quant = quant,
        Backend = backend,
        Device = device,
        Host = host,
        Commit = commit,
        LoadMs = Math.Round(result.LoadMs, 1),
        PromptTokens = result.PromptTokens,
        DecodeTokens = result.DecodeTokens,
        Depth = result.Depth,
        DecodeCtxDepth = result.DecodeCtxDepth,
        Warmup = result.Warmup is { } w ? BenchJsonRep.From(w) : null,
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

    private static void RenderSweepSummary(
        int[] depthList, BenchJsonResult[] results, string modelName,
        string quant, string backend, string deviceLabel)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[grey]depth-list sweep summary | {Markup.Escape(modelName)} {quant} | {backend} ({Markup.Escape(deviceLabel)})[/]").LeftJustified());

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("depth");
        table.AddColumn(new TableColumn("ctx depth").RightAligned());
        table.AddColumn(new TableColumn("pp tok/s").RightAligned());
        table.AddColumn(new TableColumn("tg tok/s").RightAligned());
        table.AddColumn(new TableColumn("tg best").RightAligned());

        for (int i = 0; i < depthList.Length; i++)
        {
            table.AddRow(
                depthList[i].ToString(),
                results[i].DecodeCtxDepth.ToString(),
                BenchStats.FormatTokS(results[i].PrefillTokSMedian),
                $"[bold green]{BenchStats.FormatTokS(results[i].DecodeTokSMedian)}[/]",
                BenchStats.FormatTokS(results[i].DecodeTokSBest));
        }

        AnsiConsole.Write(table);
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

        if (result.Warmup is { } warmup)
        {
            table.AddRow(
                "[dim]warmup[/]",
                $"[dim]{warmup.PrefillMs:F1}[/]",
                $"[dim]{warmup.PrefillTokS:F2}[/]",
                $"[dim]{warmup.DecodeMs:F1}[/]",
                $"[dim]{warmup.DecodeTokS:F2}[/]");
        }
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
        if (settings.KvQuantK is not null) parts.Add($"--kv-quant-k {settings.KvQuantK}");
        if (settings.KvQuantV is not null) parts.Add($"--kv-quant-v {settings.KvQuantV}");
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
