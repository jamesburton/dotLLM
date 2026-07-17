using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Engine;
using DotLLM.Server;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Launch the OpenAI-compatible API server with a built-in web chat UI.
/// </summary>
internal sealed class ServeCommand : AsyncCommand<ServeCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to a GGUF file or HuggingFace repo ID.</summary>
        [CommandArgument(0, "[model]")]
        [Description("Path to a GGUF file or HuggingFace repo ID. Omit to start without a model (load from UI).")]
        public string? Model { get; set; }

        /// <summary>Port to listen on.</summary>
        [CommandOption("--port|-p")]
        [Description("Port to listen on.")]
        [DefaultValue(8080)]
        public int Port { get; set; } = 8080;

        /// <summary>Host to bind to.</summary>
        [CommandOption("--host")]
        [Description("Host address to bind to.")]
        [DefaultValue("localhost")]
        public string Host { get; set; } = "localhost";

        /// <summary>Compute device.</summary>
        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'gpu', 'gpu:0', 'gpu:1'.")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        /// <summary>Number of GPU layers for hybrid offloading.</summary>
        [CommandOption("--gpu-layers")]
        [Description("Number of transformer layers to offload to GPU.")]
        public int? GpuLayers { get; set; }

        /// <summary>CPU thread count.</summary>
        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto.")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        /// <summary>Decode thread count.</summary>
        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode. 0 = auto.")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        /// <summary>Quantization filter.</summary>
        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }

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

        /// <summary>Disable prompt caching (enabled by default).</summary>
        [CommandOption("--no-prompt-cache")]
        [Description("Disable prompt caching. When enabled (default), KV-cache state is reused across turns.")]
        [DefaultValue(false)]
        public bool NoPromptCache { get; set; }

        /// <summary>Maximum number of cached sessions for prompt caching.</summary>
        [CommandOption("--prompt-cache-size")]
        [Description("Maximum number of cached sessions for prompt caching.")]
        [DefaultValue(4)]
        public int PromptCacheSize { get; set; } = 4;

        /// <summary>Disable warm-up (enabled by default).</summary>
        [CommandOption("--no-warmup")]
        [Description("Disable warm-up passes at startup. When enabled (default), dummy inference runs trigger JIT compilation.")]
        [DefaultValue(false)]
        public bool NoWarmup { get; set; }

        /// <summary>Number of warm-up iterations.</summary>
        [CommandOption("--warmup-iterations")]
        [Description("Number of warm-up iterations (default: 3). More iterations improve JIT optimization via Dynamic PGO.")]
        [DefaultValue(3)]
        public int WarmupIterations { get; set; } = 3;

        /// <summary>Suppress automatic browser opening.</summary>
        [CommandOption("--no-browser")]
        [Description("Don't auto-open the browser.")]
        [DefaultValue(false)]
        public bool NoBrowser { get; set; }

        /// <summary>Disable the built-in web chat UI. Serve API endpoints only.</summary>
        [CommandOption("--no-ui")]
        [Description("Disable the built-in web chat UI. Only serve API endpoints.")]
        [DefaultValue(false)]
        public bool NoUi { get; set; }

        /// <summary>Disable paged KV-cache (enabled by default for serve).</summary>
        [CommandOption("--no-paged")]
        [Description("Disable paged KV-cache. When enabled (default), uses block-based allocation for memory efficiency.")]
        [DefaultValue(false)]
        public bool NoPaged { get; set; }

        /// <summary>Draft model for speculative decoding.</summary>
        [CommandOption("--speculative-model|--draft-model")]
        [Description("Path or HuggingFace repo ID for a draft model. Enables speculative decoding for faster generation. Must share vocabulary with the main model.")]
        public string? SpeculativeModel { get; set; }

        /// <summary>Number of draft candidates per speculative step.</summary>
        [CommandOption("--speculative-k|--draft-tokens")]
        [Description("Number of draft tokens per speculative step (K). Default 5.")]
        [DefaultValue(5)]
        public int SpeculativeK { get; set; } = 5;

        /// <summary>Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog).</summary>
        [CommandOption("--prefill-chunk-size|--ubatch-size")]
        [Description("Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog). 0 = whole prompt in one pass (default). With the continuous-batch scheduler this caps prefill tokens admitted per step instead.")]
        [DefaultValue(0)]
        public int PrefillChunkSize { get; set; }

        [CommandOption("--rope-scaling")]
        [Description("RoPE scaling override: 'none', 'linear', 'yarn', 'ntk', 'dynamic'. Overrides the GGUF-derived value.")]
        public string? RopeScaling { get; set; }

        [CommandOption("--rope-freq-base")]
        [Description("RoPE base frequency (theta) override. Overrides the GGUF-derived value.")]
        public float? RopeFreqBase { get; set; }

        [CommandOption("--rope-scale")]
        [Description("RoPE scaling factor override (linear/YaRN/NTK). Overrides the GGUF-derived value.")]
        public float? RopeScale { get; set; }

        [CommandOption("--yarn-orig-ctx")]
        [Description("YaRN original context length override.")]
        public int? YarnOrigCtx { get; set; }

        [CommandOption("--yarn-attn-factor")]
        [Description("YaRN attention factor override.")]
        public float? YarnAttnFactor { get; set; }

        [CommandOption("--yarn-beta-fast")]
        [Description("YaRN beta-fast parameter override.")]
        public float? YarnBetaFast { get; set; }

        [CommandOption("--yarn-beta-slow")]
        [Description("YaRN beta-slow parameter override.")]
        public float? YarnBetaSlow { get; set; }

        /// <summary>Idle-unload duration in seconds (#369, ollama parity).</summary>
        [CommandOption("--keep-alive")]
        [Description("Idle-unload duration in seconds (ollama parity; default 300 = 5 min). 0 = unload after each request. Negative = never unload.")]
        [DefaultValue(300.0)]
        public double KeepAlive { get; set; } = 300;

        /// <summary>Maximum number of models resident at once (#369).</summary>
        [CommandOption("--max-resident-models")]
        [Description("Maximum number of models resident at once, counting the active one (default 1 = classic single-model hot-swap). Set > 1 to hold multiple models concurrently.")]
        [DefaultValue(1)]
        public int MaxResidentModels { get; set; } = 1;

        /// <summary>Total byte budget across all resident models (#369).</summary>
        [CommandOption("--resident-memory-budget")]
        [Description("Total byte budget across all resident models. 0 (default) = unlimited, only --max-resident-models bounds residency.")]
        [DefaultValue(0L)]
        public long ResidentMemoryBudgetBytes { get; set; }
    }

    /// <inheritdoc/>
    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        var serverOptions = new ServerOptions
        {
            Model = settings.Model ?? "",
            Quant = settings.Quant,
            Device = settings.Device,
            GpuLayers = settings.GpuLayers,
            Threads = settings.Threads,
            DecodeThreads = settings.DecodeThreads,
            Host = settings.Host,
            Port = settings.Port,
            CacheTypeK = settings.CacheTypeK,
            CacheTypeV = settings.CacheTypeV,
            PromptCacheEnabled = !settings.NoPromptCache,
            PromptCacheSize = settings.PromptCacheSize,
            Warmup = new WarmupOptions
            {
                Enabled = !settings.NoWarmup,
                Iterations = settings.WarmupIterations,
            },
            UsePaged = !settings.NoPaged,
            SpeculativeModel = settings.SpeculativeModel,
            SpeculativeCandidates = settings.SpeculativeK,
            PrefillChunkSize = settings.PrefillChunkSize,
            KeepAliveSeconds = settings.KeepAlive,
            MaxResidentModels = settings.MaxResidentModels,
            ResidentMemoryBudgetBytes = settings.ResidentMemoryBudgetBytes,
            ModelId = "none",
            RopeOverride = ServerOptions.BuildRopeOverride(settings.RopeScaling, settings.RopeFreqBase,
                settings.RopeScale, settings.YarnOrigCtx, settings.YarnAttnFactor,
                settings.YarnBetaFast, settings.YarnBetaSlow),
        };

        ServerState? state = null;

        if (!string.IsNullOrEmpty(settings.Model))
        {
            // Resolve and load model
            var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
            if (resolvedPath is null)
                return 1;

            string modelId = Path.GetFileNameWithoutExtension(resolvedPath);
            if (settings.Model.Contains('/'))
                modelId = settings.Model.Split('/')[^1];
            serverOptions = serverOptions with { ModelId = modelId };

            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Loading and warming up model...", _ =>
                {
                    state = ServerStartup.LoadModel(resolvedPath, serverOptions);
                });
        }
        else
        {
            // Start without a model — load from UI
            state = ServerStartup.CreateBareState(serverOptions);
        }

        // Build app (UI enabled unless --no-ui)
        var app = ServerStartup.BuildApp(state!, [], serveUi: !settings.NoUi);

        var url = $"http://{settings.Host}:{settings.Port}";

        // Print banner
        AnsiConsole.Write(new Rule($"[grey]dotllm serve[/]").LeftJustified());
        if (state!.IsReady)
        {
            AnsiConsole.MarkupLine(
                $"  [bold]{state.Config!.Architecture}[/] {state.Config.NumLayers}L/{state.Config.HiddenSize}H | " +
                $"{Markup.Escape(Path.GetFileName(state.LoadedModelPath))}");
        }
        else
        {
            AnsiConsole.MarkupLine("  [yellow]No model loaded[/] — select one from the UI");
        }
        AnsiConsole.MarkupLine($"  Listening on [link={url}]{url}[/]");
        if (!settings.NoUi)
            AnsiConsole.MarkupLine($"  Chat UI: [dim]{url}[/]");
        AnsiConsole.MarkupLine("  [dim]Single-request mode — requests processed sequentially[/]");
        AnsiConsole.MarkupLine("[dim]  Press Ctrl+C to stop.[/]");
        AnsiConsole.WriteLine();

        // Auto-open browser (skip when UI is disabled)
        if (!settings.NoUi && !settings.NoBrowser)
        {
            try
            {
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }
            catch
            {
                // Ignore on headless systems
            }
        }

        // Run server (blocks until shutdown)
        await app.RunAsync(url);

        // Cleanup
        state.Dispose();
        return 0;
    }
}
