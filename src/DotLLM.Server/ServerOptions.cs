using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Scheduler;
using DotLLM.Server.RateLimiting;

namespace DotLLM.Server;

/// <summary>
/// Server startup configuration parsed from command-line arguments.
/// </summary>
public sealed record ServerOptions
{
    /// <summary>GGUF file path or HuggingFace repo ID.</summary>
    public required string Model { get; init; }

    /// <summary>Quantization filter (e.g., Q8_0, Q4_K_M).</summary>
    public string? Quant { get; init; }

    /// <summary>Compute device: "cpu", "gpu", "gpu:0".</summary>
    public string Device { get; init; } = "cpu";

    /// <summary>Number of GPU layers for hybrid offloading.</summary>
    public int? GpuLayers { get; init; }

    /// <summary>CPU thread count (0 = auto).</summary>
    public int Threads { get; init; }

    /// <summary>Decode thread count (0 = auto).</summary>
    public int DecodeThreads { get; init; }

    /// <summary>Host to bind to.</summary>
    public string Host { get; init; } = "localhost";

    /// <summary>Port to listen on.</summary>
    public int Port { get; init; } = 8080;

    /// <summary>KV-cache key quantization type.</summary>
    public string CacheTypeK { get; init; } = "f32";

    /// <summary>KV-cache value quantization type.</summary>
    public string CacheTypeV { get; init; } = "f32";

    /// <summary>Whether prompt caching is enabled.</summary>
    public bool PromptCacheEnabled { get; init; } = true;

    /// <summary>Maximum number of cached sessions for prompt caching.</summary>
    public int PromptCacheSize { get; init; } = 4;

    /// <summary>Whether to use paged KV-cache (block-based allocation). Default true for serve.</summary>
    public bool UsePaged { get; init; } = true;

    /// <summary>Warm-up configuration for JIT pre-compilation and CUDA kernel loading.</summary>
    public WarmupOptions Warmup { get; init; } = WarmupOptions.Default;

    /// <summary>Draft model for speculative decoding. Null = disabled.</summary>
    public string? SpeculativeModel { get; init; }

    /// <summary>Number of draft candidates per speculative step (K).</summary>
    public int SpeculativeCandidates { get; init; } = 5;

    /// <summary>
    /// Maximum prompt tokens per prefill forward pass (llama.cpp <c>-ub</c> / micro-batch analog).
    /// 0 (default) = whole prompt in one forward pass. On the single-request
    /// <see cref="DotLLM.Engine.TextGenerator"/> path this chunks the prompt-suffix prefill; when
    /// the continuous-batch scheduler is active it is applied as the scheduler's per-step prefill
    /// admission cap (<see cref="ContinuousBatchSchedulerOptions.MaxPrefillTokensPerStep"/>) unless
    /// the <see cref="Scheduler"/> section already sets one — a single prompt longer than the cap
    /// still prefills in one forward pass there (admission-level cap, not intra-prompt chunking).
    /// </summary>
    public int PrefillChunkSize { get; init; }

    /// <summary>Model display name (derived from file path).</summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// RoPE scaling overrides applied on top of the GGUF-derived config at load time
    /// (llama.cpp <c>--rope-scaling</c>/<c>--rope-freq-base</c>/<c>--yarn-*</c> flag family).
    /// Null = use the GGUF-derived <see cref="Core.Models.ModelConfig.RoPEConfig"/> unchanged.
    /// </summary>
    public RoPEOverrideOptions? RopeOverride { get; init; }

    /// <summary>
    /// Whether the LoRA admin write endpoints (<c>POST /v1/lora/load</c>,
    /// <c>DELETE /v1/lora/{name}</c>) are enabled. Read-only <c>GET /v1/lora</c>
    /// is always available. Defaults to <c>false</c> — opt-in via configuration.
    /// </summary>
    public bool AllowLoraAdminApi { get; init; }

    /// <summary>
    /// Per-API-key rate-limit configuration. When <see cref="RateLimitConfig.Enabled"/>
    /// is <c>false</c> (or this property is <c>null</c>) the rate-limit middleware
    /// is not wired and the server has no per-caller limits. See <see cref="RateLimitConfig"/>
    /// for the JSON shape loaded from <c>appsettings.json</c>.
    /// </summary>
    public RateLimitConfig? RateLimit { get; init; }

    /// <summary>
    /// Continuous-batch scheduler options. When <c>null</c>, the scheduler uses its defaults
    /// (fairness off, preemption off, etc.). A host binding <see cref="ServerOptions"/> from
    /// <c>appsettings.json</c> can set this section to enable per-API-key fairness
    /// (<see cref="ContinuousBatchSchedulerOptions.EnableFairness"/>), bound recurrent concurrency,
    /// preemption, the prefill-token cap, and the active-sequence/reserve limits.
    /// </summary>
    public ContinuousBatchSchedulerOptions? Scheduler { get; init; }

    /// <summary>
    /// Server-wide default idle-unload duration in seconds (#369, ollama parity — ollama's own
    /// default is 5 min). Per-model/per-request <c>keep_alive</c> overrides take precedence when
    /// present. 0 = unload immediately after each use. Negative = never auto-unload.
    /// </summary>
    public double KeepAliveSeconds { get; init; } = 300;

    /// <summary>
    /// Maximum number of models resident at once, counting the active one (#369). Default 1
    /// reproduces the original single-model hot-swap behavior exactly — the previous model is
    /// evicted the instant a new one loads. Set &gt; 1 to hold multiple models concurrently
    /// (subject to <see cref="ResidentMemoryBudgetBytes"/>), servable via the <c>model</c> field on
    /// chat/completion requests or an explicit <c>POST /v1/models/load</c>.
    /// </summary>
    public int MaxResidentModels { get; init; } = 1;

    /// <summary>
    /// Total byte budget across all resident models (#369). 0 (default) = unlimited — only
    /// <see cref="MaxResidentModels"/> bounds residency. Accounted against each model's GGUF file
    /// size on disk as a proxy for its host-RAM (mmap) or VRAM footprint.
    /// </summary>
    public long ResidentMemoryBudgetBytes { get; init; }

    /// <summary>
    /// Interval between idle-unload sweeps (#369). Default 5s — bounds the worst-case delay
    /// between a model crossing its keep-alive and actually being unloaded (including the
    /// <c>keep_alive: 0</c> "unload after each use" case, which is enforced on the next tick
    /// rather than synchronously in the request path, to keep response latency unaffected).
    /// </summary>
    public TimeSpan IdleSweepInterval { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Parses command-line arguments into <see cref="ServerOptions"/>.
    /// </summary>
    public static ServerOptions Parse(string[] args)
    {
        string? model = null;
        string? quant = null;
        string device = "cpu";
        int? gpuLayers = null;
        int threads = 0;
        int decodeThreads = 0;
        string host = "localhost";
        int port = 8080;
        string cacheTypeK = "f32";
        string cacheTypeV = "f32";
        bool promptCacheEnabled = true;
        int promptCacheSize = 4;
        bool usePaged = true;
        bool warmupEnabled = true;
        int warmupIterations = 3;
        bool schedulerFairness = false;
        string? speculativeModel = null;
        int speculativeCandidates = 5;
        int prefillChunkSize = 0;
        string? ropeScaling = null;
        float? ropeFreqBase = null;
        float? ropeScale = null;
        int? yarnOrigCtx = null;
        float? yarnAttnFactor = null;
        float? yarnBetaFast = null;
        float? yarnBetaSlow = null;
        double keepAliveSeconds = 300;
        int maxResidentModels = 1;
        long residentMemoryBudgetBytes = 0;

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            string? next = i + 1 < args.Length ? args[i + 1] : null;

            switch (arg)
            {
                case "--model" or "-m":
                    model = next; i++; break;
                case "--quant" or "-q":
                    quant = next; i++; break;
                case "--device" or "-d":
                    device = next ?? "cpu"; i++; break;
                case "--gpu-layers":
                    gpuLayers = int.Parse(next!); i++; break;
                case "--threads":
                    threads = int.Parse(next!); i++; break;
                case "--decode-threads":
                    decodeThreads = int.Parse(next!); i++; break;
                case "--host":
                    host = next ?? "localhost"; i++; break;
                case "--port" or "-p":
                    port = int.Parse(next!); i++; break;
                case "--cache-type-k":
                    cacheTypeK = next ?? "f32"; i++; break;
                case "--cache-type-v":
                    cacheTypeV = next ?? "f32"; i++; break;
                case "--no-prompt-cache":
                    promptCacheEnabled = false; break;
                case "--prompt-cache-size":
                    promptCacheSize = int.Parse(next!); i++; break;
                case "--no-paged":
                    usePaged = false; break;
                case "--warmup":
                    warmupEnabled = true; break;
                case "--no-warmup":
                    warmupEnabled = false; break;
                case "--warmup-iterations":
                    warmupIterations = int.Parse(next!); i++; break;
                case "--scheduler-fairness":
                    schedulerFairness = true; break;
                case "--speculative-model" or "--draft-model":
                    speculativeModel = next; i++; break;
                case "--speculative-k" or "--draft-tokens":
                    speculativeCandidates = int.Parse(next!); i++; break;
                case "--prefill-chunk-size" or "--ubatch-size":
                    prefillChunkSize = int.Parse(next!); i++; break;
                case "--rope-scaling":
                    ropeScaling = next; i++; break;
                case "--rope-freq-base":
                    ropeFreqBase = float.Parse(next!); i++; break;
                case "--rope-scale":
                    ropeScale = float.Parse(next!); i++; break;
                case "--yarn-orig-ctx":
                    yarnOrigCtx = int.Parse(next!); i++; break;
                case "--yarn-attn-factor":
                    yarnAttnFactor = float.Parse(next!); i++; break;
                case "--yarn-beta-fast":
                    yarnBetaFast = float.Parse(next!); i++; break;
                case "--yarn-beta-slow":
                    yarnBetaSlow = float.Parse(next!); i++; break;
                case "--keep-alive":
                    keepAliveSeconds = double.Parse(next!); i++; break;
                case "--max-resident-models":
                    maxResidentModels = int.Parse(next!); i++; break;
                case "--resident-memory-budget":
                    residentMemoryBudgetBytes = long.Parse(next!); i++; break;
                default:
                    // Positional: treat as model if not set
                    if (model is null && !arg.StartsWith('-'))
                        model = arg;
                    break;
            }
        }

        if (model is null)
        {
            Console.Error.WriteLine("Usage: dotllm-server --model <path-or-repo> [--port 8080] [--device cpu|gpu]");
            Environment.Exit(1);
        }

        string modelId = Path.GetFileNameWithoutExtension(model);
        if (model.Contains('/'))
            modelId = model.Split('/')[^1]; // last segment of repo

        return new ServerOptions
        {
            Model = model,
            Quant = quant,
            Device = device,
            GpuLayers = gpuLayers,
            Threads = threads,
            DecodeThreads = decodeThreads,
            Host = host,
            Port = port,
            CacheTypeK = cacheTypeK,
            CacheTypeV = cacheTypeV,
            PromptCacheEnabled = promptCacheEnabled,
            PromptCacheSize = promptCacheSize,
            UsePaged = usePaged,
            Warmup = new WarmupOptions
            {
                Enabled = warmupEnabled,
                Iterations = warmupIterations,
            },
            Scheduler = schedulerFairness
                ? new ContinuousBatchSchedulerOptions { EnableFairness = true }
                : null,
            SpeculativeModel = speculativeModel,
            SpeculativeCandidates = speculativeCandidates,
            PrefillChunkSize = prefillChunkSize,
            KeepAliveSeconds = keepAliveSeconds,
            MaxResidentModels = maxResidentModels,
            ResidentMemoryBudgetBytes = residentMemoryBudgetBytes,
            ModelId = modelId,
            RopeOverride = BuildRopeOverride(ropeScaling, ropeFreqBase, ropeScale,
                yarnOrigCtx, yarnAttnFactor, yarnBetaFast, yarnBetaSlow),
        };
    }

    /// <summary>
    /// Builds a <see cref="RoPEOverrideOptions"/> from individually-parsed CLI flag values.
    /// Returns null (no-op) when none were set. Shared by the raw <see cref="Parse"/> path and
    /// (indirectly, via matching flags) the Spectre.Console-based <c>dotllm serve</c> command.
    /// </summary>
    public static RoPEOverrideOptions? BuildRopeOverride(string? scalingType, float? freqBase,
        float? scalingFactor, int? origCtx, float? attnFactor, float? betaFast, float? betaSlow)
    {
        var overrides = new RoPEOverrideOptions
        {
            ScalingType = scalingType is null ? null : ParseRopeScalingType(scalingType),
            FreqBase = freqBase,
            ScalingFactor = scalingFactor,
            OrigMaxSeqLen = origCtx,
            AttnFactor = attnFactor,
            BetaFast = betaFast,
            BetaSlow = betaSlow,
        };
        return overrides.HasAnyOverride ? overrides : null;
    }

    private static RoPEScalingType ParseRopeScalingType(string value) => value.ToLowerInvariant() switch
    {
        "none" => RoPEScalingType.None,
        "linear" => RoPEScalingType.Linear,
        "yarn" => RoPEScalingType.YaRN,
        "ntk" => RoPEScalingType.NTK,
        "dynamic" or "dynamic_ntk" or "dynamic-ntk" => RoPEScalingType.DynamicNTK,
        "su" or "longrope" => RoPEScalingType.Su,
        _ => throw new InvalidOperationException(
            $"Unknown --rope-scaling value '{value}'. Expected: none, linear, yarn, ntk, dynamic."),
    };
}
