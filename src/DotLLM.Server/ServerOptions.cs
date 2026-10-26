using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
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

    /// <summary>Compute device: "auto", "cpu", "gpu", "gpu:0", "vulkan". The <c>serve</c> CLI and <see cref="Parse"/> default to "auto" (GPU when servable; an explicit device never silently falls back to CPU).</summary>
    public string Device { get; init; } = "cpu";

    /// <summary>Number of GPU layers for hybrid offloading.</summary>
    public int? GpuLayers { get; init; }

    /// <summary>
    /// Routed-expert offload for CUDA <c>nemotron_h_moe</c> models (<c>--moe-offload</c>, <c>--moe-cache-mib</c>,
    /// <c>--moe-cache-policy</c>, issue #1070; <c>--moe-cold</c>, <c>--moe-cpu-threads</c>, <c>--cpu-moe</c>, <c>--n-cpu-moe</c>,
    /// <c>--moe-calibrate</c>, <c>--ctx-size</c>, issue #1071). Null = defaults plus the <c>DOTLLM_MOE_*</c> environment variables.
    /// </summary>
    public DotLLM.Core.Moe.MoeOffloadOptions? MoeOffload { get; init; }

    /// <summary>CPU thread count (0 = auto).</summary>
    public int Threads { get; init; }

    /// <summary>Decode thread count (0 = auto).</summary>
    public int DecodeThreads { get; init; }

    /// <summary>Host to bind to.</summary>
    public string Host { get; init; } = "localhost";

    /// <summary>Port to listen on.</summary>
    public int Port { get; init; } = 8080;

    /// <summary>KV-cache key type; <c>null</c> = not specified (the CPU paged path then defaults to Q8_0 blocks, #1284).</summary>
    public string? CacheTypeK { get; init; }

    /// <summary>KV-cache value type; <c>null</c> = not specified (see <see cref="CacheTypeK"/>).</summary>
    public string? CacheTypeV { get; init; }

    /// <summary>Whether prompt caching is enabled.</summary>
    public bool PromptCacheEnabled { get; init; } = true;

    /// <summary>Maximum number of cached sessions for prompt caching.</summary>
    public int PromptCacheSize { get; init; } = 4;

    /// <summary>Maximum recurrent prefix snapshots. Null preserves the single-slot default.</summary>
    public int? PrefixSnapshots { get; init; }

    /// <summary>Maximum page-locked host snapshot bytes. Zero disables host retention.</summary>
    public long PrefixSnapshotHostBudgetBytes { get; init; }

    /// <summary>Parent directory for ephemeral recurrent snapshot spill files.</summary>
    public string? PrefixSnapshotDiskDirectory { get; init; }

    /// <summary>Maximum snapshot spill bytes. Zero disables the disk tier.</summary>
    public long PrefixSnapshotDiskBudgetBytes { get; init; }

    /// <summary>Recurrent-state checkpoint every this many prefill tokens for mid-history resume (issue #1286); 0 disables.</summary>
    public int PromptCacheStep { get; init; }

    /// <summary>Most step checkpoints kept; null means the engine default (8).</summary>
    public int? PromptCacheStepKeep { get; init; }

    /// <summary>Validates and resolves recurrent snapshot limits for this server.</summary>
    public RecurrentPrefixCacheOptions GetPrefixSnapshotOptions()
    {
        int count = PrefixSnapshots ?? 1;
        if (count < 1) throw new ArgumentOutOfRangeException(nameof(PrefixSnapshots), "--prefix-snapshots must be at least 1.");
        if (PrefixSnapshotHostBudgetBytes < 0) throw new ArgumentOutOfRangeException(nameof(PrefixSnapshotHostBudgetBytes));
        if (PrefixSnapshotDiskBudgetBytes < 0) throw new ArgumentOutOfRangeException(nameof(PrefixSnapshotDiskBudgetBytes));
        if (PromptCacheStep < 0) throw new ArgumentOutOfRangeException(nameof(PromptCacheStep), "--prompt-cache-step must not be negative.");
        if (PromptCacheStepKeep < 1) throw new ArgumentOutOfRangeException(nameof(PromptCacheStepKeep), "--prompt-cache-step-keep must be at least 1.");
        string? directory = PrefixSnapshotDiskDirectory;
        if (directory is not null && string.IsNullOrWhiteSpace(directory))
            throw new ArgumentException("--prefix-snapshot-disk-dir must name a directory.");
        if ((directory is null) != (PrefixSnapshotDiskBudgetBytes == 0))
            throw new ArgumentException("--prefix-snapshot-disk-dir and a positive --prefix-snapshot-disk-budget must be supplied together.");
        if (directory is not null)
        {
            try { directory = Path.GetFullPath(directory); }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            { throw new ArgumentException("--prefix-snapshot-disk-dir is not a valid directory path.", ex); }
        }
        return new() { MaxEntries = count, HostBudgetBytes = PrefixSnapshotHostBudgetBytes,
            DiskDirectory = directory, DiskBudgetBytes = PrefixSnapshotDiskBudgetBytes,
            StepTokens = PromptCacheStep, MaxStepCheckpoints = PromptCacheStepKeep ?? new RecurrentPrefixCacheOptions().MaxStepCheckpoints };
    }

    /// <summary>Resolves CLI count and human-readable host byte budget before loading a model.</summary>
    public static RecurrentPrefixCacheOptions BuildPrefixSnapshotOptions(int? count, string? hostBudget,
        string? diskDirectory = null, string? diskBudget = null, int step = 0, int? stepKeep = null)
        => new ServerOptions { Model = "", PrefixSnapshots = count, PrefixSnapshotHostBudgetBytes = ParsePrefixSnapshotHostBudget(hostBudget),
            PrefixSnapshotDiskDirectory = diskDirectory, PrefixSnapshotDiskBudgetBytes = ParsePrefixSnapshotDiskBudget(diskBudget),
            PromptCacheStep = step, PromptCacheStepKeep = stepKeep }
            .GetPrefixSnapshotOptions();

    /// <summary>Parses nonnegative bytes or decimal/binary units, for example 8000000, 8GB or 8GiB.</summary>
    public static long ParsePrefixSnapshotHostBudget(string? value)
        => ParsePrefixSnapshotByteBudget(value, "--prefix-snapshot-host-budget");

    /// <summary>Parses the nonnegative spill budget using the same decimal/binary byte units as the host tier.</summary>
    public static long ParsePrefixSnapshotDiskBudget(string? value)
        => ParsePrefixSnapshotByteBudget(value, "--prefix-snapshot-disk-budget");

    private static long ParsePrefixSnapshotByteBudget(string? value, string flag)
    {
        if (value is null) return 0;
        string number = value.Trim();
        long multiplier = 1;
        foreach (var (suffix, scale) in new (string, long)[]
        {
            ("TiB", 1L << 40), ("GiB", 1L << 30), ("MiB", 1L << 20), ("KiB", 1L << 10),
            ("TB", 1_000_000_000_000L), ("GB", 1_000_000_000L), ("MB", 1_000_000L), ("KB", 1_000L), ("B", 1L),
        })
        {
            if (!number.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) continue;
            number = number[..^suffix.Length].TrimEnd();
            multiplier = scale;
            break;
        }
        if (!decimal.TryParse(number, System.Globalization.NumberStyles.AllowDecimalPoint,
                System.Globalization.CultureInfo.InvariantCulture, out decimal amount)
            || amount < 0 || amount > long.MaxValue / (decimal)multiplier)
            throw new ArgumentException($"{flag} must be a nonnegative byte size, such as 8GiB.", nameof(value));
        decimal bytes = amount * multiplier;
        if (bytes != decimal.Truncate(bytes))
            throw new ArgumentException($"{flag} must resolve to a whole number of bytes.", nameof(value));
        return (long)bytes;
    }

    /// <summary>Whether to use paged KV-cache (block-based allocation). Default true for serve.</summary>
    public bool UsePaged { get; init; } = true;

    /// <summary>Warm-up configuration for JIT pre-compilation and CUDA kernel loading.</summary>
    public WarmupOptions Warmup { get; init; } = WarmupOptions.Default;

    /// <summary>Draft model for speculative decoding. Null = disabled.</summary>
    public string? SpeculativeModel { get; init; }

    /// <summary>Number of draft candidates per speculative step (K). Also used as K for MTP
    /// self-speculative decoding (<see cref="MtpEnabled"/>) — both are the same "candidates per
    /// round" concept.</summary>
    public int SpeculativeCandidates { get; init; } = DotLLM.Engine.TextGenerator.DefaultSpeculativeCandidates;

    /// <summary>
    /// Multi-Token Prediction (MTP) self-speculative decoding (issue #253). <b>Defaults to <c>true</c></b>:
    /// when the loaded GGUF carries an embedded MTP head (<c>IModel.SupportsMtp</c>), no external draft model
    /// (<see cref="SpeculativeModel"/>) is given, and <see cref="ExpectedConcurrency"/> is below 5, <c>serve</c>
    /// enables it automatically with the adaptive gate (it backs off to plain decode when drafts stop paying).
    /// It is a no-op for any GGUF without a head. Opt out with <c>--no-mtp</c>; <c>--mtp</c> is still accepted
    /// (back-compat) and only changes the startup log wording - it does not override the concurrency or
    /// draft-model rules below.
    /// <para>
    /// Tradeoff: while MTP is active the continuous-batch scheduler is off for that model (MTP's single-sequence
    /// self-speculative loop does not batch requests), so many concurrent clients are better served with
    /// <c>--no-mtp</c> or <c>--expected-concurrency 5</c> (or more), which drops MTP for the model.
    /// </para>
    /// </summary>
    public bool MtpEnabled { get; init; } = true;

    /// <summary>True when <c>--mtp</c> was passed explicitly (back-compat); only affects log wording.</summary>
    public bool MtpExplicit { get; init; }

    /// <summary>
    /// Path of a separate MTP head GGUF for architectures that ship it beside the trunk (qwen4exp: <c>mtp-*.gguf</c>). Null = auto-detect a sibling
    /// file (see <c>Qwen4ExpMtpHeadResolver</c>). An explicit path always attaches; an auto-detected one attaches only when the trunk is large
    /// enough for MTP to pay (see <c>Qwen4ExpMtpHeadResolver.AutoAttachMinTrunkBytes</c>).
    /// </summary>
    public string? MtpHeadPath { get; init; }

    /// <summary>DFlash / DFlash2 drafter GGUF (<c>--dflash</c>, issue #1175); it takes the model's MTP slot. Null = none.</summary>
    public string? DFlashPath { get; init; }

    /// <summary>
    /// How reasoning ("thinking") output is separated from the answer (#767): <c>--reasoning-format
    /// none|auto|deepseek</c>. <see cref="DotLLM.Tokenizers.Reasoning.ReasoningFormat.Auto"/> (default) splits
    /// into <c>reasoning_content</c> + <c>content</c> when the chat template opened a think block or the model
    /// emits <c>&lt;think&gt;</c> itself; <c>none</c> returns the raw output in <c>content</c> as before.
    /// A request may override it with its own <c>reasoning_format</c> field.
    /// </summary>
    public DotLLM.Tokenizers.Reasoning.ReasoningFormat ReasoningFormat { get; init; } = DotLLM.Tokenizers.Reasoning.ReasoningFormat.Auto;

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

    /// <summary>
    /// Served context window in tokens (<c>--ctx-size</c>, llama.cpp <c>-c</c>). Null or 0 (default): models that size their
    /// own device memory (the CUDA hybrid) fit the window to free VRAM at load and report it in <c>/props</c>; others keep
    /// the checkpoint's window. A value overrides the fit (capped at the checkpoint's window) and is logged with a warning
    /// when it exceeds what fits: requests past the fit are then refused with an out-of-memory error rather than spilled
    /// (issue #1068).
    /// </summary>
    public int? ContextSize { get; init; }

    /// <summary>Default for <see cref="EffectiveMaxMessages"/> when <see cref="MaxMessages"/> is unset.</summary>
    public const int DefaultMaxMessages = 8192;

    /// <summary>Environment variable that supplies <see cref="MaxMessages"/> when <c>--max-messages</c> is not given.</summary>
    public const string MaxMessagesEnvVar = "DOTLLM_MAX_MESSAGES";

    /// <summary>
    /// Maximum number of messages accepted in one chat request on <c>/v1/chat/completions</c> and
    /// <c>/v1/messages</c> (#835). Null = <see cref="DefaultMaxMessages"/>; 0 = unlimited; negative is
    /// invalid (rejected by <see cref="ResolveMaxMessages"/>). Set with <c>--max-messages</c> or
    /// <c>DOTLLM_MAX_MESSAGES</c>. A cheap structural bound only - the prompt-length-vs-context check
    /// is the real protection for the engine.
    /// </summary>
    public int? MaxMessages { get; init; }

    /// <summary>The message-count cap in force: <see cref="MaxMessages"/> or the default. 0 means unlimited.</summary>
    public int EffectiveMaxMessages => MaxMessages ?? DefaultMaxMessages;

    /// <summary>
    /// Resolves the message cap from the CLI value and the <see cref="MaxMessagesEnvVar"/> value; the CLI wins.
    /// Returns null when neither is set (so the default applies). Throws <see cref="ArgumentException"/> for a
    /// negative or non-integer value, so a typo fails at startup instead of silently disabling the cap.
    /// </summary>
    public static int? ResolveMaxMessages(int? cliValue, string? envValue)
    {
        if (cliValue is int cli)
            return cli >= 0 ? cli : throw new ArgumentException(
                $"--max-messages must be >= 0 (0 = unlimited), got {cli}.");

        if (string.IsNullOrWhiteSpace(envValue))
            return null;

        if (!int.TryParse(envValue.Trim(), System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int env) || env < 0)
            throw new ArgumentException(
                $"{MaxMessagesEnvVar} must be an integer >= 0 (0 = unlimited), got '{envValue}'.");
        return env;
    }

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
    /// Whether the model-administration write endpoints introduced in #454 are enabled:
    /// <c>POST /v1/models/unload</c>, <c>POST /v1/models/pull</c> (+ its cancel route),
    /// <c>POST /v1/models/enable</c>, <c>POST /v1/models/disable</c> and
    /// <c>PUT /v1/settings</c>. The matching read-only routes (<c>GET /v1/settings</c>,
    /// <c>GET /v1/devices</c>, <c>GET /v1/models/pull</c>) are always available.
    /// Defaults to <c>false</c> — opt in via <c>--allow-model-admin</c>, mirroring
    /// <see cref="AllowLoraAdminApi"/>. Deliberately a separate flag: unloading models and
    /// rewriting residency settings is a different trust boundary from registering a LoRA adapter.
    /// </summary>
    public bool AllowModelAdminApi { get; init; }

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
    /// Expected number of concurrently decoding requests (<c>--expected-concurrency</c>). Vulkan hybrid models serve through the serial
    /// per-request generator by default; at <c>&gt;= 5</c> concurrent decode streams the continuous-batch scheduler measured +73% aggregate
    /// decode throughput on Tev1-4B (parity at &lt;= 4, slower for a repeated identical prompt, and it is off when MTP is active), so a hint of
    /// 5 or more enables it automatically and also drops default-on MTP for the model (see <see cref="MtpEnabled"/>). <c>DOTLLM_VK_SCHEDULER=0</c>/<c>1</c> overrides the hint.
    /// </summary>
    public int ExpectedConcurrency { get; init; }

    /// <summary>
    /// Download a missing Hugging Face model when a request or load names one (<c>--auto-pull</c>, issue #714). Off by default so a remote client
    /// cannot start large downloads; the CLI's own <c>run/chat/serve</c> always pull their main model.
    /// </summary>
    public bool AutoPull { get; init; }

    /// <summary>The device a model actually loaded on when <see cref="Device"/> is <c>auto</c> (<c>cpu</c>, <c>vulkan</c>, <c>gpu:0</c>); null otherwise.</summary>
    public string? ResolvedDevice { get; init; }

    /// <summary>Non-null when <c>--device auto</c> fell back from a faster device because the model is unsupported there (reason + perf consequence); surfaced in <c>/props</c>.</summary>
    public string? DeviceFallbackWarning { get; init; }

    /// <summary>
    /// Logit temperature for <c>/v1/systemone</c> probabilities (<c>--decision-temperature</c>); 0 or 1 = the raw restricted softmax. See
    /// <c>DecisionEvaluator.Temperature</c> for the measured effect on Tev1-4B.
    /// </summary>
    public double DecisionTemperature { get; init; }

    /// <summary>Option orderings averaged per noul/choice question (<c>--decision-orderings</c>): 1 (default) or 2 (forward + reversed).</summary>
    public int DecisionOrderings { get; init; } = 1;

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
        string device = "auto";
        int? gpuLayers = null;
        string? moeOffloadMode = null, moeCacheMib = null, moeCachePolicy = null, moeCold = null;
        int? moeCpuThreads = null, nCpuMoe = null, ctxSize = null;
        bool cpuMoe = false, moeCalibrate = false;
        int threads = 0;
        int decodeThreads = 0;
        string host = "localhost";
        int port = 8080;
        string? cacheTypeK = null;
        string? cacheTypeV = null;
        bool promptCacheEnabled = true;
        int promptCacheSize = 4;
        int? prefixSnapshots = null;
        string? prefixSnapshotHostBudget = null;
        string? prefixSnapshotDiskDirectory = null, prefixSnapshotDiskBudget = null;
        bool usePaged = true;
        bool warmupEnabled = true;
        int warmupIterations = 3;
        bool schedulerFairness = false;
        string? speculativeModel = null;
        int speculativeCandidates = DotLLM.Engine.TextGenerator.DefaultSpeculativeCandidates;
        bool mtpEnabled = true;
        bool mtpExplicit = false;
        string? mtpHeadPath = null;
        string? dflashPath = null;
        var reasoningFormat = DotLLM.Tokenizers.Reasoning.ReasoningFormat.Auto;
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
        bool allowModelAdmin = false;
        bool allowLoraAdmin = false;
        int? maxMessagesCli = null;
        int promptCacheStep = 0;
        int? promptCacheStepKeep = null;

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
                case "--moe-offload":
                    moeOffloadMode = next; i++; break;
                case "--moe-cache-mib":
                    moeCacheMib = next; i++; break;
                case "--moe-cache-policy":
                    moeCachePolicy = next; i++; break;
                case "--moe-cold":
                    moeCold = next; i++; break;
                case "--moe-cpu-threads":
                    moeCpuThreads = int.Parse(next!, System.Globalization.CultureInfo.InvariantCulture); i++; break;
                case "--n-cpu-moe":
                    nCpuMoe = int.Parse(next!, System.Globalization.CultureInfo.InvariantCulture); i++; break;
                case "--ctx-size" or "--ctx":
                    ctxSize = int.Parse(next!, System.Globalization.CultureInfo.InvariantCulture); i++; break;
                case "--cpu-moe":
                    cpuMoe = true; break;
                case "--moe-calibrate":
                    moeCalibrate = true; break;
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
                case "--mtp":
                    mtpEnabled = true; mtpExplicit = true; break;
                case "--no-mtp":
                    mtpEnabled = false; mtpExplicit = false; break;
                case "--mtp-head":
                    mtpHeadPath = next; i++; break;
                case "--dflash":
                    dflashPath = next; i++; break;
                case "--reasoning-format":
                    reasoningFormat = DotLLM.Tokenizers.Reasoning.ReasoningFormats.Parse(next ?? ""); i++; break;
                case "--prefill-chunk-size" or "--ubatch-size":
                    prefillChunkSize = int.Parse(next!); i++; break;
                case "--prefix-snapshots":
                    prefixSnapshots = int.Parse(next ?? throw new ArgumentException("--prefix-snapshots requires a count."), System.Globalization.CultureInfo.InvariantCulture); i++; break;
                case "--prefix-snapshot-host-budget":
                    prefixSnapshotHostBudget = next ?? throw new ArgumentException("--prefix-snapshot-host-budget requires a size."); i++; break;
                case "--prefix-snapshot-disk-dir":
                    prefixSnapshotDiskDirectory = next ?? throw new ArgumentException("--prefix-snapshot-disk-dir requires a path."); i++; break;
                case "--prefix-snapshot-disk-budget":
                    prefixSnapshotDiskBudget = next ?? throw new ArgumentException("--prefix-snapshot-disk-budget requires a size."); i++; break;
                case "--prompt-cache-step":
                    promptCacheStep = int.Parse(next ?? throw new ArgumentException("--prompt-cache-step requires a token count."), System.Globalization.CultureInfo.InvariantCulture); i++; break;
                case "--prompt-cache-step-keep":
                    promptCacheStepKeep = int.Parse(next ?? throw new ArgumentException("--prompt-cache-step-keep requires a count."), System.Globalization.CultureInfo.InvariantCulture); i++; break;
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
                case "--allow-model-admin":
                    allowModelAdmin = true; break;
                case "--allow-lora-admin":
                    allowLoraAdmin = true; break;
                case "--max-messages":
                    maxMessagesCli = int.Parse(next!, System.Globalization.CultureInfo.InvariantCulture); i++; break;
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

        // Repo ids keep their final component; local GGUF paths (either separator, relative or absolute) drop the extension,
        // so one file gets one id however its path was written (#1275).
        int lastSeparator = Math.Max(model.LastIndexOf('/'), model.LastIndexOf('\\'));
        string modelId = model[(lastSeparator + 1)..];
        if (modelId.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            modelId = modelId[..^5];

        var snapshotOptions = BuildPrefixSnapshotOptions(prefixSnapshots, prefixSnapshotHostBudget, prefixSnapshotDiskDirectory, prefixSnapshotDiskBudget,
            promptCacheStep, promptCacheStepKeep);
        return new ServerOptions
        {
            Model = model,
            Quant = quant,
            Device = device,
            GpuLayers = gpuLayers,
            MoeOffload = DotLLM.Core.Moe.MoeOffloadOptions.FromCli(new DotLLM.Core.Moe.MoeCliValues(moeOffloadMode, moeCacheMib,
                moeCachePolicy, moeCold, moeCpuThreads, cpuMoe, nCpuMoe, moeCalibrate, ctxSize)),
            Threads = threads,
            DecodeThreads = decodeThreads,
            Host = host,
            Port = port,
            CacheTypeK = cacheTypeK,
            CacheTypeV = cacheTypeV,
            PromptCacheEnabled = promptCacheEnabled,
            PromptCacheSize = promptCacheSize,
            PrefixSnapshots = snapshotOptions.MaxEntries,
            PrefixSnapshotHostBudgetBytes = snapshotOptions.HostBudgetBytes,
            PrefixSnapshotDiskDirectory = snapshotOptions.DiskDirectory,
            PrefixSnapshotDiskBudgetBytes = snapshotOptions.DiskBudgetBytes,
            PromptCacheStep = snapshotOptions.StepTokens,
            PromptCacheStepKeep = promptCacheStepKeep,
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
            MtpEnabled = mtpEnabled,
            MtpExplicit = mtpExplicit,
            MtpHeadPath = mtpHeadPath,
            DFlashPath = dflashPath,
            ReasoningFormat = reasoningFormat,
            PrefillChunkSize = prefillChunkSize,
            KeepAliveSeconds = keepAliveSeconds,
            MaxResidentModels = maxResidentModels,
            ResidentMemoryBudgetBytes = residentMemoryBudgetBytes,
            AllowModelAdminApi = allowModelAdmin,
            AllowLoraAdminApi = allowLoraAdmin,
            MaxMessages = ResolveMaxMessages(maxMessagesCli, Environment.GetEnvironmentVariable(MaxMessagesEnvVar)),
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
