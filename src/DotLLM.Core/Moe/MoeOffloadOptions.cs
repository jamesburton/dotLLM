using System.Globalization;

namespace DotLLM.Core.Moe;

/// <summary>Whether a routed-MoE model keeps its experts resident in device memory or offloads them (issue #1070).</summary>
public enum MoeOffloadMode
{
    /// <summary>Resident when the whole model fits the device, otherwise offload (the default).</summary>
    Auto,

    /// <summary>Never offload: refuse a model whose experts do not fit (the pre-#1070 behaviour).</summary>
    Off,

    /// <summary>Always offload, even when the model would fit (used to exercise the offload path on small models).</summary>
    On,
}

/// <summary>Replacement policy of the device expert cache (<see cref="ExpertCache"/>).</summary>
public enum ExpertCachePolicy
{
    /// <summary>
    /// Segmented LRU (the default): new experts enter a probation segment, a re-hit promotes them to a protected segment,
    /// and eviction takes the least recently used probation entry first (llama.cpp #30637 style).
    /// </summary>
    Slru,

    /// <summary>Plain least recently used.</summary>
    Lru,

    /// <summary>Least frequently used (lookups since insertion), ties broken by least recent use.</summary>
    Lfu,
}

/// <summary>Where an offloaded routed expert that is not in the device cache runs (issue #1071, design #1063 section 8c).</summary>
public enum MoeColdMode
{
    /// <summary>
    /// Per expert, by a cost model calibrated at load time: the CPU when computing it there is cheaper than copying it in and
    /// running it on the device, balanced so neither side waits much for the other (Fiddler rule plus a DALI-style greedy pass).
    /// Experts computed on the CPU enter the cache only through promote-only admission.
    /// </summary>
    Auto,

    /// <summary>Copy every miss into the device cache and run it there (the #1070 copy-on-miss behaviour).</summary>
    Copy,

    /// <summary>Compute every uncached expert on the CPU; the cache changes only through promote-only admission.</summary>
    Cpu,
}

/// <summary>
/// The raw values of the MoE offload command-line options shared by <c>run</c>, <c>chat</c>, <c>serve</c> and <c>bench</c>
/// (null = not given; each unset value falls back to its environment variable in <see cref="MoeOffloadOptions.FromCli(MoeCliValues, Action{string}?, Func{string, string?}?)"/>).
/// </summary>
/// <param name="Mode"><c>--moe-offload</c>.</param>
/// <param name="CacheMib"><c>--moe-cache-mib</c>.</param>
/// <param name="Policy"><c>--moe-cache-policy</c>.</param>
/// <param name="Cold"><c>--moe-cold</c>.</param>
/// <param name="CpuThreads"><c>--moe-cpu-threads</c>.</param>
/// <param name="CpuMoe"><c>--cpu-moe</c> (true = every MoE layer's experts on the CPU).</param>
/// <param name="NCpuMoe"><c>--n-cpu-moe</c>.</param>
/// <param name="Calibrate"><c>--moe-calibrate</c> (true = re-run the load-time cost calibration).</param>
/// <param name="ContextTokens"><c>--ctx-size</c>: context the device budget reserves KV cache for.</param>
public sealed record MoeCliValues(
    string? Mode = null, string? CacheMib = null, string? Policy = null, string? Cold = null, int? CpuThreads = null,
    bool CpuMoe = false, int? NCpuMoe = null, bool Calibrate = false, int? ContextTokens = null);

/// <summary>
/// Options for routed-expert offload on a GPU backend (issue #1070, design #1063 section 8b/8d; cold experts on the CPU:
/// #1071, section 8c). Every property has an environment-variable fallback (see <see cref="WithEnvironmentFallback"/>), so a
/// library caller that passes nothing still honours the <c>DOTLLM_MOE_*</c> variables.
/// </summary>
public sealed record MoeOffloadOptions
{
    /// <summary>Environment variable for <see cref="Mode"/> (<c>auto</c>, <c>off</c> or <c>on</c>).</summary>
    public const string ModeEnvVar = "DOTLLM_MOE_OFFLOAD";

    /// <summary>Environment variable for <see cref="CacheMib"/> (a positive integer, or <c>auto</c>).</summary>
    public const string CacheMibEnvVar = "DOTLLM_MOE_CACHE_MIB";

    /// <summary>Environment variable for <see cref="Policy"/> (<c>slru</c>, <c>lru</c> or <c>lfu</c>).</summary>
    public const string PolicyEnvVar = "DOTLLM_MOE_CACHE_POLICY";

    /// <summary>Environment variable for <see cref="Cold"/> (<c>auto</c>, <c>copy</c> or <c>cpu</c>).</summary>
    public const string ColdEnvVar = "DOTLLM_MOE_COLD";

    /// <summary>Environment variable for <see cref="CpuThreads"/> (a positive integer).</summary>
    public const string CpuThreadsEnvVar = "DOTLLM_MOE_CPU_THREADS";

    /// <summary>Environment variable for <see cref="Calibrate"/> (<c>1</c> forces a re-calibration).</summary>
    public const string CalibrateEnvVar = "DOTLLM_MOE_CALIBRATE";

    /// <summary>Environment variable for <see cref="ContextTokens"/> (a positive integer).</summary>
    public const string ContextEnvVar = "DOTLLM_MOE_CTX";

    /// <summary>
    /// Environment variable for <see cref="CpuMoeLayers"/> (<c>--n-cpu-moe</c>; the variable the Vulkan Qwen3MoeHybrid offload
    /// (#370) already reads).
    /// </summary>
    public const string NCpuMoeEnvVar = "DOTLLM_N_CPU_MOE";

    /// <summary>Environment variable for <c>--cpu-moe</c> (<c>1</c> = every MoE layer's experts on the CPU).</summary>
    public const string CpuMoeEnvVar = "DOTLLM_CPU_MOE";

    /// <summary>Default share of physical RAM the pinned host expert store may take (design 8b: 45 %).</summary>
    public const double DefaultPinBudgetFraction = 0.45;

    /// <summary><see cref="CpuMoeLayers"/> value of <c>--cpu-moe</c>: every MoE layer.</summary>
    public const int AllLayers = int.MaxValue;

    /// <summary>The defaults: <see cref="MoeOffloadMode.Auto"/>, an automatic cache size, SLRU.</summary>
    public static MoeOffloadOptions Default { get; } = new();

    /// <summary>Resident, offload, or decide by fit.</summary>
    public MoeOffloadMode Mode { get; init; } = MoeOffloadMode.Auto;

    /// <summary>
    /// Device expert-cache size in MiB; <see langword="null"/> = the whole measured budget. A request above the budget is
    /// clamped to it with a warning (WDDM spills instead of failing, so the budget is never exceeded).
    /// </summary>
    public long? CacheMib { get; init; }

    /// <summary>Cache replacement policy.</summary>
    public ExpertCachePolicy Policy { get; init; } = ExpertCachePolicy.Slru;

    /// <summary>Where uncached experts run when the experts are offloaded (#1071).</summary>
    public MoeColdMode Cold { get; init; } = MoeColdMode.Auto;

    /// <summary>
    /// Worker threads of the CPU cold-expert leg (one expert per worker at a time, each single-threaded);
    /// <see langword="null"/> = physical cores − 1 (<see cref="DefaultCpuThreads"/>).
    /// </summary>
    public int? CpuThreads { get; init; }

    /// <summary>
    /// llama.cpp-compatible static placement (<c>--n-cpu-moe N</c> / <c>--cpu-moe</c>): the routed experts of every MoE layer
    /// whose layer index is below this value are never cached and always run on the CPU. 0 = none,
    /// <see cref="AllLayers"/> = every MoE layer. A positive value offloads the experts even when the model would fit.
    /// </summary>
    public int CpuMoeLayers { get; init; }

    /// <summary>Re-run the load-time cost calibration instead of reading the cached record (<c>--moe-calibrate</c>).</summary>
    public bool Calibrate { get; init; }

    /// <summary>
    /// Largest share of physical RAM the offloaded expert banks may occupy as a pinned (page-locked) host copy. Above it the
    /// banks stay in the memory-mapped file and reach the device through a small pinned staging ring.
    /// </summary>
    public double PinBudgetFraction { get; init; } = DefaultPinBudgetFraction;

    /// <summary>
    /// Device memory kept free after every allocation, in MiB; <see langword="null"/> = the backend's default (512 MiB, the
    /// #1063 addendum's display-attached figure).
    /// </summary>
    public long? ReserveMib { get; init; }

    /// <summary>
    /// Context length the device budget reserves KV cache for; <see langword="null"/> = the model's maximum context (the
    /// #1063 addendum's worst case). Set from <c>--ctx-size</c>; lower it when the caller never allocates a cache that long.
    /// </summary>
    public int? ContextTokens { get; init; }

    /// <summary>
    /// Path of the calibration cache file; <see langword="null"/> = <see cref="MoeCalibrationStore.DefaultPath"/>
    /// (<c>%LOCALAPPDATA%\dotllm\moe-calibration.json</c>). Tests point it at a scratch file.
    /// </summary>
    public string? CalibrationPath { get; init; }

    /// <summary>
    /// Fixed cost model that replaces the load-time calibration (tests, or a caller that measured elsewhere);
    /// <see langword="null"/> = calibrate (or read the cached record).
    /// </summary>
    public MoeCostModel? CostModelOverride { get; init; }

    /// <summary>Model file the experts come from; keys the calibration cache (set by the loader).</summary>
    public string? ModelPath { get; init; }

    /// <summary>Optional sink for load-time notes and warnings (clamped cache size, spill probe results); null = stderr.</summary>
    public Action<string>? Log { get; init; }

    /// <summary>True when offloaded experts may run on the CPU (a cold mode other than copy, or static CPU layers).</summary>
    public bool MayUseCpuLeg => Cold != MoeColdMode.Copy || CpuMoeLayers > 0;

    /// <summary>True when <paramref name="layerIndex"/> is statically placed on the CPU (<see cref="CpuMoeLayers"/>).</summary>
    /// <param name="layerIndex">Model layer index.</param>
    /// <returns>True for a static CPU layer.</returns>
    public bool IsStaticCpuLayer(int layerIndex) => layerIndex < CpuMoeLayers;

    /// <summary>Worker threads of the CPU cold leg: <see cref="CpuThreads"/>, else <see cref="DefaultCpuThreads"/>.</summary>
    public int EffectiveCpuThreads => Math.Max(1, CpuThreads ?? DefaultCpuThreads());

    /// <summary>
    /// Default CPU cold-leg worker count: physical cores − 1 (the host thread enqueues the GPU work, then joins the workers).
    /// Physical cores are taken as half the logical processors when there are more than four (SMT), which is right for the
    /// common two-way SMT parts; at least one.
    /// </summary>
    /// <returns>The worker count.</returns>
    public static int DefaultCpuThreads()
    {
        int logical = Environment.ProcessorCount;
        int physical = logical > 4 ? logical / 2 : logical;
        return Math.Max(1, physical - 1);
    }

    /// <summary>
    /// Fills every property the caller left at its default from the <c>DOTLLM_MOE_*</c> environment variables. Explicit values
    /// (anything not equal to the default) win over the environment.
    /// </summary>
    /// <param name="options">Caller options, or null for the defaults.</param>
    /// <param name="getEnv">Environment lookup (injectable for tests); null = <see cref="Environment.GetEnvironmentVariable(string)"/>.</param>
    /// <returns>The resolved options.</returns>
    /// <exception cref="ArgumentException">An environment variable holds a value that does not parse.</exception>
    public static MoeOffloadOptions WithEnvironmentFallback(MoeOffloadOptions? options, Func<string, string?>? getEnv = null)
    {
        getEnv ??= Environment.GetEnvironmentVariable;
        var o = options ?? Default;
        if (o.Mode == MoeOffloadMode.Auto && getEnv(ModeEnvVar) is { Length: > 0 } mode)
            o = o with { Mode = ParseMode(mode) };
        if (o.CacheMib is null && getEnv(CacheMibEnvVar) is { Length: > 0 } mib)
            o = o with { CacheMib = ParseCacheMib(mib) };
        if (o.Policy == ExpertCachePolicy.Slru && getEnv(PolicyEnvVar) is { Length: > 0 } policy)
            o = o with { Policy = ParsePolicy(policy) };
        if (o.Cold == MoeColdMode.Auto && getEnv(ColdEnvVar) is { Length: > 0 } cold)
            o = o with { Cold = ParseCold(cold) };
        if (o.CpuThreads is null && getEnv(CpuThreadsEnvVar) is { Length: > 0 } threads)
            o = o with { CpuThreads = ParsePositive(threads, CpuThreadsEnvVar) };
        if (o.CpuMoeLayers == 0)
            o = o with { CpuMoeLayers = ResolveCpuMoeLayers(false, null, getEnv) };
        if (!o.Calibrate && getEnv(CalibrateEnvVar) is "1" or "true")
            o = o with { Calibrate = true };
        if (o.ContextTokens is null && getEnv(ContextEnvVar) is { Length: > 0 } ctx)
            o = o with { ContextTokens = ParsePositive(ctx, ContextEnvVar) };
        return o;
    }

    /// <summary>
    /// The static CPU-layer count of <c>--cpu-moe</c> / <c>--n-cpu-moe N</c>, shared by the CUDA expert offload and the Vulkan
    /// Qwen3MoeHybrid offload (#370): <c>--cpu-moe</c> wins (every layer), then an explicit N, then <c>DOTLLM_CPU_MOE=1</c>, then
    /// <c>DOTLLM_N_CPU_MOE</c>; 0 when none is set.
    /// </summary>
    /// <param name="cpuMoe"><c>--cpu-moe</c>.</param>
    /// <param name="nCpuMoe"><c>--n-cpu-moe</c>, or null when not given.</param>
    /// <param name="getEnv">Environment lookup (injectable for tests); null = the process environment.</param>
    /// <returns>Layers (by index, from 0) whose experts run on the CPU; <see cref="AllLayers"/> for all.</returns>
    /// <exception cref="ArgumentException">N is negative, or an environment variable does not parse.</exception>
    public static int ResolveCpuMoeLayers(bool cpuMoe, int? nCpuMoe, Func<string, string?>? getEnv = null)
    {
        if (cpuMoe) return AllLayers;
        if (nCpuMoe is { } n)
        {
            if (n < 0) throw new ArgumentException($"--n-cpu-moe must be >= 0, got {n}.", nameof(nCpuMoe));
            return n;
        }
        getEnv ??= Environment.GetEnvironmentVariable;
        if (getEnv(CpuMoeEnvVar) is "1" or "true") return AllLayers;
        if (getEnv(NCpuMoeEnvVar) is { Length: > 0 } raw)
        {
            if (!int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int env) || env < 0)
                throw new ArgumentException($"Invalid {NCpuMoeEnvVar} '{raw}'. Expected a non-negative integer.", nameof(getEnv));
            return env;
        }
        return 0;
    }

    /// <summary>Parses <c>auto</c>, <c>off</c> or <c>on</c> (case-insensitive).</summary>
    /// <param name="value">The text.</param>
    /// <returns>The mode.</returns>
    /// <exception cref="ArgumentException">Not one of the three values.</exception>
    public static MoeOffloadMode ParseMode(string value) => value.Trim().ToLowerInvariant() switch
    {
        "auto" => MoeOffloadMode.Auto,
        "off" => MoeOffloadMode.Off,
        "on" => MoeOffloadMode.On,
        _ => throw new ArgumentException($"Unknown MoE offload mode '{value}'. Expected auto, off or on.", nameof(value)),
    };

    /// <summary>Parses <c>slru</c>, <c>lru</c> or <c>lfu</c> (case-insensitive).</summary>
    /// <param name="value">The text.</param>
    /// <returns>The policy.</returns>
    /// <exception cref="ArgumentException">Not one of the three values.</exception>
    public static ExpertCachePolicy ParsePolicy(string value) => value.Trim().ToLowerInvariant() switch
    {
        "slru" => ExpertCachePolicy.Slru,
        "lru" => ExpertCachePolicy.Lru,
        "lfu" => ExpertCachePolicy.Lfu,
        _ => throw new ArgumentException($"Unknown MoE cache policy '{value}'. Expected slru, lru or lfu.", nameof(value)),
    };

    /// <summary>Parses <c>auto</c>, <c>copy</c> or <c>cpu</c> (case-insensitive).</summary>
    /// <param name="value">The text.</param>
    /// <returns>The cold mode.</returns>
    /// <exception cref="ArgumentException">Not one of the three values.</exception>
    public static MoeColdMode ParseCold(string value) => value.Trim().ToLowerInvariant() switch
    {
        "auto" => MoeColdMode.Auto,
        "copy" => MoeColdMode.Copy,
        "cpu" => MoeColdMode.Cpu,
        _ => throw new ArgumentException($"Unknown MoE cold mode '{value}'. Expected auto, copy or cpu.", nameof(value)),
    };

    /// <summary>Parses a cache size in MiB: a positive integer, or <c>auto</c> for <see langword="null"/>.</summary>
    /// <param name="value">The text.</param>
    /// <returns>The size, or null for automatic.</returns>
    /// <exception cref="ArgumentException">Neither <c>auto</c> nor a positive integer.</exception>
    public static long? ParseCacheMib(string value)
    {
        string v = value.Trim();
        if (v.Equals("auto", StringComparison.OrdinalIgnoreCase)) return null;
        if (long.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out long mib) && mib > 0) return mib;
        throw new ArgumentException($"Invalid MoE cache size '{value}' (MiB). Expected a positive integer or 'auto'.", nameof(value));
    }

    private static int ParsePositive(string value, string name)
    {
        if (int.TryParse(value.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0) return n;
        throw new ArgumentException($"Invalid {name} '{value}'. Expected a positive integer.", nameof(value));
    }

    /// <summary>
    /// Builds options from the three #1070 CLI values; each unset value falls back to its environment variable.
    /// </summary>
    /// <param name="mode"><c>--moe-offload</c>, or null.</param>
    /// <param name="cacheMib"><c>--moe-cache-mib</c>, or null.</param>
    /// <param name="policy"><c>--moe-cache-policy</c>, or null.</param>
    /// <param name="log">Optional sink for load-time notes and warnings.</param>
    /// <param name="getEnv">Environment lookup (injectable for tests).</param>
    /// <returns>The resolved options.</returns>
    /// <exception cref="ArgumentException">A value (CLI or environment) does not parse.</exception>
    public static MoeOffloadOptions FromCli(string? mode, string? cacheMib, string? policy, Action<string>? log = null,
        Func<string, string?>? getEnv = null)
        => FromCli(new MoeCliValues(mode, cacheMib, policy), log, getEnv);

    /// <summary>
    /// Builds options from the CLI values shared by <c>run</c>, <c>chat</c>, <c>serve</c> and <c>bench</c>; each unset value
    /// falls back to its environment variable.
    /// </summary>
    /// <param name="values">The raw CLI values.</param>
    /// <param name="log">Optional sink for load-time notes and warnings.</param>
    /// <param name="getEnv">Environment lookup (injectable for tests).</param>
    /// <returns>The resolved options.</returns>
    /// <exception cref="ArgumentException">A value (CLI or environment) does not parse, or the values conflict.</exception>
    public static MoeOffloadOptions FromCli(MoeCliValues values, Action<string>? log = null, Func<string, string?>? getEnv = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        getEnv ??= Environment.GetEnvironmentVariable;
        var o = new MoeOffloadOptions { Log = log };

        // An explicit CLI value wins even when it equals the default, so apply the fallback per unset option only.
        if (Pick(values.Mode, ModeEnvVar) is { } mode) o = o with { Mode = ParseMode(mode) };
        if (Pick(values.CacheMib, CacheMibEnvVar) is { } mib) o = o with { CacheMib = ParseCacheMib(mib) };
        if (Pick(values.Policy, PolicyEnvVar) is { } policy) o = o with { Policy = ParsePolicy(policy) };
        if (Pick(values.Cold, ColdEnvVar) is { } cold) o = o with { Cold = ParseCold(cold) };

        if (values.CpuThreads is { } threads)
            o = o with { CpuThreads = threads > 0 ? threads : throw new ArgumentException($"--moe-cpu-threads must be positive, got {threads}.", nameof(values)) };
        else if (getEnv(CpuThreadsEnvVar) is { Length: > 0 } envThreads)
            o = o with { CpuThreads = ParsePositive(envThreads, CpuThreadsEnvVar) };

        if (values.ContextTokens is { } ctx)
            o = o with { ContextTokens = ctx > 0 ? ctx : throw new ArgumentException($"--ctx-size must be positive, got {ctx}.", nameof(values)) };
        else if (getEnv(ContextEnvVar) is { Length: > 0 } envCtx)
            o = o with { ContextTokens = ParsePositive(envCtx, ContextEnvVar) };

        o = o with
        {
            CpuMoeLayers = ResolveCpuMoeLayers(values.CpuMoe, values.NCpuMoe, getEnv),
            Calibrate = values.Calibrate || getEnv(CalibrateEnvVar) is "1" or "true",
        };
        if (o.CpuMoeLayers > 0 && o.Mode == MoeOffloadMode.Off)
            throw new ArgumentException("--cpu-moe / --n-cpu-moe place experts on the CPU, which --moe-offload off forbids.", nameof(values));
        return o;

        string? Pick(string? cli, string env) => !string.IsNullOrWhiteSpace(cli) ? cli : getEnv(env) is { Length: > 0 } e ? e : null;
    }

    /// <summary>Returns a user-facing error for invalid CLI values, else null (for Spectre <c>Validate</c>).</summary>
    /// <param name="mode"><c>--moe-offload</c>, or null.</param>
    /// <param name="cacheMib"><c>--moe-cache-mib</c>, or null.</param>
    /// <param name="policy"><c>--moe-cache-policy</c>, or null.</param>
    /// <returns>The error message, or null when all three are valid.</returns>
    public static string? Validate(string? mode, string? cacheMib, string? policy)
        => Validate(new MoeCliValues(mode, cacheMib, policy), getEnv: _ => null);

    /// <summary>
    /// Returns a user-facing error for invalid CLI values (and, unless <paramref name="getEnv"/> hides them, invalid
    /// <c>DOTLLM_MOE_*</c> fallbacks), else null.
    /// </summary>
    /// <param name="values">The raw CLI values.</param>
    /// <param name="getEnv">Environment lookup; null = the process environment.</param>
    /// <returns>The error message, or null.</returns>
    public static string? Validate(MoeCliValues values, Func<string, string?>? getEnv = null)
    {
        try
        {
            FromCli(values, getEnv: getEnv);
            return null;
        }
        catch (ArgumentException ex)
        {
            return ex.Message.Split(" (Parameter", 2)[0];
        }
    }

    /// <summary>Help text of <c>--moe-offload</c>, shared by every command.</summary>
    public const string ModeHelp =
        "Routed-MoE expert placement on CUDA (nemotron_h_moe): 'auto' (default: keep experts in VRAM when the model fits, else " +
        "offload them to a host store with a VRAM expert cache), 'off' (refuse a model that does not fit), 'on' (always offload). " +
        "Env fallback: " + ModeEnvVar + ".";

    /// <summary>Help text of <c>--moe-cache-mib</c>.</summary>
    public const string CacheMibHelp =
        "VRAM expert-cache size in MiB when experts are offloaded; default 'auto' = the measured free-VRAM budget. Larger values are " +
        "clamped to the budget. Env fallback: " + CacheMibEnvVar + ".";

    /// <summary>Help text of <c>--moe-cache-policy</c>.</summary>
    public const string PolicyHelp =
        "Expert-cache replacement policy: 'slru' (default), 'lru' or 'lfu'. Env fallback: " + PolicyEnvVar + ".";

    /// <summary>Help text of <c>--moe-cold</c>.</summary>
    public const string ColdHelp =
        "Where offloaded experts that are not in the VRAM cache run: 'auto' (default: per expert, CPU or copy-to-GPU by a cost " +
        "model calibrated at load), 'copy' (copy every miss into the cache), 'cpu' (compute every uncached expert on the CPU; " +
        "an expert enters the cache only when its recent use beats the weakest cached one). Env fallback: " + ColdEnvVar + ".";

    /// <summary>Help text of <c>--moe-cpu-threads</c>.</summary>
    public const string CpuThreadsHelp =
        "Worker threads for MoE experts computed on the CPU (one expert per worker); default physical cores - 1. Env fallback: " +
        CpuThreadsEnvVar + ".";

    /// <summary>Help text of <c>--cpu-moe</c>.</summary>
    public const string CpuMoeHelp =
        "Keep every MoE layer's routed experts on the CPU (llama.cpp --cpu-moe). CUDA nemotron_h_moe: never cached, always computed " +
        "on the CPU; Vulkan Qwen3MoeHybrid (bench): as --n-cpu-moe with every layer. Env fallback: " + CpuMoeEnvVar + "=1.";

    /// <summary>Help text of <c>--n-cpu-moe</c>.</summary>
    public const string NCpuMoeHelp =
        "Keep the routed experts of the first N layers (by layer index, llama.cpp --n-cpu-moe) on the CPU. CUDA nemotron_h_moe: " +
        "never cached, always computed on the CPU; Vulkan Qwen3MoeHybrid: CPU expert compute for those layers (#370). " +
        "Env fallback: " + NCpuMoeEnvVar + ".";

    /// <summary>Help text of <c>--moe-calibrate</c>.</summary>
    public const string CalibrateHelp =
        "Re-run the MoE offload cost calibration (PCIe bandwidth, sync latency, CPU time per expert) instead of reading the cached " +
        "record in %LOCALAPPDATA%/dotllm/moe-calibration.json. Env fallback: " + CalibrateEnvVar + "=1.";

    /// <summary>Help text of <c>--ctx-size</c>.</summary>
    public const string ContextHelp =
        "Context length the MoE expert-offload VRAM budget reserves KV cache for (default: the model's maximum context, e.g. 262144 " +
        "for Nemotron 3.5 Bolt). Lower it to leave more VRAM for the expert cache. Env fallback: " + ContextEnvVar + ".";
}
