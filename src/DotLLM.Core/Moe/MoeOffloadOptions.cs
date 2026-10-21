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

/// <summary>
/// Options for routed-expert offload on a GPU backend (issue #1070, design #1063 section 8b/8d). Every property has an
/// environment-variable fallback (see <see cref="WithEnvironmentFallback"/>), so a library caller that passes nothing still honours
/// <c>DOTLLM_MOE_OFFLOAD</c>, <c>DOTLLM_MOE_CACHE_MIB</c> and <c>DOTLLM_MOE_CACHE_POLICY</c>.
/// </summary>
public sealed record MoeOffloadOptions
{
    /// <summary>Environment variable for <see cref="Mode"/> (<c>auto</c>, <c>off</c> or <c>on</c>).</summary>
    public const string ModeEnvVar = "DOTLLM_MOE_OFFLOAD";

    /// <summary>Environment variable for <see cref="CacheMib"/> (a positive integer, or <c>auto</c>).</summary>
    public const string CacheMibEnvVar = "DOTLLM_MOE_CACHE_MIB";

    /// <summary>Environment variable for <see cref="Policy"/> (<c>slru</c>, <c>lru</c> or <c>lfu</c>).</summary>
    public const string PolicyEnvVar = "DOTLLM_MOE_CACHE_POLICY";

    /// <summary>Default share of physical RAM the pinned host expert store may take (design 8b: 45 %).</summary>
    public const double DefaultPinBudgetFraction = 0.45;

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
    /// #1063 addendum's worst case). Lower it when the caller never allocates a cache that long.
    /// </summary>
    public int? ContextTokens { get; init; }

    /// <summary>Optional sink for load-time notes and warnings (clamped cache size, spill probe results); null = stderr.</summary>
    public Action<string>? Log { get; init; }

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
        return o;
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

    /// <summary>
    /// Builds options from the three CLI values shared by <c>run</c>, <c>chat</c>, <c>serve</c> and <c>bench</c>; each unset value
    /// falls back to its environment variable.
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
    {
        var o = new MoeOffloadOptions { Log = log };
        if (!string.IsNullOrWhiteSpace(mode)) o = o with { Mode = ParseMode(mode) };
        if (!string.IsNullOrWhiteSpace(cacheMib)) o = o with { CacheMib = ParseCacheMib(cacheMib) };
        if (!string.IsNullOrWhiteSpace(policy)) o = o with { Policy = ParsePolicy(policy) };

        // An explicit CLI value wins even when it equals the default, so apply the fallback per unset option only.
        getEnv ??= Environment.GetEnvironmentVariable;
        if (string.IsNullOrWhiteSpace(mode) && getEnv(ModeEnvVar) is { Length: > 0 } envMode) o = o with { Mode = ParseMode(envMode) };
        if (string.IsNullOrWhiteSpace(cacheMib) && getEnv(CacheMibEnvVar) is { Length: > 0 } envMib) o = o with { CacheMib = ParseCacheMib(envMib) };
        if (string.IsNullOrWhiteSpace(policy) && getEnv(PolicyEnvVar) is { Length: > 0 } envPolicy) o = o with { Policy = ParsePolicy(envPolicy) };
        return o;
    }

    /// <summary>Returns a user-facing error for invalid CLI values, else null (for Spectre <c>Validate</c>).</summary>
    /// <param name="mode"><c>--moe-offload</c>, or null.</param>
    /// <param name="cacheMib"><c>--moe-cache-mib</c>, or null.</param>
    /// <param name="policy"><c>--moe-cache-policy</c>, or null.</param>
    /// <returns>The error message, or null when all three are valid.</returns>
    public static string? Validate(string? mode, string? cacheMib, string? policy)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(mode)) ParseMode(mode);
            if (!string.IsNullOrWhiteSpace(cacheMib)) ParseCacheMib(cacheMib);
            if (!string.IsNullOrWhiteSpace(policy)) ParsePolicy(policy);
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
}
