namespace DotLLM.Server.Tuning;

/// <summary>Where a setting's value came from, reported on <c>/v1/status</c> as <c>flag</c>, <c>auto</c> or <c>default</c>.</summary>
public enum SettingSource
{
    /// <summary>The built-in default.</summary>
    Default,
    /// <summary>An explicit command-line flag.</summary>
    Flag,
    /// <summary>A saved tuning result applied by <c>--auto</c>.</summary>
    Auto,
}

/// <summary>The command a result is being applied to; TurboQuant eligibility depends on it.</summary>
public enum AutoCommand
{
    /// <summary><c>dotllm serve</c>.</summary>
    Serve,
    /// <summary><c>dotllm run</c>.</summary>
    Run,
    /// <summary><c>dotllm chat</c>.</summary>
    Chat,
    /// <summary><c>dotllm bench</c>.</summary>
    Bench,
}

/// <summary>What a command is, as far as the applier's constraints go.</summary>
/// <param name="Command">The command.</param>
/// <param name="SingleUser">True for a single-user serve (the <c>--single-user</c> profile arrives in slice 4; slice 1 never sets it).</param>
public readonly record struct AutoPolicy(AutoCommand Command, bool SingleUser = false);

/// <summary>The outcome of <see cref="AutoSettings.Apply"/>.</summary>
/// <param name="Applied">The settings to apply; null members mean "keep what the command has" (a flag or the default).</param>
/// <param name="Sources">Source of every known setting.</param>
/// <param name="Match">The store match.</param>
/// <param name="Warnings">Warnings to log.</param>
/// <param name="Advice">The line telling the user what to run, or null for a clean hit.</param>
public sealed record AutoApplied(TunedSettings Applied, IReadOnlyDictionary<string, SettingSource> Sources, TuningMatch Match,
                                 IReadOnlyList<string> Warnings, string? Advice);

/// <summary>Whether the current build can run a candidate's settings; null = yes, else the reason it cannot.</summary>
public delegate string? SettingRefusal(TunedSettings candidate);

/// <summary>
/// Whether the saved placement is admitted against free memory now, at <paramref name="servedContext"/>: null when it is, else the reason it
/// is not (with required and free bytes when both are known). A peak that was never recorded, or free memory that cannot be read, is a
/// refusal, never an admission.
/// </summary>
public delegate string? PlacementAdmission(TunedSettings candidate, int servedContext);

/// <summary>
/// Applies a saved tuning result under the precedence rules (#1902): explicit flags win; a miss applies nothing; a saved TurboQuant pair is
/// applied whole or not at all; a setting the build refuses is dropped, never substituted; and the saved placement must still fit free
/// memory now, or every saved setting is dropped (no spill). Admission only admits what the saved result bounds: its peak bytes were
/// measured at <see cref="TunedSettings.TunedContext"/>, so a result without one, or a served context above it, is not admitted.
/// </summary>
public static class AutoSettings
{
    /// <summary>Canonical setting names.</summary>
    public static readonly IReadOnlyList<string> Names =
    [
        "device", "gpu_layers", "moe_offload", "cache_type_k", "cache_type_v", "paged", "prefill_chunk_size", "mtp",
        "speculative_k", "threads", "decode_threads", "expected_concurrency", "ctx_size",
    ];

    private static readonly Dictionary<string, string> s_flags = new(StringComparer.Ordinal)
    {
        ["--device"] = "device", ["-d"] = "device",
        ["--gpu-layers"] = "gpu_layers",
        ["--moe-offload"] = "moe_offload", ["--cpu-moe"] = "moe_offload", ["--n-cpu-moe"] = "moe_offload", ["--moe-cache-mib"] = "moe_offload",
        ["--cache-type-k"] = "cache_type_k", ["--kv-quant-k"] = "cache_type_k",
        ["--cache-type-v"] = "cache_type_v", ["--kv-quant-v"] = "cache_type_v",
        ["--paged"] = "paged", ["--no-paged"] = "paged",
        ["--prefill-chunk-size"] = "prefill_chunk_size", ["--ubatch-size"] = "prefill_chunk_size",
        ["--mtp"] = "mtp", ["--no-mtp"] = "mtp",
        ["--speculative-k"] = "speculative_k", ["--draft-tokens"] = "speculative_k",
        ["--threads"] = "threads",
        ["--decode-threads"] = "decode_threads",
        ["--expected-concurrency"] = "expected_concurrency",
        ["--ctx-size"] = "ctx_size", ["-c"] = "ctx_size", ["--ctx"] = "ctx_size",
    };

    /// <summary>The canonical setting names given on the command line (synonyms normalised; <c>--flag=value</c> handled; <c>--auto-pull</c> is not one).</summary>
    /// <param name="rawArgs">The raw command-line arguments.</param>
    public static IReadOnlySet<string> ExplicitFlags(IReadOnlyList<string> rawArgs)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (string a in rawArgs)
        {
            if (a.Length < 2 || a[0] != '-') continue;
            int eq = a.IndexOf('=');
            string flag = eq > 0 ? a[..eq] : a;
            if (s_flags.TryGetValue(flag, out string? name)) set.Add(name);
        }
        return set;
    }

    /// <summary>True for a <c>tq</c>, <c>tq1..tq8</c> or <c>turboquant</c> KV format.</summary>
    public static bool IsTurboQuant(string? kv) =>
        kv is not null && (kv.StartsWith("tq", StringComparison.OrdinalIgnoreCase) || kv.StartsWith("turboquant", StringComparison.OrdinalIgnoreCase));

    /// <summary>Applies <paramref name="match"/> under the rules; see the type remarks.</summary>
    /// <param name="explicitFlags">Settings given as flags (from <see cref="ExplicitFlags"/>).</param>
    /// <param name="match">The store match.</param>
    /// <param name="tuneCommand">The <c>dotllm tune ...</c> command to print.</param>
    /// <param name="servedContext">The context being served: the flag's value when <c>ctx_size</c> was given, else the command's default.</param>
    /// <param name="admit">Re-admission of the saved placement against free memory.</param>
    /// <param name="refuse">Whether this build refuses the saved KV group.</param>
    /// <param name="policy">The command, for the TurboQuant constraint.</param>
    /// <param name="supported">Settings the command can take; others are ignored. Null means all.</param>
    public static AutoApplied Apply(IReadOnlySet<string> explicitFlags, TuningMatch match, string tuneCommand, int servedContext,
                                    PlacementAdmission admit, SettingRefusal refuse, AutoPolicy policy, IReadOnlySet<string>? supported = null)
    {
        var sources = Names.ToDictionary(n => n, n => explicitFlags.Contains(n) ? SettingSource.Flag : SettingSource.Default);
        var warnings = new List<string>();

        if (match.Kind == TuningMatchKind.Miss || match.Result?.Settings is null)
            return new(new TunedSettings(), sources, match, warnings,
                $"no tuning result for this model/device/profile ({match.Reason}); run: {tuneCommand}");

        string? advice = match.Kind switch
        {
            TuningMatchKind.Stale => $"saved result is stale ({match.Reason}); re-measure: {tuneCommand} --recheck",
            TuningMatchKind.Partial => $"saved result is incomplete ({match.Reason}); finish: {tuneCommand}",
            _ => null,
        };

        bool Free(string name) => !explicitFlags.Contains(name) && (supported is null || supported.Contains(name));
        var s = match.Result.Settings;
        var c = new TunedSettings
        {
            Device = Free("device") ? s.Device : null,
            GpuLayers = Free("gpu_layers") ? s.GpuLayers : null,
            MoeOffload = Free("moe_offload") ? s.MoeOffload : null,
            NCpuMoe = Free("moe_offload") ? s.NCpuMoe : null,
            MoeCacheMib = Free("moe_offload") ? s.MoeCacheMib : null,
            CacheTypeK = Free("cache_type_k") ? s.CacheTypeK : null,
            CacheTypeV = Free("cache_type_v") ? s.CacheTypeV : null,
            Paged = Free("paged") ? s.Paged : null,
            PrefillChunkSize = Free("prefill_chunk_size") ? s.PrefillChunkSize : null,
            MtpEnabled = Free("mtp") ? s.MtpEnabled : null,
            SpeculativeK = Free("speculative_k") ? s.SpeculativeK : null,
            Threads = Free("threads") ? s.Threads : null,
            DecodeThreads = Free("decode_threads") ? s.DecodeThreads : null,
            ExpectedConcurrency = Free("expected_concurrency") ? s.ExpectedConcurrency : null,
            ContextSize = Free("ctx_size") ? s.ContextSize : null,
            TunedContext = s.TunedContext,
            PeakDeviceBytes = s.PeakDeviceBytes,
            PeakHostBytes = s.PeakHostBytes,
        };

        // A TurboQuant pair is applied whole or not at all (a mixed tq pair is refused, #1895).
        bool savedTq = IsTurboQuant(s.CacheTypeK) || IsTurboQuant(s.CacheTypeV);
        if (savedTq && (explicitFlags.Contains("cache_type_k") || explicitFlags.Contains("cache_type_v")))
        {
            c = DropKv(c);
            warnings.Add($"saved TurboQuant KV ({s.CacheTypeK}/{s.CacheTypeV}) dropped: a flag set one side, and a TurboQuant pair must be applied whole");
        }
        // TurboQuant needs --no-paged, which only run, chat, bench and a single-user serve may use (#1893 lifts this for serve).
        else if (savedTq && policy.Command == AutoCommand.Serve && !policy.SingleUser)
        {
            c = DropKv(c);
            warnings.Add("saved TurboQuant KV dropped: it needs --no-paged, which is only for run, chat, bench or a single-user serve");
        }

        if (c.CacheTypeK is not null || c.CacheTypeV is not null || c.Paged is not null)
            if (refuse(c) is { } reason)
            {
                c = DropKv(c);
                warnings.Add($"saved KV settings dropped: {reason}");
            }

        int context = explicitFlags.Contains("ctx_size") ? servedContext : c.ContextSize ?? servedContext;
        if (Present(c).Any(p => p.Value))
        {
            // The recorded peaks bound memory only up to the context they were measured at (no spill: never admit an unbounded peak).
            string? refused = c.TunedContext is not int tuned || tuned <= 0
                ? "the saved result does not record the context its peak memory was measured at"
                : context > tuned
                    ? $"serving context {context} is above the tuned context {tuned}, so the recorded peak memory does not bound it"
                    : admit(c, context);
            if (refused is not null)
            {
                c = new TunedSettings();
                warnings.Add($"tuned placement not admitted ({refused}); running the defaults");
            }
        }

        foreach (var (name, value) in Present(c)) if (value) sources[name] = SettingSource.Auto;
        return new(c, sources, match, warnings, advice);
    }

    private static TunedSettings DropKv(TunedSettings c) => c with { CacheTypeK = null, CacheTypeV = null, Paged = null };

    private static IEnumerable<(string Name, bool Value)> Present(TunedSettings c)
    {
        yield return ("device", c.Device is not null);
        yield return ("gpu_layers", c.GpuLayers is not null);
        yield return ("moe_offload", c.MoeOffload is not null || c.NCpuMoe is not null || c.MoeCacheMib is not null);
        yield return ("cache_type_k", c.CacheTypeK is not null);
        yield return ("cache_type_v", c.CacheTypeV is not null);
        yield return ("paged", c.Paged is not null);
        yield return ("prefill_chunk_size", c.PrefillChunkSize is not null);
        yield return ("mtp", c.MtpEnabled is not null);
        yield return ("speculative_k", c.SpeculativeK is not null);
        yield return ("threads", c.Threads is not null);
        yield return ("decode_threads", c.DecodeThreads is not null);
        yield return ("expected_concurrency", c.ExpectedConcurrency is not null);
        yield return ("ctx_size", c.ContextSize is not null);
    }

    /// <summary>Bytes as GiB with two decimals.</summary>
    internal static string Format(long bytes) => $"{bytes / (1024.0 * 1024 * 1024):F2} GiB";
}
