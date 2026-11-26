using System.Collections;

namespace DotLLM.Server.Tuning;

/// <summary>
/// The set <c>DOTLLM_*</c> tuning and A/B variables that can move the optimum (#1902). A variable set at serve time but not at tune time
/// (for example <c>DOTLLM_CUDA_STEP_GRAPH=0</c>) makes a saved result stale.
/// </summary>
public static class EnvOverrides
{
    /// <summary>Variables that change where files live or what is logged, never performance.</summary>
    private static readonly HashSet<string> s_ignored = new(StringComparer.Ordinal)
    {
        "DOTLLM_TUNING_DIR", "DOTLLM_HASH_CACHE_DIR", "DOTLLM_MODELS_DIR", "DOTLLM_PROFILES_DIR",
        "DOTLLM_URL", "DOTLLM_NO_PROMPT", "DOTLLM_DIRECTSTORAGE_DIR", "DOTLLM_REGEN_CLI_DOCS", "DOTLLM_DEBUG_ERRORS",
        "DOTLLM_LOG_TIMINGS", "DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER",
    };

    private static readonly string[] s_ignoredSuffixes = ["_GGUF", "_TRACE", "_DEBUG", "_STATS", "_VERBOSE", "_LOG"];

    /// <summary>True for a <c>DOTLLM_*</c> name that is not a performance input (paths, fixtures, logging and trace switches).</summary>
    public static bool IsIgnored(string name) =>
        s_ignored.Contains(name) || name.StartsWith("DOTLLM_TEST_", StringComparison.Ordinal) || s_ignoredSuffixes.Any(s => name.EndsWith(s, StringComparison.Ordinal));

    /// <summary>The sorted <c>name=value</c> pairs and their hash; both null when no relevant variable is set.</summary>
    /// <param name="read">Source of variables; null reads the process environment. Injected by tests.</param>
    public static (string? Hash, IReadOnlyList<string>? Pairs) Current(Func<IEnumerable<KeyValuePair<string, string>>>? read = null)
    {
        IEnumerable<KeyValuePair<string, string>> all = read is not null ? read() : ProcessEnvironment();
        var pairs = all
            .Where(kv => kv.Key.StartsWith("DOTLLM_", StringComparison.Ordinal) && !string.IsNullOrEmpty(kv.Value) && !IsIgnored(kv.Key))
            .Select(kv => kv.Key + "=" + kv.Value)
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToList();
        return pairs.Count == 0 ? (null, null) : (ModelHash.Sha256Hex(string.Join('\n', pairs)), pairs);
    }

    private static IEnumerable<KeyValuePair<string, string>> ProcessEnvironment()
    {
        foreach (DictionaryEntry e in Environment.GetEnvironmentVariables())
            if (e.Key is string k && e.Value is string v) yield return new(k, v);
    }
}
