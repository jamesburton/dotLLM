using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.HuggingFace;

/// <summary>
/// A named, persisted model configuration - dotLLM's Modelfile equivalent (issue #716): a base model reference plus the defaults to apply when
/// that name is requested. Stored as <c>~/.dotllm/profiles/{name}.json</c>; every field except <see cref="From"/> is optional.
/// </summary>
/// <remarks>
/// All members are nullable with no initialisers: this type is deserialised through a source-generated context, which silently drops property
/// initialisers on <c>init</c> members (see the JSON DTO rules in CLAUDE.md).
/// </remarks>
public sealed record ModelProfile
{
    /// <summary>The base model: any reference <see cref="ModelResolver"/> accepts (path, <c>owner/repo[:tag]</c>, local stem), or another profile name.</summary>
    [JsonPropertyName("from")] public string? From { get; init; }

    /// <summary>Free text shown by <c>model show</c>.</summary>
    [JsonPropertyName("description")] public string? Description { get; init; }

    /// <summary>System prompt prepended when a chat request carries none.</summary>
    [JsonPropertyName("system")] public string? System { get; init; }

    /// <summary>Sampling temperature used when a request omits it.</summary>
    [JsonPropertyName("temperature")] public float? Temperature { get; init; }

    /// <summary>Top-P used when a request omits it.</summary>
    [JsonPropertyName("top_p")] public float? TopP { get; init; }

    /// <summary>Top-K used when a request omits it.</summary>
    [JsonPropertyName("top_k")] public int? TopK { get; init; }

    /// <summary>Min-P used when a request omits it.</summary>
    [JsonPropertyName("min_p")] public float? MinP { get; init; }

    /// <summary>Repetition penalty used when a request omits it.</summary>
    [JsonPropertyName("repeat_penalty")] public float? RepeatPenalty { get; init; }

    /// <summary>Maximum generated tokens used when a request omits it.</summary>
    [JsonPropertyName("max_tokens")] public int? MaxTokens { get; init; }

    /// <summary>Seed used when a request omits it.</summary>
    [JsonPropertyName("seed")] public int? Seed { get; init; }

    /// <summary>Stop sequences always applied (in addition to the request's).</summary>
    [JsonPropertyName("stop")] public string[]? Stop { get; init; }

    /// <summary>Device this model loads on (<c>cpu</c>, <c>vulkan</c>, <c>gpu:0</c>); the server default otherwise.</summary>
    [JsonPropertyName("device")] public string? Device { get; init; }

    /// <summary>GPU layers for a partial offload.</summary>
    [JsonPropertyName("gpu_layers")] public int? GpuLayers { get; init; }

    /// <summary>Keep-alive seconds for this model (0 = unload after each request, negative = never).</summary>
    [JsonPropertyName("keep_alive")] public double? KeepAlive { get; init; }
}

[JsonSourceGenerationOptions(WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ModelProfile))]
internal partial class ModelProfileJsonContext : JsonSerializerContext;

/// <summary>Reads and writes <see cref="ModelProfile"/> files. Names are case-insensitive (stored lower-case); <c>name:latest</c> is the same as <c>name</c>.</summary>
public static class ModelProfileStore
{
    /// <summary>Profiles directory: <c>DOTLLM_PROFILES_DIR</c>, else <c>~/.dotllm/profiles</c>.</summary>
    public static string DefaultDirectory
    {
        get
        {
            string? env = Environment.GetEnvironmentVariable("DOTLLM_PROFILES_DIR");
            if (!string.IsNullOrWhiteSpace(env)) return env;
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "profiles");
        }
    }

    /// <summary>Normalises a name (trims, drops a trailing <c>:latest</c>) or returns null when it is not a legal profile name.</summary>
    public static string? NormalizeName(string? name)
    {
        if (string.IsNullOrWhiteSpace(name)) return null;
        string n = name.Trim();
        if (n.EndsWith(":latest", StringComparison.OrdinalIgnoreCase)) n = n[..^":latest".Length];
        if (n.Length is 0 or > 80) return null;
        // letters, digits, '.', '_', '-', and one ':' for an ollama-style tag. No separators or reserved names: a profile name is also a file name.
        foreach (char c in n)
            if (!(char.IsAsciiLetterOrDigit(c) || c is '.' or '_' or '-' or ':')) return null;
        if (n.Count(c => c == ':') > 1 || n[0] is '.' or ':' or '-' || n[^1] is '.' or ':') return null;
        // Canonical form is lower-case: names are case-insensitive, and a file name must be too on case-sensitive file systems.
        return n.ToLowerInvariant();
    }

    private static string FilePath(string name, string? dir) =>
        Path.Combine(dir ?? DefaultDirectory, name.Replace(':', '~') + ".json");

    /// <summary>Loads a profile, or null when there is none (or the file is unreadable / has no <c>from</c>).</summary>
    public static ModelProfile? TryGet(string? name, string? dir = null)
    {
        string? n = NormalizeName(name);
        if (n is null) return null;
        string path = FilePath(n, dir);
        try
        {
            if (!File.Exists(path)) return null;
            var p = JsonSerializer.Deserialize(File.ReadAllText(path), ModelProfileJsonContext.Default.ModelProfile);
            return p is { From.Length: > 0 } ? p : null;
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException) { return null; }
    }

    /// <summary>Writes (or replaces) a profile. Throws <see cref="ArgumentException"/> for an illegal name or a missing <c>from</c>.</summary>
    public static void Save(string name, ModelProfile profile, string? dir = null)
    {
        string n = NormalizeName(name) ?? throw new ArgumentException($"'{name}' is not a valid profile name (letters, digits, . _ - and one ':tag').", nameof(name));
        if (string.IsNullOrWhiteSpace(profile.From)) throw new ArgumentException("A profile needs a 'from' base model.", nameof(profile));
        string path = FilePath(n, dir);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(profile, ModelProfileJsonContext.Default.ModelProfile));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Removes a profile. Returns false when it did not exist.</summary>
    public static bool Delete(string name, string? dir = null)
    {
        string? n = NormalizeName(name);
        if (n is null) return false;
        string path = FilePath(n, dir);
        if (!File.Exists(path)) return false;
        File.Delete(path);
        return true;
    }

    /// <summary>All profiles, by name.</summary>
    public static List<(string Name, ModelProfile Profile)> List(string? dir = null)
    {
        var result = new List<(string, ModelProfile)>();
        dir ??= DefaultDirectory;
        if (!Directory.Exists(dir)) return result;
        foreach (string f in Directory.GetFiles(dir, "*.json"))
        {
            string name = Path.GetFileNameWithoutExtension(f).Replace('~', ':');
            if (TryGet(name, dir) is { } p) result.Add((name, p));
        }
        return result.OrderBy(x => x.Item1, StringComparer.OrdinalIgnoreCase).ToList();
    }

    /// <summary>Flattens a profile chain (outermost first): each field comes from the first profile that sets it.</summary>
    public static ModelProfile Merge(IReadOnlyList<ModelProfile> chain)
    {
        T? First<T>(Func<ModelProfile, T?> pick) where T : class
        { foreach (var p in chain) if (pick(p) is { } v) return v; return null; }
        T? FirstV<T>(Func<ModelProfile, T?> pick) where T : struct
        { foreach (var p in chain) if (pick(p) is { } v) return v; return null; }
        return new ModelProfile
        {
            From = chain[^1].From,
            Description = First(p => p.Description), System = First(p => p.System), Stop = First(p => p.Stop), Device = First(p => p.Device),
            Temperature = FirstV(p => p.Temperature), TopP = FirstV(p => p.TopP), TopK = FirstV(p => p.TopK), MinP = FirstV(p => p.MinP),
            RepeatPenalty = FirstV(p => p.RepeatPenalty), MaxTokens = FirstV(p => p.MaxTokens), Seed = FirstV(p => p.Seed),
            GpuLayers = FirstV(p => p.GpuLayers), KeepAlive = FirstV(p => p.KeepAlive),
        };
    }

    /// <summary>
    /// Resolves a name through profiles to the underlying non-profile reference and the chain of profiles crossed (outermost first), following
    /// <c>from</c> links. Returns null when <paramref name="name"/> is not a profile; cycles and chains deeper than 8 are cut and reported null.
    /// </summary>
    public static (string BaseReference, IReadOnlyList<ModelProfile> Chain)? Resolve(string? name, string? dir = null)
    {
        var chain = new List<ModelProfile>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string? current = name;
        while (TryGet(current, dir) is { } p)
        {
            if (chain.Count >= 8 || !seen.Add(NormalizeName(current)!)) return null;
            chain.Add(p);
            current = p.From;
        }
        return chain.Count == 0 ? null : (current!, chain);
    }
}
