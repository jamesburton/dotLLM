using System.Security.Cryptography;
using System.Text.Json;

namespace DotLLM.HuggingFace;

/// <summary>An ollama model reference: <c>[host/][namespace/]name[:tag]</c>, e.g. <c>llama3.2:3b</c> or <c>ollama:user/model:q4</c>.</summary>
/// <param name="Name">Model name; <c>library/</c> is implicit and omitted (<c>llama3.2</c>, <c>user/model</c>).</param>
/// <param name="Tag">Tag; <c>latest</c> when none was given.</param>
/// <param name="Explicit">True when the reference carried an <c>ollama:</c> / registry prefix (it can only mean an ollama model).</param>
public readonly record struct OllamaRef(string Name, string Tag, bool Explicit)
{
    /// <summary>Canonical display form: <c>name:tag</c>.</summary>
    public override string ToString() => $"{Name}:{Tag}";

    /// <summary>
    /// Parses <c>ollama:name[:tag]</c>, <c>registry.ollama.ai/library/name:tag</c>, <c>ollama.com/library/name:tag</c> (explicit) or a bare
    /// <c>name[:tag]</c> with no path separators (implicit - only meaningful after local lookups missed). Returns null for paths, repo ids and
    /// anything else that cannot be an ollama name.
    /// </summary>
    public static OllamaRef? TryParse(string? arg)
    {
        if (string.IsNullOrWhiteSpace(arg)) return null;
        string s = arg.Trim();
        bool explicitRef = false;
        foreach (string prefix in new[] { "ollama:", "ollama://", "https://registry.ollama.ai/", "registry.ollama.ai/", "ollama.com/", "https://ollama.com/" })
            if (s.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) { s = s[prefix.Length..]; explicitRef = true; break; }
        if (s.StartsWith("library/", StringComparison.OrdinalIgnoreCase)) s = s["library/".Length..];

        string tag = "latest";
        int colon = s.LastIndexOf(':');
        if (colon > 0 && !s[(colon + 1)..].Contains('/')) { tag = s[(colon + 1)..]; s = s[..colon]; }
        if (s.Length == 0 || tag.Length == 0) return null;
        if (s.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || Path.IsPathRooted(s) || s.Contains('\\')) return null;
        if (!explicitRef && s.Contains('/')) return null;      // owner/repo is a Hugging Face reference; namespaced ollama models need the prefix
        if (s.Split('/').Length > 2) return null;
        foreach (char c in s + tag)
            if (!(char.IsAsciiLetterOrDigit(c) || c is '.' or '_' or '-' or '/')) return null;
        return new OllamaRef(s.ToLowerInvariant(), tag.ToLowerInvariant(), explicitRef);
    }
}

/// <summary>A model in an ollama store: where its GGUF blob is, and the system prompt / parameters its manifest carries.</summary>
public sealed record OllamaModel(
    OllamaRef Ref, string BlobPath, long SizeBytes, string? System, IReadOnlyDictionary<string, JsonElement>? Params, DateTimeOffset ModifiedAt);

/// <summary>Read-only access to an existing ollama installation's model store (<c>OLLAMA_MODELS</c> or <c>~/.ollama/models</c>).</summary>
public static class OllamaStore
{
    private const string ModelLayer = "application/vnd.ollama.image.model";
    private const string SystemLayer = "application/vnd.ollama.image.system";
    private const string ParamsLayer = "application/vnd.ollama.image.params";

    /// <summary>The ollama models directory: <c>OLLAMA_MODELS</c>, else <c>~/.ollama/models</c>.</summary>
    public static string DefaultRoot
    {
        get
        {
            string? env = Environment.GetEnvironmentVariable("OLLAMA_MODELS");
            if (!string.IsNullOrWhiteSpace(env)) return env;
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        }
    }

    /// <summary>Blob file for a <c>sha256:hex</c> digest.</summary>
    public static string BlobPathFor(string digest, string? root = null) =>
        Path.Combine(root ?? DefaultRoot, "blobs", digest.Replace(':', '-'));

    /// <summary>True when <paramref name="path"/> is a blob inside the given (or default) ollama store.</summary>
    public static bool IsBlobPath(string path, string? root = null)
    {
        string blobs = Path.GetFullPath(Path.Combine(root ?? DefaultRoot, "blobs")) + Path.DirectorySeparatorChar;
        return Path.GetFullPath(path).StartsWith(blobs, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Every model with a manifest and a present model blob. Never throws.</summary>
    public static List<OllamaModel> ListAll(string? root = null)
    {
        root ??= DefaultRoot;
        var result = new List<OllamaModel>();
        string manifests = Path.Combine(root, "manifests");
        if (!Directory.Exists(manifests)) return result;
        try
        {
            foreach (string file in Directory.EnumerateFiles(manifests, "*", SearchOption.AllDirectories))
            {
                string[] parts = Path.GetRelativePath(manifests, file).Split(Path.DirectorySeparatorChar);
                if (parts.Length < 4) continue;   // host / namespace / model / tag
                string host = parts[0], ns = parts[1], model = string.Join('/', parts[2..^1]), tag = parts[^1];
                string name = ns.Equals("library", StringComparison.OrdinalIgnoreCase) ? model : $"{ns}/{model}";
                if (!host.Equals("registry.ollama.ai", StringComparison.OrdinalIgnoreCase)) name = $"{host}/{name}";
                if (ReadManifest(file, root, new OllamaRef(name.ToLowerInvariant(), tag.ToLowerInvariant(), false)) is { } m) result.Add(m);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* unreadable store: report what was read */ }
        return result.OrderBy(m => m.Ref.Name, StringComparer.Ordinal).ThenBy(m => m.Ref.Tag, StringComparer.Ordinal).ToList();
    }

    /// <summary>Finds one model by reference, or null.</summary>
    public static OllamaModel? TryFind(OllamaRef reference, string? root = null)
    {
        root ??= DefaultRoot;
        string rel = reference.Name.Contains('/') ? reference.Name : "library/" + reference.Name;
        string file = Path.Combine(root, "manifests", "registry.ollama.ai", rel.Replace('/', Path.DirectorySeparatorChar), reference.Tag);
        return File.Exists(file) ? ReadManifest(file, root, reference) : null;
    }

    private static OllamaModel? ReadManifest(string manifestFile, string root, OllamaRef reference)
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(manifestFile));
            if (!doc.RootElement.TryGetProperty("layers", out var layers)) return null;
            string? modelDigest = null; long size = 0; string? system = null; Dictionary<string, JsonElement>? prm = null;
            foreach (var layer in layers.EnumerateArray())
            {
                string type = layer.GetProperty("mediaType").GetString() ?? "";
                string digest = layer.GetProperty("digest").GetString() ?? "";
                if (type == ModelLayer) { modelDigest = digest; size = layer.GetProperty("size").GetInt64(); }
                else if (type == SystemLayer) system = TryReadBlobText(root, digest);
                else if (type == ParamsLayer && TryReadBlobText(root, digest) is { } json)
                {
                    using var pd = JsonDocument.Parse(json);
                    prm = pd.RootElement.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.Clone());
                }
            }
            if (modelDigest is null) return null;
            string blob = BlobPathFor(modelDigest, root);
            return File.Exists(blob) ? new OllamaModel(reference, blob, size, system, prm, File.GetLastWriteTimeUtc(manifestFile)) : null;
        }
        catch (Exception ex) when (ex is IOException or JsonException or KeyNotFoundException or InvalidOperationException) { return null; }
    }

    private static string? TryReadBlobText(string root, string digest)
    {
        try { string p = BlobPathFor(digest, root); return File.Exists(p) && new FileInfo(p).Length < 1 << 20 ? File.ReadAllText(p) : null; }
        catch (IOException) { return null; }
    }

    /// <summary>Maps an ollama <c>params</c> layer onto a <see cref="ModelProfile"/> (unknown keys such as num_ctx are ignored).</summary>
    public static ModelProfile ToProfile(OllamaModel m, string from)
    {
        float? F(string k) => m.Params is not null && m.Params.TryGetValue(k, out var v) && v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : null;
        int? I(string k) => m.Params is not null && m.Params.TryGetValue(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
        string[]? stop = m.Params is not null && m.Params.TryGetValue("stop", out var s) && s.ValueKind == JsonValueKind.Array
            ? s.EnumerateArray().Select(e => e.GetString() ?? "").Where(x => x.Length > 0).ToArray() : null;
        return new ModelProfile
        {
            From = from, Description = $"imported from ollama ({m.Ref})", System = m.System,
            Temperature = F("temperature"), TopP = F("top_p"), TopK = I("top_k"), MinP = F("min_p"), RepeatPenalty = F("repeat_penalty"),
            MaxTokens = I("num_predict") is > 0 and var np ? np : null, Seed = I("seed"), Stop = stop is { Length: > 0 } ? stop : null,
        };
    }
}

/// <summary>Pulls models from the ollama registry (<c>registry.ollama.ai</c>) into dotLLM's own model store.</summary>
public sealed class OllamaRegistry : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsClient;
    private readonly string _baseUrl;

    /// <summary>Creates a client. <paramref name="baseUrl"/> exists for tests; the default is the public registry.</summary>
    public OllamaRegistry(HttpClient? http = null, string baseUrl = "https://registry.ollama.ai")
    {
        _ownsClient = http is null;
        _http = http ?? new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        _baseUrl = baseUrl.TrimEnd('/');
        if (!_http.DefaultRequestHeaders.UserAgent.Any()) _http.DefaultRequestHeaders.UserAgent.ParseAdd("dotLLM/0.1");
    }

    /// <summary>Where a pulled model's GGUF lives in the model store: <c>models/ollama/{name}/{name}-{tag}.gguf</c>.</summary>
    public static string TargetPath(OllamaRef r, string? modelsDir = null) =>
        Path.Combine(modelsDir ?? HuggingFaceDownloader.DefaultModelsDirectory, "ollama", r.Name.Replace('/', '_'), $"{r.Name.Replace('/', '_')}-{r.Tag}.gguf");

    /// <summary>
    /// Downloads the model layer (resumable via <c>.incomplete</c>, sha256-verified before it is moved into place) and returns the GGUF path plus a
    /// profile carrying the manifest's system prompt and parameters. The ollama prompt template is NOT carried over: it is a Go template, and
    /// dotLLM renders the chat template embedded in the GGUF.
    /// </summary>
    public async Task<(string Path, ModelProfile Profile)> PullAsync(
        OllamaRef r, IProgress<(long bytesDownloaded, long? totalBytes)>? progress, CancellationToken ct, string? modelsDir = null)
    {
        string repo = r.Name.Contains('/') ? r.Name : "library/" + r.Name;
        using var req = new HttpRequestMessage(HttpMethod.Get, $"{_baseUrl}/v2/{repo}/manifests/{r.Tag}");
        req.Headers.Accept.ParseAdd("application/vnd.docker.distribution.manifest.v2+json");
        using var resp = await _http.SendAsync(req, ct).ConfigureAwait(false);
        if (resp.StatusCode == System.Net.HttpStatusCode.NotFound)
            throw new InvalidOperationException($"'{r}' was not found on the ollama registry.");
        resp.EnsureSuccessStatusCode();

        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false));
        string? modelDigest = null; long modelSize = 0; string? systemDigest = null, paramsDigest = null;
        foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
        {
            string type = layer.GetProperty("mediaType").GetString() ?? "", digest = layer.GetProperty("digest").GetString() ?? "";
            if (type == "application/vnd.ollama.image.model") { modelDigest = digest; modelSize = layer.GetProperty("size").GetInt64(); }
            else if (type == "application/vnd.ollama.image.system") systemDigest = digest;
            else if (type == "application/vnd.ollama.image.params") paramsDigest = digest;
        }
        if (modelDigest is null) throw new InvalidOperationException($"'{r}' has no model layer (not a GGUF model).");

        string target = TargetPath(r, modelsDir);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        await DownloadBlobAsync(repo, modelDigest, modelSize, target, progress, ct).ConfigureAwait(false);

        string? system = systemDigest is null ? null : await GetTextBlobAsync(repo, systemDigest, ct).ConfigureAwait(false);
        string? prm = paramsDigest is null ? null : await GetTextBlobAsync(repo, paramsDigest, ct).ConfigureAwait(false);
        var model = new OllamaModel(r, target, modelSize, system,
            prm is null ? null : JsonDocument.Parse(prm).RootElement.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.Clone()), DateTimeOffset.UtcNow);
        return (target, OllamaStore.ToProfile(model, target));
    }

    private async Task<string> GetTextBlobAsync(string repo, string digest, CancellationToken ct)
    {
        using var resp = await _http.GetAsync($"{_baseUrl}/v2/{repo}/blobs/{digest}", ct).ConfigureAwait(false);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
    }

    private async Task DownloadBlobAsync(string repo, string digest, long size, string target, IProgress<(long, long?)>? progress, CancellationToken ct)
    {
        string expected = digest.StartsWith("sha256:", StringComparison.Ordinal) ? digest["sha256:".Length..] : throw new InvalidOperationException($"Unsupported digest '{digest}'.");
        if (File.Exists(target) && new FileInfo(target).Length == size && await HashFileAsync(target, ct).ConfigureAwait(false) == expected) return;   // already pulled

        string part = target + ".incomplete";
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        long have = 0;
        if (File.Exists(part))
        {
            have = new FileInfo(part).Length;
            if (have > size) { File.Delete(part); have = 0; }
            else
            {
                // Re-hash the part we already have so the final digest covers the whole file.
                await using var existing = File.OpenRead(part);
                var buf = new byte[1 << 20]; int n;
                while ((n = await existing.ReadAsync(buf, ct).ConfigureAwait(false)) > 0) hash.AppendData(buf, 0, n);
            }
        }

        using var req = new HttpRequestMessage(HttpMethod.Get, $"{_baseUrl}/v2/{repo}/blobs/{digest}");
        if (have > 0) req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(have, null);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        if (have > 0 && resp.StatusCode != System.Net.HttpStatusCode.PartialContent)
        {
            // The server ignored the Range: start over rather than append a full body to a partial file.
            have = 0; hash.GetHashAndReset(); File.Delete(part);
        }
        resp.EnsureSuccessStatusCode();

        await using (var src = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
        await using (var dst = new FileStream(part, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write))
        {
            var buf = new byte[1 << 20]; int n;
            while ((n = await src.ReadAsync(buf, ct).ConfigureAwait(false)) > 0)
            {
                await dst.WriteAsync(buf.AsMemory(0, n), ct).ConfigureAwait(false);
                hash.AppendData(buf, 0, n);
                have += n;
                progress?.Report((have, size));
            }
        }

        string actual = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
        if (actual != expected)
        {
            File.Delete(part);   // a corrupt partial must not be resumed
            throw new InvalidOperationException($"sha256 mismatch for {digest}: got {actual}. The partial download was discarded.");
        }
        File.Move(part, target, overwrite: true);
    }

    private static async Task<string> HashFileAsync(string path, CancellationToken ct)
    {
        await using var fs = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(fs, ct).ConfigureAwait(false)).ToLowerInvariant();
    }

    /// <inheritdoc/>
    public void Dispose() { if (_ownsClient) _http.Dispose(); }
}
