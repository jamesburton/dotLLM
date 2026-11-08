using System.Net;
using System.Text.Json;
using DotLLM.ChunkCache.Client;

namespace DotLLM.ChunkCache;

/// <summary>What <c>GET /v1/chunk-cache</c> says about the server's model and store (the fields the CLI uses).</summary>
/// <param name="ModelId">Content model id (<c>model.id</c>); names the index file.</param>
/// <param name="Producer">Producer identity (<c>model.producer</c>, 64 hex): the key of no tokens; its first 8 digits name the index file.</param>
/// <param name="TokenizerId"><c>model.tokenizer_id</c>.</param>
/// <param name="Supported">Whether chunk reuse supports the model.</param>
/// <param name="Entries">Store entries of the caller's namespace.</param>
/// <param name="Bytes">Their bytes.</param>
/// <param name="BudgetBytes">The disk budget (0 = none).</param>
/// <param name="MaxChunksPerRequest">The most spans one request may carry.</param>
internal sealed record ServerIdentity(string? ModelId, string? Producer, string? TokenizerId, bool? Supported, long? Entries, long? Bytes,
    long? BudgetBytes, int? MaxChunksPerRequest);

/// <summary>
/// The connection of a server-backed run: <see cref="ChunkCacheHttpClient"/> for prewarm / evict / explain, and one capabilities
/// read for the server's model identity. The API key comes from <see cref="ApiKeyVariable"/> and nowhere else; it is never
/// printed (messages go through <see cref="CliContext.Redact"/>) and the base URL is reduced to scheme, host, port and path.
/// </summary>
internal sealed class ServerConnection : IDisposable
{
    /// <summary>The only source of the API key (contract ruling 18): never a command-line argument.</summary>
    public const string ApiKeyVariable = "CHUNKCACHE_API_KEY";

    private readonly HttpClient _http;
    private readonly string _root;
    private readonly string? _apiKey;

    private ServerConnection(HttpClient http, ChunkCacheHttpClient client, string baseUrl, string root, string? apiKey)
    {
        _http = http;
        Client = client;
        BaseUrl = baseUrl;
        _root = root;
        _apiKey = apiKey;
    }

    /// <summary>The client for prewarm, evict and explain.</summary>
    public ChunkCacheHttpClient Client { get; }

    /// <summary>What the server said about its model and store (set by <see cref="Common.ConnectServer"/>).</summary>
    public ServerIdentity? Identity { get; set; }

    /// <summary>The base URL without user info, query or fragment.</summary>
    public string BaseUrl { get; }

    /// <summary>Opens a connection to <paramref name="url"/> (http or https).</summary>
    public static ServerConnection Open(CliContext ctx, string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) || uri.Scheme is not ("http" or "https") || string.IsNullOrEmpty(uri.Host))
            throw new UsageException("--server must be an absolute http:// or https:// URL.");
        string? key = ctx.GetEnv(ApiKeyVariable);
        if (string.IsNullOrEmpty(key)) key = null;
        if (key is not null && key.Any(char.IsControl))
            throw new UsageException($"{ApiKeyVariable} contains a control character; it is not used.");
        string b = uri.GetComponents(UriComponents.SchemeAndServer | UriComponents.Path, UriFormat.UriEscaped).TrimEnd('/');
        if (b.EndsWith("/v1", StringComparison.OrdinalIgnoreCase)) b = b[..^3];
        if (key is not null && uri.Scheme == "http" && !uri.IsLoopback)
            ctx.Info($"warning: {ApiKeyVariable} is sent over plain http to {uri.Host}; use https unless the network is trusted.");
        HttpClient http = ctx.HttpHandler is { } h ? new HttpClient(h, disposeHandler: false) : new HttpClient();
        http.Timeout = TimeSpan.FromMinutes(10);
        var client = new ChunkCacheHttpClient(http, new Uri(b + "/"), key);
        return new ServerConnection(http, client, b, b + "/v1/chunk-cache", key);
    }

    /// <summary>Reads <c>GET /v1/chunk-cache</c>: the server's model identity and store figures.</summary>
    public async Task<ServerIdentity> GetIdentityAsync(CancellationToken ct = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, _root);
        if (_apiKey is not null) request.Headers.TryAddWithoutValidation("X-API-Key", _apiKey);
        using HttpResponseMessage response = await _http.SendAsync(request, ct).ConfigureAwait(false);
        string body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.NotFound)
            throw new InvalidOperationException($"{BaseUrl} does not serve the chunk-cache convention (GET /v1/chunk-cache is 404); start it with --chunk-cache.");
        if (!response.IsSuccessStatusCode)
            throw new ChunkCacheRequestException(response.StatusCode, null, null, $"GET {_root} failed with {(int)response.StatusCode} {response.StatusCode}.");
        try
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            JsonElement root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) throw new JsonException("not an object");
            JsonElement model = root.TryGetProperty("model", out JsonElement m) && m.ValueKind == JsonValueKind.Object ? m : default;
            JsonElement store = root.TryGetProperty("store", out JsonElement s) && s.ValueKind == JsonValueKind.Object ? s : default;
            return new ServerIdentity(Str(model, "id"), Str(model, "producer"), Str(model, "tokenizer_id"),
                model.ValueKind == JsonValueKind.Object && model.TryGetProperty("supported", out JsonElement sup) && sup.ValueKind is JsonValueKind.True or JsonValueKind.False ? sup.GetBoolean() : null,
                Long(store, "entries"), Long(store, "bytes"), Long(store, "budget_bytes"),
                Long(root, "max_chunks_per_request") is long mc and > 0 and <= int.MaxValue ? (int)mc : null);
        }
        catch (JsonException)
        {
            throw new InvalidOperationException($"{BaseUrl} answered GET /v1/chunk-cache with a body that is not the chunk-cache capabilities JSON.");
        }
    }

    /// <inheritdoc/>
    public void Dispose() => _http.Dispose();

    private static string? Str(JsonElement e, string name) =>
        e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static long? Long(JsonElement e, string name) =>
        e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt64(out long l) ? l : null;
}
