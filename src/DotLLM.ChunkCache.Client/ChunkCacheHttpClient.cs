// SPDX-License-Identifier: MIT
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// <see cref="IChunkCacheClient"/> over a dotLLM server's <c>/v1/chunk-cache</c> endpoints (spec 5.9, 7.2), plus local
/// index queries for <c>select</c> and <c>status</c>, which the server does not serve.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><c>GET /v1/chunk-cache</c>: <see langword="null"/> on 404, and on a 200 body that is not JSON or whose
/// <c>versions</c> lacks 1 (convention absent). Any other error status throws <see cref="ChunkCacheRequestException"/>
/// (a server that refuses the key is present, and the status helps diagnose it). <b>Either way the caller strips
/// annotations</b>: contract ruling 16 makes any exception or non-2xx from the probe mean "not capable". Use
/// <see cref="ChunkCacheProbe.ProbeAsync"/> for that fail-closed form.</item>
/// <item><c>POST /v1/chunk-cache/prewarm</c>: <c>{"texts": [{id, text, source?: {path}}], "max_tokens"?}</c>. Text only;
/// a <c>select</c> input is resolved locally with <c>include_text</c> and its hashes are never sent. <c>source.path</c> is
/// advisory (ruling 14): the server applies its deny-list when present but cannot rely on it.</item>
/// <item><c>POST /v1/chunk-cache/evict</c>: <c>{mode, path?, namespace?, dry_run}</c>, with <c>mode</c> and
/// <c>dry_run</c> always sent resolved (<c>orphans</c>, <c>true</c>).</item>
/// <item><c>POST /v1/chunk-cache/explain</c>: <c>{"request": &lt;the full request body&gt;}</c> (pinned by ruling 15).</item>
/// </list>
/// The API key, when given and non-empty, is sent as <c>X-API-Key</c>. User info in the base URL is dropped. Tool-level
/// <c>store</c> / <c>server</c> inputs are not sent. A 2xx body that is not the expected JSON throws
/// <see cref="ChunkCacheRequestException"/> too, so callers handle one exception type.
/// </remarks>
public sealed class ChunkCacheHttpClient : IChunkCacheClient
{
    private readonly HttpClient _http;
    private readonly string _root;
    private readonly string? _apiKey;
    private readonly IChunkIndexQueries _queries;

    /// <summary>Creates the client.</summary>
    /// <param name="http">The HTTP client (its handler pipeline is used as is; this class never disposes it).</param>
    /// <param name="baseUrl">The server base URL, with or without a trailing <c>/v1</c>.</param>
    /// <param name="apiKey">Sent as <c>X-API-Key</c> when not <see langword="null"/>.</param>
    /// <param name="queries">Local index queries for select / status; <see langword="null"/> = <see cref="CliChunkIndexQueries"/>.</param>
    public ChunkCacheHttpClient(HttpClient http, Uri baseUrl, string? apiKey, IChunkIndexQueries? queries = null)
    {
        ArgumentNullException.ThrowIfNull(http);
        ArgumentNullException.ThrowIfNull(baseUrl);
        if (!baseUrl.IsAbsoluteUri) throw new ArgumentException("The base URL must be absolute.", nameof(baseUrl));
        _http = http;
        _apiKey = string.IsNullOrEmpty(apiKey) ? null : apiKey;
        _queries = queries ?? new CliChunkIndexQueries();

        // Scheme, host, port and path only: user info never reaches request URLs or exception messages.
        string b = baseUrl.GetComponents(UriComponents.SchemeAndServer | UriComponents.Path, UriFormat.UriEscaped).TrimEnd('/');
        if (b.EndsWith("/v1", StringComparison.OrdinalIgnoreCase)) b = b[..^3];
        _root = b + "/v1/chunk-cache";
    }

    /// <inheritdoc />
    public async ValueTask<ChunkCacheCapabilities?> GetCapabilitiesAsync(CancellationToken ct)
    {
        using var request = NewRequest(HttpMethod.Get, _root, content: null);
        using HttpResponseMessage response = await _http.SendAsync(request, ct).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.NotFound)
            return null;
        string body = await ReadBodyAsync(response, ct).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw Failure("GET", _root, response.StatusCode, body);

        ChunkCacheCapabilities? caps;
        try
        {
            caps = JsonSerializer.Deserialize(body, ChunkCacheJsonContext.Default.ChunkCacheCapabilities);
        }
        catch (JsonException)
        {
            return null; // not this convention (e.g. an HTML page from a proxy)
        }

        return caps is not null && caps.SupportsVersion(1) ? caps : null;
    }

    /// <inheritdoc />
    public ValueTask<SelectResult> SelectAsync(SelectRequest request, CancellationToken ct) => _queries.SelectAsync(request, ct);

    /// <inheritdoc />
    public ValueTask<StatusResult> StatusAsync(StatusRequest request, CancellationToken ct) => _queries.StatusAsync(request, ct);

    /// <inheritdoc />
    /// <remarks>
    /// When no text remains to send (a <c>select</c> that matched nothing), no request is made and an all-zero
    /// <see cref="PrewarmResult"/> is returned; it is indistinguishable from a server reply with zero counts.
    /// </remarks>
    public async ValueTask<PrewarmResult> PrewarmAsync(PrewarmRequest request, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Texts is null && request.Select is null)
            throw new ArgumentException("Prewarm needs texts or a select input.", nameof(request));

        var texts = new List<PrewarmText>();
        foreach (PrewarmText t in request.Texts ?? [])
        {
            if (string.IsNullOrEmpty(t.Text))
                throw new ArgumentException($"Prewarm text '{t.Id}' has no text; prewarm takes text only, never a hash.", nameof(request));
            texts.Add(t);
        }

        if (request.Select is { } select)
        {
            SelectResult selected = await _queries.SelectAsync(
                select with { IncludeText = true, Store = select.Store ?? request.Store }, ct).ConfigureAwait(false);
            foreach (SelectedChunk c in selected.Chunks ?? [])
            {
                if (string.IsNullOrEmpty(c.Text))
                    throw new InvalidOperationException(
                        $"Selected chunk '{c.Id}' came back without its text; prewarm sends text only, never a hash.");
                texts.Add(new PrewarmText
                {
                    Id = c.Id,
                    Text = c.Text,
                    Source = c.Path is null ? null : new PrewarmSource { Path = c.Path },
                });
            }
        }

        if (texts.Count == 0)
            return new PrewarmResult();

        var body = new PrewarmBody { Texts = texts, MaxTokens = request.MaxTokens };
        return await PostAsync("prewarm", Serialize(body, ChunkCacheHttpJsonContext.Default.PrewarmBody),
            ChunkCacheJsonContext.Default.PrewarmResult, ct).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public ValueTask<EvictResult> EvictAsync(EvictRequest request, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(request);
        var body = new EvictBody
        {
            Mode = request.EffectiveMode,
            Path = request.Path,
            Namespace = request.Namespace,
            DryRun = request.EffectiveDryRun,
        };
        return PostAsync("evict", Serialize(body, ChunkCacheHttpJsonContext.Default.EvictBody), ChunkCacheJsonContext.Default.EvictResult, ct);
    }

    /// <inheritdoc />
    public ValueTask<ExplainResult> ExplainAsync(JsonObject requestBody, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(requestBody);
        var envelope = new JsonObject { ["request"] = requestBody.DeepClone() };
        return PostAsync("explain", envelope.ToJsonString(), ChunkCacheJsonContext.Default.ExplainResult, ct);
    }

    private static string Serialize<T>(T value, JsonTypeInfo<T> type) => JsonSerializer.Serialize(value, type);

    private async ValueTask<T> PostAsync<T>(string endpoint, string json, JsonTypeInfo<T> resultType, CancellationToken ct)
        where T : class
    {
        string url = _root + "/" + endpoint;
        using var request = NewRequest(HttpMethod.Post, url, new StringContent(json, Encoding.UTF8, "application/json"));
        using HttpResponseMessage response = await _http.SendAsync(request, ct).ConfigureAwait(false);
        string body = await ReadBodyAsync(response, ct).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw Failure("POST", url, response.StatusCode, body);

        T? result;
        try
        {
            result = JsonSerializer.Deserialize(body, resultType);
        }
        catch (JsonException ex)
        {
            throw new ChunkCacheRequestException(response.StatusCode, null, null,
                $"POST {url} returned a body that is not the expected JSON: {ex.Message}", ex);
        }

        return result ?? throw new ChunkCacheRequestException(response.StatusCode, null, null, $"POST {url} returned JSON null.");
    }

    private HttpRequestMessage NewRequest(HttpMethod method, string url, HttpContent? content)
    {
        var request = new HttpRequestMessage(method, url) { Content = content };
        if (_apiKey is not null)
            request.Headers.TryAddWithoutValidation("X-API-Key", _apiKey);
        return request;
    }

    private static async Task<string> ReadBodyAsync(HttpResponseMessage response, CancellationToken ct) =>
        await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

    private static ChunkCacheRequestException Failure(string method, string url, HttpStatusCode status, string body)
    {
        ErrorBody? error = null;
        try
        {
            error = JsonSerializer.Deserialize(body, ChunkCacheHttpJsonContext.Default.ErrorEnvelope)?.Error;
        }
        catch (JsonException)
        {
            // not an error envelope; report the status alone
        }

        string detail = error?.Message ?? "no error envelope";
        return new ChunkCacheRequestException(status, error?.Type, error?.Code,
            $"{method} {url} failed with {(int)status} {status}: {detail}");
    }
}
