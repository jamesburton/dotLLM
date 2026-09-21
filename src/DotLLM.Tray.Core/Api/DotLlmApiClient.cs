using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace DotLLM.Tray.Api;

/// <summary>
/// Typed client for the dotLLM server's management API (#454).
/// </summary>
/// <remarks>
/// <para>
/// This is the tray's <b>only</b> route into the server. There is deliberately no in-process path:
/// the tray links neither DotLLM.Engine nor DotLLM.Server, so "reach around the API" is not merely
/// discouraged, it does not compile.
/// </para>
/// <para>
/// The client owns no <see cref="HttpClient"/> lifetime policy of its own — one is injected — so
/// tests drive it through a stub <see cref="HttpMessageHandler"/> and the UI shares a single
/// long-lived instance.
/// </para>
/// </remarks>
public sealed class DotLlmApiClient
{
    /// <summary>
    /// Per-call budget for the <c>/health</c> and <c>/ready</c> probes.
    /// </summary>
    /// <remarks>
    /// The shared <see cref="HttpClient"/> is given a long timeout because an unload waits behind
    /// an in-flight generation and a load can take minutes. The probes must not inherit it: the
    /// tray polls every few seconds, so a wedged server would otherwise stack minutes' worth of
    /// outstanding probes.
    /// </remarks>
    private static readonly TimeSpan ProbeTimeout = TimeSpan.FromSeconds(3);

    private readonly HttpClient _http;

    /// <summary>Creates a client over an already-configured <see cref="HttpClient"/>.</summary>
    /// <param name="http">Transport. Its <see cref="HttpClient.BaseAddress"/> must be set.</param>
    public DotLlmApiClient(HttpClient http)
    {
        ArgumentNullException.ThrowIfNull(http);
        if (http.BaseAddress is null)
            throw new ArgumentException("BaseAddress must be set.", nameof(http));
        _http = http;
    }

    /// <summary>The server base address this client talks to.</summary>
    public Uri BaseAddress => _http.BaseAddress!;

    // ───────────────────────────────── probes ─────────────────────────────────

    /// <summary>
    /// Probes <c>GET /health</c>. Returns false rather than throwing when nothing is listening —
    /// this is the "is a server already running here?" question, and an unreachable port is a
    /// legitimate answer, not an error.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    public Task<bool> IsHealthyAsync(CancellationToken ct = default) => ProbeAsync("/health", ct);

    /// <summary>
    /// Probes <c>GET /ready</c>. False means the process is up but has no model loaded — the
    /// server allows starting without one, so this is a distinct state from "not running".
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    public Task<bool> IsReadyAsync(CancellationToken ct = default) => ProbeAsync("/ready", ct);

    private async Task<bool> ProbeAsync(string route, CancellationToken ct)
    {
        using var budget = CancellationTokenSource.CreateLinkedTokenSource(ct);
        budget.CancelAfter(ProbeTimeout);
        try
        {
            using var response = await _http.GetAsync(route, budget.Token).ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch (HttpRequestException)
        {
            // Connection refused: nothing is listening. A legitimate answer, not an error.
            return false;
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            // Connect or probe timeout. Treated as "not answering" rather than propagated, so a
            // wedged server shows as down instead of throwing out of a timer tick.
            return false;
        }
    }

    // ───────────────────────────────── models ─────────────────────────────────

    /// <summary>Lists resident models with keep-alive/expiry state (<c>GET /v1/models</c>, ungated).</summary>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayModelList> GetResidentModelsAsync(CancellationToken ct = default) =>
        GetAsync("/v1/models", TrayJsonContext.Default.TrayModelList, ct);

    /// <summary>Lists locally downloaded models (<c>GET /v1/models/available</c>, ungated).</summary>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayAvailableModelList> GetAvailableModelsAsync(CancellationToken ct = default) =>
        GetAsync("/v1/models/available", TrayJsonContext.Default.TrayAvailableModelList, ct);

    /// <summary>Loads or hot-swaps a model (<c>POST /v1/models/load</c>).</summary>
    /// <param name="request">Load parameters.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayLoadResult> LoadModelAsync(TrayLoadRequest request, CancellationToken ct = default) =>
        PostAsync("/v1/models/load", request, TrayJsonContext.Default.TrayLoadRequest,
            TrayJsonContext.Default.TrayLoadResult, ct);

    /// <summary>
    /// Unloads one model, or all of them (<c>POST /v1/models/unload</c>, gated).
    /// </summary>
    /// <remarks>
    /// The server waits behind an in-flight generation rather than interrupting it, so this call
    /// can take as long as the current response. The tray must not assume it is instant.
    /// </remarks>
    /// <param name="request">Which model(s) to unload.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayUnloadResult> UnloadModelAsync(TrayUnloadRequest request, CancellationToken ct = default) =>
        PostAsync("/v1/models/unload", request, TrayJsonContext.Default.TrayUnloadRequest,
            TrayJsonContext.Default.TrayUnloadResult, ct);

    /// <summary>Re-enables a model key (<c>POST /v1/models/enable</c>, gated).</summary>
    /// <param name="modelKey">Model key.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayEnableResult> EnableModelAsync(string modelKey, CancellationToken ct = default) =>
        PostAsync("/v1/models/enable", new TrayEnableRequest { Model = modelKey },
            TrayJsonContext.Default.TrayEnableRequest, TrayJsonContext.Default.TrayEnableResult, ct);

    /// <summary>
    /// Disables a model key (<c>POST /v1/models/disable</c>, gated).
    /// </summary>
    /// <remarks>
    /// Disabling blocks the <i>next</i> activation; it does not unload a model that is serving now
    /// (<see cref="TrayEnableResult.StillLoaded"/> reports that), and the curation is in-memory
    /// only — it is lost when the server restarts. The tray says both in its UI.
    /// </remarks>
    /// <param name="modelKey">Model key.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayEnableResult> DisableModelAsync(string modelKey, CancellationToken ct = default) =>
        PostAsync("/v1/models/disable", new TrayEnableRequest { Model = modelKey },
            TrayJsonContext.Default.TrayEnableRequest, TrayJsonContext.Default.TrayEnableResult, ct);

    // ──────────────────────────────── settings ────────────────────────────────

    /// <summary>Reads the live runtime settings (<c>GET /v1/settings</c>, ungated).</summary>
    /// <param name="ct">Cancellation token.</param>
    public Task<TraySettingsDto> GetSettingsAsync(CancellationToken ct = default) =>
        GetAsync("/v1/settings", TrayJsonContext.Default.TraySettingsDto, ct);

    /// <summary>Applies a partial settings update (<c>PUT /v1/settings</c>, gated).</summary>
    /// <param name="update">Only the non-null fields are sent and applied.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<TraySettingsUpdateResult> UpdateSettingsAsync(
        TraySettingsUpdate update, CancellationToken ct = default)
    {
        using var content = JsonContent.Create(update, TrayJsonContext.Default.TraySettingsUpdate);
        using var response = await _http.PutAsync("/v1/settings", content, ct).ConfigureAwait(false);
        return await ReadAsync(response, "PUT /v1/settings",
            TrayJsonContext.Default.TraySettingsUpdateResult, ct).ConfigureAwait(false);
    }

    // ───────────────────────────────── devices ────────────────────────────────

    /// <summary>
    /// Enumerates compute backends (<c>GET /v1/devices</c>, ungated).
    /// </summary>
    /// <remarks>
    /// Callers must gate a device picker on <see cref="TrayBackendInfo.Servable"/>, not
    /// <see cref="TrayBackendInfo.Available"/>: the server's load path dispatches to CPU or CUDA
    /// only, so Vulkan reports available-but-not-servable and offering it would produce a load
    /// that silently lands on the CPU.
    /// </remarks>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayDeviceList> GetDevicesAsync(CancellationToken ct = default) =>
        GetAsync("/v1/devices", TrayJsonContext.Default.TrayDeviceList, ct);

    // ────────────────────────────────── pull ──────────────────────────────────

    /// <summary>
    /// Starts a download and returns immediately with the job (<c>POST /v1/models/pull</c> with
    /// <c>stream: false</c>, gated). The caller polls, or streams a separate job-scoped request.
    /// </summary>
    /// <param name="request">Repo/file to fetch.</param>
    /// <param name="ct">Cancellation token — cancels the <i>request</i>, never the download.</param>
    public async Task<TrayPullJob> StartPullAsync(TrayPullRequest request, CancellationToken ct = default)
    {
        // `stream` is always sent explicitly: the server treats an omitted value as streaming.
        var body = request with { Stream = false };
        using var content = JsonContent.Create(body, TrayJsonContext.Default.TrayPullRequest);
        using var response = await _http.PostAsync("/v1/models/pull", content, ct).ConfigureAwait(false);
        return await ReadAsync(response, "POST /v1/models/pull",
            TrayJsonContext.Default.TrayPullJob, ct).ConfigureAwait(false);
    }

    /// <summary>Lists all known pull jobs (<c>GET /v1/models/pull</c>, ungated).</summary>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayPullJobList> GetPullJobsAsync(CancellationToken ct = default) =>
        GetAsync("/v1/models/pull", TrayJsonContext.Default.TrayPullJobList, ct);

    /// <summary>Polls one pull job (<c>GET /v1/models/pull/{id}</c>, ungated).</summary>
    /// <param name="jobId">Job id.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<TrayPullJob> GetPullJobAsync(string jobId, CancellationToken ct = default) =>
        GetAsync($"/v1/models/pull/{Uri.EscapeDataString(jobId)}", TrayJsonContext.Default.TrayPullJob, ct);

    /// <summary>
    /// Cancels a download (<c>DELETE /v1/models/pull/{id}</c>, gated).
    /// </summary>
    /// <remarks>
    /// This is the <i>only</i> thing that stops a download. Closing the progress stream, closing
    /// the tray window, or even exiting the tray does not — the job's lifetime is deliberately
    /// decoupled from the HTTP connection so a multi-gigabyte pull survives a UI restart.
    /// </remarks>
    /// <param name="jobId">Job id.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<TrayStatusResponse> CancelPullAsync(string jobId, CancellationToken ct = default)
    {
        var route = $"/v1/models/pull/{Uri.EscapeDataString(jobId)}";
        using var response = await _http.DeleteAsync(route, ct).ConfigureAwait(false);
        return await ReadAsync(response, $"DELETE {route}",
            TrayJsonContext.Default.TrayStatusResponse, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Opens an SSE progress stream for a pull (<c>POST /v1/models/pull</c> with <c>stream: true</c>,
    /// gated), yielding each job snapshot until the terminal <c>data: [DONE]</c>.
    /// </summary>
    /// <remarks>
    /// Abandoning the enumeration (or cancelling <paramref name="ct"/>) ends the stream only. The
    /// download continues; use <see cref="CancelPullAsync"/> to actually stop it.
    /// </remarks>
    /// <param name="request">Repo/file to fetch.</param>
    /// <param name="ct">Cancellation token for the stream.</param>
    public async IAsyncEnumerable<TrayPullJob> StreamPullAsync(
        TrayPullRequest request,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var body = request with { Stream = true };
        using var message = new HttpRequestMessage(HttpMethod.Post, "/v1/models/pull")
        {
            Content = JsonContent.Create(body, TrayJsonContext.Default.TrayPullRequest),
        };

        using var response = await _http
            .SendAsync(message, HttpCompletionOption.ResponseHeadersRead, ct)
            .ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
            throw await ToExceptionAsync(response, "POST /v1/models/pull (stream)", ct).ConfigureAwait(false);

        await using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var reader = new StreamReader(stream);

        while (true)
        {
            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line is null)
                break;
            if (line.Length == 0 || !line.StartsWith("data:", StringComparison.Ordinal))
                continue;

            var payload = line[5..].Trim();
            if (payload is "[DONE]")
                yield break;

            TrayPullJob? job;
            try
            {
                job = JsonSerializer.Deserialize(payload, TrayJsonContext.Default.TrayPullJob);
            }
            catch (JsonException)
            {
                // A malformed frame is not worth tearing the stream down for; the next tick, or
                // the terminal frame, carries the same state.
                continue;
            }

            if (job is not null)
                yield return job;
        }
    }

    // ───────────────────────────────── plumbing ───────────────────────────────

    private async Task<T> GetAsync<T>(
        string route, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken ct)
    {
        using var response = await _http.GetAsync(route, ct).ConfigureAwait(false);
        return await ReadAsync(response, $"GET {route}", typeInfo, ct).ConfigureAwait(false);
    }

    private async Task<TResponse> PostAsync<TRequest, TResponse>(
        string route,
        TRequest request,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo<TRequest> requestInfo,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo<TResponse> responseInfo,
        CancellationToken ct)
    {
        using var content = JsonContent.Create(request, requestInfo);
        using var response = await _http.PostAsync(route, content, ct).ConfigureAwait(false);
        return await ReadAsync(response, $"POST {route}", responseInfo, ct).ConfigureAwait(false);
    }

    private static async Task<T> ReadAsync<T>(
        HttpResponseMessage response,
        string route,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo,
        CancellationToken ct)
    {
        if (!response.IsSuccessStatusCode)
            throw await ToExceptionAsync(response, route, ct).ConfigureAwait(false);

        var value = await response.Content.ReadFromJsonAsync(typeInfo, ct).ConfigureAwait(false);
        return value ?? throw new DotLlmApiException(
            response.StatusCode, $"{route} returned an empty body.", route);
    }

    /// <summary>
    /// Turns a failure response into a <see cref="DotLlmApiException"/>, preferring the server's
    /// own <c>error</c> text. The gated routes put the flag to set in there.
    /// </summary>
    private static async Task<DotLlmApiException> ToExceptionAsync(
        HttpResponseMessage response, string route, CancellationToken ct)
    {
        string message;
        try
        {
            var error = await response.Content
                .ReadFromJsonAsync(TrayJsonContext.Default.TrayErrorResponse, ct)
                .ConfigureAwait(false);
            message = string.IsNullOrWhiteSpace(error?.Error)
                ? DescribeStatus(response.StatusCode, route)
                : error!.Error;
        }
        catch (Exception ex) when (ex is JsonException or NotSupportedException or HttpRequestException)
        {
            message = DescribeStatus(response.StatusCode, route);
        }

        return new DotLlmApiException(response.StatusCode, message, route);
    }

    private static string DescribeStatus(HttpStatusCode status, string route) =>
        $"{route} failed: {(int)status} {status}.";
}
