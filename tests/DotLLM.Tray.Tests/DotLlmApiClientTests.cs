using System.Net;
using DotLLM.Tray.Api;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// Exercises the tray's API client over a stub transport.
/// </summary>
/// <remarks>
/// These are route/verb/body assertions, not contract assertions — the wire shape is
/// <see cref="TrayContractTests"/>'s job. What is checked here is that the client calls the route
/// #454 documents, with the verb it documents, and that it translates the server's failure modes
/// (notably the admin gate's 403) into something the UI can act on.
/// </remarks>
public sealed class DotLlmApiClientTests
{
    [Fact]
    public async Task IsHealthy_ReturnsFalseWhenNothingIsListening()
    {
        // The "is a server already running?" probe. A refused connection is an answer, not a
        // crash — the supervisor calls this before deciding to spawn.
        using var http = new HttpClient(
            new ThrowingHandler(new HttpRequestException("No connection could be made.")),
            disposeHandler: true)
        {
            BaseAddress = new Uri("http://localhost:18080/"),
        };

        Assert.False(await new DotLlmApiClient(http).IsHealthyAsync(CancellationToken.None));
    }

    [Fact]
    public async Task IsHealthy_TrueButNotReady_IsARunningServerWithoutAModel()
    {
        using var http = StubHttpMessageHandler.Client(
            request => request.RequestUri!.AbsolutePath == "/health"
                ? StubHttpMessageHandler.Json("""{"status":"ok"}""")
                : new HttpResponseMessage(HttpStatusCode.ServiceUnavailable),
            out _);

        var client = new DotLlmApiClient(http);
        Assert.True(await client.IsHealthyAsync(CancellationToken.None));
        Assert.False(await client.IsReadyAsync(CancellationToken.None));
    }

    [Fact]
    public async Task GetResidentModels_CallsTheDocumentedRoute()
    {
        using var http = StubHttpMessageHandler.Client(
            _ => StubHttpMessageHandler.Json(
                """{"data":[{"id":"m","is_active":true,"expires_in_seconds":30,"size_bytes":10}]}"""),
            out var handler);

        var result = await new DotLlmApiClient(http)
            .GetResidentModelsAsync(CancellationToken.None);

        Assert.Equal("/v1/models", Assert.Single(handler.Requests).Uri.AbsolutePath);
        Assert.Equal(30, Assert.Single(result.Data).ExpiresInSeconds);
    }

    [Fact]
    public async Task UpdateSettings_UsesPutAndSendsOnlyTheEditedField()
    {
        using var http = StubHttpMessageHandler.Client(
            _ => StubHttpMessageHandler.Json(
                """{"settings":{"keep_alive_seconds":60},"applied":["keep_alive_seconds"],"restart_required":[],"evicted":[]}"""),
            out var handler);

        var result = await new DotLlmApiClient(http).UpdateSettingsAsync(
            new TraySettingsUpdate { KeepAliveSeconds = 60 }, CancellationToken.None);

        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Put, request.Method);
        Assert.Equal("/v1/settings", request.Uri.AbsolutePath);
        Assert.Contains("keep_alive_seconds", request.Body!, StringComparison.Ordinal);
        Assert.DoesNotContain("max_resident_models", request.Body!, StringComparison.Ordinal);
        Assert.Equal(["keep_alive_seconds"], result.Applied);
    }

    [Fact]
    public async Task AdminGateRefusal_SurfacesTheServersOwnMessage()
    {
        // The #454 gate answers 403 with a body naming the flag. A tray that swallowed it would
        // leave the user with buttons that do nothing and no explanation.
        const string GateBody = """
            {"error":"POST /v1/models/unload is disabled. Start the server with --allow-model-admin (ServerOptions.AllowModelAdminApi) to enable the model-administration API."}
            """;

        using var http = StubHttpMessageHandler.Client(
            _ => StubHttpMessageHandler.Json(GateBody, HttpStatusCode.Forbidden), out _);

        var exception = await Assert.ThrowsAsync<DotLlmApiException>(
            () => new DotLlmApiClient(http).UnloadModelAsync(
                new TrayUnloadRequest { All = true }, CancellationToken.None));

        Assert.True(exception.IsAdminGateRefusal);
        Assert.Contains("--allow-model-admin", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task NonJsonFailureBody_StillProducesAUsableMessage()
    {
        // Kestrel can answer with an HTML error page or nothing at all. The client must not throw
        // a JsonException out of a call site that is only prepared for DotLlmApiException.
        using var http = StubHttpMessageHandler.Client(
            _ => new HttpResponseMessage(HttpStatusCode.InternalServerError)
            {
                Content = new StringContent("<html>boom</html>", System.Text.Encoding.UTF8, "text/html"),
            },
            out _);

        var exception = await Assert.ThrowsAsync<DotLlmApiException>(
            () => new DotLlmApiClient(http).GetSettingsAsync(CancellationToken.None));

        Assert.Equal(HttpStatusCode.InternalServerError, exception.StatusCode);
        Assert.Contains("/v1/settings", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task StartPull_ForcesStreamFalseSoTheCallReturnsImmediately()
    {
        using var http = StubHttpMessageHandler.Client(
            _ => StubHttpMessageHandler.Json(
                """{"id":"j1","repo_id":"o/r","filename":"f.gguf","revision":"main","status":"running","bytes_downloaded":0,"started_at":1}""",
                HttpStatusCode.Accepted),
            out var handler);

        var job = await new DotLlmApiClient(http).StartPullAsync(
            new TrayPullRequest { RepoId = "o/r", Filename = "f.gguf" }, CancellationToken.None);

        // Even though the caller left Stream null, the client must send false — otherwise the
        // server opens an SSE stream and this call never returns.
        Assert.Contains("\"stream\":false", Assert.Single(handler.Requests).Body!, StringComparison.Ordinal);
        Assert.Equal("j1", job.Id);
    }

    [Fact]
    public async Task StreamPull_ParsesEveryFrameAndStopsAtDone()
    {
        const string Stream = """
            data: {"id":"j1","repo_id":"o/r","filename":"f.gguf","revision":"main","status":"running","bytes_downloaded":0,"total_bytes":100,"percent":0,"started_at":1}

            data: {"id":"j1","repo_id":"o/r","filename":"f.gguf","revision":"main","status":"running","bytes_downloaded":50,"total_bytes":100,"percent":50,"started_at":1}

            data: {"id":"j1","repo_id":"o/r","filename":"f.gguf","revision":"main","status":"completed","bytes_downloaded":100,"total_bytes":100,"percent":100,"started_at":1,"completed_at":2}

            data: [DONE]


            """;

        using var http = StubHttpMessageHandler.Client(_ => StubHttpMessageHandler.Sse(Stream), out var handler);

        var jobs = new List<TrayPullJob>();
        await foreach (var job in new DotLlmApiClient(http).StreamPullAsync(
            new TrayPullRequest { RepoId = "o/r", Filename = "f.gguf" }, CancellationToken.None))
        {
            jobs.Add(job);
        }

        Assert.Contains("\"stream\":true", Assert.Single(handler.Requests).Body!, StringComparison.Ordinal);
        Assert.Equal(3, jobs.Count);
        Assert.Equal([0d, 50d, 100d], jobs.Select(j => j.Percent!.Value));
        Assert.True(jobs[^1].IsTerminal);
    }

    [Fact]
    public async Task StreamPull_SkipsAMalformedFrameRatherThanTearingDownTheStream()
    {
        const string Stream = """
            data: {"id":"j1","repo_id":"o/r","filename":"f.gguf","revision":"main","status":"running","bytes_downloaded":1,"started_at":1}

            data: {not json at all

            data: {"id":"j1","repo_id":"o/r","filename":"f.gguf","revision":"main","status":"completed","bytes_downloaded":2,"started_at":1}

            data: [DONE]


            """;

        using var http = StubHttpMessageHandler.Client(_ => StubHttpMessageHandler.Sse(Stream), out _);

        var jobs = new List<TrayPullJob>();
        await foreach (var job in new DotLlmApiClient(http).StreamPullAsync(
            new TrayPullRequest { RepoId = "o/r", Filename = "f.gguf" }, CancellationToken.None))
        {
            jobs.Add(job);
        }

        // The terminal frame is what matters; losing an intermediate tick is survivable, losing
        // the completion is not.
        Assert.Equal(2, jobs.Count);
        Assert.Equal("completed", jobs[^1].Status);
    }

    [Fact]
    public async Task CancelPull_UsesDelete_TheOnlyVerbThatStopsADownload()
    {
        using var http = StubHttpMessageHandler.Client(
            _ => StubHttpMessageHandler.Json("""{"status":"cancelling"}"""), out var handler);

        var result = await new DotLlmApiClient(http).CancelPullAsync("j1", CancellationToken.None);

        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Delete, request.Method);
        Assert.Equal("/v1/models/pull/j1", request.Uri.AbsolutePath);
        Assert.Equal("cancelling", result.Status);
    }

    [Fact]
    public async Task EnableAndDisable_HitTheirOwnRoutes()
    {
        using var http = StubHttpMessageHandler.Client(
            _ => StubHttpMessageHandler.Json("""{"model":"m","enabled":false,"still_loaded":true}"""),
            out var handler);

        var client = new DotLlmApiClient(http);
        await client.DisableModelAsync("m", CancellationToken.None);
        await client.EnableModelAsync("m", CancellationToken.None);

        Assert.Equal("/v1/models/disable", handler.Requests[0].Uri.AbsolutePath);
        Assert.Equal("/v1/models/enable", handler.Requests[1].Uri.AbsolutePath);
        Assert.All(handler.Requests, r => Assert.Equal(HttpMethod.Post, r.Method));
    }

    [Fact]
    public void Constructor_RejectsAClientWithoutABaseAddress()
    {
        using var http = new HttpClient();
        Assert.Throws<ArgumentException>(() => new DotLlmApiClient(http));
    }

    private sealed class ThrowingHandler(Exception exception) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromException<HttpResponseMessage>(exception);
    }
}
