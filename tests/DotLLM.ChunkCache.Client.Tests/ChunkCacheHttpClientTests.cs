using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary><see cref="ChunkCacheHttpClient"/> against a fake handler (no server).</summary>
public sealed class ChunkCacheHttpClientTests : IDisposable
{
    private static readonly Uri s_base = new("http://localhost:8080");
    private readonly List<HttpClient> _http = [];

    public void Dispose()
    {
        foreach (HttpClient h in _http) h.Dispose();
    }

    private ChunkCacheHttpClient Client(StubHandler handler, string? apiKey = null, IChunkIndexQueries? queries = null,
        Uri? baseUrl = null)
    {
        var http = new HttpClient(handler);
        _http.Add(http);
        return new ChunkCacheHttpClient(http, baseUrl ?? s_base, apiKey, queries ?? new FakeQueries(new SelectResult()));
    }

    private static string CapabilitiesFixture => Encoding.UTF8.GetString(Fixtures.Bytes("capabilities.json"));

    // ---------------------------------------------------------------- capabilities

    /// <summary>A 404 means "no convention: strip annotations" (spec 5.9).</summary>
    [Fact]
    public async Task Capabilities_404_IsNull()
    {
        var handler = StubHandler.Json(HttpStatusCode.NotFound, """{"error":{"message":"not found","type":"not_found_error","code":null}}""");

        Assert.Null(await Client(handler).GetCapabilitiesAsync(CancellationToken.None));

        var (req, _) = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Get, req.Method);
        Assert.Equal("http://localhost:8080/v1/chunk-cache", req.RequestUri!.ToString());
    }

    [Fact]
    public async Task Capabilities_ParsesFixture()
    {
        var caps = await Client(StubHandler.Json(HttpStatusCode.OK, CapabilitiesFixture)).GetCapabilitiesAsync(CancellationToken.None);

        Assert.NotNull(caps);
        Assert.Equal([1], caps.Versions!);
        Assert.True(caps.SupportsVersion(1));
        Assert.Equal(["exact", "shadow", "balanced", "fast", "auto"], caps.Policies!);
        Assert.Equal(["chunk_starts"], caps.Selectors!);
        Assert.Equal(["symbol-512"], caps.WireChunkers!);
        Assert.Empty(caps.AutoValidated!);
        Assert.Equal("qwen3-4b", caps.Model!.Id);
        Assert.True(caps.Model.Supported);
        Assert.Null(caps.Model.Reason);
        Assert.Equal(0, caps.Store!.Entries);
        Assert.Equal(0, caps.Store.BudgetBytes);
        Assert.Equal("f32", caps.Store.Dtype);
        Assert.Equal(4096, caps.MaxChunksPerRequest);
    }

    /// <summary>A body that does not list version 1 is a server this client cannot speak to: treated as absent.</summary>
    [Theory]
    [InlineData("""{"versions":[2],"policies":["exact"]}""")]
    [InlineData("""{"versions":[],"policies":["exact"]}""")]
    [InlineData("""{"policies":["exact"]}""")]
    [InlineData("""<html>not json</html>""")]
    public async Task Capabilities_UnknownVersionOnly_IsNull(string body)
    {
        Assert.Null(await Client(StubHandler.Json(HttpStatusCode.OK, body)).GetCapabilitiesAsync(CancellationToken.None));
    }

    /// <summary>
    /// Only a 404 means "absent". A 401 / 403 / 500 is a present server refusing or failing, and must not silently turn
    /// annotations off (nor on).
    /// </summary>
    [Theory]
    [InlineData(HttpStatusCode.Unauthorized, "authentication_error")]
    [InlineData(HttpStatusCode.Forbidden, "permission_error")]
    [InlineData(HttpStatusCode.InternalServerError, "api_error")]
    public async Task Capabilities_OtherErrors_Throw(HttpStatusCode status, string type)
    {
        var handler = StubHandler.Json(status, $$$"""{"error":{"message":"no","type":"{{{type}}}","code":null}}""");
        var ex = await Assert.ThrowsAsync<ChunkCacheRequestException>(() => Client(handler).GetCapabilitiesAsync(CancellationToken.None).AsTask());
        Assert.Equal(status, ex.StatusCode);
        Assert.Equal(type, ex.ErrorType);
    }

    /// <summary>OpenAI-style base URLs end in <c>/v1</c>; the probe path must not double it.</summary>
    [Theory]
    [InlineData("http://h:1")]
    [InlineData("http://h:1/")]
    [InlineData("http://h:1/v1")]
    [InlineData("http://h:1/v1/")]
    public async Task BaseUrl_WithOrWithoutV1_ProbesOnePath(string baseUrl)
    {
        var handler = StubHandler.Json(HttpStatusCode.NotFound, "{}");
        await Client(handler, baseUrl: new Uri(baseUrl)).GetCapabilitiesAsync(CancellationToken.None);
        Assert.Equal("http://h:1/v1/chunk-cache", handler.Requests[0].Request.RequestUri!.ToString());
    }

    [Fact]
    public async Task BaseUrl_WithPathPrefix_IsKept()
    {
        var handler = StubHandler.Json(HttpStatusCode.NotFound, "{}");
        await Client(handler, baseUrl: new Uri("http://h:1/proxy/llm/v1")).GetCapabilitiesAsync(CancellationToken.None);
        Assert.Equal("http://h:1/proxy/llm/v1/chunk-cache", handler.Requests[0].Request.RequestUri!.ToString());
    }

    // ---------------------------------------------------------------- api key

    /// <summary>The key goes in <c>X-API-Key</c> on every call, exactly; never in a URL.</summary>
    [Fact]
    public async Task ApiKey_SentAsXApiKey()
    {
        using var handler = new StubHandler(r => r.RequestUri!.AbsolutePath.EndsWith("chunk-cache", StringComparison.Ordinal)
            ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(CapabilitiesFixture) }
            : new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("{}") });
        var client = Client(handler, apiKey: "sk-test-123");

        await client.GetCapabilitiesAsync(CancellationToken.None);
        await client.EvictAsync(new EvictRequest(), CancellationToken.None);
        await client.PrewarmAsync(new PrewarmRequest { Texts = [new PrewarmText { Id = "a", Text = "x" }] }, CancellationToken.None);
        await client.ExplainAsync(new JsonObject { ["messages"] = new JsonArray() }, CancellationToken.None);

        Assert.Equal(4, handler.Requests.Count);
        foreach (var (req, _) in handler.Requests)
        {
            Assert.Equal(["sk-test-123"], req.Headers.GetValues("X-API-Key"));
            Assert.DoesNotContain("sk-test-123", req.RequestUri!.ToString(), StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task ApiKey_Null_SendsNoKeyHeader()
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, CapabilitiesFixture);
        await Client(handler, apiKey: null).GetCapabilitiesAsync(CancellationToken.None);
        Assert.False(handler.Requests[0].Request.Headers.Contains("X-API-Key"));
        Assert.Null(handler.Requests[0].Request.Headers.Authorization);
    }

    // ---------------------------------------------------------------- prewarm

    /// <summary>
    /// Prewarm sends texts, never hashes (a hash-only API would be a membership oracle, spec 7.2). A <c>select</c> input
    /// is resolved locally with <c>include_text</c> and its chunk texts are sent; the select's hashes are not.
    /// </summary>
    [Fact]
    public async Task Prewarm_SendsTextNeverHashesOnly()
    {
        var selected = new SelectResult
        {
            Chunks =
            [
                new SelectedChunk { Id = "a.cs:1-3@0", Path = "a.cs", StartLine = 1, EndLine = 3, Hash = ChunkHash.Of("class A {}\n"), Text = "class A {}\n" },
                new SelectedChunk { Id = "b.cs:1-2@0", Path = "b.cs", StartLine = 1, EndLine = 2, Hash = ChunkHash.Of("class B {}\n"), Text = "class B {}\n" },
            ],
        };
        var queries = new FakeQueries(selected);
        var handler = StubHandler.Json(HttpStatusCode.OK, """{"requested":3,"already_stored":1,"queued":2,"queued_tokens":40,"refused":0,"refused_reasons":[]}""");

        var result = await Client(handler, queries: queries).PrewarmAsync(new PrewarmRequest
        {
            Store = "C:/store",
            Select = new SelectRequest { Path = "src", Paths = ["*.cs"] },
            Texts = [new PrewarmText { Id = "extra", Text = "extra text\n" }],
            MaxTokens = 1000,
        }, CancellationToken.None);

        Assert.True(queries.LastSelect!.IncludeText);
        Assert.Equal("C:/store", queries.LastSelect.Store);
        Assert.Equal(["*.cs"], queries.LastSelect.Paths!);

        var (req, body) = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, req.Method);
        Assert.Equal("http://localhost:8080/v1/chunk-cache/prewarm", req.RequestUri!.ToString());
        JsonObject sent = JsonNode.Parse(body!)!.AsObject();
        Assert.Equal(["texts", "max_tokens"], sent.Select(p => p.Key));
        Assert.Equal(1000, (int)sent["max_tokens"]!);
        JsonArray texts = sent["texts"]!.AsArray();
        Assert.Equal(["extra text\n", "class A {}\n", "class B {}\n"], texts.Select(t => (string)t!["text"]!));
        Assert.Equal(["extra", "a.cs:1-3@0", "b.cs:1-2@0"], texts.Select(t => (string)t!["id"]!));
        Assert.Equal("a.cs", (string)texts[1]!["source"]!["path"]!);
        Assert.DoesNotContain("hash", body!, StringComparison.Ordinal);
        Assert.DoesNotContain("sha256:", body!, StringComparison.Ordinal);

        Assert.Equal(3, result.Requested);
        Assert.Equal(1, result.AlreadyStored);
        Assert.Equal(2, result.Queued);
        Assert.Equal(40, result.QueuedTokens);
    }

    /// <summary>A selected chunk without text cannot be prewarmed by hash: the call fails before anything is sent.</summary>
    [Fact]
    public async Task Prewarm_SelectedChunkWithoutText_ThrowsAndSendsNothing()
    {
        var queries = new FakeQueries(new SelectResult { Chunks = [new SelectedChunk { Id = "a", Hash = ChunkHash.Of("a") }] });
        var handler = StubHandler.Json(HttpStatusCode.OK, "{}");

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            Client(handler, queries: queries).PrewarmAsync(new PrewarmRequest { Select = new SelectRequest() }, CancellationToken.None).AsTask());
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task Prewarm_TextMissing_ThrowsAndSendsNothing()
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, "{}");
        await Assert.ThrowsAsync<ArgumentException>(() =>
            Client(handler).PrewarmAsync(new PrewarmRequest { Texts = [new PrewarmText { Id = "a" }] }, CancellationToken.None).AsTask());
        await Assert.ThrowsAsync<ArgumentException>(() =>
            Client(handler).PrewarmAsync(new PrewarmRequest(), CancellationToken.None).AsTask());
        Assert.Empty(handler.Requests);
    }

    // ---------------------------------------------------------------- evict

    /// <summary>
    /// Evict always sends <c>mode</c> and <c>dry_run</c> resolved: an unset <c>dry_run</c> goes out as <c>true</c>, so a
    /// caller who did not ask to delete never depends on the server's default to stay safe.
    /// </summary>
    [Fact]
    public async Task Evict_Defaults_SentExplicitlyAsDryRunOrphans()
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, """{"evicted":3,"freed_bytes":4096,"kept_referenced":9,"dry_run":true}""");

        var result = await Client(handler).EvictAsync(new EvictRequest { Store = "C:/store", Server = "http://x" }, CancellationToken.None);

        var (req, body) = Assert.Single(handler.Requests);
        Assert.Equal("http://localhost:8080/v1/chunk-cache/evict", req.RequestUri!.ToString());
        JsonObject sent = JsonNode.Parse(body!)!.AsObject();
        Assert.Equal("orphans", (string)sent["mode"]!);
        Assert.True((bool)sent["dry_run"]!);
        Assert.False(sent.ContainsKey("store"));
        Assert.False(sent.ContainsKey("server"));

        Assert.Equal(3, result.Evicted);
        Assert.Equal(4096, result.FreedBytes);
        Assert.Equal(9, result.KeptReferenced);
        Assert.True(result.DryRun);
    }

    [Fact]
    public async Task Evict_ExplicitPathMode_SendsFields()
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, """{"evicted":0,"freed_bytes":0,"kept_referenced":0,"dry_run":false}""");
        await Client(handler).EvictAsync(new EvictRequest { Mode = "path", Path = "src/**", DryRun = false }, CancellationToken.None);

        JsonObject sent = JsonNode.Parse(handler.Requests[0].Body!)!.AsObject();
        Assert.Equal("path", (string)sent["mode"]!);
        Assert.Equal("src/**", (string)sent["path"]!);
        Assert.False((bool)sent["dry_run"]!);
    }

    /// <summary>Server errors surface with the spec 5.8 code (here the one 503 code).</summary>
    [Fact]
    public async Task Evict_StoreUnavailable_ThrowsWithCode()
    {
        var handler = StubHandler.Json(HttpStatusCode.ServiceUnavailable,
            """{"type":"error","error":{"message":"store lock timeout","type":"api_error","param":null,"code":"chunk_store_unavailable"}}""");

        var ex = await Assert.ThrowsAsync<ChunkCacheRequestException>(() => Client(handler).EvictAsync(new EvictRequest(), CancellationToken.None).AsTask());
        Assert.Equal(HttpStatusCode.ServiceUnavailable, ex.StatusCode);
        Assert.Equal("api_error", ex.ErrorType);
        Assert.Equal("chunk_store_unavailable", ex.Code);
        Assert.Contains("store lock timeout", ex.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- explain

    /// <summary>Explain posts the full request (text) under <c>request</c>, unchanged, and parses the spec 7.2 output.</summary>
    [Fact]
    public async Task Explain_PostsRequestBody_ParsesResult()
    {
        JsonObject manifest = Fixtures.Json("manifest.json");
        var handler = StubHandler.Json(HttpStatusCode.OK,
            """{"policy_would_apply":"exact","reason":null,"chunks":[{"id":"a","status":"miss","why":"CRLF vs index"}],"predicted_ms":{"reuse":12.5,"full_prefill":40},"warnings":["chunk_ignored_too_short"]}""");

        var result = await Client(handler).ExplainAsync(manifest, CancellationToken.None);

        var (req, body) = Assert.Single(handler.Requests);
        Assert.Equal("http://localhost:8080/v1/chunk-cache/explain", req.RequestUri!.ToString());
        JsonObject sent = JsonNode.Parse(body!)!.AsObject();
        Assert.True(JsonNode.DeepEquals(Fixtures.Json("manifest.json"), sent["request"]));
        Assert.Null(manifest.Parent); // the caller's object is copied, not re-parented into the envelope

        Assert.Equal("exact", result.PolicyWouldApply);
        Assert.Equal("miss", result.Chunks![0].Status);
        Assert.Equal("CRLF vs index", result.Chunks[0].Why);
        Assert.Equal(12.5, result.PredictedMs!.Reuse);
        Assert.Equal(40, result.PredictedMs.FullPrefill);
        Assert.Equal(["chunk_ignored_too_short"], result.Warnings!);
    }

    // ---------------------------------------------------------------- select / status

    /// <summary>Select and status are local index operations: delegated, never sent to the server.</summary>
    [Fact]
    public async Task SelectAndStatus_DelegateToQueries_NoHttp()
    {
        var queries = new FakeQueries(new SelectResult { SelectedTokens = 5 });
        var handler = StubHandler.Json(HttpStatusCode.OK, "{}");
        var client = Client(handler, queries: queries);

        Assert.Equal(5, (await client.SelectAsync(new SelectRequest { Symbol = "X" }, CancellationToken.None)).SelectedTokens);
        Assert.Equal(7, (await client.StatusAsync(new StatusRequest { Path = "p" }, CancellationToken.None)).TotalChunks);
        Assert.Equal("X", queries.LastSelect!.Symbol);
        Assert.Equal("p", queries.LastStatus!.Path);
        Assert.Empty(handler.Requests);
    }
}
