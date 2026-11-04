using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// The fail-closed probe (contract ruling 16) and the Task 6 review's wire fixes: prewarm <c>source</c> (ruling 14),
/// lone surrogates (ruling 17), nullable <c>dry_run</c>, one exception type, key and URL hygiene.
/// </summary>
public sealed class ProbeAndReviewFixTests : IDisposable
{
    private readonly List<HttpClient> _http = [];

    public void Dispose()
    {
        foreach (HttpClient h in _http) h.Dispose();
    }

    private ChunkCacheHttpClient Client(StubHandler handler, string? apiKey = null, string baseUrl = "http://localhost:8080",
        IChunkIndexQueries? queries = null)
    {
        var http = new HttpClient(handler);
        _http.Add(http);
        return new ChunkCacheHttpClient(http, new Uri(baseUrl), apiKey, queries ?? new FakeQueries(new SelectResult()));
    }

    // ---------------------------------------------------------------- ruling 16: probe failure = strip

    [Fact]
    public async Task Probe_Capable_WhenServerSpeaksV1()
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, Encoding.UTF8.GetString(Fixtures.Bytes("capabilities.json")));
        ChunkCacheProbeResult r = await Client(handler).ProbeAsync(CancellationToken.None);

        Assert.True(r.IsCapable);
        Assert.NotNull(r.Capabilities);
        Assert.Null(r.Failure);
    }

    /// <summary>404, 401, 403, 5xx, a transport error and an HTTP timeout all mean "not capable: strip annotations".</summary>
    [Theory]
    [InlineData(HttpStatusCode.NotFound, false)]
    [InlineData(HttpStatusCode.Unauthorized, true)]
    [InlineData(HttpStatusCode.Forbidden, true)]
    [InlineData(HttpStatusCode.InternalServerError, true)]
    [InlineData(HttpStatusCode.ServiceUnavailable, true)]
    public async Task Probe_ErrorStatus_IsNotCapable(HttpStatusCode status, bool hasFailure)
    {
        var handler = StubHandler.Json(status, """{"error":{"message":"x","type":"api_error","code":null}}""");
        ChunkCacheProbeResult r = await Client(handler).ProbeAsync(CancellationToken.None);

        Assert.False(r.IsCapable);
        Assert.Null(r.Capabilities);
        Assert.Equal(hasFailure, r.Failure is not null);
    }

    [Fact]
    public async Task Probe_TransportErrorOrHttpTimeout_IsNotCapable()
    {
        using var refused = new StubHandler(_ => throw new HttpRequestException("connection refused"));
        Assert.False((await Client(refused).ProbeAsync(CancellationToken.None)).IsCapable);

        // HttpClient.Timeout surfaces as TaskCanceledException although the caller did not cancel.
        using var timeout = new StubHandler(_ => throw new TaskCanceledException("HttpClient.Timeout"));
        ChunkCacheProbeResult r = await Client(timeout).ProbeAsync(CancellationToken.None);
        Assert.False(r.IsCapable);
        Assert.IsType<TaskCanceledException>(r.Failure);
    }

    /// <summary>The caller's own cancellation is not swallowed into "not capable".</summary>
    [Fact]
    public async Task Probe_CallerCancellation_Propagates()
    {
        using var cts = new CancellationTokenSource();
        using var handler = new StubHandler(_ =>
        {
            cts.Cancel();
            throw new TaskCanceledException();
        });
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Client(handler).ProbeAsync(cts.Token).AsTask());
    }

    // ---------------------------------------------------------------- ruling 14: prewarm source is {path} only

    [Fact]
    public async Task Prewarm_SourceCarriesPathOnly()
    {
        var queries = new FakeQueries(new SelectResult
        {
            Chunks = [new SelectedChunk { Id = "a", Path = "src/a.cs", StartLine = 3, EndLine = 9, Text = "class A {}\n" }],
        });
        var handler = StubHandler.Json(HttpStatusCode.OK, """{"requested":1}""");
        await Client(handler, queries: queries).PrewarmAsync(new PrewarmRequest { Select = new SelectRequest() }, CancellationToken.None);

        JsonObject source = JsonNode.Parse(handler.Requests[0].Body!)!["texts"]![0]!["source"]!.AsObject();
        Assert.Equal(["path"], source.Select(p => p.Key));
        Assert.Equal("src/a.cs", (string)source["path"]!);
    }

    /// <summary>No resolved texts: nothing is sent and an all-zero result comes back (documented).</summary>
    [Fact]
    public async Task Prewarm_SelectFindsNothing_SendsNothing()
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, "{}");
        var r = await Client(handler).PrewarmAsync(new PrewarmRequest { Select = new SelectRequest() }, CancellationToken.None);
        Assert.Empty(handler.Requests);
        Assert.Equal(0, r.Requested);
    }

    // ---------------------------------------------------------------- ruling 17: lone surrogates

    /// <summary>A lone surrogate hashes as U+FFFD (EF BF BD), the default <c>Encoding.UTF8</c> replacement.</summary>
    [Fact]
    public void Hash_LoneSurrogate_IsReplacementCharacter()
    {
        static string Hex(byte[] b) => "sha256:" + Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(b)).ToLowerInvariant();
        Assert.Equal(Hex([0x61, 0xEF, 0xBF, 0xBD, 0x62]), ChunkHash.Of("a\ud800b"));
        Assert.Equal(Hex([0x61, 0xEF, 0xBF, 0xBD, 0x62]), ChunkHash.Of("a\udc00b"));
        Assert.Equal(ChunkHash.Of("a�b"), ChunkHash.Of("a\ud800b"));
    }

    // ---------------------------------------------------------------- nits

    /// <summary>A server that omits <c>dry_run</c> must not read as "deleted for real".</summary>
    [Fact]
    public async Task EvictResult_DryRunOmitted_IsNull()
    {
        var r = await Client(StubHandler.Json(HttpStatusCode.OK, """{"evicted":1}""")).EvictAsync(new EvictRequest(), CancellationToken.None);
        Assert.Null(r.DryRun);
    }

    /// <summary>A 2xx body that is not the expected JSON is a <see cref="ChunkCacheRequestException"/>, like any failure.</summary>
    [Theory]
    [InlineData("<html/>")]
    [InlineData("""{"evicted":"many"}""")]
    [InlineData("null")]
    public async Task Post_BadSuccessBody_ThrowsRequestException(string body)
    {
        var handler = StubHandler.Json(HttpStatusCode.OK, body);
        var ex = await Assert.ThrowsAsync<ChunkCacheRequestException>(() => Client(handler).EvictAsync(new EvictRequest(), CancellationToken.None).AsTask());
        Assert.Equal(HttpStatusCode.OK, ex.StatusCode);
    }

    [Fact]
    public async Task ApiKey_Empty_SendsNoHeader()
    {
        var handler = StubHandler.Json(HttpStatusCode.NotFound, "{}");
        await Client(handler, apiKey: "").GetCapabilitiesAsync(CancellationToken.None);
        Assert.False(handler.Requests[0].Request.Headers.Contains("X-API-Key"));
    }

    /// <summary>Credentials in the base URL never reach exception messages (or the request URL).</summary>
    [Fact]
    public async Task BaseUrlUserInfo_NotInMessagesOrRequests()
    {
        var handler = StubHandler.Json(HttpStatusCode.InternalServerError, "{}");
        var ex = await Assert.ThrowsAsync<ChunkCacheRequestException>(() =>
            Client(handler, baseUrl: "http://user:s3cret@h:1/v1").EvictAsync(new EvictRequest(), CancellationToken.None).AsTask());

        Assert.DoesNotContain("s3cret", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("s3cret", handler.Requests[0].Request.RequestUri!.ToString(), StringComparison.Ordinal);
    }
}
