using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Xunit;

// Test code: handlers, requests and helpers are owned by the HttpClient / HttpMessageInvoker chains each test disposes.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>Tests from the Task 8 review (S1-S9 and the nits).</summary>
public sealed class ReviewFixTests
{
    private static readonly TimeSpan s_wait = TimeSpan.FromSeconds(15);

    private static string Text(char c) => "// file\n" + new string(c, 80) + "\n";

    private static ChunkPart[] PartsOf(string text, string id = "x") => [new ChunkPart(text, [new ChunkSpan(id, ChunkHash.Of(text))])];

    private static bool HasManifest(byte[]? body) => body is not null && Encoding.UTF8.GetString(body).Contains("\"chunk_cache\"", StringComparison.Ordinal);

    // ---- S1: a refusal turns annotations off for the failure period ------------------------------------------------

    /// <summary>
    /// A server that claims the convention but refuses every non-strict manifest: the first request costs a refused POST
    /// plus the plain resend; the second, within the failure TTL, costs one plain POST and no probe.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AlwaysRefusingServer_SecondRequestCostsNoExtraPost(bool cache)
    {
        string text = Text('r');
        byte[] input = Pipeline.StringBody(("tool", text));
        var capture = new CaptureHandler((_, body) => HasManifest(body)
            ? new HttpResponseMessage(HttpStatusCode.BadRequest) { Content = new StringContent("{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"chunk_span_invalid\"}}", Encoding.UTF8, "application/json") }
            : new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("{}", Encoding.UTF8, "application/json") });
        var probe = new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        IChunkCapabilitySource source = cache ? new ChunkCapabilityCache(probe) : new FixedChunkCapabilities(Pipeline.Caps());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(source) { InnerHandler = capture });

        for (int i = 0; i < 2; i++)
        {
            using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
            using HttpResponseMessage r = await invoker.SendAsync(Pipeline.Post(input), CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, r.StatusCode);
            if (i == 1) Assert.Equal(ChunkSkipReasons.NotCapable, scope.Outcome!.SkipReason);
        }

        Assert.Equal(3, capture.Seen.Count);
        Assert.Equal([true, false, false], capture.Seen.Select(s => HasManifest(s.Body)));
        if (cache) Assert.Equal(1, probe.Calls);
    }

    /// <summary>After the failure TTL, a refused server is probed and annotated again (the refusal is not permanent).</summary>
    [Fact]
    public async Task Refusal_ExpiresAfterFailureTtl()
    {
        var time = new ManualTime();
        var probe = new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        var cache = new ChunkCapabilityCache(probe, failureTtl: TimeSpan.FromSeconds(30), timeProvider: time);
        Assert.NotNull(await cache.GetAsync("m", CancellationToken.None));
        cache.MarkNotCapable();
        Assert.Null(await cache.GetAsync("m", CancellationToken.None));
        Assert.Null(await cache.GetAsync("other", CancellationToken.None));
        time.Advance(TimeSpan.FromSeconds(31));
        Assert.NotNull(await cache.GetAsync("m", CancellationToken.None));
    }

    /// <summary><see cref="ChunkCacheBodyHandler.FallBackOnChunkError"/> off: the refusal reaches the caller; nothing is resent.</summary>
    [Fact]
    public async Task FallbackOff_RefusalReachesCaller()
    {
        string text = Text('f');
        var capture = new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.BadRequest)
        { Content = new StringContent("{\"error\":{\"code\":\"chunk_hash_mismatch\"}}", Encoding.UTF8, "application/json") });
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture, FallBackOnChunkError = false });
        using HttpResponseMessage r = await invoker.SendAsync(Pipeline.Post(Pipeline.StringBody(("tool", text))), CancellationToken.None);

        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        Assert.Single(capture.Seen);
        Assert.False(scope.Outcome!.FellBackToPlain);
    }

    // ---- S6 and the probe-context nit --------------------------------------------------------------------------------

    /// <summary>Two models on one server, alternating: one probe per model within the TTL.</summary>
    [Fact]
    public async Task Cache_AlternatingModels_OneProbeEach()
    {
        var probe = new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        var cache = new ChunkCapabilityCache(probe);
        foreach (string m in new[] { "a", "b", "a", "b", "a", "b" })
            Assert.NotNull(await cache.GetAsync(m, CancellationToken.None));
        Assert.Equal(2, probe.Calls);
    }

    /// <summary>The model map is bounded: the least recently probed model is dropped first.</summary>
    [Fact]
    public async Task Cache_ModelMap_Bounded()
    {
        var probe = new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        var cache = new ChunkCapabilityCache(probe);
        for (int i = 0; i <= ChunkCapabilityCache.MaxModels; i++)
            await cache.GetAsync("m" + i, CancellationToken.None);
        Assert.Equal(ChunkCapabilityCache.MaxModels + 1, probe.Calls);
        await cache.GetAsync("m" + ChunkCapabilityCache.MaxModels, CancellationToken.None); // still cached
        Assert.Equal(ChunkCapabilityCache.MaxModels + 1, probe.Calls);
        await cache.GetAsync("m0", CancellationToken.None);                                   // dropped: probed again
        Assert.Equal(ChunkCapabilityCache.MaxModels + 2, probe.Calls);
    }

    /// <summary>The shared probe runs without the first caller's execution context: no chunk scope, no Activity parent.</summary>
    [Fact]
    public async Task Cache_ProbeDoesNotInheritCallerContext()
    {
        Activity? seenActivity = new("sentinel");
        ChunkManifestScope? seenScope = null;
        bool ran = false;
        var probe = new ScriptedProbe((_, _) =>
        {
            ran = true;
            seenActivity = Activity.Current;
            seenScope = ChunkManifestScope.Current;
            return Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps());
        });
        var cache = new ChunkCapabilityCache(probe);
        using var activity = new Activity("caller").Start();
        using var scope = ChunkManifestScope.Begin([], new ChunkRequestOptions());

        Assert.NotNull(await cache.GetAsync("m", CancellationToken.None));

        Assert.True(ran);
        Assert.Null(seenActivity);
        Assert.Null(seenScope);
        Assert.Same(activity, Activity.Current); // the caller's own context is unchanged
    }

    // ---- S2: SSE scanner ---------------------------------------------------------------------------------------------

    private static (List<JsonObject> Reports, SseChunkCacheScanner Scanner, byte[] Out) ScanWithZeroReads(string sse, int chunk)
    {
        var reports = new List<JsonObject>();
        var scanner = new SseChunkCacheScanner(new Trickle(Encoding.UTF8.GetBytes(sse), chunk), reports.Add);
        var ms = new MemoryStream();
        var buf = new byte[chunk];
        while (true)
        {
            Assert.Equal(0, scanner.Read([]));          // a zero-byte read between every real read
            int n = scanner.Read(buf, 0, buf.Length);
            if (n == 0) break;
            ms.Write(buf, 0, n);
        }

        return (reports, scanner, ms.ToArray());
    }

    /// <summary>A zero-byte read is not end of stream: a report split across it is still found, and so is an unterminated last line.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(64)]
    public void Scanner_ZeroByteReads_NotEndOfStream(int chunk)
    {
        string final = "data: " + Fixtures.Json("response-stream-final.json").ToJsonString();
        (List<JsonObject> reports, _, byte[] output) = ScanWithZeroReads("data: {\"x\":1}\n\n" + final + "\n\ndata: [DONE]\n\n", chunk);
        Assert.Single(reports);
        Assert.Contains("[DONE]", Encoding.UTF8.GetString(output), StringComparison.Ordinal);

        (List<JsonObject> tail, _, _) = ScanWithZeroReads("data: {\"x\":1}\n\n" + final, chunk);
        Assert.Single(tail);
    }

    /// <summary>Bare <c>\r</c> line ends (allowed by the SSE spec), and <c>\r\n</c> split across reads, are one line end each.</summary>
    [Theory]
    [InlineData("\r", 1)]
    [InlineData("\r", 7)]
    [InlineData("\r\n", 1)]
    [InlineData("\r\n", 3)]
    public void Scanner_CrLineEnds(string eol, int chunk)
    {
        string sse = "event: message_delta" + eol + "data: " + Fixtures.Json("response-anthropic-delta.json").ToJsonString() + eol + eol
                     + "data: {\"type\":\"message_stop\"}" + eol + eol;
        (List<JsonObject> reports, SseChunkCacheScanner scanner, _) = ScanWithZeroReads(sse, chunk);
        Assert.True(JsonNode.DeepEquals(Fixtures.Json("response-anthropic-delta.json")["chunk_cache"], Assert.Single(reports)));
        Assert.Equal(1, scanner.ParsedEvents);
    }

    /// <summary>A line over <see cref="SseChunkCacheScanner.MaxLineBytes"/> is skipped (and counted); the next line is still parsed.</summary>
    [Fact]
    public void Scanner_OversizedLine_SkippedNextParsed()
    {
        string huge = "data: {\"chunk_cache\":{\"v\":1},\"pad\":\"" + new string('p', SseChunkCacheScanner.MaxLineBytes) + "\"}\n\n";
        string final = "data: " + Fixtures.Json("response-stream-final.json").ToJsonString() + "\n\n";
        var reports = new List<JsonObject>();
        using var scanner = new SseChunkCacheScanner(new MemoryStream(Encoding.UTF8.GetBytes(huge + final)), reports.Add);
        scanner.CopyTo(Stream.Null, 8192);

        Assert.Equal(1, scanner.SkippedLines);
        Assert.True(JsonNode.DeepEquals(Fixtures.Json("response-stream-final.json")["chunk_cache"], Assert.Single(reports)));
    }

    // ---- S7: responses are disposed on every exit path ---------------------------------------------------------------

    /// <summary>When reading the response body for the report (or the error code) throws, the response is disposed and the error propagates.</summary>
    [Theory]
    [InlineData(200)]
    [InlineData(400)]
    public async Task ReadFailure_DisposesResponse(int status)
    {
        string text = Text('d');
        var content = new ThrowingContent();
        var capture = new CaptureHandler((_, _) => new HttpResponseMessage((HttpStatusCode)status) { Content = content });
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });

        async Task Send()
        {
            using HttpResponseMessage _ = await invoker.SendAsync(Pipeline.Post(Pipeline.StringBody(("tool", text))), CancellationToken.None);
        }

        Exception ex = await Assert.ThrowsAnyAsync<Exception>(Send);
        Assert.True(ex is IOException || ex.InnerException is IOException, ex.ToString());

        Assert.True(content.Disposed);
    }

    /// <summary>A blocking JSON body over <see cref="ChunkCacheBodyHandler.MaxReportBodyBytes"/> is not read for the report.</summary>
    [Fact]
    public async Task OversizedJsonBody_NotReadForReport()
    {
        string text = Text('o');
        byte[] big = Encoding.UTF8.GetBytes("{\"chunk_cache\":{\"v\":1},\"pad\":\"" + new string('p', (int)ChunkCacheBodyHandler.MaxReportBodyBytes) + "\"}");
        var capture = new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(big) { Headers = { ContentType = new("application/json") } } });
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using HttpResponseMessage _ = await invoker.SendAsync(Pipeline.Post(Pipeline.StringBody(("tool", text))), CancellationToken.None);

        Assert.True(scope.Outcome!.Annotated);
        Assert.Null(scope.ResponseReport);
    }

    // ---- S8: Anthropic tool_result is not located in v1 --------------------------------------------------------------

    /// <summary>
    /// A registered text inside an Anthropic <c>tool_result</c> block is not located (v1 limitation, ruling 19's converter
    /// mapping is the server's): the body goes out untouched and the spans count as not located.
    /// </summary>
    [Fact]
    public async Task AnthropicToolResult_Untouched_NotLocated()
    {
        string text = Text('a');
        byte[] input = Encoding.UTF8.GetBytes(new JsonObject
        {
            ["model"] = "m", ["max_tokens"] = 16,
            ["messages"] = new JsonArray(new JsonObject
            {
                ["role"] = "user",
                ["content"] = new JsonArray(new JsonObject
                {
                    ["type"] = "tool_result", ["tool_use_id"] = "t1",
                    ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = text }),
                }),
            }),
        }.ToJsonString());
        var capture = CaptureHandler.Ok();
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using HttpResponseMessage _ = await invoker.SendAsync(Pipeline.Post(input, "http://dotllm.test/v1/messages"), CancellationToken.None);

        Assert.Equal(input, capture.LastBody);
        Assert.Equal(ChunkSkipReasons.NoSpans, scope.Outcome!.SkipReason);
        Assert.Equal(1, scope.Outcome.NotLocated);
    }

    // ---- S9: missing tests ------------------------------------------------------------------------------------------

    /// <summary>An encoded request body (Content-Encoding) is never parsed or annotated.</summary>
    [Fact]
    public async Task ContentEncodedBody_Untouched()
    {
        string text = Text('g');
        byte[] input = Pipeline.StringBody(("tool", text));
        HttpRequestMessage request = Pipeline.Post(input);
        request.Content!.Headers.ContentEncoding.Add("gzip");
        var capture = CaptureHandler.Ok();
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using HttpResponseMessage _ = await invoker.SendAsync(request, CancellationToken.None);

        Assert.Equal(input, capture.LastBody);
        Assert.Equal(ChunkSkipReasons.NotJson, scope.Outcome!.SkipReason);
    }

    /// <summary>A BOM before the body and whitespace / CRLF after the closing brace survive the splice byte for byte.</summary>
    [Fact]
    public async Task BomAndTrailingWhitespace_SplicedBeforeTheBrace()
    {
        string text = Text('b');
        byte[] json = Pipeline.StringBody(("tool", text));
        byte[] input = [0xEF, 0xBB, 0xBF, .. json, (byte)' ', (byte)'\r', (byte)'\n', (byte)'\t'];
        var capture = CaptureHandler.Ok();
        using var scope = ChunkManifestScope.Begin(PartsOf(text, "bom"), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using HttpResponseMessage _ = await invoker.SendAsync(Pipeline.Post(input), CancellationToken.None);

        string manifest = Pipeline.SplicedManifest(input, capture.LastBody);
        Assert.Equal("bom", (string)JsonNode.Parse(manifest)!["chunks"]![0]!["id"]!);
        Assert.Equal(new byte[] { 0xEF, 0xBB, 0xBF }, capture.LastBody[..3]);
        Assert.EndsWith("} \r\n\t", Encoding.UTF8.GetString(capture.LastBody), StringComparison.Ordinal);
    }

    /// <summary>Only POST is rewritten, even on the chat path.</summary>
    [Theory]
    [InlineData("PUT")]
    [InlineData("PATCH")]
    public async Task NonPostOnChatPath_Untouched(string method)
    {
        string text = Text('p');
        byte[] input = Pipeline.StringBody(("tool", text));
        HttpRequestMessage request = Pipeline.Post(input);
        request.Method = new HttpMethod(method);
        var capture = CaptureHandler.Ok();
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using HttpResponseMessage _ = await invoker.SendAsync(request, CancellationToken.None);

        Assert.Equal(input, capture.LastBody);
        Assert.Null(scope.Outcome);
    }

    /// <summary>
    /// Two concurrent decorated calls with different registered parts on one shared HttpClient: each request carries only
    /// its own manifest (the AsyncLocal scope is per call), with both requests in flight at once.
    /// </summary>
    [Fact]
    public async Task ConcurrentCalls_SharedHttpClient_EachOwnManifest()
    {
        string textA = Text('A'), textB = Text('B');
        var regA = new ChunkSpanRegistry();
        regA.Register(textA, [new ChunkSpan("only-A", ChunkHash.Of(textA))]);
        var regB = new ChunkSpanRegistry();
        regB.Register(textB, [new ChunkSpan("only-B", ChunkHash.Of(textB))]);
        var bothArrived = new CountdownEvent(2);
        var terminal = new BarrierHandler(bothArrived);
        using HttpClient http = Pipeline.Client(terminal, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var clientA = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), regA, ExactPolicy.Instance);
        using var clientB = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), regB, ExactPolicy.Instance);

        Task<ChatResponse> a = clientA.GetResponseAsync([new ChatMessage(ChatRole.User, "q"), new ChatMessage(ChatRole.User, textA), new ChatMessage(ChatRole.User, textB)]);
        Task<ChatResponse> b = clientB.GetResponseAsync([new ChatMessage(ChatRole.User, "q"), new ChatMessage(ChatRole.User, textA), new ChatMessage(ChatRole.User, textB)]);
        await Task.WhenAll(a, b).WaitAsync(s_wait);

        string[] ids = terminal.Bodies.Select(body => string.Join(",", JsonNode.Parse(body)!["chunk_cache"]!["chunks"]!.AsArray().Select(c => (string)c!["id"]!)))
            .Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(["only-A", "only-B"], ids);
    }

    // ---- nits --------------------------------------------------------------------------------------------------------

    /// <summary><c>CallCompleted</c> fires when the call throws (a strict refusal surfaces as an SDK exception), with the outcome.</summary>
    [Fact]
    public async Task CallCompleted_FiresWhenTheCallThrows()
    {
        string text = Text('t');
        var registry = new ChunkSpanRegistry();
        registry.Register(text, [new ChunkSpan("t", ChunkHash.Of(text))]);
        var capture = new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.BadRequest)
        { Content = new StringContent("{\"error\":{\"message\":\"m\",\"type\":\"invalid_request_error\",\"code\":\"chunk_hash_mismatch\"}}", Encoding.UTF8, "application/json") });
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry,
            new StrictPolicy());
        ChunkCacheCallEventArgs? seen = null;
        client.CallCompleted += (_, e) => seen = e;

        await Assert.ThrowsAnyAsync<Exception>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, text)]));

        Assert.NotNull(seen);
        Assert.True(seen!.Options.Strict);
        Assert.Equal(1, seen.Outcome!.SentSpans);
    }

    /// <summary><c>CallCompleted</c> fires when the caller abandons a stream after the first update.</summary>
    [Fact]
    public async Task CallCompleted_FiresWhenStreamAbandoned()
    {
        string text = Text('s');
        var registry = new ChunkSpanRegistry();
        registry.Register(text, [new ChunkSpan("s", ChunkHash.Of(text))]);
        using var client = new ChunkAnnotatingChatClient(new EndlessClient(), registry, ExactPolicy.Instance);
        int events = 0;
        client.CallCompleted += (_, _) => events++;

        await foreach (ChatResponseUpdate _ in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, text)]))
            break;

        Assert.Equal(1, events);
    }

    /// <summary>
    /// A resend whose content was replaced above this handler since the first attempt (a body rewriter re-wrapping it)
    /// keeps that new content: the stored original is restored only over this handler's own rewrite.
    /// </summary>
    [Fact]
    public async Task Resend_AfterUpstreamRewrite_KeepsNewContent()
    {
        string text = Text('u');
        var capture = CaptureHandler.Ok();
        using var scope = ChunkManifestScope.Begin(PartsOf(text), new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        HttpRequestMessage request = Pipeline.Post(Pipeline.StringBody(("tool", text)));

        using (await invoker.SendAsync(request, CancellationToken.None)) { }
        var edited = new JsonObject { ["model"] = Pipeline.Model, ["max_tokens"] = 7, ["messages"] = new JsonArray(new JsonObject { ["role"] = "tool", ["content"] = text }) };
        request.Content = new StringContent(edited.ToJsonString(), Encoding.UTF8, "application/json");
        using (await invoker.SendAsync(request, CancellationToken.None)) { }

        JsonObject second = JsonNode.Parse(capture.Seen[1].Body!)!.AsObject();
        Assert.Equal(7, (int)second["max_tokens"]!);
        Assert.NotNull(second["chunk_cache"]);
    }

    // ---- helpers -----------------------------------------------------------------------------------------------------

    private sealed class StrictPolicy : IRecomputePolicy
    {
        public ChunkRequestOptions Decide(ChunkRequestContext context) => new(Strict: true);
    }

    private sealed class EndlessClient : IChatClient
    {
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            while (true)
            {
                await Task.Yield();
                yield return new ChatResponseUpdate(ChatRole.Assistant, "x");
            }
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    /// <summary>Answers only once both requests have arrived, so the two calls are provably in flight together.</summary>
    private sealed class BarrierHandler(CountdownEvent arrived) : HttpMessageHandler
    {
        public List<byte[]> Bodies { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            byte[] body = await request.Content!.ReadAsByteArrayAsync(cancellationToken);
            lock (Bodies) Bodies.Add(body);
            arrived.Signal();
            await Task.Run(() => arrived.Wait(s_wait), cancellationToken);
            Assert.True(arrived.IsSet, "both requests were not in flight together");
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Fixtures.Bytes("response-blocking.json")) { Headers = { ContentType = new("application/json") } } };
        }
    }

    private sealed class ThrowingContent : HttpContent
    {
        public ThrowingContent() => Headers.ContentType = new("application/json");

        public bool Disposed { get; private set; }

        protected override Task SerializeToStreamAsync(Stream stream, TransportContext? context) => throw new IOException("connection reset");

        protected override bool TryComputeLength(out long length)
        {
            length = 0;
            return false;
        }

        protected override void Dispose(bool disposing)
        {
            Disposed = true;
            base.Dispose(disposing);
        }
    }

    private sealed class Trickle(byte[] data, int size) : Stream
    {
        private int _pos;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => data.Length;

        public override long Position { get => _pos; set => throw new NotSupportedException(); }

        public override void Flush() { }

        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(Math.Min(count, size), data.Length - _pos);
            Array.Copy(data, _pos, buffer, offset, n);
            _pos += n;
            return n;
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
