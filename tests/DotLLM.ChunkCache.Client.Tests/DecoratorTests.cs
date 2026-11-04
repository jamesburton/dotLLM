using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Xunit;

// Test code: handlers, requests and helpers are handed to HttpClient / HttpMessageInvoker chains that each test
// disposes (ownership the analyzer cannot follow), and continuations may resume on xUnit's context.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// <see cref="ChunkAnnotatingChatClient"/> over the real MEAI OpenAI bridge, so the body under test is the one the provider
/// actually serialises (escapes, content shapes, its own request content type), and the AsyncLocal hand-off crosses the
/// OpenAI SDK pipeline and <see cref="HttpClient"/> (spec open question 1, for a plain HttpClient stack).
/// </summary>
public sealed class DecoratorTests
{
    private static readonly TimeSpan s_wait = TimeSpan.FromSeconds(15);

    /// <summary>A file-read result with non-ASCII and a surrogate pair (the fixture part), as two adjacent chunks.</summary>
    private static (string Text, ChunkSpan[] Spans) FileRead()
    {
        string part = Pipeline.PartText(Fixtures.Json("nonascii-crlf.json"), 3, 1);
        int s = Pipeline.Utf16Of(part, 35), e = Pipeline.Utf16Of(part, 374);
        Assert.Equal(part.Length, e); // the fixture's span runs to the end of the part
        return (part, [new ChunkSpan("docs/notes-fr.md:1-12@0", ChunkHash.Of(part[s..e]), s, e), new ChunkSpan("head", ChunkHash.Of(part[..s]), 0, s)]);
    }

    private static List<ChatMessage> Conversation(string toolResult) =>
    [
        new(ChatRole.System, "You are a coding agent. Use tools to read files."),
        new(ChatRole.User, "Why does ChunkIndex.Update drop entries? (é)"),
        new(ChatRole.Assistant, [new FunctionCallContent("call_7", "read_file", new Dictionary<string, object?>(StringComparer.Ordinal) { ["path"] = "docs/notes-fr.md" })]),
        new(ChatRole.Tool, [new FunctionResultContent("call_7", toolResult)]),
    ];

    private static CaptureHandler Responding(string fixture) =>
        new((_, _) => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Fixtures.Bytes(fixture)) { Headers = { ContentType = new("application/json") } } });

    /// <summary>
    /// Blocking: the manifest is computed from the bridge's own serialised body (string content is part 0), offsets are
    /// UTF-8 bytes of the decoded text, and the response's report is lifted into
    /// <see cref="ChatResponse.AdditionalProperties"/> and the <see cref="ChunkAnnotatingChatClient.CallCompleted"/> event.
    /// </summary>
    [Fact]
    public async Task Blocking_ManifestFromRealSerializer_ReportInAdditionalProperties()
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);
        CaptureHandler capture = Responding("response-blocking.json");
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance);
        ChunkCacheCallEventArgs? seen = null;
        client.CallCompleted += (_, e) => seen = e;

        ChatResponse response = await client.GetResponseAsync(Conversation(text));

        JsonObject body = JsonNode.Parse(capture.LastBody)!.AsObject();
        Assert.Equal(text, (string)body["messages"]![3]!["content"]!); // the bridge sends tool content as a string
        JsonArray chunks = body["chunk_cache"]!["chunks"]!.AsArray();
        Assert.Equal("{\"message\":3,\"part\":0,\"start\":0,\"end\":35}", chunks[0]!["at"]!.ToJsonString());
        Assert.Equal("{\"message\":3,\"part\":0,\"start\":35,\"end\":374}", chunks[1]!["at"]!.ToJsonString());

        Assert.True(ChunkCacheReport.TryGet(response, out JsonObject? report));
        Assert.True(JsonNode.DeepEquals(Fixtures.Json("response-blocking.json")["chunk_cache"], report));
        Assert.NotNull(seen);
        Assert.Equal(2, seen!.Outcome!.SentSpans);
        Assert.Same(report, seen.Report);
        Assert.Null(ChunkManifestScope.Current);
    }

    /// <summary>
    /// Streaming over a real socket: tokens flow before the server writes its final chunk, the report arrives on an
    /// update (not a buffered response), and <c>ToChatResponse</c> carries it.
    /// </summary>
    [Fact]
    public async Task Streaming_ReportOnUpdate_NotBuffered()
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);
        var firstTokenSeen = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        string final = Fixtures.Json("response-stream-final.json").ToJsonString();
        await using var server = new FakeHttpServer(async (req, s, ct) =>
        {
            await FakeHttpServer.WriteSseHeadAsync(s, ct);
            await FakeHttpServer.WriteRawAsync(s, "data: {\"id\":\"chatcmpl-0002\",\"object\":\"chat.completion.chunk\",\"created\":1793577601,\"model\":\"qwen3-4b\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n", ct);
            await firstTokenSeen.Task.WaitAsync(s_wait, ct);
            await FakeHttpServer.WriteRawAsync(s, "data: " + final + "\n\ndata: [DONE]\n\n", ct);
        });
        using HttpClient http = Pipeline.Client(new SocketsHttpHandler(), new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, server.BaseUri), registry, ExactPolicy.Instance);

        var updates = new List<ChatResponseUpdate>();
        await foreach (ChatResponseUpdate u in client.GetStreamingResponseAsync(Conversation(text)).WithCancellation(new CancellationTokenSource(s_wait).Token))
        {
            updates.Add(u);
            if (string.Equals(u.Text, "Hello", StringComparison.Ordinal))
            {
                Assert.False(ChunkCacheReport.TryGet(u, out _));
                firstTokenSeen.TrySetResult();
            }
        }

        JsonObject expected = Fixtures.Json("response-stream-final.json")["chunk_cache"]!.AsObject();
        ChatResponseUpdate carrier = Assert.Single(updates, u => ChunkCacheReport.TryGet(u, out _));
        Assert.True(ChunkCacheReport.TryGet(carrier, out JsonObject? onUpdate));
        Assert.True(JsonNode.DeepEquals(expected, onUpdate));
        Assert.True(ChunkCacheReport.TryGet(updates.ToChatResponse(), out JsonObject? merged));
        Assert.True(JsonNode.DeepEquals(expected, merged));
        Assert.Equal("Hello", updates.ToChatResponse().Text);
        Assert.Contains("\"chunk_cache\"", Assert.Single(server.Snapshot).BodyText, StringComparison.Ordinal);
    }

    /// <summary>
    /// A retrying chat client above the decorator (CodeSharp's placement: decorator below <c>Retrying</c>) and a retrying
    /// handler on the HttpClient both resend the same manifest.
    /// </summary>
    [Fact]
    public async Task Retry_AboveDecorator_AndInHttpClient_SameManifest()
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);
        int n = 0;
        var capture = new CaptureHandler((_, _) => Interlocked.Increment(ref n) % 2 == 1
            ? new HttpResponseMessage(HttpStatusCode.ServiceUnavailable)
            : new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Fixtures.Bytes("response-blocking.json")) { Headers = { ContentType = new("application/json") } } });
        using HttpClient http = Pipeline.Client(capture, new RetryOn503Handler(1, TimeSpan.FromMilliseconds(10)), new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var decorated = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance);
        using var retrying = new RetryingChatClient(decorated, attempts: 2, failFirst: true);

        ChatResponse response = await retrying.GetResponseAsync(Conversation(text));

        Assert.Equal(4, capture.Seen.Count); // two decorator calls, each retried once below
        Assert.All(capture.Seen, s => Assert.Equal(capture.Seen[0].Body, s.Body));
        Assert.Contains("\"chunk_cache\"", Encoding.UTF8.GetString(capture.Seen[0].Body!), StringComparison.Ordinal);
        Assert.True(ChunkCacheReport.TryGet(response, out _));
    }

    /// <summary>
    /// Without capabilities (fixed null, a 401 probe, a probe timeout) the decorated pipeline sends exactly the bytes of
    /// an undecorated one, even though registered spans would match.
    /// </summary>
    [Theory]
    [InlineData("fixed-null")]
    [InlineData("probe-401")]
    [InlineData("probe-timeout")]
    [InlineData("probe-404")]
    public async Task NotCapable_BodyIdenticalToUndecoratedPipeline(string how)
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);

        CaptureHandler plainCapture = Responding("response-blocking.json");
        using (HttpClient plainHttp = Pipeline.Client(plainCapture))
            await Pipeline.OpenAiChat(plainHttp, new Uri("http://dotllm.test/")).GetResponseAsync(Conversation(text));

        IChunkCapabilitySource source = how switch
        {
            "fixed-null" => new FixedChunkCapabilities(null),
            "probe-401" => new ChunkCapabilityCache(new ScriptedProbe((_, _) => throw new ChunkCacheRequestException(HttpStatusCode.Unauthorized, null, null, "401"))),
            "probe-404" => new ChunkCapabilityCache(new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(null))),
            _ => new ChunkCapabilityCache(new ScriptedProbe(async (_, ct) => { await Task.Delay(Timeout.Infinite, ct); return null; }), probeTimeout: TimeSpan.FromMilliseconds(100)),
        };
        CaptureHandler capture = Responding("response-blocking.json");
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(source));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance, new ChunkAnnotatingOptions { Capabilities = source });

        ChatResponse response = await client.GetResponseAsync(Conversation(text));

        Assert.Equal(plainCapture.LastBody, capture.LastBody);
        Assert.False(ChunkCacheReport.TryGet(response, out _));
    }

    /// <summary>Nothing registered: no scope, and the body equals an undecorated pipeline's.</summary>
    [Fact]
    public async Task NothingRegistered_BodyIdentical()
    {
        CaptureHandler plainCapture = Responding("response-blocking.json");
        using (HttpClient plainHttp = Pipeline.Client(plainCapture))
            await Pipeline.OpenAiChat(plainHttp, new Uri("http://dotllm.test/")).GetResponseAsync(Conversation("plain tool output"));

        CaptureHandler capture = Responding("response-blocking.json");
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), new ChunkSpanRegistry(), ExactPolicy.Instance);
        int events = 0;
        client.CallCompleted += (_, _) => events++;

        await client.GetResponseAsync(Conversation("plain tool output"));

        Assert.Equal(plainCapture.LastBody, capture.LastBody);
        Assert.Equal(0, events);
    }

    /// <summary>
    /// Nothing lossy by default: a policy returning <c>balanced</c> without the user's opt-in is sent as <c>exact</c>
    /// (omitted, the default); with the opt-in it is sent as asked. <c>shadow</c> needs no opt-in.
    /// </summary>
    [Theory]
    [InlineData("balanced", false, null)]
    [InlineData("fast", false, null)]
    [InlineData("auto", false, null)]
    [InlineData("balanced", true, "balanced")]
    [InlineData("shadow", false, "shadow")]
    [InlineData("exact", false, null)]
    public async Task LossyPolicy_RequiresOptIn(string decided, bool optedIn, string? onWire)
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);
        CaptureHandler capture = Responding("response-blocking.json");
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        var policy = new FixedPolicy(new ChunkRequestOptions(Policy: decided));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry, policy,
            new ChunkAnnotatingOptions { SessionFlags = _ => new ChunkSessionFlags(false, false, optedIn), Capabilities = new FixedChunkCapabilities(Pipeline.Caps()) });

        await client.GetResponseAsync(Conversation(text), new ChatOptions { ModelId = Pipeline.Model });

        JsonNode? sent = JsonNode.Parse(capture.LastBody)!["chunk_cache"]!["policy"];
        Assert.Equal(onWire, sent is null ? null : (string)sent!);
        Assert.Equal(optedIn, policy.Seen!.UserOptedIntoApproximation);
        Assert.NotNull(policy.Seen.Server);
        Assert.Equal(Pipeline.Model, policy.Seen.ModelId);
        Assert.True(policy.Seen.ChunkTokens > 0 && policy.Seen.PromptTokensEstimate >= policy.Seen.ChunkTokens);
    }

    /// <summary>Two stacked decorators annotate once, exactly as one does (idempotent).</summary>
    [Fact]
    public async Task StackedDecorators_SameAsOne()
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);

        CaptureHandler one = Responding("response-blocking.json");
        using (HttpClient h1 = Pipeline.Client(one, new ChunkCacheBodyHandler(Pipeline.Caps())))
        using (var c1 = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(h1, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance))
            await c1.GetResponseAsync(Conversation(text));

        CaptureHandler two = Responding("response-blocking.json");
        using HttpClient h2 = Pipeline.Client(two, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var inner = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(h2, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance);
        using var outer = new ChunkAnnotatingChatClient(inner, registry, new FixedPolicy(new ChunkRequestOptions(Policy: "shadow")));
        ChatResponse r = await outer.GetResponseAsync(Conversation(text));

        // The outermost decorator's scope wins; the inner one passes through, so there is one manifest with one policy.
        JsonObject manifest = JsonNode.Parse(two.LastBody)!["chunk_cache"]!.AsObject();
        Assert.Equal("shadow", (string)manifest["policy"]!);
        JsonObject single = JsonNode.Parse(one.LastBody)!["chunk_cache"]!.AsObject();
        Assert.True(JsonNode.DeepEquals(single["chunks"], manifest["chunks"]));
        Assert.True(ChunkCacheReport.TryGet(r, out _));
    }

    /// <summary>A hook-appended suffix: the registry finds the registered prefix and the spans survive.</summary>
    [Fact]
    public async Task AppendedSuffix_FoundThroughRegistryPrefix()
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);
        CaptureHandler capture = Responding("response-blocking.json");
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance);

        await client.GetResponseAsync(Conversation(text + "\n[PostToolUse] additional context"));

        JsonArray chunks = JsonNode.Parse(capture.LastBody)!["chunk_cache"]!["chunks"]!.AsArray();
        Assert.Equal(2, chunks.Count);
        Assert.Equal(Encoding.UTF8.GetByteCount(text), (int)chunks[1]!["at"]!["end"]!);
    }

    /// <summary>
    /// Probe flapping through the whole pipeline: capable, then failing (annotations stripped, body plain), then capable
    /// again after the failure TTL.
    /// </summary>
    [Fact]
    public async Task ProbeFlapping_AnnotationsOnOffOn()
    {
        (string text, ChunkSpan[] spans) = FileRead();
        var registry = new ChunkSpanRegistry();
        registry.Register(text, spans);
        var time = new ManualTime();
        var probe = new ScriptedProbe((call, _) => call == 2
            ? throw new HttpRequestException("connection refused")
            : Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        var cache = new ChunkCapabilityCache(probe, successTtl: TimeSpan.FromMinutes(1), failureTtl: TimeSpan.FromSeconds(10), timeProvider: time);
        CaptureHandler capture = Responding("response-blocking.json");
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(cache));
        using var client = new ChunkAnnotatingChatClient(Pipeline.OpenAiChat(http, new Uri("http://dotllm.test/")), registry, ExactPolicy.Instance, new ChunkAnnotatingOptions { Capabilities = cache });

        var annotated = new List<bool>();
        for (int i = 0; i < 4; i++)
        {
            await client.GetResponseAsync(Conversation(text));
            annotated.Add(Encoding.UTF8.GetString(capture.LastBody).Contains("\"chunk_cache\"", StringComparison.Ordinal));
            time.Advance(i == 0 ? TimeSpan.FromMinutes(2) : TimeSpan.FromSeconds(i == 1 ? 4 : 7));
        }

        // probe 1 ok; TTL passes; probe 2 fails (cached 10 s); 4 s later still failed; 11 s after it probe 3 ok.
        Assert.Equal([true, false, false, true], annotated);
        Assert.Equal(3, probe.Calls);
    }

    /// <summary>
    /// Streaming: the scope is visible to the inner client on every step, not only the first (an async iterator resumes
    /// on its caller's context, where the AsyncLocal is not set), and a report that arrives after the last update is
    /// carried on one extra, content-free update.
    /// </summary>
    [Fact]
    public async Task Streaming_ScopeOnEveryStep_LateReportCarried()
    {
        string text = new('s', 80);
        var registry = new ChunkSpanRegistry();
        registry.Register(text, [new ChunkSpan("s", ChunkHash.Of(text))]);
        var inner = new SteppingClient();
        using var client = new ChunkAnnotatingChatClient(inner, registry, ExactPolicy.Instance);

        var updates = new List<ChatResponseUpdate>();
        await foreach (ChatResponseUpdate u in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.Tool, [new FunctionResultContent("c", text)])]))
            updates.Add(u);

        Assert.Equal(3, inner.ScopesSeen.Count);
        Assert.All(inner.ScopesSeen, Assert.NotNull);
        Assert.Equal(4, updates.Count);
        Assert.True(ChunkCacheReport.TryGet(updates[^1], out JsonObject? report));
        Assert.Equal(1, (int)report!["v"]!);
        Assert.Equal("r1", updates[^1].ResponseId);
        Assert.Empty(updates[^1].Contents);
        Assert.Equal("abc", updates.ToChatResponse().Text);
    }

    /// <summary>
    /// With a capability source that says "not capable", the decorator opens no scope at all (the inner client never sees
    /// one, no event), independently of the body handler's own gate.
    /// </summary>
    [Fact]
    public async Task NotCapableSource_NoScopeOpened()
    {
        string text = new('n', 80);
        var registry = new ChunkSpanRegistry();
        registry.Register(text, [new ChunkSpan("n", ChunkHash.Of(text))]);
        var inner = new SteppingClient();
        using var client = new ChunkAnnotatingChatClient(inner, registry, ExactPolicy.Instance, new ChunkAnnotatingOptions { Capabilities = new FixedChunkCapabilities(null) });
        int events = 0;
        client.CallCompleted += (_, _) => events++;

        await foreach (ChatResponseUpdate _ in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.Tool, [new FunctionResultContent("c", text)])])) { }
        await client.GetResponseAsync([new ChatMessage(ChatRole.Tool, [new FunctionResultContent("c", text)])]);

        Assert.Equal(4, inner.ScopesSeen.Count);
        Assert.All(inner.ScopesSeen, Assert.Null);
        Assert.Equal(0, events);
    }

    /// <summary>An inner streaming client that records the scope on each step and sets the report after its last update.</summary>
    private sealed class SteppingClient : IChatClient
    {
        public List<ChunkManifestScope?> ScopesSeen { get; } = [];

        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            ScopesSeen.Add(ChunkManifestScope.Current);
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "ok")));
        }

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (string piece in new[] { "a", "b", "c" })
            {
                await Task.Yield();
                ScopesSeen.Add(ChunkManifestScope.Current);
                yield return new ChatResponseUpdate(ChatRole.Assistant, piece) { ResponseId = "r1", MessageId = "m1" };
            }

            ChunkManifestScope.Current?.SetReport(new JsonObject { ["v"] = 1 });
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    private sealed class FixedPolicy(ChunkRequestOptions options) : IRecomputePolicy
    {
        public ChunkRequestContext? Seen { get; private set; }

        public ChunkRequestOptions Decide(ChunkRequestContext context)
        {
            Seen = context;
            return options;
        }
    }

    /// <summary>Calls the inner client again when the first call throws (here: the first call's 503 surfaced as an exception is simulated).</summary>
    private sealed class RetryingChatClient(IChatClient inner, int attempts, bool failFirst) : DelegatingChatClient(inner)
    {
        public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            ChatResponse? last = null;
            List<ChatMessage> list = messages.ToList();
            for (int i = 0; i < attempts; i++)
            {
                last = await base.GetResponseAsync(list, options, cancellationToken);
                if (!(failFirst && i == 0)) break; // pretend the first answer was rejected by a higher-level check
            }

            return last!;
        }
    }
}
