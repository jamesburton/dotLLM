using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

// Test code: handlers, requests and helpers are handed to HttpClient / HttpMessageInvoker chains that each test
// disposes (ownership the analyzer cannot follow), and continuations may resume on xUnit's context.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// The body handler against the contract fixtures (plan Task 8 Step 1), driven with an explicit
/// <see cref="ChunkManifestScope"/>; the decorator's half is in <see cref="DecoratorTests"/>.
/// </summary>
public sealed class PipelineTests
{
    private static readonly TimeSpan s_wait = TimeSpan.FromSeconds(15);

    private static async Task<(byte[] Sent, ChunkManifestScope Scope, CaptureHandler Capture)> SendAsync(
        byte[] body, IReadOnlyList<ChunkPart> parts, ChunkRequestOptions? options = null, ChunkCacheCapabilities? caps = null,
        bool noCaps = false, string? path = null, CaptureHandler? capture = null)
    {
        capture ??= CaptureHandler.Ok();
        var scope = ChunkManifestScope.Begin(parts, options ?? new ChunkRequestOptions());
        try
        {
            using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(noCaps ? null : caps ?? Pipeline.Caps()));
            using HttpRequestMessage request = path is null ? Pipeline.Post(body) : Pipeline.Post(body, path);
            using HttpResponseMessage response = await http.SendAsync(request);
            return (capture.LastBody, scope, capture);
        }
        finally
        {
            scope.Dispose();
        }
    }

    // ---- plan-named tests ------------------------------------------------------------------------------------------

    /// <summary>
    /// The handler's manifest for the registered spans of <c>manifest.json</c> is the fixture's <c>chunk_cache</c>,
    /// byte for byte in compact form, and every other byte of the body is unchanged (the member is spliced in).
    /// </summary>
    [Fact]
    public async Task Manifest_MatchesRegisteredSpans()
    {
        JsonObject fixture = Fixtures.Json("manifest.json");
        byte[] input = Pipeline.FixtureWithoutManifest("manifest.json");
        string part1 = Pipeline.PartText(fixture, 3, 1);
        string part2 = Pipeline.PartText(fixture, 3, 2);
        int end2 = Pipeline.Utf16Of(part2, 1830);
        Assert.NotEqual(part2.Length, end2); // glue follows the span in the same part

        ChunkPart[] parts =
        [
            new(part1, [new ChunkSpan("src/Store/ChunkIndex.cs:365-412@14211", ChunkHash.Of(part1))]),
            new(part2, [new ChunkSpan("src/Store/ChunkRef.cs:1-40@0", ChunkHash.Of(part2[..end2]), 0, end2)]),
        ];

        (byte[] sent, ChunkManifestScope scope, CaptureHandler capture) = await SendAsync(input, parts, new ChunkRequestOptions(Policy: "shadow"));

        string manifest = Pipeline.SplicedManifest(input, sent);
        Assert.Equal(fixture["chunk_cache"]!.ToJsonString(), manifest);
        Assert.True(scope.Outcome!.Annotated);
        Assert.Equal(2, scope.Outcome.SentSpans);
        Assert.Equal("application/json; charset=utf-8", capture.Seen[^1].ContentType);
        Assert.Equal(sent.Length, capture.Seen[^1].ContentLength);
    }

    /// <summary>
    /// Review Focus: the contract body escapes non-ASCII (<c>é</c>, a surrogate pair) on the wire; the handler hashes
    /// the decoded text and still produces the fixture manifest exactly.
    /// </summary>
    [Fact]
    public async Task EscapedNonAscii_StillLocated()
    {
        byte[] input = Pipeline.FixtureWithoutManifest("nonascii-crlf.json");
        string raw = Encoding.UTF8.GetString(input);
        foreach (JsonNode? s in Fixtures.Json("expected.json")["requests"]!["nonascii-crlf.json"]!["raw_contains"]!.AsArray())
            Assert.Contains((string)s!, raw, StringComparison.Ordinal);

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, NonAsciiParts());

        Assert.Equal(Fixtures.Json("nonascii-crlf.json")["chunk_cache"]!.ToJsonString(), Pipeline.SplicedManifest(input, sent));
        Assert.Equal(2, scope.Outcome!.SentSpans);
    }

    /// <summary>The same body written by a serializer that does not escape non-ASCII gives the same manifest.</summary>
    [Fact]
    public async Task RawUtf8Body_SameManifestAsEscaped()
    {
        JsonObject doc = Fixtures.Json("nonascii-crlf.json");
        doc.Remove("chunk_cache");
        byte[] input = Encoding.UTF8.GetBytes(doc.ToJsonString(new JsonSerializerOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        Assert.Contains("é", Encoding.UTF8.GetString(input), StringComparison.Ordinal);

        (byte[] sent, _, _) = await SendAsync(input, NonAsciiParts());

        Assert.Equal(Fixtures.Json("nonascii-crlf.json")["chunk_cache"]!.ToJsonString(), Pipeline.SplicedManifest(input, sent));
    }

    /// <summary>
    /// Review Focus: a surrogate pair and other multi-byte characters precede the span, so UTF-16, code-point and UTF-8
    /// offsets all differ; the wire gets the UTF-8 byte offsets (35, 374) the fixture pins.
    /// </summary>
    [Fact]
    public async Task SurrogatePairBeforeSpan_Utf8OffsetsExact()
    {
        string part = Pipeline.PartText(Fixtures.Json("nonascii-crlf.json"), 3, 1);
        int start16 = Pipeline.Utf16Of(part, 35);
        int end16 = Pipeline.Utf16Of(part, 374);
        Assert.NotEqual(35, start16);
        Assert.NotEqual(374, end16);
        Assert.NotEqual(start16, part[..start16].EnumerateRunes().Count()); // code points differ from UTF-16 too

        (byte[] sent, _, _) = await SendAsync(Pipeline.FixtureWithoutManifest("nonascii-crlf.json"), NonAsciiParts());

        JsonObject at = JsonNode.Parse(Pipeline.SplicedManifest(Pipeline.FixtureWithoutManifest("nonascii-crlf.json"), sent))!["chunks"]![0]!["at"]!.AsObject();
        Assert.Equal(35, (int)at["start"]!);
        Assert.Equal(374, (int)at["end"]!);
    }

    /// <summary>A part whose text was changed after registration yields no annotation, and the body is untouched.</summary>
    [Fact]
    public async Task MutatedPart_NoAnnotation()
    {
        string text = Pipeline.PartText(Fixtures.Json("manifest.json"), 3, 1);
        string mutated = text.Replace("seen", "Seen", StringComparison.Ordinal);
        Assert.NotEqual(text, mutated);
        byte[] input = Pipeline.StringBody(("user", "why?"), ("tool", mutated));

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan("a", ChunkHash.Of(text))])]);

        Assert.Equal(input, sent);
        Assert.False(scope.Outcome!.Annotated);
        Assert.Equal(ChunkSkipReasons.NoSpans, scope.Outcome.SkipReason);
        Assert.Equal(1, scope.Outcome.NotLocated);
    }

    /// <summary>
    /// A span whose text no longer hashes to its registered hash is dropped (never a wrong annotation); the other span of
    /// the same part survives.
    /// </summary>
    [Fact]
    public async Task HashMismatch_DropsOnlyThatSpan()
    {
        string text = "header line\n" + new string('x', 64) + "\n" + new string('y', 64) + "\n";
        int mid = 12 + 65;
        ChunkSpan good = new("good", ChunkHash.Of(text[12..mid]), 12, mid);
        ChunkSpan bad = new("bad", ChunkHash.Of("something else"), mid, text.Length);
        byte[] input = Pipeline.StringBody(("tool", text));

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(text, [good, bad])]);

        JsonArray chunks = JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]!.AsArray();
        Assert.Equal("good", (string)Assert.Single(chunks)!["id"]!);
        Assert.Equal(1, scope.Outcome!.HashMismatch);
    }

    /// <summary>
    /// The client does not normalise: a CRLF part registered under its LF-normalised hash (what an LF index holds) is
    /// dropped, while the CRLF hash is accepted (fixture <c>nonascii-crlf.json</c>, <c>lf_variant</c>).
    /// </summary>
    [Fact]
    public async Task CrlfPart_LfIndexedHash_Dropped()
    {
        JsonObject fixture = Fixtures.Json("nonascii-crlf.json");
        string crlf = Pipeline.PartText(fixture, 3, 2);
        string lfHash = (string)Fixtures.Json("expected.json")["requests"]!["nonascii-crlf.json"]!["lf_variant"]!["lf_hash"]!;
        Assert.Equal(lfHash, ChunkHash.Of(crlf.Replace("\r\n", "\n", StringComparison.Ordinal)));
        byte[] input = Pipeline.FixtureWithoutManifest("nonascii-crlf.json");

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(crlf, [new ChunkSpan("lf", lfHash)])]);

        Assert.Equal(input, sent);
        Assert.Equal(1, scope.Outcome!.HashMismatch);
    }

    /// <summary>
    /// A suffix appended after registration (a hook's additional context) keeps the spans: offsets are unchanged, and a
    /// registered whole-part span is sent with an explicit end, so the suffix is never inside it.
    /// </summary>
    [Fact]
    public async Task AppendedSuffix_KeepsSpan()
    {
        const string glue = "// src/A.cs\n";
        string body = string.Concat(Enumerable.Repeat("int a = 1; // é\n", 8));
        string text = glue + body;
        int k = glue.Length + (4 * 16);
        const string suffix = "\n\n<hook>additional context</hook>";
        ChunkSpan head = new("head", ChunkHash.Of(text[glue.Length..k]), glue.Length, k);
        ChunkSpan tail = new("tail", ChunkHash.Of(text[k..]), k, null); // registered "to the end of the part"
        byte[] input = Pipeline.StringBody(("user", "q"), ("tool", text + suffix));

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(text, [tail, head])]);

        JsonArray chunks = JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]!.AsArray();
        Assert.Equal(["head", "tail"], chunks.Select(c => (string)c!["id"]!));
        Assert.Equal(glue.Length, (int)chunks[0]!["at"]!["start"]!);
        Assert.Equal(Encoding.UTF8.GetByteCount(text[..k]), (int)chunks[0]!["at"]!["end"]!);
        Assert.Equal(Encoding.UTF8.GetByteCount(text[..k]), (int)chunks[1]!["at"]!["start"]!);
        Assert.Equal(Encoding.UTF8.GetByteCount(text), (int)chunks[1]!["at"]!["end"]!); // never into the suffix
        Assert.Equal(0, scope.Outcome!.HashMismatch);
    }

    /// <summary>String <c>content</c> is part 0 (the only v1 form on the OpenAI surface, ruling 19); a whole-part span omits <c>start</c>/<c>end</c>.</summary>
    [Fact]
    public async Task StringContent_IsPartZero()
    {
        string text = "public sealed record ChunkRef(string Id, string Path);\n// trailing comment that is long enough\n";
        byte[] input = Pipeline.StringBody(("system", "sys"), ("user", "q"), ("tool", text));

        (byte[] sent, _, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan("ref", ChunkHash.Of(text))])]);

        JsonObject chunk = JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]![0]!.AsObject();
        Assert.Equal("{\"message\":2,\"part\":0}", chunk["at"]!.ToJsonString());
    }

    /// <summary>
    /// No capabilities (probe failed or convention absent, ruling 16): the body bytes and content headers are exactly
    /// the caller's, even with registered spans that would match.
    /// </summary>
    [Theory]
    [InlineData("null-caps")]
    [InlineData("no-v1")]
    [InlineData("source-null")]
    public async Task NoCapabilities_BodyByteIdentical(string how)
    {
        string text = Pipeline.PartText(Fixtures.Json("manifest.json"), 3, 1);
        byte[] input = Pipeline.StringBody(("user", "q"), ("tool", text));
        ChunkPart[] parts = [new(text, [new ChunkSpan("a", ChunkHash.Of(text))])];
        var capture = CaptureHandler.Ok();

        using (var scope = ChunkManifestScope.Begin(parts, new ChunkRequestOptions()))
        {
            ChunkCacheBodyHandler handler = how switch
            {
                "null-caps" => new ChunkCacheBodyHandler((ChunkCacheCapabilities?)null),
                "no-v1" => new ChunkCacheBodyHandler(new ChunkCacheCapabilities { Versions = [2] }),
                _ => new ChunkCacheBodyHandler(new FixedChunkCapabilities(null)),
            };
            using HttpClient http = Pipeline.Client(capture, handler);
            using HttpResponseMessage r = await http.SendAsync(Pipeline.Post(input));
            Assert.Equal(ChunkSkipReasons.NotCapable, scope.Outcome!.SkipReason);
        }

        Assert.Equal(input, capture.LastBody);
        Assert.Equal("application/json; charset=utf-8", capture.Seen[^1].ContentType);
        Assert.Equal(input.Length, capture.Seen[^1].ContentLength);
    }

    /// <summary>
    /// The report is read from the final SSE chunk while the stream flows: the first event reaches the reader before the
    /// server has written the last one (a buffering handler would deadlock here and time out).
    /// </summary>
    [Fact]
    public async Task Streaming_ReportFromFinalChunk_NoBuffering()
    {
        string text = Pipeline.PartText(Fixtures.Json("manifest.json"), 3, 1);
        byte[] input = Pipeline.StringBody(("user", "q"), ("tool", text));
        var clientSawFirst = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        string final = Fixtures.Json("response-stream-final.json").ToJsonString();

        await using var server = new FakeHttpServer(async (req, s, ct) =>
        {
            await FakeHttpServer.WriteSseHeadAsync(s, ct);
            await FakeHttpServer.WriteRawAsync(s, "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n", ct);
            await clientSawFirst.Task.WaitAsync(s_wait, ct);
            await FakeHttpServer.WriteRawAsync(s, "data: " + final + "\n\ndata: [DONE]\n\n", ct);
        });

        using var scope = ChunkManifestScope.Begin([new ChunkPart(text, [new ChunkSpan("a", ChunkHash.Of(text))])], new ChunkRequestOptions());
        using HttpClient http = Pipeline.Client(new SocketsHttpHandler(), new ChunkCacheBodyHandler(Pipeline.Caps()));
        using HttpRequestMessage request = Pipeline.Post(input, new Uri(server.BaseUri, "v1/chat/completions").ToString());
        using HttpResponseMessage response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead).WaitAsync(s_wait);
        using var reader = new StreamReader(await response.Content.ReadAsStreamAsync());

        string? first = await reader.ReadLineAsync().WaitAsync(s_wait);
        Assert.StartsWith("data: {", first, StringComparison.Ordinal);
        Assert.Null(scope.ResponseReport);
        clientSawFirst.SetResult();
        string rest = await reader.ReadToEndAsync().WaitAsync(s_wait);

        Assert.Contains("[DONE]", rest, StringComparison.Ordinal);
        Assert.True(JsonNode.DeepEquals(Fixtures.Json("response-stream-final.json")["chunk_cache"], scope.ResponseReport));
        Assert.Contains("\"chunk_cache\"", Assert.Single(server.Snapshot).BodyText, StringComparison.Ordinal);
    }

    /// <summary>
    /// A retrying handler between the decorator's scope and the body handler resends the same request message after an
    /// awaited delay; the AsyncLocal scope survives, and both attempts carry the same manifest. The reverse order (retry
    /// below the body handler) resends the rewritten bytes unchanged.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Retry_ResendsSameManifest(bool retryAboveBodyHandler)
    {
        string text = Pipeline.PartText(Fixtures.Json("manifest.json"), 3, 1);
        byte[] input = Pipeline.StringBody(("user", "q"), ("tool", text));
        int n = 0;
        var capture = new CaptureHandler((_, _) => Interlocked.Increment(ref n) == 1
            ? new HttpResponseMessage(HttpStatusCode.ServiceUnavailable)
            : new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(Fixtures.Json("response-blocking.json").ToJsonString(), Encoding.UTF8, "application/json") });
        var retry = new RetryOn503Handler(2, TimeSpan.FromMilliseconds(20));
        var body = new ChunkCacheBodyHandler(Pipeline.Caps());

        using var scope = ChunkManifestScope.Begin([new ChunkPart(text, [new ChunkSpan("a", ChunkHash.Of(text))])], new ChunkRequestOptions());
        using HttpClient http = retryAboveBodyHandler ? Pipeline.Client(capture, retry, body) : Pipeline.Client(capture, body, retry);
        using HttpResponseMessage response = await http.SendAsync(Pipeline.Post(input));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(2, capture.Seen.Count);
        Assert.Equal(capture.Seen[0].Body, capture.Seen[1].Body);
        Assert.Contains("\"chunk_cache\"", Encoding.UTF8.GetString(capture.Seen[1].Body!), StringComparison.Ordinal);
        Assert.NotNull(scope.ResponseReport);
    }

    // ---- handler edge cases ----------------------------------------------------------------------------------------

    /// <summary>A body that already carries <c>chunk_cache</c> or an inline <c>chunk</c> is never touched (idempotent, no conflict).</summary>
    [Theory]
    [InlineData("manifest.json")]
    [InlineData("inline.json")]
    [InlineData("conflict.json")]
    public async Task AlreadyAnnotated_LeftUntouched(string fixtureName)
    {
        byte[] input = Fixtures.Bytes(fixtureName);
        JsonObject fixture = Fixtures.Json(fixtureName);
        string text = Pipeline.PartText(fixture, 3, 1);

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan("x", ChunkHash.Of(text))])]);

        Assert.Equal(input, sent);
        Assert.Equal(ChunkSkipReasons.AlreadyAnnotated, scope.Outcome!.SkipReason);
    }

    /// <summary>Only chat-completion and Messages POSTs are rewritten.</summary>
    [Theory]
    [InlineData("http://dotllm.test/v1/embeddings")]
    [InlineData("http://dotllm.test/v1/chunk-cache/explain")]
    [InlineData("http://dotllm.test/v1/completions")]
    public async Task OtherEndpoints_Untouched(string url)
    {
        string text = new('z', 80);
        byte[] input = Pipeline.StringBody(("tool", text));

        (byte[] sent, _, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan("x", ChunkHash.Of(text))])], path: url);

        Assert.Equal(input, sent);
    }

    /// <summary>The Anthropic Messages path is rewritten too (top-level <c>system</c> is not a message).</summary>
    [Fact]
    public async Task AnthropicMessagesPath_Annotated()
    {
        string text = new('q', 80);
        byte[] input = Encoding.UTF8.GetBytes(new JsonObject
        {
            ["model"] = "m", ["max_tokens"] = 16, ["system"] = "sys",
            ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = "glue\n" }, new JsonObject { ["type"] = "text", ["text"] = text }) }),
        }.ToJsonString());

        (byte[] sent, _, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan("x", ChunkHash.Of(text))])], path: "http://dotllm.test/v1/messages");

        Assert.Equal("{\"message\":0,\"part\":1}", JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]![0]!["at"]!.ToJsonString());
    }

    /// <summary>
    /// Manifest order is document order (message, part, start) whatever the registration order; a span overlapping an
    /// earlier one of the same part is dropped (the server would drop it too, ruling 1); the same text in two messages is
    /// annotated in both.
    /// </summary>
    [Fact]
    public async Task Spans_DocumentOrder_OverlapDropped_RepeatedTextAnnotatedTwice()
    {
        string a = "AAAA aaaa AAAA aaaa AAAA aaaa AAAA aaaa\n";
        string b = "BBBB bbbb BBBB bbbb BBBB bbbb BBBB bbbb\n";
        string text = a + b;
        ChunkSpan second = new("b", ChunkHash.Of(b), a.Length, text.Length);
        ChunkSpan first = new("a", ChunkHash.Of(a), 0, a.Length);
        ChunkSpan overlapping = new("ab", ChunkHash.Of(text[10..50]), 10, 50);
        byte[] input = Pipeline.StringBody(("tool", text), ("user", "next"), ("tool", text));

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(text, [second, overlapping, first])]);

        JsonArray chunks = JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]!.AsArray();
        Assert.Equal(["a", "b", "a", "b"], chunks.Select(c => (string)c!["id"]!));
        Assert.Equal([0, 0, 2, 2], chunks.Select(c => (int)c!["at"]!["message"]!));
        Assert.Equal(2, scope.Outcome!.Overlapping);
    }

    /// <summary>At most <c>max_chunks_per_request</c> annotations, the first ones in document order.</summary>
    [Fact]
    public async Task MaxChunksPerRequest_KeepsFirstInDocumentOrder()
    {
        var texts = Enumerable.Range(0, 5).Select(i => $"chunk number {i} " + new string((char)('a' + i), 60)).ToArray();
        byte[] input = Pipeline.StringBody(texts.Select(t => ("tool", t)).ToArray());

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input,
            texts.Reverse().Select((t, i) => new ChunkPart(t, [new ChunkSpan("c" + t[13], ChunkHash.Of(t))])).ToArray(), caps: Pipeline.Caps(maxChunks: 3));

        JsonArray chunks = JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]!.AsArray();
        Assert.Equal(["c0", "c1", "c2"], chunks.Select(c => (string)c!["id"]!));
        Assert.Equal(2, scope.Outcome!.OverLimit);
    }

    /// <summary>Ids over 256 UTF-16 chars are cut to 256 without splitting a surrogate pair (the id is telemetry only).</summary>
    [Fact]
    public async Task LongId_CutTo256()
    {
        string text = new('k', 100);
        string id = new string('i', 255) + "😀tail";
        byte[] input = Pipeline.StringBody(("tool", text));

        (byte[] sent, _, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan(id, ChunkHash.Of(text))])]);

        string wireId = (string)JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]![0]!["id"]!;
        Assert.Equal(new string('i', 255), wireId);
    }

    /// <summary>
    /// A part holding a lone surrogate is skipped (STJ, like the server's reader, refuses to decode it; review nit 11),
    /// and a span boundary inside a surrogate pair is invalid.
    /// </summary>
    [Fact]
    public async Task Surrogates_LoneSkipped_SplitPairInvalid()
    {
        string lone = "abc\ud800def" + new string('l', 60);
        string pair = "xx😀" + new string('p', 60);
        var cut = new ChunkSpan("cut", ChunkHash.Of(pair[3..]), 3, null); // starts on the low surrogate
        var good = new ChunkSpan("good", ChunkHash.Of(pair[4..]), 4, null);
        // STJ writers replace a lone surrogate with U+FFFD, so write the escape by hand, as a careless serializer might.
        byte[] input = Encoding.UTF8.GetBytes("{\"model\":\"m\",\"messages\":[{\"role\":\"tool\",\"content\":\"abc\\ud800def" + new string('l', 60)
            + "\"},{\"role\":\"tool\",\"content\":\"xx\\ud83d\\ude00" + new string('p', 60) + "\"}]}");

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input,
            [new ChunkPart(lone, [new ChunkSpan("lone", ChunkHash.Of(lone))]), new ChunkPart(pair, [cut, good])]);

        JsonArray chunks = JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]!.AsArray();
        JsonNode only = Assert.Single(chunks)!;
        Assert.Equal("good", (string)only["id"]!);
        Assert.Equal(6, (int)only["at"]!["start"]!); // "xx" + 4-byte emoji
        Assert.Equal(1, scope.Outcome!.Invalid);    // the span starting inside the pair
        Assert.Equal(1, scope.Outcome.NotLocated);  // the lone-surrogate part is never decoded, so never matched
    }

    /// <summary>Request options: non-default values are sent; defaults are omitted (the fixtures' form); span flags map to the wire.</summary>
    [Fact]
    public async Task Options_NonDefaultsSent_DefaultsOmitted()
    {
        string text = new('o', 70);
        byte[] input = Pipeline.StringBody(("tool", text));
        var span = new ChunkSpan("o", ChunkHash.Of(text), Chunker: "symbol-512", Source: new ChunkSourceInfo("src/O.cs", 1, 3, "abc"), Volatile: true, NoStore: true);
        var options = new ChunkRequestOptions(Policy: "shadow", Selector: "chunk_starts", Budget: 0.1, OnMiss: "encode", Strict: true, Report: "none");

        (byte[] sent, _, _) = await SendAsync(input, [new ChunkPart(text, [span])], options);

        Assert.Equal(
            "{\"v\":1,\"policy\":\"shadow\",\"recompute\":{\"selector\":\"chunk_starts\",\"budget\":0.1},\"on_miss\":\"encode\",\"strict\":true,\"report\":\"none\","
            + "\"chunks\":[{\"id\":\"o\",\"hash\":\"" + ChunkHash.Of(text) + "\",\"at\":{\"message\":0,\"part\":0},\"chunker\":\"symbol-512\","
            + "\"source\":{\"path\":\"src/O.cs\",\"start_line\":1,\"end_line\":3,\"rev\":\"abc\"},\"stability\":\"volatile\",\"no_store\":true}]}",
            Pipeline.SplicedManifest(input, sent));

        (byte[] sent2, _, _) = await SendAsync(input, [new ChunkPart(text, [new ChunkSpan("o", ChunkHash.Of(text))])]);
        Assert.Equal("{\"v\":1,\"chunks\":[{\"id\":\"o\",\"hash\":\"" + ChunkHash.Of(text) + "\",\"at\":{\"message\":0,\"part\":0}}]}", Pipeline.SplicedManifest(input, sent2));
    }

    /// <summary>The blocking response's <c>chunk_cache</c> reaches the scope; the response body stays readable and unchanged.</summary>
    [Theory]
    [InlineData("response-blocking.json")]
    [InlineData("response-shadow.json")]
    public async Task BlockingReport_ParsedIntoScope(string responseFixture)
    {
        string text = new('r', 70);
        byte[] input = Pipeline.StringBody(("tool", text));
        byte[] responseBytes = Fixtures.Bytes(responseFixture);
        var capture = new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(responseBytes) { Headers = { ContentType = new("application/json") } } });

        using var scope = ChunkManifestScope.Begin([new ChunkPart(text, [new ChunkSpan("r", ChunkHash.Of(text))])], new ChunkRequestOptions());
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using HttpResponseMessage response = await http.SendAsync(Pipeline.Post(input));

        Assert.Equal(responseBytes, await response.Content.ReadAsByteArrayAsync());
        Assert.True(JsonNode.DeepEquals(Fixtures.Json(responseFixture)["chunk_cache"], scope.ResponseReport));
    }

    /// <summary>A non-JSON, non-SSE response body (e.g. NDJSON) is neither buffered nor read: the caller gets the stream as is.</summary>
    [Fact]
    public async Task NonJsonResponse_NotBuffered()
    {
        string text = new('n', 70);
        byte[] input = Pipeline.StringBody(("tool", text));
        var ndjson = new MemoryStream(Encoding.UTF8.GetBytes("{\"chunk_cache\":{\"v\":1}}\n"));
        var body = new StreamContent(ndjson);
        body.Headers.ContentType = new("application/x-ndjson");
        var capture = new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.OK) { Content = body });

        using var scope = ChunkManifestScope.Begin([new ChunkPart(text, [new ChunkSpan("n", ChunkHash.Of(text))])], new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using HttpResponseMessage response = await invoker.SendAsync(Pipeline.Post(input), CancellationToken.None);

        Assert.Same(body, response.Content);
        Assert.Equal(0, ndjson.Position);
        Assert.Null(scope.ResponseReport);
    }

    /// <summary>
    /// The same request message sent twice re-derives the manifest from the caller's original bytes, so a capability
    /// lost between attempts sends the original body (never a stale manifest).
    /// </summary>
    [Fact]
    public async Task Resend_AfterCapabilityLost_SendsOriginalBytes()
    {
        string text = new('s', 70);
        byte[] input = Pipeline.StringBody(("tool", text));
        var caps = new FlippingCaps { Value = Pipeline.Caps() };
        var capture = CaptureHandler.Ok();
        using var scope = ChunkManifestScope.Begin([new ChunkPart(text, [new ChunkSpan("s", ChunkHash.Of(text))])], new ChunkRequestOptions());
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(caps) { InnerHandler = capture });
        using HttpRequestMessage request = Pipeline.Post(input);

        using (await invoker.SendAsync(request, CancellationToken.None)) { }
        caps.Value = null;
        using (await invoker.SendAsync(request, CancellationToken.None)) { }

        Assert.Contains("\"chunk_cache\"", Encoding.UTF8.GetString(capture.Seen[0].Body!), StringComparison.Ordinal);
        Assert.Equal(input, capture.Seen[1].Body);
    }

    /// <summary>
    /// Strict mode: a server <c>chunk_*</c> 400 reaches the caller unchanged and nothing is resent (the caller asked for
    /// errors). Non-strict: the request is resent once without annotations (byte-identical to the caller's body) and the
    /// capability answer is invalidated. A non-chunk 400 is never resent.
    /// </summary>
    [Theory]
    [InlineData(true, "chunk_hash_mismatch", 400, 1)]
    [InlineData(false, "chunk_hash_mismatch", 400, 2)]
    [InlineData(false, "chunk_store_unavailable", 503, 2)]
    [InlineData(false, "context_length_exceeded", 400, 1)]
    [InlineData(true, "chunk_store_unavailable", 503, 1)]
    public async Task ChunkErrors_StrictPassesThrough_NonStrictFallsBackOnce(bool strict, string code, int status, int expectedRequests)
    {
        string text = new('e', 70);
        byte[] input = Pipeline.StringBody(("tool", text));
        string error = $"{{\"error\":{{\"message\":\"m\",\"type\":\"{(status == 503 ? "api_error" : "invalid_request_error")}\",\"code\":\"{code}\"}}}}";
        var capture = new CaptureHandler((_, body) => Encoding.UTF8.GetString(body!).Contains("\"chunk_cache\"", StringComparison.Ordinal)
            ? new HttpResponseMessage((HttpStatusCode)status) { Content = new StringContent(error, Encoding.UTF8, "application/json") }
            : new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("{}", Encoding.UTF8, "application/json") });
        var caps = new FlippingCaps { Value = Pipeline.Caps() };

        using var scope = ChunkManifestScope.Begin([new ChunkPart(text, [new ChunkSpan("e", ChunkHash.Of(text))])], new ChunkRequestOptions(Strict: strict));
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(caps));
        using HttpResponseMessage response = await http.SendAsync(Pipeline.Post(input));

        Assert.Equal(expectedRequests, capture.Seen.Count);
        if (expectedRequests == 2)
        {
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            Assert.Equal(input, capture.Seen[1].Body);
            Assert.True(scope.Outcome!.FellBackToPlain);
            Assert.Equal(1, caps.Invalidations);
        }
        else
        {
            Assert.Equal((HttpStatusCode)status, response.StatusCode);
            Assert.Contains(code, await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);
            Assert.Equal(0, caps.Invalidations);
        }
    }

    /// <summary>
    /// A registered text found inside a longer serialised part (a bridge that joined two texts, or glue in front) is
    /// annotated at its offset; a span ending inside a surrogate pair is dropped.
    /// </summary>
    [Fact]
    public async Task JoinedPart_SpanLocatedInside()
    {
        string text = "😀 chunk text that is long enough to be a real chunk of a file\n";
        string joined = "glue header é\n" + text;
        var whole = new ChunkSpan("whole", ChunkHash.Of(text));
        var halfPair = new ChunkSpan("half", ChunkHash.Of(text[..1]), 0, 1);
        byte[] input = Pipeline.StringBody(("user", "q"), ("tool", joined));

        (byte[] sent, ChunkManifestScope scope, _) = await SendAsync(input, [new ChunkPart(text, [halfPair, whole])]);

        JsonNode chunk = Assert.Single(JsonNode.Parse(Pipeline.SplicedManifest(input, sent))!["chunks"]!.AsArray())!;
        Assert.Equal("whole", (string)chunk["id"]!);
        Assert.Equal(Encoding.UTF8.GetByteCount("glue header é\n"), (int)chunk["at"]!["start"]!);
        Assert.Equal(Encoding.UTF8.GetByteCount(joined), (int)chunk["at"]!["end"]!);
        Assert.Equal(1, scope.Outcome!.Invalid);
    }

    /// <summary>Scopes nest: disposing restores the enclosing one.</summary>
    [Fact]
    public void Scope_NestsAndRestores()
    {
        Assert.Null(ChunkManifestScope.Current);
        ChunkManifestScope outer = ChunkManifestScope.Begin([], new ChunkRequestOptions());
        ChunkManifestScope inner = ChunkManifestScope.Begin([], new ChunkRequestOptions(Policy: "shadow"));
        Assert.Same(inner, ChunkManifestScope.Current);
        inner.Dispose();
        Assert.Same(outer, ChunkManifestScope.Current);
        outer.Dispose();
        Assert.Null(ChunkManifestScope.Current);
    }

    /// <summary>A request without a scope (no decorator above) is never parsed or changed.</summary>
    [Fact]
    public async Task NoScope_Untouched()
    {
        Assert.Null(ChunkManifestScope.Current);
        byte[] input = Fixtures.Bytes("manifest.json");
        var capture = CaptureHandler.Ok();
        using HttpClient http = Pipeline.Client(capture, new ChunkCacheBodyHandler(Pipeline.Caps()));
        using HttpResponseMessage _ = await http.SendAsync(Pipeline.Post(input));
        Assert.Equal(input, capture.LastBody);
    }

    private static ChunkPart[] NonAsciiParts()
    {
        JsonObject fixture = Fixtures.Json("nonascii-crlf.json");
        string part1 = Pipeline.PartText(fixture, 3, 1);
        string part2 = Pipeline.PartText(fixture, 3, 2);
        int s = Pipeline.Utf16Of(part1, 35), e = Pipeline.Utf16Of(part1, 374);
        return
        [
            new(part1, [new ChunkSpan("docs/notes-fr.md:1-12@0", ChunkHash.Of(part1[s..e]), s, e)]),
            new(part2, [new ChunkSpan("src/Win/Math.cs:1-14@0", ChunkHash.Of(part2))]),
        ];
    }

    private sealed class FlippingCaps : IChunkCapabilitySource
    {
        public ChunkCacheCapabilities? Value { get; set; }

        public int Invalidations { get; private set; }

        public ValueTask<ChunkCacheCapabilities?> GetAsync(string? modelId, CancellationToken ct) => ValueTask.FromResult(Value);

        public void Invalidate() => Invalidations++;
    }
}
