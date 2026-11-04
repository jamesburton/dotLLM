using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// <see cref="ChunkAnnotationParser"/>: the <c>chunk_cache</c> wire convention's parsing and validation (spec §5.1–5.3,
/// §5.8; contract <c>ChunkCache/Fixtures/v1</c>, rulings 1–22). No store, no reuse: only what the request says and
/// whether it matches the text the server received.
/// </summary>
public sealed class ChunkAnnotationParserTests
{
    private static readonly ChunkCacheOptions On = new(true, ChunkCacheOptions.DefaultMaxChunks);
    private const string Ns = "";

    internal static string Sha(string text) =>
        "sha256:" + Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));

    private static int Utf8(string s) => Encoding.UTF8.GetByteCount(s);

    private static ChatCompletionRequest OpenAi(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    private static AnthropicMessagesRequest Anthropic(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;

    private static AnthropicMessagesRequest Anthropic(JsonNode node) => Anthropic(node.ToJsonString());

    private static ChatCompletionRequest OpenAi(JsonNode node) => OpenAi(node.ToJsonString());

    private static string[] Codes(ParsedChunkRequest p) => p.Warnings.Select(w => $"{w.Code}x{w.Count}").ToArray();

    // ------------------------------------------------------------------------------------- contract fixtures

    public static TheoryData<string> RequestFixtures()
    {
        var data = new TheoryData<string>();
        foreach (var p in ContractFixtures.Expected["requests"]!.AsObject())
            data.Add(p.Key);
        return data;
    }

    /// <summary>
    /// Every request fixture, read in the contract's own serialised-message model (string content = part 0, arrays
    /// index their parts), gives exactly <c>expected.json</c>'s non-strict warnings, strict error, accepted span ids
    /// and defect list. The array-content shape is not an OpenAI v1 body (ruling 19), so the parser's core is fed
    /// directly; <see cref="Fixtures_ThroughTheAnthropicSurface_MatchExpected"/> runs the same requests end to end.
    /// </summary>
    [Theory]
    [MemberData(nameof(RequestFixtures))]
    public void Fixtures_MatchExpected(string name)
    {
        JsonNode exp = ContractFixtures.Expected["requests"]![name]!;
        foreach (bool strict in new[] { false, true })
        {
            JsonNode req = ContractFixtures.Request(name);
            if (strict)
                SetStrict(req);
            var parsed = ContractView.Parse(req, On);
            AssertMatchesExpected(name, exp, parsed, strict);
        }
    }

    /// <summary>
    /// The same fixtures translated to the Anthropic surface (system to the top level, assistant tool calls to
    /// <c>tool_use</c> blocks, the tool message to a user message of text blocks, <c>image_url</c> to an <c>image</c>
    /// block, manifest message indices shifted by the removed system message) parse to the same result, and every
    /// accepted span is the exact substring of the engine message it maps to.
    /// </summary>
    [Theory]
    [MemberData(nameof(RequestFixtures))]
    public void Fixtures_ThroughTheAnthropicSurface_MatchExpected(string name)
    {
        JsonNode exp = ContractFixtures.Expected["requests"]![name]!;
        foreach (bool strict in new[] { false, true })
        {
            JsonNode req = ToAnthropic(ContractFixtures.Request(name));
            if (strict)
                SetStrict(req);
            var request = Anthropic(req);
            var parsed = ChunkAnnotationParser.Parse(request, On, Ns);
            AssertMatchesExpected(name, exp, parsed, strict);
            AssertEngineSubstrings(AnthropicConverter.ToMessages(request), parsed!);
        }
    }

    private static void SetStrict(JsonNode req)
    {
        if (req["chunk_cache"] is JsonObject cc)
            cc["strict"] = true;
        else
            req["chunk_cache"] = new JsonObject { ["strict"] = true }; // options only: not a conflict (ruling 2)
    }

    private static void AssertMatchesExpected(string name, JsonNode exp, ParsedChunkRequest? parsed, bool strict)
    {
        Assert.NotNull(parsed);
        var expWarnings = exp["warnings"]!.AsArray().Select(w => $"{(string)w!["code"]!}x{(int)w["count"]!}").ToArray();
        Assert.Equal(expWarnings, Codes(parsed));

        var expIds = exp["accepted_ids"]!.AsArray().Select(x => (string)x!).ToArray();
        Assert.Equal(expIds, parsed.Spans.Select(s => s.Id).ToArray());

        var expDefects = (exp["defects"]?.AsArray() ?? []).Select(d => $"{(int)d!["chunk"]!}:{(string)d["kind"]!}").ToArray();
        if (!name.StartsWith("conflict", StringComparison.Ordinal))
            Assert.Equal(expDefects, parsed.Defects.Select(d => $"{d.Index}:{d.Kind}").ToArray());

        JsonNode? expError = exp["strict_error"];
        if (!strict || expError is null)
        {
            Assert.Null(parsed.Error);
        }
        else
        {
            Assert.NotNull(parsed.Error);
            Assert.Equal((int)expError["status"]!, parsed.Error.Status);
            Assert.Equal((string)expError["type"]!, parsed.Error.Type);
            Assert.Equal((string)expError["code"]!, parsed.Error.Code);
        }

        foreach (var span in parsed.Spans)
            Assert.Equal(span.Hash, Sha(span.Text));
    }

    private static void AssertEngineSubstrings(IReadOnlyList<ChatMessage> engine, ParsedChunkRequest parsed)
    {
        foreach (var s in parsed.Spans)
        {
            string content = engine[s.Engine.EngineMessage].Content;
            Assert.Equal(s.Text, content[s.Engine.StartChar..s.Engine.EndChar]);
        }
    }

    /// <summary>Translates a contract request (OpenAI-shaped, array content) to an Anthropic body with the same parts.</summary>
    internal static JsonNode ToAnthropic(JsonNode req)
    {
        var src = req["messages"]!.AsArray();
        var messages = new JsonArray();
        string? system = null;
        foreach (var m in src)
        {
            string role = (string)m!["role"]!;
            if (string.Equals(role, "system", StringComparison.Ordinal))
            {
                system = (string)m["content"]!;
                continue;
            }

            JsonNode? content = m["content"]?.DeepClone();
            if (string.Equals(role, "assistant", StringComparison.Ordinal) && m["tool_calls"] is JsonArray calls)
            {
                var blocks = new JsonArray();
                foreach (var c in calls)
                    blocks.Add(new JsonObject
                    {
                        ["type"] = "tool_use", ["id"] = c!["id"]!.DeepClone(), ["name"] = c["function"]!["name"]!.DeepClone(),
                        ["input"] = JsonNode.Parse((string)c["function"]!["arguments"]!),
                    });
                content = blocks;
            }
            else if (content is JsonArray parts)
            {
                foreach (var p in parts)
                    if (string.Equals((string?)p!["type"], "image_url", StringComparison.Ordinal))
                    {
                        p.AsObject().Remove("image_url");
                        p["type"] = "image";
                        p["source"] = new JsonObject { ["type"] = "base64", ["media_type"] = "image/png", ["data"] = "iVBORw0KGgo=" };
                    }
            }

            messages.Add(new JsonObject { ["role"] = string.Equals(role, "tool", StringComparison.Ordinal) ? "user" : role, ["content"] = content });
        }

        var body = new JsonObject { ["model"] = "qwen3-4b", ["max_tokens"] = 16, ["messages"] = messages };
        if (system is not null)
            body["system"] = system;
        if (req["chunk_cache"]?.DeepClone() is JsonObject cc)
        {
            if (cc["chunks"] is JsonArray chunks && system is not null)
                foreach (var ch in chunks)
                    ch!["at"]!["message"] = (int)ch["at"]!["message"]! - 1;
            body["chunk_cache"] = cc;
        }
        return body;
    }

    [Fact]
    public void Fixture_NonAscii_ByteOffsetsDifferFromCharOffsets_AndCrlfIsHashedAsSent()
    {
        var parsed = ContractView.Parse(ContractFixtures.Request("nonascii-crlf.json"), On)!;
        var first = parsed.Spans[0];
        Assert.Equal((35, 374), (first.StartByte, first.EndByte));
        // 35 UTF-8 bytes of "// notes (😀)\n"-style prefix are fewer UTF-16 chars: the engine offsets are chars.
        Assert.NotEqual(first.StartByte, first.Engine.StartChar);
        Assert.Contains("\r\n", parsed.Spans[1].Text, StringComparison.Ordinal);
        Assert.NotEqual(Sha(parsed.Spans[1].Text.Replace("\r\n", "\n", StringComparison.Ordinal)), parsed.Spans[1].Hash);
    }

    // ------------------------------------------------------------------------------------------ flag

    [Fact]
    public void FlagOff_ReturnsNull_EvenWithAnnotations()
    {
        foreach (var p in ContractFixtures.Expected["requests"]!.AsObject())
            Assert.Null(ChunkAnnotationParser.Parse(Anthropic(ToAnthropic(ContractFixtures.Request(p.Key))), ChunkCacheOptions.Disabled, Ns));

        var openAi = OpenAi(OpenAiManifest("hello world", 0, 5, Sha("hello")));
        Assert.Null(ChunkAnnotationParser.Parse(openAi, ChunkCacheOptions.Disabled, Ns));
        Assert.NotNull(ChunkAnnotationParser.Parse(openAi, On, Ns)); // control: the same body parses with the flag on
    }

    [Fact]
    public void NoAnnotations_ReturnsNull()
    {
        Assert.Null(ChunkAnnotationParser.Parse(OpenAi("""{"messages":[{"role":"user","content":"hi"}]}"""), On, Ns));
        Assert.Null(ChunkAnnotationParser.Parse(
            Anthropic("""{"max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"""), On, Ns));
    }

    // ------------------------------------------------------------------------------------------ hashing

    /// <summary>The hash is over the UTF-8 bytes of the DECODED string: <c>é</c> on the wire is é (2 bytes).</summary>
    [Fact]
    public void HashOverDecodedString()
    {
        string escaped = "caf\\u00e9 cr\\u00e8me \\ud83d\\ude00\\n";
        string decoded = "café crème 😀\n";
        string body(string hash) =>
            $$$"""{"messages":[{"role":"user","content":"{{{escaped}}}"}],"chunk_cache":{"chunks":[{"id":"a","hash":"{{{hash}}}","at":{"message":0,"part":0}}]}}""";

        Assert.Contains("\\u00e9", body(Sha(decoded)), StringComparison.Ordinal);
        var ok = ChunkAnnotationParser.Parse(OpenAi(body(Sha(decoded))), On, Ns)!;
        Assert.Equal(["a"], ok.Spans.Select(s => s.Id));
        Assert.Equal(decoded, ok.Spans[0].Text);
        Assert.Equal(Utf8(decoded), ok.Spans[0].EndByte);

        var wire = ChunkAnnotationParser.Parse(OpenAi(body(Sha(escaped))), On, Ns)!; // hash of the escaped wire text
        Assert.Empty(wire.Spans);
        Assert.Equal([$"{ChunkCacheCodes.HashMismatch}x1"], Codes(wire));
    }

    /// <summary>Ruling 17: a lone surrogate hashes as UTF-8 with U+FFFD replacement (<see cref="Encoding.UTF8"/>).</summary>
    [Fact]
    public void LoneSurrogate_HashesWithReplacementChar()
    {
        string text = "a\ud800b";
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = text }],
            ChunkCache = JsonSerializer.Deserialize<JsonElement>(
                $$$"""{"chunks":[{"id":"s","hash":"{{{Sha("a\uFFFDb")}}}","at":{"message":0,"part":0}}]}"""),
        };
        var parsed = ChunkAnnotationParser.Parse(request, On, Ns)!;
        Assert.Equal(["s"], parsed.Spans.Select(s => s.Id));
        Assert.Equal(text, parsed.Spans[0].Text);
        Assert.Equal((0, 3), (parsed.Spans[0].Engine.StartChar, parsed.Spans[0].Engine.EndChar));
    }

    [Fact]
    public void UppercaseHex_IsAMismatch()
    {
        string hash = "sha256:" + Convert.ToHexString(SHA256.HashData("hello"u8.ToArray()));
        var parsed = ChunkAnnotationParser.Parse(OpenAi(OpenAiManifest("hello", 0, 5, hash)), On, Ns)!;
        Assert.Empty(parsed.Spans);
        Assert.Equal([$"{ChunkCacheCodes.HashMismatch}x1"], Codes(parsed));
    }

    // ------------------------------------------------------------------------------------------ ids, counts, chunkers

    [Fact]
    public void IdLongerThan256_Rejected()
    {
        string Body(string id) =>
            $$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"chunks":[{"id":"{{{id}}}","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":0}}]}}""";

        var ok = ChunkAnnotationParser.Parse(OpenAi(Body(new string('x', 256))), On, Ns)!;
        Assert.Single(ok.Spans);

        // 256 characters that are 512 UTF-16 code units: the limit counts characters, as the contract's check.py does.
        var emoji = ChunkAnnotationParser.Parse(OpenAi(Body(string.Concat(Enumerable.Repeat("😀", 256)))), On, Ns)!;
        Assert.Single(emoji.Spans);

        var tooLong = ChunkAnnotationParser.Parse(OpenAi(Body(new string('x', 257))), On, Ns)!;
        Assert.Empty(tooLong.Spans);
        Assert.Equal([$"{ChunkCacheCodes.SpanInvalid}x1"], Codes(tooLong));
        Assert.Equal("id_invalid", tooLong.Defects.Single().Kind);

        var empty = ChunkAnnotationParser.Parse(OpenAi(Body("")), On, Ns)!;
        Assert.Empty(empty.Spans);
    }

    [Fact]
    public void TooMany_Warns_StrictIs400()
    {
        string content = "aaaa bbbb cccc";
        string Chunk(string id, int s, int e) =>
            $$$"""{"id":"{{{id}}}","hash":"{{{Sha(content[s..e])}}}","at":{"message":0,"part":0,"start":{{{s}}},"end":{{{e}}}}}""";
        string Body(bool strict) =>
            $$$"""{"messages":[{"role":"user","content":"{{{content}}}"}],"chunk_cache":{"strict":{{{(strict ? "true" : "false")}}},"chunks":[{{{Chunk("a", 0, 4)}}},{{{Chunk("b", 5, 9)}}},{{{Chunk("c", 10, 14)}}}]}}""";
        var two = new ChunkCacheOptions(true, 2);

        var lax = ChunkAnnotationParser.Parse(OpenAi(Body(false)), two, Ns)!;
        Assert.Equal(["a", "b"], lax.Spans.Select(s => s.Id));
        Assert.Equal([$"{ChunkCacheCodes.TooMany}x1"], Codes(lax));
        Assert.Null(lax.Error);
        Assert.Equal(3, lax.Annotated);

        var strict = ChunkAnnotationParser.Parse(OpenAi(Body(true)), two, Ns)!;
        Assert.Equal(new ChunkCacheError(400, "invalid_request_error", ChunkCacheCodes.TooMany, strict.Error!.Message), strict.Error);

        Assert.Equal(3, ChunkAnnotationParser.Parse(OpenAi(Body(false)), On, Ns)!.Spans.Count); // under the default cap
    }

    [Fact]
    public void UnknownChunker_NotWireEligible_WarningAndStrict400()
    {
        string Body(string chunker, bool strict) =>
            $$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"strict":{{{(strict ? "true" : "false")}}},"chunks":[{"id":"a","hash":"{{{Sha("hello")}}}","chunker":"{{{chunker}}}","at":{"message":0,"part":0}}]}}""";

        var fixedChunks = ChunkAnnotationParser.Parse(OpenAi(Body("fixed-512", false)), On, Ns)!;
        Assert.Empty(fixedChunks.Spans);
        Assert.Equal([$"{ChunkCacheCodes.NotWireEligible}x1"], Codes(fixedChunks));

        var strict = ChunkAnnotationParser.Parse(OpenAi(Body("fixed-512", true)), On, Ns)!;
        // Ruling 26: a warning, and a 400 when strict.
        Assert.Equal(new ChunkCacheError(400, "invalid_request_error", ChunkCacheCodes.NotWireEligible, strict.Error!.Message), strict.Error);

        var symbol = ChunkAnnotationParser.Parse(OpenAi(Body("symbol-512", false)), On, Ns)!;
        Assert.Equal("symbol-512", symbol.Spans.Single().Chunker);
        Assert.Empty(symbol.Warnings);
    }

    // ------------------------------------------------------------------------------------------ request options

    [Fact]
    public void QueryDefaultsToLastUserMessage()
    {
        const string messages = """
            [{"role":"system","content":"sys"},{"role":"user","content":"first task"},{"role":"assistant","content":"ok"},
             {"role":"user","content":"the task"},{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},
             {"role":"tool","tool_call_id":"c","content":"file text"}]
            """;
        Assert.Equal("the task", ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":{{{messages}}},"chunk_cache":{}}"""), On, Ns)!.QueryText);
        Assert.Equal("explicit", ChunkAnnotationParser.Parse(
            OpenAi($$$$"""{"messages":{{{{messages}}}},"chunk_cache":{"query":{"text":"explicit"}}}"""), On, Ns)!.QueryText);
        Assert.Equal("first task", ChunkAnnotationParser.Parse(
            OpenAi($$$$"""{"messages":{{{{messages}}}},"chunk_cache":{"query":{"message":1}}}"""), On, Ns)!.QueryText);
        // Ruling 34: an out-of-range query.message is malformed (warning); the default query still applies.
        var outOfRange = ChunkAnnotationParser.Parse(
            OpenAi($$$$"""{"messages":{{{{messages}}}},"chunk_cache":{"query":{"message":99}}}"""), On, Ns)!;
        Assert.Equal("the task", outOfRange.QueryText);
        Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(outOfRange));

        // Anthropic: tool results arrive in a user turn, but the converter makes them `tool` messages; the query is
        // the last real user text.
        var anthropic = Anthropic("""
            {"max_tokens":1,"chunk_cache":{},"messages":[
              {"role":"user","content":"the task"},
              {"role":"assistant","content":[{"type":"tool_use","id":"t","name":"f","input":{}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"file text"}]}]}
            """);
        Assert.Equal("the task", ChunkAnnotationParser.Parse(anthropic, On, Ns)!.QueryText);
    }

    [Fact]
    public void Options_ParseWithDefaults()
    {
        var d = ChunkAnnotationParser.Parse(OpenAi("""{"messages":[{"role":"user","content":"x"}],"chunk_cache":{}}"""), On, "k-1")!;
        Assert.Equal(ChunkPolicy.Exact, d.Requested);
        Assert.False(d.Strict);
        Assert.Equal("summary", d.Report);
        Assert.Equal(32, d.MinChunkTokens);
        Assert.Equal("prefill", d.OnMiss);
        Assert.Equal("k-1", d.Namespace);
        Assert.Empty(d.Spans);
        Assert.Empty(d.Warnings);
        Assert.Equal(0, d.Annotated);

        var o = ChunkAnnotationParser.Parse(OpenAi("""
            {"messages":[{"role":"user","content":"x"}],"chunk_cache":{"v":1,"policy":"shadow","strict":true,"report":"none",
             "min_chunk_tokens":8,"on_miss":"encode"}}
            """), On, "k-1")!;
        Assert.Equal(ChunkPolicy.Shadow, o.Requested);
        Assert.True(o.Strict);
        Assert.Equal("none", o.Report);
        Assert.Equal(8, o.MinChunkTokens);
        Assert.Equal("encode", o.OnMiss);
        Assert.Equal("k-1", o.Namespace);
        Assert.Null(o.Error);
        Assert.Empty(o.Warnings);

        // No authentication yet: the caller's namespace defaults to the single-user one.
        Assert.Equal(ChunkCacheWire.DefaultNamespace, ChunkAnnotationParser.Parse(
            OpenAi("""{"messages":[{"role":"user","content":"x"}],"chunk_cache":{}}"""), On, ChunkCacheWire.DefaultNamespace)!.Namespace);
    }

    /// <summary>
    /// The namespace is server-assigned (§5.2): a body <c>chunk_cache.namespace</c> naming anyone else
    /// is ignored with <c>chunk_namespace_ignored</c>, the caller's own namespace stays, spans are kept; strict is a 400.
    /// </summary>
    [Fact]
    public void BodyNamespace_OtherThanCallers_IgnoredWithWarning_StrictIs400()
    {
        string Body(string ns, bool strict) =>
            $$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"namespace":{{{ns}}},"strict":{{{(strict ? "true" : "false")}}},"chunks":[{"id":"a","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":0}}]}}""";

        foreach (string ns in new[] { "\"k-0123456789abcdef\"", "\"\"", "\"K-1\"" })
        {
            var lax = ChunkAnnotationParser.Parse(OpenAi(Body(ns, false)), On, "k-1")!;
            Assert.Equal("k-1", lax.Namespace);
            Assert.Equal([$"{ChunkCacheCodes.NamespaceIgnored}x1"], Codes(lax));
            Assert.Equal(["a"], lax.Spans.Select(s => s.Id));
            Assert.Null(lax.Error);

            var strict = ChunkAnnotationParser.Parse(OpenAi(Body(ns, true)), On, "k-1")!;
            Assert.Equal(new ChunkCacheError(400, "invalid_request_error", ChunkCacheCodes.NamespaceIgnored, strict.Error!.Message), strict.Error);
        }

        // Anthropic surface, default caller namespace: a body value that is not the default is ignored too.
        var anthropic = ChunkAnnotationParser.Parse(Anthropic(
            """{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"chunk_cache":{"namespace":"team-a"}}"""), On, ChunkCacheWire.DefaultNamespace)!;
        Assert.Equal(ChunkCacheWire.DefaultNamespace, anthropic.Namespace);
        Assert.Equal([$"{ChunkCacheCodes.NamespaceIgnored}x1"], Codes(anthropic));
    }

    [Fact]
    public void BodyNamespace_EqualToCallers_Accepted()
    {
        var own = ChunkAnnotationParser.Parse(OpenAi(
            """{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"namespace":"k-1","strict":true}}"""), On, "k-1")!;
        Assert.Equal("k-1", own.Namespace);
        Assert.Empty(own.Warnings);
        Assert.Null(own.Error);

        var defaultNs = ChunkAnnotationParser.Parse(OpenAi(
            """{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"namespace":"","strict":true}}"""), On, ChunkCacheWire.DefaultNamespace)!;
        Assert.Empty(defaultNs.Warnings);

        var nullNs = ChunkAnnotationParser.Parse(OpenAi(
            """{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"namespace":null}}"""), On, "k-1")!;
        Assert.Empty(nullNs.Warnings);
    }

    [Theory]
    [InlineData("exact", nameof(ChunkPolicy.Exact))]
    [InlineData("shadow", nameof(ChunkPolicy.Shadow))]
    [InlineData("balanced", nameof(ChunkPolicy.Balanced))]
    [InlineData("fast", nameof(ChunkPolicy.Fast))]
    [InlineData("auto", nameof(ChunkPolicy.Auto))]
    public void Policy_ParsesToItsEnumValue_NoWarning(string wire, string expected)
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"policy":"{{{wire}}}"}}"""), On, Ns)!;
        Assert.Equal(Enum.Parse<ChunkPolicy>(expected), p.Requested);
        Assert.Empty(p.Warnings);
    }

    [Fact]
    public void UnknownPolicy_IsExactWithPolicyUnavailable_StrictError()
    {
        var lax = ChunkAnnotationParser.Parse(OpenAi("""{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"policy":"turbo"}}"""), On, Ns)!;
        Assert.Equal(ChunkPolicy.Exact, lax.Requested);
        Assert.Equal([$"{ChunkCacheCodes.PolicyUnavailable}x1"], Codes(lax));
        var strict = ChunkAnnotationParser.Parse(OpenAi("""{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"policy":"turbo","strict":true}}"""), On, Ns)!;
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable, strict.Error!.Code);
        Assert.Equal(400, strict.Error.Status);
    }

    [Fact]
    public void Version_Unsupported_DropsEverything()
    {
        var v2 = ChunkAnnotationParser.Parse(OpenAi(OpenAiManifest("hello", 0, 5, Sha("hello"), extra: "\"v\":2,")), On, Ns)!;
        Assert.Empty(v2.Spans);
        Assert.Equal([$"{ChunkCacheCodes.VersionUnsupported}x1"], Codes(v2));

        // A manifest entry restating another version is a version defect for that entry only.
        var entryV2 = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"chunks":[{"v":2,"id":"a","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":0}}]}}
            """), On, Ns)!;
        Assert.Equal("version", entryV2.Defects.Single().Kind);
        Assert.Equal([$"{ChunkCacheCodes.VersionUnsupported}x1"], Codes(entryV2));

        // Inline v is required (ruling 12): missing is a malformed annotation (ruling 23).
        var noV = ChunkAnnotationParser.Parse(Anthropic($$$"""
            {"max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"hello","chunk":{"id":"a","hash":"{{{Sha("hello")}}}"}}]}]}
            """), On, Ns)!;
        Assert.Empty(noV.Spans);
        Assert.Equal("malformed", noV.Defects.Single().Kind);
        Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(noV));
    }

    [Fact]
    public void Positioning_OnlyIsolated()
    {
        string Body(string positioning) =>
            $$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"chunks":[{"id":"a","hash":"{{{Sha("hello")}}}","positioning":"{{{positioning}}}","at":{"message":0,"part":0}}]}}""";
        Assert.Single(ChunkAnnotationParser.Parse(OpenAi(Body("isolated")), On, Ns)!.Spans);
        var other = ChunkAnnotationParser.Parse(OpenAi(Body("in_place")), On, Ns)!;
        Assert.Empty(other.Spans);
        Assert.Equal("positioning", other.Defects.Single().Kind);
    }

    [Fact]
    public void StabilityAndNoStore_AreCarried()
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"chunks":[
              {"id":"a","hash":"{{{Sha("hello")}}}","stability":"volatile","no_store":true,"source":{"path":"x"},"at":{"message":0,"part":0}}]}}
            """), On, Ns)!;
        var s = p.Spans.Single();
        Assert.True(s.Volatile);
        Assert.True(s.NoStore);
    }

    // ------------------------------------------------------------------------------------------ span rules

    [Theory]
    [InlineData(2, 2)] // empty (ruling 3)
    [InlineData(3, 2)] // reversed
    [InlineData(0, 6)] // past the end
    [InlineData(-1, 2)]
    public void BadRanges_AreSpanInvalid(int start, int end)
    {
        var p = ChunkAnnotationParser.Parse(OpenAi(OpenAiManifest("hello", start, end, Sha("he"))), On, Ns)!;
        Assert.Empty(p.Spans);
        Assert.Equal([$"{ChunkCacheCodes.SpanInvalid}x1"], Codes(p));
        Assert.Equal("range_invalid", p.Defects.Single().Kind);
    }

    [Fact]
    public void Overlap_LaterSpanDropped_AdjacentSpansAccepted()
    {
        string content = "abcdefgh";
        string Chunk(string id, int s, int e) =>
            $$$"""{"id":"{{{id}}}","hash":"{{{Sha(content[s..e])}}}","at":{"message":0,"part":0,"start":{{{s}}},"end":{{{e}}}}}""";
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"{{{content}}}"}],"chunk_cache":{"chunks":[{{{Chunk("a", 0, 4)}}},{{{Chunk("b", 4, 8)}}},{{{Chunk("c", 3, 5)}}}]}}
            """), On, Ns)!;
        Assert.Equal(["a", "b"], p.Spans.Select(s => s.Id));
        Assert.Equal(["2:overlap"], p.Defects.Select(d => $"{d.Index}:{d.Kind}"));
    }

    /// <summary>Ruling 4: the strict error is the first defect in document order, not the most common one.</summary>
    [Fact]
    public void Strict_FirstDefectInDocumentOrderWins()
    {
        string content = "abcdefgh";
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"{{{content}}}"}],"chunk_cache":{"strict":true,"chunks":[
              {"id":"h","hash":"{{{Sha("nope")}}}","at":{"message":0,"part":0,"start":0,"end":2}},
              {"id":"r1","hash":"{{{Sha("x")}}}","at":{"message":0,"part":0,"start":6,"end":99}},
              {"id":"r2","hash":"{{{Sha("x")}}}","at":{"message":0,"part":3}}]}}
            """), On, Ns)!;
        Assert.Equal(ChunkCacheCodes.HashMismatch, p.Error!.Code);
        Assert.Equal([$"{ChunkCacheCodes.HashMismatch}x1", $"{ChunkCacheCodes.SpanInvalid}x2"], Codes(p));
    }

    [Fact]
    public void MalformedAnnotations_AreAnnotationMalformed_NotAThrow()
    {
        string h = Sha("hello");
        foreach (string chunks in new[]
                 {
                     """[5]""", $$$"""[{"id":"a","hash":"{{{h}}}"}]""", $$$"""[{"id":"a","hash":"{{{h}}}","at":{"message":"0","part":0}}]""",
                     $$$"""[{"id":"a","hash":"{{{h}}}","at":{"message":0,"part":0,"start":1.5}}]""", $$$"""[{"id":7,"hash":"{{{h}}}","at":{"message":0,"part":0}}]""",
                     """[{"id":"a","at":{"message":0,"part":0}}]""", """[{"id":"a","hash":5,"at":{"message":0,"part":0}}]""",
                     $$$"""[{"v":"1","id":"a","hash":"{{{h}}}","at":{"message":0,"part":0}}]""", $$$"""[{"id":"a","hash":"{{{h}}}","chunker":5,"at":{"message":0,"part":0}}]""",
                     $$$"""[{"id":"a","hash":"{{{h}}}","no_store":"yes","at":{"message":0,"part":0}}]""", $$$"""[{"id":"a","hash":"{{{h}}}","at":[0,0]}]""",
                 })
        {
            foreach (bool strict in new[] { false, true })
            {
                var p = ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"strict":{{{(strict ? "true" : "false")}}},"chunks":{{{chunks}}}}}"""), On, Ns)!;
                Assert.Empty(p.Spans);
                Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(p));
                Assert.Equal("malformed", p.Defects.Single().Kind);
                if (strict)
                    Assert.Equal((400, "invalid_request_error", ChunkCacheCodes.AnnotationMalformed), (p.Error!.Status, p.Error.Type, p.Error.Code));
            }
        }

        // A well-formed but wrong hash string is a mismatch, not malformed.
        var wrongHash = ChunkAnnotationParser.Parse(OpenAi("""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"chunks":[{"id":"a","hash":"sha256:00","at":{"message":0,"part":0}}]}}"""), On, Ns)!;
        Assert.Equal([$"{ChunkCacheCodes.HashMismatch}x1"], Codes(wrongHash));

        foreach (string cc in new[] { "5", "\"x\"", "[]", """{"chunks":{}}""", """{"strict":"yes"}""", """{"v":"1"}""",
                     """{"policy":3}""", """{"report":true}""", """{"on_miss":1}""", """{"min_chunk_tokens":-1}""", """{"query":"x"}""",
                     """{"query":{"message":"0"}}""", """{"namespace":5}""" })
        {
            var p = ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{{{cc}}}}"""), On, Ns)!;
            Assert.Empty(p.Spans);
            Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(p));
        }

        Assert.Null(ChunkAnnotationParser.Parse(OpenAi("""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":null}"""), On, Ns));
    }

    // ------------------------------------------------------------------------------------------ OpenAI surface

    /// <summary>OpenAI: a manifest span on string content (part 0), byte offsets past non-ASCII glue.</summary>
    [Fact]
    public void OpenAi_StringContentManifest_Parsed()
    {
        string glue = "// Ünïcode 😀 header — café\n";
        string body = "public int Add(int a, int b) => a + b;\n";
        string content = glue + body + "// trailing glue\n";
        var request = OpenAi(new JsonObject
        {
            ["messages"] = new JsonArray(
                new JsonObject { ["role"] = "system", ["content"] = "sys" },
                new JsonObject { ["role"] = "user", ["content"] = content }),
            ["chunk_cache"] = new JsonObject
            {
                ["v"] = 1, ["policy"] = "shadow",
                ["chunks"] = new JsonArray(new JsonObject
                {
                    ["id"] = "src/A.cs:1-1@0", ["hash"] = Sha(body), ["chunker"] = "symbol-512",
                    ["at"] = new JsonObject { ["message"] = 1, ["part"] = 0, ["start"] = Utf8(glue), ["end"] = Utf8(glue + body) },
                }),
            },
        });

        var parsed = ChunkAnnotationParser.Parse(request, On, Ns)!;
        var s = parsed.Spans.Single();
        Assert.Equal((1, 0, Utf8(glue), Utf8(glue + body)), (s.Message, s.Part, s.StartByte, s.EndByte));
        Assert.Equal(new EngineSpan(1, glue.Length, glue.Length + body.Length), s.Engine);
        Assert.NotEqual(s.StartByte, s.Engine.StartChar);
        Assert.Equal(body, s.Text);
        Assert.Equal(body, RequestConverter.ToMessages(request.Messages)[1].Content[s.Engine.StartChar..s.Engine.EndChar]);
        Assert.Equal(ChunkPolicy.Shadow, parsed.Requested);
    }

    [Fact]
    public void OpenAi_PartBeyondZero_OrNullContent_IsSpanInvalid()
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":null}],"chunk_cache":{"chunks":[
              {"id":"a","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":1}},
              {"id":"b","hash":"{{{Sha("hello")}}}","at":{"message":1,"part":0}},
              {"id":"c","hash":"{{{Sha("hello")}}}","at":{"message":2,"part":0}}]}}
            """), On, Ns)!;
        Assert.Empty(p.Spans);
        Assert.Equal(["0:part_out_of_range", "1:part_out_of_range", "2:part_out_of_range"], p.Defects.Select(d => $"{d.Index}:{d.Kind}"));
    }

    /// <summary>A <c>chunk</c> field on an OpenAI message object is not a placement: it stays an ignored unknown field.</summary>
    [Fact]
    public void OpenAi_InlineChunkField_Ignored()
    {
        string chunk = $$$"""{"v":1,"id":"a","hash":"{{{Sha("hello")}}}"}""";
        Assert.Null(ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":[{"role":"user","content":"hello","chunk":{{{chunk}}}}]}"""), On, Ns));

        var withOptions = ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":[{"role":"user","content":"hello","chunk":{{{chunk}}}}],"chunk_cache":{"policy":"shadow"}}"""), On, Ns)!;
        Assert.Empty(withOptions.Spans);
        Assert.Empty(withOptions.Warnings);
        Assert.Equal(0, withOptions.Annotated);
    }

    // ------------------------------------------------------------------------------------------ Anthropic surface

    [Fact]
    public void Anthropic_TwoTextBlocks_SecondBlockSpanShiftedByFirstPlusNewline()
    {
        string head = "// src/A.cs (lines 1-2)\n";
        string body = "class A { int é; }\n";
        var request = Anthropic($$$"""
            {"max_tokens":1,"system":"sys","messages":[{"role":"user","content":[
              {"type":"text","text":{{{JsonSerializer.Serialize(head)}}}},
              {"type":"text","text":{{{JsonSerializer.Serialize(body)}}},"chunk":{"v":1,"id":"a","hash":"{{{Sha(body)}}}"}}]}]}
            """);
        var s = ChunkAnnotationParser.Parse(request, On, Ns)!.Spans.Single();
        // engine message 1: the top-level system prompt is engine message 0
        Assert.Equal(new EngineSpan(1, head.Length + 1, head.Length + 1 + body.Length), s.Engine);
        Assert.Equal((0, 1, 0, Utf8(body)), (s.Message, s.Part, s.StartByte, s.EndByte));
        Assert.Equal(head + "\n" + body, AnthropicConverter.ToMessages(request)[1].Content);

        // Manifest form, a sub-range of the second block: the shift applies to start and end alike.
        var manifest = Anthropic($$$"""
            {"max_tokens":1,"messages":[{"role":"user","content":[
              {"type":"text","text":{{{JsonSerializer.Serialize(head)}}}},{"type":"text","text":{{{JsonSerializer.Serialize(body)}}}}]}],
             "chunk_cache":{"chunks":[{"id":"m","hash":"{{{Sha("int é;")}}}","at":{"message":0,"part":1,"start":10,"end":{{{10 + Utf8("int é;")}}}}}]}}
            """);
        var m = ChunkAnnotationParser.Parse(manifest, On, Ns)!.Spans.Single();
        Assert.Equal(new EngineSpan(0, head.Length + 1 + 10, head.Length + 1 + 10 + "int é;".Length), m.Engine);
    }

    /// <summary>The separator is written only once text has accumulated: after an empty first block, no shift.</summary>
    [Fact]
    public void Anthropic_EmptyFirstBlock_NoSeparatorShift()
    {
        var request = Anthropic($$$"""
            {"max_tokens":1,"messages":[{"role":"user","content":[
              {"type":"text","text":""},{"type":"text","text":"body","chunk":{"v":1,"id":"a","hash":"{{{Sha("body")}}}"}}]}]}
            """);
        Assert.Equal(new EngineSpan(0, 0, 4), ChunkAnnotationParser.Parse(request, On, Ns)!.Spans.Single().Engine);
    }

    [Fact]
    public void Anthropic_ToolResultBlock_MapsToOwnEngineMessage()
    {
        var request = Anthropic($$$"""
            {"max_tokens":1,"messages":[
              {"role":"user","content":"read it"},
              {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read","input":{}},{"type":"tool_use","id":"t2","name":"read","input":{}}]},
              {"role":"user","content":[
                {"type":"text","text":"before"},
                {"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"hdr"},{"type":"text","text":"BODY","chunk":{"v":1,"id":"inner","hash":"{{{Sha("BODY")}}}"}}]},
                {"type":"tool_result","tool_use_id":"t2","content":"STRING RESULT"},
                {"type":"text","text":"after","chunk":{"v":1,"id":"after","hash":"{{{Sha("after")}}}"}}]}]}
            """);
        var engine = AnthropicConverter.ToMessages(request);
        Assert.Equal(["user", "assistant", "user", "tool", "tool", "user"], engine.Select(m => m.Role));

        var spans = ChunkAnnotationParser.Parse(request, On, Ns)!.Spans;
        Assert.Equal(["inner", "after"], spans.Select(s => s.Id));
        Assert.Equal(new EngineSpan(3, 4, 8), spans[0].Engine);     // "hdr\nBODY"
        Assert.Equal((2, 1, 1), (spans[0].Message, spans[0].Part, spans[0].InnerPart));
        Assert.Equal(new EngineSpan(5, 0, 5), spans[1].Engine);     // a new user message after the results: offset reset
    }

    /// <summary>
    /// Manifest on a <c>tool_result</c> with string content addresses that string; one with array content is not a
    /// text part (its inner blocks are reachable only inline), nor is a <c>thinking</c> block.
    /// </summary>
    [Fact]
    public void Anthropic_ToolResultManifest_StringContentIsText_ArrayContentIsNot()
    {
        const string messages = """
            [{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read","input":{}},{"type":"tool_use","id":"t2","name":"read","input":{}}]},
             {"role":"user","content":[
               {"type":"tool_result","tool_use_id":"t1","content":"STRING RESULT"},
               {"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"x"}]},
               {"type":"thinking","thinking":"hmm"}]}]
            """;
        var p = ChunkAnnotationParser.Parse(Anthropic($$$"""
            {"max_tokens":1,"messages":{{{messages}}},"chunk_cache":{"chunks":[
              {"id":"s","hash":"{{{Sha("RESULT")}}}","at":{"message":1,"part":0,"start":7,"end":13}},
              {"id":"a","hash":"{{{Sha("x")}}}","at":{"message":1,"part":1}},
              {"id":"t","hash":"{{{Sha("hmm")}}}","at":{"message":1,"part":2}}]}}
            """), On, Ns)!;
        Assert.Equal(["s"], p.Spans.Select(s => s.Id));
        Assert.Equal(new EngineSpan(1, 7, 13), p.Spans[0].Engine);
        Assert.Equal(["1:non_text", "2:non_text"], p.Defects.Select(d => $"{d.Index}:{d.Kind}"));
    }

    [Fact]
    public void Anthropic_InlineChunkOnNonTextBlock_IsSpanInvalid()
    {
        var p = ChunkAnnotationParser.Parse(Anthropic($$$"""
            {"max_tokens":1,"messages":[{"role":"assistant","content":[
              {"type":"tool_use","id":"t","name":"f","input":{},"chunk":{"v":1,"id":"x","hash":"{{{Sha("{}")}}}"}}]}]}
            """), On, Ns)!;
        Assert.Empty(p.Spans);
        Assert.Equal("non_text", p.Defects.Single().Kind);
    }

    /// <summary>
    /// Property test over random Anthropic block layouts (empty blocks, non-ASCII, tool_use / thinking interleaved,
    /// tool_results with string and array content): every annotated text maps to the exact substring of its engine
    /// message, inline and manifest alike.
    /// </summary>
    [Fact]
    public void Anthropic_MappedSpan_SubstringOfEngineMessageEqualsSpanText()
    {
        string[] pieces = ["a", "é", "😀", "\n", " ", "line\r\n", "<think>", "—", "x y z", ""];
        var rng = new Random(1234);
        int checkedSpans = 0;

        for (int iter = 0; iter < 300; iter++)
        {
            string Text()
            {
                var sb = new StringBuilder();
                int n = rng.Next(0, 5);
                for (int i = 0; i < n; i++) sb.Append(pieces[rng.Next(pieces.Length)]);
                return sb.ToString();
            }

            bool inline = rng.Next(2) == 0;
            var messages = new JsonArray();
            var manifest = new JsonArray();
            int expected = 0;
            int nMessages = rng.Next(1, 4);
            for (int m = 0; m < nMessages; m++)
            {
                if (rng.Next(3) == 0)
                {
                    string t = Text();
                    messages.Add(new JsonObject { ["role"] = "user", ["content"] = t });
                    if (!inline && t.Length > 0)
                    {
                        manifest.Add(Whole($"m{m}", m, 0, t));
                        expected++;
                    }
                    continue;
                }

                var blocks = new JsonArray();
                int nBlocks = rng.Next(1, 6);
                for (int b = 0; b < nBlocks; b++)
                {
                    switch (rng.Next(6))
                    {
                        case 0:
                            blocks.Add(new JsonObject { ["type"] = "tool_use", ["id"] = $"t{m}{b}", ["name"] = "f", ["input"] = new JsonObject() });
                            break;
                        case 1:
                        {
                            var inner = new JsonArray();
                            int nInner = rng.Next(0, 4);
                            for (int k = 0; k < nInner; k++)
                            {
                                string t = Text();
                                var item = new JsonObject { ["type"] = "text", ["text"] = t };
                                if (inline && t.Length > 0)
                                {
                                    item["chunk"] = InlineChunk($"i{m}.{b}.{k}", t);
                                    expected++;
                                }
                                inner.Add(item);
                            }
                            blocks.Add(new JsonObject { ["type"] = "tool_result", ["tool_use_id"] = $"t{m}{b}", ["content"] = inner });
                            break;
                        }
                        case 3:
                            // thinking: replayed as reasoning, never part of the content, so it must not shift offsets
                            blocks.Add(new JsonObject { ["type"] = "thinking", ["thinking"] = Text() + "x" });
                            break;
                        case 2:
                        {
                            string t = Text();
                            blocks.Add(new JsonObject { ["type"] = "tool_result", ["tool_use_id"] = $"t{m}{b}", ["content"] = t });
                            if (!inline && t.Length > 0)
                            {
                                manifest.Add(Whole($"r{m}.{b}", m, b, t));
                                expected++;
                            }
                            break;
                        }
                        default:
                        {
                            string t = Text();
                            var block = new JsonObject { ["type"] = "text", ["text"] = t };
                            if (t.Length > 0)
                            {
                                if (inline) block["chunk"] = InlineChunk($"b{m}.{b}", t);
                                else manifest.Add(SubRange($"b{m}.{b}", m, b, t));
                                expected++;
                            }
                            blocks.Add(block);
                            break;
                        }
                    }
                }
                messages.Add(new JsonObject { ["role"] = m % 2 == 0 ? "user" : "assistant", ["content"] = blocks });
            }

            var body = new JsonObject { ["max_tokens"] = 1, ["messages"] = messages };
            if (rng.Next(2) == 0) body["system"] = "system prompt";
            if (!inline) body["chunk_cache"] = new JsonObject { ["chunks"] = manifest };
            var request = Anthropic(body);
            var parsed = ChunkAnnotationParser.Parse(request, On, Ns);
            if (expected == 0 && inline)
            {
                Assert.Null(parsed);
                continue;
            }

            Assert.NotNull(parsed);
            Assert.Empty(parsed.Warnings);
            Assert.Equal(expected, parsed.Spans.Count);
            var engine = AnthropicConverter.ToMessages(request);
            foreach (var s in parsed.Spans)
            {
                Assert.Equal(s.Text, engine[s.Engine.EngineMessage].Content[s.Engine.StartChar..s.Engine.EndChar]);
                Assert.Equal(s.Hash, Sha(s.Text));
                checkedSpans++;
            }
        }

        Assert.True(checkedSpans > 300, $"only {checkedSpans} spans checked");

        JsonObject InlineChunk(string id, string text) => new() { ["v"] = 1, ["id"] = id, ["hash"] = Sha(text) };

        JsonObject Whole(string id, int message, int part, string text) =>
            new() { ["id"] = id, ["hash"] = Sha(text), ["at"] = new JsonObject { ["message"] = message, ["part"] = part } };

        // A random sub-range on character (rune) boundaries.
        JsonObject SubRange(string id, int message, int part, string text)
        {
            var runes = text.EnumerateRunes().ToArray();
            int a = rng.Next(0, runes.Length), b = rng.Next(a + 1, runes.Length + 1);
            string prefix = string.Concat(runes[..a].Select(r => r.ToString()));
            string span = string.Concat(runes[a..b].Select(r => r.ToString()));
            return new JsonObject
            {
                ["id"] = id, ["hash"] = Sha(span),
                ["at"] = new JsonObject { ["message"] = message, ["part"] = part, ["start"] = Utf8(prefix), ["end"] = Utf8(prefix + span) },
            };
        }
    }

    // ------------------------------------------------------------------------------------------ review fix round

    /// <summary>
    /// B2: System.Text.Json parses a lone-surrogate escape but cannot return it as a string. In any annotation field the
    /// parser reads, that is <c>chunk_annotation_malformed</c> (400 when strict), never an exception (a 500 once wired).
    /// </summary>
    [Theory]
    [InlineData("id")]
    [InlineData("hash")]
    [InlineData("chunker")]
    [InlineData("positioning")]
    [InlineData("stability")]
    [InlineData("policy")]
    [InlineData("namespace")]
    [InlineData("report")]
    [InlineData("on_miss")]
    [InlineData("query.text")]
    public void LoneSurrogateEscape_InAnnotationField_IsMalformed_NeverThrows(string field)
    {
        const string bad = "\"x\\ud800y\"";
        var entry = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["id"] = "\"a\"", ["hash"] = $"\"{Sha("hello")}\"", ["at"] = """{"message":0,"part":0}""",
        };
        var request = new Dictionary<string, string>(StringComparer.Ordinal);
        if (field == "query.text")
            request["query"] = $$$"""{"text":{{{bad}}}}""";
        else if (field is "policy" or "namespace" or "report" or "on_miss")
            request[field] = bad;
        else
            entry[field] = bad;

        foreach (bool strict in new[] { false, true })
        {
            request["strict"] = strict ? "true" : "false";
            request["chunks"] = "[{" + string.Join(",", entry.Select(kv => $"\"{kv.Key}\":{kv.Value}")) + "}]";
            string body = """{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"""
                + string.Join(",", request.Select(kv => $"\"{kv.Key}\":{kv.Value}")) + "}}";
            Assert.Contains("\\ud800", body, StringComparison.Ordinal);

            var parsed = ChunkAnnotationParser.Parse(OpenAi(body), On, Ns)!; // must not throw
            Assert.Contains(parsed.Warnings, w => w.Code == ChunkCacheCodes.AnnotationMalformed && w.Count == 1);
            if (strict)
                Assert.Equal(ChunkCacheCodes.AnnotationMalformed, parsed.Error!.Code);
        }

        // Same through the Anthropic inline placement for the per-entry fields.
        if (entry.ContainsKey(field) && field != "at")
        {
            string chunk = "{\"v\":1," + string.Join(",", entry.Where(kv => kv.Key != "at").Select(kv => $"\"{kv.Key}\":{kv.Value}")) + "}";
            var anthropic = ChunkAnnotationParser.Parse(Anthropic(
                $$$"""{"max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"hello","chunk":{{{chunk}}}}]}]}"""), On, Ns)!;
            Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(anthropic));
        }
    }

    /// <summary>
    /// B3: 4096 whole-part spans on one 4 MiB part. The part is encoded once and only the first span is hashed (the
    /// rest are overlaps, rejected before hashing), so the parse is fast and allocates about one copy of the part.
    /// </summary>
    [Fact]
    public void CostBound_ManyWholePartSpansOnALargePart_EncodeOnceHashOnce()
    {
        string content = new string('a', 4 << 20);
        var chunks = new StringBuilder("[");
        string hash = Sha(content);
        for (int i = 0; i < 4096; i++)
            chunks.Append(i == 0 ? "" : ",").Append($$$"""{"id":"c{{{i}}}","hash":"{{{hash}}}","at":{"message":0,"part":0}}""");
        chunks.Append(']');
        var request = OpenAi($$$"""{"messages":[{"role":"user","content":"{{{content}}}"}],"chunk_cache":{"chunks":{{{chunks}}}}}""");

        ChunkAnnotationParser.HashComputations = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var parsed = ChunkAnnotationParser.Parse(request, On, Ns)!;
        sw.Stop();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(["c0"], parsed.Spans.Select(s => s.Id));
        Assert.Equal(4095, parsed.Defects.Count(d => d.Kind == "overlap"));
        Assert.Equal(1, ChunkAnnotationParser.HashComputations);
        Assert.True(allocated < 64L << 20, $"allocated {allocated >> 20} MiB (one copy of the part is 4 MiB)");
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"took {sw.Elapsed}");
    }

    /// <summary>B3: a manifest far longer than the cap is cut by its array length; the extra entries are never read.</summary>
    [Fact]
    public void CostBound_HugeManifest_OnlyTheCapIsRead()
    {
        const int entries = 1_000_000;
        var chunks = new StringBuilder("[");
        for (int i = 0; i < entries; i++)
            chunks.Append(i == 0 ? "{}" : ",{}");
        chunks.Append(']');
        var request = OpenAi($$$"""{"messages":[{"role":"user","content":"hello"}],"chunk_cache":{"chunks":{{{chunks}}}}}""");
        var cap = new ChunkCacheOptions(true, 16);

        long before = GC.GetAllocatedBytesForCurrentThread();
        var parsed = ChunkAnnotationParser.Parse(request, cap, Ns)!;
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(16, parsed.Defects.Count);
        Assert.Equal([$"{ChunkCacheCodes.TooMany}x{entries - 16}", $"{ChunkCacheCodes.AnnotationMalformed}x16"], Codes(parsed));
        Assert.Equal(entries, parsed.Annotated);
        Assert.True(allocated < 4L << 20, $"allocated {allocated >> 10} KiB for 16 entries");
    }

    /// <summary>Amended ruling 24: overlap is checked before the hash, so an overlapping span with a bad hash is an overlap.</summary>
    [Fact]
    public void CheckOrder_OverlapBeforeHash()
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"abcdefgh"}],"chunk_cache":{"chunks":[
              {"id":"a","hash":"{{{Sha("abcd")}}}","at":{"message":0,"part":0,"start":0,"end":4}},
              {"id":"b","hash":"sha256:00","at":{"message":0,"part":0,"start":2,"end":6}}]}}
            """), On, Ns)!;
        Assert.Equal(["a"], p.Spans.Select(s => s.Id));
        Assert.Equal(["1:overlap"], p.Defects.Select(d => $"{d.Index}:{d.Kind}"));
    }

    /// <summary>
    /// Ruling 34: an unknown <c>report</c> or <c>on_miss</c> string, and a <c>query.message</c> outside the request's
    /// messages, are malformed annotations: a warning and the default otherwise, a 400 when strict. Spans are kept.
    /// </summary>
    [Theory]
    [InlineData("\"report\":\"verbose\"")]
    [InlineData("\"report\":\"\"")]
    [InlineData("\"report\":\"SUMMARY\"")]
    [InlineData("\"on_miss\":\"skip\"")]
    [InlineData("\"on_miss\":\"Encode\"")]
    [InlineData("\"query\":{\"message\":99}")]
    [InlineData("\"query\":{\"message\":1}")]
    [InlineData("\"query\":{\"message\":-1}")]
    [InlineData("\"query\":{\"message\":2147483648}")]
    public void UnknownOptionValue_IsMalformed_DefaultApplies_StrictIs400(string option)
    {
        string Body(bool strict) => $$$"""
            {"messages":[{"role":"user","content":"hello"}],"chunk_cache":{{{{option}}},"strict":{{{(strict ? "true" : "false")}}},
             "chunks":[{"id":"a","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":0}}]}}
            """;

        var lax = ChunkAnnotationParser.Parse(OpenAi(Body(false)), On, Ns)!;
        Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(lax));
        Assert.Equal("summary", lax.Report);
        Assert.Equal("prefill", lax.OnMiss);
        Assert.Equal("hello", lax.QueryText);
        Assert.Equal(["a"], lax.Spans.Select(s => s.Id));
        Assert.Null(lax.Error);

        var strict = ChunkAnnotationParser.Parse(OpenAi(Body(true)), On, Ns)!;
        Assert.Equal((400, "invalid_request_error", ChunkCacheCodes.AnnotationMalformed),
            (strict.Error!.Status, strict.Error.Type, strict.Error.Code));
    }

    /// <summary>The known values of the same options stay accepted with no warning (ruling 34 does not over-reach).</summary>
    [Theory]
    [InlineData("\"report\":\"none\"", "none", "prefill")]
    [InlineData("\"report\":\"per_chunk\"", "per_chunk", "prefill")]
    [InlineData("\"report\":\"summary\"", "summary", "prefill")]
    [InlineData("\"on_miss\":\"encode\"", "summary", "encode")]
    [InlineData("\"on_miss\":\"prefill\"", "summary", "prefill")]
    [InlineData("\"query\":{\"message\":0}", "summary", "prefill")]
    [InlineData("\"query\":{}", "summary", "prefill")]
    public void KnownOptionValues_NoWarning(string option, string report, string onMiss)
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"hello"}],"chunk_cache":{{{{option}}},"strict":true}}
            """), On, Ns)!;
        Assert.Empty(p.Warnings);
        Assert.Null(p.Error);
        Assert.Equal((report, onMiss, "hello"), (p.Report, p.OnMiss, p.QueryText));
    }

    /// <summary>
    /// Ruling 33: the check order is structure, version, id, <b>eligibility</b>, range/boundary, overlap, hash. An
    /// ineligible span is reported as such even when its range is also bad, a bad id wins over eligibility, and an
    /// ineligible span never reserves its range (a later eligible span over the same bytes is accepted).
    /// </summary>
    [Fact]
    public void CheckOrder_EligibilityAfterId_BeforeRange_ReservesNoRange()
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"abcdefgh"}],"chunk_cache":{"chunks":[
              {"id":"bad-range","hash":"{{{Sha("abcd")}}}","chunker":"fixed-512","at":{"message":0,"part":0,"start":0,"end":99}},
              {"id":"","hash":"{{{Sha("abcd")}}}","chunker":"fixed-512","at":{"message":0,"part":0,"start":0,"end":4}},
              {"id":"ineligible","hash":"{{{Sha("abcd")}}}","chunker":"fixed-512","at":{"message":0,"part":0,"start":0,"end":4}},
              {"id":"eligible","hash":"{{{Sha("abcd")}}}","chunker":"symbol-512","at":{"message":0,"part":0,"start":0,"end":4}}]}}
            """), On, Ns)!;
        Assert.Equal(["0:not_wire_eligible", "1:id_invalid", "2:not_wire_eligible"], p.Defects.Select(d => $"{d.Index}:{d.Kind}"));
        Assert.Equal(["eligible"], p.Spans.Select(s => s.Id));
    }

    /// <summary>Amended ruling 4: manifest entries count in array order, not in (message, part) order.</summary>
    [Fact]
    public void Strict_ManifestDefects_InArrayOrder()
    {
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"hello"},{"role":"user","content":"world"}],"chunk_cache":{"strict":true,"chunks":[
              {"id":"late","hash":"{{{Sha("world")}}}","at":{"message":1,"part":0,"start":0,"end":99}},
              {"id":"early","hash":"sha256:00","at":{"message":0,"part":0}}]}}
            """), On, Ns)!;
        Assert.Equal(ChunkCacheCodes.SpanInvalid, p.Error!.Code);
    }

    /// <summary>S7: the engine-substring check catches a mapping that is off by one char (the only guard if a converter edit
    /// bypassed the recorder).</summary>
    [Fact]
    public void WrongEngineOffset_IsNotCarried()
    {
        var engine = new List<ChatMessage> { new() { Role = "user", Content = "xhello" } };
        foreach (int offset in new[] { 0, 2 })
        {
            var wire = new List<ChunkAnnotationParser.WireMessage>
            {
                new("user", [new ChunkAnnotationParser.WirePart(true, "hello", null, 0, offset)]),
            };
            var cc = JsonSerializer.Deserialize<JsonElement>($$$"""{"chunks":[{"id":"a","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":0}}]}""");
            var p = ChunkAnnotationParser.ParseCore(wire, engine, cc, On, Ns)!;
            Assert.Empty(p.Spans);
            Assert.Equal("not_carried", p.Defects.Single().Kind);
        }

        var right = ChunkAnnotationParser.ParseCore(
            [new("user", [new ChunkAnnotationParser.WirePart(true, "hello", null, 0, 1)])], engine,
            JsonSerializer.Deserialize<JsonElement>($$$"""{"chunks":[{"id":"a","hash":"{{{Sha("hello")}}}","at":{"message":0,"part":0}}]}"""), On, Ns)!;
        Assert.Equal(new EngineSpan(0, 1, 6), right.Spans.Single().Engine);
    }

    /// <summary>S5: missing messages or null entries (rejected later by the validators) never throw here.</summary>
    [Fact]
    public void NullMessages_ReturnNull()
    {
        var cc = JsonSerializer.Deserialize<JsonElement>("""{"chunks":[]}""");
        Assert.Null(ChunkAnnotationParser.Parse(new ChatCompletionRequest { Messages = null!, ChunkCache = cc }, On, Ns));
        Assert.Null(ChunkAnnotationParser.Parse(new ChatCompletionRequest { Messages = [null!], ChunkCache = cc }, On, Ns));
        Assert.Null(ChunkAnnotationParser.Parse(Anthropic("""{"max_tokens":1,"messages":null,"chunk_cache":{}}"""), On, Ns));
        Assert.Null(ChunkAnnotationParser.Parse(Anthropic("""{"max_tokens":1,"messages":[null],"chunk_cache":{}}"""), On, Ns));
    }

    /// <summary>
    /// S4: an unannotated Anthropic request is not converted at all. Observable: content System.Text.Json cannot decode
    /// (a lone surrogate) would throw in the converter, but the pre-scan returns first.
    /// </summary>
    [Fact]
    public void Anthropic_Unannotated_SkipsTheConversion()
    {
        var request = Anthropic("""{"max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"x\ud800y"}]}]}""");
        Assert.Throws<InvalidOperationException>(() => AnthropicConverter.ToMessages(request)); // control
        Assert.Null(ChunkAnnotationParser.Parse(request, On, Ns));
    }

    /// <summary>
    /// The codes are wire strings (spec §5.8, rulings 23, 26, 32): pinned literally, because every other test compares
    /// against the constants and could not see a constant with the wrong value. Also pins which codes are warnings only.
    /// </summary>
    [Fact]
    public void Codes_AreTheSpecStrings()
    {
        Assert.Equal(
            ["chunk_version_unsupported", "chunk_hash_mismatch", "chunk_span_invalid", "chunk_manifest_conflict",
             "chunk_template_altered", "chunk_special_tokens", "chunk_model_unsupported", "chunk_policy_unavailable",
             "chunk_too_many", "chunk_store_unavailable", "chunk_namespace_ignored", "chunk_annotation_malformed",
             "chunk_not_wire_eligible", "chunk_ignored_too_short", "chunk_encode_skipped", "chunk_encode_queued"],
            [ChunkCacheCodes.VersionUnsupported, ChunkCacheCodes.HashMismatch, ChunkCacheCodes.SpanInvalid,
             ChunkCacheCodes.ManifestConflict, ChunkCacheCodes.TemplateAltered, ChunkCacheCodes.SpecialTokens,
             ChunkCacheCodes.ModelUnsupported, ChunkCacheCodes.PolicyUnavailable, ChunkCacheCodes.TooMany,
             ChunkCacheCodes.StoreUnavailable, ChunkCacheCodes.NamespaceIgnored, ChunkCacheCodes.AnnotationMalformed,
             ChunkCacheCodes.NotWireEligible, ChunkCacheCodes.IgnoredTooShort, ChunkCacheCodes.EncodeSkipped,
             ChunkCacheCodes.EncodeQueued]);

        string[] warningOnly = ["chunk_ignored_too_short", "chunk_encode_skipped", "chunk_encode_queued"];
        foreach (string code in new[] { "chunk_annotation_malformed", "chunk_not_wire_eligible", "chunk_namespace_ignored", "chunk_span_invalid" })
            Assert.False(ChunkCacheCodes.IsWarningOnly(code), code);
        foreach (string code in warningOnly)
            Assert.True(ChunkCacheCodes.IsWarningOnly(code), code);
        Assert.Equal((503, "api_error"), (ChunkCacheCodes.ErrorFor("chunk_store_unavailable", "m").Status, ChunkCacheCodes.ErrorFor("chunk_store_unavailable", "m").Type));
    }

    /// <summary>Nit N1: client JSON echoed into an error message is clipped.</summary>
    [Fact]
    public void ErrorMessage_ClipsEchoedClientJson()
    {
        string longPolicy = new string('p', 10_000);
        var p = ChunkAnnotationParser.Parse(OpenAi($$$"""{"messages":[{"role":"user","content":"x"}],"chunk_cache":{"strict":true,"policy":"{{{longPolicy}}}"}}"""), On, Ns)!;
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable, p.Error!.Code);
        Assert.True(p.Error.Message.Length < 200, $"message is {p.Error.Message.Length} chars");
    }

    /// <summary>Nit N2: the records' ToString never prints span text, hashes, the query or the namespace (spec §10).</summary>
    [Fact]
    public void ToString_RedactsTextHashQueryAndNamespace()
    {
        var parsed = ChunkAnnotationParser.Parse(OpenAi($$$"""
            {"messages":[{"role":"user","content":"SECRET-TEXT"}],"chunk_cache":{"query":{"text":"SECRET-QUERY"},"chunks":[{"id":"a","hash":"{{{Sha("SECRET-TEXT")}}}","at":{"message":0,"part":0}}]}}
            """), On, "k-secretns")!;
        string all = parsed + " " + parsed.Spans.Single();
        foreach (string secret in new[] { "SECRET-TEXT", "SECRET-QUERY", "k-secretns", Sha("SECRET-TEXT")[7..] })
            Assert.DoesNotContain(secret, all, StringComparison.Ordinal);
        Assert.Contains("Spans = 1", all, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------ helpers

    private static string OpenAiManifest(string content, int start, int end, string hash, string extra = "") =>
        $$$"""{"messages":[{"role":"user","content":{{{JsonSerializer.Serialize(content)}}}}],"chunk_cache":{{{{extra}}}"chunks":[{"id":"a","hash":"{{{hash}}}","at":{"message":0,"part":0,"start":{{{start}}},"end":{{{end}}}}}]}}""";
}

/// <summary>Loads the contract fixtures copied under <c>ChunkCache/Fixtures/v1</c>.</summary>
internal static class ContractFixtures
{
    internal static string Dir => Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1");

    internal static byte[] Bytes(string name) => File.ReadAllBytes(Path.Combine(Dir, name));

    internal static JsonNode Expected { get; } = JsonNode.Parse(Bytes("expected.json"))!;

    internal static JsonNode Request(string name) => JsonNode.Parse(Bytes(name))!;
}

/// <summary>
/// The contract's own serialised-message model (<c>check.py</c>'s <c>parts_of</c>): string content is part 0, an
/// array indexes its parts, only <c>type: "text"</c> is text. Feeds <see cref="ChunkAnnotationParser"/>'s core, with
/// one engine message per wire message whose content is its text parts concatenated.
/// </summary>
internal static class ContractView
{
    internal static ParsedChunkRequest? Parse(JsonNode request, ChunkCacheOptions options)
    {
        using var doc = JsonDocument.Parse(request.ToJsonString());
        var root = doc.RootElement;
        var wire = new List<ChunkAnnotationParser.WireMessage>();
        var engine = new List<ChatMessage>();
        int m = 0;
        foreach (var msg in root.GetProperty("messages").EnumerateArray())
        {
            string role = msg.GetProperty("role").GetString()!;
            var parts = new List<ChunkAnnotationParser.WirePart>();
            var content = new StringBuilder();
            if (msg.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
            {
                parts.Add(new(true, c.GetString(), null, m, 0));
                content.Append(c.GetString());
            }
            else if (c.ValueKind == JsonValueKind.Array)
            {
                foreach (var p in c.EnumerateArray())
                {
                    JsonElement? inline = p.TryGetProperty("chunk", out var ch) ? ch.Clone() : null;
                    if (string.Equals(p.GetProperty("type").GetString(), "text", StringComparison.Ordinal))
                    {
                        parts.Add(new(true, p.GetProperty("text").GetString(), inline, m, content.Length));
                        content.Append(p.GetProperty("text").GetString());
                    }
                    else
                    {
                        parts.Add(new(false, null, inline, -1, 0));
                    }
                }
            }
            wire.Add(new(role, parts));
            engine.Add(new ChatMessage { Role = role, Content = content.ToString() });
            m++;
        }

        JsonElement? cc = root.TryGetProperty("chunk_cache", out var e) ? e.Clone() : null;
        return ChunkAnnotationParser.ParseCore(wire, engine, cc, options, "");
    }
}
