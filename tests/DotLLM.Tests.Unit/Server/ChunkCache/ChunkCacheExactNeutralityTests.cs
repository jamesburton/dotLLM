using System.Globalization;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// <c>exact</c> / <c>shadow</c> over real HTTP on the synthetic model (Phase I Task 13; Review Focus "exact neutrality on a
/// warm slot"): an annotated <c>exact</c> or <c>shadow</c> request generates exactly what the unannotated request generates,
/// with the same <c>cached_tokens</c>, blocking and streaming, on both prompt-cache paths; with the flag off an annotated
/// body is answered byte for byte like the plain one (volatile ids, timestamps and timings blanked); the report sits on the
/// blocking response, the finish-reason chunk and the Anthropic <c>message_delta</c>; <c>/metrics</c> counts it.
/// </summary>
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkCacheExactNeutralityTests : IDisposable
{
    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string ChunkB =
        "public static int Mul(int a, int b)\n{\n    return a * b; // the product of both operands\n}\n";

    private static readonly string User = "Read these files.\n// A.cs\n" + ChunkA + "// B.cs\n" + ChunkB + "What does Add return?";

    private static readonly (string Role, string Content)[] Conversation = [("system", "You are terse."), ("user", User)];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-neutral-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string _model;

    public ChunkCacheExactNeutralityTests() => _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static JsonObject Manifest(string policy, string extra = "")
    {
        var cc = new JsonObject
        {
            ["v"] = 1,
            ["policy"] = policy,
            ["chunks"] = new JsonArray(ChunkCacheTestServer.Entry("a", 1, User, ChunkA), ChunkCacheTestServer.Entry("b", 1, User, ChunkB)),
        };
        if (extra.Length > 0)
            foreach (var p in JsonNode.Parse(extra)!.AsObject().ToList())
                cc[p.Key] = p.Value?.DeepClone();
        return cc;
    }

    private static ChunkCacheOptions On(string? store = null) => ChunkCacheOptions.Create(true, null, store);

    /// <summary>The conversation on a surface. Anthropic has no system role in <c>messages</c> (it goes to <c>system</c>), so
    /// the manifest's message indices shift down by one there.</summary>
    private static string Body(string surface, JsonObject? cc, bool stream)
    {
        if (surface == "openai")
            return ChunkCacheTestServer.OpenAi(Conversation, cc, stream);
        var shifted = cc?.DeepClone().AsObject();
        if (shifted?["chunks"] is JsonArray chunks)
            foreach (var c in chunks)
                c!["at"]!["message"] = c["at"]!["message"]!.GetValue<int>() - 1;
        return ChunkCacheTestServer.Anthropic(Conversation, shifted, stream);
    }

    private static string PathOf(string surface) => surface == "openai" ? "/v1/chat/completions" : "/v1/messages";

    /// <summary>
    /// #1632: an annotated Anthropic text block holding a lone surrogate. Over HTTP the decoder yields U+FFFD (the
    /// converter's InvalidOperationException only occurs for requests built in-process, see
    /// <c>Anthropic_Unannotated_SkipsTheConversion</c>), so the request is served and never becomes a server error; the
    /// annotations that no longer match are ignored with <c>chunk_span_invalid</c>, not an exception.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Anthropic_LoneSurrogateInAnnotatedTextBlock_IsNeverAServerError(bool stream)
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        string streamField = stream ? ",\"stream\":true" : "";
        string body = "{\"max_tokens\":4" + streamField + ",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\ud800y\"}]}],"
            + "\"chunk_cache\":" + Manifest("exact").ToJsonString() + "}";
        var response = await server.PostAsync("/v1/messages", body);
        Assert.True(response.Status is 200 or 400, response.Body);
        if (!stream && response.Status == 200)
            Assert.Contains(ChunkCacheCodes.SpanInvalid, response.Body, StringComparison.Ordinal);
    }

    /// <summary>The body without the chunk-cache additions (<c>chunk_cache</c>, <c>chunk_n</c>, <c>chunk_recomputed_n</c>), normalized.</summary>
    private static string WithoutChunkFields(string body)
    {
        string n = ChunkCacheTestServer.Normalize(body);
        if (n.StartsWith("{", StringComparison.Ordinal))
            return Strip(JsonNode.Parse(n)!).ToJsonString();
        var lines = n.Split('\n').Select(l => l.StartsWith("data: {", StringComparison.Ordinal)
            ? "data: " + Strip(JsonNode.Parse(l["data: ".Length..])!).ToJsonString()
            : l);
        return string.Join('\n', lines);

        static JsonNode Strip(JsonNode node)
        {
            if (node is JsonObject o)
            {
                o.Remove("chunk_cache");
                o.Remove("chunk_n");
                o.Remove("chunk_recomputed_n");
                foreach (var p in o.ToList())
                    if (p.Value is not null) Strip(p.Value);
            }
            else if (node is JsonArray a)
            {
                foreach (var item in a)
                    if (item is not null) Strip(item);
            }
            return node;
        }
    }

    private static int CachedTokens(string surface, string body)
    {
        if (surface == "openai")
        {
            JsonNode usage = body.StartsWith("{", StringComparison.Ordinal)
                ? JsonNode.Parse(body)!["usage"]!
                : ChunkCacheTestServer.SseData(body).Last(d => d["usage"] is not null)["usage"]!;
            return usage["prompt_tokens_details"]!["cached_tokens"]!.GetValue<int>();
        }
        // Anthropic usage has no cache field: the report's prefix_reused equals timings cache_n; for the plain
        // request use the OpenAI comparison. Here: the generated text and usage are what is compared.
        return -1;
    }

    // ------------------------------------------------------------------------------------------------ flag off

    /// <summary>
    /// Flag off: annotated bodies (valid shadow, strict with defects, malformed, inline + manifest) get the same response as
    /// the plain body on every surface, blocking and streaming, and nothing chunk-related appears.
    /// </summary>
    [Fact]
    public async Task FlagOff_AnnotatedBodies_AnsweredLikePlain_EverySurface()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Disabled);
        string[] manifests =
        [
            Manifest("shadow").ToJsonString(),
            Manifest("balanced", """{"strict":true}""").ToJsonString(),
            """{"v":2,"strict":true}""",
            "[1,2]",
        ];
        foreach (string surface in new[] { "openai", "anthropic" })
        {
            foreach (bool stream in new[] { false, true })
            {
                string plain = Body(surface, null, stream);
                await server.PostAsync(PathOf(surface), plain); // warm the prefix cache: later requests all see the same state
                var expected = await server.PostAsync(PathOf(surface), plain);
                Assert.Equal(200, expected.Status);
                string expectedNorm = ChunkCacheTestServer.Normalize(expected.Body);
                foreach (string m in manifests)
                {
                    var annotated = JsonNode.Parse(plain)!;
                    annotated["chunk_cache"] = JsonNode.Parse(m);
                    var got = await server.PostAsync(PathOf(surface), annotated.ToJsonString());
                    Assert.Equal(200, got.Status);
                    Assert.Equal(expectedNorm, ChunkCacheTestServer.Normalize(got.Body));
                    Assert.DoesNotContain("chunk_", got.Body, StringComparison.Ordinal);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ exact neutrality

    /// <summary>
    /// Flag on: an annotated <c>exact</c> request (cold, then warm) generates the same text with the same usage and
    /// <c>cached_tokens</c> as the unannotated request on an identical server; only the chunk-cache fields are added.
    /// </summary>
    [Theory]
    [InlineData("openai", false, "paged")]
    [InlineData("openai", true, "paged")]
    [InlineData("openai", false, "trie")]
    [InlineData("openai", true, "trie")]
    [InlineData("openai", false, "single")]
    [InlineData("openai", true, "single")]
    [InlineData("anthropic", false, "paged")]
    [InlineData("anthropic", true, "trie")]
    [InlineData("anthropic", false, "single")]
    public async Task Exact_IsBitIdenticalToUnannotated(string surface, bool stream, string path)
    {
        await using var annotated = await Start(path);
        await using var plain = await Start(path);
        // Anthropic usage cannot show prefix reuse: an OpenAI twin of the same conversation (same rendered prompt) on a third
        // identical server gives the cached_tokens the annotated Anthropic report's prefix_reused must equal.
        await using var twin = surface == "anthropic" ? await Start(path) : null;
        for (int turn = 0; turn < 2; turn++)
        {
            var a = await annotated.PostAsync(PathOf(surface), Body(surface, Manifest("exact"), stream));
            var p = await plain.PostAsync(PathOf(surface), Body(surface, null, stream));
            Assert.Equal(200, a.Status);
            Assert.Equal(200, p.Status);
            Assert.Contains("\"chunk_cache\"", a.Body, StringComparison.Ordinal);
            Assert.DoesNotContain("chunk_", p.Body, StringComparison.Ordinal);
            Assert.Equal(WithoutChunkFields(p.Body), WithoutChunkFields(a.Body));
            int cached;
            if (twin is not null)
            {
                var t = await twin.PostAsync("/v1/chat/completions", Body("openai", null, stream: false));
                var report = Report(a.Body);
                var usage = JsonNode.Parse(t.Body)!["usage"]!;
                Assert.Equal(usage["prompt_tokens"]!.GetValue<int>(), report["tokens"]!["prompt"]!.GetValue<int>());
                cached = usage["prompt_tokens_details"]!["cached_tokens"]!.GetValue<int>();
                Assert.Equal(cached, report["tokens"]!["prefix_reused"]!.GetValue<int>());
            }
            else
            {
                cached = CachedTokens(surface, a.Body);
                Assert.Equal(CachedTokens(surface, p.Body), cached);
            }
            if (turn == 1 && path != "paged")
                Assert.True(cached > 0, "the warm turn must reuse the exact prefix");
        }
    }

    /// <summary>
    /// The three prompt-cache configurations: <c>single</c> (single-slot prefix cache, generator path), <c>paged</c> (the
    /// default: paged KV + scheduler, which <c>ServerStartup</c> runs without the prefix trie, so there is no cross-request
    /// prefix reuse: <c>cached_tokens</c> 0 both ways) and <c>trie</c> (paged + scheduler with the trie wired in).
    /// </summary>
    private Task<ChunkCacheTestServer> Start(string path, string? store = null) => path switch
    {
        "single" => ChunkCacheTestServer.StartAsync(_model, On(store), paged: false),
        "trie" => ChunkCacheTestServer.StartAsync(_model, On(store), paged: true, schedulerTrie: true),
        _ => ChunkCacheTestServer.StartAsync(_model, On(store), paged: true),
    };

    /// <summary>The <c>chunk_cache</c> object of a blocking body or of the event that carries it in a stream.</summary>
    private static JsonNode Report(string body) => body.StartsWith("{", StringComparison.Ordinal)
        ? JsonNode.Parse(body)!["chunk_cache"]!
        : ChunkCacheTestServer.SseData(body).Single(d => d["chunk_cache"] is not null)["chunk_cache"]!;

    /// <summary><c>cached_tokens</c> / <c>cache_n</c> are the prefix cache's, annotated or not, and the report's
    /// <c>prefix_reused</c> equals them.</summary>
    [Fact]
    public async Task CachedTokens_UnchangedByAnnotations()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(), paged: false);
        await server.PostAsync("/v1/chat/completions", Body("openai", null, false));
        var plain = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", null, false))).Body)!;
        foreach (string policy in new[] { "exact", "shadow", "balanced", "auto" })
        {
            var r = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", Manifest(policy), false))).Body)!;
            int cached = r["usage"]!["prompt_tokens_details"]!["cached_tokens"]!.GetValue<int>();
            Assert.Equal(plain["usage"]!["prompt_tokens_details"]!["cached_tokens"]!.GetValue<int>(), cached);
            Assert.True(cached > 0);
            Assert.Equal(cached, r["timings"]!["cache_n"]!.GetValue<int>());
            Assert.Equal(cached, r["chunk_cache"]!["tokens"]!["prefix_reused"]!.GetValue<int>());
            Assert.Equal(0, r["chunk_cache"]!["tokens"]!["prefix_reused_approx"]!.GetValue<int>());
            Assert.Equal(0, r["timings"]!["chunk_n"]!.GetValue<int>());
            Assert.Equal(0, r["timings"]!["chunk_recomputed_n"]!.GetValue<int>());
        }
    }

    /// <summary>
    /// Review Focus: <c>exact</c> right after <c>shadow</c> on the same slot returns the same tokens and the same
    /// <c>cached_tokens</c> as the unannotated request after an unannotated request.
    /// </summary>
    [Theory]
    [InlineData(false, "paged")]
    [InlineData(true, "paged")]
    [InlineData(false, "trie")]
    [InlineData(true, "trie")]
    [InlineData(false, "single")]
    [InlineData(true, "single")]
    public async Task Exact_AfterShadow_SameSlot_Identical(bool stream, string path)
    {
        await using var annotated = await Start(path);
        await using var plain = await Start(path);

        var shadow = await annotated.PostAsync("/v1/chat/completions", Body("openai", Manifest("shadow"), stream));
        await plain.PostAsync("/v1/chat/completions", Body("openai", null, stream));
        Assert.Contains("\"policy_requested\":\"shadow\"", shadow.Body, StringComparison.Ordinal);

        var exact = await annotated.PostAsync("/v1/chat/completions", Body("openai", Manifest("exact"), stream));
        var reference = await plain.PostAsync("/v1/chat/completions", Body("openai", null, stream));
        Assert.Equal(WithoutChunkFields(reference.Body), WithoutChunkFields(exact.Body));
        Assert.Equal(CachedTokens("openai", reference.Body), CachedTokens("openai", exact.Body));
        if (path != "paged") // the default scheduler path has no cross-request prefix reuse (cached_tokens 0 both ways)
            Assert.True(CachedTokens("openai", exact.Body) > 0);
    }

    /// <summary>
    /// Task 12 lineage on the serving path: an approximate prefix snapshot (rows after <c>r</c> computed from other tokens)
    /// on the slot is restored only up to its first approximate row; the annotated <c>exact</c> request then generates what
    /// a cold unannotated request generates, and <c>cached_tokens</c> is exactly <c>r</c>.
    /// </summary>
    [Fact]
    public async Task Exact_WithApproxSnapshotOnSlot_RestoresExactRowsOnly()
    {
        await using var cold = await ChunkCacheTestServer.StartAsync(_model, On(), paged: false);
        var reference = JsonNode.Parse((await cold.PostAsync("/v1/chat/completions", Body("openai", null, false))).Body)!;

        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(), paged: false);
        var state = server.State;
        var messages = RequestConverter.ToMessages(
            System.Text.Json.JsonSerializer.Deserialize(Body("openai", null, false), ServerJsonContext.Default.ChatCompletionRequest)!.Messages!);
        Assert.True(ReasoningSupport.TryApply(state.ChatTemplate!, messages, ReasoningSupport.BuildTemplateOptions(null, null, null, null, false),
            out string prompt, out _, out _));
        int[] p = state.Tokenizer!.Encode(prompt);
        int r = p.Length / 2;

        // Rows < r are the prompt's own; rows >= r are K/V of different tokens: an "approximate" snapshot.
        int[] q = p.Select((t, i) => i < r ? t : (t + 1) % state.Config!.VocabSize).ToArray();
        var kv = new SimpleKvCache(KvGeometry.FromConfig(state.Config!, state.Model!.SupportsCompactSlidingKv), p.Length + 64);
        using (state.Model.Forward(q, Enumerable.Range(0, q.Length).ToArray(), deviceId: -1, kv)) { }
        state.PrefixCache!.Clear();
        state.PrefixCache.Store(p, kv, firstApproxRow: r);

        var got = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("exact"), false))).Body)!;
        Assert.Equal(r, got["usage"]!["prompt_tokens_details"]!["cached_tokens"]!.GetValue<int>());
        Assert.Equal(r, got["chunk_cache"]!["tokens"]!["prefix_reused"]!.GetValue<int>());
        Assert.Equal(0, got["chunk_cache"]!["tokens"]!["prefix_reused_approx"]!.GetValue<int>());
        Assert.Equal(reference["choices"]![0]!["message"]!["content"]!.GetValue<string>(), got["choices"]![0]!["message"]!["content"]!.GetValue<string>());
        Assert.Equal(reference["usage"]!["completion_tokens"]!.GetValue<int>(), got["usage"]!["completion_tokens"]!.GetValue<int>());
    }

    // ------------------------------------------------------------------------------------------------ report placement

    [Fact]
    public async Task StreamingFinalChunk_CarriesReport()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        var r = await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("exact"), stream: true));
        var data = ChunkCacheTestServer.SseData(r.Body);
        var carriers = data.Where(d => d["chunk_cache"] is not null).ToList();
        var final = Assert.Single(carriers);
        Assert.NotNull(final["choices"]![0]!["finish_reason"]);
        Assert.Equal(0, final["timings"]!["chunk_n"]!.GetValue<int>());
        Assert.Equal(0, final["timings"]!["chunk_recomputed_n"]!.GetValue<int>());
        var report = final["chunk_cache"]!;
        Assert.Equal("exact", report["policy_requested"]!.GetValue<string>());
        Assert.Equal("exact", report["policy_applied"]!.GetValue<string>());
        Assert.Equal(2, report["chunks"]!["annotated"]!.GetValue<int>());
        var usage = data.Last(d => d["usage"] is not null)["usage"]!;
        Assert.Equal(usage["prompt_tokens"]!.GetValue<int>(), report["tokens"]!["prompt"]!.GetValue<int>());
        Assert.Equal(usage["prompt_tokens_details"]!["cached_tokens"]!.GetValue<int>(), report["tokens"]!["prefix_reused"]!.GetValue<int>());
        Assert.Equal(final["timings"]!["cache_n"]!.GetValue<int>(), report["tokens"]!["prefix_reused"]!.GetValue<int>());
    }

    [Fact]
    public async Task AnthropicMessageDelta_CarriesReport()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        var stream = await server.PostAsync("/v1/messages", Body("anthropic", Manifest("shadow"), stream: true));
        var events = ChunkCacheTestServer.SseData(stream.Body);
        var carrier = Assert.Single(events, e => e["chunk_cache"] is not null);
        Assert.Equal("message_delta", carrier["type"]!.GetValue<string>());
        Assert.Equal("shadow", carrier["chunk_cache"]!["policy_requested"]!.GetValue<string>());
        Assert.Equal("exact", carrier["chunk_cache"]!["policy_applied"]!.GetValue<string>());
        Assert.NotNull(carrier["chunk_cache"]!["plan"]);

        var blocking = JsonNode.Parse((await server.PostAsync("/v1/messages", Body("anthropic", Manifest("exact"), stream: false))).Body)!;
        Assert.Equal("exact", blocking["chunk_cache"]!["policy_applied"]!.GetValue<string>());
        Assert.Equal(blocking["usage"]!["input_tokens"]!.GetValue<int>(), blocking["chunk_cache"]!["tokens"]!["prompt"]!.GetValue<int>());
    }

    /// <summary><c>report: "none"</c> omits the object (the request is still served exactly).</summary>
    [Fact]
    public async Task ReportNone_OmitsObject()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        var r = await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("shadow", """{"report":"none"}"""), false));
        Assert.Equal(200, r.Status);
        Assert.DoesNotContain("\"chunk_cache\"", r.Body, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ store, plan, errors

    /// <summary>A store holding chunk A (encoded by the CPU backend under the server's identity): <c>shadow</c> plans one hit and one miss.</summary>
    [Fact]
    public async Task Shadow_OverHttp_CountsStoreHitsAndPlans()
    {
        string root = Path.Combine(_scratch, "store");
        Directory.CreateDirectory(root);
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root));
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        Assert.True(identity.Supported, identity.UnsupportedReason);
        int[] ids = ChunkTokenization.Encode(server.State.Tokenizer!, ChunkA);
        var store = new FileChunkStore(root);
        store.Put(identity.KeyOf(ids, ChunkNamespace.Default), new ChunkEncoder(identity.Fingerprint).Encode(server.State.Model!, ids));

        var r = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("shadow"), false))).Body)!;
        var cc = r["chunk_cache"]!;
        Assert.Equal(1, cc["chunks"]!["hit"]!.GetValue<int>());
        Assert.Equal(1, cc["chunks"]!["miss"]!.GetValue<int>());
        Assert.Equal(0, cc["chunks"]!["ignored"]!.GetValue<int>());
        var plan = cc["plan"]!;
        Assert.Equal(ids.Length, plan["tokens"]!["chunk_reused"]!.GetValue<int>());
        Assert.Equal((int)Math.Ceiling(ids.Length * ChunkCacheReport.DefaultPlanBudget), plan["tokens"]!["chunk_recomputed"]!.GetValue<int>());
        Assert.Equal(0, cc["tokens"]!["chunk_reused"]!.GetValue<int>()); // nothing applied

        // exact (on a cold server over the same store) counts the same hit (ruling 5) and has no plan.
        await using var other = await ChunkCacheTestServer.StartAsync(_model, On(root));
        var e = JsonNode.Parse((await other.PostAsync("/v1/chat/completions", Body("openai", Manifest("exact"), false))).Body)!["chunk_cache"]!;
        Assert.Null(e["plan"]);
        Assert.Equal(1, e["chunks"]!["hit"]!.GetValue<int>());
        Assert.Equal(1, e["chunks"]!["miss"]!.GetValue<int>());
    }

    /// <summary>A configured store root that does not exist: <c>chunk_store_unavailable</c> warning; strict 503 api_error.</summary>
    [Fact]
    public async Task MissingStoreRoot_Warns_StrictIs503()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(Path.Combine(_scratch, "no-such-store")));
        var r = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("exact"), false))).Body)!;
        Assert.Contains(r["chunk_cache"]!["warnings"]!.AsArray(), w => w!["code"]!.GetValue<string>() == ChunkCacheCodes.StoreUnavailable);
        Assert.Equal(2, r["chunk_cache"]!["chunks"]!["miss"]!.GetValue<int>() + r["chunk_cache"]!["chunks"]!["prefix"]!.GetValue<int>());
        Assert.False(Directory.Exists(Path.Combine(_scratch, "no-such-store")));

        var strict = await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("exact", """{"strict":true}"""), false));
        Assert.Equal(503, strict.Status);
        var err = JsonNode.Parse(strict.Body)!["error"]!;
        Assert.Equal("api_error", err["type"]!.GetValue<string>());
        Assert.Equal(ChunkCacheCodes.StoreUnavailable, err["code"]!.GetValue<string>());
    }

    /// <summary>Strict errors reach the caller on both surfaces: 400 invalid_request_error with the code (Anthropic: <c>error.code</c>).</summary>
    [Fact]
    public async Task StrictErrors_AreReturned_BothSurfaces()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());

        var policy = await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("balanced", """{"strict":true}"""), false));
        Assert.Equal(400, policy.Status);
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable, JsonNode.Parse(policy.Body)!["error"]!["code"]!.GetValue<string>());
        Assert.Equal("invalid_request_error", JsonNode.Parse(policy.Body)!["error"]!["type"]!.GetValue<string>());

        var bad = Manifest("exact", """{"strict":true}""");
        bad["chunks"]![0]!["hash"] = ChunkAnnotationParserTests.Sha("something else");
        var hash = await server.PostAsync("/v1/chat/completions", Body("openai", bad, false));
        Assert.Equal(400, hash.Status);
        Assert.Equal(ChunkCacheCodes.HashMismatch, JsonNode.Parse(hash.Body)!["error"]!["code"]!.GetValue<string>());

        foreach (bool stream in new[] { false, true })
        {
            var anth = await server.PostAsync("/v1/messages", Body("anthropic", Manifest("fast", """{"strict":true}"""), stream));
            Assert.Equal(400, anth.Status);
            var body = JsonNode.Parse(anth.Body)!;
            Assert.Equal("error", body["type"]!.GetValue<string>());
            Assert.Equal("invalid_request_error", body["error"]!["type"]!.GetValue<string>());
            Assert.Equal(ChunkCacheCodes.PolicyUnavailable, body["error"]!["code"]!.GetValue<string>());
        }

        // Non-strict: served exact with the reason and the warning.
        var lax = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", Manifest("balanced"), false))).Body)!["chunk_cache"]!;
        Assert.Equal("balanced", lax["policy_requested"]!.GetValue<string>());
        Assert.Equal("exact", lax["policy_applied"]!.GetValue<string>());
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable, lax["reason"]!.GetValue<string>());
    }

    /// <summary><c>/metrics</c> gains <c>dotllm_chunk_cache_chunks_total{status}</c> and <c>dotllm_chunk_cache_tokens_total{kind}</c>.</summary>
    [Fact]
    public async Task Metrics_CountersIncrement()
    {
        string root = Path.Combine(_scratch, "metrics-store");
        Directory.CreateDirectory(root);
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root));
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        int[] ids = ChunkTokenization.Encode(server.State.Tokenizer!, ChunkA);
        new FileChunkStore(root).Put(identity.KeyOf(ids, ChunkNamespace.Default), new ChunkEncoder(identity.Fingerprint).Encode(server.State.Model!, ids));
        long Value(string text, string series)
        {
            string? line = text.Split('\n').FirstOrDefault(l => l.StartsWith(series + " ", StringComparison.Ordinal));
            return line is null ? 0 : long.Parse(line[(series.Length + 1)..], CultureInfo.InvariantCulture);
        }

        string before = await server.GetAsync("/metrics");
        var bad = Manifest("exact");
        bad["chunks"]![1]!["hash"] = ChunkAnnotationParserTests.Sha("not the text"); // chunk A a store hit, chunk B ignored (hash)
        var r1 = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", bad, false))).Body)!["chunk_cache"]!;
        var r2 = JsonNode.Parse((await server.PostAsync("/v1/chat/completions", Body("openai", null, false))).Body)!;
        Assert.Null(r2["chunk_cache"]);
        string after = await server.GetAsync("/metrics");

        foreach (string status in new[] { "prefix", "hit", "miss", "ignored" })
        {
            string series = $"dotllm_chunk_cache_chunks_total{{status=\"{status}\"}}";
            Assert.Equal(r1["chunks"]![status]!.GetValue<int>(), Value(after, series) - Value(before, series));
        }
        Assert.Equal(1, r1["chunks"]!["ignored"]!.GetValue<int>());
        Assert.Equal(1, r1["chunks"]!["hit"]!.GetValue<int>());
        foreach (string kind in new[] { "prompt", "prefix_reused", "prefilled" })
        {
            string series = $"dotllm_chunk_cache_tokens_total{{kind=\"{kind}\"}}";
            Assert.Equal(r1["tokens"]![kind]!.GetValue<int>(), Value(after, series) - Value(before, series));
        }
        Assert.True(Value(after, "dotllm_chunk_cache_tokens_total{kind=\"prompt\"}") > 0);
        Assert.DoesNotContain("namespace", after, StringComparison.Ordinal);
    }
}
