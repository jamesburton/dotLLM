using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Cli.Commands;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Endpoints;
using DotLLM.Server.RateLimiting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Spectre.Console.Cli;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// The chunk-cache control plane and its authentication (Phase I Task 14, spec §5.9, §7.2, §10) over real HTTP on the
/// synthetic model: capabilities against the contract fixture, explain, 401 / 403, namespace isolation between two keys,
/// per-chunk status only for an authenticated caller, no namespace / key / digest in any body or log, rate limiting, and
/// the flag-off 404.
/// </summary>
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkCacheEndpointTests : IDisposable
{
    private const string KeyA = "test-key-alpha-0123456789abcdef0123";
    private const string KeyB = "test-key-bravo-9876543210fedcba9876";
    private const string KeyC = "test-key-charlie-5555555555aaaaaaaa";

    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string ChunkB =
        "public static int Mul(int a, int b)\n{\n    return a * b; // the product of both operands\n}\n";

    private static readonly string User = "Read these files.\n// A.cs\n" + ChunkA + "// B.cs\n" + ChunkB + "What does Add return?";

    private static readonly (string Role, string Content)[] Conversation = [("system", "You are terse."), ("user", User)];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-endpoint-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string _model;

    public ChunkCacheEndpointTests() => _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private string KeysFile(params string[] lines)
    {
        Directory.CreateDirectory(_scratch);
        string path = Path.Combine(_scratch, "keys-" + Guid.NewGuid().ToString("N")[..8] + ".txt");
        File.WriteAllLines(path, lines);
        return path;
    }

    private ChunkCacheOptions On(string? store = null, string? keysFile = null, bool allowShortKeys = false) =>
        ChunkCacheOptions.Create(true, null, store, keysFile, allowShortKeys);

    /// <summary>A POST with an explicit content type and extra headers.</summary>
    private static async Task<(int Status, string Body)> PostRaw(ChunkCacheTestServer server, string path, HttpContent content, params (string Name, string Value)[] headers)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = content };
        foreach (var (n, v) in headers)
            request.Headers.TryAddWithoutValidation(n, v);
        using var resp = await server.Http.SendAsync(request);
        return ((int)resp.StatusCode, await resp.Content.ReadAsStringAsync());
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

    private static string ChatBody(JsonObject? cc) => ChunkCacheTestServer.OpenAi(Conversation, (JsonObject?)cc?.DeepClone());

    private static string ExplainBody(JsonObject? cc) =>
        new JsonObject { ["request"] = JsonNode.Parse(ChatBody(cc)) }.ToJsonString();

    /// <summary>Encodes <see cref="ChunkA"/> into <paramref name="root"/> under <paramref name="ns"/> with the server's identity.</summary>
    private static void PutChunkA(ChunkCacheTestServer server, string root, string ns)
    {
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        Assert.True(identity.Supported, identity.UnsupportedReason);
        int[] ids = ChunkTokenization.Encode(server.State.Tokenizer!, ChunkA);
        new FileChunkStore(root).Put(identity.KeyOf(ids, ns), new ChunkEncoder(identity.Fingerprint).Encode(server.State.Model!, ids));
    }

    private string NewStore()
    {
        string root = Path.Combine(_scratch, "store-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        return root;
    }

    private static (string, string) Key(string key) => ("X-API-Key", key);

    private static (string, string) Bearer(string key) => ("Authorization", "Bearer " + key);

    /// <summary>Makes every request look as if it came from <paramref name="address"/> (a remote caller).</summary>
    private static Action<WebApplication> From(string address) => app => app.Use((ctx, next) =>
    {
        ctx.Connection.RemoteIpAddress = IPAddress.Parse(address);
        return next(ctx);
    });

    private static string? Status(JsonNode explain, string id) =>
        explain["chunks"]!.AsArray().Single(c => c!["id"]!.GetValue<string>() == id)!["status"]!.GetValue<string>();

    // ------------------------------------------------------------------------------------------------ capabilities

    /// <summary>
    /// Capabilities match the contract's <c>capabilities.json</c>: the same top-level keys and types, the same lists, the
    /// store keys exactly, and the model's spec keys plus the key-identity fields (model id, tokenizer id, producer of the
    /// caller's namespace, quantisation, backend, K/V dtype, RoPE).
    /// </summary>
    [Fact]
    public async Task Capabilities_MatchesFixtureShape()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        var (status, body) = await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null);
        Assert.Equal(200, status);
        var got = JsonNode.Parse(body)!.AsObject();
        var fixture = JsonNode.Parse(ContractFixtures.Bytes("capabilities.json"))!.AsObject();

        Assert.Equal(fixture.Select(p => p.Key).Order(StringComparer.Ordinal), got.Select(p => p.Key).Order(StringComparer.Ordinal));
        foreach (string list in new[] { "versions", "policies", "selectors", "wire_chunkers", "auto_validated" })
            Assert.Equal(fixture[list]!.ToJsonString(), got[list]!.ToJsonString());
        Assert.Equal(fixture["store"]!.AsObject().Select(p => p.Key).Append("configured").Order(StringComparer.Ordinal),
            got["store"]!.AsObject().Select(p => p.Key).Order(StringComparer.Ordinal));
        Assert.False(got["store"]!["configured"]!.GetValue<bool>()); // ruling 52: no --chunk-store
        Assert.Equal(0, got["store"]!["entries"]!.GetValue<long>());
        Assert.Equal(0, got["store"]!["bytes"]!.GetValue<long>());
        Assert.Equal(0, got["store"]!["budget_bytes"]!.GetValue<long>());
        Assert.Equal("f32", got["store"]!["dtype"]!.GetValue<string>());
        Assert.Equal(ChunkCacheOptions.DefaultMaxChunks, got["max_chunks_per_request"]!.GetValue<int>());

        var model = got["model"]!.AsObject();
        foreach (var p in fixture["model"]!.AsObject())
            Assert.True(model.ContainsKey(p.Key), $"model.{p.Key} missing");
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        Assert.True(model["supported"]!.GetValue<bool>());
        Assert.Null(model["reason"]);
        Assert.Equal(identity.ModelId, model["id"]!.GetValue<string>());
        Assert.Equal("sampled", model["id_kind"]!.GetValue<string>());
        Assert.Equal(identity.TokenizerId, model["tokenizer_id"]!.GetValue<string>());
        Assert.Equal(identity.KeyOf([], ChunkNamespace.Default).Hex, model["producer"]!.GetValue<string>());
        Assert.Equal(identity.Quantization, model["quantization"]!.GetValue<string>());
        Assert.Equal("cpu", model["backend"]!.GetValue<string>());
        Assert.Equal("f32", model["kv_dtype"]!.GetValue<string>());
        Assert.Equal(identity.Rope.RopeDim, model["rope"]!["dim"]!.GetValue<int>());
        Assert.Equal(identity.Rope.Theta, model["rope"]!["theta"]!.GetValue<float>());
    }

    /// <summary>With the flag off neither route exists: the same bare 404 as any unmapped path, for every method.</summary>
    [Fact]
    public async Task FlagOff_404()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Disabled);
        var unmapped = await server.SendAsync(HttpMethod.Get, "/v1/no-such-route", null);
        Assert.Equal(404, unmapped.Status);
        foreach (var method in new[] { HttpMethod.Get, HttpMethod.Post })
        {
            foreach (string path in new[] { "/v1/chunk-cache", "/v1/chunk-cache/explain" })
            {
                var r = await server.SendAsync(method, path, method == HttpMethod.Post ? ExplainBody(Manifest("exact")) : null, Key(KeyA));
                Assert.Equal(unmapped, r);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ authentication

    /// <summary>No keys configured: a non-loopback caller gets 403 permission_error on both routes; inference is never refused.</summary>
    [Fact]
    public async Task NoKeysConfigured_NonLoopback_403()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(), configure: From("10.1.2.3"));
        foreach (var (method, path, body) in new[] { (HttpMethod.Get, "/v1/chunk-cache", (string?)null), (HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("exact"))) })
        {
            var r = await server.SendAsync(method, path, body);
            Assert.Equal(403, r.Status);
            Assert.Equal(ChunkCacheAuth.PermissionErrorType, JsonNode.Parse(r.Body)!["error"]!["type"]!.GetValue<string>());
        }
        var inference = await server.PostAsync("/v1/chat/completions", ChatBody(Manifest("exact", """{"report":"per_chunk"}""")));
        Assert.Equal(200, inference.Status);
        Assert.Null(JsonNode.Parse(inference.Body)!["chunk_cache"]!["per_chunk"]);
    }

    /// <summary>Without keys: loopback callers only. A browser (an <c>Origin</c> header) or a non-loopback <c>Host</c> (DNS rebinding) is refused too.</summary>
    [Fact]
    public async Task EnabledWithoutKeys_LoopbackOnly()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null)).Status);
        Assert.Equal(403, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, ("Origin", "https://example.com"))).Status);
        Assert.Equal(403, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, ("Host", "rebound.example.com"))).Status);
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, ("Host", "localhost"))).Status);

        // The rule itself, on synthetic connections.
        Assert.True(Local("127.0.0.1", "127.0.0.1"));
        Assert.True(Local("::1", "[::1]"));
        Assert.True(Local("::ffff:127.0.0.1", "localhost"));
        Assert.False(Local(null, "localhost"));          // unknown remote (Unix socket, in-memory host)
        Assert.False(Local("192.168.1.20", "localhost"));
        Assert.False(Local("127.0.0.1", "evil.example"));
        Assert.False(Local("127.0.0.1", "localhost", origin: "null"));
        Assert.True(Local("127.0.0.1", "127.0.0.1:5000"));
        Assert.True(Local("127.0.0.1", "localhost:8080"));
        Assert.True(Local("::1", "[::1]:80"));
        Assert.False(Local("127.0.0.1", "evil.example:80"));
        Assert.False(Local("127.0.0.1", "127.0.0.1.evil.example:80"));

        static bool Local(string? remote, string host, string? origin = null)
        {
            var ctx = new DefaultHttpContext();
            ctx.Connection.RemoteIpAddress = remote is null ? null : IPAddress.Parse(remote);
            ctx.Request.Host = new HostString(host);
            if (origin is not null)
                ctx.Request.Headers.Origin = origin;
            return ChunkCacheAuth.IsLocal(ctx);
        }
    }

    /// <summary>
    /// Keys configured: no key, an unknown key, a near miss (prefix, extension, other case) or two <c>X-API-Key</c> values
    /// are 401 authentication_error with <c>WWW-Authenticate</c>; a listed key is admitted through either header; a loopback
    /// caller without a key is refused too (keys replace the loopback rule).
    /// </summary>
    [Fact]
    public async Task KeysConfigured_MissingKey_401()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(keysFile: KeysFile(KeyA, KeyB)));
        (string, string)[][] refused =
        [
            Array.Empty<(string, string)>(),
            new[] { Key("not-a-configured-key-000") },
            new[] { Key(KeyA[..^1]) },
            new[] { Key(KeyA + "x") },
            new[] { Key(KeyA.ToUpperInvariant()) },
            new[] { Key(KeyA), Key(KeyB) },
            new[] { ("Authorization", "Basic " + KeyA) },
            new[] { Bearer("") },
        ];
        foreach (var headers in refused)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, "/v1/chunk-cache");
            foreach (var (n, v) in headers)
                request.Headers.TryAddWithoutValidation(n, v);
            using var resp = await server.Http.SendAsync(request);
            Assert.Equal(HttpStatusCode.Unauthorized, resp.StatusCode);
            Assert.Equal("Bearer", resp.Headers.WwwAuthenticate.Single().Scheme);
            Assert.Equal(ChunkCacheAuth.AuthenticationErrorType, JsonNode.Parse(await resp.Content.ReadAsStringAsync())!["error"]!["type"]!.GetValue<string>());
        }
        Assert.Equal(401, (await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("exact")))).Status);
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA))).Status);
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Bearer(KeyB))).Status);
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(" " + KeyA + " "))).Status);
    }

    /// <summary>A key's namespace is <c>ChunkNamespace.ForApiKey(key)</c>: the producer in capabilities is the key of no tokens in it.</summary>
    [Fact]
    public async Task KeysConfigured_ValidKey_NamespaceFromKey()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(keysFile: KeysFile(KeyA, KeyB + " admin")));
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        string pa = JsonNode.Parse((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA))).Body)!["model"]!["producer"]!.GetValue<string>();
        string pb = JsonNode.Parse((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Bearer(KeyB))).Body)!["model"]!["producer"]!.GetValue<string>();
        Assert.Equal(identity.KeyOf([], ChunkNamespace.ForApiKey(KeyA)).Hex, pa);
        Assert.Equal(identity.KeyOf([], ChunkNamespace.ForApiKey(KeyB)).Hex, pb);
        Assert.NotEqual(pa, pb);
        Assert.NotEqual(identity.KeyOf([], ChunkNamespace.Default).Hex, pa);

        var ctx = new DefaultHttpContext();
        ctx.Request.Headers["X-API-Key"] = KeyB;
        var caller = ChunkCacheAuth.Resolve(ctx, server.State.Options.ChunkCache, out var failure);
        Assert.Equal(ChunkAuthFailure.None, failure);
        Assert.Equal(new ChunkCaller(true, ChunkNamespace.ForApiKey(KeyB)) { IsAdmin = true }, caller);
        Assert.DoesNotContain(caller.Namespace, caller.ToString(), StringComparison.Ordinal);

        // Two X-API-Key values (HttpClient folds them into one line; a raw client can send two) are ambiguous: no key.
        var two = new DefaultHttpContext();
        two.Request.Headers["X-API-Key"] = new Microsoft.Extensions.Primitives.StringValues([KeyA, KeyB]);
        Assert.Null(ChunkCacheAuth.PresentedKey(two.Request.Headers));
        ChunkCacheAuth.Resolve(two, server.State.Options.ChunkCache, out var twoFailure);
        Assert.Equal(ChunkAuthFailure.Unauthorized, twoFailure);
    }

    /// <summary>
    /// Two keys share a store but not entries: a chunk stored in A's namespace is a hit for A and a miss for B, in
    /// explain and in shadow plan counts; B naming A's namespace in the body is ignored (<c>chunk_namespace_ignored</c>).
    /// (Evict arrives with plan Task 16; its namespace rule is tested there.)
    /// </summary>
    [Fact]
    public async Task TwoKeys_CannotSeeOrEvictEachOthersEntries()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root, KeysFile(KeyA, KeyB)));
        PutChunkA(server, root, ChunkNamespace.ForApiKey(KeyA));

        var a = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("shadow")), Key(KeyA))).Body)!;
        var b = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("shadow")), Key(KeyB))).Body)!;
        Assert.Equal(1, a["counts"]!["hit"]!.GetValue<int>());
        Assert.Equal(0, b["counts"]!["hit"]!.GetValue<int>());
        Assert.Equal(2, b["counts"]!["miss"]!.GetValue<int>());

        var sa = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chat/completions", ChatBody(Manifest("shadow")), Key(KeyA))).Body)!["chunk_cache"]!;
        var sb = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chat/completions", ChatBody(Manifest("shadow")), Key(KeyB))).Body)!["chunk_cache"]!;
        Assert.Equal(1, sa["chunks"]!["hit"]!.GetValue<int>());
        Assert.True(sa["plan"]!["tokens"]!["chunk_reused"]!.GetValue<int>() > 0);
        Assert.Equal(0, sb["chunks"]!["hit"]!.GetValue<int>());
        Assert.Equal(0, sb["plan"]!["tokens"]!["chunk_reused"]!.GetValue<int>());

        // B claims A's namespace in the body: ignored, still B's own.
        string claimed = ExplainBody(Manifest("shadow", $$"""{"namespace":"{{ChunkNamespace.ForApiKey(KeyA)}}"}"""));
        var bc = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", claimed, Key(KeyB))).Body)!;
        Assert.Equal(0, bc["counts"]!["hit"]!.GetValue<int>());
        Assert.Contains(ChunkCacheCodes.NamespaceIgnored, bc["warnings"]!.AsArray().Select(w => w!.GetValue<string>()));
    }

    // ------------------------------------------------------------------------------------------------ explain

    /// <summary>
    /// Explain gives an authenticated caller per-chunk status in its own namespace only (A: <c>a</c> hit; B: miss), with
    /// <c>why</c> for ignored chunks; an unauthenticated caller would get counts only.
    /// </summary>
    [Fact]
    public async Task Explain_PerChunkOnlyForOwnNamespace()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root, KeysFile(KeyA, KeyB)));
        PutChunkA(server, root, ChunkNamespace.ForApiKey(KeyA));

        var cc = Manifest("exact");
        var bad = ChunkCacheTestServer.Entry("c", 1, User, "What does Add return?");
        bad["hash"] = ChunkAnnotationParserTests.Sha("something else");
        cc["chunks"]!.AsArray().Add(bad);

        var (statusA, bodyA) = await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(cc), Key(KeyA));
        Assert.Equal(200, statusA);
        var a = JsonNode.Parse(bodyA)!;
        Assert.Equal("exact", a["policy_would_apply"]!.GetValue<string>());
        Assert.Equal(["a", "b", "c"], a["chunks"]!.AsArray().Select(c => c!["id"]!.GetValue<string>()));
        Assert.Equal("hit", Status(a, "a"));
        Assert.Equal("miss", Status(a, "b"));
        Assert.Equal("ignored", Status(a, "c"));
        Assert.Equal(ChunkCacheCodes.HashMismatch, a["chunks"]![2]!["why"]!.GetValue<string>());
        Assert.True(a["chunks"]![0]!.AsObject().ContainsKey("why")); // ruling 55: a code or null
        Assert.Null(a["chunks"]![0]!["why"]);
        Assert.Equal(3, a["counts"]!["annotated"]!.GetValue<int>());
        Assert.True(a["predicted_ms"]!["full_prefill"]!.GetValue<double>() > 0);

        var b = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(cc), Key(KeyB))).Body)!;
        Assert.Equal("miss", Status(b, "a"));

        // An unauthenticated caller never gets per-chunk status (the control plane refuses it; the builder enforces it too).
        var plan = new ChunkCachePlan(ChunkPolicy.Exact, null, false, 1, [], [], null, "summary", "cpu", 10)
        {
            Statuses = [new SpanStatus(0, "a", SpanStatus.Hit, null, 0, 5)],
        };
        Assert.Null(ChunkCacheEndpoint.BuildExplain(plan, ChunkCaller.Anonymous).Chunks);
        Assert.Single(ChunkCacheEndpoint.BuildExplain(plan, new ChunkCaller(true, ChunkNamespace.Default)).Chunks!);
    }

    /// <summary>
    /// Explain takes <c>{"request": ...}</c> only: a hash list, an extra field, a non-object, a non-JSON body or an invalid
    /// chat body is 400 invalid_request_error (never a 500); a strict request's defect is the same error the chat endpoint
    /// gives.
    /// </summary>
    [Fact]
    public async Task Explain_HashListBody_Rejected()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        string hash = ChunkAnnotationParserTests.Sha(ChunkA);
        string[] bodies =
        [
            $$"""{"hashes":["{{hash}}"]}""",
            $$"""{"chunks":[{"id":"a","hash":"{{hash}}"}]}""",
            $$"""{"request":{{ChatBody(Manifest("exact"))}},"hashes":["{{hash}}"]}""",
            $$"""{"request":{{ChatBody(Manifest("exact"))}},"request":{{ChatBody(null)}}}""",
            """["a"]""",
            "not json",
            "",
            """{"request":"hello"}""",
            """{"request":{"messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}}""",
            """{"request":{"messages":"nope"}}""",
            """{"request":{}}""",
        ];
        foreach (string body in bodies)
        {
            var (status, text) = await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", body);
            Assert.True(status == 400, $"{body[..Math.Min(40, body.Length)]}: {status} {text}");
            Assert.Equal("invalid_request_error", JsonNode.Parse(text)!["error"]!["type"]!.GetValue<string>());
        }

        var strict = Manifest("exact", """{"strict":true}""");
        strict["chunks"]![0]!["hash"] = ChunkAnnotationParserTests.Sha("something else");
        var viaExplain = await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(strict));
        var viaChat = await server.PostAsync("/v1/chat/completions", ChatBody(strict));
        Assert.Equal(400, viaExplain.Status);
        Assert.Equal(viaChat.Status, viaExplain.Status);
        Assert.Equal(JsonNode.Parse(viaChat.Body)!["error"]!.ToJsonString(), JsonNode.Parse(viaExplain.Body)!["error"]!.ToJsonString());

        // No annotation: nothing to explain, not an error.
        var plain = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(null))).Body)!;
        Assert.Equal(0, plain["counts"]!["annotated"]!.GetValue<int>());
    }

    // ------------------------------------------------------------------------------------------------ privacy

    /// <summary>
    /// No response body and no log line (console and <c>ILogger</c>, every level) carries a caller's namespace, key or key
    /// digest, across capabilities, explain, 401s and annotated inference (blocking and streaming, per-chunk).
    /// </summary>
    [Fact]
    public async Task NamespaceNeverInResponseBodiesOrLogs()
    {
        string root = NewStore();
        var logs = new CapturingLoggerProvider();
        var console = new StringWriter();
        TextWriter oldOut = Console.Out, oldErr = Console.Error;
        var bodies = new List<string>();
        Console.SetOut(console);
        Console.SetError(console);
        try
        {
            await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root, KeysFile(KeyA, "tiny-qz7"), allowShortKeys: true), logs: logs);
            PutChunkA(server, root, ChunkNamespace.ForApiKey(KeyA));
            bodies.Add((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA + "x"))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("shadow")), Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", "{\"hashes\":[]}", Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, "/v1/chat/completions", ChatBody(Manifest("shadow", """{"report":"per_chunk"}""")), Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, "/v1/chat/completions",
                ChunkCacheTestServer.OpenAi(Conversation, Manifest("shadow", """{"report":"per_chunk"}"""), stream: true), Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, "/v1/chat/completions",
                ChatBody(Manifest("exact", $$"""{"strict":true,"namespace":"{{ChunkNamespace.ForApiKey(KeyA)}}x"}""")), Key(KeyA))).Body);
        }
        finally
        {
            Console.SetOut(oldOut);
            Console.SetError(oldErr);
        }

        string all = string.Join("\n", bodies) + "\n" + console + "\n" + logs.Text;
        Assert.Contains("hit", bodies[2], StringComparison.Ordinal); // the bodies are real answers
        Assert.Contains("line 2", console.ToString(), StringComparison.Ordinal); // the short-key warning names the line
        foreach (string key in new[] { KeyA, "tiny-qz7" })
        {
            string ns = ChunkNamespace.ForApiKey(key);
            string digest = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(key)));
            Assert.DoesNotContain(key, all, StringComparison.Ordinal);
            Assert.DoesNotContain(ns, all, StringComparison.Ordinal);
            Assert.DoesNotContain(ns[2..], all, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(digest, all, StringComparison.OrdinalIgnoreCase);
        }
    }

    // ------------------------------------------------------------------------------------------------ keys file

    /// <summary>
    /// The keys file: keys are trimmed (spaces, tabs); comments and blank lines skipped; <c>admin</c> marks an admin key;
    /// a non-ASCII or control character (also a no-break space) refuses the file naming the line, never the key; a key
    /// under 32 characters is refused (line number, never the key) unless <c>--chunk-cache-allow-short-keys</c>, which
    /// turns it into a warning; duplicates, an unknown field and an empty file are refused; both option paths read the
    /// flags, which need <c>--chunk-cache</c> (and the short-key override needs a keys file).
    /// </summary>
    [Fact]
    public void KeysFile_TrimsKeys_RefusesNonAscii_WarnsShortKeys()
    {
        var shortRefused = Assert.Throws<ArgumentException>(() => ChunkCacheKeys.Parse([KeyA, "only-twenty-chars-xx"], "k.txt", allowShortKeys: false));
        Assert.Contains("line 2", shortRefused.Message, StringComparison.Ordinal);
        Assert.Contains("--chunk-cache-allow-short-keys", shortRefused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("only-twenty-chars-xx", shortRefused.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => ChunkCacheKeys.Parse([new string('k', 31)], "k.txt", allowShortKeys: false));
        Assert.Equal(1, ChunkCacheKeys.Parse([new string('k', 32)], "k.txt", allowShortKeys: false).Count);

        var keys = ChunkCacheKeys.Parse(["# comment", "", "   " + KeyA + " \t", "\t" + KeyB + "  admin ", "tiny"], "k.txt", allowShortKeys: true);
        Assert.Equal(3, keys.Count);
        Assert.Equal(1, keys.AdminCount);
        Assert.True(keys.TryMatch(KeyA, out bool adminA) && !adminA);
        Assert.True(keys.TryMatch(KeyB, out bool adminB) && adminB);
        Assert.True(keys.TryMatch("tiny", out _));
        Assert.False(keys.TryMatch("   " + KeyA + " x", out _));
        Assert.False(keys.TryMatch(KeyA[..10], out _));
        Assert.False(keys.TryMatch("# comment", out _));
        Assert.Equal(["--chunk-cache-api-keys: k.txt, line 5: key shorter than 32 characters, accepted because of --chunk-cache-allow-short-keys (guessable offline from the store's file names)."], keys.Warnings);
        Assert.DoesNotContain("tiny", keys.ToString(), StringComparison.Ordinal);

        foreach (var (lines, line) in new (string[], int)[]
        {
            ([KeyA, "clé-secrète-0123456789"], 2),
            (["ok-key-0123456789abcdef", KeyA + " "], 2),
            (["　" + KeyA], 1),
            ([KeyA + "\u0001"], 1),
            ([KeyA, KeyB, KeyA], 3),
            ([KeyA + " root"], 1),
            ([KeyA + " admin extra"], 1),
            ([KeyA + " admin admin"], 1),
            ([KeyA + " ns=default"], 1),
            ([KeyA + " ns="], 1),
            ([KeyA + " ns=Team/X"], 1),
            ([KeyA + " ns=a ns=b"], 1),
            ([KeyB, KeyA + " ns=" + new string('n', 65)], 2),
        })
        {
            var ex = Assert.Throws<ArgumentException>(() => ChunkCacheKeys.Parse(lines, "k.txt", allowShortKeys: true));
            Assert.Contains($"line {line}", ex.Message, StringComparison.Ordinal);
            foreach (string l in lines)
                Assert.DoesNotContain(l.Trim(), ex.Message, StringComparison.Ordinal);
        }
        Assert.Throws<ArgumentException>(() => ChunkCacheKeys.Parse(["# only a comment", ""], "k.txt", allowShortKeys: false));

        string file = KeysFile(KeyA);
        Assert.Equal(1, ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-cache-api-keys", file]).ChunkCache.ApiKeys!.Count);
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache-api-keys", file]));
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-cache-api-keys", Path.Combine(_scratch, "missing.txt")]));
        Assert.True(new ServeCommand.Settings { ChunkCache = true, ChunkCacheApiKeys = file }.Validate().Successful);
        Assert.False(new ServeCommand.Settings { ChunkCacheApiKeys = file }.Validate().Successful);
        Assert.False(new ServeCommand.Settings { ChunkCache = true, ChunkCacheApiKeys = KeysFile("bad key here") }.Validate().Successful);

        // Short keys: refused by default on both paths, accepted with the override (which needs a keys file).
        string shortFile = KeysFile("short-key-0123");
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-cache-api-keys", shortFile]));
        Assert.Equal(1, ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-cache-api-keys", shortFile, "--chunk-cache-allow-short-keys"]).ChunkCache.ApiKeys!.Count);
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-cache-allow-short-keys"]));
        Assert.False(new ServeCommand.Settings { ChunkCache = true, ChunkCacheApiKeys = shortFile }.Validate().Successful);
        Assert.True(new ServeCommand.Settings { ChunkCache = true, ChunkCacheApiKeys = shortFile, ChunkCacheAllowShortKeys = true }.Validate().Successful);

        // An unusable path is a clean configuration error (ArgumentException), never another exception type.
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, null, "bad\0path"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, null, Path.Combine(_scratch, "no-such-dir", "keys.txt")));
        Assert.False(new ServeCommand.Settings { ChunkCache = true, ChunkCacheApiKeys = "bad\0path" }.Validate().Successful);
        var names = typeof(ServeCommand.Settings).GetProperties()
            .Select(p => p.GetCustomAttribute<CommandOptionAttribute>())
            .Where(a => a is not null).SelectMany(a => a!.LongNames).ToHashSet(StringComparer.Ordinal);
        Assert.Contains("chunk-cache-api-keys", names);
        Assert.Contains("chunk-cache-allow-short-keys", names);
    }

    /// <summary>
    /// <c>ns=&lt;namespace&gt;</c> on a key line maps the key to that namespace instead of the derived one (the rotation path:
    /// a new key keeps the old entries); two keys may share one deliberately; a key without it keeps its own.
    /// </summary>
    [Fact]
    public async Task KeysFile_NsField_MapsKeyToNamedNamespace()
    {
        var keys = ChunkCacheKeys.Parse([KeyA + " ns=team-x", KeyB + " admin ns=team-x", KeyC], "k.txt", allowShortKeys: false);
        Assert.True(keys.TryMatch(KeyA, out bool adminA, out string? nsA) && !adminA);
        Assert.Equal("team-x", nsA);
        Assert.True(keys.TryMatch(KeyB, out bool adminB, out string? nsB) && adminB);
        Assert.Equal("team-x", nsB);
        Assert.True(keys.TryMatch(KeyC, out _, out string? nsC));
        Assert.Null(nsC);

        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root, KeysFile(KeyA + " ns=team-x", KeyB + "\tns=team-x  admin", KeyC)));
        PutChunkA(server, root, "team-x");
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        foreach (string key in new[] { KeyA, KeyB })
        {
            var e = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("exact")), Key(key))).Body)!;
            Assert.Equal("hit", Status(e, "a"));
            string producer = JsonNode.Parse((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(key))).Body)!["model"]!["producer"]!.GetValue<string>();
            Assert.Equal(identity.KeyOf([], "team-x").Hex, producer);
        }
        var c = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("exact")), Key(KeyC))).Body)!;
        Assert.Equal("miss", Status(c, "a"));
        var ctx = new DefaultHttpContext();
        ctx.Request.Headers["X-API-Key"] = KeyC;
        Assert.Equal(ChunkNamespace.ForApiKey(KeyC), ChunkCacheAuth.Resolve(ctx, server.State.Options.ChunkCache, out _).Namespace);
    }

    /// <summary>Ruling 52: <c>store</c> is always an object; <c>configured: false</c> without <c>--chunk-store</c>, absent with one.</summary>
    [Fact]
    public async Task Capabilities_StoreConfiguredFalse_WithoutStore()
    {
        await using (var none = await ChunkCacheTestServer.StartAsync(_model, On()))
        {
            var store = JsonNode.Parse((await none.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null)).Body)!["store"]!.AsObject();
            Assert.False(store["configured"]!.GetValue<bool>());
            Assert.Equal(0, store["entries"]!.GetValue<long>());
        }
        await using var with = await ChunkCacheTestServer.StartAsync(_model, On(NewStore()));
        var s2 = JsonNode.Parse((await with.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null)).Body)!["store"]!.AsObject();
        Assert.False(s2.ContainsKey("configured"));
    }

    /// <summary>
    /// CSRF hardening (before prewarm / evict): a control-plane POST must be <c>application/json</c> (else 415, so a
    /// browser must preflight); <c>Sec-Fetch-Site: cross-site</c> is refused (403) on both routes, even with a valid key.
    /// </summary>
    [Fact]
    public async Task ControlPlane_RequiresJson_RefusesCrossSite()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(keysFile: KeysFile(KeyA)));
        string body = ExplainBody(Manifest("exact"));
        Assert.Equal(415, (await PostRaw(server, "/v1/chunk-cache/explain", new StringContent(body, Encoding.UTF8, "text/plain"), Key(KeyA))).Status);
        var form = new StringContent(body, Encoding.UTF8);
        form.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/x-www-form-urlencoded");
        Assert.Equal(415, (await PostRaw(server, "/v1/chunk-cache/explain", form, Key(KeyA))).Status);
        var none = new ByteArrayContent(Encoding.UTF8.GetBytes(body));
        Assert.Equal(415, (await PostRaw(server, "/v1/chunk-cache/explain", none, Key(KeyA))).Status);
        Assert.Equal(200, (await PostRaw(server, "/v1/chunk-cache/explain", new StringContent(body, Encoding.UTF8, "application/json"), Key(KeyA))).Status);

        Assert.Equal(403, (await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", body, Key(KeyA), ("Sec-Fetch-Site", "cross-site"))).Status);
        Assert.Equal(403, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA), ("Sec-Fetch-Site", "cross-site"))).Status);
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA), ("Sec-Fetch-Site", "same-origin"))).Status);
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA), ("Sec-Fetch-Site", "none"))).Status);
    }

    /// <summary>
    /// Explain is bounded: at most <c>MaxConcurrentExplains</c> (2) run at once, the next gets 429 with <c>Retry-After</c>;
    /// a body over <c>MaxExplainBodyBytes</c> is 413, with or without a <c>Content-Length</c>.
    /// </summary>
    [Fact]
    public async Task Explain_ConcurrencyCapped_429_BodyLimited_413()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On());
        string body = ExplainBody(Manifest("exact"));
        var gate = server.State.ChunkCacheRuntime.ExplainGate;
        Assert.Equal(ChunkCacheEndpoint.MaxConcurrentExplains, gate.CurrentCount);
        Assert.True(gate.Wait(0) && gate.Wait(0));
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chunk-cache/explain") { Content = new StringContent(body, Encoding.UTF8, "application/json") };
            using var resp = await server.Http.SendAsync(request);
            Assert.Equal(HttpStatusCode.TooManyRequests, resp.StatusCode);
            Assert.NotNull(resp.Headers.RetryAfter);
        }
        finally
        {
            gate.Release(2);
        }
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", body)).Status);

        string big = "{\"request\":{\"messages\":[{\"role\":\"user\",\"content\":\"" + new string('x', ChunkCacheEndpoint.MaxExplainBodyBytes) + "\"}]}}";
        var sized = await PostRaw(server, "/v1/chunk-cache/explain", new StringContent(big, Encoding.UTF8, "application/json"));
        Assert.Equal(413, sized.Status);
        var streamed = new StreamContent(new MemoryStream(Encoding.UTF8.GetBytes(big)));
        streamed.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
        using (var chunked = new HttpRequestMessage(HttpMethod.Post, "/v1/chunk-cache/explain") { Content = streamed })
        {
            chunked.Headers.TransferEncodingChunked = true;
            using var resp = await server.Http.SendAsync(chunked);
            Assert.Equal(HttpStatusCode.RequestEntityTooLarge, resp.StatusCode);
        }
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", body)).Status);
        Assert.Equal(ChunkCacheEndpoint.MaxConcurrentExplains, gate.CurrentCount); // every path released its slot
    }

    /// <summary>The exchange's two namespace sources (the parse and the caller) must agree.</summary>
    [Fact]
    public void ExchangeBegin_NamespaceMismatch_Refused()
    {
        var parsed = new ParsedChunkRequest(ChunkPolicy.Exact, [], null, false, "summary", ChunkNamespace.ForApiKey(KeyA), [], null);
        Assert.Throws<ArgumentException>(() => ChunkCacheExchange.Begin(parsed, new ChunkCaller(true, ChunkNamespace.Default)));
        Assert.NotNull(ChunkCacheExchange.Begin(parsed, new ChunkCaller(true, ChunkNamespace.ForApiKey(KeyA))));
    }

    /// <summary>
    /// A non-admin caller sees its own namespace's figures only: with entries of another namespace in the store, its
    /// capabilities show 0 entries and 0 bytes, not the store-wide totals.
    /// </summary>
    [Fact]
    public async Task NonAdmin_CapabilitiesShowOwnNamespaceFiguresOnly()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root, KeysFile(KeyA, KeyB)));
        PutChunkA(server, root, ChunkNamespace.ForApiKey(KeyA));
        Assert.True(FileChunkStore.OpenExisting(root).TotalBytes > 0);

        var store = JsonNode.Parse((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyB))).Body)!["store"]!;
        Assert.Equal(0, store["entries"]!.GetValue<long>());
        Assert.Equal(0, store["bytes"]!.GetValue<long>());
    }

    // ------------------------------------------------------------------------------------------------ rate limiting

    /// <summary>Both routes are metered like inference: with one request per minute, the second gets 429.</summary>
    [Fact]
    public async Task Endpoints_AreRateLimited()
    {
        Assert.True(RateLimitMiddleware.IsMeteredPath(ChunkCacheEndpoint.CapabilitiesPath));
        Assert.True(RateLimitMiddleware.IsMeteredPath(ChunkCacheEndpoint.ExplainPath));

        var config = new RateLimitConfig { Enabled = true, DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 1 } };
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(),
            configure: app => app.UseDotLLMRateLimiting(new RateLimitManager(config), new HeaderApiKeyResolver()));
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null)).Status);
        Assert.Equal(429, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null)).Status);
        Assert.Equal(429, (await server.SendAsync(HttpMethod.Post, "/v1/chunk-cache/explain", ExplainBody(Manifest("exact")))).Status);
    }

    // ------------------------------------------------------------------------------------------------ inference

    /// <summary>
    /// Annotated inference is never refused, but an unauthenticated caller (keys configured, none sent) gets the summary for
    /// <c>report: "per_chunk"</c> and no store lookups (its chunk stored in the default namespace is not a hit); a key holder
    /// gets per-chunk status for its own namespace.
    /// </summary>
    [Fact]
    public async Task ReportPerChunk_UnauthenticatedGetsSummary()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(root, KeysFile(KeyA)));
        PutChunkA(server, root, ChunkNamespace.Default);
        PutChunkA(server, root, ChunkNamespace.ForApiKey(KeyA));
        string body = ChatBody(Manifest("exact", """{"report":"per_chunk"}"""));

        // A different system prompt: the default paged path reuses prefixes across requests (#1596), and the anonymous request
        // must not leave a prefix that makes the key holder's chunk a "prefix" instead of a "hit".
        string anonBody = body.Replace("You are terse.", "You are verbose.", StringComparison.Ordinal);
        Assert.NotEqual(body, anonBody);
        var anon = await server.PostAsync("/v1/chat/completions", anonBody);
        Assert.Equal(200, anon.Status);
        var ac = JsonNode.Parse(anon.Body)!["chunk_cache"]!;
        Assert.Null(ac["per_chunk"]);
        Assert.Equal(0, ac["chunks"]!["hit"]!.GetValue<int>());
        Assert.Equal(2, ac["chunks"]!["miss"]!.GetValue<int>());

        var keyed = JsonNode.Parse((await server.SendAsync(HttpMethod.Post, "/v1/chat/completions", body, Key(KeyA))).Body)!["chunk_cache"]!;
        Assert.Equal(1, keyed["chunks"]!["hit"]!.GetValue<int>());
        var perChunk = keyed["per_chunk"]!.AsArray();
        Assert.Equal(["a", "b"], perChunk.Select(c => c!["id"]!.GetValue<string>()));
        Assert.Equal(["hit", "miss"], perChunk.Select(c => c!["status"]!.GetValue<string>()));

        // Streaming carries the same per-chunk report on the finish-reason chunk.
        var stream = await server.SendAsync(HttpMethod.Post, "/v1/chat/completions",
            ChunkCacheTestServer.OpenAi(Conversation, Manifest("exact", """{"report":"per_chunk"}"""), stream: true), Key(KeyA));
        var report = ChunkCacheTestServer.SseData(stream.Body).Select(d => d["chunk_cache"]).Single(c => c is not null)!;
        Assert.Equal(2, report["per_chunk"]!.AsArray().Count);
    }

    /// <summary>Captures every <c>ILogger</c> line of a host.</summary>
    private sealed class CapturingLoggerProvider : ILoggerProvider
    {
        private readonly StringBuilder _text = new();

        public string Text
        {
            get
            {
                lock (_text) return _text.ToString();
            }
        }

        public ILogger CreateLogger(string categoryName) => new Logger(this, categoryName);

        public void Dispose()
        {
        }

        private sealed class Logger(CapturingLoggerProvider owner, string category) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                lock (owner._text)
                    owner._text.Append(category).Append(": ").Append(formatter(state, exception)).Append(' ').Append(exception).Append('\n');
            }
        }
    }
}
