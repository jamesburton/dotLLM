using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Endpoints;
using Microsoft.AspNetCore.Builder;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// The write side of the chunk-cache control plane (Phase I Task 16, spec §7.2, §10): <c>POST /v1/chunk-cache/prewarm</c>
/// (text only, bypasses only the seen-twice rule, deny-list on <c>source.path</c>, caps) and <c>POST /v1/chunk-cache/evict</c>
/// (<c>orphans | path | namespace | all</c>, <c>dry_run</c> by default, forget of the four kinds, fail closed), namespaces
/// and admin keys, capabilities store figures, and the per-namespace budget, over real HTTP on the synthetic model (CPU).
/// </summary>
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkCacheWriteEndpointTests : IDisposable
{
    private const string KeyA = "test-key-alpha-0123456789abcdef0123";
    private const string KeyB = "test-key-bravo-9876543210fedcba9876";
    private const string KeyAdmin = "test-key-admin-5555555555aaaaaaaa55";

    private const string Prewarm = "/v1/chunk-cache/prewarm";
    private const string Evict = "/v1/chunk-cache/evict";

    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string ChunkB =
        "public static int Mul(int a, int b)\n{\n    return a * b; // the product of both operands\n}\n";

    private static readonly string ChunkC =
        "public static int Sub(int a, int b)\n{\n    return a - b; // the difference of both operands\n}\n";

    private static readonly string ChunkD =
        "public static int Div(int a, int b)\n{\n    return a / b; // the quotient of both operands\n}\n";

    private static readonly TimeSpan Drain = TimeSpan.FromSeconds(60);

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-write-ep-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string _model;

    public ChunkCacheWriteEndpointTests() => _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private string NewStore()
    {
        string root = Path.Combine(_scratch, "store-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        return root;
    }

    private string KeysFile(params string[] lines)
    {
        Directory.CreateDirectory(_scratch);
        string path = Path.Combine(_scratch, "keys-" + Guid.NewGuid().ToString("N")[..8] + ".txt");
        File.WriteAllLines(path, lines);
        return path;
    }

    private static ChunkCacheOptions Writing(string root, string? keysFile = null, string budget = "1GiB", string? namespaceBudget = null) =>
        ChunkCacheOptions.Create(true, null, root, keysFile, storeWrite: true, storeBudget: budget, storeNamespaceBudget: namespaceBudget);

    private static (string, string) Key(string key) => ("X-API-Key", key);

    private static string Ns(string key) => ChunkNamespace.ForApiKey(key);

    private static ChunkStoreService Writer(ChunkCacheTestServer server) => server.State.ChunkCacheRuntime.Writer!;

    private static ChunkKey KeyOf(ChunkCacheTestServer server, string text, string ns)
    {
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        Assert.True(identity.Supported, identity.UnsupportedReason);
        return identity.KeyOf(ChunkTokenization.Encode(server.State.Tokenizer!, text), ns);
    }

    private static int TokensOf(ChunkCacheTestServer server, string text) => ChunkTokenization.Encode(server.State.Tokenizer!, text).Length;

    /// <summary>Encodes <paramref name="text"/> straight into the store (a CLI <c>--offline</c> entry: no ledger row).</summary>
    private static ChunkKey PutDirect(ChunkCacheTestServer server, string root, string text, string ns)
    {
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        int[] ids = ChunkTokenization.Encode(server.State.Tokenizer!, text);
        var key = identity.KeyOf(ids, ns);
        new FileChunkStore(root).Put(key, new ChunkEncoder(identity.Fingerprint).Encode(server.State.Model!, ids));
        return key;
    }

    private static bool Stored(string root, ChunkKey key) => File.Exists(FileChunkStore.EntryPath(root, key.Hex));

    private static long SizeOf(string root, ChunkKey key) => new FileInfo(FileChunkStore.EntryPath(root, key.Hex)).Length;

    private static HashSet<string> LedgerKeys(string root) =>
        File.Exists(ServerLedger.DefaultPath(root))
            ? ServerLedger.Load(root).Rows.Select(r => r.KeyHex).ToHashSet(StringComparer.Ordinal)
            : [];

    /// <summary>The index file of <paramref name="ns"/> for the served model (Task 4b name), referencing <paramref name="keys"/>.</summary>
    private static string WriteNamespaceIndex(ChunkCacheTestServer server, string root, string ns, params ChunkKey[] keys)
    {
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        string path = ChunkIndex.DefaultPath(root, "symbol-512", identity.ModelId, identity.KeyOf([], ns).Hex, ns);
        WriteIndexAt(path, keys);
        return path;
    }

    private static void WriteIndexAt(string path, params ChunkKey[] keys)
    {
        var chunks = new JsonArray(keys.Select(k => (JsonNode)new JsonObject
        {
            ["startLine"] = 1, ["endLine"] = 2, ["startOffset"] = 0, ["endOffset"] = 1, ["tokens"] = 1, ["key"] = k.Hex,
        }).ToArray());
        var doc = new JsonObject
        {
            ["formatVersion"] = 1,
            ["producer"] = new string('0', 64),
            ["chunker"] = "symbol-512",
            ["files"] = new JsonArray(new JsonObject { ["path"] = "src/a.cs", ["sha256"] = new string('0', 64), ["chunks"] = chunks }),
        };
        File.WriteAllText(path, doc.ToJsonString());
    }

    private static string PrewarmBody(params (string Text, string? Path)[] items) => new JsonObject
    {
        ["texts"] = new JsonArray(items.Select((t, i) =>
        {
            var o = new JsonObject { ["id"] = "c" + i, ["text"] = t.Text };
            if (t.Path is not null)
                o["source"] = new JsonObject { ["path"] = t.Path };
            return (JsonNode)o;
        }).ToArray()),
    }.ToJsonString();

    private static async Task<JsonNode> PrewarmOk(ChunkCacheTestServer server, string body, params (string, string)[] headers)
    {
        var (status, text) = await server.SendAsync(HttpMethod.Post, Prewarm, body, headers);
        Assert.True(status == 200, $"prewarm {status}: {text}");
        return JsonNode.Parse(text)!;
    }

    private static async Task<JsonNode> EvictOk(ChunkCacheTestServer server, string body, params (string, string)[] headers)
    {
        var (status, text) = await server.SendAsync(HttpMethod.Post, Evict, body, headers);
        Assert.True(status == 200, $"evict {status}: {text}");
        return JsonNode.Parse(text)!;
    }

    /// <summary>Prewarms <paramref name="items"/> and waits until the writer has stored them.</summary>
    private static async Task Store(ChunkCacheTestServer server, (string, string)[] headers, params (string Text, string? Path)[] items)
    {
        var r = await PrewarmOk(server, PrewarmBody(items), headers);
        Assert.Equal(items.Length, r["queued"]!.GetValue<int>());
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
    }

    private static int Int(JsonNode node, string name) => node[name]!.GetValue<int>();

    private static long Long(JsonNode node, string name) => node[name]!.GetValue<long>();

    private static string[] Strings(JsonNode? array) => array?.AsArray().Select(x => x!.GetValue<string>()).ToArray() ?? [];

    private static Action<WebApplication> From(string address) => app => app.Use((ctx, next) =>
    {
        ctx.Connection.RemoteIpAddress = IPAddress.Parse(address);
        return next(ctx);
    });

    // ------------------------------------------------------------------------------------------------ prewarm

    /// <summary>
    /// Prewarm takes text only, bypasses only the seen-twice rule (the first and only call queues), records no sighting,
    /// stores a ledger row, and reports <c>already_stored</c> on a repeat. Hash lists and hash-only items are refused.
    /// </summary>
    [Fact]
    public async Task Prewarm_TextOnly_QueuesWithoutSightings()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var key = KeyOf(server, ChunkA, ChunkNamespace.Default);

        var r = await PrewarmOk(server, PrewarmBody((ChunkA, "src/A.cs")));
        Assert.Equal(1, Int(r, "requested"));
        Assert.Equal(0, Int(r, "already_stored"));
        Assert.Equal(1, Int(r, "queued"));
        Assert.Equal(TokensOf(server, ChunkA), Int(r, "queued_tokens"));
        Assert.Equal(0, Int(r, "refused"));
        Assert.Empty(Strings(r["refused_reasons"]));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.True(Stored(root, key));
        Assert.Contains(key.Hex, LedgerKeys(root));
        Assert.Equal(0, Writer(server).Sightings.Count);
        Assert.Equal(0, Writer(server).Sightings.CountOf(ChunkNamespace.Default, WireChunking.Hash(ChunkA)));
        Assert.Equal("src/A.cs", ServerLedger.Load(root).Rows.Single().SourcePath);

        var again = await PrewarmOk(server, PrewarmBody((ChunkA, null)));
        Assert.Equal(1, Int(again, "already_stored"));
        Assert.Equal(0, Int(again, "queued"));

        foreach (string bad in new[]
        {
            """{"hashes":["sha256:0000000000000000000000000000000000000000000000000000000000000000"]}""",
            """{"texts":[{"id":"x","hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000"}]}""",
            """{"texts":[{"id":"x"}]}""",
            """{"texts":[{"text":42}]}""",
            """{"texts":"abc"}""",
            """[]""",
            "not json",
        })
        {
            var (status, body) = await server.SendAsync(HttpMethod.Post, Prewarm, bad);
            Assert.True(status == 400, $"{bad}: {status} {body}");
            Assert.Equal("invalid_request_error", JsonNode.Parse(body)!["error"]!["type"]!.GetValue<string>());
        }
    }

    /// <summary>
    /// The deny-list applies to <c>source.path</c> and the PEM sniff to the text: refused, counted with reasons, never stored;
    /// an allowed item in the same call is queued.
    /// </summary>
    [Fact]
    public async Task Prewarm_DeniedPath_Refused()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        string pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAx0123456789abcdefghijklmnopqrstuvwxyz\n-----END RSA PRIVATE KEY-----\n";
        var r = await PrewarmOk(server, PrewarmBody((ChunkA, ".env"), (ChunkB, "config/secrets.json"), (pem, "src/Key.cs"), (ChunkC, "src/C.cs")));
        Assert.Equal(4, Int(r, "requested"));
        Assert.Equal(3, Int(r, "refused"));
        Assert.Equal(1, Int(r, "queued"));
        Assert.Equal(["denied_path", "secret_content"], Strings(r["refused_reasons"]).Order(StringComparer.Ordinal));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.False(Stored(root, KeyOf(server, ChunkA, "")));
        Assert.False(Stored(root, KeyOf(server, ChunkB, "")));
        Assert.True(Stored(root, KeyOf(server, ChunkC, "")));
        Assert.Single(LedgerKeys(root));
    }

    /// <summary>
    /// Caps: more than <c>MaxPrewarmItems</c> items is a 400 (<c>chunk_too_many</c>), a body over the byte cap a 413, a
    /// chunk over the 512-token CPU cap is refused (<c>too_large_for_cpu</c>, ruling 65), and <c>max_tokens</c> bounds the
    /// tokens one call queues (the rest refused <c>max_tokens</c>).
    /// </summary>
    [Fact]
    public async Task Prewarm_Caps_ItemsBytesTokens()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));

        var many = Enumerable.Range(0, ChunkCacheEndpoint.MaxPrewarmItems + 1).Select(i => ("text " + i, (string?)null)).ToArray();
        var (s1, b1) = await server.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody(many));
        Assert.Equal(400, s1);
        Assert.Equal(ChunkCacheCodes.TooMany, JsonNode.Parse(b1)!["error"]!["code"]!.GetValue<string>());

        string huge = new('x', ChunkCacheEndpoint.MaxPrewarmBodyBytes + 10);
        var (s2, _) = await server.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((huge, null)));
        Assert.Equal(413, s2);

        string big = string.Concat(Enumerable.Repeat(ChunkA, 8));
        Assert.True(TokensOf(server, big) > ChunkStoreService.MaxCpuEncodeTokens);
        var r = await PrewarmOk(server, PrewarmBody((big, null)));
        Assert.Equal(1, Int(r, "refused"));
        Assert.Equal(["too_large_for_cpu"], Strings(r["refused_reasons"]));

        var body = JsonNode.Parse(PrewarmBody((ChunkA, null), (ChunkB, null)))!;
        body["max_tokens"] = TokensOf(server, ChunkA);
        var capped = await PrewarmOk(server, body.ToJsonString());
        Assert.Equal(1, Int(capped, "queued"));
        Assert.Equal(1, Int(capped, "refused"));
        Assert.Equal(["max_tokens"], Strings(capped["refused_reasons"]));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.True(Stored(root, KeyOf(server, ChunkA, "")));
        Assert.False(Stored(root, KeyOf(server, ChunkB, "")));
    }

    /// <summary>
    /// Security gates on both write routes: flag off 404; no <c>--chunk-store-write</c> 503; cross-site 403 and a non-JSON
    /// content type 415 (CSRF, Task 14 S6); keys configured without a key 401; no keys and a remote caller 403. None changes
    /// the store.
    /// </summary>
    [Fact]
    public async Task WriteRoutes_FlagWriterCsrfAndAuthGates()
    {
        await using (var off = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Disabled))
        {
            Assert.Equal(404, (await off.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((ChunkA, null)))).Status);
            Assert.Equal(404, (await off.SendAsync(HttpMethod.Post, Evict, "{}")).Status);
        }

        string readOnly = NewStore();
        await using (var ro = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Create(true, null, readOnly)))
        {
            var (sp, bp) = await ro.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((ChunkA, null)));
            Assert.Equal(503, sp);
            Assert.Equal(ChunkCacheCodes.StoreUnavailable, JsonNode.Parse(bp)!["error"]!["code"]!.GetValue<string>());
            Assert.Equal(503, (await ro.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""")).Status);
        }

        string root = NewStore();
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA))))
        {
            await Store(server, [Key(KeyA)], (ChunkA, "src/A.cs"));
            var key = KeyOf(server, ChunkA, Ns(KeyA));
            string evictAll = """{"mode":"namespace","dry_run":false}""";
            foreach (string path in new[] { Prewarm, Evict })
            {
                string body = path == Prewarm ? PrewarmBody((ChunkB, null)) : evictAll;
                Assert.Equal(403, (await server.SendAsync(HttpMethod.Post, path, body, Key(KeyA), ("Sec-Fetch-Site", "cross-site"))).Status);
                using (var plain = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, "text/plain") })
                {
                    plain.Headers.TryAddWithoutValidation("X-API-Key", KeyA);
                    using var resp = await server.Http.SendAsync(plain);
                    Assert.Equal(415, (int)resp.StatusCode);
                }
                Assert.Equal(401, (await server.SendAsync(HttpMethod.Post, path, body)).Status);
                Assert.Equal(401, (await server.SendAsync(HttpMethod.Post, path, body, Key(KeyA + "x"))).Status);
            }
            Assert.True(Writer(server).Queue.WaitDrained(Drain));
            Assert.True(Stored(root, key));
            Assert.False(Stored(root, KeyOf(server, ChunkB, Ns(KeyA))));
        }

        string remote = NewStore();
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(remote), configure: From("10.1.2.3")))
        {
            Assert.Equal(403, (await server.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((ChunkA, null)))).Status);
            Assert.Equal(403, (await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"all","dry_run":false}""")).Status);
            Assert.True(Writer(server).Queue.WaitDrained(Drain));
            Assert.Empty(new FileChunkStore(remote).Keys());
        }
    }

    // ------------------------------------------------------------------------------------------------ evict

    /// <summary>
    /// <c>dry_run</c> defaults to true: the counts of what would go, and nothing deleted (file, ledger row, sighting stay);
    /// only an explicit JSON <c>false</c> deletes; a non-boolean <c>dry_run</c>, an unknown mode or field is a 400.
    /// </summary>
    [Fact]
    public async Task Evict_DryRunDefault_DeletesNothing()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        await Store(server, [], (ChunkA, "src/A.cs"));
        var key = KeyOf(server, ChunkA, "");
        Writer(server).Sightings.See("", WireChunking.Hash(ChunkB), "src/B.cs");
        string ledgerBefore = File.ReadAllText(ServerLedger.DefaultPath(root));

        var dry = await EvictOk(server, """{"mode":"namespace"}""");
        Assert.True(dry["dry_run"]!.GetValue<bool>());
        Assert.Equal(1, Int(dry, "evicted"));
        Assert.Equal(SizeOf(root, key), Long(dry, "freed_bytes"));
        Assert.Equal(0, Int(dry, "kept_referenced"));
        Assert.True(Stored(root, key));
        Assert.Equal(ledgerBefore, File.ReadAllText(ServerLedger.DefaultPath(root)));
        Assert.Equal(1, Writer(server).Sightings.CountOf("", WireChunking.Hash(ChunkB)));

        var defaults = await EvictOk(server, "{}");
        Assert.True(defaults["dry_run"]!.GetValue<bool>());
        Assert.True(Stored(root, key));

        foreach (string bad in new[]
        {
            """{"mode":"namespace","dry_run":"false"}""",
            """{"mode":"namespace","dry_run":0}""",
            """{"mode":"everything","dry_run":false}""",
            """{"mode":"all-for-model","dry_run":false}""",
            """{"mode":"namespace","dry_run":false,"keys":["x"]}""",
            """{"mode":"path","dry_run":false}""",
            """{"mode":"path","path":"../x","dry_run":false}""",
            """{"mode":"path","path":"!src/**","dry_run":false}""",
            """{"mode":"namespace","path":"src/**","dry_run":false}""",
            """{"mode":"namespace","dry_run":null}""",
            """{"mode":"namespace","dry_run":true,"dry_run":false}""",
            "{\"mode\":\"path\",\"dry_run\":false,\"path\":\"src/" + new string('a', 1100) + "\"}",
        })
        {
            var (status, body) = await server.SendAsync(HttpMethod.Post, Evict, bad);
            Assert.True(status == 400, $"{bad}: {status} {body}");
        }
        Assert.True(Stored(root, key));
        Assert.Equal(ledgerBefore, File.ReadAllText(ServerLedger.DefaultPath(root)));

        var real = await EvictOk(server, """{"mode":"namespace","dry_run":false}""");
        Assert.False(real["dry_run"]!.GetValue<bool>());
        Assert.Equal(1, Int(real, "evicted"));
        Assert.False(Stored(root, key));
    }

    /// <summary>
    /// <c>orphans</c> deletes only files that neither an index nor the server ledger references (store-wide, so admin only:
    /// a local caller without keys is admin; a keyed non-admin gets 403).
    /// </summary>
    [Fact]
    public async Task Evict_Orphans_KeepsLedgerAndIndexKeys()
    {
        string root = NewStore();
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            await Store(server, [], (ChunkA, null));
            var ledgerKey = KeyOf(server, ChunkA, "");
            var indexed = PutDirect(server, root, ChunkB, "");
            WriteIndexAt(Path.Combine(root, "index-symbol-512.json"), indexed);
            var orphan = PutDirect(server, root, ChunkC, "");
            long orphanBytes = SizeOf(root, orphan);

            var dry = await EvictOk(server, """{"mode":"orphans"}""");
            Assert.Equal(1, Int(dry, "evicted"));
            Assert.True(Stored(root, orphan));
            Assert.Equal(orphanBytes, Long(dry, "freed_bytes"));
            var (refused, refusedBody) = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"orphans","dry_run":false}""");
            Assert.Equal(403, refused);
            Assert.Equal("chunk_cache_admin_key_required", JsonNode.Parse(refusedBody)!["error"]!["code"]!.GetValue<string>());
            Assert.True(Stored(root, orphan));
        }

        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyAdmin + " admin"))))
        {
            var ledgerKey = KeyOf(server, ChunkA, "");
            var indexed = KeyOf(server, ChunkB, "");
            var orphan = KeyOf(server, ChunkC, "");
            long orphanBytes = SizeOf(root, orphan);
            var real = await EvictOk(server, """{"mode":"orphans","dry_run":false}""", Key(KeyAdmin));
            Assert.Equal(1, Int(real, "evicted"));
            Assert.Equal(orphanBytes, Long(real, "freed_bytes"));
            Assert.False(Stored(root, orphan));
            Assert.True(Stored(root, ledgerKey));
            Assert.True(Stored(root, indexed));
            Assert.Contains(ledgerKey.Hex, LedgerKeys(root));
        }

        string keyed = NewStore();
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(keyed, KeysFile(KeyA, KeyAdmin + " admin"))))
        {
            var orphan = PutDirect(server, keyed, ChunkC, "");
            var (status, body) = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"orphans","dry_run":false}""", Key(KeyA));
            Assert.Equal(403, status);
            Assert.Equal("permission_error", JsonNode.Parse(body)!["error"]!["type"]!.GetValue<string>());
            Assert.True(Stored(keyed, orphan));
            var admin = await EvictOk(server, """{"mode":"orphans","dry_run":false}""", Key(KeyAdmin));
            Assert.Equal(1, Int(admin, "evicted"));
            Assert.False(Stored(keyed, orphan));
        }
    }

    /// <summary>
    /// <c>path</c> ("forget this repo") removes, in the caller's namespace, the K/V and ledger rows whose provenance matches
    /// the glob and the sighting rows with that provenance; an entry a (CLI) index still references keeps its file and is
    /// counted in <c>kept_referenced</c> (the server owns no index references). Other paths and namespaces are untouched.
    /// </summary>
    [Fact]
    public async Task Evict_Path_ForgetsAllFourKinds()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA, KeyB)));
        await Store(server, [Key(KeyA)], (ChunkA, "src/A.cs"), (ChunkB, "lib/B.cs"), (ChunkC, "src/sub/C.cs"));
        await Store(server, [Key(KeyB)], (ChunkA, "src/A.cs"));
        string nsA = Ns(KeyA), nsB = Ns(KeyB);
        var a = KeyOf(server, ChunkA, nsA);
        var b = KeyOf(server, ChunkB, nsA);
        var c = KeyOf(server, ChunkC, nsA);
        var otherA = KeyOf(server, ChunkA, nsB);
        WriteNamespaceIndex(server, root, nsA, c); // a CLI index in A's namespace still references C
        var sightings = Writer(server).Sightings;
        sightings.See(nsA, WireChunking.Hash(ChunkD), "src/D.cs");
        sightings.See(nsA, WireChunking.Hash(ChunkD + "x"), "lib/E.cs");
        sightings.See(nsB, WireChunking.Hash(ChunkD), "src/D.cs");
        string evict = """{"mode":"path","path":"src/**","dry_run":false}""";

        var dry = await EvictOk(server, """{"mode":"path","path":"src/**"}""", Key(KeyA));
        Assert.Equal(1, Int(dry, "evicted"));
        Assert.Equal(1, Int(dry, "kept_referenced"));
        Assert.True(Stored(root, a));

        var r = await EvictOk(server, evict, Key(KeyA));
        Assert.Equal(1, Int(r, "evicted"));
        Assert.Equal(1, Int(r, "kept_referenced"));
        Assert.False(r["dry_run"]!.GetValue<bool>());
        Assert.False(Stored(root, a));               // K/V
        Assert.True(Stored(root, c));                // referenced by a CLI index: file kept, counted
        Assert.True(Stored(root, b));                // other path
        Assert.True(Stored(root, otherA));           // other namespace
        var ledger = LedgerKeys(root);
        Assert.DoesNotContain(a.Hex, ledger);        // ledger rows
        Assert.DoesNotContain(c.Hex, ledger);
        Assert.Contains(b.Hex, ledger);
        Assert.Contains(otherA.Hex, ledger);
        Assert.Equal(0, sightings.CountOf(nsA, WireChunking.Hash(ChunkD)));        // sightings
        Assert.Equal(1, sightings.CountOf(nsA, WireChunking.Hash(ChunkD + "x")));
        Assert.Equal(1, sightings.CountOf(nsB, WireChunking.Hash(ChunkD)));
    }

    /// <summary>
    /// Forget compares <c>StoredPath</c> of the requested path (review: a path over 256 characters is stored as a prefix
    /// plus a hash, so the raw path would never match it).
    /// </summary>
    [Fact]
    public async Task Evict_Path_LongPath_MatchesStoredForm()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        string longPath = "src/" + string.Concat(Enumerable.Repeat("very-long-directory-name/", 14)) + "File.cs";
        Assert.True(longPath.Length > SightingTable.StoredPathChars);
        var sightings = Writer(server).Sightings;
        sightings.See("", WireChunking.Hash(ChunkD), longPath);
        sightings.See("", WireChunking.Hash(ChunkC), longPath[..^7] + "Other.cs");
        var body = new JsonObject { ["mode"] = "path", ["path"] = longPath, ["dry_run"] = false }.ToJsonString();
        await EvictOk(server, body);
        Assert.Equal(0, sightings.CountOf("", WireChunking.Hash(ChunkD)));
        Assert.Equal(1, sightings.CountOf("", WireChunking.Hash(ChunkC)));
    }

    /// <summary>
    /// <c>namespace</c> and a non-admin's <c>all</c> act on the caller's namespace only: the other key's entries, rows and
    /// sightings stay (TwoKeys…Evict, carried from Task 14).
    /// </summary>
    [Fact]
    public async Task Evict_Namespace_CannotTouchOtherNamespace()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA, KeyB)));
        await Store(server, [Key(KeyA)], (ChunkA, null));
        await Store(server, [Key(KeyB)], (ChunkA, null), (ChunkB, null));
        var a = KeyOf(server, ChunkA, Ns(KeyA));
        Writer(server).Sightings.See(Ns(KeyA), WireChunking.Hash(ChunkD), null);

        foreach (string mode in new[] { "namespace", "all" })
        {
            var r = await EvictOk(server, $$"""{"mode":"{{mode}}","dry_run":false}""", Key(KeyB));
            Assert.Equal(mode == "namespace" ? 2 : 0, Int(r, "evicted"));
            Assert.True(Stored(root, a));
            Assert.Contains(a.Hex, LedgerKeys(root));
            Assert.Equal(1, Writer(server).Sightings.CountOf(Ns(KeyA), WireChunking.Hash(ChunkD)));
        }
        Assert.False(Stored(root, KeyOf(server, ChunkA, Ns(KeyB))));
        Assert.False(Stored(root, KeyOf(server, ChunkB, Ns(KeyB))));
        Assert.Single(LedgerKeys(root));
    }

    /// <summary>
    /// A non-admin's body <c>namespace</c> is ignored (own namespace, warning <c>chunk_namespace_ignored</c>); an admin key's
    /// is honoured (<c>default</c> names the default namespace; an invalid name is a 400); an admin <c>all</c> without a
    /// namespace wipes the store (orphans too).
    /// </summary>
    [Fact]
    public async Task NonAdmin_EvictBodyNamespace_Ignored()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA, KeyB, KeyAdmin + " admin")));
        await Store(server, [Key(KeyA)], (ChunkA, null));
        await Store(server, [Key(KeyB)], (ChunkB, null));
        var a = KeyOf(server, ChunkA, Ns(KeyA));
        var b = KeyOf(server, ChunkB, Ns(KeyB));

        foreach (string claimed in new[] { "default", "", "Default" })
        {
            var ignored = await EvictOk(server, $$"""{"mode":"namespace","namespace":"{{claimed}}"}""", Key(KeyB));
            Assert.Contains(ChunkCacheCodes.NamespaceIgnored, Strings(ignored["warnings"]));
            Assert.Equal(1, Int(ignored, "evicted")); // B's own entry, never the default namespace's
        }
        var r = await EvictOk(server, $$"""{"mode":"namespace","namespace":"{{Ns(KeyA)}}","dry_run":false}""", Key(KeyB));
        Assert.Contains(ChunkCacheCodes.NamespaceIgnored, Strings(r["warnings"]));
        Assert.True(Stored(root, a));
        Assert.False(Stored(root, b)); // B's own namespace

        // Orphans figures are store-wide: a keyed non-admin is refused even for a dry run (Task 16 fix round, U9).
        var orphansDry = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"orphans"}""", Key(KeyB));
        Assert.Equal(403, orphansDry.Status);
        Assert.Equal("chunk_cache_admin_only", JsonNode.Parse(orphansDry.Body)!["error"]!["code"]!.GetValue<string>());

        Assert.Equal(400, (await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","namespace":"Bad Name","dry_run":false}""", Key(KeyAdmin))).Status);
        var admin = await EvictOk(server, $$"""{"mode":"namespace","namespace":"{{Ns(KeyA)}}","dry_run":false}""", Key(KeyAdmin));
        Assert.Empty(Strings(admin["warnings"]));
        Assert.Equal(1, Int(admin, "evicted"));
        Assert.False(Stored(root, a));

        var d = PutDirect(server, root, ChunkC, "");
        Assert.EndsWith(".default.json", WriteNamespaceIndex(server, root, ChunkNamespace.Default, d), StringComparison.Ordinal);
        var dflt = await EvictOk(server, """{"mode":"namespace","namespace":"default"}""", Key(KeyAdmin));
        Assert.Equal(1, Int(dflt, "evicted"));

        await Store(server, [Key(KeyA)], (ChunkA, null));
        var orphan = PutDirect(server, root, ChunkD, "");
        var wipe = await EvictOk(server, """{"mode":"all","dry_run":false}""", Key(KeyAdmin));
        Assert.Equal(3, Int(wipe, "evicted"));
        Assert.Empty(new FileChunkStore(root).Keys());
        Assert.Empty(LedgerKeys(root));
        Assert.Empty(Directory.EnumerateFiles(root, "index-*.json"));
        Assert.False(Stored(root, orphan));
    }

    /// <summary>
    /// Fail closed: a ledger that cannot be read (corrupted while the server runs, or at startup) refuses every evict,
    /// dry runs included, with 503 <c>chunk_store_unavailable</c>, and deletes nothing (file bytes unchanged).
    /// </summary>
    [Fact]
    public async Task Evict_UnreadableLedger_FailsClosed()
    {
        string root = NewStore();
        ChunkKey key;
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            await Store(server, [], (ChunkA, null));
            key = KeyOf(server, ChunkA, "");
            var orphan = PutDirect(server, root, ChunkC, "");
            File.WriteAllText(ServerLedger.DefaultPath(root), "{ not a ledger");
            foreach (string body in new[] { """{"mode":"namespace","dry_run":false}""", """{"mode":"orphans"}""",
                         """{"mode":"all"}""", """{"mode":"namespace"}""", """{"mode":"path","path":"src/**","dry_run":false}""" })
            {
                var (status, text) = await server.SendAsync(HttpMethod.Post, Evict, body);
                Assert.True(status == 503, $"{body}: {status} {text}");
                Assert.Equal(ChunkCacheCodes.StoreUnavailable, JsonNode.Parse(text)!["error"]!["code"]!.GetValue<string>());
            }
            Assert.True(Stored(root, key));
            Assert.True(Stored(root, orphan));
            Assert.Equal("{ not a ledger", File.ReadAllText(ServerLedger.DefaultPath(root)));
        }

        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            Assert.True(Writer(server).WritesDisabled);
            var (status, _) = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""");
            Assert.Equal(503, status);
            Assert.True(Stored(root, key));
            Assert.Equal("{ not a ledger", File.ReadAllText(ServerLedger.DefaultPath(root)));
        }
    }

    /// <summary>
    /// An unreadable index file also fails closed (nothing is provably unreferenced): 503, nothing deleted.
    /// </summary>
    [Fact]
    public async Task Evict_UnreadableIndex_FailsClosed()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        await Store(server, [], (ChunkA, null));
        var key = KeyOf(server, ChunkA, "");
        File.WriteAllText(Path.Combine(root, "index-broken.json"), "{ nope");
        foreach (string body in new[] { """{"mode":"namespace","dry_run":false}""", """{"mode":"orphans"}""" })
        {
            var (status, _) = await server.SendAsync(HttpMethod.Post, Evict, body);
            Assert.Equal(503, status);
        }
        Assert.True(Stored(root, key));
        Assert.Contains(key.Hex, LedgerKeys(root));
    }

    /// <summary>
    /// The store lock held by another writer: evict answers 503 within its lock timeout without deleting anything, and a
    /// concurrent inference request on the same server is not held up (the evict waits asynchronously).
    /// </summary>
    [Fact]
    public async Task Evict_StoreLockHeld_LockTimeout_NothingDeleted()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        await Store(server, [], (ChunkA, null));
        var key = KeyOf(server, ChunkA, "");
        Writer(server).LockTimeout = TimeSpan.FromSeconds(2);
        using (ChunkIndex.AcquireStoreLock(root, TimeSpan.FromSeconds(5)))
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (status, body) = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""");
            Assert.Equal(503, status);
            Assert.Contains("lock", body, StringComparison.OrdinalIgnoreCase);
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(15));
        }
        Assert.True(Stored(root, key));
        Assert.Contains(key.Hex, LedgerKeys(root));
    }

    /// <summary>
    /// <c>namespace</c> removes that namespace's index files (Task 4b names), its ledger rows, the entries only they
    /// referenced and its sighting rows; another namespace's index file, entries and sightings stay; an entry an index of
    /// another (legacy) name still references keeps its file (<c>kept_referenced</c>).
    /// </summary>
    [Fact]
    public async Task EvictNamespaceMode_RemovesThatNamespacesIndexFilesRowsAndSightingsOnly()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA, KeyB)));
        string nsA = Ns(KeyA), nsB = Ns(KeyB);
        await Store(server, [Key(KeyA)], (ChunkA, null));
        var serverA = KeyOf(server, ChunkA, nsA);
        var cliA = PutDirect(server, root, ChunkB, nsA);
        var legacyA = PutDirect(server, root, ChunkC, nsA);
        var cliB = PutDirect(server, root, ChunkB, nsB);
        string indexA = WriteNamespaceIndex(server, root, nsA, cliA, legacyA);
        string indexB = WriteNamespaceIndex(server, root, nsB, cliB);
        string legacy = Path.Combine(root, "index-symbol-512.json");
        WriteIndexAt(legacy, legacyA);
        var sightings = Writer(server).Sightings;
        sightings.See(nsA, WireChunking.Hash(ChunkD), null);
        sightings.See(nsB, WireChunking.Hash(ChunkD), null);

        long freed = SizeOf(root, serverA) + SizeOf(root, cliA);
        var dry = await EvictOk(server, """{"mode":"namespace"}""", Key(KeyA));
        Assert.Equal(2, Int(dry, "evicted"));
        Assert.True(File.Exists(indexA));

        var r = await EvictOk(server, """{"mode":"namespace","dry_run":false}""", Key(KeyA));
        Assert.Equal(2, Int(r, "evicted"));
        Assert.Equal(1, Int(r, "kept_referenced"));
        Assert.Equal(freed, Long(r, "freed_bytes"));
        Assert.False(File.Exists(indexA));
        Assert.True(File.Exists(indexB));
        Assert.True(File.Exists(legacy));
        Assert.False(Stored(root, serverA));
        Assert.False(Stored(root, cliA));
        Assert.True(Stored(root, legacyA));
        Assert.True(Stored(root, cliB));
        Assert.Empty(LedgerKeys(root));
        Assert.Equal(0, sightings.CountOf(nsA, WireChunking.Hash(ChunkD)));
        Assert.Equal(1, sightings.CountOf(nsB, WireChunking.Hash(ChunkD)));
        Assert.Equal(1, r["removed"]!["index_files"]!.GetValue<int>());
        Assert.Equal(1, r["removed"]!["ledger_rows"]!.GetValue<int>());
        Assert.Equal(1, r["removed"]!["sightings"]!.GetValue<int>());
    }

    /// <summary>
    /// Forget clears the sighting rows and leaves no marker: after a path forget the next request is a first sighting again
    /// (skipped), the one after it is queued and stored; the ledger file holds only ordinary rows.
    /// </summary>
    [Fact]
    public async Task Forget_ClearsSightings_NoMarker()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        string user = "Read this file.\n// A.cs\n" + ChunkA + "What does Add return?";
        var entry = ChunkCacheTestServer.Entry("a", 1, user, ChunkA);
        entry["source"] = new JsonObject { ["path"] = "src/A.cs" };
        string chat = ChunkCacheTestServer.OpenAi([("system", "You are terse."), ("user", user)],
            new JsonObject { ["v"] = 1, ["policy"] = "exact", ["chunks"] = new JsonArray(entry) });
        string hash = WireChunking.Hash(ChunkA);
        var key = KeyOf(server, ChunkA, "");

        Assert.Equal(200, (await server.PostAsync("/v1/chat/completions", chat)).Status);
        Assert.Equal(1, Writer(server).Sightings.CountOf("", hash));
        await EvictOk(server, """{"mode":"path","path":"src/A.cs","dry_run":false}""");
        Assert.Equal(0, Writer(server).Sightings.CountOf("", hash));

        var (_, b1) = await server.PostAsync("/v1/chat/completions", chat);
        Assert.Contains(ChunkCacheCodes.EncodeSkipped, b1, StringComparison.Ordinal);
        Assert.Equal(1, Writer(server).Sightings.CountOf("", hash));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.False(Stored(root, key));
        var (_, b2) = await server.PostAsync("/v1/chat/completions", chat);
        Assert.Contains(ChunkCacheCodes.EncodeQueued, b2, StringComparison.Ordinal);
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.True(Stored(root, key));

        await EvictOk(server, """{"mode":"path","path":"src/**","dry_run":false}""");
        Assert.False(Stored(root, key));
        string ledgerText = File.ReadAllText(ServerLedger.DefaultPath(root));
        Assert.DoesNotContain("forgot", ledgerText, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(ServerLedger.Load(root).Rows);
    }

    /// <summary>
    /// Forget also drops what is queued (and an in-flight encode of that namespace): nothing of the forgotten namespace is
    /// stored after the evict returns.
    /// </summary>
    [Fact]
    public async Task Forget_DropsQueuedAndInFlightItems()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var state = server.State;
        state.ChunkCacheRuntime.Dispose();
        var slow = new ChunkStoreServiceTests.FakeHost { Block = new ManualResetEventSlim(false) };
        var writer = state.ChunkCacheRuntime.StartWriter(state, slow)!;
        writer.Queue.QuietPeriod = TimeSpan.Zero;

        var r = await PrewarmOk(server, PrewarmBody((ChunkA, "src/A.cs"), (ChunkB, "src/B.cs")));
        Assert.Equal(2, Int(r, "queued"));
        Assert.True(slow.Entered.Wait(TimeSpan.FromSeconds(10))); // ChunkA in flight, ChunkB queued
        var evictTask = server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""");
        await Task.Delay(300);
        slow.Block.Set();
        var (status, body) = await evictTask;
        Assert.True(status == 200, body);
        Assert.True(writer.Queue.WaitDrained(Drain));
        Assert.Empty(new FileChunkStore(root).Keys());
        Assert.Empty(LedgerKeys(root));
        Assert.True(JsonNode.Parse(body)!["removed"]!["queued"]!.GetValue<int>() >= 1);
        Assert.Equal(1, slow.Encodes); // the queued item was dropped, never encoded and then discarded
    }

    // ------------------------------------------------------------------------------------------------ capabilities

    /// <summary>
    /// Capabilities report <c>store.entries</c>, <c>bytes</c>, <c>budget_bytes</c>, <c>dtype</c>: a non-admin sees its own
    /// namespace's ledger entries and the budget that applies to it (the per-namespace budget when set); an admin sees
    /// store-wide figures (every entry file, orphans included) and the store budget.
    /// </summary>
    [Fact]
    public async Task Capabilities_ReportsStoreFigures()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model,
            Writing(root, KeysFile(KeyA, KeyB, KeyAdmin + " admin"), budget: "1GiB", namespaceBudget: "64MiB"));
        await Store(server, [Key(KeyA)], (ChunkA, null));
        await Store(server, [Key(KeyB)], (ChunkA, null), (ChunkB, null));
        var orphan = PutDirect(server, root, ChunkC, "");
        long a = SizeOf(root, KeyOf(server, ChunkA, Ns(KeyA)));
        long b = SizeOf(root, KeyOf(server, ChunkA, Ns(KeyB))) + SizeOf(root, KeyOf(server, ChunkB, Ns(KeyB)));

        JsonNode Figures(string key) => JsonNode.Parse(server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(key)).GetAwaiter().GetResult().Body)!["store"]!;
        var sa = Figures(KeyA);
        Assert.Equal(1, Long(sa, "entries"));
        Assert.Equal(a, Long(sa, "bytes"));
        Assert.Equal(64L << 20, Long(sa, "budget_bytes"));
        Assert.Equal("f32", sa["dtype"]!.GetValue<string>());
        var sb = Figures(KeyB);
        Assert.Equal(2, Long(sb, "entries"));
        Assert.Equal(b, Long(sb, "bytes"));
        var admin = Figures(KeyAdmin);
        Assert.Equal(4, Long(admin, "entries"));
        Assert.Equal(a + b + SizeOf(root, orphan), Long(admin, "bytes"));
        Assert.Equal(1L << 30, Long(admin, "budget_bytes"));
    }

    /// <summary>
    /// The store-wide figures walk every entry file and the route has no slot: the walk is reused for
    /// <c>StoreFiguresTtl</c> instead of running per call (review I3). Stale by at most that long, never wrong after it.
    /// </summary>
    [Fact]
    public async Task Capabilities_StoreWideFigures_AreWalkedOncePerTtl()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var runtime = server.State.ChunkCacheRuntime;
        long Entries() => Long(JsonNode.Parse(server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null).GetAwaiter().GetResult().Body)!["store"]!, "entries");

        runtime.StoreFiguresTtl = TimeSpan.FromMinutes(10);
        int walks = runtime.StoreFigureWalks;
        Assert.Equal(0, Entries());
        for (int i = 0; i < 5; i++) Assert.Equal(0, Entries());
        Assert.Equal(walks + 1, runtime.StoreFigureWalks);            // six calls, one walk

        await Store(server, [], (ChunkA, null));
        Assert.Equal(0, Entries());                                   // inside the TTL: the old figures, and still no new walk
        Assert.Equal(walks + 1, runtime.StoreFigureWalks);

        runtime.StoreFiguresTtl = TimeSpan.Zero;                      // control: without the cache every call walks and sees the entry
        Assert.Equal(1, Entries());
        Assert.Equal(1, Entries());
        Assert.Equal(walks + 3, runtime.StoreFigureWalks);
    }

    // ------------------------------------------------------------------------------------------------ namespace budget

    /// <summary>
    /// <c>--chunk-store-namespace-budget</c>: budget eviction takes only the writing namespace's rows (one tenant's prewarm
    /// never evicts another's entries): over its own share a namespace evicts its own LRU rows; with the store full of
    /// others' rows and none of its own, its entry is dropped instead. Without the option the shared LRU applies (the
    /// control). The flag parses on both option paths and needs <c>--chunk-store-write</c>.
    /// </summary>
    /// <summary>
    /// A path forget covers every matching item queued before it, wherever the worker has it. The window it closes: the worker
    /// takes a matching item from the queue after the forget looked at the current item and before it emptied the queue; that
    /// item is then neither queued nor "current" as the forget saw it, and without a fence it is encoded and written back
    /// after the response reported the forget. Such an item is modelled directly: right sequence number, not in the queue.
    /// </summary>
    [Fact]
    public async Task PathForget_FencesEveryMatchingItemQueuedBeforeIt_NotOnlyTheOneBeingEncoded()
    {
        string Text(char ch) => new string(ch, 64) + "\n";
        long e = ChunkStoreServiceTests.FileBytes(Text('a'), 4);
        var x = new ChunkCaller(true, "tenant-x");
        var y = new ChunkCaller(true, "tenant-y");
        using var s = StartService(NewStore(), budget: 100 * e, nsBudget: 0);
        Put(s, x, Text('1'));
        Put(s, x, Text('2'));
        long last = s.Queue.LastSeq;
        Assert.True(last >= 2);
        ChunkEncodeItem Item(char ch, ChunkCaller who, string path, long seq) =>
            new(who.Namespace, ChunkStoreServiceTests.KeyFor(Text(ch), who.Namespace), Text(ch), 65, path) { Seq = seq };

        var raced = Item('r', x, "src/A.cs", last);
        Assert.True(s.ShouldEncode(raced)); // control: wanted until the forget
        await s.EvictAsync(new EvictRequest(EvictMode.Path, x.Namespace, "src/**", false), CancellationToken.None);

        Assert.False(s.ShouldEncode(raced));                                 // taken by the worker around the forget: still forgotten
        Assert.False(s.ShouldEncode(Item('q', x, "src/deep/B.cs", 1)));      // any earlier matching item
        Assert.True(s.ShouldEncode(Item('o', x, "docs/B.md", last)));        // another path
        Assert.True(s.ShouldEncode(Item('y', y, "src/A.cs", last)));         // another namespace
        Assert.True(s.ShouldEncode(Item('n', x, "src/A.cs", last + 1)));     // queued after the forget

        // A dry run fences nothing.
        await s.EvictAsync(new EvictRequest(EvictMode.Path, x.Namespace, "docs/**", true), CancellationToken.None);
        Assert.True(s.ShouldEncode(Item('o', x, "docs/B.md", last)));
    }

    /// <summary>
    /// The namespace and whole-store forgets fence by sequence number too: an item queued at or before the forget is never
    /// encoded or written afterwards, in the forgotten namespace (namespace mode) or in any (all without a namespace); an
    /// item queued later is.
    /// </summary>
    [Fact]
    public async Task NamespaceAndStoreWideForgets_FenceItemsQueuedBeforeThem()
    {
        string Text(char ch) => new string(ch, 64) + "\n";
        long e = ChunkStoreServiceTests.FileBytes(Text('a'), 4);
        var x = new ChunkCaller(true, "tenant-x");
        var y = new ChunkCaller(true, "tenant-y");
        using var s = StartService(NewStore(), budget: 100 * e, nsBudget: 0);
        Put(s, x, Text('1'));
        Put(s, y, Text('2'));
        long last = s.Queue.LastSeq;
        ChunkEncodeItem Item(char ch, ChunkCaller who, long seq) =>
            new(who.Namespace, ChunkStoreServiceTests.KeyFor(Text(ch), who.Namespace), Text(ch), 65, "src/A.cs") { Seq = seq };

        await s.EvictAsync(new EvictRequest(EvictMode.Namespace, x.Namespace, null, false), CancellationToken.None);
        Assert.False(s.ShouldEncode(Item('a', x, last)));
        Assert.True(s.ShouldEncode(Item('b', y, last)));       // control: the other namespace is untouched
        Assert.True(s.ShouldEncode(Item('c', x, last + 1)));   // queued after the forget

        await s.EvictAsync(new EvictRequest(EvictMode.All, null, null, false), CancellationToken.None);
        Assert.False(s.ShouldEncode(Item('b', y, last)));      // the whole store: every namespace
        Assert.False(s.ShouldEncode(Item('d', x, 1)));
        Assert.True(s.ShouldEncode(Item('e', y, last + 1)));
    }

    [Fact]
    public async Task NamespaceBudget_PrewarmEvictsOnlyOwnRows()
    {
        string Text(char ch) => new string(ch, 64) + "\n";
        long e = ChunkStoreServiceTests.FileBytes(Text('a'), 4);
        var x = new ChunkCaller(true, "tenant-x");
        var y = new ChunkCaller(true, "tenant-y");

        // (1) Over its own share: the namespace evicts its own least recently used row.
        {
            string root = NewStore();
            using var s = StartService(root, budget: 10 * e, nsBudget: 2 * e + 10);
            Put(s, x, Text('1'));
            Put(s, y, Text('2'));
            Put(s, x, Text('3'));
            Put(s, x, Text('4'));
            Assert.False(Stored(root, ChunkStoreServiceTests.KeyFor(Text('1'), x.Namespace)));
            Assert.True(Stored(root, ChunkStoreServiceTests.KeyFor(Text('3'), x.Namespace)));
            Assert.True(Stored(root, ChunkStoreServiceTests.KeyFor(Text('2'), y.Namespace)));
        }

        // (2) Store full of others' rows: with the namespace budget the newcomer's entry is dropped, nobody is evicted ...
        foreach (bool perNamespace in new[] { true, false })
        {
            string root = NewStore();
            var host = new ChunkStoreServiceTests.FakeHost();
            using var s = StartService(root, budget: 3 * e + 10, nsBudget: perNamespace ? 3 * e : 0, host);
            Put(s, x, Text('a'));
            Put(s, x, Text('b'));
            Put(s, x, Text('c'));
            Put(s, y, Text('d'));
            bool xOldestKept = Stored(root, ChunkStoreServiceTests.KeyFor(Text('a'), x.Namespace));
            bool yStored = Stored(root, ChunkStoreServiceTests.KeyFor(Text('d'), y.Namespace));
            // ... while the shared LRU (the control) evicts x's oldest row for y.
            Assert.Equal(perNamespace, xOldestKept);
            Assert.Equal(!perNamespace, yStored);
            if (!perNamespace)
                continue;

            // S1: once blocked, y's next misses are refused at admission (no encode spent and thrown away), with the detail.
            int encodes = host.Encodes;
            var refused = s.Admit(ChunkStoreServiceTests.Span(Text('e')), ChunkStoreServiceTests.KeyFor(Text('e'), y.Namespace), 65, y, true, "cpu");
            Assert.Equal(new AdmissionResult(false, AdmissionReasons.NamespaceBudgetBlocked), refused);
            var plan = s.ApplyTo(PlanOf((Text('f'), ChunkStoreServiceTests.KeyFor(Text('f'), y.Namespace))), ParsedEncode(), y);
            Assert.Equal(AdmissionReasons.NamespaceBudgetBlocked, plan.Warnings.Single(w => w.Code == ChunkCacheCodes.EncodeSkipped).Detail);
            Assert.True(s.Queue.WaitDrained(Drain));
            Assert.Equal(encodes, host.Encodes);
            // x is not blocked.
            Assert.True(s.Admit(ChunkStoreServiceTests.Span(Text('g')), ChunkStoreServiceTests.KeyFor(Text('g'), x.Namespace), 65, x, true, "cpu").Queued);
            Assert.True(s.Queue.WaitDrained(Drain));

            // Room again (x forgotten): y is admitted and stored.
            await s.EvictAsync(new EvictRequest(EvictMode.Namespace, x.Namespace, null, false), CancellationToken.None);
            Put(s, y, Text('h'));
            Assert.True(Stored(root, ChunkStoreServiceTests.KeyFor(Text('h'), y.Namespace)));
        }

        // (3) S5: the per-namespace over_budget admission check: a chunk that could never fit the namespace budget is refused
        // before any encode, although it fits the store budget.
        {
            string root = NewStore();
            var host = new ChunkStoreServiceTests.FakeHost();
            using var s = StartService(root, budget: 20 * e, nsBudget: e + 10, host);
            Put(s, x, Text('a')); // the writer learns this model's entry size
            int encodes = host.Encodes;
            string big = new string('z', 200);
            var r = s.Admit(ChunkStoreServiceTests.Span(big), ChunkStoreServiceTests.KeyFor(big, x.Namespace), big.Length, x, true, "cpu");
            Assert.Equal(new AdmissionResult(false, AdmissionReasons.OverBudget), r);
            Assert.True(s.Queue.WaitDrained(Drain));
            Assert.Equal(encodes, host.Encodes);
        }

        // (4) A cap, not a reservation: a namespace budget at or above the store budget refuses startup; a sum of caps over
        // the store budget warns (namespaces in the keys file, or 2 without one).
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, NewStore(), storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "1GiB"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, NewStore(), storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "2GiB"));
        Assert.NotNull(ChunkCacheOptions.Create(true, null, NewStore(), storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "600MiB").NamespaceBudgetWarning());
        Assert.Null(ChunkCacheOptions.Create(true, null, NewStore(), storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "500MiB").NamespaceBudgetWarning());
        string keys = KeysFile(KeyA, KeyB, KeyAdmin, "test-key-delta-0000000000000000000000 ns=team", "test-key-echo-11111111111111111111111 ns=team");
        var three = ChunkCacheOptions.Create(true, null, NewStore(), keys, storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "300MiB");
        Assert.NotNull(three.NamespaceBudgetWarning()); // 4 namespaces x 300 MiB > 1 GiB
        Assert.Null(ChunkCacheOptions.Create(true, null, NewStore(), keys, storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "256MiB").NamespaceBudgetWarning());
        Assert.Null(ChunkCacheOptions.Create(true, null, NewStore(), storeWrite: true, storeBudget: "1GiB").NamespaceBudgetWarning());

        // Options: both paths, needs --chunk-store-write and a size.
        var parsed = ServerOptions.Parse(["m.gguf", "--chunk-cache", "--chunk-store", NewStore(), "--chunk-store-write", "--chunk-store-budget", "1GiB",
            "--chunk-store-namespace-budget", "100MiB"]).ChunkCache;
        Assert.Null(parsed.NamespaceBudgetWarning());
        Assert.Equal(100L << 20, parsed.StoreNamespaceBudgetBytes);
        Assert.Equal(0, ServerOptions.Parse(["m.gguf", "--chunk-cache", "--chunk-store", NewStore(), "--chunk-store-write", "--chunk-store-budget", "1GiB"])
            .ChunkCache.StoreNamespaceBudgetBytes);
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, NewStore(), storeNamespaceBudget: "1MiB"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, NewStore(), storeWrite: true, storeBudget: "1GiB", storeNamespaceBudget: "0"));
    }

    private ChunkStoreService StartService(string root, long budget, long nsBudget) => StartService(root, budget, nsBudget, new ChunkStoreServiceTests.FakeHost());

    private static ChunkCachePlan PlanOf(params (string Text, ChunkKey Key)[] spans) =>
        new(ChunkPolicy.Exact, null, false, spans.Length,
            spans.Select((sp, i) => new PlannedSpan(i * 100, sp.Text.Length, false) { Span = ChunkStoreServiceTests.Span(sp.Text), Key = sp.Key }).ToList(),
            [], null, "summary", "cpu", 1000);

    private static ParsedChunkRequest ParsedEncode() =>
        new(ChunkPolicy.Exact, [], null, false, "summary", ChunkNamespace.Default, [], null) { OnMiss = "encode" };

    private ChunkStoreService StartService(string root, long budget, long nsBudget, ChunkStoreServiceTests.FakeHost host)
    {
        var options = new ChunkCacheOptions(true, ChunkCacheOptions.DefaultMaxChunks, root, null, StoreWrite: true, StoreBudgetBytes: budget)
        {
            StoreNamespaceBudgetBytes = nsBudget,
        };
        var service = ChunkStoreService.Start(options, host);
        service.Queue.QuietPeriod = TimeSpan.Zero;
        return service;
    }

    private static void Put(ChunkStoreService s, ChunkCaller caller, string text)
    {
        var key = ChunkStoreServiceTests.KeyFor(text, caller.Namespace);
        Assert.True(s.Admit(ChunkStoreServiceTests.Span(text), key, text.Length, caller, true, "cpu").Queued);
        Assert.True(s.Queue.WaitDrained(Drain));
        Thread.Sleep(20); // distinct last-use times
    }

    // ------------------------------------------------------------------------------------------------ fix round (review S1-S6)

    /// <summary>
    /// Locality is unchanged from before the store routes (owner's decision, 2026-11-20): a forwarding header
    /// (<c>Forwarded</c>, <c>X-Forwarded-*</c>, <c>X-Real-IP</c>) does NOT make a loopback caller non-local, so existing keyless
    /// setups behind a same-host proxy keep working. The exposure that leaves (a proxied caller is local, and local is admin
    /// for figures, dry runs and the default namespace) is documented; store-wide and cross-namespace deletes still need an
    /// admin key (<see cref="Keyless_StoreWideDestructive_NeedsAdminKey"/>). An opt-in rule is a separate issue.
    /// </summary>
    [Fact]
    public async Task Keyless_ForwardedHeaders_StayLocal_ButStoreWideDeletesStillNeedAnAdminKey()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        foreach (var header in new[] { ("Forwarded", "for=203.0.113.7"), ("X-Forwarded-For", "203.0.113.7"), ("X-Real-IP", "203.0.113.7") })
        {
            Assert.Equal(200, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, header)).Status);
            Assert.Equal(200, (await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace"}""", header)).Status);
            Assert.Equal(403, (await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"orphans","dry_run":false}""", header)).Status);
        }
        // Control: what does make a loopback caller non-local is unchanged (a browser's Origin header).
        Assert.Equal(403, (await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, ("Origin", "https://example.org"))).Status);
    }

    /// <summary>
    /// S2 ruling: a keyless server grants no store-wide destructive admin. Its local caller prewarms and evicts / forgets its
    /// own (default) namespace and may dry-run anything, but <c>orphans</c>, a store-wide <c>all</c> and another namespace
    /// with <c>dry_run: false</c> need an admin key (403 <c>chunk_cache_admin_key_required</c>).
    /// </summary>
    [Fact]
    public async Task Keyless_StoreWideDestructive_NeedsAdminKey()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        await Store(server, [], (ChunkA, null));
        var own = KeyOf(server, ChunkA, "");
        var orphan = PutDirect(server, root, ChunkC, "");
        var other = PutDirect(server, root, ChunkB, "k-0123456789abcdef");
        WriteNamespaceIndex(server, root, "k-0123456789abcdef", other);

        foreach (string body in new[] { """{"mode":"orphans","dry_run":false}""", """{"mode":"all","dry_run":false}""",
                     """{"mode":"namespace","namespace":"k-0123456789abcdef","dry_run":false}""",
                     """{"mode":"path","path":"**","namespace":"k-0123456789abcdef","dry_run":false}""" })
        {
            var (status, text) = await server.SendAsync(HttpMethod.Post, Evict, body);
            Assert.True(status == 403, $"{body}: {status} {text}");
            Assert.Equal("chunk_cache_admin_key_required", JsonNode.Parse(text)!["error"]!["code"]!.GetValue<string>());
        }
        Assert.True(Stored(root, orphan) && Stored(root, other) && Stored(root, own));

        var wideDry = await EvictOk(server, """{"mode":"all"}""");
        Assert.Equal(3, Int(wideDry, "evicted")); // a dry run may look store-wide
        Assert.Equal(1, Int(await EvictOk(server, """{"mode":"orphans"}"""), "evicted"));
        var mine = await EvictOk(server, """{"mode":"namespace","dry_run":false}""");
        Assert.Equal(1, Int(mine, "evicted"));
        Assert.False(Stored(root, own));
        Assert.True(Stored(root, orphan) && Stored(root, other));
    }

    /// <summary>A body that never finishes cannot hold a write slot (S3).</summary>
    private sealed class SlowContent(string start) : HttpContent
    {
        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            await stream.WriteAsync(Encoding.UTF8.GetBytes(start));
            await stream.FlushAsync();
            await Task.Delay(TimeSpan.FromSeconds(6));
        }

        protected override bool TryComputeLength(out long length)
        {
            length = 0;
            return false;
        }
    }

    private async Task<int> PostSlow(ChunkCacheTestServer server, string path, string key)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new SlowContent("{\"mode\":") };
        request.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
        request.Headers.TryAddWithoutValidation("X-API-Key", key);
        try
        {
            using var resp = await server.Http.SendAsync(request);
            return (int)resp.StatusCode;
        }
        catch (HttpRequestException)
        {
            return -1; // the server closed the connection on the unfinished body
        }
        catch (TaskCanceledException)
        {
            return -2;
        }
    }

    /// <summary>
    /// S3: the body is read, with a deadline and the size limit, before a write slot is taken: two trickling bodies from
    /// one tenant hold no slot (another tenant's evict answers at once) and end with 408 at the deadline; explain likewise.
    /// One tenant has at most one write call at once (429), and the server at most two (429).
    /// </summary>
    [Fact]
    public async Task WriteRoutes_BodyReadBeforeTheGate_PerNamespaceAndGlobalSlots()
    {
        string root = NewStore();
        const string KeyC = "test-key-charlie-5555555555aaaaaaaa";
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA, KeyB, KeyC)));
        var saved = ChunkCacheEndpoint.BodyReadTimeout;
        ChunkCacheEndpoint.BodyReadTimeout = TimeSpan.FromSeconds(2);
        try
        {
            var slow = new[] { PostSlow(server, Evict, KeyA), PostSlow(server, Prewarm, KeyA), PostSlow(server, "/v1/chunk-cache/explain", KeyA) };
            await Task.Delay(300);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var fast = await EvictOk(server, """{"mode":"namespace"}""", Key(KeyB));
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1.5), sw.Elapsed.ToString());
            Assert.True(fast["dry_run"]!.GetValue<bool>());
            Assert.Equal(0, Int(await EvictOk(server, """{"mode":"namespace"}""", Key(KeyB)), "evicted")); // a second fast call too
            foreach (int status in await Task.WhenAll(slow))
                Assert.True(status is 408 or -1, status.ToString());
        }
        finally
        {
            ChunkCacheEndpoint.BodyReadTimeout = saved;
        }

        // Slots: hold the store lock so two evicts wait inside their slot.
        await Store(server, [Key(KeyA)], (ChunkA, null));
        await Store(server, [Key(KeyB)], (ChunkA, null));
        Writer(server).LockTimeout = TimeSpan.FromSeconds(8);
        Task<(int Status, string Body)> a, b;
        using (ChunkIndex.AcquireStoreLock(root, TimeSpan.FromSeconds(5)))
        {
            a = server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""", Key(KeyA));
            await Task.Delay(500);
            var sameTenant = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace"}""", Key(KeyA));
            Assert.Equal(429, sameTenant.Status);
            Assert.Equal("chunk_control_busy", JsonNode.Parse(sameTenant.Body)!["error"]!["code"]!.GetValue<string>());
            b = server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""", Key(KeyB));
            await Task.Delay(500);
            var third = await server.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((ChunkC, null)), Key(KeyC));
            Assert.Equal(429, third.Status);
        }
        Assert.Equal(200, (await a).Status);
        Assert.Equal(200, (await b).Status);
        Assert.Empty(LedgerKeys(root));
        Assert.Equal(200, (await server.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((ChunkC, null)), Key(KeyC))).Status);
    }

    /// <summary>
    /// S4: a forget that could not delete everything says so (warning <c>chunk_evict_partial</c> with counts, counts only);
    /// a failed ledger save is a 503 whose message says sightings and queued encodes may already be dropped while no file
    /// was deleted.
    /// </summary>
    [Fact]
    public async Task Evict_PartialDelete_WarnsWithCounts_FailedSave_SaysWhatWasDropped()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        await Store(server, [], (ChunkA, null), (ChunkB, null));
        var a = KeyOf(server, ChunkA, "");
        var b = KeyOf(server, ChunkB, "");
        var cli = PutDirect(server, root, ChunkC, "");
        string index = WriteNamespaceIndex(server, root, "", cli);

        JsonNode r;
        using (new FileStream(FileChunkStore.EntryPath(root, a.Hex), FileMode.Open, FileAccess.Read, FileShare.Read))
        using (new FileStream(index, FileMode.Open, FileAccess.Read, FileShare.Read))
            r = await EvictOk(server, """{"mode":"namespace","dry_run":false}""");
        if (OperatingSystem.IsWindows())
        {
            // Windows: a file held open without FileShare.Delete cannot be deleted, so the evict reports a partial delete.
            Assert.Contains("chunk_evict_partial", Strings(r["warnings"]));
            Assert.Equal(1, r["failed"]!["entry_files"]!.GetValue<int>());
            Assert.Equal(1, r["failed"]!["index_files"]!.GetValue<int>());
            Assert.True(Stored(root, a));   // held open: not deleted, reported
            Assert.False(Stored(root, b));
            Assert.True(Stored(root, cli)); // its index could not be deleted, so it stays referenced
        }
        else
        {
            // POSIX: unlinking an open file succeeds, so there is nothing to report (the partial path is Windows-reachable only here).
            Assert.DoesNotContain("chunk_evict_partial", Strings(r["warnings"]));
            Assert.Null(r["failed"]);
            Assert.False(Stored(root, a));
            Assert.False(Stored(root, b));
            Assert.False(Stored(root, cli));
        }
        Assert.Empty(LedgerKeys(root));

        var clean = await EvictOk(server, """{"mode":"namespace","dry_run":false}""");
        Assert.Null(clean["failed"]);
        Assert.Empty(Strings(clean["warnings"]));

        await Store(server, [], (ChunkD, null));
        var d = KeyOf(server, ChunkD, "");
        Writer(server).LedgerForTest!.BeforeRename = _ => throw new IOException("disk full (test)");
        try
        {
            var (status, body) = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""");
            Assert.Equal(503, status);
            string message = JsonNode.Parse(body)!["error"]!["message"]!.GetValue<string>();
            Assert.Contains("no entry or index file was deleted", message, StringComparison.Ordinal);
            Assert.Contains("sightings and queued encodes", message, StringComparison.Ordinal);
            Assert.True(Stored(root, d));
        }
        finally
        {
            if (Writer(server).LedgerForTest is { } l) l.BeforeRename = null;
        }
    }

    /// <summary>
    /// S5: an evict that starts while the background writer holds the ledger gate (mid-write, before the store lock) waits
    /// for it and then sees the new row (and deletes it); held past the lock timeout, the evict answers 503 cleanly and the
    /// writer completes afterwards (no deadlock).
    /// </summary>
    [Fact]
    public async Task Evict_WhileWriterHoldsLedgerGate_WaitsThenSeesRow_OrTimesOutCleanly()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var state = server.State;
        state.ChunkCacheRuntime.Dispose();
        var host = new ChunkStoreServiceTests.FakeHost();
        var writer = state.ChunkCacheRuntime.StartWriter(state, host)!;
        writer.Queue.QuietPeriod = TimeSpan.Zero;
        using var entered = new SemaphoreSlim(0);
        using var release = new SemaphoreSlim(0);
        writer.WriteEnteredForTest = () => { entered.Release(); release.Wait(TimeSpan.FromSeconds(30)); };

        Assert.Equal(1, Int(await PrewarmOk(server, PrewarmBody((ChunkA, null))), "queued"));
        Assert.True(await entered.WaitAsync(TimeSpan.FromSeconds(10)));
        var evict = server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""");
        await Task.Delay(500);
        Assert.False(evict.IsCompleted);
        release.Release();
        var (status, body) = await evict;
        Assert.True(status == 200, body);
        Assert.Equal(1, Int(JsonNode.Parse(body)!, "evicted"));
        Assert.True(writer.Queue.WaitDrained(Drain));
        Assert.Empty(new FileChunkStore(root).Keys());

        writer.LockTimeout = TimeSpan.FromSeconds(1);
        Assert.Equal(1, Int(await PrewarmOk(server, PrewarmBody((ChunkB, null))), "queued"));
        Assert.True(await entered.WaitAsync(TimeSpan.FromSeconds(10)));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (timedOut, _) = await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","dry_run":false}""");
        Assert.Equal(503, timedOut);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(10));
        writer.WriteEnteredForTest = null;
        release.Release();
        Assert.True(writer.Queue.WaitDrained(Drain));
        Assert.Single(new FileChunkStore(root).Keys()); // the writer finished its entry
    }

    /// <summary>
    /// S5 / S6: the write-time fence alone keeps an in-flight encode of a forgotten namespace out of the store (nothing else
    /// is queued); a path forget drops only the matching items (a non-matching queued item is still encoded and stored).
    /// </summary>
    [Fact]
    public async Task Forget_WriteTimeFence_InFlightOnly_PathDropsMatchingItemsOnly()
    {
        string root = NewStore();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var state = server.State;
        state.ChunkCacheRuntime.Dispose();
        var slow = new ChunkStoreServiceTests.FakeHost { Block = new ManualResetEventSlim(false) };
        var writer = state.ChunkCacheRuntime.StartWriter(state, slow)!;
        writer.Queue.QuietPeriod = TimeSpan.Zero;

        Assert.Equal(1, Int(await PrewarmOk(server, PrewarmBody((ChunkA, "src/A.cs"))), "queued"));
        Assert.True(slow.Entered.Wait(TimeSpan.FromSeconds(10)));
        var r = await EvictOk(server, """{"mode":"namespace","dry_run":false}""");
        Assert.Equal(0, r["removed"]!["queued"]!.GetValue<int>()); // nothing queued: only the fence can stop it
        slow.Block.Set();
        Assert.True(writer.Queue.WaitDrained(Drain));
        Assert.Empty(new FileChunkStore(root).Keys());

        slow.Block.Reset();
        slow.Entered.Reset();
        Assert.Equal(3, Int(await PrewarmOk(server, PrewarmBody((ChunkB, "src/B.cs"), (ChunkC, "lib/C.cs"), (ChunkD, "src/D.cs"))), "queued"));
        Assert.True(slow.Entered.Wait(TimeSpan.FromSeconds(10))); // B in flight; C and D queued
        int before = slow.Encodes;
        var p = await EvictOk(server, """{"mode":"path","path":"src/**","dry_run":false}""");
        Assert.Equal(1, p["removed"]!["queued"]!.GetValue<int>()); // D
        slow.Block.Set();
        Assert.True(writer.Queue.WaitDrained(Drain));
        Assert.Equal(before + 1, slow.Encodes); // C only: D was never encoded
        Assert.Single(new FileChunkStore(root).Keys());
        Assert.True(Stored(root, KeyOf(server, ChunkC, "")));
    }

    // ------------------------------------------------------------------------------------------------ privacy

    /// <summary>No namespace, key, key digest, chunk text, text hash or source path in any write-route body or log.</summary>
    [Fact]
    public async Task WriteRoutes_NoNamespaceKeyTextOrPathInBodiesOrLogs()
    {
        string root = NewStore();
        var console = new StringWriter();
        TextWriter oldOut = Console.Out, oldErr = Console.Error;
        var bodies = new List<string>();
        Console.SetOut(console);
        Console.SetError(console);
        try
        {
            await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, KeysFile(KeyA, KeyAdmin + " admin")));
            bodies.Add((await server.SendAsync(HttpMethod.Post, Prewarm, PrewarmBody((ChunkA, "src/Secretive.cs"), (ChunkB, ".env")), Key(KeyA))).Body);
            Assert.True(Writer(server).Queue.WaitDrained(Drain));
            bodies.Add((await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"path","path":"src/Secretive.cs"}""", Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, Evict, $$"""{"mode":"namespace","namespace":"{{Ns(KeyA)}}"}""", Key(KeyAdmin))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, Evict, $$"""{"mode":"namespace","namespace":"{{Ns(KeyA)}}","dry_run":false}""", Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Post, Evict, """{"mode":"namespace","namespace":"Bad Name"}""", Key(KeyAdmin))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyA))).Body);
            bodies.Add((await server.SendAsync(HttpMethod.Get, "/v1/chunk-cache", null, Key(KeyAdmin))).Body);
        }
        finally
        {
            Console.SetOut(oldOut);
            Console.SetError(oldErr);
        }
        string all = string.Join("\n", bodies) + "\n" + console;
        Assert.Contains("evicted", bodies[1], StringComparison.Ordinal);
        foreach (string key in new[] { KeyA, KeyAdmin })
        {
            string ns = ChunkNamespace.ForApiKey(key);
            Assert.DoesNotContain(key, all, StringComparison.Ordinal);
            Assert.DoesNotContain(ns[2..], all, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(key))), all, StringComparison.OrdinalIgnoreCase);
        }
        Assert.DoesNotContain("Secretive", all, StringComparison.Ordinal);
        Assert.DoesNotContain("the sum of both operands", all, StringComparison.Ordinal);
        Assert.DoesNotContain(WireChunking.Hash(ChunkA)[7..], all, StringComparison.Ordinal);
    }
}
