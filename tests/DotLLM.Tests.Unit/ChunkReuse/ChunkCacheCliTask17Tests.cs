using System.Diagnostics;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.ChunkCache;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase I Task 17: the <c>chunkcache</c> CLI revised for the server-backed design (<c>--json</c> against the §7.2 schemas, the
/// secrets deny-list, wire eligibility, <c>--server</c> / <c>--offline</c>, the sole-writer owner lock, the API key from the
/// environment only). Server calls go to a fake <see cref="HttpMessageHandler"/> that serves the contract's own
/// <c>capabilities.json</c> (patched with the identity fields of Task 14) and records every request as the real client sent it
/// (real wire bytes); <see cref="ChunkCacheCliServerTests"/> runs the same verbs against the real server code.
/// </summary>
public sealed partial class ChunkCacheCliTests
{
    private const string FakeUrl = "http://fake.test:8080";
    private const string ApiKeyVar = "CHUNKCACHE_API_KEY";

    // ------------------------------------------------------------------------------------------------ Task 17 tests

    [Fact]
    public void EachVerb_Json_MatchesSchema()
    {
        string tree = NewTree();
        string store = NewDir("store");
        using ModelHandle model = OpenModel(_fx.ModelA);
        var fake = new FakeServer(s => (s.Method, s.Path) switch
        {
            ("GET", "/v1/chunk-cache") => (HttpStatusCode.OK, CapsJson(model.ModelId, model.Producer)),
            ("POST", "/v1/chunk-cache/prewarm") => (HttpStatusCode.OK, """{"requested":1,"already_stored":0,"queued":1,"queued_tokens":10,"refused":0,"refused_reasons":[]}"""),
            ("POST", "/v1/chunk-cache/evict") => (HttpStatusCode.OK, """{"evicted":2,"freed_bytes":1024,"kept_referenced":3,"dry_run":true}"""),
            ("POST", "/v1/chunk-cache/explain") => (HttpStatusCode.OK, """{"policy_would_apply":"exact","reason":null,"chunks":[{"id":"a","status":"hit","why":null}],"predicted_ms":{"reuse":1.5,"full_prefill":3.0},"warnings":[]}"""),
            _ => (HttpStatusCode.NotFound, "{}"),
        });
        string texts = Path.Combine(NewDir("in"), "texts.json");
        File.WriteAllText(texts, """[{"id": "a", "text": "class A { }"}]""");
        string request = Path.Combine(Path.GetDirectoryName(texts)!, "request.json");
        File.WriteAllText(request, """{"model": "m", "messages": [{"role": "user", "content": "hi"}]}""");

        var outputs = new (string Schema, (int Code, string Out, string Err) Run)[]
        {
            ("build", CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--offline", "--json"])),
            ("status", CliRaw(["status", "--model", _fx.ModelA, "--store", store, "--no-scan", "--json"])),
            ("select", CliRaw(["select", "--model", _fx.ModelA, "--store", store, "--path", "src/Calc.cs", "--json"])),
            ("evict", CliRaw(["evict", "--server", FakeUrl, "--json"], http: fake)),
            ("prewarm", CliRaw(["prewarm", "--server", FakeUrl, "--texts", texts, "--json"], http: fake)),
            ("explain", CliRaw(["explain", "--server", FakeUrl, "--request", request, "--json"], http: fake)),
            ("error", CliRaw(["frobnicate", "--json"])),
        };
        foreach ((string schema, var run) in outputs)
        {
            if (schema != "error") Assert.True(run.Code == 0, $"{schema}: exit {run.Code}: {run.Err}");
            Assert.Empty(SchemaErrors(schema, run.Out));
        }

        // The check can fail: a renamed required field, a wrong type and an unknown enum value are all reported.
        JsonObject build = JsonNode.Parse(outputs[0].Run.Out)!.AsObject();
        string renamed = build.ToJsonString().Replace("\"denied_files\"", "\"denied\"", StringComparison.Ordinal);
        Assert.Contains(SchemaErrors("build", renamed), e => e.Contains("denied_files", StringComparison.Ordinal));
        build["files"] = "three";
        Assert.NotEmpty(SchemaErrors("build", build.ToJsonString()));
        Assert.NotEmpty(SchemaErrors("explain", """{"chunks":[{"id":"a","status":"stored"}],"warnings":[]}"""));
    }

    [Fact]
    public void Build_DeniedFiles_CountedNotChunked()
    {
        string tree = NewTree();
        const string secret = "class Secret { string Token = \"hunter2-hunter2-hunter2\"; }\n";
        File.WriteAllText(Path.Combine(tree, "src", "id_token.cs"), secret); // id_*
        File.WriteAllText(Path.Combine(tree, "src", "id_rsa.cs"), secret.Replace("Secret", "Rsa", StringComparison.Ordinal)); // id_* (the directory corpus reads .cs only)
        File.WriteAllText(Path.Combine(tree, "src", "Keep.cs"), "class Keep { int One() => 1; }\n"); // not on the list
        string store = NewDir("store");

        JsonElement r = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--offline", "--json"]));
        Assert.Equal(2, r.GetProperty("denied_files").GetInt32());
        Assert.Equal(4, r.GetProperty("files").GetInt32()); // Calc, Shapes, Text, Keep

        using ModelHandle model = OpenModel(_fx.ModelA);
        ChunkIndex index = ChunkIndex.Load(Directory.GetFiles(store, "index-*.json").Single(), model.KeyOf);
        Assert.DoesNotContain(index.Paths, p => p.Contains("id_", StringComparison.Ordinal));
        Assert.Contains("src/Keep.cs", index.Paths); // control: a file that is not on the list is chunked
        // Nothing of the secret reached the store: no entry for its tokens, and no file under the store mentions it.
        var stored = new FileChunkStore(store).Keys().Select(k => k.Hex).ToHashSet(StringComparer.Ordinal);
        Assert.DoesNotContain(model.KeyOf(ChunkTokenization.Encode(model.Tokenizer, secret)).Hex, stored);
        Assert.DoesNotContain(Directory.EnumerateFiles(store, "*", SearchOption.AllDirectories).Where(f => !f.EndsWith(".ckv", StringComparison.Ordinal)),
            f => File.ReadAllText(f).Contains("hunter2", StringComparison.Ordinal));
        Assert.Equal(index.Paths.Sum(p => index.ChunksOf(p).Count), r.GetProperty("chunks").GetInt32());
    }

    [Fact]
    public void Build_ChunkerNotInEligible_RefusedWithoutLab()
    {
        string tree = NewTree();
        string store = NewDir("store");
        foreach (string[] chunker in new[] { new[] { "--chunker", "fixed" }, new[] { "--chunk-tokens", "64" } }) // fixed-512, symbol-64
        {
            var refused = CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--offline", .. chunker, "--json"]);
            Assert.Equal(2, refused.Code);
            Assert.Contains("--lab", refused.Err, StringComparison.Ordinal);
            Assert.Equal("usage", JsonDocument.Parse(refused.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
            Assert.Empty(Directory.EnumerateFileSystemEntries(store)); // refused before anything is written
        }
        // With --lab the build runs, and every chunk of a chunker that is off the wire counts as wire-ineligible.
        JsonElement lab = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--offline", "--chunker", "fixed", "--lab", "--json"]));
        Assert.Equal("fixed-512", lab.GetProperty("chunker").GetString());
        Assert.True(lab.GetProperty("chunks").GetInt32() > 0);
        Assert.Equal(lab.GetProperty("chunks").GetInt32(), lab.GetProperty("wire_ineligible_chunks").GetInt32());
        // The wire chunker needs no flag.
        JsonElement wire = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--index-only", "--json"]));
        Assert.Equal("symbol-512", wire.GetProperty("chunker").GetString());
        Assert.Equal(0, wire.GetProperty("wire_ineligible_chunks").GetInt32());
    }

    [Fact]
    public void Build_Encode_CallsPrewarm_NeverPut()
    {
        string tree = NewTree();
        string store = NewDir("store");
        using ModelHandle model = OpenModel(_fx.ModelA);
        var fake = PrewarmingServer(model, "served-model", skewed: true);

        JsonElement r = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--server", FakeUrl, "--encode", "--json"], http: fake));
        Assert.Equal("prewarm", r.GetProperty("mode").GetString());

        // Only a capabilities read and prewarm posts: no PUT, no DELETE, no other path (the server alone writes entries).
        Assert.All(fake.Requests, q => Assert.True(q is { Method: "GET", Path: "/v1/chunk-cache" } or { Method: "POST", Path: "/v1/chunk-cache/prewarm" }, $"{q.Method} {q.Path}"));
        var posts = fake.Requests.Where(q => q.Method == "POST").ToList();
        Assert.NotEmpty(posts);

        // The wire bytes: {"texts":[{id, text, source:{path}}]} and nothing else (no hash, no token ids, no K/V).
        var sent = new List<(string Id, string Text, string Path)>();
        foreach (Seen p in posts)
        {
            JsonObject body = JsonNode.Parse(p.Body)!.AsObject();
            Assert.Equal(["texts"], body.Select(kv => kv.Key));
            foreach (JsonNode? t in body["texts"]!.AsArray())
            {
                Assert.Equal(["id", "source", "text"], t!.AsObject().Select(kv => kv.Key).Order(StringComparer.Ordinal));
                sent.Add((t["id"]!.GetValue<string>(), t["text"]!.GetValue<string>(), t["source"]!["path"]!.GetValue<string>()));
            }
        }
        // Oracle: the texts are exactly the indexed chunks' text slices of the files, one per distinct key.
        ChunkIndex index = ChunkIndex.Load(Directory.GetFiles(store, "index-*.json").Single(), model.KeyOf);
        var expected = index.Paths.SelectMany(p => index.ChunksOf(p).Select(c => File.ReadAllText(Path.Combine(tree, p))[c.StartOffset..c.EndOffset]))
            .Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(expected, sent.Select(s => s.Text).Order(StringComparer.Ordinal));
        Assert.Equal(r.GetProperty("distinct_keys").GetInt32(), sent.Count);

        // The CLI wrote no entry and no ledger; the server's own answer is what it reports as queued.
        Assert.Empty(Directory.EnumerateFiles(store, "*.ckv", SearchOption.AllDirectories));
        Assert.False(File.Exists(ServerLedger.DefaultPath(store)));
        // encode_queued_tokens is what the SERVER queued (it already held one text: 7 tokens fewer than sent), not what we planned.
        long sentTokens = sent.Sum(s => ChunkTokenization.Encode(model.Tokenizer, s.Text).Length);
        Assert.Equal(sentTokens - 7, r.GetProperty("encode_queued_tokens").GetInt64());
        Assert.Equal(sent.Count - 1, r.GetProperty("prewarm").GetProperty("queued").GetInt32());
        Assert.Equal(1, r.GetProperty("prewarm").GetProperty("already_stored").GetInt32());

        // --offline is the other writer, not a second way to call the server.
        Assert.Equal(2, CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--server", FakeUrl, "--offline"], http: fake).Code);
        // Without a stated writer a build refuses (exit 2) instead of silently encoding here.
        Assert.Equal(2, CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree]).Code);
    }

    [Fact]
    public void Evict_WithoutServerOrOffline_Exit2()
    {
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", NewTree(), "--chunk-tokens", "64"]).Code);
        string[] before = Directory.EnumerateFileSystemEntries(store, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToArray();
        foreach (string[] extra in new[] { Array.Empty<string>(), ["--apply"], ["--mode", "all"] })
        {
            var r = CliRaw(["evict", "--store", store, .. extra, "--json"]);
            Assert.Equal(2, r.Code);
            Assert.Equal("usage", JsonDocument.Parse(r.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
        }
        Assert.Equal(before, Directory.EnumerateFileSystemEntries(store, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(2, CliRaw(["evict", "--server", FakeUrl, "--offline", "--store", store]).Code); // both is no choice either
    }

    [Fact]
    public void Evict_Server_SendsTheResolvedModeAndDryRun()
    {
        using ModelHandle model = OpenModel(_fx.ModelA);
        var fake = PrewarmingServer(model, model.ModelId);
        Assert.Equal(0, CliRaw(["evict", "--server", FakeUrl], http: fake).Code);
        Assert.Equal(0, CliRaw(["evict", "--server", FakeUrl, "--apply", "--mode", "path", "--path", "src/**"], http: fake).Code);
        Assert.Equal(0, CliRaw(["evict", "--server", FakeUrl, "--mode", "namespace", "--namespace", "team-a"], http: fake).Code);
        JsonObject[] bodies = fake.Requests.Where(q => q is { Method: "POST", Path: "/v1/chunk-cache/evict" }).Select(q => JsonNode.Parse(q.Body)!.AsObject()).ToArray();
        Assert.Equal(3, bodies.Length);
        // The default is a dry run of the orphans (nothing is deleted unless --apply says so)...
        Assert.Equal("orphans", bodies[0]["mode"]!.GetValue<string>());
        Assert.True(bodies[0]["dry_run"]!.GetValue<bool>());
        // ...and --apply with a glob goes out as dry_run false.
        Assert.Equal(("path", "src/**", false), (bodies[1]["mode"]!.GetValue<string>(), bodies[1]["path"]!.GetValue<string>(), bodies[1]["dry_run"]!.GetValue<bool>()));
        Assert.Equal(("namespace", "team-a", true), (bodies[2]["mode"]!.GetValue<string>(), bodies[2]["namespace"]!.GetValue<string>(), bodies[2]["dry_run"]!.GetValue<bool>()));
    }

    [Fact]
    public void EvictAllForModelApply_ServerOwnsStore_Refused()
    {
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", NewTree(), "--chunk-tokens", "64"]).Code);
        int entries = new FileChunkStore(store).GetStats().Entries;
        string[] indexes = Directory.GetFiles(store, "index-*.json");
        Assert.NotEmpty(indexes);
        string[] cmd = ["evict", "--offline", "--store", store, "--mode", "all", "--model", _fx.ModelA, "--apply", "--json"];

        using (StoreOwnerLock? server = StoreOwnerLock.TryAcquire(store)) // a running server owns the store
        {
            Assert.NotNull(server);
            var refused = CliRaw(cmd);
            Assert.Equal(1, refused.Code);
            Assert.Equal("store_owned", JsonDocument.Parse(refused.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
            Assert.Equal(entries, new FileChunkStore(store).GetStats().Entries);
            Assert.Equal(indexes.Order(StringComparer.Ordinal), Directory.GetFiles(store, "index-*.json").Order(StringComparer.Ordinal));
        }
        // Control: once nobody owns the store the same command runs.
        Assert.Equal(0, CliRaw(cmd).Code);
        Assert.Equal(0, new FileChunkStore(store).GetStats().Entries);
        Assert.Empty(Directory.GetFiles(store, "index-*.json"));

        // Through the server there is no model-scoped eviction: --model belongs to --offline.
        Assert.Equal(2, CliRaw(["evict", "--server", FakeUrl, "--mode", "all", "--model", _fx.ModelA, "--apply"]).Code);
    }

    [Fact]
    public void Offline_EvictApply_NeedsOwnershipLock()
    {
        string tree = NewTree();
        string store = NewDir("store");
        (HashSet<string> orphans, _) = MakeOrphan(tree, store);
        var fs = new FileChunkStore(store);

        using (StoreOwnerLock? other = StoreOwnerLock.TryAcquire(store))
        {
            Assert.NotNull(other);
            var refused = CliRaw(["evict", "--offline", "--store", store, "--apply", "--json"]);
            Assert.Equal(1, refused.Code);
            Assert.Equal("store_owned", JsonDocument.Parse(refused.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
            Assert.All(orphans, o => Assert.True(fs.Contains(new ChunkKey(o))));
            // A dry run deletes nothing and needs no ownership.
            Assert.Equal(orphans.Count, Json(CliRaw(["evict", "--offline", "--store", store, "--json"])).GetProperty("evicted").GetInt32());
        }

        // Free: the applying pass takes the lock and holds it while it deletes (a probe from outside sees it held), then lets go.
        bool? heldDuringPass = null;
        EvictCommand.TestHookInsideApply = () => heldDuringPass = StoreOwnerLock.IsHeldByAnother(store);
        try
        {
            Assert.Equal(0, CliRaw(["evict", "--offline", "--store", store, "--apply"]).Code);
        }
        finally
        {
            EvictCommand.TestHookInsideApply = null;
        }
        Assert.True(heldDuringPass);
        Assert.False(StoreOwnerLock.IsHeldByAnother(store));
        Assert.All(orphans, o => Assert.False(fs.Contains(new ChunkKey(o))));
    }

    [Fact]
    public void Cli_NeverWritesLedger()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        using ModelHandle model = OpenModel(_fx.ModelA);
        string serverKey = new FileChunkStore(store).Keys().First().Hex;
        ServerLedger ledger = ServerLedger.Load(store);
        ledger.Record(new ChunkKey(serverKey), "k-0123456789abcdef", "src/Calc.cs", 1, DateTimeOffset.UtcNow);
        using (StoreLockHandle held = ChunkIndex.AcquireStoreLock(store, TimeSpan.FromSeconds(30))) ledger.Save(held);
        string path = ServerLedger.DefaultPath(store);
        byte[] bytes = File.ReadAllBytes(path);
        DateTime stamp = File.GetLastWriteTimeUtc(path);

        var fake = PrewarmingServer(model, model.ModelId);
        string texts = Path.Combine(NewDir("in"), "t.json");
        File.WriteAllText(texts, """[{"id": "a", "text": "class A { }"}]""");
        string request = Path.Combine(Path.GetDirectoryName(texts)!, "r.json");
        File.WriteAllText(request, """{"model": "m", "messages": []}""");
        string[] common = ["--model", _fx.ModelA, "--store", store];
        var verbs = new (string[] Args, bool Http)[]
        {
            (["build", .. common, "--directory", tree, "--chunk-tokens", "64", "--lab", "--offline"], false),
            (["build", .. common, "--directory", tree, "--chunk-tokens", "64", "--lab", "--dry-run"], false),
            (["build", .. common, "--directory", tree, "--index-only"], false),
            (["build", .. common, "--directory", tree, "--server", FakeUrl, "--encode"], true),
            (["status", .. common, "--chunk-tokens", "64", "--no-scan"], false),
            (["select", .. common, "--chunk-tokens", "64", "--path", "src/Calc.cs"], false),
            (["evict", "--offline", "--store", store], false),
            (["evict", "--offline", "--store", store, "--apply"], false),
            (["evict", "--offline", "--store", store, "--mode", "all", "--model", _fx.ModelA, "--apply"], false),
            (["evict", "--server", FakeUrl, "--apply"], true),
            (["prewarm", "--server", FakeUrl, "--texts", texts], true),
            (["explain", "--server", FakeUrl, "--request", request], true),
        };
        foreach (var (args, http) in verbs)
        {
            var run = CliRaw(args, http: http ? fake : null);
            Assert.True(run.Code == 0, $"{string.Join(' ', args)} => {run.Code}: {run.Err}");
            Assert.Equal(bytes, File.ReadAllBytes(path));
            Assert.Equal(stamp, File.GetLastWriteTimeUtc(path));
        }
    }

    [Fact]
    public void StatusStrict_UnreadableLedger_Fails()
    {
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", NewTree(), "--chunk-tokens", "64"]).Code);
        string[] status = ["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan"];
        Assert.Equal(0, CliRaw([.. status, "--strict"]).Code); // control: a healthy (absent) ledger passes
        File.WriteAllText(ServerLedger.DefaultPath(store), """{"formatVersion": 99, "rows": []}""");
        Assert.Equal(0, CliRaw(status).Code); // without --strict status still answers
        var strict = CliRaw([.. status, "--strict"]);
        Assert.Equal(3, strict.Code);
        Assert.Contains("ledger", strict.Err, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Offline_SecondWriter_ExitsNonZeroWithinTimeout()
    {
        string tree = NewTree();
        string store = NewDir("store");
        string[] build = ["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--offline", "--json"];
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        Task<(int Code, string Out, string Err)> first = Task.Run(() =>
        {
            BuildCommand.TestHookAfterOwnerLock = () =>
            {
                reached.Set();
                Assert.True(release.Wait(TimeSpan.FromSeconds(120)));
            };
            return CliRaw(build);
        });
        try
        {
            Assert.True(reached.Wait(TimeSpan.FromSeconds(60)), "the first writer never took the owner lock");
            var clock = Stopwatch.StartNew();
            var second = CliRaw(build);
            clock.Stop();
            Assert.Equal(1, second.Code);
            Assert.Equal("store_owned", JsonDocument.Parse(second.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), $"the second writer took {clock.Elapsed}");
            Assert.Empty(Directory.GetFiles(store, "index-*.json")); // it wrote nothing
        }
        finally
        {
            release.Set();
        }
        Assert.Equal(0, (await first).Code);
        Assert.NotEmpty(Directory.GetFiles(store, "index-*.json"));
        Assert.False(StoreOwnerLock.IsHeldByAnother(store)); // released at the end of the run
    }

    [Fact]
    public void Select_HashEqualsWireHash()
    {
        string tree = NewTree();
        File.WriteAllText(Path.Combine(tree, "src", "Crlf.cs"), "namespace Demo;\r\n\r\npublic static class Crlf\r\n{\r\n    public static int One() => 1;\r\n\r\n    public static int Two() => 2;\r\n}\r\n");
        JsonElement r = Json(CliRaw(["select", "--model", _fx.ModelA, "--directory", tree, "--path", "src/Crlf.cs", "--include-text", "--json"]));
        JsonElement[] chunks = r.GetProperty("chunks").EnumerateArray().ToArray();
        Assert.NotEmpty(chunks);
        foreach (JsonElement c in chunks)
        {
            string text = c.GetProperty("text").GetString()!;
            Assert.DoesNotContain('\r', text); // the wire carries the LF text the index holds (§5.3 item 1)
            string expected = "sha256:" + Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
            Assert.Equal(expected, c.GetProperty("hash").GetString());
        }
        Assert.Contains(chunks, c => c.GetProperty("text").GetString()!.Contains("One()", StringComparison.Ordinal));
    }

    [Fact]
    public void Select_Server_ExplainBody_PinsEachChunkByUtf8ByteRange_AndStoredIsTheServersAnswer()
    {
        string tree = NewTree();
        File.WriteAllText(Path.Combine(tree, "src", "Umlaut.cs"), "namespace Demo;\n\npublic static class Überall\n{\n    public static string Grüß() => \"grüß gott 日本 😀\";\n\n    public static int Zwei() => 2;\n}\n");
        using ModelHandle model = OpenModel(_fx.ModelA);
        var fake = new FakeServer(s =>
        {
            if (s.Method == "GET") return (HttpStatusCode.OK, CapsJson(model.ModelId, model.Producer));
            // The server's per-chunk answer: hit for the even spans, miss for the odd ones.
            JsonArray spans = JsonNode.Parse(s.Body)!["request"]!["chunk_cache"]!["chunks"]!.AsArray();
            string chunks = string.Join(",", spans.Select((c, i) => $$"""{"id":"{{c!["id"]!.GetValue<string>()}}","status":"{{(i % 2 == 0 ? "hit" : "miss")}}","why":null}"""));
            return (HttpStatusCode.OK, $$"""{"policy_would_apply":"exact","chunks":[{{chunks}}],"warnings":[]}""");
        });

        JsonElement r = Json(CliRaw(["select", "--model", _fx.ModelA, "--directory", tree, "--path", "src/*.cs", "--include-text", "--server", FakeUrl, "--json"], http: fake));
        Seen explain = Assert.Single(fake.Requests, q => q.Method == "POST");
        Assert.Equal("/v1/chunk-cache/explain", explain.Path);
        JsonObject envelope = JsonNode.Parse(explain.Body)!.AsObject();
        Assert.Equal(["request"], envelope.Select(kv => kv.Key)); // ruling 15
        JsonObject request = envelope["request"]!.AsObject();
        JsonArray messages = request["messages"]!.AsArray();
        Assert.Equal("user", messages.Single()!["role"]!.GetValue<string>());
        byte[] content = Encoding.UTF8.GetBytes(messages.Single()!["content"]!.GetValue<string>()); // ruling 19: one string content
        JsonObject cc = request["chunk_cache"]!.AsObject();
        JsonElement[] selected = r.GetProperty("chunks").EnumerateArray().ToArray();
        JsonArray spans = cc["chunks"]!.AsArray();
        Assert.Equal(selected.Length, spans.Count);
        for (int i = 0; i < spans.Count; i++)
        {
            JsonObject at = spans[i]!["at"]!.AsObject();
            Assert.Equal([0, 0], [at["message"]!.GetValue<int>(), at["part"]!.GetValue<int>()]);
            string text = Encoding.UTF8.GetString(content[at["start"]!.GetValue<int>()..at["end"]!.GetValue<int>()]);
            Assert.Equal(selected[i].GetProperty("text").GetString(), text); // the byte range is the chunk's text
            Assert.Equal("sha256:" + Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text))), spans[i]!["hash"]!.GetValue<string>());
            Assert.Equal("symbol-512", spans[i]!["chunker"]!.GetValue<string>());
            Assert.Equal(i % 2 == 0, selected[i].GetProperty("stored").GetBoolean()); // `stored` is what the server answered, per chunk
        }
        Assert.Contains(spans, sp => Encoding.UTF8.GetString(content[sp!["at"]!["start"]!.GetValue<int>()..sp["at"]!["end"]!.GetValue<int>()]).Contains("😀", StringComparison.Ordinal));
        Assert.Equal(selected.Count(c => c.GetProperty("stored").GetBoolean()), r.GetProperty("stored_chunks").GetInt32());
    }

    [Fact]
    public void Json_Error_IsObjectOnStdout_NonZeroExit()
    {
        string store = NewDir("store");
        string request = Path.Combine(NewDir("in"), "r.json");
        File.WriteAllText(request, """{"model": "m", "messages": []}""");
        using ModelHandle model = OpenModel(_fx.ModelA);
        var failing = new FakeServer(s => s.Method == "GET" ? (HttpStatusCode.OK, CapsJson(model.ModelId, model.Producer))
            : (HttpStatusCode.InternalServerError, """{"error":{"message":"boom","type":"api_error","code":"chunk_store_unavailable"}}"""));
        var unreachable = new FakeServer(_ => throw new HttpRequestException("connection refused"));
        var cases = new (string[] Args, HttpMessageHandler? Http, string Code)[]
        {
            (["frobnicate", "--json"], null, "usage"),
            (["build", "--model", _fx.ModelA, "--store", store, "--json"], null, "usage"),
            (["evict", "--store", store, "--json"], null, "usage"),
            (["status", "--model", _fx.ModelA, "--store", Path.Combine(store, "nope"), "--json"], null, "error"),
            (["explain", "--server", FakeUrl, "--request", request, "--json"], failing, "server_error"),
            (["explain", "--server", FakeUrl, "--request", request, "--json"], unreachable, "server_unreachable"),
        };
        foreach (var (args, http, code) in cases)
        {
            var r = CliRaw(args, http: http);
            Assert.NotEqual(0, r.Code);
            JsonElement root = JsonDocument.Parse(r.Out).RootElement; // exactly one JSON document, nothing else on stdout
            Assert.Equal(JsonValueKind.Object, root.ValueKind);
            Assert.Equal(["error"], root.EnumerateObject().Select(p => p.Name));
            Assert.Equal(code, root.GetProperty("error").GetProperty("code").GetString());
            Assert.False(string.IsNullOrWhiteSpace(root.GetProperty("error").GetProperty("message").GetString()));
            Assert.Empty(SchemaErrors("error", r.Out));
            Assert.NotEmpty(r.Err); // the human-readable form stays on stderr
        }
    }

    [Fact]
    public void ApiKey_FromEnvOnly()
    {
        const string key = "ck-secret-0123456789abcdefABCDEF0123456789";
        Func<string, string?> env = n => n == ApiKeyVar ? key : null;
        string tree = NewTree();
        using ModelHandle model = OpenModel(_fx.ModelA);

        // Not an argument, in any spelling.
        foreach (string flag in new[] { "--api-key", "--key", "--token" })
        {
            var r = CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--server", FakeUrl, flag, key, "--json"]);
            Assert.Equal(2, r.Code);
            Assert.DoesNotContain(key, r.Out + r.Err, StringComparison.Ordinal);
        }

        // From the environment it rides as X-API-Key on every call (capabilities and prewarm); without it there is no header.
        var fake = PrewarmingServer(model, model.ModelId);
        string store = NewDir("store");
        var ok = CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--server", FakeUrl, "--json"], http: fake, env: env);
        Assert.Equal(0, ok.Code);
        Assert.True(fake.Requests.Count >= 2);
        Assert.All(fake.Requests, q => Assert.Equal(key, q.Headers["X-API-Key"]));
        Assert.DoesNotContain(key, ok.Out + ok.Err, StringComparison.Ordinal);
        var anonymous = PrewarmingServer(model, model.ModelId);
        Assert.Equal(0, CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--server", FakeUrl, "--json"], http: anonymous).Code);
        Assert.All(anonymous.Requests, q => Assert.False(q.Headers.ContainsKey("X-API-Key")));

        // A server that echoes the key back in its error body does not get it into our output (stderr or the JSON error).
        var echo = new FakeServer(s => s.Method == "GET" ? (HttpStatusCode.OK, CapsJson(model.ModelId, model.Producer))
            : (HttpStatusCode.Unauthorized, $$$"""{"error":{"message":"bad key {{{key}}}","type":"authentication_error","code":"invalid_api_key"}}"""));
        string texts = Path.Combine(NewDir("in"), "t.json");
        File.WriteAllText(texts, """[{"id": "a", "text": "class A { }"}]""");
        var denied = CliRaw(["prewarm", "--server", FakeUrl, "--texts", texts, "--json", "--verbose"], http: echo, env: env);
        Assert.Equal(1, denied.Code);
        Assert.DoesNotContain(key, denied.Out + denied.Err, StringComparison.Ordinal);
        Assert.Contains("***", denied.Out, StringComparison.Ordinal); // redacted, not dropped: the message still says what happened
        Assert.Equal("server_error", JsonDocument.Parse(denied.Out).RootElement.GetProperty("error").GetProperty("code").GetString());

        // A key that could split a header line is refused without being echoed.
        var bad = CliRaw(["prewarm", "--server", FakeUrl, "--texts", texts], http: echo, env: n => n == ApiKeyVar ? "abc\r\nX-Evil: 1" : null);
        Assert.Equal(2, bad.Code);
        Assert.DoesNotContain("X-Evil", bad.Out + bad.Err, StringComparison.Ordinal);
    }

    [Fact]
    public void ServerBacked_IndexNamedByServerModelAndProducer()
    {
        string tree = NewTree();
        string store = NewDir("store");
        using ModelHandle model = OpenModel(_fx.ModelA);
        const string served = "served-model-7";
        Assert.NotEqual(served, model.ModelId);
        var fake = new FakeServer(s => s.Method == "GET" ? (HttpStatusCode.OK, CapsJson(served, model.Producer, entries: 11, bytes: 2222, budget: 33333)) : (HttpStatusCode.NotFound, "{}"));

        Assert.Equal(0, CliRaw(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--server", FakeUrl, "--index-only"], http: fake).Code);
        string expected = Path.GetFileName(ChunkIndex.DefaultPath(store, "symbol-512", served, model.Producer, ChunkNamespace.Default));
        Assert.Equal([expected], Directory.GetFiles(store, "index-*.json").Select(Path.GetFileName));
        Assert.False(File.Exists(ChunkIndex.DefaultPath(store, "symbol-512", model.ModelId, model.Producer, ChunkNamespace.Default))); // not the local id's name
        Assert.Equal(served, StoreRegistry.Load(store).Get(expected)?.ModelId); // the registry row says so too

        // status finds the same index, and the server's own figures (with the budget only the server knows) fill the store fields.
        JsonElement st = Json(CliRaw(["status", "--model", _fx.ModelA, "--store", store, "--no-scan", "--server", FakeUrl, "--json"], http: fake));
        Assert.Equal(33333, st.GetProperty("budget_bytes").GetInt64());
        Assert.Equal(2222, st.GetProperty("store_bytes").GetInt64());
        Assert.Equal(11, st.GetProperty("entries").GetInt32());
        // Without --server the budget is not known: no budget_bytes (and the index is the local id's, so this is another store).
        string local = NewDir("store");
        Assert.Equal(0, CliRaw(["build", "--model", _fx.ModelA, "--store", local, "--directory", tree, "--index-only"]).Code);
        Assert.False(Json(CliRaw(["status", "--model", _fx.ModelA, "--store", local, "--no-scan", "--json"])).TryGetProperty("budget_bytes", out _));

        // A server whose producer is another model's (other GGUF, backend or dtype) is refused: its keys are not ours.
        var other = new FakeServer(s => s.Method == "GET" ? (HttpStatusCode.OK, CapsJson(served, new string('a', 64))) : (HttpStatusCode.NotFound, "{}"));
        var mismatch = CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--server", FakeUrl, "--index-only"], http: other);
        Assert.Equal(1, mismatch.Code);
        Assert.Contains("producer", mismatch.Err, StringComparison.Ordinal);
    }

    [Fact]
    public void MaxEncodeTokens_CountsTokens()
    {
        string tree = NewTree();
        using ModelHandle model = OpenModel(_fx.ModelA);
        var planner = PrewarmingServer(model, model.ModelId);
        JsonElement plan = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--server", FakeUrl, "--dry-run", "--json", "--show", "0"], http: planner));
        int[] tokens = plan.GetProperty("would_encode").EnumerateArray().Select(c => c.GetProperty("tokens").GetInt32()).ToArray();
        Assert.True(tokens.Length >= 4, $"fixture too small: {tokens.Length} chunks");
        // A budget that holds the first two chunks and not the third: a count of 2 chunks, never "N chunks" for N tokens.
        int cap = tokens[0] + tokens[1] + Math.Min(tokens[2], 1) ;
        int fit = 0;
        for (int sum = 0; fit < tokens.Length && sum + tokens[fit] <= cap; fit++) sum += tokens[fit];
        Assert.InRange(fit, 2, tokens.Length - 1);
        Assert.True(cap > tokens.Length); // it is a token budget: more tokens than chunks exist

        var fake = PrewarmingServer(model, model.ModelId);
        JsonElement r = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--server", FakeUrl, "--max-encode-tokens", cap.ToString(), "--json"], http: fake));
        int posted = fake.Requests.Where(q => q.Method == "POST").Sum(q => JsonNode.Parse(q.Body)!["texts"]!.AsArray().Count);
        long postedTokens = fake.Requests.Where(q => q.Method == "POST").SelectMany(q => JsonNode.Parse(q.Body)!["texts"]!.AsArray())
            .Sum(t => ChunkTokenization.Encode(model.Tokenizer, t!["text"]!.GetValue<string>()).Length);
        Assert.Equal(fit, posted);
        Assert.True(postedTokens <= cap, $"{postedTokens} tokens posted for a budget of {cap}");
        Assert.Equal(postedTokens, r.GetProperty("encode_queued_tokens").GetInt64());

        // --max-chunks counts chunks: the same number as a token budget is not the same thing.
        var byChunks = PrewarmingServer(model, model.ModelId);
        Json(CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--server", FakeUrl, "--max-chunks", "2", "--json"], http: byChunks));
        Assert.Equal(2, byChunks.Requests.Where(q => q.Method == "POST").Sum(q => JsonNode.Parse(q.Body)!["texts"]!.AsArray().Count));
    }

    [Fact]
    public void Evict_OldModeName_RejectedWithHint()
    {
        string store = NewDir("store");
        foreach (string[] old in new[] { new[] { "--all-for-model" }, ["--mode", "all_for_model"], ["--mode", "all-for-model"] })
        {
            var r = CliRaw(["evict", "--offline", "--store", store, .. old, "--model", _fx.ModelA, "--json"]);
            Assert.Equal(2, r.Code);
            Assert.Contains("is now", r.Err, StringComparison.Ordinal); // the message says it was renamed...
            Assert.Contains("--mode all", r.Err, StringComparison.Ordinal); // ...and names the new spelling
            Assert.Equal("usage", JsonDocument.Parse(r.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
        }
        var unknown = CliRaw(["evict", "--offline", "--store", store, "--mode", "frobnicate"]);
        Assert.Equal(2, unknown.Code);
        Assert.Contains("orphans | path | namespace | all", unknown.Err, StringComparison.Ordinal);
        // The new modes are accepted by the parser: path needs --path, and path / namespace need a server.
        Assert.Equal(2, CliRaw(["evict", "--server", FakeUrl, "--mode", "path"]).Code);
        Assert.Equal(2, CliRaw(["evict", "--offline", "--store", store, "--mode", "path", "--path", "src/**"]).Code);
    }

    [Fact]
    public void Build_ControlTokenSpellingTree_EligibilityUsesChunkTokenization()
    {
        string tree = NewDir("tree");
        Directory.CreateDirectory(Path.Combine(tree, "src"));
        File.WriteAllText(Path.Combine(tree, "src", "Tags.cs"), "namespace Demo;\n\n/// <summary>Reasoning tags.</summary>\npublic static class Tags\n{\n    /// <summary>The model writes <think> before it reasons, then </think>.</summary>\n    public const string Open = \"<think>\";\n}\n");
        using ModelHandle model = OpenModel(_fx.ModelSpecials);
        // Precondition: this model parses the spelling as a control token under a bare EncodeRaw but not under ChunkTokenization.
        Assert.Single(model.Tokenizer.EncodeRaw("<think>"));
        Assert.True(ChunkTokenization.Encode(model.Tokenizer, "<think>").Length > 1);

        JsonElement r = Json(CliRaw(["build", "--model", _fx.ModelSpecials, "--store", NewDir("store"), "--directory", tree, "--index-only", "--json"]));
        Assert.Equal("symbol-512", r.GetProperty("chunker").GetString());
        Assert.True(r.GetProperty("chunks").GetInt32() >= 1);
        Assert.Equal(0, r.GetProperty("wire_ineligible_chunks").GetInt32()); // the chunk ids and the wire encode agree

        // And the chunk with the spelled-out tag is SENT: a per-chunk eligibility with a bare EncodeRaw would drop it silently.
        var fake = PrewarmingServer(model, model.ModelId);
        JsonElement queued = Json(CliRaw(["build", "--model", _fx.ModelSpecials, "--store", NewDir("store"), "--directory", tree, "--server", FakeUrl, "--json"], http: fake));
        string[] texts = fake.Requests.Where(q => q.Method == "POST").SelectMany(q => JsonNode.Parse(q.Body)!["texts"]!.AsArray()).Select(t => t!["text"]!.GetValue<string>()).ToArray();
        Assert.Contains(texts, t => t.Contains("<think>", StringComparison.Ordinal));
        Assert.Equal(0, queued.GetProperty("prewarm").GetProperty("skipped_ineligible").GetInt32());
    }

    [Fact]
    public void Build_EligibilityRequiresAllowListSliceAndRoundTrip()
    {
        using ModelHandle model = OpenModel(_fx.ModelA);
        const string text = "int One() => 1;";
        int[] ids = ChunkTokenization.Encode(model.Tokenizer, text);
        Chunk Good() => new("a.cs", 1, 1, ids) { StartOffset = 0, EndOffset = text.Length };
        string TextOf(Chunk _) => text;
        int Count(string chunker, params Chunk[] chunks) => WireEligibility.CountIneligible(chunker, chunks, TextOf, model.Tokenizer);

        Assert.Equal(0, Count("symbol-512", Good()));
        Assert.Equal(1, Count("fixed-512", Good())); // off the allow-list
        Assert.Equal(2, Count("symbol-64", Good(), Good())); // the name carries the size: only symbol-512 is on the list
        Assert.Equal(1, Count("symbol-512", Good() with { WireEligible = false })); // a token-id slice
        Assert.Equal(1, Count("symbol-512", Good() with { TokenIds = ids[..^1] })); // the ids are not what the text encodes to
        Assert.Equal(1, Count("symbol-512", Good() with { TokenIds = [.. ids, ids[0]] }));
        // All three kinds at once, beside two good chunks: each is counted once, none twice.
        Chunk slice = Good() with { WireEligible = false }, broken = Good() with { TokenIds = ids[1..] };
        Assert.Equal(2, Count("symbol-512", Good(), slice, broken, Good()));
        Assert.Equal(4, Count("fixed-512", Good(), slice, broken, Good()));

        // The same through the CLI. An oversized member is cut into TEXT pieces, each tokenised alone: they stay on the wire. Only
        // the fixed-size chunker makes token-id slices, and it is off the allow-list anyway (so that case is the unit rows above).
        string tree = NewDir("tree");
        Directory.CreateDirectory(Path.Combine(tree, "src"));
        string body = string.Concat(Enumerable.Range(0, 120).Select(i => $"        total += Step{i}(value);\n"));
        string source = "namespace Demo;\n\npublic static class Big\n{\n    public static int Run(int value)\n    {\n        int total = 0;\n" + body + "        return total;\n    }\n}\n";
        File.WriteAllText(Path.Combine(tree, "src", "Big.cs"), source);
        var file = new SourceFile("src/Big.cs", source);
        Chunk[] pieces = new DotLLM.ChunkReuse.CodeCorpus.SymbolChunker(model.Tokenizer, 512).Chunk(file).ToArray();
        Assert.True(pieces.Length >= 2, "fixture: the member must be cut into pieces");
        Assert.All(pieces, c => Assert.True(c.WireEligible));
        JsonElement r = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--index-only", "--json"]));
        Assert.Equal(pieces.Length, r.GetProperty("chunks").GetInt32());
        Assert.Equal(0, r.GetProperty("wire_ineligible_chunks").GetInt32());
        JsonElement lab = Json(CliRaw(["build", "--model", _fx.ModelA, "--store", NewDir("store"), "--directory", tree, "--index-only", "--chunker", "fixed", "--lab", "--json"]));
        Assert.Equal(lab.GetProperty("chunks").GetInt32(), lab.GetProperty("wire_ineligible_chunks").GetInt32()); // slices, and off the list
    }

    [Fact]
    public void Prewarm_DeniedAndEscapingPaths_RefusedBeforeAnythingIsSent()
    {
        using ModelHandle model = OpenModel(_fx.ModelA);
        var fake = PrewarmingServer(model, model.ModelId);
        string texts = Path.Combine(NewDir("in"), "texts.json");
        File.WriteAllText(texts, """
            [
              {"id": "ok", "text": "class A { }", "source": {"path": "src/A.cs"}},
              {"id": "nosource", "text": "class B { }"},
              {"id": "env", "text": "KEY=1", "source": {"path": "config/.env.production"}},
              {"id": "pem", "text": "-----BEGIN", "source": {"path": "certs/server.pem"}},
              {"id": "git", "text": "[core]", "source": {"path": ".git/config"}},
              {"id": "up", "text": "class C { }", "source": {"path": "../outside/C.cs"}},
              {"id": "abs", "text": "class D { }", "source": {"path": "/etc/passwd"}},
              {"id": "drive", "text": "class E { }", "source": {"path": "C:\\Users\\x\\E.cs"}}
            ]
            """);
        JsonElement r = Json(CliRaw(["prewarm", "--server", FakeUrl, "--texts", texts, "--json"], http: fake));
        string[] sent = fake.Requests.Where(q => q.Method == "POST").SelectMany(q => JsonNode.Parse(q.Body)!["texts"]!.AsArray()).Select(t => t!["id"]!.GetValue<string>()).ToArray();
        Assert.Equal(["nosource", "ok"], sent.Order(StringComparer.Ordinal));
        Assert.Equal(8, r.GetProperty("requested").GetInt32());
        Assert.Equal(6, r.GetProperty("refused").GetInt32());
        Assert.Equal(["denied_path", "path_outside_corpus_root"], r.GetProperty("refused_reasons").EnumerateArray().Select(x => x.GetString()!).Order(StringComparer.Ordinal));
        // Hashes are not accepted in place of text.
        File.WriteAllText(texts, """[{"id": "h", "hash": "sha256:abc"}]""");
        Assert.Equal(1, CliRaw(["prewarm", "--server", FakeUrl, "--texts", texts], http: fake).Code);
        Assert.Equal(2, CliRaw(["prewarm", "--texts", texts]).Code); // no server, no prewarm
    }

    // ------------------------------------------------------------------------------------------------ Task 17 helpers

    /// <summary>One request as the CLI sent it.</summary>
    private sealed record Seen(string Method, string Path, IReadOnlyDictionary<string, string> Headers, string Body);

    /// <summary>A fake server: records every request (method, path, headers, body bytes as sent) and answers from a function.</summary>
    private sealed class FakeServer(Func<Seen, (HttpStatusCode Status, string Body)> answer) : HttpMessageHandler
    {
        public List<Seen> Requests { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancellationToken);
            var headers = request.Headers.ToDictionary(h => h.Key, h => string.Join(",", h.Value), StringComparer.OrdinalIgnoreCase);
            var seen = new Seen(request.Method.Method, request.RequestUri!.AbsolutePath, headers, body);
            lock (Requests) Requests.Add(seen);
            (HttpStatusCode status, string text) = answer(seen);
            return new HttpResponseMessage(status) { Content = new StringContent(text, Encoding.UTF8, "application/json") };
        }
    }

    /// <summary>A server with the capabilities of <paramref name="model"/> (as <paramref name="modelId"/>) that accepts prewarm texts.</summary>
    private FakeServer PrewarmingServer(ModelHandle model, string modelId, bool skewed = false) => new(s =>
    {
        if (s is { Method: "GET", Path: "/v1/chunk-cache" }) return (HttpStatusCode.OK, CapsJson(modelId, model.Producer));
        if (s is { Method: "POST", Path: "/v1/chunk-cache/prewarm" })
        {
            JsonArray texts = JsonNode.Parse(s.Body)!["texts"]!.AsArray();
            long tokens = texts.Sum(t => ChunkTokenization.Encode(model.Tokenizer, t!["text"]!.GetValue<string>()).Length);
            // skewed: the server already holds one text and so queues one less (and fewer tokens) than were sent.
            if (skewed)
                return (HttpStatusCode.OK, $$"""{"requested":{{texts.Count}},"already_stored":1,"queued":{{texts.Count - 1}},"queued_tokens":{{tokens - 7}},"refused":0,"refused_reasons":[]}""");
            return (HttpStatusCode.OK, $$"""{"requested":{{texts.Count}},"already_stored":0,"queued":{{texts.Count}},"queued_tokens":{{tokens}},"refused":0,"refused_reasons":[]}""");
        }
        if (s is { Method: "POST", Path: "/v1/chunk-cache/evict" })
            return (HttpStatusCode.OK, """{"evicted":0,"freed_bytes":0,"kept_referenced":0,"dry_run":false}""");
        if (s is { Method: "POST", Path: "/v1/chunk-cache/explain" })
            return (HttpStatusCode.OK, """{"policy_would_apply":"exact","chunks":[],"warnings":[]}""");
        return (HttpStatusCode.NotFound, "{}");
    });

    /// <summary>The contract's <c>capabilities.json</c> with the Task 14 identity fields of the model (and store figures) patched in.</summary>
    private static string CapsJson(string modelId, string producer, long entries = 0, long bytes = 0, long budget = 0)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1", "capabilities.json");
        JsonObject caps = JsonNode.Parse(File.ReadAllBytes(path))!.AsObject();
        JsonObject model = caps["model"]!.AsObject();
        model["id"] = modelId;
        model["producer"] = producer;
        model["tokenizer_id"] = "tok-test";
        JsonObject store = caps["store"]!.AsObject();
        store["entries"] = entries;
        store["bytes"] = bytes;
        store["budget_bytes"] = budget;
        return caps.ToJsonString();
    }

    /// <summary>Violations of <paramref name="json"/> against the schema file <c>Fixtures/v1/schemas/&lt;name&gt;.json</c> (empty = valid).</summary>
    private static List<string> SchemaErrors(string name, string json)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1", "schemas", name + ".json");
        using JsonDocument schema = JsonDocument.Parse(File.ReadAllText(path));
        using JsonDocument doc = JsonDocument.Parse(json); // one JSON document, nothing else
        var errors = new List<string>();
        Validate(doc.RootElement, schema.RootElement, "$", errors);
        return errors;
    }

    // The subset of JSON Schema the §7.2 files use: type, enum, required, properties, additionalProperties: false, items.
    private static void Validate(JsonElement value, JsonElement schema, string path, List<string> errors)
    {
        if (schema.TryGetProperty("type", out JsonElement type))
        {
            string t = type.GetString()!;
            bool ok = t switch
            {
                "object" => value.ValueKind == JsonValueKind.Object,
                "array" => value.ValueKind == JsonValueKind.Array,
                "string" => value.ValueKind == JsonValueKind.String,
                "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
                "number" => value.ValueKind == JsonValueKind.Number,
                "integer" => value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out _),
                _ => throw new InvalidOperationException("schema type " + t),
            };
            if (!ok)
            {
                errors.Add($"{path}: expected {t}, found {value.ValueKind}");
                return;
            }
        }
        if (schema.TryGetProperty("enum", out JsonElement allowed) && !allowed.EnumerateArray().Any(a => a.ToString() == value.ToString()))
            errors.Add($"{path}: '{value}' is not one of {allowed}");
        if (value.ValueKind == JsonValueKind.Object)
        {
            if (schema.TryGetProperty("required", out JsonElement required))
                foreach (JsonElement name in required.EnumerateArray())
                    if (!value.TryGetProperty(name.GetString()!, out _)) errors.Add($"{path}: missing required property {name.GetString()}");
            schema.TryGetProperty("properties", out JsonElement props);
            foreach (JsonProperty p in value.EnumerateObject())
            {
                if (props.ValueKind == JsonValueKind.Object && props.TryGetProperty(p.Name, out JsonElement sub)) Validate(p.Value, sub, path + "." + p.Name, errors);
                else if (schema.TryGetProperty("additionalProperties", out JsonElement extra) && extra.ValueKind == JsonValueKind.False)
                    errors.Add($"{path}: unexpected property {p.Name}");
            }
        }
        if (value.ValueKind == JsonValueKind.Array && schema.TryGetProperty("items", out JsonElement items))
        {
            int i = 0;
            foreach (JsonElement item in value.EnumerateArray()) Validate(item, items, $"{path}[{i++}]", errors);
        }
    }
}
