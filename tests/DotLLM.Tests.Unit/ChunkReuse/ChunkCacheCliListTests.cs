using System.Net;
using System.Security.Cryptography;
using System.Text.Json;
using DotLLM.ChunkCache;
using DotLLM.ChunkReuse.Store;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <c>chunkcache list</c> (dotLLM issue 1721): a read-only inventory of one or more stores (entries and bytes, the owner-lock probe, the
/// server ledger, and per index the model, chunker and chunk counts against what the store holds). It never takes a lock, never writes
/// the registry, ledger or an index, and never creates a directory.
/// </summary>
public sealed partial class ChunkCacheCliTests
{
    private string BuildStore(string store, string tree, string model, params string[] extra) =>
        Json(CliRaw(["build", "--model", model, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--lab", "--offline", "--json", .. extra])).GetRawText();

    /// <summary>Every file and directory under <paramref name="root"/> with its size, last write and content hash (a change anywhere shows).</summary>
    private static List<string> Snapshot(string root)
    {
        var lines = new List<string>();
        foreach (string entry in Directory.EnumerateFileSystemEntries(root, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            string rel = Path.GetRelativePath(root, entry);
            if (Directory.Exists(entry)) lines.Add("D " + rel);
            else lines.Add($"F {rel} {new FileInfo(entry).Length} {File.GetLastWriteTimeUtc(entry).Ticks} {Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(entry)))}");
        }
        return lines;
    }

    private static JsonElement OnlyStore(JsonElement list)
    {
        JsonElement stores = list.GetProperty("stores");
        Assert.Equal(1, stores.GetArrayLength());
        return stores[0];
    }

    [Fact]
    public void List_ReportsEntriesBytesAndPerIndexModelAndChunkCounts_MatchesSchema()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement a = JsonDocument.Parse(BuildStore(store, tree, _fx.ModelA)).RootElement;
        JsonElement b = JsonDocument.Parse(BuildStore(store, tree, _fx.ModelB)).RootElement;
        var run = CliRaw(["list", "--store", store, "--json"]);
        Assert.True(run.Code == 0, run.Err);
        Assert.Empty(SchemaErrors("list", run.Out));
        JsonElement s = OnlyStore(JsonDocument.Parse(run.Out).RootElement);

        // Oracle: the library's own count of the store, and the figures each build printed.
        StoreStats stats = new FileChunkStore(store).GetStats();
        Assert.Equal(store, s.GetProperty("path").GetString());
        Assert.Equal(stats.Entries, s.GetProperty("entries").GetInt32());
        Assert.Equal(stats.EntryBytes, s.GetProperty("entry_bytes").GetInt64());
        Assert.Equal("free", s.GetProperty("owner_lock").GetString());
        Assert.Equal("absent", s.GetProperty("ledger").GetProperty("state").GetString());
        JsonElement[] indexes = s.GetProperty("indexes").EnumerateArray().ToArray();
        Assert.Equal(2, indexes.Length);
        foreach (JsonElement built in new[] { a, b })
        {
            JsonElement ix = Assert.Single(indexes, i => i.GetProperty("model_id").GetString() == built.GetProperty("model_id").GetString());
            Assert.Equal("ok", ix.GetProperty("state").GetString());
            Assert.Equal("symbol-64", ix.GetProperty("chunker").GetString());
            Assert.Equal(built.GetProperty("files").GetInt32(), ix.GetProperty("files").GetInt32());
            Assert.Equal(built.GetProperty("chunks").GetInt32(), ix.GetProperty("chunks").GetInt32());
            Assert.Equal(built.GetProperty("tokens").GetInt64(), ix.GetProperty("tokens").GetInt64());
            Assert.Equal(built.GetProperty("distinct_keys").GetInt32(), ix.GetProperty("distinct_keys").GetInt32());
            Assert.Equal(ix.GetProperty("distinct_keys").GetInt32(), ix.GetProperty("stored_keys").GetInt32());
            Assert.Equal(0, ix.GetProperty("missing_keys").GetInt32());
            Assert.Equal(tree, ix.GetProperty("source_path").GetString());
            Assert.Equal("directory", ix.GetProperty("source_kind").GetString());
            Assert.StartsWith("index-symbol-64.", ix.GetProperty("file").GetString(), StringComparison.Ordinal);
        }
        Assert.NotEqual(indexes[0].GetProperty("file").GetString(), indexes[1].GetProperty("file").GetString());

        // The schema check can fail: dropping a required field or adding a stray one is reported.
        Assert.NotEmpty(SchemaErrors("list", run.Out.Replace("\"owner_lock\"", "\"owner\"", StringComparison.Ordinal)));
        Assert.NotEmpty(SchemaErrors("list", run.Out.Replace("\"entries\"", "\"entries\": 1, \"stray\"", StringComparison.Ordinal)));
    }

    [Fact]
    public void List_CountsChunksTheStoreLacks()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement built = JsonDocument.Parse(BuildStore(store, tree, _fx.ModelA)).RootElement;
        string[] files = Directory.GetFiles(store, "*.ckv", SearchOption.AllDirectories).Order(StringComparer.Ordinal).Take(3).ToArray();
        foreach (string f in files) File.Delete(f);

        JsonElement s = OnlyStore(Json(CliRaw(["list", "--store", store, "--json"])));
        JsonElement ix = Assert.Single(s.GetProperty("indexes").EnumerateArray());
        Assert.Equal(3, ix.GetProperty("missing_keys").GetInt32());
        Assert.Equal(built.GetProperty("distinct_keys").GetInt32() - 3, ix.GetProperty("stored_keys").GetInt32());
        Assert.Equal(built.GetProperty("distinct_keys").GetInt32() - 3, s.GetProperty("entries").GetInt32());

        // An index-only store holds nothing: every key is missing.
        string bare = NewDir("bare");
        JsonElement indexOnly = JsonDocument.Parse(Json(CliRaw(["build", "--model", _fx.ModelA, "--store", bare, "--directory", tree, "--chunk-tokens", "64", "--lab", "--index-only", "--json"])).GetRawText()).RootElement;
        JsonElement bs = OnlyStore(Json(CliRaw(["list", "--store", bare, "--json"])));
        Assert.Equal(0, bs.GetProperty("entries").GetInt32());
        JsonElement bi = Assert.Single(bs.GetProperty("indexes").EnumerateArray());
        Assert.Equal(0, bi.GetProperty("stored_keys").GetInt32());
        Assert.Equal(indexOnly.GetProperty("distinct_keys").GetInt32(), bi.GetProperty("missing_keys").GetInt32());
    }

    [Fact]
    public void List_ReportsAHeldOwnerLock_WithoutTakingOneOrWaitingOnAStoreLock()
    {
        string tree = NewTree();
        string store = NewDir("store");
        BuildStore(store, tree, _fx.ModelA);
        Assert.Equal("free", OnlyStore(Json(CliRaw(["list", "--store", store, "--json"]))).GetProperty("owner_lock").GetString());

        using (StoreOwnerLock? server = StoreOwnerLock.TryAcquire(store)) // a running server owns the store
        {
            Assert.NotNull(server);
            var run = CliRaw(["list", "--store", store, "--json"]);
            Assert.True(run.Code == 0, run.Err); // a held lock is information, not an error
            Assert.Equal("held", OnlyStore(JsonDocument.Parse(run.Out).RootElement).GetProperty("owner_lock").GetString());
            Assert.Contains("owner lock: held", CliRaw(["list", "--store", store]).Out, StringComparison.Ordinal);
        }

        // list released what it probed: the next writer can take the store at once.
        using (StoreOwnerLock? next = StoreOwnerLock.TryAcquire(store)) Assert.NotNull(next);
        Assert.Equal("free", OnlyStore(Json(CliRaw(["list", "--store", store, "--json"]))).GetProperty("owner_lock").GetString());

        // It takes no store lock either: with the short store lock held it still answers at once (a waiting list would hit the 30 s default).
        using (ChunkIndex.AcquireStoreLock(store, TimeSpan.FromSeconds(5)))
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            Assert.Equal(0, CliRaw(["list", "--store", store, "--json"]).Code);
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(20), $"list took {sw.Elapsed}");
        }
    }

    [Fact]
    public void List_IsReadOnly_WritesNothingAndCreatesNothing()
    {
        string tree = NewTree();
        string store = NewDir("store");
        BuildStore(store, tree, _fx.ModelA);
        List<string> before = Snapshot(store);
        Assert.Equal(0, CliRaw(["list", "--store", store, "--json"]).Code);
        Assert.Equal(0, CliRaw(["list", "--store", store]).Code);
        Assert.Equal(before, Snapshot(store));

        // No registry: the library rebuilds one in memory from the index files; list must not save it.
        string registry = Path.Combine(store, StoreLayout.ConfigFileName);
        File.Delete(registry);
        File.Delete(Path.Combine(store, StoreOwnerLock.FileName));
        before = Snapshot(store);
        var rebuilt = CliRaw(["list", "--store", store, "--json"]);
        Assert.Equal(0, rebuilt.Code);
        Assert.False(File.Exists(registry), "list wrote the registry");
        Assert.False(File.Exists(Path.Combine(store, StoreOwnerLock.FileName)), "list created the owner lock file");
        Assert.Equal(before, Snapshot(store));
        JsonElement s = OnlyStore(JsonDocument.Parse(rebuilt.Out).RootElement);
        Assert.Equal(1, s.GetProperty("indexes").GetArrayLength()); // found from the index file on disk
        Assert.True(s.TryGetProperty("registry_note", out _));

        // A corrupt registry is reported and left exactly as it is.
        File.WriteAllText(registry, "{ not json");
        before = Snapshot(store);
        Assert.Equal(0, CliRaw(["list", "--store", store, "--json"]).Code);
        Assert.Equal(before, Snapshot(store));
        Assert.Equal("{ not json", File.ReadAllText(registry));

        // A server ledger is read, never written.
        File.WriteAllText(Path.Combine(store, ServerLedger.FileName), "{ broken");
        before = Snapshot(store);
        JsonElement withLedger = OnlyStore(Json(CliRaw(["list", "--store", store, "--json"])));
        Assert.Equal("unreadable", withLedger.GetProperty("ledger").GetProperty("state").GetString());
        Assert.Equal(before, Snapshot(store));
    }

    [Fact]
    public void List_OfAMissingStore_FailsWithoutCreatingTheDirectory()
    {
        string parent = NewDir("parent");
        string missing = Path.Combine(parent, "no", "such", "store");
        var run = CliRaw(["list", "--store", missing, "--json"]);
        Assert.Equal(1, run.Code);
        JsonElement err = JsonDocument.Parse(run.Out).RootElement.GetProperty("error");
        Assert.Equal("error", err.GetProperty("code").GetString());
        Assert.Equal(1, err.GetProperty("exit_code").GetInt32());
        Assert.Empty(SchemaErrors("error", run.Out));
        Assert.Contains("Store not found", run.Err, StringComparison.Ordinal);
        Assert.False(Directory.Exists(Path.Combine(parent, "no")), "list created the store directory");
        Assert.Empty(Directory.EnumerateFileSystemEntries(parent));

        // One missing store among several fails the whole call and still creates nothing.
        string tree = NewTree();
        string good = NewDir("good");
        BuildStore(good, tree, _fx.ModelA);
        Assert.Equal(1, CliRaw(["list", "--store", good, "--store", missing]).Code);
        Assert.Empty(Directory.EnumerateFileSystemEntries(parent));
    }

    [Fact]
    public void List_SeveralStores_AndUsageErrors()
    {
        string tree = NewTree();
        string one = NewDir("one");
        string two = NewDir("two");
        BuildStore(one, tree, _fx.ModelA);
        BuildStore(two, tree, _fx.ModelB);
        JsonElement list = Json(CliRaw(["list", "--store", one, "--store", two, "--json"]));
        string[] paths = list.GetProperty("stores").EnumerateArray().Select(x => x.GetProperty("path").GetString()!).ToArray();
        Assert.Equal([one, two], paths); // in the order given

        Assert.Equal(2, CliRaw(["list"]).Code); // --store is required
        Assert.Equal(2, CliRaw(["list", "--store", one + "," + two]).Code); // no comma lists
        Assert.Equal(2, CliRaw(["list", "--store", one, "--model", _fx.ModelA]).Code); // list reads no model
        Assert.Equal(2, CliRaw(["list", "--store", one, "--no-such-option"]).Code);
        Assert.Equal("usage", JsonDocument.Parse(CliRaw(["list", "--json"]).Out).RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public void List_ReportsAnUnreadableAndAMissingIndexFile_AsStatesNotErrors()
    {
        string tree = NewTree();
        string store = NewDir("store");
        BuildStore(store, tree, _fx.ModelA);
        string index = Directory.GetFiles(store, "index-*.json").Single();
        string good = File.ReadAllText(index);
        File.WriteAllText(index, "{ this is not an index");
        JsonElement bad = Assert.Single(OnlyStore(Json(CliRaw(["list", "--store", store, "--json"]))).GetProperty("indexes").EnumerateArray());
        Assert.Equal("unreadable", bad.GetProperty("state").GetString());
        Assert.False(string.IsNullOrEmpty(bad.GetProperty("error").GetString()));

        File.Delete(index);
        // The registry still lists it; the file is gone (the registry reconciles against the disk, so it may drop the row: either is reported, never thrown).
        var gone = CliRaw(["list", "--store", store, "--json"]);
        Assert.Equal(0, gone.Code);
        Assert.Empty(SchemaErrors("list", gone.Out));
        File.WriteAllText(index, good);
        Assert.Equal("ok", Assert.Single(OnlyStore(Json(CliRaw(["list", "--store", store, "--json"]))).GetProperty("indexes").EnumerateArray()).GetProperty("state").GetString());
    }

    [Fact]
    public void List_WithServer_AddsTheServersOwnFigures_AndNeverPrintsTheKey()
    {
        string tree = NewTree();
        string store = NewDir("store");
        BuildStore(store, tree, _fx.ModelA);
        using ModelHandle model = OpenModel(_fx.ModelA);
        var fake = new FakeServer(s => s is { Method: "GET", Path: "/v1/chunk-cache" }
            ? (HttpStatusCode.OK, CapsJson(model.ModelId, model.Producer, entries: 12, bytes: 3456, budget: 99999))
            : (HttpStatusCode.NotFound, "{}"));
        const string key = "list-secret-api-key-0123456789abcdef";
        var run = CliRaw(["list", "--store", store, "--server", FakeUrl, "--json"], http: fake, env: n => n == ApiKeyVar ? key : null);
        Assert.True(run.Code == 0, run.Err);
        Assert.Empty(SchemaErrors("list", run.Out));
        JsonElement server = JsonDocument.Parse(run.Out).RootElement.GetProperty("server");
        Assert.Equal(FakeUrl, server.GetProperty("url").GetString());
        Assert.Equal(model.ModelId, server.GetProperty("model_id").GetString());
        Assert.Equal(model.Producer, server.GetProperty("producer").GetString());
        Assert.Equal(12, server.GetProperty("entries").GetInt64());
        Assert.Equal(3456, server.GetProperty("bytes").GetInt64());
        Assert.Equal(99999, server.GetProperty("budget_bytes").GetInt64());
        Assert.Equal(key, Assert.Single(fake.Requests).Headers["X-API-Key"]); // sent to the server, and only there
        Assert.DoesNotContain(key, run.Out + run.Err, StringComparison.Ordinal);

        // An unreachable server is a runtime error (exit 1), reported as server_unreachable, and the key still never shows.
        var down = new FakeServer(_ => throw new HttpRequestException("connection refused: " + key));
        var failed = CliRaw(["list", "--store", store, "--server", FakeUrl, "--json"], http: down, env: n => n == ApiKeyVar ? key : null);
        Assert.Equal(1, failed.Code);
        Assert.Equal("server_unreachable", JsonDocument.Parse(failed.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
        Assert.DoesNotContain(key, failed.Out + failed.Err, StringComparison.Ordinal);
    }

    [Fact]
    public void List_TextOutput_NamesTheStoreItsIndexesAndTheAdvisoryLockState()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement built = JsonDocument.Parse(BuildStore(store, tree, _fx.ModelA)).RootElement;
        var run = CliRaw(["list", "--store", store]);
        Assert.Equal(0, run.Code);
        Assert.Contains($"store {store}: {Directory.GetFiles(store, "*.ckv", SearchOption.AllDirectories).Length} entries", run.Out, StringComparison.Ordinal);
        Assert.Contains("owner lock: free (advisory", run.Out, StringComparison.Ordinal);
        Assert.Contains("symbol-64", run.Out, StringComparison.Ordinal);
        Assert.Contains($"{built.GetProperty("chunks").GetInt32()} chunks", run.Out, StringComparison.Ordinal);
        // Under --json the text goes to stderr and stdout is one document.
        var json = CliRaw(["list", "--store", store, "--json"]);
        JsonDocument.Parse(json.Out).Dispose();
    }
}
