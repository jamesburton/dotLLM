using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <see cref="ServerLedger"/> (Phase I Task 3, spec L2): the server's record of the entries it wrote, and the ownership-aware
/// mark-and-sweep (<see cref="ChunkIndex.StoreWideOrphans"/>, <see cref="ChunkIndex.ReferencedKeysUnder"/>) that must never
/// treat a ledger key as an orphan, and must fail closed (throw, delete nothing) when the ledger cannot be read. The oracles are
/// the ledger file on disk (parsed with <see cref="JsonNode"/>, not the ledger's own reader) and the store's files.
/// </summary>
public sealed class ServerLedgerTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private const string TenantA = "k-0123456789abcdef";
    private const string TenantB = "tenant-b";
    private static readonly DateTimeOffset T0 = new(2026, 11, 2, 10, 0, 0, TimeSpan.Zero);

    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-ledger-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    private static ChunkKey Key(string ns, params int[] ids) => ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, ids, @namespace: ns);

    /// <summary>Puts a small real entry under <paramref name="key"/> and returns its file size.</summary>
    private static long Put(FileChunkStore store, ChunkKey key, int seed = 1)
    {
        store.Put(key, new ChunkKvEntry("fp-a", [seed, seed + 1], [[1f, -2f]], [[0.5f, 4f]]));
        return new FileInfo(store.PathOf(key)).Length;
    }

    private string LedgerFile => ServerLedger.DefaultPath(_root);

    /// <summary>Saves under the store lock, passing it as proof (the server's put + record + save scope).</summary>
    private void SaveLocked(ServerLedger ledger)
    {
        using StoreLockHandle held = ChunkIndex.AcquireStoreLock(_root, TimeSpan.FromSeconds(30));
        ledger.Save(held);
    }

    // ------------------------------------------------------------------------------------------------ format and round trip

    [Fact]
    public void Ledger_RoundTrip()
    {
        var store = new FileChunkStore(_root);
        ChunkKey a1 = Key(TenantA, 1), a2 = Key(TenantA, 2), d1 = Key(ChunkNamespace.Default, 1), b1 = Key(TenantB, 1);
        foreach (ChunkKey k in new[] { a1, a2, d1, b1 }) Put(store, k);

        ServerLedger ledger = ServerLedger.Load(_root);
        Assert.Empty(ledger.Rows); // missing file = empty
        Assert.False(File.Exists(LedgerFile)); // Load never creates it
        var plus2h = new DateTimeOffset(2026, 11, 2, 12, 30, 0, TimeSpan.FromHours(2));
        ledger.Record(a1, TenantA, "src/App/A.cs", 100, T0);
        ledger.Record(a2, TenantA, null, 200, T0.AddMinutes(1));
        ledger.Record(d1, ChunkNamespace.Default, "lib\\B.cs", 300, plus2h); // normalised to '/'
        ledger.Record(b1, TenantB, "x.cs", 400, T0.AddMinutes(3));
        // A second sighting keeps the first-seen time and provenance, takes the new size and the later use.
        ledger.Record(a1, TenantA, "other/Z.cs", 150, T0.AddMinutes(10));
        // A sighting with an older clock does not move LastUsed backwards.
        ledger.Record(a2, TenantA, "late/P.cs", 200, T0.AddMinutes(-5));
        SaveLocked(ledger);

        // Independent oracle: the file itself.
        JsonNode doc = JsonNode.Parse(File.ReadAllText(LedgerFile))!;
        Assert.Equal(ServerLedger.FormatVersion, doc["formatVersion"]!.GetValue<int>());
        Assert.Equal(1, ServerLedger.FormatVersion);
        JsonArray rows = doc["rows"]!.AsArray();
        Assert.Equal(4, rows.Count);
        // Rows are written grouped by namespace (ordinal), then by key: deterministic bytes for identical content.
        string[] order = rows.Select(r => r!["namespace"]!.GetValue<string>() + "/" + r["key"]!.GetValue<string>()).ToArray();
        Assert.Equal(order.Order(StringComparer.Ordinal), order);

        ServerLedger back = ServerLedger.Load(_root);
        Assert.Equal(0, back.DroppedOnLoad);
        Assert.Equal(4, back.Rows.Count);
        LedgerRow r1 = back.Rows.Single(r => r.KeyHex == a1.Hex);
        Assert.Equal(new LedgerRow(a1.Hex, TenantA, "src/App/A.cs", 150, T0, T0.AddMinutes(10)), r1);
        LedgerRow r2 = back.Rows.Single(r => r.KeyHex == a2.Hex);
        Assert.Equal(new LedgerRow(a2.Hex, TenantA, "late/P.cs", 200, T0.AddMinutes(1), T0.AddMinutes(1)), r2); // first non-null path adopted
        LedgerRow r3 = back.Rows.Single(r => r.KeyHex == d1.Hex);
        Assert.Equal(new LedgerRow(d1.Hex, ChunkNamespace.Default, "lib/B.cs", 300, plus2h, plus2h), r3);
        Assert.Equal(TimeSpan.FromHours(2), r3.FirstSeen.Offset); // the offset survives the round trip
        Assert.Equal(new LedgerRow(b1.Hex, TenantB, "x.cs", 400, T0.AddMinutes(3), T0.AddMinutes(3)), back.Rows.Single(r => r.KeyHex == b1.Hex));

        // Saving the loaded ledger again writes the same bytes.
        byte[] before = File.ReadAllBytes(LedgerFile);
        SaveLocked(back);
        Assert.Equal(before, File.ReadAllBytes(LedgerFile));
    }

    [Fact]
    public void Ledger_Record_RefusesBadInput()
    {
        new FileChunkStore(_root);
        ServerLedger ledger = ServerLedger.Load(_root);
        ChunkKey a1 = Key(TenantA, 1);
        Assert.Throws<ArgumentException>(() => ledger.Record(a1, "Tenant", null, 1, T0));      // namespace rule
        Assert.Throws<ArgumentException>(() => ledger.Record(a1, "a/b", null, 1, T0));
        Assert.Throws<ArgumentNullException>(() => ledger.Record(a1, null!, null, 1, T0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ledger.Record(a1, TenantA, null, -1, T0));
        Assert.Throws<ArgumentException>(() => ledger.Record(new ChunkKey("not-a-key"), TenantA, null, 1, T0));
        Assert.Throws<ArgumentException>(() => ledger.Record(a1, TenantA, "/abs/A.cs", 1, T0)); // provenance is tree-relative
        Assert.Throws<ArgumentException>(() => ledger.Record(a1, TenantA, "../A.cs", 1, T0));
        Assert.Empty(ledger.Rows);
        ledger.Record(a1, TenantA, null, 1, T0);
        // A key belongs to one namespace (the namespace is hashed into it): recording it under another is a caller bug.
        Assert.Throws<InvalidOperationException>(() => ledger.Record(a1, TenantB, null, 1, T0));
        Assert.Equal(TenantA, Assert.Single(ledger.Rows).Namespace);
        Assert.Throws<ArgumentException>(() => ledger.InNamespace("UPPER").ToList());
    }

    [Fact]
    public void Ledger_RowWithoutStoreFile_DroppedOnLoad()
    {
        var store = new FileChunkStore(_root);
        ChunkKey kept = Key(TenantA, 1), gone = Key(TenantA, 2), neverWritten = Key(TenantB, 3);
        Put(store, kept);
        Put(store, gone);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(kept, TenantA, null, 1, T0);
        ledger.Record(gone, TenantA, null, 1, T0);
        ledger.Record(neverWritten, TenantB, null, 1, T0); // e.g. a manual delete or an out-of-band eviction
        SaveLocked(ledger);
        store.Evict(gone);

        ServerLedger back = ServerLedger.Load(_root);
        Assert.Equal(2, back.DroppedOnLoad);
        Assert.Equal([kept.Hex], back.Rows.Select(r => r.KeyHex));
        SaveLocked(back);
        Assert.Single(JsonNode.Parse(File.ReadAllText(LedgerFile))!["rows"]!.AsArray()); // the next save persists the drop
    }

    [Fact]
    public void Ledger_LruOrder_Deterministic()
    {
        var store = new FileChunkStore(_root);
        ChunkKey[] keys = Enumerable.Range(1, 5).Select(i => Key(i % 2 == 0 ? TenantA : TenantB, i)).ToArray();
        foreach (ChunkKey k in keys) Put(store, k);
        ServerLedger ledger = ServerLedger.Load(_root);
        // FirstSeen ascends with i, LastUsed descends: LRU by FirstSeen would give the reverse order.
        for (int i = 0; i < 5; i++)
        {
            ledger.Record(keys[i], i % 2 == 1 ? TenantA : TenantB, null, 10, T0.AddMinutes(i));
            ledger.Touch(keys[i], T0.AddHours(10 - i));
        }
        // Two rows share LastUsed: the tie goes to the ordinal key order.
        ledger.Touch(keys[0], T0.AddHours(7)); // keys[0] was at +10 h; Touch never goes backwards
        ledger.Touch(keys[3], T0.AddHours(8)); // keys[3] was at +7 h, now ties keys[2] at +8 h
        Assert.False(ledger.Touch(Key(TenantA, 99), T0)); // unknown key: no row created
        Assert.True(ledger.Touch(keys[4], T0.AddHours(6)));

        string[] expected =
        [
            keys[4].Hex,                                                                  // +6 h
            .. new[] { keys[2].Hex, keys[3].Hex }.Order(StringComparer.Ordinal),          // +8 h tie
            keys[1].Hex,                                                                  // +9 h
            keys[0].Hex,                                                                  // +10 h
        ];
        Assert.Equal(expected, ledger.LeastRecentlyUsed().Select(r => r.KeyHex));
        SaveLocked(ledger);
        Assert.Equal(expected, ServerLedger.Load(_root).LeastRecentlyUsed().Select(r => r.KeyHex)); // same after a reload
    }

    [Fact]
    public void Ledger_UnderPath_UsesGlobAndPathComparer()
    {
        var store = new FileChunkStore(_root);
        ChunkKey a = Key(TenantA, 1), b = Key(TenantA, 2), c = Key(TenantA, 3), n = Key(TenantA, 4), d = Key(TenantB, 5);
        foreach (ChunkKey k in new[] { a, b, c, n, d }) Put(store, k);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(a, TenantA, "src/App/A.cs", 1, T0);
        ledger.Record(b, TenantA, "SRC/app/B.cs", 1, T0);
        ledger.Record(c, TenantA, "lib/App/C.cs", 1, T0);
        ledger.Record(n, TenantA, null, 1, T0); // no provenance: never under any path
        ledger.Record(d, TenantB, "src\\App\\Deep\\D.cs", 1, T0);

        string[] Under(string glob) => ledger.UnderPath(glob).Select(r => r.KeyHex).Order(StringComparer.Ordinal).ToArray();
        string[] Sorted(params ChunkKey[] ks) => ks.Select(k => k.Hex).Order(StringComparer.Ordinal).ToArray();

        bool caseInsensitive = ChunkIndex.PathComparer.Equals("A", "a");
        Assert.Equal(OperatingSystem.IsWindows(), caseInsensitive);
        Assert.Equal(caseInsensitive ? Sorted(a, b, d) : Sorted(a, d), Under("src/App/"));
        Assert.Equal(caseInsensitive ? Sorted(a, b) : Sorted(a), Under("src/App/*.cs"));
        Assert.Equal(Sorted(a, b, c, d), Under("*.cs"));
        Assert.Equal(Sorted(c), Under("lib/**"));
        Assert.Empty(Under("App/"));
        Assert.Throws<ArgumentException>(() => ledger.UnderPath("/src/**").ToList());
        // A negated glob selects the rows with provenance that do not match; rows without provenance never match.
        Assert.Equal(caseInsensitive ? Sorted(c) : Sorted(b, c), Under("!src/App/"));
        Assert.Equal(caseInsensitive ? Sorted(c) : Sorted(b, c), Under("! src/**"));
        // UnderPath crosses namespaces (an admin view); a tenant scopes it with InNamespace.
        Assert.Equal(Sorted(d), ledger.UnderPath("src/App/").Intersect(ledger.InNamespace(TenantB)).Select(r => r.KeyHex).ToArray());
    }

    [Fact]
    public void Ledger_TwoNamespaces_DoNotAffectEachOther()
    {
        var store = new FileChunkStore(_root);
        ChunkKey a1 = Key(TenantA, 1), a2 = Key(TenantA, 2), b1 = Key(TenantB, 1), d1 = Key(ChunkNamespace.Default, 1);
        foreach (ChunkKey k in new[] { a1, a2, b1, d1 }) Put(store, k);
        Assert.NotEqual(a1, b1); // same tokens, other namespace, other key
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(a1, TenantA, "p.cs", 1, T0);
        ledger.Record(a2, TenantA, "p.cs", 1, T0);
        ledger.Record(b1, TenantB, "p.cs", 1, T0);
        ledger.Record(d1, ChunkNamespace.Default, "p.cs", 1, T0);

        Assert.Equal(new[] { a1.Hex, a2.Hex }.Order(StringComparer.Ordinal), ledger.InNamespace(TenantA).Select(r => r.KeyHex).Order(StringComparer.Ordinal));
        Assert.Equal([b1.Hex], ledger.InNamespace(TenantB).Select(r => r.KeyHex));
        Assert.Equal([d1.Hex], ledger.InNamespace(ChunkNamespace.Default).Select(r => r.KeyHex));
        Assert.Empty(ledger.InNamespace("tenant-c"));

        // Forgetting all of tenant A (Task 16's evict --mode namespace) leaves B's and the default namespace's rows, and only
        // A's entries become orphans.
        foreach (LedgerRow r in ledger.InNamespace(TenantA).ToList()) Assert.True(ledger.Remove(new ChunkKey(r.KeyHex)));
        Assert.False(ledger.Remove(a1)); // already gone
        SaveLocked(ledger);
        Assert.Equal(new[] { a1, a2 }.OrderBy(k => k.Hex, StringComparer.Ordinal), ChunkIndex.StoreWideOrphans(store, _root));
        ServerLedger back = ServerLedger.Load(_root);
        Assert.Equal([b1.Hex], back.InNamespace(TenantB).Select(r => r.KeyHex));
        Assert.Equal([d1.Hex], back.InNamespace(ChunkNamespace.Default).Select(r => r.KeyHex));
    }

    // ------------------------------------------------------------------------------------------------ ownership-aware orphans

    [Fact]
    public void StoreWideOrphans_KeepsLedgerKeys()
    {
        var store = new FileChunkStore(_root);
        var index = new ChunkIndex(ids => Key(ChunkNamespace.Default, ids.ToArray()));
        index.Update([new SourceFile("A.txt", "0123456789\n")], new FixedSizeChunker(new CharTokenizer(), chunkTokens: 4));
        index.Save(ChunkIndex.DefaultPath(_root, "fixed-4"));
        foreach (ChunkKey k in index.ReferencedKeys()) Put(store, k);
        ChunkKey serverA = Key(TenantA, 900), serverB = Key(TenantB, 900), stray = Key(ChunkNamespace.Default, 901);
        Put(store, serverA);
        Put(store, serverB);
        Put(store, stray);

        // Without a ledger the server's entries are orphans (the hazard L2 removes).
        Assert.Equal(new[] { serverA, serverB, stray }.OrderBy(k => k.Hex, StringComparer.Ordinal), ChunkIndex.StoreWideOrphans(store, _root));

        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(serverA, TenantA, null, 1, T0);
        ledger.Record(serverB, TenantB, "src/B.cs", 1, T0);
        SaveLocked(ledger);
        Assert.Equal([stray], ChunkIndex.StoreWideOrphans(store, _root));
        IReadOnlySet<ChunkKey> referenced = ChunkIndex.ReferencedKeysUnder(_root);
        Assert.Contains(serverA, referenced);
        Assert.Contains(serverB, referenced);
        Assert.DoesNotContain(stray, referenced);
        // Skipping the caller's own index file never skips the ledger.
        Assert.Contains(serverA, ChunkIndex.ReferencedKeysUnder(_root, ChunkIndex.DefaultPath(_root, "fixed-4")));
        Assert.Contains(serverA, ChunkIndex.ReferencedKeysUnder(_root, LedgerFile));

        // A ledger key leaves protection only when the ledger itself drops it (forget / budget eviction).
        ledger.Remove(serverB);
        SaveLocked(ledger);
        Assert.Equal(new[] { serverB, stray }.OrderBy(k => k.Hex, StringComparer.Ordinal), ChunkIndex.StoreWideOrphans(store, _root));
    }

    public static TheoryData<string, string> UnreadableLedgers => new()
    {
        { "not-json", "{ not json" },
        { "json-null", "null" },
        { "future-version", """{"formatVersion": 2, "rows": []}""" },
        { "no-version", """{"rows": []}""" },
        { "no-rows", """{"formatVersion": 1}""" },
        { "null-row", """{"formatVersion": 1, "rows": [null]}""" },
        { "bad-key", """{"formatVersion": 1, "rows": [{"key": "xyz", "namespace": "", "bytes": 1, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "bad-namespace", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namespace": "Tenant/A", "bytes": 1, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "no-namespace", """{"formatVersion": 1, "rows": [{"key": "@KEY", "bytes": 1, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "negative-bytes", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namespace": "", "bytes": -1, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "no-last-used", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namespace": "", "bytes": 1, "firstSeen": "2026-11-02T10:00:00+00:00"}]}""" },
        { "rooted-source", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namespace": "", "sourcePath": "/etc/x", "bytes": 1, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "duplicate-key", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namespace": "", "bytes": 1, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}, {"key": "@KEY", "namespace": "", "bytes": 2, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "duplicate-property", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namespace": "", "bytes": 1, "bytes": 2, "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T10:00:00+00:00"}]}""" },
        { "duplicate-version", """{"formatVersion": 1, "formatVersion": 1, "rows": []}""" },
        { "truncated", """{"formatVersion": 1, "rows": [{"key": "@KEY", "namesp""" },
    };

    [Theory]
    [MemberData(nameof(UnreadableLedgers))]
    public void StoreWideOrphans_UnreadableLedger_ThrowsAndDeletesNothing(string name, string content)
    {
        var store = new FileChunkStore(_root);
        ChunkKey server = Key(TenantA, 900), stray = Key(ChunkNamespace.Default, 901);
        Put(store, server);
        Put(store, stray);
        File.WriteAllText(LedgerFile, content.Replace("@KEY", server.Hex, StringComparison.Ordinal));

        var ex = Assert.Throws<LedgerUnreadableException>(() => ChunkIndex.StoreWideOrphans(store, _root));
        Assert.IsAssignableFrom<IOException>(ex); // InvalidDataException is sealed; callers already treat IOException as "cannot decide"
        Assert.Equal(LedgerFile, ex.Path);
        Assert.Contains(LedgerFile, ex.Message, StringComparison.Ordinal); // the error names the file
        Assert.DoesNotContain("Tenant/A", ex.Message, StringComparison.Ordinal); // namespace values never reach messages
        Assert.Throws<LedgerUnreadableException>(() => ChunkIndex.ReferencedKeysUnder(_root));
        Assert.Throws<LedgerUnreadableException>(() => ServerLedger.Load(_root));

        // The caller that evicts what StoreWideOrphans returns deletes nothing, and the file is left for repair.
        Assert.Equal(new[] { server, stray }.OrderBy(k => k.Hex).ToArray(), store.Keys().OrderBy(k => k.Hex).ToArray());
        Assert.Equal(content.Replace("@KEY", server.Hex, StringComparison.Ordinal), File.ReadAllText(LedgerFile));
        _ = name;
    }

    [Fact]
    public void Ledger_HeldExclusivelyByAnotherProcess_IsUnreadable_NotEmpty()
    {
        var store = new FileChunkStore(_root);
        ChunkKey server = Key(TenantA, 900);
        Put(store, server);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(server, TenantA, null, 1, T0);
        SaveLocked(ledger);
        using (new FileStream(LedgerFile, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            if (OperatingSystem.IsWindows()) // POSIX has no mandatory sharing modes
                Assert.Throws<LedgerUnreadableException>(() => ChunkIndex.StoreWideOrphans(store, _root));
        }
        Assert.Empty(ChunkIndex.StoreWideOrphans(store, _root));
    }

    [Fact]
    public void Ledger_UnknownFieldsAreIgnored_WithinTheVersion()
    {
        var store = new FileChunkStore(_root);
        ChunkKey server = Key(TenantA, 900);
        Put(store, server);
        File.WriteAllText(LedgerFile, $$"""
            {"formatVersion": 1, "writer": "dotllm-server 0.9", "rows": [
              {"key": "{{server.Hex}}", "namespace": "{{TenantA}}", "sourcePath": null, "bytes": 7,
               "firstSeen": "2026-11-02T10:00:00+00:00", "lastUsed": "2026-11-02T11:00:00+00:00", "hits": 3}]}
            """);
        ServerLedger ledger = ServerLedger.Load(_root);
        Assert.Equal(new LedgerRow(server.Hex, TenantA, null, 7, T0, T0.AddHours(1)), Assert.Single(ledger.Rows));
        Assert.Empty(ChunkIndex.StoreWideOrphans(store, _root));
    }

    // ------------------------------------------------------------------------------------------------ atomic save, concurrency

    [Fact]
    public void Ledger_SaveIsAtomic()
    {
        var store = new FileChunkStore(_root);
        ChunkKey k1 = Key(TenantA, 1), k2 = Key(TenantA, 2);
        Put(store, k1);
        Put(store, k2);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(k1, TenantA, null, 1, T0);
        SaveLocked(ledger);
        byte[] good = File.ReadAllBytes(LedgerFile);

        // A crashed writer's truncated temp file is never read and never replaces the ledger.
        string stale = LedgerFile + "." + Guid.NewGuid().ToString("N") + ".tmp";
        string fresh = LedgerFile + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(stale, "{\"formatVersion\": 1, \"rows\": [");
        File.SetLastWriteTimeUtc(stale, DateTime.UtcNow.AddHours(-2));
        File.WriteAllText(fresh, "{\"formatVersion\": 1, \"ro");
        Assert.Equal([k1.Hex], ServerLedger.Load(_root).Rows.Select(r => r.KeyHex));
        Assert.Equal(good, File.ReadAllBytes(LedgerFile));

        // A save that dies after writing the temp file and before the rename leaves the old ledger intact and no temp file.
        ledger.Record(k2, TenantA, null, 2, T0);
        ledger.BeforeRename = temp =>
        {
            Assert.True(File.Exists(temp));
            Assert.Equal(2, JsonNode.Parse(File.ReadAllText(temp))!["rows"]!.AsArray().Count); // complete and flushed
            throw new IOException("simulated crash before rename");
        };
        Assert.Throws<IOException>(() => SaveLocked(ledger));
        Assert.Equal(good, File.ReadAllBytes(LedgerFile));
        ledger.BeforeRename = null;
        string[] temps = Directory.GetFiles(_root, Path.GetFileName(LedgerFile) + ".*.tmp");
        Assert.Equal([fresh], temps); // the failed save's temp is gone; the fresh one may belong to a live writer

        // The next save succeeds and sweeps the stale temp file.
        SaveLocked(ledger);
        Assert.False(File.Exists(stale));
        Assert.Equal(new[] { k1.Hex, k2.Hex }.Order(StringComparer.Ordinal), ServerLedger.Load(_root).Rows.Select(r => r.KeyHex).Order(StringComparer.Ordinal));
        // Temp files and the ledger are not index files and not store entries.
        Assert.Equal(2, store.Keys().Count());
    }

    [Fact]
    public async Task Ledger_ConcurrentSaves_EvictionNeverSeesAPartialLedger()
    {
        var store = new FileChunkStore(_root);
        ChunkKey pinned = Key(TenantA, 1), stray = Key(ChunkNamespace.Default, 2);
        Put(store, pinned);
        Put(store, stray);
        var extra = Enumerable.Range(10, 40).Select(i => Key(TenantB, i)).ToArray();
        foreach (ChunkKey k in extra) Put(store, k, seed: 7);
        ServerLedger writer = ServerLedger.Load(_root);
        writer.Record(pinned, TenantA, null, 1, T0);
        SaveLocked(writer);

        using var stop = new CancellationTokenSource();
        Task saving = Task.Run(() =>
        {
            int i = 0;
            while (!stop.IsCancellationRequested)
            {
                // Alternate between a small and a large ledger; the pinned key is in both.
                if (i++ % 2 == 0) foreach (ChunkKey k in extra) writer.Record(k, TenantB, "src/x.cs", 1, T0);
                else foreach (ChunkKey k in extra) writer.Remove(k);
                SaveLocked(writer);
            }
        });
        try
        {
            for (int n = 0; n < 150; n++)
            {
                IReadOnlyList<ChunkKey> orphans = ChunkIndex.StoreWideOrphans(store, _root);
                Assert.DoesNotContain(pinned, orphans);
                Assert.Contains(stray, orphans);
            }
        }
        finally
        {
            stop.Cancel();
            await saving;
        }
    }

    [Fact]
    public void Ledger_PathIsADirectory_IsUnreadable_NotAbsent()
    {
        var store = new FileChunkStore(_root);
        ChunkKey stray = Key(ChunkNamespace.Default, 901);
        Put(store, stray);
        Directory.CreateDirectory(LedgerFile);
        Assert.Throws<LedgerUnreadableException>(() => ServerLedger.Load(_root));
        var ex = Assert.Throws<LedgerUnreadableException>(() => ChunkIndex.StoreWideOrphans(store, _root));
        Assert.Equal(LedgerFile, ex.Path);
        Assert.True(store.Contains(stray));
    }

    [Fact]
    public void Ledger_UnprobeablePath_IsUnreadable_OnlyNotFoundIsAbsent()
    {
        var store = new FileChunkStore(_root);
        ChunkKey server = Key(TenantA, 900);
        Put(store, server);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(server, TenantA, null, 1, T0);
        SaveLocked(ledger);
        string ledgerFile = LedgerFile;
        try
        {
            foreach (Exception fault in new Exception[] { new UnauthorizedAccessException("acl"), new IOException("device") })
            {
                ServerLedger.ProbeHook = p => p == ledgerFile ? throw fault : File.GetAttributes(p);
                Assert.Throws<LedgerUnreadableException>(() => ServerLedger.Load(_root));
                Assert.Throws<LedgerUnreadableException>(() => ChunkIndex.StoreWideOrphans(store, _root));
            }
            foreach (Exception absent in new Exception[] { new FileNotFoundException("gone"), new DirectoryNotFoundException("gone") })
            {
                ServerLedger.ProbeHook = p => p == ledgerFile ? throw absent : File.GetAttributes(p);
                Assert.Empty(ServerLedger.Load(_root).Rows);
                Assert.Equal([server], ChunkIndex.StoreWideOrphans(store, _root));
            }
        }
        finally
        {
            ServerLedger.ProbeHook = null;
        }
    }

    [Fact]
    public void Ledger_EntryStateUnknown_RowKeptAndReported_OnlyNotFoundDrops()
    {
        var store = new FileChunkStore(_root);
        ChunkKey unknown = Key(TenantA, 1), gone = Key(TenantA, 2), present = Key(TenantA, 3), dirGone = Key(TenantA, 4);
        foreach (ChunkKey k in new[] { unknown, gone, present, dirGone }) Put(store, k);
        ServerLedger ledger = ServerLedger.Load(_root);
        foreach (ChunkKey k in new[] { unknown, gone, present, dirGone }) ledger.Record(k, TenantA, null, 1, T0);
        SaveLocked(ledger);
        string Entry(ChunkKey k) => store.PathOf(k);
        try
        {
            ServerLedger.ProbeHook = p =>
                p == Entry(unknown) ? throw new UnauthorizedAccessException("shard acl")
                : p == Entry(gone) ? throw new FileNotFoundException("gone")
                : p == Entry(dirGone) ? throw new DirectoryNotFoundException("shard gone")
                : File.GetAttributes(p);
            ServerLedger back = ServerLedger.Load(_root);
            Assert.Equal(2, back.DroppedOnLoad);
            Assert.Equal(1, back.UndeterminedOnLoad);
            Assert.Equal(new[] { unknown.Hex, present.Hex }.Order(StringComparer.Ordinal), back.Rows.Select(r => r.KeyHex).Order(StringComparer.Ordinal));

            ServerLedger.ProbeHook = p => p == Entry(unknown) ? throw new IOException("sharing") : File.GetAttributes(p);
            Assert.Equal(1, ServerLedger.Load(_root).UndeterminedOnLoad);
            Assert.Contains(unknown.Hex, ServerLedger.Load(_root).Rows.Select(r => r.KeyHex));
        }
        finally
        {
            ServerLedger.ProbeHook = null;
        }
        // Without faults every entry exists: nothing dropped, nothing undetermined.
        ServerLedger clean = ServerLedger.Load(_root);
        Assert.Equal((0, 0, 4), (clean.DroppedOnLoad, clean.UndeterminedOnLoad, clean.Rows.Count));
    }

    [Fact]
    public void Ledger_EntryPathIsADirectory_RowKeptAndReportedUndetermined()
    {
        var store = new FileChunkStore(_root);
        ChunkKey key = Key(TenantA, 5);
        Put(store, key);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(key, TenantA, null, 1, T0);
        SaveLocked(ledger);
        string entry = store.PathOf(key);
        File.Delete(entry);
        Directory.CreateDirectory(entry); // a directory where the entry file belongs: not "not found", so the row is not dropped

        ServerLedger back = ServerLedger.Load(_root);

        Assert.Equal((0, 1), (back.DroppedOnLoad, back.UndeterminedOnLoad));
        Assert.Contains(key.Hex, back.Rows.Select(r => r.KeyHex));
    }

    [Fact]
    public void Ledger_SaveFailure_IsNotMaskedByTempCleanup()
    {
        new FileChunkStore(_root);
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(Key(TenantA, 1), TenantA, null, 1, T0);
        string? held = null;
        FileStream? lockTemp = null;
        ledger.BeforeRename = temp =>
        {
            held = temp;
            // Hold the temp file open without delete sharing, so the cleanup's delete fails too (on Windows).
            lockTemp = new FileStream(temp, FileMode.Open, FileAccess.Read, FileShare.Read);
            throw new IOException("original failure");
        };
        try
        {
            var ex = Assert.Throws<IOException>(() => SaveLocked(ledger));
            Assert.Equal("original failure", ex.Message);
        }
        finally
        {
            lockTemp?.Dispose();
            if (held is not null && File.Exists(held)) File.Delete(held);
        }
    }

    [Fact]
    public void Ledger_IsNotAnIndexFile_AndDefaultPathIsInTheStoreRoot()
    {
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), "ledger-server.json"), ServerLedger.DefaultPath(_root));
        Assert.Throws<ArgumentException>(() => ServerLedger.DefaultPath(""));
        // A missing store root loads an empty ledger and creates nothing.
        string missing = Path.Combine(_root, "nope");
        Assert.Empty(ServerLedger.Load(missing).Rows);
        Assert.False(Directory.Exists(missing));
    }
}
