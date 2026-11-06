using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Index naming by chunker, model and namespace and the store registry (Phase I Task 4b, integration spec L9):
/// <c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;ns-or-default&gt;.json</c> and its <c>.lock</c>, <c>chunkcache.json</c>, and
/// the migration of pre-L9 index names. Oracles: file names computed here from SHA-256 independently of the library, the
/// files on disk, and <see cref="ChunkIndex.StoreWideOrphans"/> / <see cref="ChunkIndex.EvictStoreWideOrphans"/>.
/// </summary>
public sealed class StoreRegistryTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private static readonly TimeSpan Wait = TimeSpan.FromSeconds(10);
    private const string ModelA = "gguf-sha256:aaaa1111";
    private const string ModelB = "gguf-sha256:bbbb2222";
    private const string TenantA = "k-0123456789abcdef";
    private const string TenantB = "tenant.b";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-registry-" + Guid.NewGuid().ToString("N"));

    public StoreRegistryTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    /// <summary>The model's key function in a namespace (the model id stands in for the fingerprint).</summary>
    private static ChunkKeyFunction KeyOf(string modelId, string ns) =>
        ids => ChunkKey.Create(modelId, "tok", Rope, "cpu", KvDtype.F32, ids, @namespace: ns);

    /// <summary>Independent oracle for the model tag: first 12 hex of SHA-256 of the UTF-8 model id.</summary>
    private static string Tag(string modelId) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(modelId)))[..12];

    /// <summary>The producer identity of a model in a namespace (64 hex): the key of no tokens.</summary>
    private static string P(string modelId, string ns) => KeyOf(modelId, ns)([]).Hex;

    /// <summary>Its 8-hex file name tag.</summary>
    private static string P8(string modelId, string ns) => P(modelId, ns)[..8];

    private static FixedSizeChunker Chunker => new(new CharTokenizer(), chunkTokens: 8);

    private static ChunkKvEntry Entry(ChunkKey key) => new("fp", [1, 2], [[1f, -2f]], [[0.5f, 4f]]);

    /// <summary>Builds and saves an index of <paramref name="files"/> at its default path, and stores every key it references.</summary>
    private ChunkIndex Build(FileChunkStore store, string modelId, string ns, params SourceFile[] files)
    {
        var index = new ChunkIndex(KeyOf(modelId, ns));
        index.Update(files, Chunker);
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, modelId, P(modelId, ns), ns, Wait))
            index.Save(ChunkIndex.DefaultPath(_root, Chunker.Name, modelId, P(modelId, ns), ns));
        foreach (ChunkKey k in index.ReferencedKeys()) store.Put(k, Entry(k));
        return index;
    }

    // ------------------------------------------------------------------------------------------------ names

    [Fact]
    public void TwoModelsSameChunker_TwoIndexFiles()
    {
        var store = new FileChunkStore(_root);
        string pathA = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default);
        string pathB = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelB, P(ModelB, ChunkNamespace.Default), ChunkNamespace.Default);
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), $"index-fixed-8.{Tag(ModelA)}.{P8(ModelA, ChunkNamespace.Default)}.default.json"), pathA);
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), $"index-fixed-8.{Tag(ModelB)}.{P8(ModelB, ChunkNamespace.Default)}.default.json"), pathB);
        Assert.Equal(Tag(ModelA), ChunkIndex.ModelTag(ModelA));

        ChunkIndex a = Build(store, ModelA, ChunkNamespace.Default, new SourceFile("A.txt", "same text in both\n"));
        ChunkIndex b = Build(store, ModelB, ChunkNamespace.Default, new SourceFile("A.txt", "same text in both\n"));
        Assert.True(File.Exists(pathA));
        Assert.True(File.Exists(pathB));
        // Each loads its own model's index (neither clobbered the other), and both protect their keys.
        Assert.Null(ChunkIndex.Load(pathA, KeyOf(ModelA, ChunkNamespace.Default)).LoadNote);
        Assert.Equal(["A.txt"], ChunkIndex.Load(pathB, KeyOf(ModelB, ChunkNamespace.Default)).Paths);
        Assert.Empty(ChunkIndex.StoreWideOrphans(store, _root));
        Assert.Empty(a.ReferencedKeys().Intersect(b.ReferencedKeys()));
        // Locks follow the names.
        Assert.Equal(Path.ChangeExtension(pathA, ".lock"), ChunkIndex.IndexLockPath(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default));
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default, Wait))
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelB, P(ModelB, ChunkNamespace.Default), ChunkNamespace.Default, TimeSpan.Zero)) { }
    }

    [Fact]
    public void TwoNamespaces_TwoIndexFilesAndLocks()
    {
        string def = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default);
        string a = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA);
        string b = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, TenantB), TenantB);
        Assert.Equal($"index-fixed-8.{Tag(ModelA)}.{P8(ModelA, TenantA)}.{TenantA}.json", Path.GetFileName(a));
        Assert.Equal($"index-fixed-8.{Tag(ModelA)}.{P8(ModelA, TenantB)}.tenant.b.json", Path.GetFileName(b));
        Assert.Equal(3, new[] { def, a, b }.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        // Distinct locks: holding A's does not block B's or the default's; the same namespace serialises.
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA, Wait))
        {
            using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, TenantB), TenantB, TimeSpan.Zero)) { }
            using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default, TimeSpan.Zero)) { }
            Assert.Throws<ChunkStoreLockTimeoutException>(() => ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA, TimeSpan.FromMilliseconds(50)).Dispose());
        }
        Assert.Equal(3, new[] { def, a, b }.Select(p => Path.ChangeExtension(p, ".lock")).Distinct(StringComparer.OrdinalIgnoreCase).Count());

        // Both namespaces index the same text under one chunker and model: two files, neither loads empty.
        var store = new FileChunkStore(_root);
        Build(store, ModelA, TenantA, new SourceFile("A.txt", "shared text\n"));
        Build(store, ModelA, TenantB, new SourceFile("A.txt", "shared text\n"));
        Assert.Null(ChunkIndex.Load(a, KeyOf(ModelA, TenantA)).LoadNote);
        Assert.Null(ChunkIndex.Load(b, KeyOf(ModelA, TenantB)).LoadNote);
        Assert.Equal(["A.txt"], ChunkIndex.Load(a, KeyOf(ModelA, TenantA)).Paths);
    }

    [Fact]
    public void Names_ValidatedBeforeTheyReachAFileName_AndParseBack()
    {
        foreach (string bad in new[] { "Fixed-8", "a.b", "a/b", "a\\b", "", "x:y", new string('a', 65) })
            Assert.ThrowsAny<ArgumentException>(() => ChunkIndex.DefaultPath(_root, bad, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default));
        foreach (string ns in new[] { "Tenant", "a/b", "..\\x", "default", "é", new string('a', 65) })
            Assert.ThrowsAny<ArgumentException>(() => ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ns));
        Assert.ThrowsAny<ArgumentException>(() => ChunkIndex.DefaultPath(_root, Chunker.Name, "", P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default));
        foreach (string badProducer in new[] { "", "0123", new string('A', 64), new string('a', 63) })
            Assert.ThrowsAny<ArgumentException>(() => ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, badProducer, ChunkNamespace.Default));
        Assert.False(ChunkNamespace.IsValid(ChunkNamespace.ReservedDefaultName)); // "default" is the Default's file tag only

        foreach (string ns in new[] { ChunkNamespace.Default, TenantA, TenantB, ".", "..", "a..b", "x-" })
        {
            string name = Path.GetFileName(ChunkIndex.DefaultPath(_root, "symbol-512", ModelB, P(ModelB, ns), ns));
            Assert.True(ChunkIndex.TryParseIndexFileName(name, out IndexFileName parts), name);
            Assert.Equal(new IndexFileName("symbol-512", Tag(ModelB), P8(ModelB, ns), ns), parts);
        }
        foreach (string notCurrent in new[] { "index-symbol-512.json", $"index-symbol-512.{Tag(ModelA)}.json", "index-a.0123456789AB.default.json",
                     "index-a.0123456789ab.json", "index-a.0123456789ab..json", "index-a.0123456789ab.Bad.json", "chunkcache.json", "ledger-server.json",
                     "index-a.0123456789ab.default.lock", "index-a.0123456789ab.default.json" /* Task 4b name, no p8 */,
                     "index-a.0123456789AB.0123abcd.default.json" /* upper-case model tag, valid producer tag */,
                     "index-a.0123456789ab.0123ABCD.default.json","index-a.0123456789ab.0123abc.default.json", "index-a.0123456789ab.0123abcd..json" })
            Assert.False(ChunkIndex.TryParseIndexFileName(notCurrent, out _), notCurrent);
    }

    // ------------------------------------------------------------------------------------------------ orphans

    [Fact]
    public void StoreWideOrphans_FromOneNamespace_NeverDeletesOtherNamespacesEntries()
    {
        var store = new FileChunkStore(_root);
        SourceFile one = new("One.txt", "first file text\n"), two = new("Two.txt", "second file text\n");
        ChunkIndex a = Build(store, ModelA, TenantA, one, two);
        ChunkIndex b = Build(store, ModelA, TenantB, one, two);
        var bKeys = b.ReferencedKeys().ToHashSet();
        var aBefore = a.ReferencedKeys().ToHashSet();

        // Tenant A drops Two.txt and saves its own index: only A's Two.txt keys become orphans.
        a.Update([one], Chunker);
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA, Wait))
            a.Save(ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA));
        var expected = aBefore.Except(a.ReferencedKeys()).OrderBy(k => k.Hex, StringComparer.Ordinal).ToList();
        Assert.NotEmpty(expected);
        IReadOnlyList<ChunkKey> evicted = ChunkIndex.EvictStoreWideOrphans(store, _root, Wait);
        Assert.Equal(expected, evicted);
        Assert.All(bKeys, k => Assert.True(store.Contains(k), "tenant B lost an entry"));
        Assert.All(a.ReferencedKeys(), k => Assert.True(store.Contains(k)));
        Assert.Empty(ChunkIndex.StoreWideOrphans(store, _root));
    }

    [Fact]
    public void Orphans_CountKeysFromAllModelIndexes_RegistryIsNotAuthoritative()
    {
        var store = new FileChunkStore(_root);
        ChunkIndex a = Build(store, ModelA, ChunkNamespace.Default, new SourceFile("A.txt", "model a text\n"));
        ChunkIndex b = Build(store, ModelB, TenantB, new SourceFile("B.txt", "model b text\n"));
        // A legacy-named index (another producer) is an index too.
        var legacy = new ChunkIndex(KeyOf("gguf-sha256:old", ChunkNamespace.Default));
        legacy.Update([new SourceFile("L.txt", "legacy text\n")], Chunker);
        legacy.Save(ChunkIndex.DefaultPath(_root, Chunker.Name));
        foreach (ChunkKey k in legacy.ReferencedKeys()) store.Put(k, Entry(k));
        ChunkKey stray = KeyOf(ModelA, ChunkNamespace.Default)([99, 98]);
        store.Put(stray, Entry(stray));

        // The registry lists only model A's index; model B's file and the legacy file are not listed.
        StoreRegistry.Update(_root, Wait, r =>
        {
            foreach (RegistryEntry e in r.Entries.ToList()) r.Remove(e.File);
            r.Upsert(StoreRegistry.EntryFor(Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default));
        });
        JsonNode doc = JsonNode.Parse(File.ReadAllText(StoreRegistry.DefaultPath(_root)))!;
        Assert.Single(doc["indexes"]!.AsArray());
        Assert.Equal([stray], ChunkIndex.StoreWideOrphans(store, _root));

        // A corrupt registry changes nothing either.
        File.WriteAllText(StoreRegistry.DefaultPath(_root), "{ corrupt");
        Assert.Equal([stray], ChunkIndex.EvictStoreWideOrphans(store, _root, Wait));
        Assert.All(a.ReferencedKeys().Concat(b.ReferencedKeys()).Concat(legacy.ReferencedKeys()), k => Assert.True(store.Contains(k)));
    }

    // ------------------------------------------------------------------------------------------------ registry

    [Fact]
    public void Registry_ListsEveryIndex_AndRebuildsWhenMissing()
    {
        var store = new FileChunkStore(_root);
        Build(store, ModelA, ChunkNamespace.Default, new SourceFile("A.txt", "a\n"));
        Build(store, ModelA, TenantA, new SourceFile("A.txt", "a\n"));
        Build(store, ModelB, TenantB, new SourceFile("B.txt", "b\n"));
        DateTimeOffset when = new(2026, 11, 3, 9, 0, 0, TimeSpan.Zero);
        StoreRegistry.Update(_root, Wait, r =>
        {
            r.Upsert(StoreRegistry.EntryFor(Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default) with
            {
                TokenizerId = "tok", Backend = "cpu", Producer = KeyOf(ModelA, ChunkNamespace.Default)([]).Hex, Updated = when,
                Tool = new Dictionary<string, string?> { ["root"] = "C:/src", ["source_kind"] = "directory" },
            });
        });

        // The file: snake_case rows with the plan's fields, one per index (two discovered from disk), sorted by file name.
        JsonNode doc = JsonNode.Parse(File.ReadAllText(StoreRegistry.DefaultPath(_root)))!;
        Assert.Equal(2, doc["format_version"]!.GetValue<int>());
        JsonArray rows = doc["indexes"]!.AsArray();
        Assert.Equal(3, rows.Count);
        JsonNode first = rows.Single(r => r!["namespace"]!.GetValue<string>() == "" && r["model12"]!.GetValue<string>() == Tag(ModelA))!;
        Assert.Equal($"index-fixed-8.{Tag(ModelA)}.{P8(ModelA, ChunkNamespace.Default)}.default.json", first["file"]!.GetValue<string>());
        Assert.Equal("fixed-8", first["chunker"]!.GetValue<string>());
        Assert.Equal(ModelA, first["model_id"]!.GetValue<string>());
        Assert.Equal("tok", first["tokenizer_id"]!.GetValue<string>());
        Assert.Equal("cpu", first["backend"]!.GetValue<string>());
        Assert.Equal(when, first["updated"]!.GetValue<DateTimeOffset>());
        Assert.Equal("C:/src", first["tool"]!["root"]!.GetValue<string>());
        string[] files = rows.Select(r => r!["file"]!.GetValue<string>()).ToArray();
        Assert.Equal(files.Order(StringComparer.Ordinal), files);

        // Find by chunker, model and namespace.
        StoreRegistry loaded = StoreRegistry.Load(_root);
        Assert.Null(loaded.LoadNote);
        Assert.False(loaded.Rebuilt);
        Assert.Equal(ChunkIndex.DefaultPath(_root, Chunker.Name, ModelB, P(ModelB, TenantB), TenantB), loaded.Find(Chunker.Name, ModelB, P(ModelB, TenantB), TenantB));
        Assert.Null(loaded.Find(Chunker.Name, ModelB, P(ModelB, ChunkNamespace.Default), ChunkNamespace.Default));
        Assert.Null(loaded.Find("symbol-512", ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default));

        // Missing registry: rebuilt from the index file names (chunker, model tag, namespace).
        File.Delete(StoreRegistry.DefaultPath(_root));
        StoreRegistry rebuilt = StoreRegistry.Load(_root);
        Assert.True(rebuilt.Rebuilt);
        Assert.Equal(files, rebuilt.Entries.Select(e => e.File).ToArray());
        RegistryEntry tb = rebuilt.Entries.Single(e => e.Namespace == TenantB);
        Assert.Equal(("fixed-8", Tag(ModelB)), (tb.Chunker, tb.ModelTag));
        Assert.Null(tb.ModelId); // not recoverable from the name
        Assert.Equal(ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA), rebuilt.Find(Chunker.Name, ModelA, P(ModelA, TenantA), TenantA));
        Assert.False(File.Exists(StoreRegistry.DefaultPath(_root))); // Load writes nothing

        // A row whose index file is gone is dropped on load (the registry on disk lists it first).
        StoreRegistry.Update(_root, Wait, _ => { });
        Assert.Contains(TenantA, File.ReadAllText(StoreRegistry.DefaultPath(_root)), StringComparison.Ordinal);
        File.Delete(ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, TenantA), TenantA));
        Assert.Contains("dropped 1 row", StoreRegistry.Load(_root).LoadNote, StringComparison.Ordinal);
        StoreRegistry.Update(_root, Wait, _ => { });
        Assert.Equal(2, StoreRegistry.Load(_root).Entries.Count);
        Assert.DoesNotContain(TenantA, File.ReadAllText(StoreRegistry.DefaultPath(_root)), StringComparison.Ordinal);
    }

    [Fact]
    public void Registry_SaveIsAtomic_AndOnlyUnderTheStoreLock()
    {
        var store = new FileChunkStore(_root);
        Build(store, ModelA, ChunkNamespace.Default, new SourceFile("A.txt", "a\n"));
        StoreRegistry.Update(_root, Wait, _ => { });
        string path = StoreRegistry.DefaultPath(_root);
        byte[] before = File.ReadAllBytes(path);

        // Not under this store's held lock: refused (no handle, a released handle, another store's handle).
        StoreRegistry r = StoreRegistry.Load(_root);
        Assert.Throws<ArgumentNullException>(() => r.Save(null!));
        StoreLockHandle released = ChunkIndex.AcquireStoreLock(_root, Wait);
        released.Dispose();
        Assert.Throws<InvalidOperationException>(() => r.Save(released));
        string other = Path.Combine(_root, "other-store");
        using (StoreLockHandle otherLock = ChunkIndex.AcquireStoreLock(other, Wait))
            Assert.Throws<InvalidOperationException>(() => r.Save(otherLock));
        Assert.Equal(before, File.ReadAllBytes(path));

        // A crash just before the rename: the old file is intact, the complete temp is removed.
        Build(store, ModelB, ChunkNamespace.Default, new SourceFile("B.txt", "b\n"));
        StoreRegistry crash = StoreRegistry.Load(_root);
        string? seenTemp = null;
        crash.BeforeRename = temp =>
        {
            seenTemp = temp;
            Assert.Equal(2, JsonNode.Parse(File.ReadAllText(temp))!["indexes"]!.AsArray().Count); // the temp is complete
            throw new IOException("simulated crash before the rename");
        };
        using (StoreLockHandle held = ChunkIndex.AcquireStoreLock(_root, Wait))
            Assert.Throws<IOException>(() => crash.Save(held));
        Assert.Equal(before, File.ReadAllBytes(path));
        Assert.False(File.Exists(seenTemp));

        // A truncated temp left by a crashed writer is never read.
        File.WriteAllText(path + ".0123456789abcdef0123456789abcdef.tmp", "{\"format_version\": 2, \"inde");
        StoreRegistry afterCrash = StoreRegistry.Load(_root);
        Assert.False(afterCrash.Rebuilt);
        Assert.Equal(2, afterCrash.Entries.Count); // the saved row plus model B's index discovered on disk
    }

    [Fact]
    public void Registry_Corrupt_IsRebuilt_KeptAsBad_AndNewerVersionIsNeverOverwritten()
    {
        var store = new FileChunkStore(_root);
        Build(store, ModelA, ChunkNamespace.Default, new SourceFile("A.txt", "a\n"));
        string path = StoreRegistry.DefaultPath(_root);
        File.WriteAllText(path, "{ not json");
        StoreRegistry corrupt = StoreRegistry.Load(_root);
        Assert.True(corrupt.Rebuilt);
        Assert.Contains("not valid JSON", corrupt.LoadNote, StringComparison.Ordinal);
        Assert.Single(corrupt.Entries);
        StoreRegistry.Update(_root, Wait, _ => { });
        string firstBad = Assert.Single(Directory.GetFiles(_root, StoreRegistry.FileName + StoreRegistry.BadCopyPrefix + "*"));
        Assert.Equal("{ not json", File.ReadAllText(firstBad));
        Assert.False(StoreRegistry.Load(_root).Rebuilt);
        // A second corruption keeps the first copy (review N4).
        File.WriteAllText(path, "{ corrupt again");
        StoreRegistry.Update(_root, Wait, _ => { });
        string[] bads = Directory.GetFiles(_root, StoreRegistry.FileName + StoreRegistry.BadCopyPrefix + "*");
        Assert.Equal(2, bads.Length);
        Assert.Equal(new[] { "{ corrupt again", "{ not json" }, bads.Select(File.ReadAllText).Order(StringComparer.Ordinal).ToArray());

        string newer = """{"format_version": 3, "indexes": [], "whatever": true}""";
        File.WriteAllText(path, newer);
        StoreRegistry future = StoreRegistry.Load(_root);
        Assert.True(future.Rebuilt);
        Assert.Single(future.Entries); // readable for lookups
        Assert.Throws<InvalidDataException>(() => StoreRegistry.Update(_root, Wait, _ => { }));
        Assert.Equal(newer, File.ReadAllText(path));

        // Rows naming files outside the store are refused on load (never reach a delete) and reported.
        File.WriteAllText(path, """{"format_version": 2, "indexes": [{"file": "..\\victim.json", "chunker": "fixed-8", "namespace": ""}, {"file": "index-x.json", "chunker": "x", "namespace": "", "future_field": 7}]}""");
        File.WriteAllText(Path.Combine(_root, "index-x.json"), "{}");
        StoreRegistry edited = StoreRegistry.Load(_root);
        Assert.Single(edited.InvalidRows);
        StoreRegistry.Update(_root, Wait, _ => { });
        Assert.Contains("\"future_field\": 7", File.ReadAllText(path), StringComparison.Ordinal); // unknown row fields kept
        Assert.DoesNotContain("victim", File.ReadAllText(path), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Registry_TransientReadError_IsRetried_PersistentOneBlocksOnlyThatSave()
    {
        // Review N3: a sharing violation while loading (AV scanner, indexer) must not make the registry "never overwrite" for
        // good: the load retries briefly, and a later load reads it again.
        var store = new FileChunkStore(_root);
        Build(store, ModelA, ChunkNamespace.Default, new SourceFile("A.txt", "a\n"));
        StoreRegistry.Update(_root, Wait, r => r.Upsert(r.Entries.Single() with { Backend = "cpu" }));
        string path = StoreRegistry.DefaultPath(_root);

        var holder = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
        var release = Task.Run(async () => { await Task.Delay(300); holder.Dispose(); });
        StoreRegistry transient = StoreRegistry.Load(_root);
        await release;
        Assert.False(transient.Rebuilt);
        Assert.Equal("cpu", Assert.Single(transient.Entries).Backend);

        TimeSpan saved = StoreRegistry.ReadRetryBudget;
        StoreRegistry.ReadRetryBudget = TimeSpan.FromMilliseconds(100);
        try
        {
            StoreRegistry blocked;
            using (new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None))
                blocked = StoreRegistry.Load(_root);
            Assert.True(blocked.Rebuilt);
            using (StoreLockHandle held = ChunkIndex.AcquireStoreLock(_root, Wait))
                Assert.Throws<InvalidDataException>(() => blocked.Save(held)); // that instance never overwrites what it could not read
        }
        finally
        {
            StoreRegistry.ReadRetryBudget = saved;
        }
        StoreRegistry.Update(_root, Wait, _ => { }); // the next writer reads it again and saves
        Assert.Equal("cpu", Assert.Single(StoreRegistry.Load(_root).Entries).Backend);
    }

    [Fact]
    public void Registry_RebuiltRow_RecordsTheIndexProducer()
    {
        // Review N2: a rebuilt row gets the producer the index file records (its name has only 8 hex of it).
        var store = new FileChunkStore(_root);
        Build(store, ModelA, TenantA, new SourceFile("A.txt", "a\n"));
        RegistryEntry row = Assert.Single(StoreRegistry.Load(_root).Entries);
        Assert.Equal(P(ModelA, TenantA), row.Producer);
        Assert.Equal(P8(ModelA, TenantA), row.ProducerTag);
    }

    [Fact]
    public void TwoProducersSameModelAndNamespace_TwoIndexFiles()
    {
        // Review ruling S3: a CPU and a Vulkan producer of one model and namespace keep separate indexes.
        ChunkKeyFunction cpu = KeyOf(ModelA, ChunkNamespace.Default);
        ChunkKeyFunction vulkan = ids => ChunkKey.Create(ModelA, "tok", Rope, "vulkan", KvDtype.F32, ids);
        string pc = cpu([]).Hex, pv = vulkan([]).Hex;
        Assert.NotEqual(pc[..8], pv[..8]);
        string fc = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, pc, ChunkNamespace.Default);
        string fv = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, pv, ChunkNamespace.Default);
        Assert.NotEqual(fc, fv);
        var a = new ChunkIndex(cpu);
        a.Update([new SourceFile("A.txt", "same\n")], Chunker);
        a.Save(fc);
        var b = new ChunkIndex(vulkan);
        b.Update([new SourceFile("A.txt", "same\n")], Chunker);
        b.Save(fv);
        Assert.Null(ChunkIndex.Load(fc, cpu).LoadNote);
        Assert.Null(ChunkIndex.Load(fv, vulkan).LoadNote);
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, pc, ChunkNamespace.Default, Wait))
        using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, pv, ChunkNamespace.Default, TimeSpan.Zero)) { }
        StoreRegistry reg = StoreRegistry.Load(_root);
        Assert.Equal(fc, reg.Find(Chunker.Name, ModelA, pc, ChunkNamespace.Default));
        Assert.Equal(fv, reg.Find(Chunker.Name, ModelA, pv, ChunkNamespace.Default));
    }

    [Fact]
    public void Registry_ConcurrentWriters_LoseNoRow()
    {
        var store = new FileChunkStore(_root);
        const int Writers = 6;
        for (int w = 0; w < Writers; w++) Build(store, $"gguf-sha256:model-{w}", ChunkNamespace.Default, new SourceFile("A.txt", $"text {w}\n"));
        File.Delete(StoreRegistry.DefaultPath(_root)); // (Build does not write it; make sure)
        Parallel.For(0, Writers, new ParallelOptions { MaxDegreeOfParallelism = Writers }, w =>
            StoreRegistry.Update(_root, TimeSpan.FromSeconds(30), r =>
            {
                Thread.Sleep(20); // widen the read-modify-write window
                r.Upsert(StoreRegistry.EntryFor(Chunker.Name, $"gguf-sha256:model-{w}", P($"gguf-sha256:model-{w}", ChunkNamespace.Default), ChunkNamespace.Default) with { Backend = "cpu", TokenizerId = $"tok-{w}" });
            }));
        StoreRegistry final = StoreRegistry.Load(_root);
        Assert.Equal(Writers, final.Entries.Count);
        Assert.All(final.Entries, e => Assert.Equal("cpu", e.Backend)); // every writer's own fields survived
    }

    // ------------------------------------------------------------------------------------------------ migration

    [Fact]
    public void OldIndexName_MigratedWhenProducerMatches()
    {
        var store = new FileChunkStore(_root);
        ChunkKeyFunction keyOf = KeyOf(ModelA, ChunkNamespace.Default);
        var old = new ChunkIndex(keyOf);
        old.Update([new SourceFile("A.txt", "legacy library name\n")], Chunker);
        string legacy = ChunkIndex.DefaultPath(_root, Chunker.Name); // index-fixed-8.json
        old.Save(legacy);
        byte[] bytes = File.ReadAllBytes(legacy);
        string target = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default);

        // Read-only callers read it in place.
        Assert.Equal(legacy, ChunkIndex.ResolveIndexPath(_root, Chunker.Name, ModelA, ChunkNamespace.Default, keyOf));
        // The key stays protected at every moment: checked inside the migration, after the rename.
        bool? protectedAfterRename = null;
        ChunkIndex.AfterMigrationRename = () => protectedAfterRename = old.ReferencedKeys().All(ChunkIndex.ReferencedKeysUnder(_root).Contains);
        IndexMigration m;
        try
        {
            using (ChunkIndex.AcquireIndexLock(_root, Chunker.Name, ModelA, P(ModelA, ChunkNamespace.Default), ChunkNamespace.Default, Wait))
                m = ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelA, ChunkNamespace.Default, keyOf, Wait);
        }
        finally
        {
            ChunkIndex.AfterMigrationRename = null;
        }
        Assert.True(protectedAfterRename);
        Assert.Equal(legacy, m.MigratedFrom);
        Assert.Equal(target, m.IndexPath);
        Assert.False(File.Exists(legacy));
        Assert.Equal(bytes, File.ReadAllBytes(target)); // renamed, not rewritten
        Assert.Equal(["A.txt"], ChunkIndex.Load(target, keyOf).Paths);
        Assert.NotNull(StoreRegistry.Load(_root).Get(Path.GetFileName(target)));
        // Read once: a second call does nothing.
        Assert.Null(ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelA, ChunkNamespace.Default, keyOf, Wait).MigratedFrom);

        // The Phase S CLI name (index-<chunker>.<producer12>.json) with a version-1 registry row: the row moves with the file
        // and keeps the CLI's fields (now under "tool").
        ChunkKeyFunction keyB = KeyOf(ModelB, ChunkNamespace.Default);
        string producerB = keyB([]).Hex;
        var cli = new ChunkIndex(keyB);
        cli.Update([new SourceFile("B.txt", "phase s cli name\n")], Chunker);
        string cliName = $"index-{Chunker.Name}.{producerB[..12]}.json";
        cli.Save(Path.Combine(_root, cliName));
        File.WriteAllText(StoreRegistry.DefaultPath(_root), $$"""
            {
              "format_version": 1,
              "indexes": [
                { "index_file": "{{cliName}}", "chunker": "fixed-8", "producer": "{{producerB}}", "model_fingerprint": "fp-b",
                  "model_id": "{{ModelB}}", "model_path": "C:/m/b.gguf", "source_kind": "directory", "source_path": "C:/src",
                  "root": "C:/src", "updated_utc": "2026-11-02T10:00:00Z" }
              ]
            }
            """);
        IndexMigration mb = ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelB, ChunkNamespace.Default, keyB, Wait);
        string targetB = ChunkIndex.DefaultPath(_root, Chunker.Name, ModelB, P(ModelB, ChunkNamespace.Default), ChunkNamespace.Default);
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), cliName), mb.MigratedFrom);
        RegistryEntry row = StoreRegistry.Load(_root).Get(Path.GetFileName(targetB))!;
        Assert.Equal((ModelB, producerB, "C:/src", "directory"), (row.ModelId, row.Producer, row.Tool!["root"], row.Tool["source_kind"]));
        Assert.Equal(new DateTimeOffset(2026, 11, 2, 10, 0, 0, TimeSpan.Zero), row.Updated);
        Assert.Null(StoreRegistry.Load(_root).Get(cliName));
        Assert.Equal(2, JsonNode.Parse(File.ReadAllText(StoreRegistry.DefaultPath(_root)))!["format_version"]!.GetValue<int>());

        // The Task 4b name (index-<chunker>.<model12>.<ns>.json, no producer segment) migrates too (review ruling S3).
        ChunkKeyFunction keyT = KeyOf(ModelA, TenantA);
        var t4b = new ChunkIndex(keyT);
        t4b.Update([new SourceFile("T.txt", "task 4b name\n")], Chunker);
        string t4bName = $"index-{Chunker.Name}.{Tag(ModelA)}.{TenantA}.json";
        t4b.Save(Path.Combine(_root, t4bName));
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), t4bName), ChunkIndex.ResolveIndexPath(_root, Chunker.Name, ModelA, TenantA, keyT));
        IndexMigration mt = ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelA, TenantA, keyT, Wait);
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), t4bName), mt.MigratedFrom);
        Assert.Equal($"index-{Chunker.Name}.{Tag(ModelA)}.{P8(ModelA, TenantA)}.{TenantA}.json", Path.GetFileName(mt.IndexPath));
        Assert.Equal(["T.txt"], ChunkIndex.Load(mt.IndexPath, keyT).Paths);

        // Renamed, but the registry cannot be saved (newer format): the error says the index was renamed (review N9).
        File.WriteAllText(StoreRegistry.DefaultPath(_root), """{"format_version": 9, "indexes": []}""");
        ChunkKeyFunction keyN = KeyOf(ModelB, TenantB);
        var n = new ChunkIndex(keyN);
        n.Update([new SourceFile("N.txt", "newer registry\n")], Chunker);
        n.Save(ChunkIndex.DefaultPath(_root, Chunker.Name));
        var ex = Assert.Throws<InvalidDataException>(() => ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelB, TenantB, keyN, Wait));
        Assert.Contains("was renamed", ex.Message, StringComparison.Ordinal);
        Assert.True(File.Exists(ChunkIndex.DefaultPath(_root, Chunker.Name, ModelB, P(ModelB, TenantB), TenantB)));
    }

    [Fact]
    public void OldIndexName_OtherProducer_LeftAndStillReferenced()
    {
        var store = new FileChunkStore(_root);
        var other = new ChunkIndex(KeyOf(ModelB, ChunkNamespace.Default)); // another model wrote the legacy index-fixed-8.json
        other.Update([new SourceFile("O.txt", "other producer\n")], Chunker);
        string legacy = ChunkIndex.DefaultPath(_root, Chunker.Name);
        other.Save(legacy);
        foreach (ChunkKey k in other.ReferencedKeys()) store.Put(k, Entry(k));

        ChunkKeyFunction keyA = KeyOf(ModelA, ChunkNamespace.Default);
        IndexMigration m = ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelA, ChunkNamespace.Default, keyA, Wait);
        Assert.Null(m.MigratedFrom);
        Assert.Contains(m.LeftBehind, l => l.Contains("another producer", StringComparison.Ordinal));
        Assert.True(File.Exists(legacy));
        Assert.False(File.Exists(m.IndexPath));
        Assert.Equal(m.IndexPath, ChunkIndex.ResolveIndexPath(_root, Chunker.Name, ModelA, ChunkNamespace.Default, keyA));
        // Still an index: its keys stay protected.
        Assert.Empty(ChunkIndex.StoreWideOrphans(store, _root));
        Assert.Superset(other.ReferencedKeys().ToHashSet(), ChunkIndex.ReferencedKeysUnder(_root).ToHashSet());

        // An existing current-name index wins too: a matching legacy file is then left, not merged or overwritten.
        var mine = new ChunkIndex(keyA);
        mine.Update([new SourceFile("M.txt", "mine\n")], Chunker);
        mine.Save(m.IndexPath);
        var stale = new ChunkIndex(keyA);
        stale.Update([new SourceFile("S.txt", "stale legacy\n")], Chunker);
        File.Delete(legacy);
        stale.Save(legacy);
        IndexMigration again = ChunkIndex.MigrateLegacyIndex(_root, Chunker.Name, ModelA, ChunkNamespace.Default, keyA, Wait);
        Assert.Null(again.MigratedFrom);
        Assert.Contains(again.LeftBehind, l => l.Contains("already exists", StringComparison.Ordinal));
        Assert.Equal(["M.txt"], ChunkIndex.Load(m.IndexPath, keyA).Paths);
    }
}
