using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// The sole writer (Phase I Task 15, spec §5.10, §10, §14 L1/L2) driven directly: admission, the bounded queue and its
/// idle-only worker, the one-scope put / ledger row / save, budget eviction that respects index references, the
/// unreadable-ledger and lock-timeout rules. A fake host stands in for the model (entries of a chosen size); the
/// real-server path is <see cref="ChunkStoreServerTests"/>.
/// </summary>
public sealed class ChunkStoreServiceTests : IDisposable
{
    private static readonly TimeSpan Drain = TimeSpan.FromSeconds(30);
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-writer-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly List<ChunkStoreService> _services = [];

    public void Dispose()
    {
        foreach (var s in _services) s.Dispose();
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    // ------------------------------------------------------------------------------------------------ fakes and helpers

    /// <summary>The model stand-in: idle on demand, entries of <see cref="Stride"/> floats per row and token.</summary>
    internal sealed class FakeHost : IChunkEncodeHost
    {
        private int _leases;

        public volatile bool Idle = true;
        public int Stride = 4;
        public int Encodes;
        public int EncodesOutsideLease;

        public IDisposable? TryEnterIdle()
        {
            if (!Idle) return null;
            Interlocked.Increment(ref _leases);
            return new Lease(this);
        }

        public ChunkKvEntry? Encode(ChunkEncodeItem item)
        {
            if (Volatile.Read(ref _leases) != 1) Interlocked.Increment(ref EncodesOutsideLease);
            Interlocked.Increment(ref Encodes);
            return EntryFor(item.Text, Stride);
        }

        private sealed class Lease(FakeHost host) : IDisposable
        {
            public void Dispose() => Interlocked.Decrement(ref host._leases);
        }
    }

    internal static ChunkKvEntry EntryFor(string text, int stride)
    {
        int n = text.Length;
        var k = new float[n * stride];
        var v = new float[n * stride];
        for (int i = 0; i < k.Length; i++) { k[i] = i; v[i] = -i; }
        return new ChunkKvEntry("fp", text.Select(c => (int)c).ToArray(), [k], [v], "cpu");
    }

    internal static long FileBytes(string text, int stride) =>
        FileChunkStore.HeaderBytes("fp", "cpu", 1) + 4L * text.Length + 8L * text.Length * stride;

    /// <summary>A key for a text in a namespace (any well-formed key works: admission takes the request's key).</summary>
    internal static ChunkKey KeyFor(string text, string ns = "") =>
        new(Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(ns + "\u0000" + text))));

    internal static ChunkSpanRef Span(string text, bool isVolatile = false, bool noStore = false, string? source = null, bool sourceUnreadable = false) =>
        new(0, 0, 0, Encoding.UTF8.GetByteCount(text), "id", ChunkAnnotationParserTests.Sha(text), null, isVolatile, noStore, text,
            new EngineSpan(0, 0, text.Length)) { Index = 0, SourcePath = source, SourceUnreadable = sourceUnreadable };

    private static readonly ChunkCaller Local = new(true, ChunkNamespace.Default);

    private string NewRoot()
    {
        string root = Path.Combine(_scratch, "store-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        return root;
    }

    private ChunkStoreService Start(string root, FakeHost host, long budget = 1L << 30, bool indexOnly = false, IChunkStore? store = null,
        TimeProvider? time = null)
    {
        var options = new ChunkCacheOptions(true, ChunkCacheOptions.DefaultMaxChunks, root, null, StoreWrite: true, StoreBudgetBytes: budget, StoreIndexOnly: indexOnly);
        var service = ChunkStoreService.Start(options, host, store, time);
        _services.Add(service);
        return service;
    }

    /// <summary>A plan holding one verified chunk per text (hit or miss) for <see cref="ChunkStoreService.ApplyTo"/>.</summary>
    private static ChunkCachePlan Plan(params (string Text, bool Hit, ChunkKey Key)[] spans) =>
        new(ChunkPolicy.Exact, null, false, spans.Length,
            spans.Select((s, i) => new PlannedSpan(i * 100, s.Text.Length, s.Hit) { Span = Span(s.Text), Key = s.Key }).ToList(),
            [], null, "summary", "cpu", 1000);

    private static ParsedChunkRequest Parsed(string onMiss = "prefill") =>
        new(ChunkPolicy.Exact, [], null, false, "summary", ChunkNamespace.Default, [], null) { OnMiss = onMiss };

    private static ChunkWarning? Warning(ChunkCachePlan plan, string code) => plan.Warnings.FirstOrDefault(w => w.Code == code);

    private static string Text(char c, int n = 64) => new string(c, n) + "\n";

    private static HashSet<string> LedgerKeysOnDisk(string root) =>
        File.Exists(ServerLedger.DefaultPath(root))
            ? ServerLedger.Load(root).Rows.Select(r => r.KeyHex).ToHashSet(StringComparer.Ordinal)
            : [];

    private static string RawLedgerText(string root) =>
        File.Exists(ServerLedger.DefaultPath(root)) ? File.ReadAllText(ServerLedger.DefaultPath(root)) : "";

    /// <summary>Writes a CLI-style index file under the store root that references <paramref name="keys"/>.</summary>
    internal static void WriteIndex(string root, string name, params ChunkKey[] keys)
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
        File.WriteAllText(Path.Combine(root, $"index-{name}.json"), doc.ToJsonString());
    }

    private static void Admitted(ChunkStoreService s, string text, ChunkKey key, bool onMissEncode = false) =>
        Assert.True(s.Admit(Span(text), key, text.Length, Local, onMissEncode, "cpu").Queued);

    private sealed class ManualTime : TimeProvider
    {
        public DateTimeOffset Now = new(2026, 11, 3, 12, 0, 0, TimeSpan.Zero);
        public override DateTimeOffset GetUtcNow() => Now;
    }

    /// <summary>A store whose puts can be intercepted (before and after the inner put).</summary>
    private sealed class HookedStore(FileChunkStore inner) : IChunkStore
    {
        public Action<ChunkKey>? BeforePut;
        public Action<ChunkKey>? AfterPut;

        public bool TryGet(ChunkKey key, out ChunkKvEntry entry) => inner.TryGet(key, out entry);
        public void Put(ChunkKey key, ChunkKvEntry entry)
        {
            BeforePut?.Invoke(key);
            inner.Put(key, entry);
            AfterPut?.Invoke(key);
        }
        public bool Contains(ChunkKey key) => inner.Contains(key);
        public IEnumerable<ChunkKey> Keys() => inner.Keys();
        public long TotalBytes => inner.TotalBytes;
        public void Evict(ChunkKey key) => inner.Evict(key);
    }

    // ------------------------------------------------------------------------------------------------ admission (§5.10)

    /// <summary>A miss is queued only from its second request in the same namespace; the worker then stores it with a ledger row.</summary>
    [Fact]
    public void SeenOnce_NotQueued_SeenTwice_Queued()
    {
        string root = NewRoot();
        var host = new FakeHost();
        var s = Start(root, host);
        string text = Text('a');
        var key = KeyFor(text);

        var first = s.Admit(Span(text), key, text.Length, Local, false, "cpu");
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.NotSeenTwice), first);
        // Another namespace's sighting does not count for this one.
        var other = new ChunkCaller(true, "k-0123456789abcdef");
        Assert.False(s.Admit(Span(text), KeyFor(text, other.Namespace), text.Length, other, false, "cpu").Queued);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(0, host.Encodes);
        Assert.False(new FileChunkStore(root).Contains(key));

        var second = s.Admit(Span(text), key, text.Length, Local, false, "cpu");
        Assert.Equal(new AdmissionResult(true, null), second);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(1, host.Encodes);
        Assert.True(new FileChunkStore(root).Contains(key));
        Assert.Contains(key.Hex, LedgerKeysOnDisk(root));
        Assert.False(new FileChunkStore(root).Contains(KeyFor(text, other.Namespace)));
    }

    /// <summary>
    /// <c>volatile</c>, <c>no_store</c>, a deny-listed <c>source.path</c> and an unreadable <c>source</c> are never queued,
    /// however often they are seen, and leave no sighting (spec §5.2: no_store persists neither K/V nor sighting).
    /// </summary>
    [Fact]
    public void VolatileAndNoStore_NeverQueued()
    {
        string root = NewRoot();
        var host = new FakeHost();
        var s = Start(root, host);
        var cases = new (ChunkSpanRef Span, string Reason)[]
        {
            (Span(Text('v'), isVolatile: true), AdmissionReasons.Volatile),
            (Span(Text('n'), noStore: true), AdmissionReasons.NoStore),
            (Span(Text('e'), source: ".env.local"), AdmissionReasons.DeniedPath),
            (Span(Text('p'), source: "certs/server.PEM"), AdmissionReasons.DeniedPath),
            (Span(Text('s'), source: "config\\secrets.json"), AdmissionReasons.DeniedPath),
            (Span(Text('g'), source: "repo/.git/config"), AdmissionReasons.DeniedPath),
            (Span(Text('i'), source: "home/id_ed25519"), AdmissionReasons.DeniedPath),
            (Span(Text('d'), source: "appsettings.Local.Development.json"), AdmissionReasons.DeniedPath),
            (Span(Text('u'), sourceUnreadable: true), AdmissionReasons.SourceUnreadable),
        };
        foreach (var (span, reason) in cases)
        {
            for (int i = 0; i < 3; i++)
            {
                var r = s.Admit(span, KeyFor(span.Text), span.Text.Length, Local, onMissEncode: true, "cpu");
                Assert.Equal(new AdmissionResult(false, reason), r);
            }
            Assert.Equal(0, s.Sightings.CountOf(ChunkNamespace.Default, span.Hash));
        }
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(0, host.Encodes);
        Assert.Empty(new FileChunkStore(root).Keys());
        // A plain path next to them is admitted.
        Admitted(s, Text('o'), KeyFor(Text('o')), onMissEncode: true);
    }

    /// <summary>The deny-list (spec §10): file-name patterns at any depth, <c>.git/**</c> at any depth, case-insensitive.</summary>
    [Theory]
    [InlineData(".env", true)]
    [InlineData("a/b/.env.production", true)]
    [InlineData("KEYS/Server.KEY", true)]
    [InlineData("cert.pfx", true)]
    [InlineData("C:\\users\\me\\.ssh\\id_rsa", true)]
    [InlineData("secrets.json", true)]
    [InlineData("src/appsettings.Staging.Development.json", true)]
    [InlineData(".git/HEAD", true)]
    [InlineData("vendor/lib/.git/objects/ab/cd", true)]
    [InlineData("../outside/.env", true)]
    [InlineData("src/Program.cs", false)]
    [InlineData("src/environment.cs", false)]
    [InlineData("src/appsettings.Development.json", false)]
    [InlineData("docs/git/README.md", false)]
    [InlineData("src/.gitignore", false)]
    [InlineData("keys/readme.keys", false)]
    [InlineData("", false)]
    public void DenyList_DefaultPatterns(string path, bool denied) => Assert.Equal(denied, ChunkDenyList.Default.IsDenied(path));

    /// <summary>
    /// <c>--chunk-store-write</c> needs <c>--chunk-store</c> and a positive <c>--chunk-store-budget</c> (no unlimited default),
    /// and the budget and index-only switches need <c>--chunk-store-write</c>; on both option paths.
    /// </summary>
    [Fact]
    public void BudgetRequired_StartupFailsWithoutIt()
    {
        string root = NewRoot();
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, root, storeWrite: true));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, root, storeWrite: true, storeBudget: "0"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, root, storeWrite: true, storeBudget: "lots"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, null, storeWrite: true, storeBudget: "1GiB"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, root, storeBudget: "1GiB"));
        Assert.Throws<ArgumentException>(() => ChunkCacheOptions.Create(true, null, root, storeIndexOnly: true));
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-store", root, "--chunk-store-write"]));

        var ok = ChunkCacheOptions.Create(true, null, root, storeWrite: true, storeBudget: "2MiB", storeIndexOnly: true);
        Assert.True(ok.StoreWrite);
        Assert.Equal(2L << 20, ok.StoreBudgetBytes);
        Assert.True(ok.StoreIndexOnly);
        Assert.Equal(ok, ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-store", root, "--chunk-store-write",
            "--chunk-store-budget", "2MiB", "--chunk-store-index-only"]).ChunkCache);
        // Read-only stays as it was (Task 13): no budget needed, nothing written.
        var readOnly = ChunkCacheOptions.Create(true, null, root);
        Assert.False(readOnly.StoreWrite);
        Assert.Equal(0, readOnly.StoreBudgetBytes);
    }

    /// <summary>The worker encodes only while the host reports the server idle, and only inside the lease.</summary>
    [Fact]
    public void EncodesOnlyWhenIdle()
    {
        string root = NewRoot();
        var host = new FakeHost { Idle = false };
        var s = Start(root, host);
        string text = Text('w');
        Admitted(s, text, KeyFor(text), onMissEncode: true);

        Assert.False(s.Queue.WaitDrained(TimeSpan.FromMilliseconds(400)));
        Assert.Equal(0, host.Encodes);
        Assert.Equal(1, s.Queue.Pending);

        host.Idle = true;
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(1, host.Encodes);
        Assert.Equal(0, host.EncodesOutsideLease);
        Assert.True(new FileChunkStore(root).Contains(KeyFor(text)));
    }

    /// <summary><c>on_miss: "encode"</c> bypasses the seen-twice rule only: the first sighting is queued.</summary>
    [Fact]
    public void OnMissEncode_BypassesOnlySeenTwice()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        var plan = s.ApplyTo(Plan((Text('x'), false, KeyFor(Text('x')))), Parsed("encode"), Local);
        Assert.Equal(1, Warning(plan, ChunkCacheCodes.EncodeQueued)?.Count);
        Assert.Equal(Text('x').Length, plan.EncodeQueuedTokens);
        Assert.Null(Warning(plan, ChunkCacheCodes.EncodeSkipped));
        Assert.False(s.Admit(Span(Text('y'), noStore: true), KeyFor(Text('y')), 10, Local, onMissEncode: true, "cpu").Queued);
        var once = s.ApplyTo(Plan((Text('z'), false, KeyFor(Text('z')))), Parsed(), Local);
        Assert.Equal(1, Warning(once, ChunkCacheCodes.EncodeSkipped)?.Count);
        Assert.Equal(0, once.EncodeQueuedTokens);
    }

    /// <summary>The same text twice in one request is one sighting (sightings count requests, not spans).</summary>
    [Fact]
    public void SameTextTwiceInOneRequest_IsOneSighting()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        string text = Text('t');
        var plan = s.ApplyTo(Plan((text, false, KeyFor(text)), (text, false, KeyFor(text))), Parsed(), Local);
        Assert.Equal(2, Warning(plan, ChunkCacheCodes.EncodeSkipped)?.Count);
        Assert.Null(Warning(plan, ChunkCacheCodes.EncodeQueued));
        Assert.Equal(1, s.Sightings.CountOf(ChunkNamespace.Default, ChunkAnnotationParserTests.Sha(text)));
        var next = s.ApplyTo(Plan((text, false, KeyFor(text))), Parsed(), Local);
        Assert.Equal(1, Warning(next, ChunkCacheCodes.EncodeQueued)?.Count);
    }

    /// <summary>A full queue refuses (bounded memory); one text queued twice is one item.</summary>
    [Fact]
    public void Queue_IsBounded_AndDeduplicated()
    {
        string root = NewRoot();
        var host = new FakeHost { Idle = false };
        var s = Start(root, host);
        string same = Text('q');
        Admitted(s, same, KeyFor(same), onMissEncode: true);
        Admitted(s, same, KeyFor(same), onMissEncode: true);
        Assert.InRange(s.Queue.Pending, 1, 1);
        for (int i = 1; i < ChunkEncodeQueue.MaxItems; i++)
            Admitted(s, "t" + i + "\n" + Text('q', 40), KeyFor("t" + i + "\n" + Text('q', 40)), onMissEncode: true);
        var over = s.Admit(Span("overflow\n" + Text('q', 40)), KeyFor("overflow"), 50, Local, true, "cpu");
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.QueueFull), over);
    }

    /// <summary>
    /// Only the CPU backend writes (the Vulkan schema question, plan Task 15), and chunks longer than
    /// <see cref="ChunkStoreService.MaxEncodeTokens"/> are not encoded in the background.
    /// </summary>
    [Fact]
    public void VulkanBackendAndOverlongChunks_NotQueued()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.Backend),
            s.Admit(Span(Text('b')), KeyFor(Text('b')), 65, Local, true, "vulkan"));
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.TooLong),
            s.Admit(Span(Text('l')), KeyFor(Text('l')), ChunkStoreService.MaxEncodeTokens + 1, Local, true, "cpu"));
    }

    // ------------------------------------------------------------------------------------------------ budget

    /// <summary>
    /// Over budget, the least recently used ledger row goes first (a hit refreshes a row); an entry only an index
    /// references is never evicted.
    /// </summary>
    [Fact]
    public void OverBudget_EvictsLruLedgerRows_NeverIndexOnlyEntries()
    {
        string root = NewRoot();
        var time = new ManualTime();
        string a = Text('a', 200), b = Text('b', 200), c = Text('c', 200), cli = Text('i', 200);
        long one = FileBytes(a, 4);
        // A CLI entry referenced only by an index.
        var cliKey = KeyFor(cli);
        new FileChunkStore(root).Put(cliKey, EntryFor(cli, 4));
        WriteIndex(root, "cli", cliKey);
        var host = new FakeHost();
        var s = Start(root, host, budget: 3 * one + one / 2, time: time); // room for the CLI entry and two more

        Admitted(s, a, KeyFor(a), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        time.Now += TimeSpan.FromMinutes(1);
        Admitted(s, b, KeyFor(b), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        time.Now += TimeSpan.FromMinutes(1);
        // A hit on a: b is now the least recently used.
        s.ApplyTo(Plan((a, true, KeyFor(a))), Parsed(), Local);
        time.Now += TimeSpan.FromMinutes(1);
        Admitted(s, c, KeyFor(c), true);
        Assert.True(s.Queue.WaitDrained(Drain));

        var store = new FileChunkStore(root);
        Assert.True(store.Contains(cliKey));
        Assert.True(store.Contains(KeyFor(a)));
        Assert.False(store.Contains(KeyFor(b)));
        Assert.True(store.Contains(KeyFor(c)));
        Assert.Equal(new[] { KeyFor(a).Hex, KeyFor(c).Hex }.Order(StringComparer.Ordinal), LedgerKeysOnDisk(root).Order(StringComparer.Ordinal));
        Assert.True(store.GetStats().EntryBytes <= s.BudgetBytes);
        Assert.Null(s.Problem);
    }

    /// <summary>
    /// When the server's own entries cannot free enough (index-referenced entries fill the budget), nothing is evicted,
    /// encoding stops, and responses with a miss warn <c>chunk_store_unavailable</c> (<c>budget_exhausted</c>).
    /// </summary>
    [Fact]
    public void CannotFree_StopsEncoding_Warns()
    {
        string root = NewRoot();
        string cli1 = Text('1', 200), cli2 = Text('2', 200), mine = Text('m', 200);
        long one = FileBytes(cli1, 4);
        var store = new FileChunkStore(root);
        store.Put(KeyFor(cli1), EntryFor(cli1, 4));
        store.Put(KeyFor(cli2), EntryFor(cli2, 4));
        WriteIndex(root, "cli", KeyFor(cli1), KeyFor(cli2));
        var host = new FakeHost();
        var s = Start(root, host, budget: 2 * one + one / 2);

        Admitted(s, mine, KeyFor(mine), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(StoreProblems.BudgetExhausted, s.Problem);
        Assert.True(store.Contains(KeyFor(cli1)));
        Assert.True(store.Contains(KeyFor(cli2)));
        Assert.False(store.Contains(KeyFor(mine)));
        Assert.Empty(LedgerKeysOnDisk(root));

        var plan = s.ApplyTo(Plan((Text('n', 200), false, KeyFor(Text('n', 200)))), Parsed("encode"), Local);
        var unavailable = Warning(plan, ChunkCacheCodes.StoreUnavailable);
        Assert.NotNull(unavailable);
        Assert.Equal(StoreProblems.BudgetExhausted, unavailable!.Detail);
        Assert.Equal(1, Warning(plan, ChunkCacheCodes.EncodeSkipped)?.Count);
        Assert.Null(Warning(plan, ChunkCacheCodes.EncodeQueued));
        Assert.Equal(1, host.Encodes); // stopped: the refused miss was never encoded
    }

    /// <summary>
    /// Budget eviction drops the ledger row of a key a CLI index also references but keeps its file (rule (c)); only an
    /// unreferenced row frees bytes.
    /// </summary>
    [Fact]
    public void BudgetEviction_KeyAlsoInCliIndex_KeepsFile_RemovesRowOnly()
    {
        string root = NewRoot();
        var time = new ManualTime();
        string shared = Text('s', 200), own = Text('o', 200), next = Text('n', 200);
        long one = FileBytes(shared, 4);
        var host = new FakeHost();
        var s = Start(root, host, budget: 2 * one + one / 2, time: time);
        Admitted(s, shared, KeyFor(shared), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        time.Now += TimeSpan.FromMinutes(1);
        Admitted(s, own, KeyFor(own), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        WriteIndex(root, "cli", KeyFor(shared)); // the CLI indexes the same text (content-addressed key)
        time.Now += TimeSpan.FromMinutes(1);

        Admitted(s, next, KeyFor(next), true);
        Assert.True(s.Queue.WaitDrained(Drain));

        var store = new FileChunkStore(root);
        Assert.True(store.Contains(KeyFor(shared)));   // file kept: an index references it
        Assert.DoesNotContain(KeyFor(shared).Hex, LedgerKeysOnDisk(root)); // its row went (LRU)
        Assert.False(store.Contains(KeyFor(own)));     // the unreferenced row freed the room
        Assert.True(store.Contains(KeyFor(next)));
        Assert.Contains(KeyFor(next).Hex, LedgerKeysOnDisk(root));
    }

    // ------------------------------------------------------------------------------------------------ crash, disk full, locks

    /// <summary>
    /// A crash mid-put leaves no loadable entry and no row; a failure mid-ledger-save keeps the old ledger file, rolls the
    /// in-memory ledger back and removes the just-written file; a row whose file is gone is dropped on load (Review Focus).
    /// </summary>
    [Fact]
    public void CrashMidPut_NoLoadableEntry_LedgerConsistentOnReload()
    {
        string root = NewRoot();
        var files = new FileChunkStore(root);
        var host = new FakeHost();
        var s = Start(root, host, store: files);
        string first = Text('f'), crash = Text('c'), saveFail = Text('l');

        Admitted(s, first, KeyFor(first), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        byte[] ledgerBefore = File.ReadAllBytes(ServerLedger.DefaultPath(root));

        // 1. Crash inside the put, before the rename.
        files.BeforeRenameForTest = _ => throw new IOException("simulated crash mid-put");
        Admitted(s, crash, KeyFor(crash), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        files.BeforeRenameForTest = null;
        Assert.False(files.Contains(KeyFor(crash)));
        Assert.False(File.Exists(files.PathOf(KeyFor(crash))));
        Assert.Equal(ledgerBefore, File.ReadAllBytes(ServerLedger.DefaultPath(root)));

        // 2. Failure inside the ledger save, after the entry file was put.
        s.LedgerForTest!.BeforeRename = _ => throw new IOException("simulated full disk mid-save");
        Admitted(s, saveFail, KeyFor(saveFail), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        s.LedgerForTest.BeforeRename = null;
        Assert.Equal(ledgerBefore, File.ReadAllBytes(ServerLedger.DefaultPath(root)));
        Assert.False(files.Contains(KeyFor(saveFail)));
        Assert.Equal(new[] { KeyFor(first).Hex }, s.LedgerForTest.Rows.Select(r => r.KeyHex));
        Assert.Equal(StoreProblems.WriteFailed, s.Problem);

        // Reload: every row has its file, every file has its row.
        var reloaded = ServerLedger.Load(root);
        Assert.Equal(0, reloaded.DroppedOnLoad);
        Assert.Equal(files.Keys().Select(k => k.Hex).Order(StringComparer.Ordinal), reloaded.Rows.Select(r => r.KeyHex).Order(StringComparer.Ordinal));

        // A later write succeeds and clears the problem.
        Admitted(s, Text('g'), KeyFor(Text('g')), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Null(s.Problem);

        // A row whose file was removed behind the server's back is dropped on the next load.
        files.Evict(KeyFor(first));
        Assert.Equal(1, ServerLedger.Load(root).DroppedOnLoad);
    }

    /// <summary>A put that fails (disk full) stores nothing and records no row; the chunk stays a miss and responses warn.</summary>
    [Fact]
    public void DiskFull_PutFails_IsMissAndWarning()
    {
        string root = NewRoot();
        var hooked = new HookedStore(new FileChunkStore(root))
        {
            BeforePut = _ => throw new IOException("There is not enough space on the disk."),
        };
        var s = Start(root, new FakeHost(), store: hooked);
        string text = Text('d');
        Admitted(s, text, KeyFor(text), true);
        Assert.True(s.Queue.WaitDrained(Drain));

        Assert.False(hooked.Contains(KeyFor(text)));
        Assert.Empty(LedgerKeysOnDisk(root));
        Assert.Equal(StoreProblems.WriteFailed, s.Problem);
        var plan = s.ApplyTo(Plan((text, false, KeyFor(text))), Parsed("encode"), Local);
        Assert.Equal(StoreProblems.WriteFailed, Warning(plan, ChunkCacheCodes.StoreUnavailable)?.Detail);
        Assert.Equal(1, Warning(plan, ChunkCacheCodes.EncodeQueued)?.Count); // retried: a transient failure does not stop the writer
    }

    /// <summary>
    /// The put, the ledger row and the save run in one store-lock scope: the row is never on disk before its file, another
    /// holder cannot take the lock between the put and the save, and after the scope every entry has its row.
    /// </summary>
    [Fact]
    public void PutRecordSave_InOneLockScope()
    {
        string root = NewRoot();
        var hooked = new HookedStore(new FileChunkStore(root));
        var s = Start(root, new FakeHost(), store: hooked);
        var seenBefore = new List<bool>();
        var lockFreeAfterPut = new List<bool>();
        // The raw file, not ServerLedger.Load (which drops a row whose entry file is missing: exactly the state looked for).
        hooked.BeforePut = key => seenBefore.Add(RawLedgerText(root).Contains(key.Hex, StringComparison.Ordinal));
        hooked.AfterPut = _ =>
        {
            // Another holder (another thread, as a CLI would be another process) tries the store lock right after the put.
            bool free = Task.Run(() =>
            {
                try
                {
                    using var probe = ChunkIndex.AcquireStoreLock(root, TimeSpan.Zero);
                    return true;
                }
                catch (ChunkStoreLockTimeoutException)
                {
                    return false;
                }
            }).GetAwaiter().GetResult();
            lockFreeAfterPut.Add(free);
        };

        foreach (char c in "pqr")
            Admitted(s, Text(c), KeyFor(Text(c)), true);
        Assert.True(s.Queue.WaitDrained(Drain));

        Assert.Equal(new[] { false, false, false }, seenBefore);
        Assert.Equal(new[] { false, false, false }, lockFreeAfterPut);
        using (ChunkIndex.AcquireStoreLock(root, TimeSpan.FromSeconds(5)))
            Assert.Equal(hooked.Keys().Select(k => k.Hex).Order(StringComparer.Ordinal), LedgerKeysOnDisk(root).Order(StringComparer.Ordinal));
    }

    /// <summary>
    /// Another writer racing for the store lock from the moment of each put (spinning on the lock file, as another process
    /// would) never observes, once it holds the lock, an entry file without its ledger row. The race is real, so this can
    /// never fail on a correct writer; on a writer that releases the lock between put and save it wins the gap often
    /// enough over 12 writes to fail (mutant M6).
    /// </summary>
    [Fact]
    public void RacingWriter_NeverSeesAFileWithoutItsRow()
    {
        string root = NewRoot();
        var hooked = new HookedStore(new FileChunkStore(root));
        var s = Start(root, new FakeHost(), store: hooked);
        string lockFile = Path.Combine(root, ChunkIndex.StoreLockFileName);
        var violations = new System.Collections.Concurrent.ConcurrentBag<string>();
        var racers = new List<Thread>();
        hooked.AfterPut = key =>
        {
            var racer = new Thread(() =>
            {
                var sw = Stopwatch.StartNew();
                while (sw.Elapsed < TimeSpan.FromSeconds(10))
                {
                    FileStream held;
                    try { held = new FileStream(lockFile, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); }
                    catch (IOException) { continue; }
                    catch (UnauthorizedAccessException) { continue; }
                    using (held)
                    {
                        if (File.Exists(new FileChunkStore(root).PathOf(key)) && !RawLedgerText(root).Contains(key.Hex, StringComparison.Ordinal))
                            violations.Add(key.Hex);
                    }
                    return;
                }
            }) { IsBackground = true, Priority = ThreadPriority.AboveNormal };
            racers.Add(racer);
            racer.Start();
        };

        for (int i = 0; i < 12; i++)
        {
            string text = "race " + i + "\n" + Text('r');
            Admitted(s, text, KeyFor(text), true);
            Assert.True(s.Queue.WaitDrained(Drain));
        }
        foreach (var r in racers)
            Assert.True(r.Join(TimeSpan.FromSeconds(15)));
        Assert.Empty(violations);
        Assert.Equal(12, LedgerKeysOnDisk(root).Count);
    }

    /// <summary>
    /// A store lock held elsewhere past the timeout is reported as <c>lock_timeout</c> (never as a plain busy store or a
    /// write failure), nothing is written, and the next write after the lock is released clears it.
    /// </summary>
    [Fact]
    public void LockTimeout_SurfacesAsLockTimeoutDetail()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        s.LockTimeout = TimeSpan.FromMilliseconds(200);
        string text = Text('k');
        using (ChunkIndex.AcquireStoreLock(root, TimeSpan.FromSeconds(5)))
        {
            Admitted(s, text, KeyFor(text), true);
            Assert.True(s.Queue.WaitDrained(Drain));
        }
        Assert.Equal(StoreProblems.LockTimeout, s.Problem);
        Assert.False(new FileChunkStore(root).Contains(KeyFor(text)));
        var plan = s.ApplyTo(Plan((text, false, KeyFor(text))), Parsed("encode"), Local);
        Assert.Equal(StoreProblems.LockTimeout, Warning(plan, ChunkCacheCodes.StoreUnavailable)?.Detail);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.True(new FileChunkStore(root).Contains(KeyFor(text)));
        Assert.Null(s.Problem);
    }

    /// <summary>Admission never waits for the store lock: with the lock held elsewhere it returns at once.</summary>
    [Fact]
    public void Admission_NeverWaitsForTheStoreLock()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost { Idle = false });
        using var held = ChunkIndex.AcquireStoreLock(root, TimeSpan.FromSeconds(5));
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 50; i++)
            s.ApplyTo(Plan((Text('h') + i, false, KeyFor(Text('h') + i))), Parsed("encode"), Local);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(2), $"admission took {sw.Elapsed}");
    }

    /// <summary>A second writer on the same root fails at once with a clear message; after the first stops, a new one starts.</summary>
    [Fact]
    public void SecondServerSameRoot_FailsFast()
    {
        string root = NewRoot();
        var first = Start(root, new FakeHost());
        var sw = Stopwatch.StartNew();
        var ex = Assert.Throws<InvalidServerOptionException>(() => Start(root, new FakeHost()));
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(2));
        Assert.Contains("already owned by another dotllm server", ex.Message, StringComparison.Ordinal);
        Assert.True(StoreOwnerLock.IsHeldByAnother(root));
        first.Dispose();
        Assert.False(StoreOwnerLock.IsHeldByAnother(root));
        Start(root, new FakeHost());
    }

    /// <summary>
    /// An unreadable ledger at startup disables writes for the server's lifetime: the file is never replaced (byte-identical
    /// after misses that would otherwise be stored) and misses warn <c>chunk_store_unavailable</c> (<c>ledger_unreadable</c>).
    /// </summary>
    [Fact]
    public void UnreadableLedgerAtStartup_WritesDisabled_FileUntouched()
    {
        string root = NewRoot();
        byte[] garbage = Encoding.UTF8.GetBytes("{\"formatVersion\": 1, \"rows\": [ {\"key\": \"not-a-key\"} ]}");
        File.WriteAllBytes(ServerLedger.DefaultPath(root), garbage);
        var host = new FakeHost();
        var s = Start(root, host);
        Assert.True(s.WritesDisabled);

        string text = Text('u');
        for (int i = 0; i < 3; i++)
        {
            var plan = s.ApplyTo(Plan((text, false, KeyFor(text))), Parsed("encode"), Local);
            Assert.Equal(StoreProblems.LedgerUnreadable, Warning(plan, ChunkCacheCodes.StoreUnavailable)?.Detail);
            Assert.Equal(1, Warning(plan, ChunkCacheCodes.EncodeSkipped)?.Count);
        }
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(0, host.Encodes);
        Assert.Equal(garbage, File.ReadAllBytes(ServerLedger.DefaultPath(root)));
        Assert.Empty(new FileChunkStore(root).Keys());
    }

    /// <summary>With <c>--chunk-store-index-only</c> only chunks an index references are stored; ad-hoc text never is.</summary>
    [Fact]
    public void IndexOnly_AdHocTextNeverStored()
    {
        string root = NewRoot();
        string indexed = Text('x'), adHoc = Text('y');
        WriteIndex(root, "cli", KeyFor(indexed));
        var host = new FakeHost();
        var s = Start(root, host, indexOnly: true);

        Assert.Equal(new AdmissionResult(false, AdmissionReasons.IndexOnly), s.Admit(Span(adHoc), KeyFor(adHoc), 65, Local, true, "cpu"));
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.IndexOnly), s.Admit(Span(adHoc), KeyFor(adHoc), 65, Local, true, "cpu"));
        Admitted(s, indexed, KeyFor(indexed), true);
        Assert.True(s.Queue.WaitDrained(Drain));
        var store = new FileChunkStore(root);
        Assert.True(store.Contains(KeyFor(indexed)));
        Assert.False(store.Contains(KeyFor(adHoc)));
        Assert.Equal(1, host.Encodes);
    }

    /// <summary>Shutdown stops a worker that is waiting for an idle server: no encode happens afterwards.</summary>
    [Fact]
    public void Dispose_StopsTheWorker_PendingItemsDropped()
    {
        string root = NewRoot();
        var host = new FakeHost { Idle = false };
        var s = Start(root, host);
        Admitted(s, Text('z'), KeyFor(Text('z')), true);
        var sw = Stopwatch.StartNew();
        s.Dispose();
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5));
        host.Idle = true;
        Thread.Sleep(200);
        Assert.Equal(0, host.Encodes);
        Assert.False(StoreOwnerLock.IsHeldByAnother(root));
        Assert.False(s.Admit(Span(Text('z')), KeyFor(Text('z')), 65, Local, true, "cpu").Queued);
    }
}
