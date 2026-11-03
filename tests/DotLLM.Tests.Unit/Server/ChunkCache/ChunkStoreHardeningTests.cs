using System.Diagnostics;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using Xunit;
using static DotLLM.Tests.Unit.Server.ChunkCache.ChunkStoreServiceTests;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// The sole writer's hardening (Phase I Task 15 security review, S1-S6 and nits): oversized entries never stop the writer,
/// shutdown never frees what a running encode uses, a ledger that turns unreadable mid-run stops writes until a verified
/// reload, the worker waits for a quiet period and caps each idle window, per-namespace and per-request limits, the
/// widened deny-list with content sniffing, and the most restrictive occurrence of a repeated text.
/// </summary>
public sealed class ChunkStoreHardeningTests : IDisposable
{
    private static readonly TimeSpan Drain = TimeSpan.FromSeconds(30);
    private static readonly ChunkCaller Local = new(true, ChunkNamespace.Default);
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-harden-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly List<ChunkStoreService> _services = [];

    public void Dispose()
    {
        foreach (var s in _services) s.Dispose();
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private string NewRoot()
    {
        string root = Path.Combine(_scratch, "store-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        return root;
    }

    private ChunkStoreService Start(string root, FakeHost host, long budget = 1L << 30, TimeProvider? time = null)
    {
        var options = new ChunkCacheOptions(true, ChunkCacheOptions.DefaultMaxChunks, root, null, StoreWrite: true, StoreBudgetBytes: budget);
        var service = ChunkStoreService.Start(options, host, null, time);
        service.Queue.QuietPeriod = TimeSpan.Zero;
        _services.Add(service);
        return service;
    }

    private static string Text(char c, int n = 64) => new string(c, n) + "\n";

    private static void Admitted(ChunkStoreService s, string text, ChunkKey key, ChunkCaller? caller = null) =>
        Assert.True(s.Admit(Span(text), key, text.Length, caller ?? Local, true, "cpu").Queued);

    private static ChunkCachePlan Plan(params (string Text, bool Hit, ChunkKey Key, ChunkSpanRef? Span)[] spans) =>
        new(ChunkPolicy.Exact, null, false, spans.Length,
            spans.Select((s, i) => new PlannedSpan(i * 100, s.Text.Length, s.Hit) { Span = s.Span ?? Span(s.Text), Key = s.Key }).ToList(),
            [], null, "summary", "cpu", 1000);

    private static ParsedChunkRequest Parsed(string onMiss = "prefill") =>
        new(ChunkPolicy.Exact, [], null, false, "summary", ChunkNamespace.Default, [], null) { OnMiss = onMiss };

    private static ChunkWarning? Warning(ChunkCachePlan plan, string code) => plan.Warnings.FirstOrDefault(w => w.Code == code);

    private static bool Eventually(Func<bool> condition, TimeSpan within)
    {
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed < within)
        {
            if (condition()) return true;
            Thread.Sleep(20);
        }
        return condition();
    }

    private sealed class ManualTime : TimeProvider
    {
        public DateTimeOffset Now = new(2026, 11, 3, 12, 0, 0, TimeSpan.Zero);
        public override DateTimeOffset GetUtcNow() => Now;
    }

    /// <summary>
    /// S1: an encoded entry larger than the whole budget is dropped alone as <c>entry_too_large</c>; nothing is evicted, the
    /// writer is not stopped (no <c>budget_exhausted</c>), and the next normal entry is written.
    /// </summary>
    [Fact]
    public void OversizedEntry_DroppedAsTooLarge_WriterKeepsRunning()
    {
        string root = NewRoot();
        string a = Text('a', 200), huge = Text('h', 200), b = Text('b', 200);
        var host = new FakeHost();
        var s = Start(root, host, budget: 2 * FileBytes(a, 4));
        Admitted(s, a, KeyFor(a));
        Assert.True(s.Queue.WaitDrained(Drain));

        host.Stride = 64; // far over the whole budget
        Admitted(s, huge, KeyFor(huge));
        Assert.True(s.Queue.WaitDrained(Drain));
        var store = new FileChunkStore(root);
        Assert.False(store.Contains(KeyFor(huge)));
        Assert.True(store.Contains(KeyFor(a)));
        Assert.Equal(StoreProblems.EntryTooLarge, s.Problem);

        host.Stride = 4;
        var plan = s.ApplyTo(Plan((b, false, KeyFor(b), null)), Parsed("encode"), Local);
        Assert.Equal(StoreProblems.EntryTooLarge, Warning(plan, ChunkCacheCodes.StoreUnavailable)?.Detail);
        Assert.Equal(1, Warning(plan, ChunkCacheCodes.EncodeQueued)?.Count); // not stopped for everyone
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.True(store.Contains(KeyFor(b)));
        Assert.Null(s.Problem);
    }

    /// <summary>
    /// S2: shutdown while an encode runs: <see cref="ChunkStoreService.Dispose"/> returns after its timeout, but the owner
    /// lock stays held while the worker may still run and is released when it exits; nothing is written after shutdown.
    /// </summary>
    [Fact]
    public void Dispose_WhileEncoding_KeepsOwnerLockUntilWorkerExits()
    {
        string root = NewRoot();
        var host = new FakeHost { Block = new ManualResetEventSlim(false) };
        var s = Start(root, host);
        s.Queue.ShutdownTimeout = TimeSpan.FromMilliseconds(200);
        Admitted(s, Text('s'), KeyFor(Text('s')));
        Assert.True(host.Entered.Wait(TimeSpan.FromSeconds(10)));

        var sw = Stopwatch.StartNew();
        s.Dispose();
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5));
        Assert.False(s.WorkerStopped);
        Assert.True(StoreOwnerLock.IsHeldByAnother(root)); // not released under a running worker

        host.Block.Set();
        Assert.True(Eventually(() => s.WorkerStopped, TimeSpan.FromSeconds(10)));
        Assert.True(Eventually(() => !StoreOwnerLock.IsHeldByAnother(root), TimeSpan.FromSeconds(10)));
        Assert.False(new FileChunkStore(root).Contains(KeyFor(Text('s'))));
    }

    /// <summary>
    /// S3: the ledger turns unreadable mid-run (a failed save, then reloads that keep failing): writes stop
    /// (<c>ledger_unreadable</c>), the unsaved row leaves memory, the file is never written, and only a verified reload
    /// clears the state.
    /// </summary>
    [Fact]
    public void LedgerUnreadableMidRun_StopsWrites_FileUntouched_RecoversOnVerifiedReload()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        s.ReloadRetryInterval = TimeSpan.FromMilliseconds(200);
        string first = Text('1'), second = Text('2'), third = Text('3');
        Admitted(s, first, KeyFor(first));
        Assert.True(s.Queue.WaitDrained(Drain));
        string ledgerPath = ServerLedger.DefaultPath(root);
        byte[] before = File.ReadAllBytes(ledgerPath);

        FileStream? holder = null;
        s.LedgerForTest!.BeforeRename = _ =>
        {
            holder = new FileStream(ledgerPath, FileMode.Open, FileAccess.Read, FileShare.None); // a scanner holds it on
            throw new IOException("simulated save failure");
        };
        Admitted(s, second, KeyFor(second));
        Assert.True(s.Queue.WaitDrained(Drain));
        try
        {
            Assert.Equal(StoreProblems.LedgerUnreadable, s.Problem);
            Assert.True(s.WritesDisabled);
            Assert.Equal(new AdmissionResult(false, AdmissionReasons.WritesDisabled), s.Admit(Span(third), KeyFor(third), 65, Local, true, "cpu"));
            Assert.Equal(new[] { KeyFor(first).Hex }, s.LedgerForTest!.Rows.Select(r => r.KeyHex));
            Thread.Sleep(2600); // over two worker ticks: reload attempts run and keep failing while the file is held
            Assert.Equal(StoreProblems.LedgerUnreadable, s.Problem);
        }
        finally
        {
            holder?.Dispose();
        }
        Assert.Equal(before, File.ReadAllBytes(ledgerPath));
        Assert.True(Eventually(() => s.Problem is null, TimeSpan.FromSeconds(10)));
        Assert.False(s.WritesDisabled);
        s.LedgerForTest!.BeforeRename = null;
        Admitted(s, third, KeyFor(third));
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.True(new FileChunkStore(root).Contains(KeyFor(third)));
    }

    /// <summary>S3: a ledger held open briefly at startup (sharing violation) is retried, not declared unreadable.</summary>
    [Fact]
    public async Task LedgerBrieflyLocked_AtStartup_IsRetried()
    {
        string root = NewRoot();
        var first = Start(root, new FakeHost());
        Admitted(first, Text('l'), KeyFor(Text('l')));
        Assert.True(first.Queue.WaitDrained(Drain));
        first.Dispose();
        var holder = new FileStream(ServerLedger.DefaultPath(root), FileMode.Open, FileAccess.Read, FileShare.None);
        var release = Task.Run(async () => { await Task.Delay(150); holder.Dispose(); });
        var s = Start(root, new FakeHost());
        await release;
        Assert.False(s.WritesDisabled);
        Assert.Null(s.Problem);
    }

    /// <summary>S4: the worker takes the model only after a quiet period; idle gaps shorter than it never get an encode.</summary>
    [Fact]
    public async Task QuietPeriod_ShortIdleGaps_NoEncode()
    {
        string root = NewRoot();
        var host = new FakeHost();
        var s = Start(root, host);
        // Quiet period 2.5 s against activity every 200-450 ms, observed for 4 s (longer than the quiet period, so a worker
        // that ignores the activity encodes inside the window); a correct one fails only if the runner stalls over 2 s.
        s.Queue.QuietPeriod = TimeSpan.FromSeconds(2.5);
        using var stop = new CancellationTokenSource();
        var toggler = Task.Run(async () =>
        {
            while (!stop.IsCancellationRequested)
            {
                host.Busy = true;
                Interlocked.Increment(ref host.Stamp);
                await Task.Delay(150);
                host.Busy = false;
                await Task.Delay(300); // idle gaps far shorter than the quiet period
            }
        });
        Admitted(s, Text('g'), KeyFor(Text('g')));
        await Task.Delay(4000);
        Assert.Equal(0, host.Encodes);
        stop.Cancel();
        await toggler;
        host.Busy = false;
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(1, host.Encodes);
    }

    /// <summary>S4: a request that starts and ends between two polls still restarts the quiet period (activity stamp).</summary>
    [Fact]
    public async Task QuietPeriod_RestartedByActivityBetweenPolls()
    {
        string root = NewRoot();
        var host = new FakeHost();
        var s = Start(root, host);
        // Quiet period 2.5 s against activity every 200-450 ms, observed for 4 s (longer than the quiet period, so a worker
        // that ignores the activity encodes inside the window); a correct one fails only if the runner stalls over 2 s.
        s.Queue.QuietPeriod = TimeSpan.FromSeconds(2.5);
        using var stop = new CancellationTokenSource();
        var bumper = Task.Run(async () =>
        {
            while (!stop.IsCancellationRequested)
            {
                Interlocked.Increment(ref host.Stamp); // never observed as busy, only as activity
                await Task.Delay(200);
            }
        });
        Admitted(s, Text('q'), KeyFor(Text('q')));
        await Task.Delay(4000);
        Assert.Equal(0, host.Encodes);
        stop.Cancel();
        await bumper;
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(1, host.Encodes);
    }

    /// <summary>S4: at most <c>MaxEncodesPerIdleWindow</c> encodes per idle window, then a cool-down.</summary>
    [Fact]
    public void IdleWindow_EncodeCap_ThenCooldown()
    {
        string root = NewRoot();
        var host = new FakeHost { Idle = false };
        var s = Start(root, host);
        s.Queue.QuietPeriod = TimeSpan.FromMilliseconds(50);
        s.Queue.MaxEncodesPerIdleWindow = 2;
        s.Queue.WindowCooldown = TimeSpan.FromSeconds(2);
        foreach (char c in "uvw")
            Admitted(s, Text(c), KeyFor(Text(c)));
        host.Idle = true;
        Assert.True(Eventually(() => host.Encodes >= 2, TimeSpan.FromSeconds(5)));
        Thread.Sleep(800);
        Assert.Equal(2, host.Encodes);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(3, host.Encodes);
    }

    /// <summary>S5: one namespace holds at most its share of the queue; another namespace still queues.</summary>
    [Fact]
    public void Queue_PerNamespaceShare()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost { Idle = false });
        for (int i = 0; i < ChunkEncodeQueue.MaxItemsPerNamespace; i++)
            Admitted(s, "a" + i + Text('a', 40), KeyFor("a" + i + Text('a', 40)));
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.NamespaceQueueShare),
            s.Admit(Span("over" + Text('a', 40)), KeyFor("over" + Text('a', 40)), 50, Local, true, "cpu"));
        var other = new ChunkCaller(true, "k-0123456789abcdef");
        Assert.True(s.Admit(Span("b" + Text('b', 40)), KeyFor("b" + Text('b', 40), other.Namespace), 50, other, true, "cpu").Queued);
    }

    /// <summary>S5: one request queues at most <see cref="ChunkStoreService.MaxQueuedPerRequest"/> misses.</summary>
    [Fact]
    public void PerRequest_QueueCap()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost { Idle = false });
        var spans = Enumerable.Range(0, 20)
            .Select(i => ("r" + i + Text('r', 40), false, KeyFor("r" + i + Text('r', 40)), (ChunkSpanRef?)null)).ToArray();
        var plan = s.ApplyTo(Plan(spans), Parsed("encode"), Local);
        Assert.Equal(ChunkStoreService.MaxQueuedPerRequest, Warning(plan, ChunkCacheCodes.EncodeQueued)?.Count);
        Assert.Equal(20 - ChunkStoreService.MaxQueuedPerRequest, Warning(plan, ChunkCacheCodes.EncodeSkipped)?.Count);
    }

    /// <summary>S5: one namespace cannot push out another's sightings beyond its own share; long paths are stored capped.</summary>
    [Fact]
    public void SightingTable_PerNamespaceCap_AndCappedPaths()
    {
        var t = new SightingTable(capacity: 10, perNamespaceCapacity: 4);
        t.See("b", "hb", null);
        for (int i = 0; i < 10; i++)
            t.See("a", "h" + i, null);
        Assert.Equal(1, t.CountOf("b", "hb"));
        Assert.Equal(1, t.CountOf("a", "h9"));
        Assert.Equal(0, t.CountOf("a", "h0"));
        Assert.True(t.Count <= 5);

        string p1 = "src/" + new string('x', 2000) + "/one.cs", p2 = "src/" + new string('x', 2000) + "/two.cs";
        t.See("c", "p1", p1);
        t.See("c", "p2", p2);
        string s1 = t.SourcePathOf("c", "p1")!, s2 = t.SourcePathOf("c", "p2")!;
        Assert.True(s1.Length <= SightingTable.StoredPathChars);
        Assert.StartsWith("src/xxx", s1, StringComparison.Ordinal);
        Assert.NotEqual(s1, s2);
        t.See("c", "p3", "src/short.cs");
        Assert.Equal("src/short.cs", t.SourcePathOf("c", "p3"));
    }

    /// <summary>S5: budget eviction for a namespace evicts that namespace's own rows first (one tenant cannot flush another's).</summary>
    [Fact]
    public void Eviction_PrefersTheWritingNamespacesOwnRows()
    {
        string root = NewRoot();
        var time = new ManualTime();
        string a1 = Text('a', 200), b1 = Text('b', 200), b2 = Text('c', 200);
        var s = Start(root, new FakeHost(), budget: 2 * FileBytes(a1, 4) + 64, time: time);
        var tenantA = new ChunkCaller(true, "k-aaaaaaaaaaaaaaaa");
        var tenantB = new ChunkCaller(true, "k-bbbbbbbbbbbbbbbb");
        Admitted(s, a1, KeyFor(a1, tenantA.Namespace), tenantA);
        Assert.True(s.Queue.WaitDrained(Drain));
        time.Now += TimeSpan.FromMinutes(1);
        Admitted(s, b1, KeyFor(b1, tenantB.Namespace), tenantB);
        Assert.True(s.Queue.WaitDrained(Drain));
        time.Now += TimeSpan.FromMinutes(1);
        Admitted(s, b2, KeyFor(b2, tenantB.Namespace), tenantB);
        Assert.True(s.Queue.WaitDrained(Drain));
        var store = new FileChunkStore(root);
        Assert.True(store.Contains(KeyFor(a1, tenantA.Namespace))); // the globally least recent row, kept
        Assert.False(store.Contains(KeyFor(b1, tenantB.Namespace)));
        Assert.True(store.Contains(KeyFor(b2, tenantB.Namespace)));
    }

    /// <summary>S6: chunk text holding a PEM private key is never queued or sighted, whatever its path.</summary>
    [Fact]
    public void PemPrivateKeyText_NeverQueued()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        string pem = "// config\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\n";
        Assert.True(ChunkDenyList.LooksLikeSecret(pem));
        Assert.True(ChunkDenyList.LooksLikeSecret("-----BEGIN RSA PRIVATE KEY-----\nMIIE\n"));
        Assert.False(ChunkDenyList.LooksLikeSecret("-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"));
        Assert.False(ChunkDenyList.LooksLikeSecret("// a comment about PRIVATE KEY-----\n"));
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.SecretContent),
            s.Admit(Span(pem, source: "src/a.cs"), KeyFor(pem), 60, Local, true, "cpu"));
        Assert.Equal(0, s.Sightings.CountOf(ChunkNamespace.Default, ChunkAnnotationParserTests.Sha(pem)));
    }

    /// <summary>N3: the same text twice in one request, once with <c>no_store</c>: the most restrictive occurrence wins.</summary>
    [Fact]
    public void SameTextTwice_MostRestrictiveOccurrenceWins()
    {
        string root = NewRoot();
        var s = Start(root, new FakeHost());
        string text = Text('m');
        var key = KeyFor(text);
        var applied = s.ApplyTo(Plan((text, false, key, null), (text, false, key, Span(text, noStore: true))), Parsed("encode"), Local);
        Assert.Null(Warning(applied, ChunkCacheCodes.EncodeQueued));
        Assert.Equal(2, Warning(applied, ChunkCacheCodes.EncodeSkipped)?.Count);
        Assert.Equal(0, s.Sightings.CountOf(ChunkNamespace.Default, ChunkAnnotationParserTests.Sha(text)));
    }

    /// <summary>N4: ownership is claimed before the model loads; the writer then uses that claim, and a second claim fails.</summary>
    [Fact]
    public void ClaimOwnership_BeforeLoad_ThenStartUsesIt()
    {
        string root = NewRoot();
        var options = new ChunkCacheOptions(true, ChunkCacheOptions.DefaultMaxChunks, root, null, StoreWrite: true, StoreBudgetBytes: 1L << 30);
        ChunkStoreService.ClaimOwnership(options);
        Assert.True(StoreOwnerLock.IsHeldByAnother(root));
        Assert.Throws<InvalidServerOptionException>(() => ChunkStoreService.ClaimOwnership(options));
        var s = ChunkStoreService.Start(options, new FakeHost());
        _services.Add(s);
        s.Dispose();
        Assert.False(StoreOwnerLock.IsHeldByAnother(root));
    }

    // ------------------------------------------------------------------------------------------------ re-review R1, R2, nits

    /// <summary>
    /// R1: a fresh writer on a CLI-built store (CLI entries at half the budget, empty ledger) gets one entry sized at 60% of
    /// the budget: that entry alone is dropped (<c>entry_too_large</c>), the writer is not stopped, and another tenant's
    /// normal entry is stored next.
    /// </summary>
    [Fact]
    public void FreshWriterOnCliStore_LargeEntry_DroppedNotStopped()
    {
        string root = NewRoot();
        string cli = Text('c', 200), big = Text('b', 240), small = Text('s', 40);
        var store = new FileChunkStore(root);
        store.Put(KeyFor(cli), EntryFor(cli, 4));
        WriteIndex(root, "cli", KeyFor(cli));
        var host = new FakeHost();
        var s = Start(root, host, budget: 2 * FileBytes(cli, 4));
        Assert.True(FileBytes(big, 4) > s.BudgetBytes - FileBytes(cli, 4) && FileBytes(big, 4) < s.BudgetBytes);

        Admitted(s, big, KeyFor(big));
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(StoreProblems.EntryTooLarge, s.Problem);
        Assert.False(store.Contains(KeyFor(big)));
        Assert.True(store.Contains(KeyFor(cli)));

        var other = new ChunkCaller(true, "k-0123456789abcdef");
        Assert.True(s.Admit(Span(small), KeyFor(small, other.Namespace), small.Length, other, true, "cpu").Queued);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.True(store.Contains(KeyFor(small, other.Namespace)));
        Assert.Null(s.Problem);
    }

    /// <summary>
    /// R1 reverse case: a store full of CLI entries plus one tiny server row cannot take even a minimum-size entry after
    /// evicting everything the server may evict: the writer stops (<c>budget_exhausted</c>) rather than encoding and
    /// dropping every admitted miss.
    /// </summary>
    [Fact]
    public void StoreFullOfCliEntries_TinyOwnRow_Stops()
    {
        string root = NewRoot();
        string cli = Text('c', 400), tiny = "t\n", next = Text('n', 100);
        var store = new FileChunkStore(root);
        store.Put(KeyFor(cli), EntryFor(cli, 4));
        WriteIndex(root, "cli", KeyFor(cli));
        var s = Start(root, new FakeHost(), budget: FileBytes(cli, 4) + FileBytes(tiny, 4) + 64);
        Admitted(s, tiny, KeyFor(tiny));
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.True(store.Contains(KeyFor(tiny)));

        Admitted(s, next, KeyFor(next));
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(StoreProblems.BudgetExhausted, s.Problem);
        Assert.True(store.Contains(KeyFor(tiny))); // nothing evicted for a write that could not happen
    }

    /// <summary>R1: once the entry size per token is known, a chunk that could never fit the budget is refused at admission (no encode).</summary>
    [Fact]
    public void OversizedChunk_RefusedAtAdmission_NoEncode()
    {
        string root = NewRoot();
        string first = Text('f', 100);
        var host = new FakeHost();
        var s = Start(root, host, budget: 4 * FileBytes(first, 4));
        Admitted(s, first, KeyFor(first));
        Assert.True(s.Queue.WaitDrained(Drain));
        int encodes = host.Encodes;
        string huge = Text('h', 50);
        var r = s.Admit(Span(huge), KeyFor(huge), 2_000, Local, true, "cpu"); // 2,000 tokens (under the 2,048 cap) at this size per token: far over
        Assert.Equal(new AdmissionResult(false, AdmissionReasons.OverBudget), r);
        Assert.True(s.Queue.WaitDrained(Drain));
        Assert.Equal(encodes, host.Encodes);
    }

    /// <summary>Nit: a PEM <c>PRIVATE KEY</c> anywhere on a BEGIN or END line (PGP blocks, a chunk holding only the end).</summary>
    [Fact]
    public void PemSniff_BeginOrEndLines_AnyPrivateKeyForm()
    {
        Assert.True(ChunkDenyList.LooksLikeSecret("-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF\n"));
        Assert.True(ChunkDenyList.LooksLikeSecret("Zm9vYmFy\n-----END RSA PRIVATE KEY-----\n"));
        Assert.True(ChunkDenyList.LooksLikeSecret("x\r\n-----BEGIN ENCRYPTED PRIVATE KEY-----\r\n"));
        Assert.False(ChunkDenyList.LooksLikeSecret("-----BEGIN PGP PUBLIC KEY BLOCK-----\nmQIN\n-----END PGP PUBLIC KEY BLOCK-----\n"));
        Assert.False(ChunkDenyList.LooksLikeSecret("// a PRIVATE KEY is never logged\n-----BEGIN CERTIFICATE-----\n"));
    }

    /// <summary>Nit: a long path's stored prefix never ends in half a surrogate pair.</summary>
    [Fact]
    public void StoredPath_CutsAtACharBoundary()
    {
        int cut = SightingTable.StoredPathChars - 17;
        string path = new string('a', cut - 1) + "\U0001F600" + new string('b', 100);
        string stored = SightingTable.StoredPath(path)!;
        string prefix = stored[..stored.LastIndexOf('#')];
        Assert.False(char.IsHighSurrogate(prefix[^1]));
        Assert.True(stored.Length <= SightingTable.StoredPathChars);
    }
}
