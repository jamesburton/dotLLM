using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The store lock model (Phase I Task 4, integration spec L3): the exclusive store lock (<see cref="ChunkIndex.AcquireStoreLock"/>)
/// is held only for short scopes (one put batch, one ledger or index save, one eviction pass); index-only updates take a
/// per-index lock (<see cref="ChunkIndex.AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>) instead. Lock order: an index lock before the store lock, never
/// the reverse. "Another holder" is simulated with a raw exclusive <see cref="FileStream"/> on the lock file (exactly what
/// another process holds) or with a child process that is killed (a crashed holder).
/// </summary>
public sealed class StoreLockTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private static readonly TimeSpan Short = TimeSpan.FromMilliseconds(150);
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-storelock-" + Guid.NewGuid().ToString("N"));

    public StoreLockTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    private const string Model = "gguf-sha256:test-model-a";
    private const string Ns = ChunkNamespace.Default;
    private static readonly string Producer = Key([]).Hex;

    private static ChunkKey Key(ReadOnlySpan<int> ids) => ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, ids);

    private static ChunkKvEntry Entry(int seed) => new("fp-a", [seed, seed + 1], [[1f, -2f]], [[0.5f, 4f]]);

    private string StoreLockFile => Path.Combine(_root, ChunkIndex.StoreLockFileName);

    /// <summary>What another process holding the lock file has: an exclusive handle.</summary>
    private static FileStream HoldExclusively(string path) => new(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

    private static FixedSizeChunker Fixed(int tokens) => new(new CharTokenizer(), chunkTokens: tokens);

    /// <summary>
    /// Runs <paramref name="action"/> on a thread whose execution context does not flow from the caller (another writer, as in
    /// another request or process), and returns what it threw, or null.
    /// </summary>
    private static Exception? OnOtherFlow(Action action)
    {
        Exception? caught = null;
        Thread t;
        using (ExecutionContext.SuppressFlow())
        {
            t = new Thread(() => caught = Record.Exception(action));
            t.Start();
        }
        t.Join();
        return caught;
    }

    // ------------------------------------------------------------------------------------------------ the plan's five

    [Fact]
    public void IndexOnlyUpdate_DoesNotNeedStoreLock()
    {
        FixedSizeChunker chunker = Fixed(10);
        string indexPath = ChunkIndex.DefaultPath(_root, chunker.Name, Model, Producer, Ns);
        using (HoldExclusively(StoreLockFile)) // e.g. a server's put batch or an eviction pass
        {
            using IDisposable indexLock = ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, Ns, Short);
            ChunkIndex index = ChunkIndex.Load(indexPath, Key);
            index.Update([new SourceFile("A.txt", "0123456789abcdefghij\n")], chunker);
            index.Save(indexPath);
            // ... while the store lock itself is still unavailable.
            Assert.Throws<ChunkStoreLockTimeoutException>(() => ChunkIndex.AcquireStoreLock(_root, Short).Dispose());
        }
        Assert.Equal(["A.txt"], ChunkIndex.Load(indexPath, Key).Paths);
        Assert.NotEqual(Path.GetFullPath(StoreLockFile), Path.GetFullPath(ChunkIndex.IndexLockPath(_root, chunker.Name, Model, Producer, Ns)));
    }

    [Fact]
    public void IndexLocks_DifferByNamespaceChunkerModelAndProducer_SoDifferentIndexesDoNotBlockEachOther()
    {
        FixedSizeChunker chunker = Fixed(10);
        const string otherNs = "tenant-b";
        Assert.True(ChunkNamespace.IsValid(otherNs));
        string baseLock = ChunkIndex.IndexLockPath(_root, chunker.Name, Model, Producer, Ns);
        Assert.NotEqual(baseLock, ChunkIndex.IndexLockPath(_root, chunker.Name, Model, Producer, otherNs));
        Assert.NotEqual(baseLock, ChunkIndex.IndexLockPath(_root, Fixed(20).Name, Model, Producer, Ns));
        Assert.NotEqual(baseLock, ChunkIndex.IndexLockPath(_root, chunker.Name, "gguf-sha256:test-model-b", Producer, Ns));
        Assert.NotEqual(baseLock, ChunkIndex.IndexLockPath(_root, chunker.Name, Model, Key([1]).Hex, Ns));

        // Holding one index's lock never makes another index's lock wait (a zero timeout tries once).
        using IDisposable held = ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, Ns, Short);
        using IDisposable otherNamespace = ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, otherNs, TimeSpan.Zero);
        using IDisposable otherChunker = ChunkIndex.AcquireIndexLock(_root, Fixed(20).Name, Model, Producer, Ns, TimeSpan.Zero);
    }

    [Fact]
    public void SameChunkerIndexUpdates_Serialise()
    {
        FixedSizeChunker chunker = Fixed(10);
        string lockPath = ChunkIndex.IndexLockPath(_root, chunker.Name, Model, Producer, Ns);
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), $"index-fixed-10.{ChunkIndex.ModelTag(Model)}.{Producer[..8]}.default.lock"), lockPath);

        // Deterministic: a second acquirer of the same index lock times out, with the busy exception naming the lock file.
        using (IDisposable first = ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, Ns, Short))
        {
            var sw = Stopwatch.StartNew();
            ChunkStoreLockTimeoutException ex = Assert.Throws<ChunkStoreLockTimeoutException>(
                () => ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, Ns, TimeSpan.FromMilliseconds(300)).Dispose());
            Assert.True(sw.Elapsed >= TimeSpan.FromMilliseconds(250), $"gave up after {sw.Elapsed.TotalMilliseconds:F0} ms, before its timeout");
            Assert.IsAssignableFrom<ChunkStoreBusyException>(ex);
            Assert.IsAssignableFrom<IOException>(ex);
            Assert.Equal(lockPath, ex.Path);
            Assert.Equal(TimeSpan.FromMilliseconds(300), ex.Timeout);
            Assert.Contains(lockPath, ex.Message, StringComparison.Ordinal);
        }
        using (ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, Ns, TimeSpan.Zero)) { } // released: free at once

        // Behavioural: N writers doing load-update-save of the same index under its lock lose no update.
        string indexPath = ChunkIndex.DefaultPath(_root, chunker.Name, Model, Producer, Ns);
        const int Writers = 6;
        Parallel.For(0, Writers, new ParallelOptions { MaxDegreeOfParallelism = Writers }, w =>
        {
            using IDisposable held = ChunkIndex.AcquireIndexLock(_root, chunker.Name, Model, Producer, Ns, TimeSpan.FromSeconds(30));
            ChunkIndex index = ChunkIndex.Load(indexPath, Key);
            Thread.Sleep(30); // widen the read-modify-write window
            index.Update([new SourceFile($"F{w}.txt", $"file {w} text\n")], chunker, removeMissingFiles: false);
            index.Save(indexPath);
        });
        Assert.Equal(Enumerable.Range(0, Writers).Select(w => $"F{w}.txt").Order(ChunkIndex.PathComparer), ChunkIndex.Load(indexPath, Key).Paths);
    }

    [Fact]
    public void DifferentChunkers_DoNotBlock()
    {
        using IDisposable a = ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short);
        using IDisposable b = ChunkIndex.AcquireIndexLock(_root, "symbol-512", Model, Producer, Ns, TimeSpan.Zero);
        // Neither is the store lock.
        using IDisposable store = ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero);
    }

    [Fact]
    public void EvictionPass_HoldsStoreLock_PutTimesOut()
    {
        var files = new FileChunkStore(_root);
        ChunkKey referenced = Key([1, 2]), orphan = Key([7, 7]), late = Key([9, 9]);
        files.Put(referenced, Entry(1));
        files.Put(orphan, Entry(7));
        var index = new ChunkIndex(Key);
        index.Update([new SourceFile("A.txt", "\u0001\u0002")], new IdsChunker([1, 2]));
        index.Save(ChunkIndex.DefaultPath(_root, "ids", Model, Producer, Ns));

        var locked = new LockingChunkStore(files, _root, Short);
        Exception? putFailure = null, indexLockFailure = null;
        ChunkIndex.BeforeOrphanDeletes = victims =>
        {
            Assert.Equal([orphan], victims);
            // Inside the pass: another writer's put batch waits for the store lock and times out.
            putFailure = OnOtherFlow(() => locked.Put(late, Entry(9)));
            // An index lock is still free (index-only updates are not blocked by an eviction pass).
            indexLockFailure = OnOtherFlow(() => ChunkIndex.AcquireIndexLock(_root, "ids", Model, Producer, Ns, TimeSpan.Zero).Dispose());
        };
        IReadOnlyList<ChunkKey> evicted;
        try
        {
            evicted = ChunkIndex.EvictStoreWideOrphans(files, _root, Short);
        }
        finally
        {
            ChunkIndex.BeforeOrphanDeletes = null;
        }
        Assert.Equal([orphan], evicted);
        Assert.IsType<ChunkStoreLockTimeoutException>(putFailure);
        Assert.Null(indexLockFailure);
        Assert.False(files.Contains(late)); // the timed-out put wrote nothing
        Assert.False(files.Contains(orphan));
        Assert.True(files.Contains(referenced));

        // After the pass the same put succeeds; and a pass that cannot get the lock deletes nothing.
        locked.Put(late, Entry(9));
        Assert.True(files.Contains(late));
        using (HoldExclusively(StoreLockFile))
        {
            Assert.Throws<ChunkStoreLockTimeoutException>(() => ChunkIndex.EvictStoreWideOrphans(files, _root, Short));
            Assert.Throws<ChunkStoreLockTimeoutException>(() => locked.Evict(late));
        }
        Assert.True(files.Contains(late));
    }

    [Fact]
    public void StoreLock_ReleasedOnDispose_EvenAfterException()
    {
        var boom = new InvalidOperationException("boom");
        Assert.Same(boom, Record.Exception((Action)(() =>
        {
            using IDisposable held = ChunkIndex.AcquireStoreLock(_root, Short);
            throw boom;
        })));
        using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero)) { }

        Assert.Same(boom, Record.Exception((Action)(() =>
        {
            using IDisposable held = ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short);
            throw boom;
        })));
        using (ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, TimeSpan.Zero)) { }

        // A put that throws inside the locking decorator releases the lock too.
        var failing = new LockingChunkStore(new ThrowingStore(boom), _root, Short);
        Assert.Same(boom, Record.Exception(() => failing.Put(Key([1]), Entry(1))));
        using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero)) { }

        // Dispose twice is harmless, and the flow's lock-order state is back to "no store lock held".
        IDisposable twice = ChunkIndex.AcquireStoreLock(_root, Short);
        twice.Dispose();
        twice.Dispose();
        using (ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short)) { }
    }

    // ------------------------------------------------------------------------------------------------ order, mode, crashes

    [Fact]
    public async Task LockOrder_IndexLockWhileHoldingStoreLock_IsRefused()
    {
        // The documented order is index lock, then store lock. Taking an index lock while this flow holds the store lock
        // could deadlock against a writer that holds that index lock and waits for the store lock: refused at once.
        using (ChunkIndex.AcquireStoreLock(_root, Short))
        {
            var sw = Stopwatch.StartNew();
            Assert.Throws<InvalidOperationException>(() => ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, TimeSpan.FromSeconds(10)).Dispose());
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5));
        }
        // The documented order works, and the index lock is not affected by releasing the store lock first.
        using (IDisposable index = ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short))
        {
            using (ChunkIndex.AcquireStoreLock(_root, Short)) { }
            using (ChunkIndex.AcquireStoreLock(_root, Short)) { }
        }
        // Another flow (thread) holding the store lock does not restrict this flow.
        using IDisposable elsewhere = await Task.Run(() => ChunkIndex.AcquireStoreLock(_root, Short));
        using (ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short)) { }
    }

    // ------------------------------------------------------------------------------------------------ review S2: proof by handle

    [Fact]
    public void StoreLockHandle_IsProofForItsLiveRootOnly_LedgerAndRegistry()
    {
        string other = Path.Combine(_root, "other");
        var files = new FileChunkStore(_root);
        ChunkKey k = Key([1, 2]);
        files.Put(k, Entry(1));
        ServerLedger ledger = ServerLedger.Load(_root);
        ledger.Record(k, Ns, null, 10, DateTimeOffset.UnixEpoch);
        StoreRegistry registry = StoreRegistry.Load(_root);

        using (StoreLockHandle otherLock = ChunkIndex.AcquireStoreLock(other, Short))
        {
            Assert.Throws<InvalidOperationException>(() => ledger.Save(otherLock));
            Assert.Throws<InvalidOperationException>(() => registry.Save(otherLock));
        }
        StoreLockHandle mine = ChunkIndex.AcquireStoreLock(_root + Path.DirectorySeparatorChar, Short); // same root, other spelling
        ledger.Save(mine);
        registry.Save(mine);
        mine.Dispose();
        Assert.False(mine.IsHeld);
        Assert.Throws<InvalidOperationException>(() => ledger.Save(mine));
        Assert.Throws<InvalidOperationException>(() => registry.Save(mine));
        Assert.Single(ServerLedger.Load(_root).Rows);
    }

    [Fact]
    public void LockOrder_IsPerStoreRoot()
    {
        string other = Path.Combine(_root, "other");
        using (ChunkIndex.AcquireStoreLock(_root, Short))
        {
            // Another store's index lock is not an order violation.
            using (ChunkIndex.AcquireIndexLock(other, "fixed-10", Model, Producer, Ns, Short)) { }
            Assert.Throws<InvalidOperationException>(() => ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short).Dispose());
        }
    }

    [Fact]
    public async Task HandleFromAChildFlow_IsValidProof()
    {
        // Review S2 case 1: a handle acquired in Task.Run (or an async helper) and used by the parent is still proof.
        using StoreLockHandle h = await Task.Run(() => ChunkIndex.AcquireStoreLock(_root, Short));
        StoreRegistry.Load(_root).Save(h);
        Assert.True(File.Exists(StoreRegistry.DefaultPath(_root)));
    }

    [Fact]
    public async Task ChildFlowStartedInsideAScope_IsNotRefusedAfterTheScopeEnds()
    {
        // Review S2 case 2: a child flow started inside a store-lock scope inherits it; once the handle is disposed the child
        // may take index locks (the old per-flow count kept refusing it for its lifetime).
        var go = new TaskCompletionSource();
        Task<Exception?> child;
        using (ChunkIndex.AcquireStoreLock(_root, Short))
        {
            child = Task.Run<Exception?>(async () =>
            {
                await go.Task;
                return Record.Exception(() => ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short).Dispose());
            });
        }
        go.SetResult();
        Assert.Null(await child);
    }

    [Fact]
    public async Task ChildDisposesTheParentsHandle_ParentNoLongerHoldsIt()
    {
        // Review S2 case 3: the child disposes the parent's handle; the parent's flow must not still count it (the old count
        // stayed 1, allowing registry saves without the lock and refusing index locks).
        StoreLockHandle h = ChunkIndex.AcquireStoreLock(_root, Short);
        await Task.Run(h.Dispose);
        Assert.False(h.IsHeld);
        using (ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short)) { }
        Assert.Throws<InvalidOperationException>(() => StoreRegistry.Load(_root).Save(h));
        using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero)) { } // really released
    }

    [Fact]
    public async Task LockFile_AccessDenied_IsRetriedWithinTheTimeout_ThenBusy()
    {
        // Review N5: an access-denied open (read-only attribute here; an ACL change or a delete-pending file on Windows) is
        // waited out like a sharing violation, and a timeout is the busy exception, not UnauthorizedAccessException.
        using (ChunkIndex.AcquireStoreLock(_root, Short)) { } // create the file
        File.SetAttributes(StoreLockFile, FileAttributes.ReadOnly);
        try
        {
            Assert.Throws<ChunkStoreLockTimeoutException>(() => ChunkIndex.AcquireStoreLock(_root, Short).Dispose());
            var clear = Task.Run(async () => { await Task.Delay(200); File.SetAttributes(StoreLockFile, FileAttributes.Normal); });
            using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.FromSeconds(10))) { }
            await clear;
        }
        finally
        {
            File.SetAttributes(StoreLockFile, FileAttributes.Normal);
        }
    }

    [Fact]
    public void Lock_IsExclusive_EvenAgainstReaders()
    {
        using (ChunkIndex.AcquireStoreLock(_root, Short))
        {
            Assert.ThrowsAny<IOException>(() => new FileStream(StoreLockFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete).Dispose());
        }
        string indexLock = ChunkIndex.IndexLockPath(_root, "fixed-10", Model, Producer, Ns);
        using (ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, Short))
        {
            Assert.ThrowsAny<IOException>(() => new FileStream(indexLock, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete).Dispose());
        }
    }

    [Fact]
    public void StaleLockFile_IsNotAHeldLock_AndIsNeverDeleted()
    {
        // A lock file left on disk (crash, or simply every earlier holder: lock files are never deleted) is not a held lock:
        // only an open exclusive handle is.
        File.WriteAllText(StoreLockFile, "left by a crashed writer");
        string indexLock = ChunkIndex.IndexLockPath(_root, "fixed-10", Model, Producer, Ns);
        File.WriteAllText(indexLock, "left by a crashed writer");
        using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero)) { }
        using (ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, TimeSpan.Zero)) { }
        Assert.True(File.Exists(StoreLockFile));
        Assert.True(File.Exists(indexLock));
        // Lock files are not index files: eviction and the index enumeration ignore them.
        Assert.Empty(ChunkIndex.ReferencedKeysUnder(_root));
    }

    [SkippableFact]
    public void CrashedHolder_ItsLockIsFreedByTheOs()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "Child-process lock holder uses Windows PowerShell; Linux/macOS flock semantics are not verified here.");
        foreach (string lockPath in new[] { StoreLockFile, ChunkIndex.IndexLockPath(_root, "fixed-10", Model, Producer, Ns) })
        {
            string script = $"$f = [System.IO.File]::Open('{lockPath.Replace("'", "''", StringComparison.Ordinal)}', 'OpenOrCreate', 'ReadWrite', 'None'); "
                + "[Console]::Out.WriteLine('held'); [Console]::Out.Flush(); Start-Sleep -Seconds 120";
            var psi = new ProcessStartInfo("powershell.exe")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (string a in new[] { "-NoProfile", "-NonInteractive", "-Command", script }) psi.ArgumentList.Add(a);
            using Process child = Process.Start(psi)!;
            try
            {
                Assert.Equal("held", child.StandardOutput.ReadLine());
                Assert.Throws<ChunkStoreLockTimeoutException>(() => Acquire(lockPath, Short).Dispose());
                child.Kill(entireProcessTree: true); // the holder "crashes": its handle is closed by the OS
                Assert.True(child.WaitForExit(30_000));
                using (Acquire(lockPath, TimeSpan.FromSeconds(5))) { }
                Assert.True(File.Exists(lockPath)); // the file stays; it is not a held lock
            }
            finally
            {
                if (!child.HasExited) child.Kill(entireProcessTree: true);
            }
        }

        IDisposable Acquire(string lockPath, TimeSpan timeout) => lockPath == StoreLockFile
            ? ChunkIndex.AcquireStoreLock(_root, timeout)
            : ChunkIndex.AcquireIndexLock(_root, "fixed-10", Model, Producer, Ns, timeout);
    }

    [Fact]
    public void IndexLock_RefusesChunkerNamesThatAreNotAFileNamePart()
    {
        foreach (string bad in new[] { "", "a/b", "a\\b", "x:y" })
            Assert.ThrowsAny<ArgumentException>(() => ChunkIndex.AcquireIndexLock(_root, bad, Model, Producer, Ns, Short).Dispose());
        foreach (string notIndex in new[] { "ledger-server.json", "chunkcache.json", "index-.json", "index-a.lock", ".chunkstore.lock" })
            Assert.Throws<ArgumentException>(() => ChunkIndex.AcquireIndexLockFor(Path.Combine(_root, notIndex), Short).Dispose());
        using (ChunkIndex.AcquireIndexLockFor(Path.Combine(_root, "index-symbol-64.0123456789ab.json"), Short))
            Assert.True(File.Exists(Path.Combine(_root, "index-symbol-64.0123456789ab.lock")));
        Assert.Throws<ArgumentOutOfRangeException>(() => ChunkIndex.AcquireStoreLock(_root, TimeSpan.FromSeconds(-2)));
    }

    // ------------------------------------------------------------------------------------------------ helpers

    /// <summary>
    /// The owner lock (Task 15): one holder per store root, a second try fails at once (null, no wait), a probe sees it held,
    /// another root is independent, it is released on dispose and the file is never deleted. It is not the store lock.
    /// </summary>
    [Fact]
    public void OwnerLock_OneHolderPerRoot_ProbeSeesIt_NotTheStoreLock()
    {
        Assert.False(StoreOwnerLock.IsHeldByAnother(_root)); // no file yet
        using (var owner = StoreOwnerLock.TryAcquire(_root))
        {
            Assert.NotNull(owner);
            var sw = Stopwatch.StartNew();
            Assert.Null(StoreOwnerLock.TryAcquire(_root));
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1));
            Assert.True(StoreOwnerLock.IsHeldByAnother(_root));
            string other = Path.Combine(_root, "other");
            using (var second = StoreOwnerLock.TryAcquire(other))
                Assert.NotNull(second);
            // Ownership does not hold the short-scope store lock.
            using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero)) { }
            Assert.True(owner!.IsHeld);
        }
        Assert.True(File.Exists(Path.Combine(_root, StoreOwnerLock.FileName)));
        Assert.False(StoreOwnerLock.IsHeldByAnother(_root));
        // A raw exclusive handle (what another process holds) is a holder too.
        using (new FileStream(Path.Combine(_root, StoreOwnerLock.FileName), FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            Assert.Null(StoreOwnerLock.TryAcquire(_root));
            Assert.True(StoreOwnerLock.IsHeldByAnother(_root));
        }
        using var again = StoreOwnerLock.TryAcquire(_root);
        Assert.NotNull(again);
    }

    /// <summary><c>IndexReferencedKeysUnder</c> = index keys only; <c>ReferencedKeysUnder</c> adds the ledger's (Task 15 rule (c)).</summary>
    [Fact]
    public void IndexReferencedKeys_ExcludeLedgerKeys()
    {
        var indexed = new ChunkKey(new string('a', 64));
        var ledgerOnly = new ChunkKey(new string('b', 64));
        var store = new FileChunkStore(_root);
        var entry = new ChunkKvEntry("fp", [1], [[0f]], [[0f]]);
        store.Put(indexed, entry);
        store.Put(ledgerOnly, entry);
        File.WriteAllText(Path.Combine(_root, "index-x.json"),
            "{\"formatVersion\":1,\"producer\":\"p\",\"chunker\":\"x\",\"files\":[{\"path\":\"a.cs\",\"sha256\":\"0\",\"chunks\":[{\"key\":\"" + indexed.Hex + "\"}]}]}");
        var ledger = ServerLedger.Load(_root);
        ledger.Record(ledgerOnly, ChunkNamespace.Default, null, 1, DateTimeOffset.UnixEpoch);
        using (var held = ChunkIndex.AcquireStoreLock(_root, Short))
            ledger.Save(held);

        Assert.Equal(new[] { indexed }, ChunkIndex.IndexReferencedKeysUnder(_root));
        Assert.Equal(new[] { indexed.Hex, ledgerOnly.Hex }, ChunkIndex.ReferencedKeysUnder(_root).Select(k => k.Hex).Order(StringComparer.Ordinal));
        Assert.Empty(ChunkIndex.IndexReferencedKeysUnder(Path.Combine(_root, "missing")));
    }

    /// <summary>One chunk holding the given token ids over the whole text (one char per id).</summary>
    private sealed class IdsChunker(int[] ids) : IChunker
    {
        public string Name => "ids";

        public int MaxChunkTokens => ids.Length;

        public IReadOnlyList<Chunk> Chunk(SourceFile file) =>
            [new Chunk(file.Path, 1, 1, ids) { StartOffset = 0, EndOffset = file.Text.Length }];
    }

    private sealed class ThrowingStore(Exception ex) : IChunkStore
    {
        public bool TryGet(ChunkKey key, out ChunkKvEntry entry)
        {
            entry = null!;
            return false;
        }

        public void Put(ChunkKey key, ChunkKvEntry entry) => throw ex;

        public bool Contains(ChunkKey key) => false;

        public IEnumerable<ChunkKey> Keys() => [];

        public long TotalBytes => 0;

        public void Evict(ChunkKey key) => throw ex;
    }
}
