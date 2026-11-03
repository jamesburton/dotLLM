using System.Collections.Concurrent;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;

namespace DotLLM.Server.ChunkCache;

/// <summary>Whether one miss was admitted for a background encode, and if not why (operator-facing; never sent to callers).</summary>
/// <param name="Queued">True when queued (or already queued).</param>
/// <param name="Reason">Why not (<see cref="AdmissionReasons"/>), or null.</param>
internal readonly record struct AdmissionResult(bool Queued, string? Reason);

/// <summary>Why admission refused a miss (<see cref="AdmissionResult.Reason"/>).</summary>
internal static class AdmissionReasons
{
    public const string NoStore = "no_store";
    public const string Volatile = "volatile";
    public const string DeniedPath = "denied_path";
    public const string SourceUnreadable = "source_unreadable";
    public const string Backend = "backend_not_writable";
    public const string TooLong = "too_long";
    public const string NotSeenTwice = "not_seen_twice";
    public const string IndexOnly = "index_only";
    public const string WritesDisabled = "writes_disabled";
    public const string BudgetExhausted = "budget_exhausted";
    public const string QueueFull = "queue_full";
}

/// <summary>What keeps the writer from storing (the <c>detail</c> of a <c>chunk_store_unavailable</c> warning).</summary>
internal static class StoreProblems
{
    /// <summary>The ledger exists but cannot be read: writes are disabled until restart (the file is never replaced).</summary>
    public const string LedgerUnreadable = "ledger_unreadable";

    /// <summary>The store lock was held by another writer for the whole lock timeout.</summary>
    public const string LockTimeout = "lock_timeout";

    /// <summary>The budget is reached and evicting the server's own entries cannot free enough: encoding stops.</summary>
    public const string BudgetExhausted = "budget_exhausted";

    /// <summary>The last put or ledger save failed (disk full, I/O error); nothing half-written was kept.</summary>
    public const string WriteFailed = "write_failed";
}

/// <summary>
/// The chunk store's sole writer (spec §5.10, §14 L1/L2; Phase I Task 15), started with <c>--chunk-store-write</c>. It owns
/// the store root (<see cref="StoreOwnerLock"/>; a second server on the same root fails at startup), keeps the server ledger
/// (<see cref="ServerLedger"/>), admits misses of annotated requests for a background encode (<see cref="Admit"/>), and stores
/// the entries <see cref="ChunkEncodeQueue"/>'s worker encodes, within the byte budget.
/// </summary>
/// <remarks>
/// <para>
/// <b>Admission</b> (spec §5.10, request thread, never blocking, never touching a lock file): a verified miss of an
/// authenticated caller is queued only when it is not <c>no_store</c>, not <c>volatile</c>, its <c>source.path</c> (when given)
/// is not on the deny-list (<see cref="ChunkDenyList"/>), it has been seen in at least two requests in the caller's namespace
/// (<see cref="SightingTable"/>; <c>on_miss: "encode"</c> bypasses only this rule), the writer is not stopped (unreadable
/// ledger, budget exhausted), with <c>--chunk-store-index-only</c> an index references its key, and the queue has room.
/// </para>
/// <para>
/// <b>Writing</b> (worker thread only): per entry, one store-lock scope (<see cref="ChunkIndex.AcquireStoreLock"/>, on the
/// worker thread, so its sleep-polling never runs on a request thread) holds the budget eviction, the
/// <see cref="IChunkStore.Put"/>, the ledger <see cref="ServerLedger.Record"/> and <see cref="ServerLedger.Save"/>: the
/// entry file first, then its row, so no other lock holder ever sees a row without a file, and a crash leaves at most a file
/// without a row (the one kind of orphan). The store is never wrapped in <see cref="LockingChunkStore"/> (the lock is not
/// re-entrant). A failed ledger save reloads the ledger from disk and deletes the just-written file unless an index
/// references it. A lock timeout is reported as <see cref="StoreProblems.LockTimeout"/>, never as an ordinary busy store.
/// </para>
/// <para>
/// <b>Budget</b>: the byte total of every entry file in the store (CLI entries included) must stay within
/// <c>--chunk-store-budget</c>. Over it, ledger rows go in least-recently-used order; a row's file is deleted only when no
/// index references its key (keys are content-addressed, a CLI index may share it), so only those frees count. When the
/// server's own entries cannot free enough, nothing is evicted, encoding stops (<see cref="StoreProblems.BudgetExhausted"/>)
/// and is re-checked every <see cref="BudgetRecheckInterval"/>.
/// </para>
/// <para>
/// <b>Unreadable ledger</b> at startup: writes stay disabled for the server's lifetime (the file is never saved over;
/// <see cref="LedgerUnreadableException"/> is never handled as an ordinary <see cref="IOException"/>). Inference runs;
/// responses with a miss warn <c>chunk_store_unavailable</c> (<c>ledger_unreadable</c>).
/// </para>
/// <para>Nothing here logs chunk text, hashes, keys or namespaces.</para>
/// </remarks>
internal sealed class ChunkStoreService : IChunkEncodeSink, IDisposable
{
    /// <summary>Longest chunk (tokens) encoded in the background: bounds the time a request can wait behind one encode.</summary>
    public const int MaxEncodeTokens = 2048;

    /// <summary>How often a stopped (budget-exhausted) writer re-checks the store size.</summary>
    public static readonly TimeSpan BudgetRecheckInterval = TimeSpan.FromSeconds(30);

    /// <summary>Most pending ledger touches (hits) kept between writes; more are dropped (they only refine the LRU order).</summary>
    public const int MaxPendingTouches = 4096;

    private readonly string _root;
    private readonly long _budget;
    private readonly bool _indexOnly;
    private readonly FileChunkStore _files;
    private readonly IChunkStore _store;
    private readonly StoreOwnerLock _owner;
    private readonly ServerLedger? _ledger;
    private readonly ConcurrentQueue<string> _touches = new();
    private readonly TimeProvider _time;
    private ChunkEncodeQueue? _queue;
    private int _pendingTouches;
    private long _storeBytes;
    private volatile string? _problem;
    private volatile IReadOnlySet<ChunkKey> _indexKeys = new HashSet<ChunkKey>();
    private (int Count, long Ticks, long Length) _indexStamp = (-1, 0, 0);
    private DateTimeOffset _lastBudgetCheck;
    private int _disposed;

    private ChunkStoreService(string root, long budget, bool indexOnly, FileChunkStore files, IChunkStore store,
        StoreOwnerLock owner, ServerLedger? ledger, string? problem, TimeProvider time)
    {
        _root = root;
        _budget = budget;
        _indexOnly = indexOnly;
        _files = files;
        _store = store;
        _owner = owner;
        _ledger = ledger;
        _problem = problem;
        _time = time;
    }

    /// <summary>The store root (absolute).</summary>
    public string Root => _root;

    /// <summary>The byte budget.</summary>
    public long BudgetBytes => _budget;

    /// <summary>Bytes of entry files in the store, as last counted or tracked.</summary>
    public long StoreBytes => Interlocked.Read(ref _storeBytes);

    /// <summary>Why the writer cannot store right now (<see cref="StoreProblems"/>), or null.</summary>
    public string? Problem => _problem;

    /// <summary>Whether writes are disabled for the server's lifetime (unreadable ledger).</summary>
    public bool WritesDisabled => _ledger is null;

    /// <summary>The sightings (per namespace and text hash).</summary>
    public SightingTable Sightings { get; } = new();

    /// <summary>The secrets deny-list applied to <c>source.path</c>.</summary>
    public ChunkDenyList DenyList { get; init; } = ChunkDenyList.Default;

    /// <summary>How long the worker waits for the store lock per write (tests shorten it).</summary>
    internal TimeSpan LockTimeout { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>The in-memory ledger (tests: its save hook simulates a crash or a full disk mid-save); null when writes are disabled.</summary>
    internal ServerLedger? LedgerForTest => _ledger;

    /// <summary>The worker's queue.</summary>
    internal ChunkEncodeQueue Queue => _queue!;

    /// <summary>
    /// Starts the writer for <paramref name="options"/> (<c>--chunk-store-write</c>): takes ownership of the root (created if
    /// missing), loads the ledger, counts the store, and starts the encode worker on <paramref name="host"/>.
    /// </summary>
    /// <param name="options">The chunk-cache options (store root, budget, index-only).</param>
    /// <param name="host">The model the worker encodes on.</param>
    /// <param name="store">The store to write (tests inject a faulty one); null = a <see cref="FileChunkStore"/> at the root.</param>
    /// <param name="time">Clock for ledger times; null = system.</param>
    /// <returns>The running writer.</returns>
    /// <exception cref="InvalidServerOptionException">Another server owns the store, or the root cannot be created or locked.</exception>
    public static ChunkStoreService Start(ChunkCacheOptions options, IChunkEncodeHost host, IChunkStore? store = null, TimeProvider? time = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(host);
        if (!options.StoreWrite || options.StoreRoot is null || options.StoreBudgetBytes <= 0)
            throw new ArgumentException("The chunk-store writer needs --chunk-store, --chunk-store-write and a budget.", nameof(options));
        string root = options.StoreRoot;
        StoreOwnerLock owner;
        FileChunkStore files;
        try
        {
            owner = StoreOwnerLock.TryAcquire(root)
                ?? throw new InvalidServerOptionException(
                    $"--chunk-store-write: the chunk store '{root}' is already owned by another dotllm server ({StoreOwnerLock.FileName} is held). "
                    + "One server writes a store: stop the other server, or give this one another directory (or no --chunk-store-write).");
            files = new FileChunkStore(root);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            throw new InvalidServerOptionException($"--chunk-store-write: the chunk store '{root}' cannot be created or locked ({ex.GetType().Name}: {ex.Message}).");
        }

        ServerLedger? ledger = null;
        string? problem = null;
        try
        {
            ledger = ServerLedger.Load(root);
        }
        catch (LedgerUnreadableException ex)
        {
            // Fail closed: never an empty ledger in its place (its entries would become orphans), never saved over.
            problem = StoreProblems.LedgerUnreadable;
            ServerLog.Info($"chunk-store: WRITES DISABLED. {ex.Message}");
        }

        var service = new ChunkStoreService(Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)), options.StoreBudgetBytes,
            options.StoreIndexOnly, files, store ?? files, owner, ledger, problem, time ?? TimeProvider.System);
        try
        {
            files.SweepTempFiles(TimeSpan.FromHours(1)); // a crashed writer's leftovers; never a live write (we own the store)
            service._storeBytes = files.GetStats().EntryBytes;
            if (service._indexOnly)
                service.RefreshIndexKeys(force: true);
            service._queue = new ChunkEncodeQueue(host, service);
        }
        catch
        {
            owner.Dispose();
            throw;
        }
        ServerLog.Info($"chunk-store: sole writer of '{service._root}' (budget {options.StoreBudgetBytes} bytes, {service._storeBytes} used"
            + (options.StoreIndexOnly ? ", index-only" : "") + (ledger is null ? ", writes disabled" : "") + ")");
        return service;
    }

    // ------------------------------------------------------------------------------------------------ admission

    /// <summary>
    /// Admits (or refuses) one verified miss for a background encode (see the class remarks). Request thread: never blocks
    /// on a lock file, never reads the store. Call once per request and distinct text (a sighting counts requests).
    /// </summary>
    /// <param name="span">The annotation.</param>
    /// <param name="key">Its key in the caller's namespace.</param>
    /// <param name="tokenCount">Its token count.</param>
    /// <param name="caller">The caller (authenticated; the endpoint never admits for others).</param>
    /// <param name="onMissEncode"><c>on_miss: "encode"</c>: bypass the seen-twice rule only.</param>
    /// <param name="backend">The served backend tag (only CPU entries are written until the Vulkan schema question is settled).</param>
    /// <returns>The outcome.</returns>
    public AdmissionResult Admit(ChunkSpanRef span, ChunkKey key, int tokenCount, ChunkCaller caller, bool onMissEncode, string backend)
    {
        ArgumentNullException.ThrowIfNull(span);
        ArgumentNullException.ThrowIfNull(caller);
        if (span.NoStore)
            return Refused(AdmissionReasons.NoStore);
        if (span.Volatile)
            return Refused(AdmissionReasons.Volatile);
        if (span.SourceUnreadable || span.SourcePath is { Length: > SightingTable.MaxSourcePathChars })
            return Refused(AdmissionReasons.SourceUnreadable);
        if (DenyList.IsDenied(span.SourcePath))
            return Refused(AdmissionReasons.DeniedPath);
        if (!string.Equals(backend, ChunkBackendTags.Cpu, StringComparison.Ordinal))
            return Refused(AdmissionReasons.Backend);
        if (tokenCount <= 0 || tokenCount > MaxEncodeTokens)
            return Refused(AdmissionReasons.TooLong);

        int seen = Sightings.See(caller.Namespace, span.Hash, span.SourcePath);
        if (seen < SightingTable.AdmitAfter && !onMissEncode)
            return Refused(AdmissionReasons.NotSeenTwice);
        if (_ledger is null)
            return Refused(AdmissionReasons.WritesDisabled);
        if (_problem == StoreProblems.BudgetExhausted)
            return Refused(AdmissionReasons.BudgetExhausted);
        if (_indexOnly && !_indexKeys.Contains(key))
            return Refused(AdmissionReasons.IndexOnly);
        if (_queue is null || !_queue.TryEnqueue(new ChunkEncodeItem(caller.Namespace, key, span.Text, tokenCount, span.SourcePath)))
            return Refused(AdmissionReasons.QueueFull);
        return new AdmissionResult(true, null);

        static AdmissionResult Refused(string reason) => new(false, reason);
    }

    /// <summary>
    /// Runs admission for every verified miss of <paramref name="plan"/> (once per distinct text), notes the hits as uses of
    /// their ledger rows, and returns the plan with <c>chunk_encode_queued</c> / <c>chunk_encode_skipped</c> counts, the queued
    /// tokens, and <c>chunk_store_unavailable</c> (with its detail) when the writer cannot store. Never throws.
    /// </summary>
    /// <param name="plan">The request's plan (no strict error).</param>
    /// <param name="parsed">The parse (<c>on_miss</c>).</param>
    /// <param name="caller">The caller (authenticated).</param>
    /// <returns>The plan with the admission's warnings.</returns>
    public ChunkCachePlan ApplyTo(ChunkCachePlan plan, ParsedChunkRequest parsed, ChunkCaller caller)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(parsed);
        ArgumentNullException.ThrowIfNull(caller);
        bool onMissEncode = string.Equals(parsed.OnMiss, "encode", StringComparison.Ordinal);
        int queued = 0, skipped = 0, queuedTokens = 0;
        var decided = new Dictionary<string, AdmissionResult>(StringComparer.Ordinal);
        foreach (var s in plan.Spans)
        {
            if (s.Key is not { } key || s.Span is not { } span)
                continue;
            if (s.Hit)
            {
                Touch(key);
                continue;
            }
            if (!decided.TryGetValue(key.Hex, out var result))
            {
                result = Admit(span, key, s.TokenCount, caller, onMissEncode, plan.Backend);
                decided[key.Hex] = result;
                if (result.Queued)
                    queuedTokens = (int)Math.Min(int.MaxValue, (long)queuedTokens + s.TokenCount);
            }
            if (result.Queued) queued++;
            else skipped++;
        }
        if (queued + skipped == 0)
            return plan;

        var warnings = plan.Warnings.ToList();
        if (_problem is { } problem)
            AddWarning(warnings, ChunkCacheCodes.StoreUnavailable, 1, problem);
        if (queued > 0)
            AddWarning(warnings, ChunkCacheCodes.EncodeQueued, queued, null);
        if (skipped > 0)
            AddWarning(warnings, ChunkCacheCodes.EncodeSkipped, skipped, null);
        return plan with { Warnings = warnings, EncodeQueuedTokens = queuedTokens };
    }

    private static void AddWarning(List<ChunkWarning> warnings, string code, int count, string? detail)
    {
        int i = warnings.FindIndex(w => w.Code == code);
        if (i >= 0)
            warnings[i] = warnings[i] with { Count = (int)Math.Min(int.MaxValue, (long)warnings[i].Count + count) };
        else
            warnings.Add(new ChunkWarning(code, count) { Detail = detail });
    }

    private void Touch(ChunkKey key)
    {
        if (_ledger is null || Interlocked.Increment(ref _pendingTouches) > MaxPendingTouches)
        {
            Interlocked.Decrement(ref _pendingTouches);
            return;
        }
        _touches.Enqueue(key.Hex);
    }

    // ------------------------------------------------------------------------------------------------ worker (sole writer)

    /// <inheritdoc/>
    public bool ShouldEncode(ChunkEncodeItem item) =>
        _ledger is not null && _problem != StoreProblems.BudgetExhausted && !_store.Contains(item.Key);

    /// <inheritdoc/>
    public void Tick()
    {
        ApplyTouches();
        if (_indexOnly)
            RefreshIndexKeys(force: false);
        if (_problem == StoreProblems.BudgetExhausted && _time.GetUtcNow() - _lastBudgetCheck >= BudgetRecheckInterval)
        {
            _lastBudgetCheck = _time.GetUtcNow();
            long bytes = _files.GetStats().EntryBytes;
            Interlocked.Exchange(ref _storeBytes, bytes);
            if (bytes < _budget)
                _problem = null; // room again (an evict, or another tool removed entries): the next miss tries
        }
    }

    /// <inheritdoc/>
    public void Write(ChunkEncodeItem item, ChunkKvEntry entry)
    {
        if (_ledger is not { } ledger)
            return;
        ChunkKey key = item.Key;
        string path = _files.PathOf(key);
        long estimate = EntryBytes(entry);

        StoreLockHandle storeLock;
        try
        {
            storeLock = ChunkIndex.AcquireStoreLock(_root, LockTimeout);
        }
        catch (ChunkStoreLockTimeoutException)
        {
            _problem = StoreProblems.LockTimeout;
            ServerLog.Info($"chunk-store: store lock timeout after {LockTimeout.TotalSeconds:0.#} s (another writer holds it); entry not stored");
            return;
        }

        using (storeLock)
        {
            ApplyTouches();
            if (_indexOnly)
            {
                // Authoritative re-check under the lock (admission used a snapshot).
                IReadOnlySet<ChunkKey> indexed;
                try { indexed = ChunkIndex.IndexReferencedKeysUnder(_root); }
                catch (InvalidDataException) { return; } // an unreadable index: nothing is provably referenced (fail closed)
                if (!indexed.Contains(key))
                    return;
            }

            bool existed = File.Exists(path);
            if (!existed && !EnsureRoom(estimate, storeLock, ledger))
                return;

            try
            {
                _store.Put(key, entry);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Disk full, a busy entry, an I/O error: the put wrote nothing loadable (temp file + rename), no row is recorded.
                _problem = ex is ChunkStoreLockTimeoutException ? StoreProblems.LockTimeout : StoreProblems.WriteFailed;
                ServerLog.Info($"chunk-store: entry not stored ({ex.GetType().Name})");
                return;
            }

            long bytes = FileLength(path) ?? estimate;
            try
            {
                ledger.Record(key, item.Namespace, SafeSourcePath(item.SourcePath), bytes, _time.GetUtcNow());
                ledger.Save(storeLock);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException)
            {
                // The row is not on disk: put memory back to what disk says, and drop the file just written unless an index
                // references its key (it would be an orphan; the next request can admit it again).
                ServerLog.Info($"chunk-store: ledger save failed ({ex.GetType().Name}); the entry was removed again");
                RollBackAfterFailedSave(key, existed);
                if (_problem != StoreProblems.LedgerUnreadable)
                    _problem = StoreProblems.WriteFailed;
                return;
            }
            if (!existed && File.Exists(path))
                Interlocked.Add(ref _storeBytes, bytes);
            if (_problem is StoreProblems.LockTimeout or StoreProblems.WriteFailed)
                _problem = null;
        }
    }

    /// <summary>
    /// Frees room for <paramref name="needed"/> bytes by budget eviction (store lock held): least-recently-used ledger rows
    /// go first; a row's file is deleted only when no index references its key. Evicts nothing when the server's own
    /// entries cannot free enough; then the writer stops (<see cref="StoreProblems.BudgetExhausted"/>).
    /// </summary>
    private bool EnsureRoom(long needed, StoreLockHandle storeLock, ServerLedger ledger)
    {
        if (StoreBytes + needed <= _budget)
            return true;
        long bytes = _files.GetStats().EntryBytes; // recount under the lock: CLI writes, earlier evictions
        Interlocked.Exchange(ref _storeBytes, bytes);
        if (bytes + needed <= _budget)
            return true;

        IReadOnlySet<ChunkKey> indexed;
        try { indexed = ChunkIndex.IndexReferencedKeysUnder(_root); }
        catch (InvalidDataException) { return Stop("an index file cannot be read; eviction cannot be decided"); }

        var victims = new List<(ChunkKey Key, long Size)>();
        long freeable = 0;
        foreach (var row in ledger.LeastRecentlyUsed())
        {
            if (bytes - freeable + needed <= _budget)
                break;
            var key = new ChunkKey(row.KeyHex);
            long size = indexed.Contains(key) ? 0 : FileLength(_files.PathOf(key)) ?? 0;
            victims.Add((key, size));
            freeable += size;
        }
        if (bytes - freeable + needed > _budget)
            return Stop("the server's own entries cannot free enough");

        foreach (var (key, _) in victims)
            ledger.Remove(key);
        try
        {
            ledger.Save(storeLock);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            ServerLog.Info($"chunk-store: ledger save failed during eviction ({ex.GetType().Name})");
            RollBackAfterFailedSave(null, true);
            if (_problem != StoreProblems.LedgerUnreadable)
                _problem = StoreProblems.WriteFailed;
            return false;
        }

        // After the save, under the same lock: what any index or the remaining ledger still references keeps its file.
        IReadOnlySet<ChunkKey> referenced;
        try { referenced = ChunkIndex.ReferencedKeysUnder(_root); }
        catch (Exception ex) when (ex is InvalidDataException or LedgerUnreadableException) { return Stop("references cannot be read"); }
        foreach (var (key, size) in victims)
        {
            if (size == 0 || referenced.Contains(key))
                continue;
            try
            {
                _store.Evict(key);
                Interlocked.Add(ref _storeBytes, -size);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Held open by a reader: the file stays (an orphan now, removable by an orphans pass).
            }
        }
        return StoreBytes + needed <= _budget || Stop("eviction freed less than expected");
    }

    private bool Stop(string why)
    {
        _problem = StoreProblems.BudgetExhausted;
        _lastBudgetCheck = _time.GetUtcNow();
        ServerLog.Info($"chunk-store: budget of {_budget} bytes reached and {why}; background encoding stops");
        return false;
    }

    // Keep the in-memory ledger equal to the file after a failed save; the file was never replaced.
    private void RollBackAfterFailedSave(ChunkKey? written, bool existed)
    {
        try
        {
            var reloaded = ServerLedger.Load(_root);
            ReplaceLedgerRows(reloaded);
        }
        catch (LedgerUnreadableException)
        {
            _problem = StoreProblems.LedgerUnreadable; // should not happen (we wrote it); never save over it
        }
        if (written is { } key && !existed)
        {
            try
            {
                if (!ChunkIndex.ReferencedKeysUnder(_root).Contains(key))
                    _store.Evict(key);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException)
            {
                // Left as an orphan (a file without a ledger row or index reference): removable by an orphans pass.
            }
        }
    }

    // The ledger instance is shared with nobody but this thread: copy the reloaded rows into it.
    private void ReplaceLedgerRows(ServerLedger reloaded)
    {
        var ledger = _ledger!;
        foreach (var row in ledger.Rows.ToList())
            ledger.Remove(new ChunkKey(row.KeyHex));
        foreach (var row in reloaded.Rows)
        {
            ledger.Record(new ChunkKey(row.KeyHex), row.Namespace, row.SourcePath, row.Bytes, row.FirstSeen);
            ledger.Touch(new ChunkKey(row.KeyHex), row.LastUsed);
        }
    }

    private void ApplyTouches()
    {
        if (_ledger is not { } ledger)
            return;
        var now = _time.GetUtcNow();
        while (_touches.TryDequeue(out string? hex))
        {
            Interlocked.Decrement(ref _pendingTouches);
            ledger.Touch(new ChunkKey(hex), now);
        }
    }

    /// <summary>Re-reads the index references when the index files changed (or <paramref name="force"/>). Worker / startup only.</summary>
    internal void RefreshIndexKeys(bool force)
    {
        (int, long, long) stamp;
        try
        {
            var infos = new DirectoryInfo(_root).EnumerateFiles("index-*.json", SearchOption.TopDirectoryOnly).ToList();
            stamp = (infos.Count, infos.Count == 0 ? 0 : infos.Max(f => f.LastWriteTimeUtc.Ticks), infos.Sum(f => f.Length));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            _indexKeys = new HashSet<ChunkKey>();
            return;
        }
        if (!force && stamp == _indexStamp)
            return;
        try
        {
            _indexKeys = ChunkIndex.IndexReferencedKeysUnder(_root);
            _indexStamp = stamp;
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or UnauthorizedAccessException)
        {
            _indexKeys = new HashSet<ChunkKey>(); // fail closed: with an unreadable index nothing is admitted as indexed
            _indexStamp = stamp;
            ServerLog.Info($"chunk-store: index references cannot be read ({ex.GetType().Name}); index-only admission refuses everything");
        }
    }

    private static string? SafeSourcePath(string? sourcePath)
    {
        if (sourcePath is null || sourcePath.Length > SightingTable.MaxSourcePathChars)
            return null;
        try
        {
            return ChunkIndex.NormalizePath(sourcePath);
        }
        catch (ArgumentException)
        {
            return null; // rooted or '..': not tree-relative, so no provenance is kept
        }
    }

    private static long EntryBytes(ChunkKvEntry entry)
    {
        long payload = (long)entry.TokenIds.Length * sizeof(int);
        for (int l = 0; l < entry.KeysPerLayer.Length; l++)
            payload += ((long)entry.KeysPerLayer[l].Length + entry.ValuesPerLayer[l].Length) * sizeof(float);
        return FileChunkStore.HeaderBytes(entry.ModelFingerprint, entry.Backend, entry.KeysPerLayer.Length) + payload;
    }

    private static long? FileLength(string path)
    {
        try
        {
            var info = new FileInfo(path);
            return info.Exists ? info.Length : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Stops the worker and releases ownership of the store.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _queue?.Dispose();
        _owner.Dispose();
    }
}
