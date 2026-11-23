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
    public const string NamespaceQueueShare = "namespace_queue_share";
    public const string RequestCap = "request_cap";
    public const string SecretContent = "secret_content";
    public const string OverBudget = "over_budget";
    public const string TooLargeForCpu = "too_large_for_cpu";

    /// <summary>Prewarm: the call's <c>max_tokens</c> was reached.</summary>
    public const string MaxTokens = "max_tokens";

    /// <summary>Prewarm: the text has no tokens.</summary>
    public const string Empty = "empty";

    /// <summary>
    /// With <c>--chunk-store-namespace-budget</c>: the caller's namespace is blocked because the store is full of other
    /// namespaces' entries it may not evict (the budget is a cap, not a reservation); refused before any encode until
    /// space frees (also the <c>chunk_encode_skipped</c> warning's <c>detail</c>).
    /// </summary>
    public const string NamespaceBudgetBlocked = "namespace_budget_blocked";
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

    /// <summary>An encoded entry was larger than the budget, or than what the server's own entries could free: that entry
    /// alone was dropped; the writer keeps running.</summary>
    public const string EntryTooLarge = "entry_too_large";

    /// <summary>The last put or ledger save failed (disk full, I/O error); nothing half-written was kept.</summary>
    public const string WriteFailed = "write_failed";

    /// <summary>Evict only: an index file cannot be read, so nothing is provably unreferenced (nothing was deleted).</summary>
    public const string IndexUnreadable = "index_unreadable";
}

/// <summary>What <c>POST /v1/chunk-cache/evict</c> removes (spec §7.2).</summary>
internal enum EvictMode
{
    /// <summary>Entry files that neither an index nor the server ledger references (store-wide).</summary>
    Orphans,

    /// <summary>Forget the entries whose ledger provenance matches a glob, in one namespace.</summary>
    Path,

    /// <summary>Forget one namespace.</summary>
    Namespace,

    /// <summary>Forget one namespace (a non-admin's own), or with no namespace the whole store (admin).</summary>
    All,
}

/// <summary>One evict call, resolved by the endpoint (namespaces are server-assigned; a body namespace only for an admin).</summary>
/// <param name="Mode">The mode.</param>
/// <param name="Namespace">The namespace acted on; null only for <see cref="EvictMode.Orphans"/> and a store-wide <see cref="EvictMode.All"/>.</param>
/// <param name="Glob">The path glob (<see cref="EvictMode.Path"/> only).</param>
/// <param name="DryRun">Count only (the default of the route).</param>
internal sealed record EvictRequest(EvictMode Mode, string? Namespace, string? Glob, bool DryRun)
{
    // Never print the namespace or the glob (spec §10).
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append(System.Globalization.CultureInfo.InvariantCulture, $"Mode = {Mode}, DryRun = {DryRun}");
        return true;
    }
}

/// <summary>What an evict did (or, in a dry run, would do).</summary>
/// <param name="Evicted">Entry files deleted.</param>
/// <param name="FreedBytes">Their bytes.</param>
/// <param name="KeptReferenced">Entry files of forgotten keys kept because an index (or a remaining ledger row) still references them.</param>
/// <param name="LedgerRows">Ledger rows removed.</param>
/// <param name="IndexFiles">Index files removed (the namespace's own, Task 4b names).</param>
/// <param name="Sightings">Sighting rows removed.</param>
/// <param name="Queued">Queued (not yet written) encodes dropped.</param>
/// <param name="FailedIndexFiles">Index files that could not be deleted (held open): a partial forget.</param>
/// <param name="FailedEntryFiles">Entry files that could not be deleted (held open): a partial evict.</param>
internal sealed record EvictOutcome(int Evicted, long FreedBytes, int KeptReferenced, int LedgerRows, int IndexFiles, int Sightings, int Queued,
    int FailedIndexFiles, int FailedEntryFiles)
{
    /// <summary>Whether a delete failed (the <c>chunk_evict_partial</c> warning).</summary>
    public bool Partial => FailedIndexFiles + FailedEntryFiles > 0;
}

/// <summary>An evict refused before it deleted anything (fail closed), with the <c>detail</c> of its 503.</summary>
internal sealed class EvictRefusedException(string detail, string message) : Exception(message)
{
    /// <summary><see cref="StoreProblems"/> value (<c>ledger_unreadable</c>, <c>index_unreadable</c>, <c>lock_timeout</c>, <c>write_failed</c>).</summary>
    public string Detail { get; } = detail;
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

    /// <summary>Longest chunk (tokens) encoded in the background on the CPU backend (ruling: bounds a request's wait).</summary>
    public const int MaxCpuEncodeTokens = 512;

    /// <summary>How often a stopped (budget-exhausted) writer re-checks the store size.</summary>
    public static readonly TimeSpan BudgetRecheckInterval = TimeSpan.FromSeconds(30);

    /// <summary>Most misses one request may queue.</summary>
    public const int MaxQueuedPerRequest = 16;

    /// <summary>How often a writer stopped by an unreadable ledger tries a verified reload (tests shorten it).</summary>
    internal TimeSpan ReloadRetryInterval { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>Whether the worker has exited (after <see cref="Dispose"/>; false while an encode still runs).</summary>
    public bool WorkerStopped => _queue?.WorkerExited ?? true;

    /// <summary>Most pending ledger touches (hits) kept between writes; more are dropped (they only refine the LRU order).</summary>
    public const int MaxPendingTouches = 4096;

    private readonly string _root;
    private readonly long _budget;
    private readonly bool _indexOnly;
    private readonly FileChunkStore _files;
    private readonly IChunkStore _store;
    private readonly StoreOwnerLock _owner;
    private ServerLedger? _ledger;
    private volatile bool _writesDisabled;
    private long _lastReloadTryMs = Environment.TickCount64; // monotonic: a wall-clock step never delays the retry
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

    // Every reader and writer of the in-memory ledger (the worker's Write and Tick, evict / forget from a request) holds this,
    // always BEFORE the store lock (never the reverse). Request threads wait for it asynchronously.
    private readonly SemaphoreSlim _ledgerSync = new(1, 1);

    // Forget fences (under _ledgerSync): an item queued at or before its namespace's fence (or the store-wide one) is never
    // written, so an encode in flight when its namespace was forgotten does not bring the entry back.
    private readonly Dictionary<string, long> _forgetFences = new(StringComparer.Ordinal);
    private long _forgetAllFence;

    // Per-namespace ledger figures (entries, bytes), rebuilt under _ledgerSync after every ledger change; read lock-free.
    private volatile IReadOnlyDictionary<string, (long Entries, long Bytes)> _namespaceFigures =
        new Dictionary<string, (long, long)>(StringComparer.Ordinal);

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
        _writesDisabled = ledger is null;
        _problem = problem;
        _time = time;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, StoreOwnerLock> s_claims =
        new(ChunkIndex.PathComparer);

    /// <summary>The store root (absolute).</summary>
    public string Root => _root;

    /// <summary>The byte budget.</summary>
    public long BudgetBytes => _budget;

    /// <summary>Test hook: runs on the worker in <see cref="Write"/> once the ledger gate is held, before the store lock.</summary>
    internal Action? WriteEnteredForTest { get; set; }

    // Namespaces blocked by the per-namespace budget (store full of others' rows); cleared when a minimum entry fits again.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> _blockedNamespaces = new(StringComparer.Ordinal);
    private long _lastBlockedCheckMs = Environment.TickCount64;

    // Path forgets are fenced like namespace / all forgets, by sequence number (under _ledgerSync): every item queued at or
    // before the forget that its matcher selects is forgotten, wherever it is (queued, being encoded, or just taken by the
    // worker). Checking only the queue and the current item leaves a window: the worker can take a matching item between
    // the two looks and write it back after the forget was reported.
    private readonly List<(long Seq, Func<ChunkEncodeItem, bool> Match)> _pathFences = [];

    /// <summary>Most path-forget fences kept; the oldest goes first (its items were queued long before and are done).</summary>
    internal const int MaxPathFences = 256;

    /// <summary>The per-namespace byte budget (<c>--chunk-store-namespace-budget</c>); 0 = off (one shared LRU).</summary>
    public long NamespaceBudgetBytes { get; private init; }

    /// <summary>Bytes of entry files in the store, as last counted or tracked.</summary>
    public long StoreBytes => Interlocked.Read(ref _storeBytes);

    /// <summary>Why the writer cannot store right now (<see cref="StoreProblems"/>), or null.</summary>
    public string? Problem => _problem;

    /// <summary>
    /// Whether writes are disabled because the ledger cannot be read (at startup or after a failed save): no admission, no
    /// encode, no write, the file is never written. Cleared only by a verified reload (<see cref="ReloadRetryInterval"/>).
    /// </summary>
    public bool WritesDisabled => _writesDisabled;

    /// <summary>The sightings (per namespace and text hash).</summary>
    public SightingTable Sightings { get; } = new();

    /// <summary>The secrets deny-list applied to <c>source.path</c>.</summary>
    public ChunkDenyList DenyList { get; init; } = ChunkDenyList.Default;

    /// <summary>How long the worker waits for the store lock per write (tests shorten it).</summary>
    internal TimeSpan LockTimeout { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>The in-memory ledger (tests: its save hook simulates a crash or a full disk mid-save); null until one was read.</summary>
    internal ServerLedger? LedgerForTest => _ledger;

    /// <summary>The worker's queue.</summary>
    internal ChunkEncodeQueue Queue => _queue!;

    /// <summary>
    /// Takes ownership of the store root <b>before</b> the model loads (<c>dotllm serve</c>), so a second writer fails in
    /// seconds instead of after loading a model; <see cref="Start"/> then uses the claim. No-op without
    /// <c>--chunk-store-write</c>.
    /// </summary>
    /// <param name="options">The chunk-cache options.</param>
    /// <exception cref="InvalidServerOptionException">Another server owns the store, or the root cannot be created or locked.</exception>
    public static void ClaimOwnership(ChunkCacheOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!options.Enabled || !options.StoreWrite || options.StoreRoot is not { } root)
            return;
        var owner = AcquireOwner(root);
        if (!s_claims.TryAdd(Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)), owner))
        {
            owner.Dispose();
            throw Owned(root);
        }
    }

    private static InvalidServerOptionException Owned(string root) => new(
        $"--chunk-store-write: the chunk store '{root}' is already owned by another dotllm server ({StoreOwnerLock.FileName} is held). "
        + "One server writes a store: stop the other server, or give this one another directory (or no --chunk-store-write).");

    private static StoreOwnerLock AcquireOwner(string root)
    {
        try
        {
            return StoreOwnerLock.TryAcquire(root) ?? throw Owned(root);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            throw new InvalidServerOptionException($"--chunk-store-write: the chunk store '{root}' cannot be created or locked ({ex.GetType().Name}: {ex.Message}).");
        }
    }

    /// <summary>
    /// Loads the ledger, retrying briefly when it cannot be read (a scanner or backup holding the file is a sharing
    /// violation, not corruption). Still unreadable after the retries: <see cref="LedgerUnreadableException"/>.
    /// </summary>
    private static ServerLedger LoadLedgerWithRetry(string root)
    {
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                return ServerLedger.Load(root);
            }
            catch (LedgerUnreadableException) when (attempt < 4)
            {
                Thread.Sleep(100);
            }
        }
    }

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
        if (options.StoreNamespaceBudgetBytes >= options.StoreBudgetBytes)
            throw new ArgumentException("--chunk-store-namespace-budget must be below --chunk-store-budget.", nameof(options));
        if (options.NamespaceBudgetWarning() is { } capWarning)
            ServerLog.Info(capWarning);
        string root = options.StoreRoot;
        StoreOwnerLock owner = s_claims.TryRemove(Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)), out var claimed) && claimed.IsHeld
            ? claimed
            : AcquireOwner(root);
        FileChunkStore files;
        try
        {
            files = new FileChunkStore(root);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            owner.Dispose();
            throw new InvalidServerOptionException($"--chunk-store-write: the chunk store '{root}' cannot be created ({ex.GetType().Name}: {ex.Message}).");
        }

        ServerLedger? ledger = null;
        string? problem = null;
        try
        {
            ledger = LoadLedgerWithRetry(root);
        }
        catch (LedgerUnreadableException ex)
        {
            // Fail closed: never an empty ledger in its place (its entries would become orphans), never saved over.
            problem = StoreProblems.LedgerUnreadable;
            ServerLog.Info($"chunk-store: WRITES DISABLED. {ex.Message}");
        }

        var service = new ChunkStoreService(Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)), options.StoreBudgetBytes,
            options.StoreIndexOnly, files, store ?? files, owner, ledger, problem, time ?? TimeProvider.System)
        {
            NamespaceBudgetBytes = Math.Max(0, options.StoreNamespaceBudgetBytes),
        };
        try
        {
            files.SweepTempFiles(TimeSpan.FromHours(1)); // a crashed writer's leftovers; never a live write (we own the store)
            service._storeBytes = files.GetStats().EntryBytes;
            if (ledger is not null)
                service.RefreshFigures(ledger);
            if (service._indexOnly)
                service.RefreshIndexKeys(force: true);
            service._queue = new ChunkEncodeQueue(host, service, service.OnWorkerExit);
            ChunkCacheMetrics.SetQueueDepthProvider(() => service._queue?.Pending ?? 0);
        }
        catch
        {
            owner.Dispose();
            throw;
        }
        ServerLog.Info($"chunk-store: sole writer of '{service._root}' (budget {options.StoreBudgetBytes} bytes, {service._storeBytes} used"
            + (options.StoreIndexOnly ? ", index-only" : "") + (service.NamespaceBudgetBytes > 0 ? $", {service.NamespaceBudgetBytes} bytes per namespace" : "")
            + (ledger is null ? ", writes disabled" : "") + ")");
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
    public AdmissionResult Admit(ChunkSpanRef span, ChunkKey key, int tokenCount, ChunkCaller caller, bool onMissEncode, string backend) =>
        AdmitCore(span, key, tokenCount, caller, onMissEncode, backend, prewarm: false);

    /// <summary>
    /// Admits one prewarm text (<c>POST /v1/chunk-cache/prewarm</c>, spec §7.2): every rule of <see cref="Admit"/> except the
    /// seen-twice rule, and no sighting is recorded (a prewarm is not a request that carried the text). Request thread,
    /// never blocking.
    /// </summary>
    /// <param name="span">The text as an annotation (its <c>source.path</c>, when given, is checked against the deny-list).</param>
    /// <param name="key">Its key in the caller's namespace.</param>
    /// <param name="tokenCount">Its token count.</param>
    /// <param name="caller">The caller (authenticated).</param>
    /// <param name="backend">The served backend tag.</param>
    /// <returns>The outcome.</returns>
    public AdmissionResult AdmitPrewarm(ChunkSpanRef span, ChunkKey key, int tokenCount, ChunkCaller caller, string backend) =>
        AdmitCore(span, key, tokenCount, caller, onMissEncode: true, backend, prewarm: true);

    /// <summary>Whether the store holds <paramref name="key"/> (a file probe; no lock). A hit also refreshes its ledger row's last use.</summary>
    /// <param name="key">The key.</param>
    /// <returns>Whether it is stored.</returns>
    public bool IsStored(ChunkKey key)
    {
        if (!_store.Contains(key))
            return false;
        Touch(key);
        return true;
    }

    private AdmissionResult AdmitCore(ChunkSpanRef span, ChunkKey key, int tokenCount, ChunkCaller caller, bool onMissEncode, string backend, bool prewarm)
    {
        ArgumentNullException.ThrowIfNull(span);
        ArgumentNullException.ThrowIfNull(caller);
        if (PrivacyRefusal(span) is { } privacy)
            return Refused(privacy);
        if (!string.Equals(backend, ChunkBackendTags.Cpu, StringComparison.Ordinal))
            return Refused(AdmissionReasons.Backend);
        if (tokenCount <= 0 || tokenCount > MaxEncodeTokens)
            return Refused(AdmissionReasons.TooLong);
        long perToken = Interlocked.Read(ref _bytesPerToken);
        long smallestBudget = NamespaceBudgetBytes > 0 ? Math.Min(_budget, NamespaceBudgetBytes) : _budget;
        if (perToken > 0 && Interlocked.Read(ref _headerBytes) + perToken * tokenCount > smallestBudget)
            return Refused(AdmissionReasons.OverBudget); // could never be stored: no encode is spent on it
        if (tokenCount > MaxCpuEncodeTokens) // the backend is CPU here (checked above); ruling: a request waits behind at most one 512-token encode
            return Refused(AdmissionReasons.TooLargeForCpu);

        if (!prewarm)
        {
            int seen = Sightings.See(caller.Namespace, span.Hash, span.SourcePath);
            if (seen < SightingTable.AdmitAfter && !onMissEncode)
                return Refused(AdmissionReasons.NotSeenTwice);
        }
        if (_writesDisabled)
            return Refused(AdmissionReasons.WritesDisabled);
        if (_problem == StoreProblems.BudgetExhausted)
            return Refused(AdmissionReasons.BudgetExhausted);
        if (NamespaceBudgetBytes > 0 && _blockedNamespaces.ContainsKey(caller.Namespace))
            return Refused(AdmissionReasons.NamespaceBudgetBlocked); // cap, not reservation: no encode spent and thrown away
        if (_indexOnly && !_indexKeys.Contains(key))
            return Refused(AdmissionReasons.IndexOnly);
        if (_queue is null)
            return Refused(AdmissionReasons.QueueFull);
        if (!_queue.TryEnqueue(new ChunkEncodeItem(caller.Namespace, key, span.Text, tokenCount, span.SourcePath), out string? refusal))
            return Refused(refusal ?? AdmissionReasons.QueueFull);
        return new AdmissionResult(true, null);

        static AdmissionResult Refused(string reason)
        {
            ChunkCacheMetrics.RecordSkip(reason);
            return new(false, reason);
        }
    }

    /// <summary>
    /// The privacy refusal of one occurrence (no sighting is recorded for it): <c>no_store</c>, volatile, an unreadable or
    /// over-long <c>source</c>, a deny-listed path, or secret-looking text (a PEM private key). Null when none applies.
    /// </summary>
    private string? PrivacyRefusal(ChunkSpanRef span) =>
        span.NoStore ? AdmissionReasons.NoStore
        : span.Volatile ? AdmissionReasons.Volatile
        : span.SourceUnreadable || span.SourcePath is { Length: > SightingTable.MaxSourcePathChars } ? AdmissionReasons.SourceUnreadable
        : DenyList.IsDenied(span.SourcePath) ? AdmissionReasons.DeniedPath
        : ChunkDenyList.LooksLikeSecret(span.Text) ? AdmissionReasons.SecretContent
        : null;

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
        int queued = 0, skipped = 0, queuedTokens = 0, queuedKeys = 0;
        bool tooLargeForCpu = false, blocked = false;

        // The most restrictive occurrence of each text decides (a no_store, volatile or denied occurrence refuses it).
        var deciding = new Dictionary<string, ChunkSpanRef>(StringComparer.Ordinal);
        foreach (var s in plan.Spans)
        {
            if (s.Hit || s.Key is not { } k || s.Span is not { } sp)
                continue;
            if (!deciding.TryGetValue(k.Hex, out var current) || (PrivacyRefusal(current) is null && PrivacyRefusal(sp) is not null))
                deciding[k.Hex] = sp;
        }

        var decided = new Dictionary<string, AdmissionResult>(StringComparer.Ordinal);
        foreach (var s in plan.Spans)
        {
            if (s.Key is not { } key || s.Span is null)
                continue;
            if (s.Hit)
            {
                Touch(key);
                continue;
            }
            if (!decided.TryGetValue(key.Hex, out var result))
            {
                var span = deciding[key.Hex];
                if (queuedKeys >= MaxQueuedPerRequest && PrivacyRefusal(span) is null)
                {
                    ChunkCacheMetrics.RecordSkip(AdmissionReasons.RequestCap);
                    result = new AdmissionResult(false, AdmissionReasons.RequestCap);
                }
                else
                {
                    result = Admit(span, key, s.TokenCount, caller, onMissEncode, plan.Backend);
                }
                decided[key.Hex] = result;
                if (result.Queued)
                {
                    queuedKeys++;
                    queuedTokens = (int)Math.Min(int.MaxValue, (long)queuedTokens + s.TokenCount);
                }
            }
            if (result.Queued)
            {
                queued++;
            }
            else
            {
                skipped++;
                tooLargeForCpu |= result.Reason == AdmissionReasons.TooLargeForCpu;
                blocked |= result.Reason == AdmissionReasons.NamespaceBudgetBlocked;
            }
        }
        if (queued + skipped == 0)
            return plan;

        var warnings = plan.Warnings.ToList();
        if (_problem is { } problem)
            AddWarning(warnings, ChunkCacheCodes.StoreUnavailable, 1, problem);
        if (queued > 0)
            AddWarning(warnings, ChunkCacheCodes.EncodeQueued, queued, null);
        if (skipped > 0)
            AddWarning(warnings, ChunkCacheCodes.EncodeSkipped, skipped,
                tooLargeForCpu ? AdmissionReasons.TooLargeForCpu : blocked ? AdmissionReasons.NamespaceBudgetBlocked : null);
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
        if (_writesDisabled || Interlocked.Increment(ref _pendingTouches) > MaxPendingTouches)
        {
            Interlocked.Decrement(ref _pendingTouches);
            return;
        }
        _touches.Enqueue(key.Hex);
    }

    // ------------------------------------------------------------------------------------------------ worker (sole writer)

    /// <inheritdoc/>
    public bool ShouldEncode(ChunkEncodeItem item)
    {
        if (_writesDisabled || _ledger is null || _problem == StoreProblems.BudgetExhausted
            || (NamespaceBudgetBytes > 0 && _blockedNamespaces.ContainsKey(item.Namespace)))
            return false;
        _ledgerSync.Wait();
        try
        {
            if (Forgotten(item))
                return false; // forgotten after it was queued: never encoded only to be thrown away
        }
        finally
        {
            _ledgerSync.Release();
        }
        return !_store.Contains(item.Key);
    }

    /// <inheritdoc/>
    public void Tick()
    {
        _ledgerSync.Wait();
        try
        {
            TickLocked();
        }
        finally
        {
            _ledgerSync.Release();
        }
    }

    private void TickLocked()
    {
        if (_writesDisabled && Environment.TickCount64 - _lastReloadTryMs >= (long)ReloadRetryInterval.TotalMilliseconds)
            TryVerifiedReload();
        ApplyTouches();
        if (_indexOnly)
            RefreshIndexKeys(force: false);
        if (_problem == StoreProblems.BudgetExhausted && _time.GetUtcNow() - _lastBudgetCheck >= BudgetRecheckInterval)
        {
            _lastBudgetCheck = _time.GetUtcNow();
            long bytes = _files.GetStats().EntryBytes;
            Interlocked.Exchange(ref _storeBytes, bytes);
            if (bytes + Math.Max(1, MinimumEntryBytes()) <= _budget)
                _problem = null; // room again (an evict, or another tool removed entries): the next miss tries
        }
        if (!_blockedNamespaces.IsEmpty && Environment.TickCount64 - _lastBlockedCheckMs >= (long)BudgetRecheckInterval.TotalMilliseconds)
            UnblockNamespacesIfRoom(recount: true);
    }

    /// <summary>Clears the namespace blocks when a minimum-size entry fits the store budget again.</summary>
    private void UnblockNamespacesIfRoom(bool recount)
    {
        _lastBlockedCheckMs = Environment.TickCount64;
        if (_blockedNamespaces.IsEmpty)
            return;
        long bytes = StoreBytes;
        if (recount)
        {
            bytes = _files.GetStats().EntryBytes;
            Interlocked.Exchange(ref _storeBytes, bytes);
        }
        if (bytes + Math.Max(1, MinimumEntryBytes()) <= _budget)
            _blockedNamespaces.Clear();
    }

    /// <inheritdoc/>
    public void Write(ChunkEncodeItem item, ChunkKvEntry entry)
    {
        _ledgerSync.Wait();
        try
        {
            WriteEnteredForTest?.Invoke();
            WriteLocked(item, entry);
        }
        finally
        {
            _ledgerSync.Release();
        }
    }

    /// <summary>Whether a forget of the item's namespace (or of the whole store) came after the item was queued.</summary>
    private bool Forgotten(ChunkEncodeItem item) =>
        (_forgetAllFence > 0 && item.Seq <= _forgetAllFence)
        || (_forgetFences.TryGetValue(item.Namespace, out long fence) && fence > 0 && item.Seq <= fence)
        || PathForgotten(item);

    private bool PathForgotten(ChunkEncodeItem item)
    {
        foreach ((long seq, Func<ChunkEncodeItem, bool> match) in _pathFences)
            if (item.Seq <= seq && match(item))
                return true;
        return false;
    }

    private void WriteLocked(ChunkEncodeItem item, ChunkKvEntry entry)
    {
        if (_writesDisabled || _ledger is not { } ledger)
            return;
        if (Forgotten(item))
            return; // forgotten while it was being encoded
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
            if (!existed && !EnsureRoom(estimate, MinimumEntryBytes(entry), item.Namespace, storeLock, ledger))
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
                ChunkCacheMetrics.RecordWrite("failed");
                return;
            }

            long bytes = FileLength(path) ?? estimate;
            bool hadRow = ledger.Touch(key, DateTimeOffset.MinValue); // probe only: never moves a row's last use back
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
                RollBackAfterFailedSave(key, existed, hadRow);
                if (_problem != StoreProblems.LedgerUnreadable)
                    _problem = StoreProblems.WriteFailed;
                ChunkCacheMetrics.RecordWrite("failed");
                return;
            }
            if (!existed && File.Exists(path))
                Interlocked.Add(ref _storeBytes, bytes);
            RefreshFigures(ledger);
            ChunkCacheMetrics.RecordWrite("stored");
            LearnEntrySize(entry);
            if (_problem is StoreProblems.LockTimeout or StoreProblems.WriteFailed or StoreProblems.EntryTooLarge)
                _problem = null;
        }
    }

    /// <summary>
    /// Frees room for <paramref name="needed"/> bytes by budget eviction (store lock held). The writing namespace's own
    /// ledger rows go first, then the others', each least recently used first; a row's file is deleted only when no index
    /// references its key. Nothing is evicted when that cannot make room: an entry larger than the budget, or than what the
    /// server's own rows could free, is dropped alone (<see cref="StoreProblems.EntryTooLarge"/>, the writer keeps running);
    /// only a store with nothing the server may evict stops the writer (<see cref="StoreProblems.BudgetExhausted"/>).
    /// </summary>
    private bool EnsureRoom(long needed, long minimumEntry, string ns, StoreLockHandle storeLock, ServerLedger ledger)
    {
        // Per-namespace budget (--chunk-store-namespace-budget): the writing namespace's ledger bytes must stay within it.
        long nsBudget = NamespaceBudgetBytes;
        long nsBytes = 0;
        bool nsOver = false;
        if (nsBudget > 0)
        {
            if (needed > nsBudget)
                return TooLarge();
            nsBytes = ledger.InNamespace(ns).Sum(r => r.Bytes);
            nsOver = nsBytes + needed > nsBudget;
        }
        if (!nsOver && StoreBytes + needed <= _budget)
            return true;
        if (needed > _budget)
            return TooLarge(); // before any store enumeration: it can never fit
        long bytes = _files.GetStats().EntryBytes; // recount under the lock: CLI writes, earlier evictions
        Interlocked.Exchange(ref _storeBytes, bytes);
        if (!nsOver && bytes + needed <= _budget)
            return true;

        IReadOnlySet<ChunkKey> indexed;
        try { indexed = ChunkIndex.IndexReferencedKeysUnder(_root); }
        catch (InvalidDataException) { return Stop("an index file cannot be read; eviction cannot be decided"); }

        // Candidates: the writing namespace's own rows first, least recently used first; with a per-namespace budget ONLY
        // its own rows (one tenant never evicts another's entries), else the others' rows after them (one shared LRU).
        var rows = ledger.LeastRecentlyUsed().ToList();
        var own = rows.Where(r => string.Equals(r.Namespace, ns, StringComparison.Ordinal));
        var ordered = (nsBudget > 0 ? own : own.Concat(rows.Where(r => !string.Equals(r.Namespace, ns, StringComparison.Ordinal))))
            .Select(r =>
            {
                var k = new ChunkKey(r.KeyHex);
                return (Key: k, Size: indexed.Contains(k) ? 0 : FileLength(_files.PathOf(k)) ?? 0, RowBytes: r.Bytes,
                    Own: string.Equals(r.Namespace, ns, StringComparison.Ordinal));
            })
            .ToList();
        long ownFreeable = ordered.Sum(v => v.Size);
        if (bytes - ownFreeable + needed > _budget)
        {
            // Evicting every row the server may evict would still not make room for THIS entry. The writer stops only when
            // not even a minimum-size entry would fit then (the store is genuinely full of entries the server may not
            // evict); otherwise this entry alone is dropped, whatever its size, so no caller can stop the writer for others.
            // With a per-namespace budget a namespace never stops the writer for the others: its entry is dropped.
            if (nsBudget > 0)
            {
                // Cap, not reservation: the store is full of other namespaces' rows. Block this namespace at admission until
                // a minimum entry fits again, instead of encoding and dropping each of its entries.
                if (bytes - ownFreeable + minimumEntry > _budget && _blockedNamespaces.TryAdd(ns, 0))
                    ServerLog.Info("chunk-store: a namespace is blocked by --chunk-store-namespace-budget (the store is full of other namespaces' entries)");
                return TooLarge();
            }
            return bytes - ownFreeable + minimumEntry > _budget
                ? Stop("the store is full of entries the server may not evict")
                : TooLarge();
        }

        var victims = new List<(ChunkKey Key, long Size)>();
        long freeable = 0, nsFreed = 0;
        foreach (var v in ordered)
        {
            if (bytes - freeable + needed <= _budget && (nsBudget == 0 || nsBytes - nsFreed + needed <= nsBudget))
                break;
            victims.Add((v.Key, v.Size));
            freeable += v.Size;
            if (v.Own)
                nsFreed += v.RowBytes;
        }
        if (nsBudget > 0 && nsBytes - nsFreed + needed > nsBudget)
            return TooLarge(); // cannot happen (needed <= nsBudget and every own row is a candidate); kept as a guard

        foreach (var (key, _) in victims)
            ledger.Remove(key);
        try
        {
            ledger.Save(storeLock);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            ServerLog.Info($"chunk-store: ledger save failed during eviction ({ex.GetType().Name})");
            RollBackAfterFailedSave(null, true, true);
            if (_problem != StoreProblems.LedgerUnreadable)
                _problem = StoreProblems.WriteFailed;
            return false;
        }

        RefreshFigures(ledger);

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
                ChunkCacheMetrics.RecordEviction();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Held open by a reader: the file stays (an orphan now, removable by an orphans pass).
            }
        }
        if (StoreBytes + needed <= _budget)
            return true;
        _problem = StoreProblems.WriteFailed; // a victim was held open by a reader: transient, retried by later misses
        return false;
    }

    /// <summary>Tokens of the smallest entry the stop rule considers (the default <c>min_chunk_tokens</c>).</summary>
    public const int MinimumEntryTokens = ChunkCacheWire.DefaultMinChunkTokens;

    // Bytes per token and header bytes of the entries this model writes, learned from the last STORED entry (0 = unknown);
    // admission uses them to refuse a chunk that could never fit, before any encode.
    private long _bytesPerToken;
    private long _headerBytes;

    /// <summary>The size of a <see cref="MinimumEntryTokens"/>-token entry as last learned (0 until one entry was stored).</summary>
    private long MinimumEntryBytes() => Interlocked.Read(ref _headerBytes) + MinimumEntryTokens * Interlocked.Read(ref _bytesPerToken);

    /// <summary>The size of a <see cref="MinimumEntryTokens"/>-token entry shaped like <paramref name="entry"/> (same model).</summary>
    private static long MinimumEntryBytes(ChunkKvEntry entry)
    {
        var (header, perToken) = SizeOf(entry);
        return header + MinimumEntryTokens * perToken;
    }

    private static (long Header, long PerToken) SizeOf(ChunkKvEntry entry)
    {
        long header = FileChunkStore.HeaderBytes(entry.ModelFingerprint, entry.Backend, entry.KeysPerLayer.Length);
        int tokens = Math.Max(1, entry.TokenIds.Length);
        return (header, (EntryBytes(entry) - header + tokens - 1) / tokens);
    }

    private void LearnEntrySize(ChunkKvEntry entry)
    {
        var (header, perToken) = SizeOf(entry);
        Interlocked.Exchange(ref _headerBytes, header);
        Interlocked.Exchange(ref _bytesPerToken, perToken);
    }

    private bool TooLarge()
    {
        _problem = StoreProblems.EntryTooLarge;
        ChunkCacheMetrics.RecordWrite("too_large");
        ServerLog.Info("chunk-store: an encoded entry is larger than the budget allows (or than the server's own entries could free); that entry was dropped");
        return false;
    }

    private bool Stop(string why)
    {
        _problem = StoreProblems.BudgetExhausted;
        _lastBudgetCheck = _time.GetUtcNow();
        ServerLog.Info($"chunk-store: budget of {_budget} bytes reached and {why}; background encoding stops");
        return false;
    }

    // Keep the in-memory ledger equal to the file after a failed save; the file was never replaced. When the file cannot be
    // read back (after brief retries), writes stop until a verified reload: never save over a possibly corrupt ledger.
    private void RollBackAfterFailedSave(ChunkKey? written, bool existed, bool hadRow)
    {
        try
        {
            ReplaceLedgerRows(LoadLedgerWithRetry(_root));
        }
        catch (LedgerUnreadableException ex)
        {
            if (written is { } unsaved && !hadRow)
                _ledger?.Remove(unsaved); // the row that never reached the file
            DisableWrites(ex);
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
        RefreshFigures(ledger);
    }

    private void DisableWrites(LedgerUnreadableException ex)
    {
        _writesDisabled = true;
        _problem = StoreProblems.LedgerUnreadable;
        _lastReloadTryMs = Environment.TickCount64;
        ServerLog.Info($"chunk-store: WRITES DISABLED until the ledger reads back ({ex.GetType().Name}). {ex.Message}");
    }

    /// <summary>Worker only: a disabled writer re-reads the ledger; a successful read (verified) re-enables writes.</summary>
    private void TryVerifiedReload()
    {
        _lastReloadTryMs = Environment.TickCount64;
        ServerLedger reloaded;
        try
        {
            reloaded = ServerLedger.Load(_root);
        }
        catch (LedgerUnreadableException)
        {
            return; // still unreadable: stay disabled, the file stays untouched
        }
        if (_ledger is null)
            _ledger = reloaded;
        else
            ReplaceLedgerRows(reloaded);
        RefreshFigures(_ledger);
        if (_problem == StoreProblems.LedgerUnreadable)
            _problem = null;
        _writesDisabled = false;
        ServerLog.Info("chunk-store: the ledger reads back again; writes resumed");
    }

    private void ApplyTouches()
    {
        if (_writesDisabled || _ledger is not { } ledger)
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

    // ------------------------------------------------------------------------------------------------ figures (capabilities)

    /// <summary>Rebuilds the per-namespace figures from <paramref name="ledger"/> (call under <see cref="_ledgerSync"/>).</summary>
    private void RefreshFigures(ServerLedger ledger)
    {
        var figures = new Dictionary<string, (long Entries, long Bytes)>(StringComparer.Ordinal);
        foreach (var row in ledger.Rows)
        {
            figures.TryGetValue(row.Namespace, out var f);
            figures[row.Namespace] = (f.Entries + 1, f.Bytes + row.Bytes);
        }
        _namespaceFigures = figures;
    }

    /// <summary>The ledger entries and bytes of one namespace (what a non-admin caller may see), as of the last ledger change.</summary>
    /// <param name="ns">The namespace.</param>
    /// <returns>Entries and bytes.</returns>
    public (long Entries, long Bytes) NamespaceFigures(string ns) =>
        _namespaceFigures.TryGetValue(ns, out var f) ? f : (0, 0);

    /// <summary>Store-wide figures (admin only): every entry file in the store, CLI and orphans included.</summary>
    /// <returns>Entries and bytes.</returns>
    public (long Entries, long Bytes) StoreFigures()
    {
        var stats = _files.GetStats();
        return (stats.Entries, stats.EntryBytes);
    }

    // ------------------------------------------------------------------------------------------------ evict and forget (Task 16)

    /// <summary>
    /// Evicts or forgets (spec §7.2, §10; plan Task 16). <see cref="EvictMode.Orphans"/> deletes the entry files no index and
    /// no ledger row references. The forget modes (<see cref="EvictMode.Path"/>, <see cref="EvictMode.Namespace"/>,
    /// <see cref="EvictMode.All"/>) remove, in the requested namespace (or the whole store): the ledger rows (by provenance
    /// glob for <c>path</c>), for <c>namespace</c> / <c>all</c> the namespace's own index files (Task 4b names; legacy names
    /// are never deleted), the matching sighting rows and queued encodes (the matching encode in flight is dropped at its
    /// write), and the entry files of the forgotten keys that nothing else references. A forgotten key that an index the
    /// call does not remove (a CLI index, a legacy name) still references keeps its file and is counted in
    /// <see cref="EvictOutcome.KeptReferenced"/> (Task 15 rule (c), ruling 66). No marker is kept (a later sighting starts
    /// again). A delete that fails (a file held open) is counted in <see cref="EvictOutcome.FailedEntryFiles"/> /
    /// <see cref="EvictOutcome.FailedIndexFiles"/>, never hidden.
    /// </summary>
    /// <remarks>
    /// Fail closed: with writes disabled (unreadable ledger), a ledger file on disk that does not read back (after brief,
    /// asynchronous retries; then writes are disabled as for the writer's own failed save), or an unreadable index file,
    /// nothing is deleted (<see cref="EvictRefusedException"/>), dry runs included. A dry run reads the index files and the
    /// ledger file once, before the ledger gate, so it never stalls the background writer on disk reads. Locks for a real
    /// run, in this order and all within <see cref="LockTimeout"/>, waited for asynchronously: the index locks of the index
    /// files to delete, the ledger gate, the store lock (one short scope: read and verify once, save the ledger, delete the
    /// index files, re-read the references, delete the entry files). Nothing is logged but counts.
    /// </remarks>
    /// <param name="request">The resolved request.</param>
    /// <param name="ct">Cancellation (the request's).</param>
    /// <returns>The counts.</returns>
    /// <exception cref="EvictRefusedException">Nothing was deleted, see <see cref="EvictRefusedException.Detail"/>.</exception>
    /// <exception cref="ArgumentException">The glob is empty, rooted, negated or has a <c>..</c> segment.</exception>
    public async Task<EvictOutcome> EvictAsync(EvictRequest request, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(request);
        ForgetMatcher? matcher = request.Mode == EvictMode.Path ? ForgetMatcher.Create(request.Glob!) : null;
        if (request.Mode is not EvictMode.Orphans && request.Namespace is { } named)
            ChunkNamespace.Validate(named, nameof(request));
        if (_writesDisabled)
            throw LedgerRefusal();

        var clock = System.Diagnostics.Stopwatch.StartNew();
        TimeSpan timeout = LockTimeout;
        IReadOnlyList<string> indexFiles = request.Mode is EvictMode.Namespace or EvictMode.All ? OwnIndexFiles(request.Namespace) : [];
        var indexLocks = new List<IDisposable>();
        bool synced = false;
        StoreLockHandle? storeLock = null;
        try
        {
            References? refs = null;
            if (request.DryRun)
            {
                refs = await ReadReferencesAsync(indexFiles, underGate: false, ct); // outside the ledger gate: the writer is never stalled by it
            }
            else
            {
                // Lock order: index locks, then (after the ledger gate) the store lock.
                foreach (string file in indexFiles)
                {
                    while (true)
                    {
                        try
                        {
                            indexLocks.Add(ChunkIndex.AcquireIndexLockFor(file, TimeSpan.Zero));
                            break;
                        }
                        catch (ChunkStoreLockTimeoutException) when (clock.Elapsed < timeout)
                        {
                            await Task.Delay(20, ct);
                        }
                        catch (ChunkStoreLockTimeoutException)
                        {
                            throw LockRefusal(timeout);
                        }
                    }
                }
            }
            TimeSpan left = timeout - clock.Elapsed;
            if (left < TimeSpan.Zero || !await _ledgerSync.WaitAsync(left, ct))
                throw LockRefusal(timeout);
            synced = true;
            if (_writesDisabled || _ledger is not { } ledger)
                throw LedgerRefusal();
            if (!request.DryRun)
            {
                while (true)
                {
                    try
                    {
                        storeLock = ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero);
                        break;
                    }
                    catch (ChunkStoreLockTimeoutException) when (clock.Elapsed < timeout)
                    {
                        await Task.Delay(20, ct);
                    }
                    catch (ChunkStoreLockTimeoutException)
                    {
                        throw LockRefusal(timeout);
                    }
                }
                refs = await ReadReferencesAsync(indexFiles, underGate: true, ct); // once, under the locks, before any decision
            }
            return EvictLocked(request, matcher, indexFiles, refs!, ledger, storeLock);
        }
        finally
        {
            storeLock?.Dispose();
            if (synced)
                _ledgerSync.Release();
            foreach (var l in indexLocks)
                l.Dispose();
        }
    }

    /// <summary>What the index files and the ledger file on disk reference, read once per evict.</summary>
    /// <param name="KeptByIndexes">Keys of the index files the call does not delete.</param>
    /// <param name="RemovedIndexKeys">Keys of the index files it deletes.</param>
    /// <param name="LedgerOnDisk">Keys of the ledger file.</param>
    private sealed record References(IReadOnlySet<ChunkKey> KeptByIndexes, IReadOnlySet<ChunkKey> RemovedIndexKeys, IReadOnlySet<ChunkKey> LedgerOnDisk);

    /// <summary>
    /// Reads every index file once and the ledger file (fail closed). A ledger file that does not read is retried briefly
    /// (a scanner or backup holding it is a sharing violation, not corruption), asynchronously; still unreadable, writes are
    /// disabled (as after the writer's own failed save) and the evict is refused. An unreadable index refuses the evict only.
    /// </summary>
    private async Task<References> ReadReferencesAsync(IReadOnlyList<string> indexFiles, bool underGate, CancellationToken ct)
    {
        var removing = new HashSet<string>(indexFiles, ChunkIndex.PathComparer);
        var kept = new HashSet<ChunkKey>();
        var removed = new HashSet<ChunkKey>();
        try
        {
            foreach (string file in ChunkIndex.IndexFilesUnder(_root))
                (removing.Contains(file) ? removed : kept).UnionWith(ChunkIndex.KeysOfIndexFile(file));
        }
        catch (InvalidDataException)
        {
            ServerLog.Info("chunk-store: evict refused, an index file cannot be read; nothing was evicted");
            throw new EvictRefusedException(StoreProblems.IndexUnreadable, "an index file in the store cannot be read; nothing was evicted");
        }
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                return new References(kept, removed, ServerLedger.KeysOnDisk(_root));
            }
            catch (LedgerUnreadableException) when (attempt < 4)
            {
                await Task.Delay(100, ct);
            }
            catch (LedgerUnreadableException ex)
            {
                if (underGate && !_writesDisabled)
                    DisableWrites(ex); // a real run, under the gate; a dry run never changes the writer's state
                throw LedgerRefusal();
            }
        }
    }

    private static EvictRefusedException LedgerRefusal() => new(StoreProblems.LedgerUnreadable,
        "the server ledger cannot be read; nothing was evicted (writes stay disabled until it reads back)");

    private static EvictRefusedException LockRefusal(TimeSpan timeout) => new(StoreProblems.LockTimeout,
        $"store lock timeout after {timeout.TotalSeconds:0.#} s (another writer holds the store or an index); nothing was evicted");

    /// <summary>The current-name index files of <paramref name="ns"/> (every namespace when null) directly under the root.</summary>
    private IReadOnlyList<string> OwnIndexFiles(string? ns) =>
        ChunkIndex.IndexFilesUnder(_root)
            .Where(f => ChunkIndex.TryParseIndexFileName(System.IO.Path.GetFileName(f), out var parts)
                && (ns is null || string.Equals(parts.Namespace, ns, StringComparison.Ordinal)))
            .ToList();

    private EvictOutcome EvictLocked(EvictRequest request, ForgetMatcher? matcher, IReadOnlyList<string> indexFiles, References refs,
        ServerLedger ledger, StoreLockHandle? storeLock)
    {
        bool dry = request.DryRun;
        string? ns = request.Namespace;
        IReadOnlySet<ChunkKey> keptByIndexes = refs.KeptByIndexes;
        var referencedBefore = new HashSet<ChunkKey>(refs.KeptByIndexes);
        referencedBefore.UnionWith(refs.RemovedIndexKeys);
        referencedBefore.UnionWith(refs.LedgerOnDisk);

        // What goes: ledger rows, and the keys whose entry files may go.
        List<LedgerRow> rows = request.Mode switch
        {
            EvictMode.Orphans => [],
            EvictMode.Path => ledger.InNamespace(ns!).Where(r => matcher!.Matches(r.SourcePath)).ToList(),
            _ when ns is null => ledger.Rows.ToList(),
            _ => ledger.InNamespace(ns).ToList(),
        };
        var removedRowKeys = rows.Select(r => new ChunkKey(r.KeyHex)).ToHashSet();
        IEnumerable<ChunkKey> candidates = request.Mode switch
        {
            EvictMode.Orphans => _files.Keys().Where(k => !referencedBefore.Contains(k)),
            _ when request.Mode == EvictMode.All && ns is null => _files.Keys(),
            _ => removedRowKeys.Concat(refs.RemovedIndexKeys),
        };
        var remaining = new HashSet<ChunkKey>(keptByIndexes);
        foreach (var row in ledger.Rows)
        {
            var k = new ChunkKey(row.KeyHex);
            if (!removedRowKeys.Contains(k))
                remaining.Add(k);
        }
        var victims = new List<(ChunkKey Key, long Size)>();
        int kept = 0;
        foreach (var key in candidates.Distinct())
        {
            if (FileLength(_files.PathOf(key)) is not { } size)
                continue; // no file: nothing to delete (a row without its file)
            if (remaining.Contains(key))
                kept++;
            else
                victims.Add((key, size));
        }

        Func<string, string?, bool>? sightingMatch = request.Mode switch
        {
            EvictMode.Orphans => null,
            EvictMode.Path => (rowNs, path) => string.Equals(rowNs, ns, StringComparison.Ordinal) && matcher!.Matches(path),
            _ when ns is null => (_, _) => true,
            _ => (rowNs, _) => string.Equals(rowNs, ns, StringComparison.Ordinal),
        };
        Func<ChunkEncodeItem, bool>? queueMatch = sightingMatch is null ? null : item => sightingMatch(item.Namespace, item.SourcePath);

        if (dry)
        {
            return new EvictOutcome(victims.Count, victims.Sum(v => v.Size), kept, rows.Count, indexFiles.Count,
                sightingMatch is null ? 0 : Sightings.Remove(sightingMatch, countOnly: true),
                queueMatch is null || _queue is null ? 0 : _queue.Remove(queueMatch, countOnly: true), 0, 0);
        }

        // 1. Forget the queued and in-flight items and the sightings (path: only the matching ones), then the rows (one save),
        //    before any file goes.
        int sightings = 0, queued = 0;
        if (request.Mode != EvictMode.Orphans)
        {
            if (request.Mode == EvictMode.Path)
            {
                if ((_queue?.LastSeq ?? 0) is > 0 and long pathFence)
                {
                    if (_pathFences.Count >= MaxPathFences) _pathFences.RemoveAt(0);
                    _pathFences.Add((pathFence, queueMatch!));
                }
            }
            else
            {
                long fence = _queue?.LastSeq ?? 0;
                if (ns is null)
                    _forgetAllFence = Math.Max(_forgetAllFence, fence);
                else
                    _forgetFences[ns] = Math.Max(_forgetFences.GetValueOrDefault(ns), fence);
            }
            queued = queueMatch is null || _queue is null ? 0 : _queue.Remove(queueMatch);
            sightings = Sightings.Remove(sightingMatch!);
        }
        if (rows.Count > 0)
        {
            foreach (var key in removedRowKeys)
                ledger.Remove(key);
            try
            {
                ledger.Save(storeLock!);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                ServerLog.Info($"chunk-store: evict refused, the ledger save failed ({ex.GetType().Name}); no file was deleted");
                RollBackAfterFailedSave(null, true, true);
                RefreshFigures(_ledger!);
                throw new EvictRefusedException(StoreProblems.WriteFailed,
                    $"the ledger could not be saved, so no entry or index file was deleted and the rows stay; the matching sightings and queued encodes ({sightings} and {queued}) were already dropped (they come back with new sightings)");
            }
            RefreshFigures(ledger);
        }

        // 2. The namespace's index files (their index locks are held).
        int indexDeleted = 0, indexFailed = 0;
        foreach (string file in indexFiles)
        {
            try
            {
                File.Delete(file);
                indexDeleted++;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                indexFailed++;
                ServerLog.Info($"chunk-store: an index file could not be deleted ({ex.GetType().Name}); its entries stay referenced");
            }
        }

        // 3. Entry files, decided again from what is on disk now (the saved ledger, the remaining indexes).
        IReadOnlySet<ChunkKey> referenced;
        try
        {
            referenced = ChunkIndex.ReferencedKeysUnder(_root);
        }
        catch (Exception ex) when (ex is InvalidDataException or LedgerUnreadableException)
        {
            ServerLog.Info($"chunk-store: evict stopped before deleting entry files ({ex.GetType().Name})");
            throw new EvictRefusedException(ex is LedgerUnreadableException ? StoreProblems.LedgerUnreadable : StoreProblems.IndexUnreadable,
                "the references could not be read again; rows, sightings and queued encodes were forgotten and index files deleted, but no entry file was deleted");
        }
        var ledgerKeys = ledger.Rows.Select(r => r.KeyHex).ToHashSet(StringComparer.Ordinal);
        int evicted = 0, entryFailed = 0;
        long freed = 0;
        foreach (var (key, size) in victims)
        {
            if (referenced.Contains(key) || ledgerKeys.Contains(key.Hex))
            {
                kept++;
                continue;
            }
            try
            {
                _store.Evict(key);
                evicted++;
                freed += size;
                ChunkCacheMetrics.RecordEviction();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                entryFailed++;
                ServerLog.Info($"chunk-store: an entry file could not be deleted ({ex.GetType().Name}); an orphans pass removes it later");
            }
        }
        Interlocked.Add(ref _storeBytes, -freed);
        if (evicted > 0 && _problem == StoreProblems.BudgetExhausted)
            _lastBudgetCheck = DateTimeOffset.MinValue; // re-check the budget at the next tick
        if (freed > 0)
            UnblockNamespacesIfRoom(recount: false);
        ServerLog.Info($"chunk-store: evict ({request.Mode.ToString().ToLowerInvariant()}) removed {evicted} entry file(s), {rows.Count} ledger row(s), "
            + $"{indexDeleted} index file(s), {sightings} sighting(s), {queued} queued encode(s); kept {kept} referenced"
            + (indexFailed + entryFailed > 0 ? $"; could not delete {entryFailed} entry and {indexFailed} index file(s)" : ""));
        return new EvictOutcome(evicted, freed, kept, rows.Count, indexDeleted, sightings, queued, indexFailed, entryFailed);
    }

    /// <summary>
    /// Matches a forget glob against stored provenance: the ledger's (normalised) <c>sourcePath</c> and the sighting table's
    /// capped form (<see cref="SightingTable.StoredPath"/>). A path over <see cref="SightingTable.StoredPathChars"/> characters
    /// is held as a prefix plus a hash, so the requested path is also compared in that stored form (review: the raw path would
    /// never match it).
    /// </summary>
    private sealed class ForgetMatcher
    {
        private readonly DotLLM.ChunkReuse.Tooling.PathGlob _glob;
        private readonly HashSet<string> _storedForms;

        private ForgetMatcher(DotLLM.ChunkReuse.Tooling.PathGlob glob, HashSet<string> storedForms)
        {
            _glob = glob;
            _storedForms = storedForms;
        }

        public static ForgetMatcher Create(string glob)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(glob);
            var parsed = DotLLM.ChunkReuse.Tooling.PathGlob.Parse(glob);
            if (parsed.Negated)
                throw new ArgumentException("A forget glob cannot be negated ('!').", nameof(glob));
            var forms = new HashSet<string>(ChunkIndex.PathComparer);
            foreach (string form in new[] { glob, glob.Trim(), Normal(glob.Trim()) })
                if (SightingTable.StoredPath(form) is { } stored)
                    forms.Add(stored);
            return new ForgetMatcher(parsed, forms);
        }

        public bool Matches(string? path) =>
            path is not null && (_glob.IsMatch(path) || _storedForms.Contains(path) || _storedForms.Contains(Normal(path)));

        private static string Normal(string path)
        {
            try
            {
                return ChunkIndex.NormalizePath(path);
            }
            catch (ArgumentException)
            {
                return path.Replace('\\', '/');
            }
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

    /// <summary>
    /// Stops the worker and releases ownership of the store. When the worker is still inside an encode after the queue's
    /// shutdown timeout, ownership is kept until the worker exits (it is released from the worker thread then), so no other
    /// writer can start on the store while this one may still write; callers check <see cref="WorkerStopped"/> before freeing
    /// the model.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        bool stopped = _queue?.Stop() ?? true;
        if (stopped)
            _owner.Dispose();
        else
            ServerLog.Info("chunk-store: shutdown while a background encode still runs; the store stays owned until it ends");
    }

    private void OnWorkerExit()
    {
        if (Volatile.Read(ref _disposed) != 0)
            _owner.Dispose();
    }
}
