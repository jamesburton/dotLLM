using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.Server.ChunkCache;

/// <summary>A miss admitted for a background encode (spec §5.10).</summary>
/// <param name="Namespace">The caller's namespace (server-assigned).</param>
/// <param name="Key">The key the request computed (<see cref="ServerModelIdentity.KeyOf"/>); the worker recomputes it and drops
/// the item when the served model changed meanwhile.</param>
/// <param name="Text">The chunk text, exactly as received (the worker tokenises it again with <c>ChunkTokenization.Encode</c>).</param>
/// <param name="TokenCount">Its token count at admission (for <c>tokens.encode_queued</c>).</param>
/// <param name="SourcePath">The annotation's <c>source.path</c> (the ledger row's provenance), or null.</param>
internal sealed record ChunkEncodeItem(string Namespace, ChunkKey Key, string Text, int TokenCount, string? SourcePath)
{
    /// <summary>The queue's sequence number (set on enqueue, from 1): a forget drops in-flight items queued at or before it.</summary>
    public long Seq { get; init; }

    // Never print the text, the key or the namespace (spec 10).
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append(System.Globalization.CultureInfo.InvariantCulture, $"TokenCount = {TokenCount}, TextLength = {Text.Length}");
        return true;
    }
}

/// <summary>What the encode worker runs on: the served model, entered only while the server is idle.</summary>
internal interface IChunkEncodeHost
{
    /// <summary>
    /// Non-blocking: when no request is running or waiting, takes the model (the request gate and the scheduler's model
    /// lease) and returns the lease; otherwise null at once. Requests that arrive meanwhile wait for at most one chunk encode.
    /// </summary>
    /// <returns>The lease, or null when the server is busy.</returns>
    IDisposable? TryEnterIdle();

    /// <summary>
    /// A number that changes whenever a request starts (the request gate is taken, a sequence is scheduled), so a request
    /// that starts and ends between two polls still restarts the worker's quiet period.
    /// </summary>
    /// <returns>The stamp.</returns>
    long ActivityStamp();

    /// <summary>Non-acquiring probe: whether a request is running or waiting (or a model is loading) right now.</summary>
    /// <returns>Whether the server is busy.</returns>
    bool IsBusy();

    /// <summary>
    /// Under the lease: tokenises <paramref name="item"/>'s text with <c>ChunkTokenization.Encode</c>, recomputes its key with
    /// the served model's identity, and encodes it in isolation (positions <c>0..n-1</c>) on the server's model and backend.
    /// </summary>
    /// <param name="item">The item.</param>
    /// <returns>The entry, or null when the item no longer applies (another model is served, its key differs, the backend
    /// cannot write entries, or it is too long).</returns>
    ChunkKvEntry? Encode(ChunkEncodeItem item);
}

/// <summary>What the encode worker writes to (<see cref="ChunkStoreService"/>).</summary>
internal interface IChunkEncodeSink
{
    /// <summary>Before waiting for an idle server: whether the item is still worth encoding (writes enabled, key not stored).</summary>
    bool ShouldEncode(ChunkEncodeItem item);

    /// <summary>Stores an encoded entry (store lock, eviction, put, ledger row and save in one scope).</summary>
    void Write(ChunkEncodeItem item, ChunkKvEntry entry);

    /// <summary>Periodic work on the worker thread (about once a second): refresh index references, re-check the budget.</summary>
    void Tick();
}

/// <summary>
/// The sole writer's queue and worker (Phase I Task 15): a bounded queue of admitted misses and <b>one</b> dedicated thread that
/// encodes them while the server is idle and hands each entry to the sink. Request threads only enqueue (never block, never
/// take a store lock); every store write happens on the worker thread.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><b>Bounded:</b> at most <see cref="MaxItems"/> items (the one being encoded included), at most
/// <see cref="MaxItemsPerNamespace"/> of them from one namespace, and <see cref="MaxQueuedChars"/> characters of text; an item
/// already queued (same namespace and key) is not added twice. A refusal is <c>chunk_encode_skipped</c>.</item>
/// <item><b>Never starves inference:</b> the worker takes the model only after the server has been quiet (not busy, no new
/// activity) for <see cref="QuietPeriod"/> (<see cref="DefaultQuietPeriod"/>, 2 s), encodes one chunk, releases the model
/// before any disk I/O, and after <see cref="MaxEncodesPerIdleWindow"/> encodes in one quiet window waits
/// <see cref="WindowCooldown"/>. A request that arrives during an encode waits for that one encode (at most
/// <c>ChunkStoreService.MaxEncodeTokens</c> tokens).</item>
/// <item><b>Shutdown:</b> <see cref="Dispose"/> cancels the worker, which stops between steps (an encode in progress cannot be
/// interrupted: the model's forward takes no cancellation) and waits up to <see cref="ShutdownTimeout"/>. When the worker is
/// still encoding after that, <see cref="WorkerExited"/> stays false and the owner must not free what the encode uses (the
/// model, the store's ownership); <c>onExit</c> runs on the worker thread when it finally leaves.</item>
/// <item>The worker thread starts with no execution context (<see cref="Thread.UnsafeStart()"/>), so it never inherits a
/// store-lock scope of the flow that created it (lock-order check, <see cref="StoreLockHandle"/> remarks).</item>
/// </list>
/// </remarks>
internal sealed class ChunkEncodeQueue : IDisposable
{
    /// <summary>Most queued items.</summary>
    public const int MaxItems = 256;

    /// <summary>Most queued items of one namespace (its share of <see cref="MaxItems"/>).</summary>
    public const int MaxItemsPerNamespace = 64;

    /// <summary>Most queued text, in UTF-16 chars.</summary>
    public const long MaxQueuedChars = 16L * 1024 * 1024;

    /// <summary>The server must have been quiet this long before the worker takes the model.</summary>
    public static readonly TimeSpan DefaultQuietPeriod = TimeSpan.FromSeconds(2);

    /// <summary>How often a busy server is re-checked for idleness.</summary>
    public static readonly TimeSpan IdlePollInterval = TimeSpan.FromMilliseconds(25);

    private static readonly TimeSpan TickInterval = TimeSpan.FromSeconds(1);

    private readonly object _gate = new();
    private readonly Queue<ChunkEncodeItem> _items = new();
    private readonly HashSet<(string, string)> _queued = [];
    private readonly Dictionary<string, int> _perNamespace = new(StringComparer.Ordinal);
    private readonly IChunkEncodeHost _host;
    private readonly IChunkEncodeSink _sink;
    private readonly Action? _onExit;
    private readonly CancellationTokenSource _cts = new();
    private readonly Thread _thread;
    private long _chars;
    private long _seq;
    private ChunkEncodeItem? _current;
    private int _inFlight;
    private int _disposed;
    private int _windowEncodes;
    private long _windowStamp = long.MinValue;

    /// <summary>Creates the queue and starts its worker thread.</summary>
    /// <param name="host">The model the worker encodes on.</param>
    /// <param name="sink">Where entries go.</param>
    /// <param name="onExit">Runs on the worker thread as it exits (after <see cref="Dispose"/>), or null.</param>
    public ChunkEncodeQueue(IChunkEncodeHost host, IChunkEncodeSink sink, Action? onExit = null)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
        _sink = sink ?? throw new ArgumentNullException(nameof(sink));
        _onExit = onExit;
        _thread = new Thread(Run) { IsBackground = true, Name = "dotllm chunk-store writer", Priority = ThreadPriority.BelowNormal };
        _thread.UnsafeStart();
    }

    /// <summary>Quiet period before each encode (a fixed default; tests shorten it).</summary>
    internal TimeSpan QuietPeriod { get; set; } = DefaultQuietPeriod;

    /// <summary>Most encodes in one quiet window before a cool-down.</summary>
    internal int MaxEncodesPerIdleWindow { get; set; } = 4;

    /// <summary>Cool-down after <see cref="MaxEncodesPerIdleWindow"/> encodes in one quiet window.</summary>
    internal TimeSpan WindowCooldown { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>How long <see cref="Dispose"/> waits for the worker.</summary>
    internal TimeSpan ShutdownTimeout { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>Whether the worker thread has exited.</summary>
    public bool WorkerExited => !_thread.IsAlive;

    /// <summary>Items queued or being encoded.</summary>
    public int Pending
    {
        get { lock (_gate) return _items.Count + _inFlight; }
    }

    /// <summary>Adds an item.</summary>
    /// <param name="item">The item.</param>
    /// <returns>True when queued (or already queued); false when refused.</returns>
    public bool TryEnqueue(ChunkEncodeItem item) => TryEnqueue(item, out _);

    /// <summary>Adds an item; <paramref name="refusal"/> says why not (<see cref="AdmissionReasons"/>).</summary>
    /// <param name="item">The item.</param>
    /// <param name="refusal">Null when queued; else <see cref="AdmissionReasons.QueueFull"/> or
    /// <see cref="AdmissionReasons.NamespaceQueueShare"/>.</param>
    /// <returns>True when queued (or already queued).</returns>
    public bool TryEnqueue(ChunkEncodeItem item, out string? refusal)
    {
        ArgumentNullException.ThrowIfNull(item);
        refusal = null;
        lock (_gate)
        {
            if (_disposed != 0)
            {
                refusal = AdmissionReasons.QueueFull;
                return false;
            }
            if (_queued.Contains((item.Namespace, item.Key.Hex)))
                return true;
            _perNamespace.TryGetValue(item.Namespace, out int mine);
            if (mine >= MaxItemsPerNamespace)
            {
                refusal = AdmissionReasons.NamespaceQueueShare;
                return false;
            }
            if (_items.Count + _inFlight >= MaxItems || _chars + item.Text.Length > MaxQueuedChars)
            {
                refusal = AdmissionReasons.QueueFull;
                return false;
            }
            _items.Enqueue(item with { Seq = ++_seq });
            _queued.Add((item.Namespace, item.Key.Hex));
            _perNamespace[item.Namespace] = mine + 1;
            _chars += item.Text.Length;
            Monitor.PulseAll(_gate);
            return true;
        }
    }

    /// <summary>The item the worker is on (taken from the queue, not yet written or dropped), or null.</summary>
    public ChunkEncodeItem? Current
    {
        get { lock (_gate) return _current; }
    }

    /// <summary>The sequence number of the last item queued (0 = none yet).</summary>
    public long LastSeq
    {
        get { lock (_gate) return _seq; }
    }

    /// <summary>
    /// Drops (or, with <paramref name="countOnly"/>, counts) the queued items <paramref name="match"/> selects (forget, Task
    /// 16). The item being encoded is not affected here: its write checks the forget fence (<see cref="LastSeq"/>).
    /// </summary>
    /// <param name="match">Selects an item.</param>
    /// <param name="countOnly">Count without dropping.</param>
    /// <returns>How many queued items matched.</returns>
    public int Remove(Func<ChunkEncodeItem, bool> match, bool countOnly = false)
    {
        ArgumentNullException.ThrowIfNull(match);
        lock (_gate)
        {
            var hits = _items.Where(match).ToList();
            if (countOnly || hits.Count == 0)
                return hits.Count;
            var keep = _items.Where(i => !match(i)).ToList();
            _items.Clear();
            foreach (var i in keep)
                _items.Enqueue(i);
            foreach (var item in hits)
            {
                _chars -= item.Text.Length;
                _queued.Remove((item.Namespace, item.Key.Hex));
                if (_perNamespace.TryGetValue(item.Namespace, out int n))
                {
                    if (n <= 1) _perNamespace.Remove(item.Namespace);
                    else _perNamespace[item.Namespace] = n - 1;
                }
            }
            Monitor.PulseAll(_gate);
            return hits.Count;
        }
    }

    /// <summary>Waits until nothing is queued or being encoded (tests and shutdown diagnostics).</summary>
    /// <param name="timeout">How long to wait.</param>
    /// <returns>Whether the queue drained in time.</returns>
    public bool WaitDrained(TimeSpan timeout)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        lock (_gate)
        {
            while (_items.Count + _inFlight > 0)
            {
                var left = timeout - clock.Elapsed;
                if (left <= TimeSpan.Zero || !Monitor.Wait(_gate, left))
                    return _items.Count + _inFlight == 0;
            }
            return true;
        }
    }

    private void Run()
    {
        try
        {
            var ct = _cts.Token;
            var sinceTick = System.Diagnostics.Stopwatch.StartNew();
            bool firstTick = true;
            while (!ct.IsCancellationRequested)
            {
                if (firstTick || sinceTick.Elapsed >= TickInterval)
                {
                    firstTick = false;
                    sinceTick.Restart();
                    Guard(_sink.Tick);
                }

                ChunkEncodeItem? item;
                lock (_gate)
                {
                    if (_items.Count == 0)
                    {
                        Monitor.Wait(_gate, TickInterval);
                        continue;
                    }
                    item = _items.Dequeue();
                    _chars -= item.Text.Length;
                    _inFlight++;
                    _current = item;
                }
                try
                {
                    Process(item, ct);
                }
                finally
                {
                    lock (_gate)
                    {
                        _inFlight--;
                        _current = null;
                        _queued.Remove((item.Namespace, item.Key.Hex));
                        if (_perNamespace.TryGetValue(item.Namespace, out int n))
                        {
                            if (n <= 1) _perNamespace.Remove(item.Namespace);
                            else _perNamespace[item.Namespace] = n - 1;
                        }
                        Monitor.PulseAll(_gate);
                    }
                }
            }
            lock (_gate)
            {
                _items.Clear();
                _queued.Clear();
                _perNamespace.Clear();
                _chars = 0;
                Monitor.PulseAll(_gate);
            }
        }
        finally
        {
            if (_onExit is { } exit)
                Guard(exit);
        }
    }

    private void Process(ChunkEncodeItem item, CancellationToken ct)
    {
        bool wanted = false;
        Guard(() => wanted = _sink.ShouldEncode(item));
        if (!wanted)
            return;

        IDisposable? lease = WaitForQuietModel(ct);
        if (lease is null)
            return;

        ChunkKvEntry? entry = null;
        try
        {
            Guard(() => entry = _host.Encode(item));
        }
        finally
        {
            Guard(lease.Dispose); // the model is free again before any disk I/O (never throws out of the worker)
        }
        if (entry is not null && !ct.IsCancellationRequested)
            Guard(() => _sink.Write(item, entry));
    }

    /// <summary>
    /// Waits on this thread until the server has been quiet (not busy, activity stamp unchanged) for <see cref="QuietPeriod"/>,
    /// honouring the per-window encode cap, then takes the model. Null when cancelled.
    /// </summary>
    private IDisposable? WaitForQuietModel(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            long stamp = 0;
            bool busy = true;
            Guard(() => { stamp = _host.ActivityStamp(); busy = _host.IsBusy(); });
            if (stamp != _windowStamp)
            {
                // New activity since the last encode window: a fresh window (and a fresh quiet period).
                _windowStamp = stamp;
                _windowEncodes = 0;
            }
            if (_windowEncodes >= MaxEncodesPerIdleWindow)
            {
                if (ct.WaitHandle.WaitOne(WindowCooldown))
                    return null;
                _windowEncodes = 0;
                continue;
            }
            if (busy)
            {
                if (ct.WaitHandle.WaitOne(IdlePollInterval))
                    return null;
                continue;
            }

            // Quiet so far: watch for the quiet period, restarting on any busy probe or new activity.
            var quietClock = System.Diagnostics.Stopwatch.StartNew(); // monotonic: a wall-clock step never shortens it
            bool quiet = true;
            while (quietClock.Elapsed < QuietPeriod)
            {
                if (ct.WaitHandle.WaitOne(IdlePollInterval))
                    return null;
                long now = 0;
                bool nowBusy = true;
                Guard(() => { now = _host.ActivityStamp(); nowBusy = _host.IsBusy(); });
                if (nowBusy || now != stamp)
                {
                    quiet = false;
                    break;
                }
            }
            if (!quiet)
                continue;

            IDisposable? lease = null;
            Guard(() => lease = _host.TryEnterIdle());
            if (lease is not null)
            {
                _windowEncodes++;
                return lease;
            }
            if (ct.WaitHandle.WaitOne(IdlePollInterval))
                return null;
        }
        return null;
    }

    // The worker never dies: a failure is logged by type only (no text, key or namespace) and the item is dropped.
    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            ServerLog.Info($"chunk-store: background encode step failed ({ex.GetType().Name}); item dropped");
        }
    }

    /// <summary>
    /// Stops the worker (it finishes the step it is on) and drops what is queued; waits up to <see cref="ShutdownTimeout"/>.
    /// </summary>
    /// <returns>Whether the worker has exited (false: it is still inside an encode).</returns>
    public bool Stop()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _cts.Cancel();
            lock (_gate)
                Monitor.PulseAll(_gate);
        }
        if (Thread.CurrentThread == _thread)
            return false;
        return _thread.Join(ShutdownTimeout);
    }

    /// <summary>Stops the worker (see <see cref="Stop"/>).</summary>
    public void Dispose()
    {
        // The token source is left to the finalizer: a worker that has not stopped may still read it.
        Stop();
    }
}
