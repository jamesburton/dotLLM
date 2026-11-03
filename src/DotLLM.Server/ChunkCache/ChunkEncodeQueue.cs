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
/// <item><b>Bounded:</b> at most <see cref="MaxItems"/> items and <see cref="MaxQueuedChars"/> characters of text; an item
/// already queued (same namespace and key) is not added twice. A full queue refuses (<c>chunk_encode_skipped</c>).</item>
/// <item><b>Never starves inference:</b> the worker encodes one chunk only after <see cref="IChunkEncodeHost.TryEnterIdle"/>
/// succeeds and releases the model straight after that one forward pass; while the server is busy it re-checks every
/// <see cref="IdlePollInterval"/> on its own thread.</item>
/// <item><b>Shutdown:</b> <see cref="Dispose"/> cancels the worker, which stops between steps (an encode in progress finishes);
/// queued items are dropped (they are only misses).</item>
/// <item>The worker thread starts with no execution context (<see cref="Thread.UnsafeStart()"/>), so it never inherits a
/// store-lock scope of the flow that created it (lock-order check, <see cref="StoreLockHandle"/> remarks).</item>
/// </list>
/// </remarks>
internal sealed class ChunkEncodeQueue : IDisposable
{
    /// <summary>Most queued items.</summary>
    public const int MaxItems = 256;

    /// <summary>Most queued text, in UTF-16 chars.</summary>
    public const long MaxQueuedChars = 16L * 1024 * 1024;

    /// <summary>How often a busy server is re-checked for idleness.</summary>
    public static readonly TimeSpan IdlePollInterval = TimeSpan.FromMilliseconds(25);

    private static readonly TimeSpan TickInterval = TimeSpan.FromSeconds(1);

    private readonly object _gate = new();
    private readonly Queue<ChunkEncodeItem> _items = new();
    private readonly HashSet<(string, string)> _queued = [];
    private readonly IChunkEncodeHost _host;
    private readonly IChunkEncodeSink _sink;
    private readonly CancellationTokenSource _cts = new();
    private readonly Thread _thread;
    private long _chars;
    private int _inFlight;
    private int _disposed;

    /// <summary>Creates the queue and starts its worker thread.</summary>
    /// <param name="host">The model the worker encodes on.</param>
    /// <param name="sink">Where entries go.</param>
    public ChunkEncodeQueue(IChunkEncodeHost host, IChunkEncodeSink sink)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
        _sink = sink ?? throw new ArgumentNullException(nameof(sink));
        _thread = new Thread(Run) { IsBackground = true, Name = "dotllm chunk-store writer", Priority = ThreadPriority.BelowNormal };
        _thread.UnsafeStart();
    }

    /// <summary>Items queued or being encoded.</summary>
    public int Pending
    {
        get { lock (_gate) return _items.Count + _inFlight; }
    }

    /// <summary>Adds an item.</summary>
    /// <param name="item">The item.</param>
    /// <returns>True when queued (or already queued); false when the queue is full or shut down.</returns>
    public bool TryEnqueue(ChunkEncodeItem item)
    {
        ArgumentNullException.ThrowIfNull(item);
        lock (_gate)
        {
            if (_disposed != 0)
                return false;
            if (_queued.Contains((item.Namespace, item.Key.Hex)))
                return true;
            if (_items.Count + _inFlight >= MaxItems || _chars + item.Text.Length > MaxQueuedChars)
                return false;
            _items.Enqueue(item);
            _queued.Add((item.Namespace, item.Key.Hex));
            _chars += item.Text.Length;
            Monitor.PulseAll(_gate);
            return true;
        }
    }

    /// <summary>Waits until nothing is queued or being encoded (tests and shutdown diagnostics).</summary>
    /// <param name="timeout">How long to wait.</param>
    /// <returns>Whether the queue drained in time.</returns>
    public bool WaitDrained(TimeSpan timeout)
    {
        var until = DateTime.UtcNow + timeout;
        lock (_gate)
        {
            while (_items.Count + _inFlight > 0)
            {
                var left = until - DateTime.UtcNow;
                if (left <= TimeSpan.Zero || !Monitor.Wait(_gate, left))
                    return _items.Count + _inFlight == 0;
            }
            return true;
        }
    }

    private void Run()
    {
        var ct = _cts.Token;
        var lastTick = DateTime.MinValue;
        while (!ct.IsCancellationRequested)
        {
            if (DateTime.UtcNow - lastTick >= TickInterval)
            {
                lastTick = DateTime.UtcNow;
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
                    _queued.Remove((item.Namespace, item.Key.Hex));
                    Monitor.PulseAll(_gate);
                }
            }
        }
        lock (_gate)
        {
            _items.Clear();
            _queued.Clear();
            _chars = 0;
            Monitor.PulseAll(_gate);
        }
    }

    private void Process(ChunkEncodeItem item, CancellationToken ct)
    {
        bool wanted = false;
        Guard(() => wanted = _sink.ShouldEncode(item));
        if (!wanted)
            return;

        // Wait (on this thread only) until no request runs or waits, then take the model for this one encode.
        IDisposable? lease = null;
        while (!ct.IsCancellationRequested)
        {
            Guard(() => lease = _host.TryEnterIdle());
            if (lease is not null)
                break;
            if (ct.WaitHandle.WaitOne(IdlePollInterval))
                return;
        }
        if (lease is null)
            return;

        ChunkKvEntry? entry = null;
        try
        {
            Guard(() => entry = _host.Encode(item));
        }
        finally
        {
            lease.Dispose(); // the model is free again before any disk I/O
        }
        if (entry is not null && !ct.IsCancellationRequested)
            Guard(() => _sink.Write(item, entry));
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

    /// <summary>Stops the worker (it finishes the step it is on) and drops what is queued.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _cts.Cancel();
        lock (_gate)
            Monitor.PulseAll(_gate);
        // The token source is left to the finalizer when the worker did not stop in time: it may still read it.
        if (Thread.CurrentThread != _thread && _thread.Join(TimeSpan.FromSeconds(30)))
            _cts.Dispose();
    }
}
