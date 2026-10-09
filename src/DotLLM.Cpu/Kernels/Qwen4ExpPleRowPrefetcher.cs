using System.Buffers;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>Where the n-gram (PLE) table's bytes come from at gather time.</summary>
public enum PleTableMode
{
    /// <summary>Rows are read straight out of the lazily mapped table (the OS page cache is the tier). Default.</summary>
    Mmap,
    /// <summary>Rows used so far are also kept in a process-owned row cache so repeated tokens never touch the mapping again
    /// (immune to working-set trimming / page-cache eviction of the mapped file).</summary>
    Ram,
}

/// <summary>How a chunk's table pages are pulled in ahead of the gather.</summary>
public enum PleHintMode
{
    /// <summary>No prefetch: the gather demand-faults (the pre-#822 behaviour).</summary>
    Off,
    /// <summary>OS hint only (<c>PrefetchVirtualMemory</c> on Windows, <c>madvise(WILLNEED)</c> on Linux/macOS); asynchronous.</summary>
    Os,
    /// <summary>A pool of threads reads one byte of every needed page (portable; completion is observable, IO queue depth = threads).</summary>
    Touch,
    /// <summary>The OS hint first, then the touch pool (the hint is only a request; the touch is what makes <c>Collect</c> a real barrier).</summary>
    OsThenTouch,
}

/// <summary>
/// Configuration of the n-gram table service: <c>DOTLLM_PLE_TABLE=mmap|ram</c>, <c>DOTLLM_PLE_PREFETCH=off|os|touch|auto</c>,
/// <c>DOTLLM_PLE_CACHE_MB</c>, <c>DOTLLM_PLE_TOUCH_THREADS</c>, <c>DOTLLM_PLE_STATS=1</c>.
/// </summary>
public sealed record PleTableOptions
{
    private static readonly AsyncLocal<PleTableOptions?> s_ambient = new();

    /// <summary>Table source (default <see cref="PleTableMode.Mmap"/>).</summary>
    public PleTableMode Mode { get; init; } = PleTableMode.Mmap;

    /// <summary>Prefetch strategy (default <see cref="PleHintMode.OsThenTouch"/>).</summary>
    public PleHintMode Hint { get; init; } = PleHintMode.OsThenTouch;

    /// <summary>Row-cache budget in MiB. <c>null</c> = default (0 for mmap, 64 for ram); 0 disables the cache.</summary>
    public int? CacheMegabytes { get; init; }

    /// <summary>Threads the touch pool / ram copy uses (<c>null</c> = min(8, cores)).</summary>
    public int? TouchThreads { get; init; }

    /// <summary>Samples the process page-fault counter around each gather (cheap, but process-wide).</summary>
    public bool CollectFaults { get; init; }

    /// <summary>Effective cache budget in bytes.</summary>
    public long CacheBytes => (CacheMegabytes ?? (Mode == PleTableMode.Ram ? 64 : 0)) * 1024L * 1024L;

    /// <summary>Effective touch threads.</summary>
    public int Threads => Math.Max(1, TouchThreads ?? Math.Min(8, Environment.ProcessorCount));

    /// <summary>No prefetch, no cache: exactly the pre-#822 gather. The control arm of every equivalence test.</summary>
    public static PleTableOptions Disabled { get; } = new() { Hint = PleHintMode.Off, CacheMegabytes = 0 };

    /// <summary>
    /// Options a branch built on this async flow picks up instead of the environment (tests flip arms without racing on
    /// process-wide environment variables).
    /// </summary>
    public static PleTableOptions? Ambient { get => s_ambient.Value; set => s_ambient.Value = value; }

    /// <summary>Reads the <c>DOTLLM_PLE_*</c> environment (or <see cref="Ambient"/> when set).</summary>
    public static PleTableOptions FromEnvironment()
    {
        if (Ambient is { } amb) return amb;
        var o = new PleTableOptions();
        string? mode = Environment.GetEnvironmentVariable("DOTLLM_PLE_TABLE");
        if (string.Equals(mode, "ram", StringComparison.OrdinalIgnoreCase)) o = o with { Mode = PleTableMode.Ram };
        string? hint = Environment.GetEnvironmentVariable("DOTLLM_PLE_PREFETCH");
        if (hint is not null)
            o = o with
            {
                Hint = hint.ToLowerInvariant() switch
                {
                    "off" or "0" or "none" => PleHintMode.Off,
                    "os" => PleHintMode.Os,
                    "touch" => PleHintMode.Touch,
                    _ => PleHintMode.OsThenTouch,
                },
            };
        if (int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_PLE_CACHE_MB"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int mb) && mb >= 0)
            o = o with { CacheMegabytes = mb };
        if (int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_PLE_TOUCH_THREADS"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int th) && th > 0)
            o = o with { TouchThreads = th };
        if (Environment.GetEnvironmentVariable("DOTLLM_PLE_STATS") == "1") o = o with { CollectFaults = true };
        return o;
    }

    /// <summary>True when a service object is needed at all (otherwise the branch keeps the plain gather).</summary>
    public bool NeedsService => Hint != PleHintMode.Off || CacheBytes > 0;
}

/// <summary>Counters of one <see cref="PleRowPrefetcher"/> (snapshot).</summary>
/// <param name="Requests">Prefetch requests issued.</param>
/// <param name="Gathers">Gather calls.</param>
/// <param name="Rows">Rows requested by gathers (with repeats).</param>
/// <param name="UniqueRows">Distinct rows across requests.</param>
/// <param name="UniquePages">Distinct table pages across requests (the coalesced prefetch footprint).</param>
/// <param name="Ranges">Coalesced contiguous page ranges handed to the OS / touch pool.</param>
/// <param name="CacheHits">Rows served from the row cache.</param>
/// <param name="CacheMisses">Rows read from the mapping.</param>
/// <param name="WaitMicros">Microseconds <c>Collect</c> blocked the caller (the PLE wait the prefetch failed to hide).</param>
/// <param name="WorkMicros">Microseconds the background worker spent (sort, hint, touch/copy).</param>
/// <param name="GatherMicros">Microseconds in the gather proper (after the wait).</param>
/// <param name="GatherFaults">Process page faults (soft+hard) during gathers; only when <see cref="PleTableOptions.CollectFaults"/>.</param>
/// <param name="HintFailures">OS hint calls that failed or were unsupported (fell back to touch).</param>
public readonly record struct PleTableStats(long Requests, long Gathers, long Rows, long UniqueRows, long UniquePages, long Ranges,
                                            long CacheHits, long CacheMisses, long WaitMicros, long WorkMicros, long GatherMicros,
                                            long GatherFaults, long HintFailures);

internal static class PleTelemetry
{
    internal static long Rows, UniquePages, CacheHits, CacheMisses, WaitMicros, GatherFaults;
    private static readonly Meter s_meter = new("DotLLM.Ple");
    // Observable instruments: zero cost on the hot path (the totals are plain Interlocked adds) and nothing runs without a listener.
    internal static readonly ObservableCounter<long> RowsC = s_meter.CreateObservableCounter("dotllm.ple.rows", () => Volatile.Read(ref Rows), "rows");
    internal static readonly ObservableCounter<long> PagesC = s_meter.CreateObservableCounter("dotllm.ple.unique_pages", () => Volatile.Read(ref UniquePages), "pages");
    internal static readonly ObservableCounter<long> HitC = s_meter.CreateObservableCounter("dotllm.ple.cache_hits", () => Volatile.Read(ref CacheHits), "rows");
    internal static readonly ObservableCounter<long> MissC = s_meter.CreateObservableCounter("dotllm.ple.cache_misses", () => Volatile.Read(ref CacheMisses), "rows");
    internal static readonly ObservableCounter<long> WaitC = s_meter.CreateObservableCounter("dotllm.ple.wait", () => Volatile.Read(ref WaitMicros), "us");
    internal static readonly ObservableCounter<long> FaultC = s_meter.CreateObservableCounter("dotllm.ple.gather_faults", () => Volatile.Read(ref GatherFaults), "faults");
    internal static void Touch() { }
}

/// <summary>
/// Prefetch / offload service for the Qwen4-Exp n-gram table (#822). The table is one multi-GB tensor read through a lazy mmap,
/// 16 rows per token; the row ids depend only on token ids, so every row of a chunk is known before layer 0 runs and can be
/// pulled in while the earlier layers compute.
/// </summary>
/// <remarks>
/// <para>
/// <b>Correctness invariant.</b> The prefetcher only changes <i>when</i> pages become resident, and (in <see cref="PleTableMode.Ram"/>
/// or with a cache) which copy of the same bytes <see cref="Gather"/> dequantises. Every row is dequantised from byte-identical
/// source data by the same <see cref="Dequantize"/> routine, so the output equals <see cref="Qwen4ExpPle.GatherRows"/> bit for bit,
/// whether or not a request was issued, finished, or matched the rows later gathered.
/// </para>
/// <para>
/// <b>Concurrency.</b> <see cref="Begin"/> only copies the row ids and queues work (non-blocking). One background task per request
/// sorts / dedupes / coalesces pages and runs the hint and touch phases. The row cache is guarded by a lock that is held only for
/// map updates and gather lookups, never across page faults of the worker.
/// </para>
/// </remarks>
public sealed unsafe class PleRowPrefetcher : IDisposable
{
    /// <summary>An in-flight prefetch. Pass it to <see cref="Gather"/> (or <see cref="Collect"/>) exactly once.</summary>
    public sealed class Request
    {
        internal long[]? Rows;
        internal int Count;
        internal Task? Work;
        internal Request() { }
    }

    private readonly byte* _table;
    private readonly QuantizationType _qt;
    private readonly long _tableRows;
    private readonly int _rowDim;
    private readonly long _rowBytes;
    private readonly PleTableOptions _opt;
    private readonly RowCache? _cache;
    private readonly int _pageShift;
    private long _requests, _gathers, _rows, _uniqueRows, _uniquePages, _ranges, _hits, _misses, _wait, _work, _gatherUs, _faults, _hintFail;
    private bool _disposed;
    private int _active;

    /// <summary>Creates the service over a borrowed table pointer (never copied, never freed).</summary>
    /// <param name="table">Base of the row-major table.</param>
    /// <param name="quantType">Table storage type.</param>
    /// <param name="tableRows">Rows in the table.</param>
    /// <param name="rowDim">Elements per row.</param>
    /// <param name="options">Service options.</param>
    public PleRowPrefetcher(nint table, QuantizationType quantType, long tableRows, int rowDim, PleTableOptions options)
    {
        _table = (byte*)table; _qt = quantType; _tableRows = tableRows; _rowDim = rowDim; _opt = options;
        _rowBytes = Dequantize.RowByteSize(rowDim, quantType);
        _pageShift = BitOperations.Log2((uint)Environment.SystemPageSize);
        long cacheRows = options.CacheBytes / Math.Max(1, _rowBytes);
        if (cacheRows > 0) _cache = new RowCache((int)Math.Min(cacheRows, int.MaxValue / 2), (int)_rowBytes);
        PleTelemetry.Touch();
    }

    /// <summary>True when <see cref="Begin"/> does anything (a hint mode other than Off).</summary>
    public bool PrefetchEnabled => _opt.Hint != PleHintMode.Off;

    /// <summary>Effective options.</summary>
    public PleTableOptions Options => _opt;

    /// <summary>Counters so far.</summary>
    public PleTableStats Stats => new(Volatile.Read(ref _requests), Volatile.Read(ref _gathers), Volatile.Read(ref _rows),
        Volatile.Read(ref _uniqueRows), Volatile.Read(ref _uniquePages), Volatile.Read(ref _ranges), Volatile.Read(ref _hits),
        Volatile.Read(ref _misses), Volatile.Read(ref _wait), Volatile.Read(ref _work), Volatile.Read(ref _gatherUs),
        Volatile.Read(ref _faults), Volatile.Read(ref _hintFail));

    /// <summary>Row-cache capacity in rows (0 = none).</summary>
    public int CacheCapacityRows => _cache?.Capacity ?? 0;

    /// <summary>Rows currently cached.</summary>
    public int CachedRows => _cache?.Count ?? 0;

    /// <summary>
    /// Queues the prefetch of <paramref name="rows"/> (table row ids) on a background task and returns immediately.
    /// Returns <c>null</c> (nothing to wait for) when prefetch is off.
    /// </summary>
    public Request? Begin(ReadOnlySpan<long> rows)
    {
        if (_opt.Hint == PleHintMode.Off || rows.IsEmpty) return null;
        ObjectDisposedException.ThrowIf(_disposed, this);
        var req = new Request { Rows = ArrayPool<long>.Shared.Rent(rows.Length), Count = rows.Length };
        rows.CopyTo(req.Rows);
        Interlocked.Increment(ref _requests);
        Interlocked.Increment(ref _active);
        req.Work = Task.Run(() => RunWorker(req));
        return req;
    }

    /// <summary>Blocks until <paramref name="req"/> has finished (its pages are resident / cached). No-op for <c>null</c>.</summary>
    public void Collect(Request? req)
    {
        if (req?.Work is not { } work) return;
        long t0 = Stopwatch.GetTimestamp();
        bool pending = !work.IsCompleted;
        try { work.GetAwaiter().GetResult(); }
        finally
        {
            if (pending)
            {
                long us = (Stopwatch.GetTimestamp() - t0) * 1_000_000 / Stopwatch.Frequency;
                Interlocked.Add(ref _wait, us);
                Interlocked.Add(ref PleTelemetry.WaitMicros, us);
            }
            if (req.Rows is { } r) { ArrayPool<long>.Shared.Return(r); req.Rows = null; }
            req.Work = null;
        }
    }

    /// <summary>
    /// Collects <paramref name="req"/> (if any), then gathers and dequantises <paramref name="rows"/> into <paramref name="dest"/>.
    /// Bit-identical to <see cref="Qwen4ExpPle.GatherRows"/> over the same table.
    /// </summary>
    public void Gather(Request? req, ReadOnlySpan<long> rows, Span<float> dest)
    {
        Collect(req);
        long t0 = Stopwatch.GetTimestamp();
        long f0 = _opt.CollectFaults ? ProcessFaults.Current() : 0;
        if (_cache is null)
        {
            Qwen4ExpPle.GatherRows((nint)_table, _qt, _tableRows, _rowDim, rows, dest);
            Interlocked.Add(ref _misses, rows.Length);
            Interlocked.Add(ref PleTelemetry.CacheMisses, rows.Length);
        }
        else
        {
            GatherCached(rows, dest);
        }
        long us = (Stopwatch.GetTimestamp() - t0) * 1_000_000 / Stopwatch.Frequency;
        Interlocked.Increment(ref _gathers);
        Interlocked.Add(ref _rows, rows.Length);
        Interlocked.Add(ref _gatherUs, us);
        Interlocked.Add(ref PleTelemetry.Rows, rows.Length);
        if (_opt.CollectFaults)
        {
            long df = ProcessFaults.Current() - f0;
            Interlocked.Add(ref _faults, df);
            Interlocked.Add(ref PleTelemetry.GatherFaults, df);
        }
    }

    private void GatherCached(ReadOnlySpan<long> rows, Span<float> dest)
    {
        if (dest.Length < (long)rows.Length * _rowDim) throw new ArgumentException("dest too small.", nameof(dest));
        var cache = _cache!;
        long hits = 0, misses = 0;
        lock (cache)
        {
            for (int i = 0; i < rows.Length; i++)
            {
                long r = rows[i];
                if ((ulong)r >= (ulong)_tableRows)
                    throw new ArgumentOutOfRangeException(nameof(rows), $"PLE row {r} outside table of {_tableRows} rows.");
                var d = dest.Slice(i * _rowDim, _rowDim);
                if (cache.TryGet(r, out byte* slot))
                {
                    Dequantize.ToFloat32((nint)slot, _rowDim, _qt, d);
                    hits++;
                }
                else
                {
                    // Read-through: dequantise from the mapping, then keep the raw row for the next time this token shows up.
                    byte* src = _table + r * _rowBytes;
                    Dequantize.ToFloat32((nint)src, _rowDim, _qt, d);
                    cache.Insert(r, src);
                    misses++;
                }
            }
        }
        Interlocked.Add(ref _hits, hits); Interlocked.Add(ref _misses, misses);
        Interlocked.Add(ref PleTelemetry.CacheHits, hits); Interlocked.Add(ref PleTelemetry.CacheMisses, misses);
    }

    // ── background worker ───────────────────────────────────────────────────

    private void RunWorker(Request req)
    {
        long t0 = Stopwatch.GetTimestamp();
        try { RunWorkerCore(req, t0); }
        finally { Interlocked.Decrement(ref _active); }
    }

    private void RunWorkerCore(Request req, long t0)
    {
        long[] sorted = ArrayPool<long>.Shared.Rent(req.Count);
        try
        {
            req.Rows.AsSpan(0, req.Count).CopyTo(sorted);
            int n = SortUnique(sorted.AsSpan(0, req.Count));
            // Ids outside the table are the gather's problem (it throws); never prefetch past the mapping.
            while (n > 0 && (ulong)sorted[n - 1] >= (ulong)_tableRows) n--;
            if (n == 0) return;
            var rows = sorted.AsSpan(0, n);

            var ranges = BuildPageRanges(rows, out long pages);
            Interlocked.Add(ref _uniqueRows, n);
            Interlocked.Add(ref _uniquePages, pages);
            Interlocked.Add(ref _ranges, ranges.Count);
            Interlocked.Add(ref PleTelemetry.UniquePages, pages);

            if (_opt.Hint is PleHintMode.Os or PleHintMode.OsThenTouch)
            {
                if (!OsPrefetch.Hint(ranges, _pageShift, _table)) Interlocked.Increment(ref _hintFail);
                else if (_opt.Hint == PleHintMode.Os) goto done;
            }
            // Touch phase. With a row cache the "touch" is the copy of the still-missing rows into the cache (it faults the very
            // same pages); otherwise one byte per page.
            if (_cache is not null) FillCache(rows);
            else TouchPages(ranges);
        done:;
        }
        finally
        {
            ArrayPool<long>.Shared.Return(sorted);
            Interlocked.Add(ref _work, (Stopwatch.GetTimestamp() - t0) * 1_000_000 / Stopwatch.Frequency);
        }
    }

    private static int SortUnique(Span<long> v)
    {
        v.Sort();
        int w = 0;
        for (int i = 0; i < v.Length; i++)
            if (i == 0 || v[i] != v[i - 1]) v[w++] = v[i];
        return w;
    }

    /// <summary>Sorted unique rows to coalesced page ranges (start page index, page count), ascending and non-overlapping.</summary>
    internal List<(long Page, long Count)> BuildPageRanges(ReadOnlySpan<long> sortedUniqueRows, out long uniquePages)
        => BuildPageRanges(sortedUniqueRows, _rowBytes, (long)(nuint)_table, _pageShift, out uniquePages);

    /// <summary>Pure page coalescing (testable without a mapping): rows of <paramref name="rowBytes"/> at <paramref name="baseAddress"/>.</summary>
    internal static List<(long Page, long Count)> BuildPageRanges(ReadOnlySpan<long> sortedUniqueRows, long rowBytes, long baseAddress,
                                                                  int pageShift, out long uniquePages)
    {
        var ranges = new List<(long Page, long Count)>(Math.Min(sortedUniqueRows.Length, 1 << 16));
        long total = 0, curStart = -1, curEnd = -1;   // [curStart, curEnd) in pages
        foreach (long r in sortedUniqueRows)
        {
            long a = baseAddress + r * rowBytes;
            long p0 = a >> pageShift, p1 = ((a + rowBytes - 1) >> pageShift) + 1;
            if (curStart < 0) { curStart = p0; curEnd = p1; continue; }
            if (p0 <= curEnd) { if (p1 > curEnd) curEnd = p1; continue; }   // adjacent or overlapping -> merge
            ranges.Add((curStart, curEnd - curStart)); total += curEnd - curStart;
            curStart = p0; curEnd = p1;
        }
        if (curStart >= 0) { ranges.Add((curStart, curEnd - curStart)); total += curEnd - curStart; }
        uniquePages = total;
        return ranges;
    }

    private void TouchPages(List<(long Page, long Count)> ranges)
    {
        if (ranges.Count == 0) return;
        int threads = _opt.Threads;
        long pageSize = 1L << _pageShift;
        int parts = Math.Min(ranges.Count, threads * 8);
        long sink = 0;
        Parallel.For(0, parts, new ParallelOptions { MaxDegreeOfParallelism = threads }, p =>
        {
            int lo = (int)((long)ranges.Count * p / parts), hi = (int)((long)ranges.Count * (p + 1) / parts);
            long acc = 0;
            for (int i = lo; i < hi; i++)
            {
                byte* a = (byte*)(ranges[i].Page << _pageShift);
                for (long k = 0; k < ranges[i].Count; k++) acc += Volatile.Read(ref a[k * pageSize]);
            }
            Interlocked.Add(ref sink, acc);
        });
        GC.KeepAlive(sink);
    }

    /// <summary>Copies the rows not yet cached from the mapping (in parallel; this is what faults their pages) and inserts them.</summary>
    private void FillCache(ReadOnlySpan<long> sortedUniqueRows)
    {
        var cache = _cache!;
        long[] missing = ArrayPool<long>.Shared.Rent(sortedUniqueRows.Length);
        int m = 0;
        lock (cache)
            foreach (long r in sortedUniqueRows)
                if (!cache.Contains(r)) missing[m++] = r;
        m = Math.Min(m, cache.Capacity);   // more than fits would just evict its own earlier rows
        if (m == 0) { ArrayPool<long>.Shared.Return(missing); return; }
        byte[] buf = ArrayPool<byte>.Shared.Rent(checked((int)Math.Min((long)m * _rowBytes, int.MaxValue)));
        try
        {
            int threads = _opt.Threads, rb = (int)_rowBytes;
            int parts = Math.Min(m, threads * 8);
            var miss = missing;
            Parallel.For(0, parts, new ParallelOptions { MaxDegreeOfParallelism = threads }, p =>
            {
                int lo = (int)((long)m * p / parts), hi = (int)((long)m * (p + 1) / parts);
                for (int i = lo; i < hi; i++)
                    new ReadOnlySpan<byte>(_table + miss[i] * rb, rb).CopyTo(buf.AsSpan(i * rb, rb));
            });
            lock (cache)
                fixed (byte* b = buf)
                    for (int i = 0; i < m; i++) cache.Insert(missing[i], b + (long)i * rb);
        }
        finally { ArrayPool<byte>.Shared.Return(buf); ArrayPool<long>.Shared.Return(missing); }
    }

    /// <summary>Drops outstanding work and frees the row cache.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        var spin = new SpinWait();
        while (Volatile.Read(ref _active) != 0) spin.SpinOnce();   // an abandoned request may still be reading the table / filling the cache
        _cache?.Dispose();
    }

    // ── clock row cache ─────────────────────────────────────────────────────

    private sealed class RowCache : IDisposable
    {
        private readonly Dictionary<long, int> _map;
        private readonly long[] _keys;
        private readonly byte[] _ref;
        private readonly byte* _slab;
        private readonly int _rowBytes;
        private int _used, _hand;

        public RowCache(int capacity, int rowBytes)
        {
            Capacity = capacity; _rowBytes = rowBytes;
            _map = new Dictionary<long, int>(capacity);
            _keys = new long[capacity]; _ref = new byte[capacity];
            _slab = (byte*)NativeMemory.AlignedAlloc((nuint)((long)capacity * rowBytes), 64);
        }

        public int Capacity { get; }
        public int Count => _used;
        public bool Contains(long row) => _map.ContainsKey(row);

        public bool TryGet(long row, out byte* slot)
        {
            if (_map.TryGetValue(row, out int s)) { _ref[s] = 1; slot = _slab + (long)s * _rowBytes; return true; }
            slot = null; return false;
        }

        public void Insert(long row, byte* src)
        {
            if (_map.TryGetValue(row, out int existing)) { _ref[existing] = 1; return; }
            int s;
            if (_used < Capacity) s = _used++;
            else
            {
                while (_ref[_hand] != 0) { _ref[_hand] = 0; if (++_hand == Capacity) _hand = 0; }
                s = _hand; if (++_hand == Capacity) _hand = 0;
                _map.Remove(_keys[s]);
            }
            Buffer.MemoryCopy(src, _slab + (long)s * _rowBytes, _rowBytes, _rowBytes);
            _keys[s] = row; _ref[s] = 0; _map[row] = s;
        }

        public void Dispose() => NativeMemory.AlignedFree(_slab);
    }
}

/// <summary>OS page-in hints. All best effort; failure means "fall back to the touch pool".</summary>
internal static unsafe partial class OsPrefetch
{
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryRangeEntry { public nint VirtualAddress; public nuint NumberOfBytes; }

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool PrefetchVirtualMemory(nint process, nuint numberOfEntries, MemoryRangeEntry* entries, uint flags);

    [LibraryImport("kernel32.dll")]
    private static partial nint GetCurrentProcess();

    [LibraryImport("libc", SetLastError = true)]
    private static partial int madvise(nint addr, nuint length, int advice);

    private const int MadvWillNeed = 3;

    /// <summary>Issues the hint for every range. Returns false when the platform has none or a call failed.</summary>
    internal static bool Hint(List<(long Page, long Count)> ranges, int pageShift, byte* tableBase)
    {
        if (ranges.Count == 0) return true;
        try
        {
            if (OperatingSystem.IsWindows())
            {
                const int batch = 2048;
                var buf = stackalloc MemoryRangeEntry[batch];
                nint proc = GetCurrentProcess();
                bool ok = true;
                for (int i = 0; i < ranges.Count; i += batch)
                {
                    int n = Math.Min(batch, ranges.Count - i);
                    for (int k = 0; k < n; k++)
                        buf[k] = new MemoryRangeEntry { VirtualAddress = (nint)(ranges[i + k].Page << pageShift), NumberOfBytes = (nuint)(ranges[i + k].Count << pageShift) };
                    ok &= PrefetchVirtualMemory(proc, (nuint)n, buf, 0);
                }
                return ok;
            }
            if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
            {
                bool ok = true;
                foreach (var (page, count) in ranges)
                    ok &= madvise((nint)(page << pageShift), (nuint)(count << pageShift), MadvWillNeed) == 0;
                return ok;
            }
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException) { }
        return false;
    }
}

/// <summary>Process page-fault counter and mapped-page residency probes (diagnostics only).</summary>
public static unsafe partial class ProcessFaults
{
    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessMemoryCounters
    {
        public uint cb, PageFaultCount;
        public nuint PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
                     QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage;
    }

    [LibraryImport("kernel32.dll")]
    private static partial nint GetCurrentProcess();

    [LibraryImport("kernel32.dll", EntryPoint = "K32GetProcessMemoryInfo")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetProcessMemoryInfo(nint process, ProcessMemoryCounters* counters, uint cb);

    [StructLayout(LayoutKind.Sequential)]
    private struct WorkingSetExInformation { public nint VirtualAddress; public nuint VirtualAttributes; }

    [LibraryImport("kernel32.dll", EntryPoint = "K32QueryWorkingSetEx")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool QueryWorkingSetEx(nint process, WorkingSetExInformation* info, uint cb);

    /// <summary>Process-wide page faults (soft + hard) so far; -1 when unavailable.</summary>
    public static long Current()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                ProcessMemoryCounters c = default;
                c.cb = (uint)sizeof(ProcessMemoryCounters);
                return GetProcessMemoryInfo(GetCurrentProcess(), &c, c.cb) ? c.PageFaultCount : -1;
            }
            if (OperatingSystem.IsLinux())
            {
                string s = File.ReadAllText("/proc/self/stat");
                string[] f = s[(s.LastIndexOf(')') + 2)..].Split(' ');   // fields after "(comm) state": minflt = idx 7, majflt = idx 9
                return long.Parse(f[7], CultureInfo.InvariantCulture) + long.Parse(f[9], CultureInfo.InvariantCulture);
            }
        }
        catch (Exception e) when (e is IOException or FormatException or IndexOutOfRangeException or UnauthorizedAccessException) { }
        return -1;
    }

    /// <summary>
    /// Counts how many of <paramref name="pageAddresses"/> are in the process working set (Windows only; -1 elsewhere). A page that
    /// was trimmed to the standby list reports non-resident here yet costs only a soft fault, so this is a lower bound on warmth.
    /// </summary>
    public static long CountWorkingSetResident(ReadOnlySpan<nint> pageAddresses)
    {
        if (!OperatingSystem.IsWindows()) return -1;
        var info = new WorkingSetExInformation[pageAddresses.Length];
        for (int i = 0; i < info.Length; i++) info[i].VirtualAddress = pageAddresses[i];
        fixed (WorkingSetExInformation* p = info)
            if (!QueryWorkingSetEx(GetCurrentProcess(), p, (uint)(info.Length * sizeof(WorkingSetExInformation)))) return -1;
        long valid = 0;
        foreach (var e in info) valid += (long)(e.VirtualAttributes & 1);
        return valid;
    }
}
