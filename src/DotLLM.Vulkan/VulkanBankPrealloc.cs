namespace DotLLM.Vulkan;

/// <summary>
/// Allocates device-local buffers on background threads ahead of the upload that will fill them (#874).
/// </summary>
/// <remarks>
/// On the 512 MB-BIOS-split Strix Halo, <c>vkAllocateMemory</c> of device-local (GTT-backed) memory costs ~0.45 s per GiB - 33 s of
/// the 177 s routed-expert phase of the real qwen4exp file - because the OS commits and zeroes the pages inside the call. The
/// allocations are independent of the data, so they can overlap the previous layer's copy. Sizes are keyed, not ordered: a request
/// for a size with a pending allocation takes it (FIFO per size), anything else allocates inline, so a caller that deviates from the
/// schedule loses only the overlap, never correctness.
/// </remarks>
internal sealed class VulkanBankPrealloc : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly object _lock = new();
    private readonly Dictionary<long, Queue<Task<VulkanDevice.Buffer>>> _pending = [];
    private bool _disposed;

    /// <summary>Creates a pool that allocates on <paramref name="device"/>.</summary>
    public VulkanBankPrealloc(VulkanDevice device) => _device = device;

    /// <summary>Starts allocating one buffer of each size in <paramref name="sizes"/> on the thread pool.</summary>
    public void Schedule(IEnumerable<long> sizes)
    {
        foreach (long bytes in sizes)
        {
            var task = Task.Run(() => _device.AllocateDeviceLocal(bytes));
            lock (_lock)
            {
                if (_disposed) { task.ContinueWith(t => { if (t.IsCompletedSuccessfully) t.Result.Dispose(); }); return; }
                if (!_pending.TryGetValue(bytes, out var q)) _pending[bytes] = q = new Queue<Task<VulkanDevice.Buffer>>();
                q.Enqueue(task);
            }
        }
    }

    /// <summary>Returns a pre-allocated buffer of exactly <paramref name="bytes"/> (waiting for it if still in flight), or null if none is scheduled.</summary>
    public VulkanDevice.Buffer? Take(long bytes)
    {
        Task<VulkanDevice.Buffer>? task = null;
        lock (_lock)
        {
            if (_pending.TryGetValue(bytes, out var q) && q.Count > 0) task = q.Dequeue();
        }
        if (task is null) return null;
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var buf = task.GetAwaiter().GetResult();
        Interlocked.Add(ref s_waitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - t0);
        return buf;
    }

    private static long s_waitTicks;

    /// <summary>Process-wide time callers spent blocked in <see cref="Take"/> on an allocation still in flight (#874 diagnostic).</summary>
    public static double WaitMilliseconds => Interlocked.Read(ref s_waitTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    /// <summary>Total bytes scheduled and not yet taken.</summary>
    public long PendingBytes { get { lock (_lock) return _pending.Sum(kv => kv.Key * kv.Value.Count); } }

    /// <summary>
    /// Pure placement guard: may <paramref name="need"/> more bytes be allocated AHEAD of program order without changing which heap any
    /// allocation lands on? True only when everything live, everything already scheduled and this request still fit the device-local
    /// heap with <paramref name="margin"/> to spare - i.e. nothing near the heap boundary, where order decides what falls back to the
    /// slower heap. Past that point the loader allocates inline, in the original order, so the residency report is unchanged (#874).
    /// </summary>
    public static bool MayAllocateAhead(long heapBytes, long liveBytes, long pendingBytes, long need, long margin)
        => liveBytes + pendingBytes + need + margin <= heapBytes;

    /// <summary>Number of allocations scheduled and not yet taken.</summary>
    public int PendingCount { get { lock (_lock) return _pending.Values.Sum(q => q.Count); } }

    /// <inheritdoc/>
    public void Dispose()
    {
        List<Task<VulkanDevice.Buffer>> left;
        lock (_lock)
        {
            _disposed = true;
            left = _pending.Values.SelectMany(q => q).ToList();
            _pending.Clear();
        }
        foreach (var t in left)
        {
            try { t.GetAwaiter().GetResult().Dispose(); } catch { /* an allocation that failed has nothing to free */ }
        }
    }
}
