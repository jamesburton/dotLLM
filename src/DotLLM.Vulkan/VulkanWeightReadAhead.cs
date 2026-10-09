using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace DotLLM.Vulkan;

/// <summary>
/// Pure cursor for the read-ahead window (#874): given how many bytes of a tensor the upload loop has consumed, yields the
/// not-yet-requested pieces up to <c>consumed + window</c>. Allocation-free and GPU-free so the scheduling arithmetic is
/// unit-testable.
/// </summary>
internal struct VulkanReadAheadCursor
{
    private readonly long _total;
    private readonly long _window;
    private readonly long _piece;
    private long _issued;

    /// <summary>Creates a cursor over <paramref name="total"/> bytes with a <paramref name="window"/> lookahead in <paramref name="piece"/>-sized requests.</summary>
    public VulkanReadAheadCursor(long total, long window, long piece)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(total);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(piece);
        _total = total; _window = Math.Max(0, window); _piece = piece; _issued = 0;
    }

    /// <summary>Bytes already requested.</summary>
    public readonly long Issued => _issued;

    /// <summary>
    /// Next piece to request given <paramref name="consumed"/> bytes already read by the consumer, or false when the window is
    /// full or the tensor is covered. Call repeatedly until it returns false.
    /// </summary>
    public bool TryNext(long consumed, out long offset, out long length)
    {
        long want = Math.Min(_total, consumed + _window);
        // Never lag behind the consumer: bytes below `consumed` are already read.
        if (_issued < consumed) _issued = Math.Min(consumed, _total);
        if (_issued >= want) { offset = 0; length = 0; return false; }
        offset = _issued;
        length = Math.Min(_piece, want - _issued);
        _issued += length;
        return true;
    }
}

/// <summary>
/// Parallel page-in of the mmap'd GGUF ahead of the staging copy (#874). The upload loop <c>memcpy</c>s from the mapping into
/// the staging slot; with the pages cold that is one sequential page-fault stream at ~0.4-1 GB/s, far below what the NVMe
/// delivers with several reads in flight (3+ GB/s). Worker threads request the next window of the tensor
/// (<c>PrefetchVirtualMemory</c> on Windows, a page touch elsewhere) so the faults are taken off the critical path and in
/// parallel. Purely advisory: it never changes a byte that reaches the device.
/// </summary>
internal static class VulkanWeightReadAhead
{
    /// <summary>Window in bytes (<c>DOTLLM_VULKAN_READAHEAD_MB</c>, default 1024; 0 disables).</summary>
    public static long WindowBytes { get; } = ParseMb("DOTLLM_VULKAN_READAHEAD_MB", 1024) << 20;

    /// <summary>Tensors smaller than this are not worth the bookkeeping.</summary>
    public const long MinTensorBytes = 4L << 20;

    private const long PieceBytes = 16L << 20;

    private static readonly int s_threads = (int)Math.Clamp(ParseMb("DOTLLM_VULKAN_READAHEAD_THREADS", Math.Min(12, Environment.ProcessorCount / 2)), 1, 32);
    private static readonly object s_lock = new();
    private static BlockingCollection<(nint Ptr, long Len)>? s_queue;
    private static long s_requestedBytes;

    /// <summary>Total bytes handed to the workers this process (diagnostic).</summary>
    public static long RequestedBytes => Interlocked.Read(ref s_requestedBytes);

    private static long ParseMb(string name, long dflt)
        => long.TryParse(Environment.GetEnvironmentVariable(name), out long v) && v >= 0 ? v : dflt;

    /// <summary>True when read-ahead is enabled for a tensor of <paramref name="bytes"/>.</summary>
    public static bool Applies(long bytes) => WindowBytes > 0 && bytes >= MinTensorBytes;

    /// <summary>Starts a cursor for a tensor of <paramref name="bytes"/> bytes.</summary>
    public static VulkanReadAheadCursor Begin(long bytes) => new(bytes, WindowBytes, PieceBytes);

    /// <summary>Requests every piece the cursor allows for <paramref name="consumed"/> bytes consumed.</summary>
    public static void Pump(ref VulkanReadAheadCursor cursor, nint src, long consumed)
    {
        while (cursor.TryNext(consumed, out long off, out long len))
        {
            Interlocked.Add(ref s_requestedBytes, len);
            Queue().TryAdd((src + (nint)off, len));   // a full queue just drops advisory work
        }
    }

    private static BlockingCollection<(nint Ptr, long Len)> Queue()
    {
        var q = s_queue;
        if (q is not null) return q;
        lock (s_lock)
        {
            if (s_queue is not null) return s_queue;
            q = new BlockingCollection<(nint, long)>(boundedCapacity: 256);
            for (int i = 0; i < s_threads; i++)
                new Thread(() => Work(q)) { IsBackground = true, Name = "dotllm-readahead" }.Start();
            s_queue = q;
            return q;
        }
    }

    private static unsafe void Work(BlockingCollection<(nint Ptr, long Len)> q)
    {
        nint process = OperatingSystem.IsWindows() ? GetCurrentProcess() : 0;
        foreach (var (ptr, len) in q.GetConsumingEnumerable())
        {
            try
            {
                if (OperatingSystem.IsWindows())
                {
                    var e = new MemoryRangeEntry { VirtualAddress = (void*)ptr, NumberOfBytes = (nuint)len };
                    if (PrefetchVirtualMemory(process, 1, &e, 0)) continue;
                }
                byte sink = 0;
                for (long k = 0; k < len; k += 4096) sink ^= ((byte*)ptr)[k];
                GC.KeepAlive(sink);
            }
            catch { /* advisory only */ }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct MemoryRangeEntry { public void* VirtualAddress; public nuint NumberOfBytes; }

    [DllImport("kernel32.dll")] private static extern nint GetCurrentProcess();
    [DllImport("kernel32.dll")] private static extern unsafe bool PrefetchVirtualMemory(nint hProcess, nuint numberOfEntries, MemoryRangeEntry* entries, uint flags);
}
