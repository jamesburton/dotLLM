using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace DotLLM.Vulkan.Tiering;

/// <summary>
/// Blocking batch reader for the expert-tiering miss path (#939): issues a set of reads with up to <c>queueDepth</c> outstanding at once and returns when all
/// completed. On Windows it opens each file <c>FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED</c> and drains one I/O completion port, so the bytes DMA straight into
/// the caller's destination (a mapped host-visible Vulkan buffer) and nothing enters the OS page cache (stage 0, #937: no cache growth, 2.9-3.3 GB/s on a fast region,
/// QD 4-8 saturates). Elsewhere (and with <c>unbuffered: false</c>) it falls back to synchronous positional reads, which keeps the class testable and portable.
/// </summary>
/// <remarks>
/// Unbuffered requests must have a sector-aligned file offset, a sector-multiple length and a sector-aligned destination; <see cref="Align"/> (4096) satisfies both
/// 512e and 4Kn drives. The buffered fallback has no such restriction. Not thread-safe: one batch at a time.
/// </remarks>
internal sealed unsafe partial class UnbufferedFileReader : IDisposable
{
    /// <summary>Alignment (bytes) of offset, length and destination for unbuffered reads.</summary>
    public const int Align = 4096;

    /// <summary>One read: <c>Length</c> bytes of file <c>File</c> at <c>Offset</c> into <c>Dest</c>; <c>Need</c> bytes must actually arrive (a read cut short by EOF is fine as long as this many came). Offset, Length and Dest must be multiples of <see cref="Align"/> when unbuffered.</summary>
    /// <param name="file">Index into the path list the reader was created with.</param>
    /// <param name="offset">Absolute file offset.</param>
    /// <param name="length">Bytes to request.</param>
    /// <param name="dest">Destination address.</param>
    /// <param name="need">Bytes that must arrive.</param>
    public readonly struct Request(int file, long offset, int length, byte* dest, int need)
    {
        /// <summary>Index into the path list.</summary>
        public int File { get; } = file;
        /// <summary>Absolute file offset.</summary>
        public long Offset { get; } = offset;
        /// <summary>Bytes requested.</summary>
        public int Length { get; } = length;
        /// <summary>Destination address.</summary>
        public byte* Dest { get; } = dest;
        /// <summary>Bytes that must arrive.</summary>
        public int Need { get; } = need;
    }

    private readonly nint[] _handles;
    private readonly SafeFileHandle[]? _fallback;
    private readonly nint _port;
    private readonly int _queueDepth;
    private readonly Ov* _ovs;

    /// <summary>True when reads bypass the OS page cache (Windows overlapped, no buffering).</summary>
    public bool Unbuffered { get; }

    /// <summary>Total bytes delivered by <see cref="Read"/> since creation.</summary>
    public long BytesRead { get; private set; }

    /// <summary>Total requests completed since creation.</summary>
    public long RequestsCompleted { get; private set; }

    /// <summary>Opens <paramref name="paths"/> for reading.</summary>
    /// <param name="paths">Files addressed by <see cref="Request.File"/> (symlinks are followed).</param>
    /// <param name="unbuffered">Request the page-cache-bypassing mode; silently buffered/synchronous off Windows.</param>
    /// <param name="queueDepth">Maximum outstanding requests (design: 4-8 saturates the drive).</param>
    public UnbufferedFileReader(IReadOnlyList<string> paths, bool unbuffered = true, int queueDepth = 8)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(queueDepth, 1);
        _queueDepth = queueDepth;
        _handles = new nint[paths.Count];
        if (unbuffered && OperatingSystem.IsWindows())
        {
            Unbuffered = true;
            _port = CreateIoCompletionPort(-1, 0, 0, 1);   // INVALID_HANDLE_VALUE: create a fresh port
            if (_port == 0) throw new Win32Exception(Marshal.GetLastPInvokeError(), "CreateIoCompletionPort");
            for (int i = 0; i < paths.Count; i++)
            {
                // GENERIC_READ, FILE_SHARE_READ|WRITE (the GGUF mapping holds the file open), OPEN_EXISTING, OVERLAPPED | NO_BUFFERING.
                nint h = CreateFileW(paths[i], 0x80000000, 3, 0, 3, 0x40000000u | 0x20000000u, 0);
                if (h == -1) throw new IOException($"cannot open '{paths[i]}' for unbuffered reads: {new Win32Exception(Marshal.GetLastPInvokeError()).Message}");
                _handles[i] = h;
                if (CreateIoCompletionPort(h, _port, (nuint)i, 0) == 0) throw new Win32Exception(Marshal.GetLastPInvokeError(), "CreateIoCompletionPort(file)");
            }
            _ovs = (Ov*)NativeMemory.AllocZeroed((nuint)(sizeof(Ov) * queueDepth));
        }
        else
        {
            _fallback = new SafeFileHandle[paths.Count];
            for (int i = 0; i < paths.Count; i++) _fallback[i] = File.OpenHandle(paths[i], FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        }
    }

    /// <summary>Reads every request, keeping up to the queue depth outstanding; returns when all completed. Throws on a failed or short read.</summary>
    public void Read(ReadOnlySpan<Request> requests)
    {
        if (requests.IsEmpty) return;
        if (!Unbuffered)
        {
            foreach (var r in requests)
            {
                int got = RandomAccess.Read(_fallback![r.File], new Span<byte>(r.Dest, r.Length), r.Offset);
                if (got < r.Need) throw new IOException($"short read: {got} of {r.Need} bytes at offset {r.Offset}");
                BytesRead += got; RequestsCompleted++;
            }
            return;
        }

        int issued = 0, done = 0;
        int* slotReq = stackalloc int[_queueDepth];
        // Slots are claimed in order and recycled as completions arrive.
        int initial = Math.Min(_queueDepth, requests.Length);
        for (int s = 0; s < initial; s++) Issue(requests, s, ref issued, slotReq);
        while (done < requests.Length)
        {
            if (!GetQueuedCompletionStatus(_port, out uint nb, out _, out Ov* pov, uint.MaxValue) && pov == null)
                throw new IOException($"GetQueuedCompletionStatus failed (error {Marshal.GetLastPInvokeError()})");
            int slot = (int)(pov - _ovs);
            var r = requests[slotReq[slot]];
            if (nb < r.Need)
                throw new IOException($"short or failed read: {nb} of {r.Need} bytes at offset {r.Offset} (length {r.Length}, NTSTATUS 0x{(uint)pov->Internal:X8})");
            BytesRead += nb; RequestsCompleted++; done++;
            if (issued < requests.Length) Issue(requests, slot, ref issued, slotReq);
        }
    }

    private void Issue(ReadOnlySpan<Request> requests, int slot, ref int issued, int* slotReq)
    {
        var r = requests[issued];
        _ovs[slot] = default;
        _ovs[slot].Offset = (uint)(r.Offset & 0xffffffff);
        _ovs[slot].OffsetHigh = (uint)(r.Offset >> 32);
        slotReq[slot] = issued++;
        if (!ReadFile(_handles[r.File], r.Dest, (uint)r.Length, null, &_ovs[slot]))
        {
            int e = Marshal.GetLastPInvokeError();
            if (e != 997) throw new IOException($"ReadFile failed (error {e}) at offset {r.Offset} length {r.Length}");   // 997 = ERROR_IO_PENDING
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (nint h in _handles) if (h != 0 && h != -1) CloseHandle(h);
        if (_port != 0) CloseHandle(_port);
        if (_ovs != null) NativeMemory.Free(_ovs);
        if (_fallback is not null) foreach (var h in _fallback) h?.Dispose();
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Ov { public nuint Internal, InternalHigh; public uint Offset, OffsetHigh; public nint hEvent; }

    [LibraryImport("kernel32.dll", EntryPoint = "CreateFileW", StringMarshalling = StringMarshalling.Utf16, SetLastError = true)]
    private static partial nint CreateFileW(string name, uint access, uint share, nint security, uint disposition, uint flags, nint template);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool ReadFile(nint handle, byte* buffer, uint count, uint* read, Ov* overlapped);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial nint CreateIoCompletionPort(nint file, nint port, nuint key, uint threads);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetQueuedCompletionStatus(nint port, out uint bytes, out nuint key, out Ov* overlapped, uint milliseconds);

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CloseHandle(nint handle);
}
