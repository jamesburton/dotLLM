using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DotLLM.Cuda;

/// <summary>
/// Detects WDDM "sysmem fallback" at runtime (issue #1068). On Windows, when a CUDA process allocates past dedicated
/// VRAM the driver backs the excess with shared system memory instead of failing the allocation, and every kernel that
/// touches it then reads over PCIe. Nothing in CUDA reports this; Windows' <c>GPU Process Memory</c> performance counters
/// do (<c>Shared Usage</c> per process). This polls that counter at most every <see cref="PollInterval"/> and logs a loud
/// warning each time this process's shared usage rises another <see cref="WarnStepBytes"/> above its first reading.
/// A no-op off Windows (Linux has no such fallback: the allocation fails instead).
/// </summary>
public sealed partial class GpuSpillMonitor : IDisposable
{
    /// <summary>Growth of shared usage above the baseline that triggers each warning.</summary>
    public const long WarnStepBytes = 256L << 20;

    /// <summary>Minimum time between two counter reads; a read costs a PDH collection.</summary>
    public static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(5);

    private readonly Func<long?> _readSharedBytes;
    private readonly Action<string> _warn;
    private readonly Stopwatch _sinceLastPoll = new();
    private long? _baseline;
    private long _warnedAbove;
    private PdhSharedUsageReader? _reader;

    /// <summary>Creates a monitor for the current process, warning on stderr.</summary>
    public GpuSpillMonitor()
    {
        _reader = OperatingSystem.IsWindows() ? PdhSharedUsageReader.TryCreate(Environment.ProcessId) : null;
        _readSharedBytes = () => _reader?.Read();
        _warn = message => Console.Error.WriteLine(message);
    }

    /// <summary>Test seam: a monitor reading <paramref name="readSharedBytes"/> and reporting to <paramref name="warn"/>.</summary>
    /// <param name="readSharedBytes">Returns the process's shared GPU memory usage in bytes, or null when unknown.</param>
    /// <param name="warn">Receives each warning.</param>
    internal GpuSpillMonitor(Func<long?> readSharedBytes, Action<string> warn)
    {
        _readSharedBytes = readSharedBytes;
        _warn = warn;
    }

    /// <summary>Shared usage above the baseline at the last read, in bytes (0 before two reads).</summary>
    public long SpilledBytes { get; private set; }

    /// <summary>Reads the counter unless the last read was less than <see cref="PollInterval"/> ago.</summary>
    /// <param name="context">What just happened, for the warning (e.g. "KV growth to 98304 tokens").</param>
    public void Poll(string context)
    {
        if (_sinceLastPoll.IsRunning && _sinceLastPoll.Elapsed < PollInterval) return;
        _sinceLastPoll.Restart();
        Check(context);
    }

    /// <summary>Reads the counter now and warns if shared usage has crossed a new <see cref="WarnStepBytes"/> step.</summary>
    /// <param name="context">What just happened, for the warning.</param>
    public void Check(string context)
    {
        long? shared = _readSharedBytes();
        if (shared is not long now) return;
        if (_baseline is not long baseline)
        {
            _baseline = now;
            return;
        }

        SpilledBytes = Math.Max(0, now - baseline);
        long step = SpilledBytes / WarnStepBytes * WarnStepBytes;
        if (step < WarnStepBytes || step <= _warnedAbove) return;
        _warnedAbove = step;
        _warn($"[dotllm] WARNING: GPU memory has spilled into shared system memory: this process's shared GPU usage is " +
              $"{SpilledBytes >> 20} MiB above its post-load level ({context}). The driver is backing device buffers with system RAM " +
              "(WDDM sysmem fallback), so every kernel touching them runs at PCIe speed and generation slows by orders of " +
              "magnitude. Lower the context ('dotllm serve --ctx-size'), use a smaller KV type, close other GPU programs, or set " +
              "NVIDIA Control Panel > Manage 3D settings > 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' so the " +
              "allocation fails instead (see docs/SERVER.md).");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _reader?.Dispose();
        _reader = null;
    }

    /// <summary>
    /// Reads <c>\GPU Process Memory(pid_N_*)\Shared Usage</c> through PDH, summed over the process's adapter instances.
    /// </summary>
    private sealed partial class PdhSharedUsageReader : IDisposable
    {
        private const uint PdhFmtLarge = 0x00000400;
        private const uint PdhMoreData = 0x800007D2;
        private nint _query;
        private readonly nint _counter;

        private PdhSharedUsageReader(nint query, nint counter)
        {
            _query = query;
            _counter = counter;
        }

        public static PdhSharedUsageReader? TryCreate(int pid)
        {
            try
            {
                if (PdhOpenQueryW(null, 0, out nint query) != 0) return null;
                string path = $@"\GPU Process Memory(pid_{pid}_*)\Shared Usage";
                if (PdhAddEnglishCounterW(query, path, 0, out nint counter) != 0)
                {
                    PdhCloseQuery(query);
                    return null;
                }
                return new PdhSharedUsageReader(query, counter);
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return null;
            }
        }

        public unsafe long? Read()
        {
            if (_query == 0 || PdhCollectQueryData(_query) != 0) return null;
            uint size = 0;
            uint status = PdhGetFormattedCounterArrayW(_counter, PdhFmtLarge, ref size, out uint count, 0);
            if (status != PdhMoreData || size == 0) return null;
            nint heap = Marshal.AllocHGlobal((nint)size);
            try
            {
                if (PdhGetFormattedCounterArrayW(_counter, PdhFmtLarge, ref size, out count, heap) != 0) return null;
                long total = 0;
                var items = (FormattedItem*)heap;
                for (int i = 0; i < count; i++)
                    if (items[i].CStatus == 0) total += items[i].LargeValue;
                return total;
            }
            finally
            {
                Marshal.FreeHGlobal(heap);
            }
        }

        public void Dispose()
        {
            if (_query != 0) PdhCloseQuery(_query);
            _query = 0;
        }

        // PDH_FMT_COUNTERVALUE_ITEM_W on 64-bit: name pointer, then PDH_FMT_COUNTERVALUE { DWORD CStatus; union { LONGLONG ... } }.
        [StructLayout(LayoutKind.Sequential)]
        private struct FormattedItem
        {
            public nint Name;
            public uint CStatus;
            public long LargeValue;
        }

        [LibraryImport("pdh.dll", StringMarshalling = StringMarshalling.Utf16)]
        private static partial uint PdhOpenQueryW(string? dataSource, nint userData, out nint query);

        [LibraryImport("pdh.dll", StringMarshalling = StringMarshalling.Utf16)]
        private static partial uint PdhAddEnglishCounterW(nint query, string fullCounterPath, nint userData, out nint counter);

        [LibraryImport("pdh.dll")]
        private static partial uint PdhCollectQueryData(nint query);

        [LibraryImport("pdh.dll", StringMarshalling = StringMarshalling.Utf16)]
        private static partial uint PdhGetFormattedCounterArrayW(nint counter, uint format, ref uint bufferSize, out uint itemCount, nint itemBuffer);

        [LibraryImport("pdh.dll")]
        private static partial uint PdhCloseQuery(nint query);
    }
}
