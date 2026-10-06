using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DotLLM.Cli.Diagnostics;

/// <summary>
/// Opt-in host/device memory residency probe (<c>DOTLLM_MEM_PROBE=1</c>), issue #438.
/// </summary>
/// <remarks>
/// <para>
/// Every number this prints names the instrument that produced it, because the obvious
/// ones measure different things and conflating them has cost a full session before:
/// </para>
/// <list type="bullet">
///   <item><description><b>working set</b> (<c>Process.WorkingSet64</c>) — host pages
///   charged to this process. It does <b>not</b> include device-local Vulkan memory, and
///   it does not include mmap pages the OS has trimmed to the standby list.</description></item>
///   <item><description><b>QueryWorkingSetEx over the GGUF view</b> — the only instrument
///   that answers "is the mmap resident?" directly. It reports, per 4 KiB page of the
///   mapped region, whether that page is <i>valid in this process's working set</i>.
///   A trimmed-but-standby page reads as not-valid: still physical RAM, but reclaimable
///   by the OS without I/O.</description></item>
///   <item><description><b>GlobalMemoryStatusEx.ullAvailPhys</b> — machine-wide available
///   physical memory, the same counter WMI exposes as
///   <c>Win32_OperatingSystem.FreePhysicalMemory</c>. Standby pages count as
///   <i>available</i> here.</description></item>
/// </list>
/// <para>
/// Device-local bytes are NOT measured here — <c>VulkanDevice.MemorySnapshot()</c> is the
/// instrument for that, and the caller appends it.
/// </para>
/// </remarks>
internal static unsafe partial class HostResidencyProbe
{
    /// <summary>True when <c>DOTLLM_MEM_PROBE</c> is set to 1.</summary>
    public static bool Enabled { get; } =
        Environment.GetEnvironmentVariable("DOTLLM_MEM_PROBE") == "1";

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WorkingSetExInfo
    {
        public nint VirtualAddress;
        public nuint VirtualAttributes;
    }

    [LibraryImport("kernel32.dll", EntryPoint = "GlobalMemoryStatusEx", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GlobalMemoryStatusEx(ref MemoryStatusEx buffer);

    [LibraryImport("kernel32.dll", EntryPoint = "K32QueryWorkingSetEx", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool QueryWorkingSetEx(nint hProcess, void* pv, uint cb);

    [LibraryImport("kernel32.dll", EntryPoint = "GetCurrentProcess")]
    private static partial nint GetCurrentProcess();

    /// <summary>Machine-wide available physical bytes (GlobalMemoryStatusEx.ullAvailPhys).</summary>
    public static ulong AvailablePhysicalBytes()
    {
        var s = new MemoryStatusEx { dwLength = (uint)sizeof(MemoryStatusEx) };
        return GlobalMemoryStatusEx(ref s) ? s.ullAvailPhys : 0;
    }

    /// <summary>
    /// Page census of <paramref name="length"/> bytes at <paramref name="basePtr"/>:
    /// how many 4 KiB pages are currently valid in this process's working set, and how
    /// many of those are shared (file-backed pages are shared).
    /// </summary>
    /// <returns><c>(residentBytes, sharedBytes, totalBytes)</c>; resident is 0 on failure.</returns>
    public static (long Resident, long Shared, long Total) MappedResidency(nint basePtr, long length)
    {
        if (basePtr == 0 || length <= 0) return (0, 0, 0);

        const int PageSize = 4096;
        const int BatchPages = 65536;           // 1 MiB of PSAPI records per call
        long pages = length / PageSize;
        long resident = 0, shared = 0;
        nint process = GetCurrentProcess();

        var batch = new WorkingSetExInfo[BatchPages];
        fixed (WorkingSetExInfo* p = batch)
        {
            for (long start = 0; start < pages; start += BatchPages)
            {
                int n = (int)Math.Min(BatchPages, pages - start);
                for (int i = 0; i < n; i++)
                {
                    p[i].VirtualAddress = basePtr + (nint)((start + i) * PageSize);
                    p[i].VirtualAttributes = 0;
                }
                if (!QueryWorkingSetEx(process, p, (uint)(n * sizeof(WorkingSetExInfo))))
                    return (0, 0, pages * PageSize);

                for (int i = 0; i < n; i++)
                {
                    nuint flags = p[i].VirtualAttributes;
                    if ((flags & 1) == 0) continue;                     // Valid
                    resident += PageSize;
                    if ((flags & (1u << 15)) != 0) shared += PageSize;  // Shared
                }
            }
        }
        return (resident, shared, pages * PageSize);
    }

    /// <summary>
    /// Writes one labelled probe line to stderr (stdout stays clean for <c>--json</c>).
    /// </summary>
    /// <param name="label">Probe point, e.g. "after-load".</param>
    /// <param name="mapBase">Base of the GGUF tensor-data mapping, or 0 when unmapped.</param>
    /// <param name="mapLength">Byte length of that mapping.</param>
    /// <param name="deviceSnapshot">VulkanDevice.MemorySnapshot(), or null.</param>
    public static void Report(string label, nint mapBase, long mapLength, string? deviceSnapshot)
    {
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        static double MiB(long b) => b / (1024.0 * 1024.0);

        var (res, shr, tot) = MappedResidency(mapBase, mapLength);
        ulong avail = AvailablePhysicalBytes();

        Console.Error.WriteLine(
            $"[mem-probe:{label}] " +
            $"ws={MiB(proc.WorkingSet64):F0} MiB (Process.WorkingSet64); " +
            $"private={MiB(proc.PrivateMemorySize64):F0} MiB (Process.PrivateMemorySize64); " +
            $"ws-minus-private={MiB(proc.WorkingSet64 - proc.PrivateMemorySize64):F0} MiB; " +
            $"availphys={MiB((long)avail):F0} MiB (GlobalMemoryStatusEx); " +
            $"gguf-map={MiB(tot):F0} MiB, resident-in-ws={MiB(res):F0} MiB " +
            $"({(tot > 0 ? 100.0 * res / tot : 0):F1}%), of which shared={MiB(shr):F0} MiB " +
            "(K32QueryWorkingSetEx page census)" +
            (deviceSnapshot is null ? "" : $"; device[VulkanDevice.MemorySnapshot]: {deviceSnapshot}"));
    }
}
