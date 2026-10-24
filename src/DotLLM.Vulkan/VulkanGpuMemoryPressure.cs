using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace DotLLM.Vulkan;

/// <summary>One process's GPU memory on one adapter (Windows "GPU Process Memory" counter).</summary>
internal readonly record struct GpuProcessUsage(int Pid, long SharedBytes, long DedicatedBytes, string? Name = null)
{
    /// <summary>Total bytes this process holds on the adapter.</summary>
    public long TotalBytes => SharedBytes + DedicatedBytes;
}

/// <summary>
/// What other processes hold of the GPU memory this process also draws on (#880).
/// </summary>
/// <remarks>
/// <b>Why not <c>VK_EXT_memory_budget</c>.</b> Measured on gfx1151 / Windows (2026-10-09, <c>tools/vk-alloc-probe --mode hold</c>):
/// with a second process holding 20 GiB on the device-local heap, this process still read <c>usage=0</c> and an unchanged
/// <c>budget=66,291 MiB</c>. The driver's budget/usage are per-process (DXGI <c>QueryVideoMemoryInfo</c>), so they never see
/// another process. The only in-process source for system-wide GPU memory use is the OS's own accounting: the PDH
/// <c>GPU Process Memory</c> counters, which are per (pid, adapter LUID) and split into Dedicated and Shared usage.
/// </remarks>
internal sealed class VulkanGpuMemoryPressure
{
    /// <summary>Per-process usage of OTHER processes (this pid excluded), largest first.</summary>
    public IReadOnlyList<GpuProcessUsage> Others { get; }

    /// <summary>Bytes of the other processes' usage that compete with this device's resident heaps.</summary>
    public long OtherBytes { get; }

    private readonly bool _integrated;

    /// <summary>Creates a pressure reading. <paramref name="integrated"/> selects shared+dedicated (UMA) versus dedicated only (discrete).</summary>
    public VulkanGpuMemoryPressure(IEnumerable<GpuProcessUsage> allProcesses, int ownPid, bool integrated)
    {
        _integrated = integrated;
        var others = allProcesses.Where(p => p.Pid != ownPid && Relevant(p, integrated) > 0)
            .OrderByDescending(p => Relevant(p, integrated)).ToList();
        Others = others;
        OtherBytes = others.Sum(p => Relevant(p, integrated));
    }

    /// <summary>Bytes of <paramref name="p"/> that count against the resident capacity: UMA = shared + dedicated; discrete = dedicated (VRAM) only.</summary>
    internal static long Relevant(GpuProcessUsage p, bool integrated) => integrated ? p.TotalBytes : p.DedicatedBytes;

    /// <summary>"pid 1234 (llama-server) 31.2 GiB, ..." for the top <paramref name="max"/> holders above 256 MiB.</summary>
    public string DescribeCulprits(int max = 5)
    {
        var sb = new StringBuilder();
        foreach (var p in Others.Where(p => Relevant(p, _integrated) >= 256L << 20).Take(max))
        {
            if (sb.Length > 0) sb.Append(", ");
            sb.Append(CultureInfo.InvariantCulture, $"pid {p.Pid} ({p.Name ?? "?"}) {Relevant(p, _integrated) / (double)(1L << 30):F1} GiB");
        }
        return sb.Length == 0 ? "none above 256 MiB" : sb.ToString();
    }

    /// <summary>Reads the live pressure for the adapter with <paramref name="luid"/>; null when unavailable (non-Windows, counters missing, or no LUID).</summary>
    public static VulkanGpuMemoryPressure? TryRead(ulong? luid, bool integrated)
    {
        if (luid is not { } l || !OperatingSystem.IsWindows()) return null;
        try
        {
            var procs = VulkanGpuMemoryCounters.ReadProcessUsage(l);
            return procs is null ? null : new VulkanGpuMemoryPressure(procs, Environment.ProcessId, integrated);
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }
    }
}

/// <summary>PDH access to <c>\GPU Process Memory(*)</c> (Windows) and the instance-name parser.</summary>
internal static partial class VulkanGpuMemoryCounters
{
    /// <summary>Parses <c>pid_10840_luid_0x00000000_0x00010d52_phys_0</c> into (pid, luid = high&lt;&lt;32 | low).</summary>
    internal static bool TryParseInstance(string name, out int pid, out ulong luid)
    {
        pid = 0; luid = 0;
        var parts = name.Split('_');
        // pid, N, luid, 0xHIGH, 0xLOW, phys, K
        if (parts.Length < 5 || parts[0] != "pid" || parts[2] != "luid") return false;
        if (!int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out pid)) return false;
        if (!TryHex(parts[3], out ulong hi) || !TryHex(parts[4], out ulong lo)) return false;
        luid = (hi << 32) | (lo & 0xFFFFFFFFu);
        return true;

        static bool TryHex(string s, out ulong v)
        {
            if (s.StartsWith("0x", StringComparison.OrdinalIgnoreCase)) s = s[2..];
            return ulong.TryParse(s, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out v);
        }
    }

    /// <summary>Per-process usage on adapter <paramref name="luid"/>; null if the counters cannot be read.</summary>
    internal static List<GpuProcessUsage>? ReadProcessUsage(ulong luid)
    {
        var shared = ReadArray("\\GPU Process Memory(*)\\Shared Usage");
        var dedicated = ReadArray("\\GPU Process Memory(*)\\Dedicated Usage");
        if (shared is null && dedicated is null) return null;
        var map = new Dictionary<int, (long S, long D)>();
        void Fold(List<(string Name, long Value)>? src, bool isShared)
        {
            if (src is null) return;
            foreach (var (name, value) in src)
            {
                if (!TryParseInstance(name, out int pid, out ulong l) || l != luid) continue;
                map.TryGetValue(pid, out var cur);
                map[pid] = isShared ? (cur.S + value, cur.D) : (cur.S, cur.D + value);
            }
        }
        Fold(shared, true); Fold(dedicated, false);
        var result = new List<GpuProcessUsage>(map.Count);
        foreach (var (pid, (s, d)) in map)
            result.Add(new GpuProcessUsage(pid, s, d, ProcessName(pid)));
        return result;
    }

    private static string? ProcessName(int pid)
    {
        try { using var p = Process.GetProcessById(pid); return p.ProcessName; }
        catch (ArgumentException) { return null; }
        catch (InvalidOperationException) { return null; }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PdhItem { public nint Name; public uint Status; public uint Pad; public long Value; }

    private const uint PdhMoreData = 0x800007D2;
    private const uint PdhFmtLarge = 0x00000400;

    [LibraryImport("pdh.dll", EntryPoint = "PdhOpenQueryW")]
    private static partial uint PdhOpenQuery(nint dataSource, nint userData, out nint query);
    [LibraryImport("pdh.dll", EntryPoint = "PdhAddEnglishCounterW", StringMarshalling = StringMarshalling.Utf16)]
    private static partial uint PdhAddEnglishCounter(nint query, string path, nint userData, out nint counter);
    [LibraryImport("pdh.dll")]
    private static partial uint PdhCollectQueryData(nint query);
    [LibraryImport("pdh.dll", EntryPoint = "PdhGetFormattedCounterArrayW")]
    private static unsafe partial uint PdhGetFormattedCounterArray(nint counter, uint format, ref uint bufferSize, ref uint itemCount, byte* buffer);
    [LibraryImport("pdh.dll")]
    private static partial uint PdhCloseQuery(nint query);

    private static unsafe List<(string, long)>? ReadArray(string path)
    {
        if (!OperatingSystem.IsWindows()) return null;
        if (PdhOpenQuery(0, 0, out nint q) != 0) return null;
        try
        {
            if (PdhAddEnglishCounter(q, path, 0, out nint c) != 0) return null;
            if (PdhCollectQueryData(q) != 0) return null;
            uint size = 0, count = 0;
            if (PdhGetFormattedCounterArray(c, PdhFmtLarge, ref size, ref count, null) != PdhMoreData) return null;
            var buf = new byte[size];
            fixed (byte* p = buf)
            {
                if (PdhGetFormattedCounterArray(c, PdhFmtLarge, ref size, ref count, p) != 0) return null;
                var list = new List<(string, long)>((int)count);
                var items = (PdhItem*)p;
                for (int i = 0; i < count; i++)
                {
                    if (items[i].Status != 0) continue;
                    list.Add((Marshal.PtrToStringUni(items[i].Name) ?? "", items[i].Value));
                }
                return list;
            }
        }
        finally { PdhCloseQuery(q); }
    }
}
