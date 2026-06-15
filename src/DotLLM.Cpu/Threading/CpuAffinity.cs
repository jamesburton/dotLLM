using System.Runtime.InteropServices;

namespace DotLLM.Cpu.Threading;

/// <summary>
/// Pins the current thread to a specific logical processor.
/// Supports Windows (<c>SetThreadAffinityMask</c>) and Linux (<c>sched_setaffinity</c>).
/// Returns false on failure or unsupported platforms — never throws.
/// </summary>
internal static partial class CpuAffinity
{
    /// <summary>
    /// Pins the current thread to the specified logical processor.
    /// </summary>
    /// <param name="logicalProcessorId">OS logical processor index (0-based).</param>
    /// <returns><c>true</c> if pinning succeeded, <c>false</c> on failure or unsupported platform.</returns>
    public static bool PinCurrentThread(int logicalProcessorId)
    {
        if (logicalProcessorId < 0)
            return false;

        if (OperatingSystem.IsWindows())
            return PinWindows(logicalProcessorId);
        if (OperatingSystem.IsLinux())
            return PinLinux(logicalProcessorId);

        // macOS: thread_policy_set with THREAD_AFFINITY_POLICY is advisory and
        // doesn't guarantee pinning. Return false to indicate unsupported.
        return false;
    }

    /// <summary>
    /// Returns the logical processor the calling thread is currently running on, or <c>-1</c> on
    /// unsupported platforms. Cheap (no real syscall on modern Windows/Linux). For an unpinned thread
    /// the value is a snapshot — it can change after the call — so callers use it only to bias a
    /// performance choice (e.g. a P-core vs E-core kernel), never for correctness.
    /// </summary>
    public static int GetCurrentProcessorId()
    {
        if (OperatingSystem.IsWindows())
            return (int)WindowsNative.GetCurrentProcessorNumber();
        if (OperatingSystem.IsLinux())
            return LinuxNative.sched_getcpu();
        return -1;
    }

    // ── Windows ──

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static bool PinWindows(int logicalProcessorId)
    {
        if (logicalProcessorId >= 64)
            return false; // Would need processor groups — not supported

        try
        {
            nint mask = (nint)(1UL << logicalProcessorId);
            nint result = WindowsNative.SetThreadAffinityMask(WindowsNative.GetCurrentThread(), mask);
            return result != 0;
        }
        catch
        {
            return false;
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static partial class WindowsNative
    {
        [LibraryImport("kernel32.dll")]
        internal static partial nint GetCurrentThread();

        [LibraryImport("kernel32.dll")]
        internal static partial nint SetThreadAffinityMask(nint hThread, nint dwThreadAffinityMask);

        [LibraryImport("kernel32.dll")]
        internal static partial uint GetCurrentProcessorNumber();
    }

    // ── Linux ──

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static bool PinLinux(int logicalProcessorId)
    {
        try
        {
            // Always allocate the full cpu_set_t (128 bytes = 1024 bits = CPU_SETSIZE).
            // Dynamically-sized masks may be rejected by older kernels that expect
            // the standard cpumask_size. 128 bytes covers up to 1024 CPUs.
            const int CpuSetSize = 128; // bytes
            const int UlongCount = CpuSetSize / sizeof(ulong); // 16

            Span<ulong> mask = stackalloc ulong[UlongCount];
            mask.Clear();
            mask[logicalProcessorId / 64] = 1UL << (logicalProcessorId % 64);

            int result;
            unsafe
            {
                fixed (ulong* pMask = mask)
                {
                    result = LinuxNative.sched_setaffinity(0, (nuint)CpuSetSize, pMask);
                }
            }
            return result == 0;
        }
        catch
        {
            return false;
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static unsafe partial class LinuxNative
    {
        [LibraryImport("libc")]
        internal static partial int sched_setaffinity(int pid, nuint cpusetsize, ulong* mask);

        [LibraryImport("libc")]
        internal static partial int sched_getcpu();
    }
}
