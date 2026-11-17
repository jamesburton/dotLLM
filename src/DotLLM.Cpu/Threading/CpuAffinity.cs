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

    /// <summary>
    /// Whether compute threads opt out of Windows execution-speed power throttling (EcoQoS). On by default;
    /// <c>DOTLLM_CPU_POWER_THROTTLE_OPTOUT=0</c> keeps the OS default (A/B baseline, #1675).
    /// </summary>
    /// <remarks>
    /// llama.cpp's ggml-cpu does the same for every compute thread at default priority
    /// (<c>ggml_thread_apply_priority</c>). Without it, Windows 11 may treat the threads of a windowless or background
    /// process as low-QoS work: it packs them onto few cores or E-cores and runs those at reduced clocks. A spin barrier
    /// then waits on the slowest thread. The flag is a scheduling hint only; it changes no result.
    /// </remarks>
    internal static readonly bool PowerThrottleOptOutEnabled =
        Environment.GetEnvironmentVariable("DOTLLM_CPU_POWER_THROTTLE_OPTOUT") != "0";

    /// <summary>
    /// Opts the current thread out of Windows execution-speed power throttling (EcoQoS), as llama.cpp does for its
    /// compute threads. No-op (returns <c>false</c>) on other OSes, when disabled via
    /// <see cref="PowerThrottleOptOutEnabled"/>, or when the call fails (Windows before 10 1709). Never throws.
    /// </summary>
    /// <returns><c>true</c> if the OS accepted the request.</returns>
    public static bool OptOutOfPowerThrottling()
    {
        if (!PowerThrottleOptOutEnabled || !OperatingSystem.IsWindows())
            return false;
        return SetExecutionSpeedThrottling(throttled: false);
    }

    /// <summary>
    /// Sets the current thread's execution-speed power-throttling state explicitly: <paramref name="throttled"/> =
    /// <c>false</c> opts out (always full speed), <c>true</c> opts in (EcoQoS). Windows only; never throws.
    /// </summary>
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    internal static unsafe bool SetExecutionSpeedThrottling(bool throttled)
    {
        try
        {
            var state = new WindowsNative.ThreadPowerThrottlingState
            {
                Version = WindowsNative.ThreadPowerThrottlingCurrentVersion,
                ControlMask = WindowsNative.ThreadPowerThrottlingExecutionSpeed,
                StateMask = throttled ? WindowsNative.ThreadPowerThrottlingExecutionSpeed : 0u,
            };
            return WindowsNative.SetThreadInformation(
                WindowsNative.GetCurrentThread(), WindowsNative.ThreadPowerThrottling,
                &state, (uint)sizeof(WindowsNative.ThreadPowerThrottlingState));
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Reads the current thread's execution-speed power-throttling state: <c>(ControlMask, StateMask)</c> bit 0 =
    /// EXECUTION_SPEED. A thread that opted out reports control 1 / state 0. Returns null when the query fails or off
    /// Windows. Test and diagnostic use.
    /// </summary>
    internal static unsafe (uint ControlMask, uint StateMask)? QueryExecutionSpeedThrottling()
    {
        if (!OperatingSystem.IsWindows())
            return null;
        try
        {
            var state = new WindowsNative.ThreadPowerThrottlingState { Version = WindowsNative.ThreadPowerThrottlingCurrentVersion };
            bool ok = WindowsNative.GetThreadInformation(
                WindowsNative.GetCurrentThread(), WindowsNative.ThreadPowerThrottling,
                &state, (uint)sizeof(WindowsNative.ThreadPowerThrottlingState));
            return ok ? (state.ControlMask, state.StateMask) : null;
        }
        catch
        {
            return null;
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static unsafe partial class WindowsNative
    {
        /// <summary><c>THREAD_INFORMATION_CLASS.ThreadPowerThrottling</c>.</summary>
        internal const int ThreadPowerThrottling = 3;

        /// <summary><c>THREAD_POWER_THROTTLING_CURRENT_VERSION</c>.</summary>
        internal const uint ThreadPowerThrottlingCurrentVersion = 1;

        /// <summary><c>THREAD_POWER_THROTTLING_EXECUTION_SPEED</c>.</summary>
        internal const uint ThreadPowerThrottlingExecutionSpeed = 0x1;

        /// <summary><c>THREAD_POWER_THROTTLING_STATE</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct ThreadPowerThrottlingState
        {
            public uint Version;
            public uint ControlMask;
            public uint StateMask;
        }

        [LibraryImport("kernel32.dll")]
        internal static partial nint GetCurrentThread();

        [LibraryImport("kernel32.dll")]
        internal static partial nint SetThreadAffinityMask(nint hThread, nint dwThreadAffinityMask);

        [LibraryImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool SetThreadInformation(nint hThread, int threadInformationClass, void* threadInformation, uint threadInformationSize);

        [LibraryImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool GetThreadInformation(nint hThread, int threadInformationClass, void* threadInformation, uint threadInformationSize);
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
    }
}
