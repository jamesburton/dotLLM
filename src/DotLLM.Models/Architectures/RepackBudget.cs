using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Memory guard for the CPU R4 weight repack (#792).
/// </summary>
/// <remarks>
/// <para>
/// The R4 repack is a second, committed copy of every repackable projection held next to the
/// memory-mapped GGUF. On a host with plenty of RAM that is a pure win. On a host where
/// <c>weights + repack</c> does not fit in <i>available</i> physical memory (Gemma-4-31B Q4_K_M, 17.7 GB
/// + ~17 GB on a box with ~32 GB visible RAM) the OS evicts the very pages decode is about to read:
/// 0.1 tok/s against llama.cpp's 4.8 tok/s on the same file.
/// </para>
/// <para>
/// The decision is a pure function (<see cref="AllowedRepackBytes"/>) over four numbers so it can be
/// unit-tested on any machine; the instruments that feed it are <see cref="QueryAvailablePhysicalBytes"/>
/// (the OS's own counter, NOT <c>GCMemoryInfo</c>, which is unreliable before the first GC and reports the
/// GC heap limit rather than free RAM) and the GGUF data-section length.
/// </para>
/// <para>
/// Override with <c>DOTLLM_CPU_REPACK=always|never|auto</c> (default <c>auto</c>).
/// </para>
/// </remarks>
internal static partial class RepackBudget
{
    /// <summary>Fraction of available physical RAM that <c>weights + repack</c> may occupy.</summary>
    public const double DefaultSafeFraction = 0.80;

    /// <summary>Environment variable selecting <see cref="RepackMode"/>.</summary>
    public const string EnvVar = "DOTLLM_CPU_REPACK";

    /// <summary>How the repack budget is applied.</summary>
    public enum RepackMode
    {
        /// <summary>Budget-aware (default).</summary>
        Auto,
        /// <summary>Always repack everything (pre-#792 behaviour).</summary>
        Always,
        /// <summary>Never repack; keep the mmap weights only.</summary>
        Never,
    }

    /// <summary>Parses <see cref="EnvVar"/>; unknown or empty values mean <see cref="RepackMode.Auto"/>.</summary>
    public static RepackMode ParseMode(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        "always" or "1" or "on" or "true" or "force" => RepackMode.Always,
        "never" or "0" or "off" or "false" or "none" => RepackMode.Never,
        _ => RepackMode.Auto,
    };

    /// <summary>
    /// The number of repack bytes that may be allocated: <c>clamp(fraction * available - weightBytes, 0, candidateBytes)</c>.
    /// </summary>
    /// <param name="weightBytes">Resident size of the mapped weights (GGUF data section).</param>
    /// <param name="candidateBytes">Bytes a full repack would allocate.</param>
    /// <param name="availablePhysicalBytes">OS-reported available physical memory; <c>&lt;= 0</c> = unknown.</param>
    /// <param name="safeFraction">Fraction of available memory the weights plus repack may use.</param>
    /// <returns><paramref name="candidateBytes"/> when memory is unknown or ample; 0 to skip the repack entirely;
    /// otherwise the byte budget for a partial repack.</returns>
    public static long AllowedRepackBytes(long weightBytes, long candidateBytes, long availablePhysicalBytes,
                                          double safeFraction = DefaultSafeFraction)
    {
        if (candidateBytes <= 0) return 0;
        if (availablePhysicalBytes <= 0) return candidateBytes;   // unknown: keep the default behaviour
        long headroom = (long)(availablePhysicalBytes * safeFraction) - Math.Max(0, weightBytes);
        if (headroom <= 0) return 0;
        return Math.Min(headroom, candidateBytes);
    }

    /// <summary>Resolves the byte budget for the current process and environment.</summary>
    public static long Resolve(long weightBytes, long candidateBytes, RepackMode mode, long availablePhysicalBytes)
        => mode switch
        {
            RepackMode.Always => candidateBytes,
            RepackMode.Never => 0,
            _ => AllowedRepackBytes(weightBytes, candidateBytes, availablePhysicalBytes),
        };

    /// <summary>
    /// Machine-wide available physical memory in bytes (standby/cache counts as available), or 0 when it
    /// cannot be determined. Windows: <c>GlobalMemoryStatusEx.ullAvailPhys</c>; Linux: <c>MemAvailable</c>
    /// further capped by a cgroup v2 limit; other platforms: 0 (unknown, repack stays on).
    /// </summary>
    public static long QueryAvailablePhysicalBytes()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var s = new MemoryStatusEx { dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>() };
                return GlobalMemoryStatusEx(ref s) ? (long)s.ullAvailPhys : 0;
            }
            if (OperatingSystem.IsLinux())
            {
                long avail = 0;
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal)) continue;
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2 && long.TryParse(parts[1], out long kb)) avail = kb * 1024;
                    break;
                }
                // cgroup v2 container limit: headroom = memory.max - memory.current
                try
                {
                    string max = File.ReadAllText("/sys/fs/cgroup/memory.max").Trim();
                    if (long.TryParse(max, out long limit) && long.TryParse(File.ReadAllText("/sys/fs/cgroup/memory.current").Trim(), out long cur))
                        avail = avail > 0 ? Math.Min(avail, Math.Max(0, limit - cur)) : Math.Max(0, limit - cur);
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
                return avail;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or DllNotFoundException or EntryPointNotFoundException)
        {
        }
        return 0;
    }

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

    [LibraryImport("kernel32.dll", EntryPoint = "GlobalMemoryStatusEx", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GlobalMemoryStatusEx(ref MemoryStatusEx buffer);

    /// <summary>One-line, human-readable summary of a repack decision (null when the full repack ran).</summary>
    public static string? Describe(RepackMode mode, long weightBytes, long candidateBytes, long allowedBytes,
                                   long repackedBytes, int repackedLayers, int totalLayers, long availableBytes)
    {
        static string G(long b) => $"{b / (1024.0 * 1024 * 1024):F1} GiB";
        if (mode == RepackMode.Never)
            return $"CPU R4 repack: disabled ({EnvVar}=never); keeping mmap weights ({G(weightBytes)}).";
        if (repackedBytes >= candidateBytes) return null;
        string why = $"weights {G(weightBytes)} + repack {G(candidateBytes)} > {DefaultSafeFraction:P0} of {G(availableBytes)} available RAM";
        return repackedBytes == 0
            ? $"CPU R4 repack: skipped ({why}); keeping mmap weights. Set {EnvVar}=always to override."
            : $"CPU R4 repack: partial ({why}); repacked {repackedLayers}/{totalLayers} layers ({G(repackedBytes)} of {G(candidateBytes)}), rest stay on the mmap weights. Set {EnvVar}=always to override.";
    }
}
