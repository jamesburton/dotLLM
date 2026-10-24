using System.Diagnostics;
using System.Text;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Opt-in itemisation of device allocations (<c>DOTLLM_CUDA_MEM_TRACE=1</c>, #1211), the CUDA counterpart of
/// <c>DOTLLM_VULKAN_MEM_TRACE</c>. Every <c>cuMemAlloc_v2</c> / <c>cuMemCreate</c> is attributed to the first caller
/// outside the interop layer; allocations of 1 MiB or more are logged as they happen, and <see cref="Dump"/> prints the
/// live bytes per site next to the driver's own used figure (the difference is the context, modules and cuBLAS).
/// Off: one static bool read per allocation.
/// </summary>
internal static class CudaMemTrace
{
    /// <summary>True when <c>DOTLLM_CUDA_MEM_TRACE=1</c>.</summary>
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("DOTLLM_CUDA_MEM_TRACE") == "1";

    private static readonly object Gate = new();
    private static readonly Dictionary<nint, (long Bytes, string Site)> Live = new();
    private static long _liveBytes;
    private static long _peakBytes;

    /// <summary>Records an allocation.</summary>
    /// <param name="ptr">Device pointer or allocation handle.</param>
    /// <param name="bytes">Size in bytes.</param>
    internal static void OnAlloc(nint ptr, long bytes)
    {
        string site = Site();
        lock (Gate)
        {
            Live[ptr] = (bytes, site);
            _liveBytes += bytes;
            _peakBytes = Math.Max(_peakBytes, _liveBytes);
            if (bytes >= 1 << 20)
                Console.Error.WriteLine($"[cuda-mem] +{bytes / 1048576.0,9:F2} MiB  live {_liveBytes / 1048576.0,9:F1} MiB  {site}");
        }
    }

    /// <summary>Records a free.</summary>
    /// <param name="ptr">Device pointer or allocation handle.</param>
    internal static void OnFree(nint ptr)
    {
        lock (Gate)
        {
            if (Live.Remove(ptr, out var e))
                _liveBytes -= e.Bytes;
        }
    }

    private static long _lastUsed;

    /// <summary>Prints the driver's used VRAM and the tracked live bytes when the driver figure moved by 1 MiB or more.</summary>
    /// <param name="label">Where in the run the mark is taken.</param>
    internal static void Mark(string label)
    {
        if (!Enabled)
            return;
        _ = CudaDriverApi.cuMemGetInfo_v2(out nuint free, out nuint total);
        long used = (long)total - (long)free;
        if (Math.Abs(used - _lastUsed) < 1 << 20)
            return;
        Console.Error.WriteLine($"[cuda-mem] @ {label}: driver used {used / 1048576.0:F1} MiB (+{(used - _lastUsed) / 1048576.0:F1}), tracked live {_liveBytes / 1048576.0:F1} MiB, untracked {(used - _liveBytes) / 1048576.0:F1} MiB");
        _lastUsed = used;
    }

    /// <summary>Prints live bytes per allocation site, the tracked peak and the driver's used VRAM.</summary>
    /// <param name="label">Where in the run the dump is taken.</param>
    internal static void Dump(string label)
    {
        if (!Enabled)
            return;
        var sb = new StringBuilder();
        lock (Gate)
        {
            _ = CudaDriverApi.cuMemGetInfo_v2(out nuint free, out nuint total);
            sb.AppendLine($"[cuda-mem] === {label}: tracked live {_liveBytes / 1048576.0:F1} MiB, tracked peak {_peakBytes / 1048576.0:F1} MiB, driver used {((long)total - (long)free) / 1048576.0:F1} MiB");
            foreach (var g in Live.Values.GroupBy(v => v.Site).Select(g => (Site: g.Key, Bytes: g.Sum(v => v.Bytes), N: g.Count())).OrderByDescending(g => g.Bytes))
                sb.AppendLine($"[cuda-mem]   {g.Bytes / 1048576.0,10:F2} MiB  x{g.N,-5} {g.Site}");
        }
        Console.Error.Write(sb.ToString());
    }

    private static string Site()
    {
        var frames = new StackTrace(2, false).GetFrames();
        var sb = new StringBuilder();
        int n = 0;
        foreach (var f in frames)
        {
            var m = DiagnosticMethodInfo.Create(f);
            if (m is null || m.DeclaringTypeName.EndsWith(nameof(CudaDriverApi), StringComparison.Ordinal)
                || m.DeclaringTypeName.EndsWith(nameof(CudaMemTrace), StringComparison.Ordinal))
                continue;
            if (n > 0)
                sb.Append(" < ");
            sb.Append(m.DeclaringTypeName.AsSpan(m.DeclaringTypeName.LastIndexOf('.') + 1)).Append('.').Append(m.Name);
            if (++n == 3)
                break;
        }
        return sb.ToString();
    }
}
