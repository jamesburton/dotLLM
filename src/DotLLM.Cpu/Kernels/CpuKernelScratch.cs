using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Deterministic release of the per-thread native scratch the CPU kernels keep between calls (#1892, #1927): the attention tile
/// holder (TurboQuant decode ring, packed F32 K/V tile, split-KV scores/accumulators), the split-KV partials, and the Q4_K row-lane
/// tile. Each holder is finalizable, so a thread's scratch is also released when the thread exits; this trims it now.
/// </summary>
public static unsafe class CpuKernelScratch
{
    /// <summary>
    /// Frees the scratch of the calling thread and, when <paramref name="pool"/> is given, of every pool worker (a broadcast dispatch;
    /// the caller runs as worker 0). Idempotent; buffers re-grow on the next call. Model hosts call it on dispose and
    /// <see cref="ComputeThreadPool"/> calls it before tearing its workers down.
    /// </summary>
    public static void Trim(ComputeThreadPool? pool)
    {
        if (pool is null) { TrimWorker(0, 0, 1); return; }
        pool.Dispatch(0, &TrimWorker);
    }

    internal static void TrimWorker(nint _, int __, int ___)
    {
        Attention.FreeThreadScratch();
        MatMul.FreeRowLaneTile();
    }
}
