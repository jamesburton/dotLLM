using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// The one admission check for runtime-sized CUDA device allocations (issue #1848). Under Windows WDDM a <c>cuMemAlloc</c> past
/// dedicated VRAM does not fail: the driver backs it with shared system memory, and every kernel touching it then reads over PCIe
/// (~17x slower than the CPU reading DDR5 on an eGPU link). So every allocation whose size depends on the request, the context or
/// the model (weights, KV caches, forward and attention scratch) asks here first, against <c>cuMemGetInfo</c> free memory (clamped
/// by <see cref="DeviceMemoryBudget.SimulatedBudgetEnvVar"/>) minus <see cref="ReserveBytes"/>, and is refused with
/// <see cref="DeviceMemoryBudget.DescribeRefusal"/> instead of spilling. Only growths ask: a
/// buffer that already has its size costs nothing per step.
/// </summary>
internal static class CudaAdmission
{
    private static readonly AsyncLocal<Func<long?>?> s_freeBytesOverride = new();

    /// <summary>
    /// Test seam: a fake free-memory source for the current async flow (and the tasks it starts). Returns the free bytes to admit
    /// against, or <see langword="null"/> for "no reading" (refused). Scoped by <see cref="AsyncLocal{T}"/> so a test cannot leak it
    /// into another test's allocations running in parallel.
    /// </summary>
    internal static Func<long?>? FreeBytesOverride
    {
        get => s_freeBytesOverride.Value;
        set => s_freeBytesOverride.Value = value;
    }

    /// <summary>Refusals issued by <see cref="TryFit"/>, process-wide (test observable).</summary>
    internal static long RefusalCount => Interlocked.Read(ref s_refusals);
    private static long s_refusals;

    /// <summary>Free VRAM every admission leaves untouched (<see cref="DeviceMemoryBudget.ReserveEnvVar"/>, default 256 MiB).</summary>
    internal static long ReserveBytes =>
        DeviceMemoryBudget.ResolveReserveBytes(Environment.GetEnvironmentVariable(DeviceMemoryBudget.ReserveEnvVar));

    /// <summary>
    /// Free bytes on the current context's device as admission sees them: the test override, else <c>cuMemGetInfo</c> (already
    /// clamped to the simulated budget). <see langword="null"/> when there is no reading.
    /// </summary>
    internal static long? QueryFreeBytes()
    {
        if (FreeBytesOverride is { } fake) return fake();
        try
        {
            return CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _) == 0 ? (long)free : null;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return null;   // no driver: the allocation that follows would fail anyway
        }
    }

    /// <summary>
    /// True when <paramref name="bytes"/> more can be allocated on the current device leaving <see cref="ReserveBytes"/> free.
    /// A missing reading refuses. <paramref name="releasedBytes"/> is memory the caller frees before it allocates (a buffer it
    /// replaces), which counts as free.
    /// </summary>
    /// <param name="bytes">Bytes the allocation needs.</param>
    /// <param name="freeBytes">The free bytes the decision used (0 when there was no reading).</param>
    /// <param name="releasedBytes">Bytes the caller frees first.</param>
    /// <returns>Whether the allocation is admitted.</returns>
    internal static bool TryFit(long bytes, out long freeBytes, long releasedBytes = 0)
    {
        freeBytes = 0;
        if (bytes <= 0) return true;
        bool ok = QueryFreeBytes() is long free
                  && DeviceMemoryBudget.Fits((freeBytes = free) + Math.Max(0, releasedBytes), bytes, ReserveBytes);
        if (!ok) Interlocked.Increment(ref s_refusals);
        return ok;
    }

    /// <summary>
    /// Admits <paramref name="bytes"/> or throws <see cref="DeviceMemoryAdmissionException"/> carrying the standard refusal. Call
    /// before allocating anything, so a refusal leaves nothing half-built.
    /// </summary>
    /// <param name="what">What the allocation is for, e.g. "KV cache (f16) for 98304 tokens".</param>
    /// <param name="bytes">Bytes the allocation needs.</param>
    /// <param name="releasedBytes">Bytes the caller frees first.</param>
    /// <param name="remedies">Replaces the default remedies sentence (context and KV type) where those are not the fix.</param>
    /// <exception cref="DeviceMemoryAdmissionException">The allocation does not fit.</exception>
    internal static void EnsureFits(string what, long bytes, long releasedBytes = 0, string? remedies = null)
    {
        if (!TryFit(bytes, out long free, releasedBytes))
            throw new DeviceMemoryAdmissionException(
                DeviceMemoryBudget.DescribeRefusal(what, bytes, free + Math.Max(0, releasedBytes), ReserveBytes, remedies));
    }

}
