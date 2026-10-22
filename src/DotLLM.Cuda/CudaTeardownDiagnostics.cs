using System.Diagnostics;
using System.Threading;

namespace DotLLM.Cuda;

/// <summary>
/// Records CUDA <c>*Destroy</c> results that the RAII wrappers deliberately do not throw on.
/// </summary>
/// <remarks>
/// <para>
/// Issue #484. <see cref="CudaContext.Dispose"/>, <see cref="CudaStream.Dispose"/> and
/// <see cref="CudaCublasHandle.Dispose"/> discard their destroy return codes, because throwing
/// from <c>Dispose</c> during an in-flight exception unwind would mask the original failure. The
/// side effect was that a <i>failing</i> teardown — the only way a correctly-written cleanup path
/// can still leak device memory — was completely invisible: the process saw the resource as freed
/// and only <c>cuMemGetInfo</c> disagreed.
/// </para>
/// <para>
/// This counter is the cheap middle ground: teardown still never throws, but a failed destroy is
/// counted and traced, so a leak-probe test can assert it stayed at zero and attribute a VRAM drop
/// to the driver rather than to our ownership bookkeeping.
/// </para>
/// </remarks>
public static class CudaTeardownDiagnostics
{
    private static int _failedDestroyCount;

    /// <summary>
    /// Number of CUDA/cuBLAS destroy calls that returned a non-success code in this process.
    /// Zero on a healthy run.
    /// </summary>
    public static int FailedDestroyCount => Volatile.Read(ref _failedDestroyCount);

    /// <summary>Resets <see cref="FailedDestroyCount"/> to zero. Intended for test isolation.</summary>
    public static void ResetFailedDestroyCount() => Volatile.Write(ref _failedDestroyCount, 0);

    /// <summary>
    /// Records the result of a destroy call. Non-zero results increment
    /// <see cref="FailedDestroyCount"/> and emit a trace line; success is free.
    /// </summary>
    /// <param name="resourceKind">Short name of the resource, e.g. <c>"CUcontext"</c>.</param>
    /// <param name="resultCode">The driver/cuBLAS status code returned by the destroy call.</param>
    internal static void RecordDestroy(string resourceKind, int resultCode)
    {
        if (resultCode == 0) return;
        Interlocked.Increment(ref _failedDestroyCount);
        Trace.WriteLine($"[dotLLM.Cuda] destroy of {resourceKind} failed with status {resultCode} — device memory may be retained.");
    }
}
