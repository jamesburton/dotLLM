using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #484 attribution probe. Isolates the three CUDA primitives that every model factory
/// creates before it can throw — <see cref="CudaContext"/>, <see cref="CudaStream"/>,
/// <see cref="CudaCublasHandle"/> — and measures each one's create/destroy cycle against
/// <c>cuMemGetInfo</c> on its own.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> #484 reported ~30 MB of device memory lost per failed bad-<c>ptxDir</c>
/// load on an RTX 3060, at two commits that already carried #383's dispose-on-failure catch
/// blocks. Static analysis of that path rules out an orphaned allocation: <c>new CudaKernels</c>
/// throws at its first <c>File.ReadAllBytes</c>, before a single <c>cuModuleLoad</c> or
/// <c>cuMemAlloc</c>, so the only CUDA work per iteration was
/// <c>cuCtxCreate → cuStreamCreate → cublasCreate → [throw] → cublasDestroy → cuStreamDestroy →
/// cuCtxDestroy</c>, all of which the catch block ran. #484's fix makes that moot by validating the
/// PTX directory before <c>cuCtxCreate</c>, so nothing is allocated at all — but that removes the
/// symptom without naming its cause.
/// </para>
/// <para>
/// <b>What this test decides.</b> The four blocks below are a factorial over the trio, so a
/// monotonic VRAM drop is attributed to a specific primitive rather than to "a failed load".
/// Every destroy goes through the wrappers, and
/// <see cref="CudaTeardownDiagnostics.FailedDestroyCount"/> is asserted to stay at zero —
/// distinguishing "the driver refused to destroy the resource" (our bug) from "the driver returned
/// success and kept the memory anyway" (a driver/WDDM behaviour, and then #484's real answer is
/// that the original test's measurement was wrong). The context-only block runs the most
/// iterations because a context is the largest of the three and the prime suspect for 30 MB.
/// </para>
/// <para>
/// It is deliberately tolerant — it reports rather than asserting a tight bound on the per-cycle
/// deltas, except for one coarse guard on the full trio — because its job on the next T5500 run is
/// to produce an attribution, not to gate the build on an unexplained driver number.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CudaPrimitiveTeardownLeakTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Creates the probe with xUnit's output sink.</summary>
    public CudaPrimitiveTeardownLeakTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void CreateDestroyCycles_DoNotRetainDeviceMemory()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        CudaTeardownDiagnostics.ResetFailedDestroyCount();

        // A long-lived probe context: cuMemGetInfo queries whatever context is current, and the
        // cycles below destroy theirs. Created first so the one-time driver init cost is paid
        // before any measurement.
        using var probe = CudaContext.Create(deviceId: 0);
        long warmup = FreeBytes(probe);
        _output.WriteLine($"Free VRAM after probe context: {warmup / (1024 * 1024)} MB");

        long ctxOnly = Measure(probe, "context only", iterations: 20, () =>
        {
            using var c = CudaContext.Create(deviceId: 0);
        });

        long ctxStream = Measure(probe, "context + stream", iterations: 10, () =>
        {
            using var c = CudaContext.Create(deviceId: 0);
            using var s = CudaStream.Create();
        });

        long ctxCublas = Measure(probe, "context + cuBLAS", iterations: 10, () =>
        {
            using var c = CudaContext.Create(deviceId: 0);
            using var b = CudaCublasHandle.Create();
        });

        long trio = Measure(probe, "context + stream + cuBLAS (the failed-load footprint)", iterations: 10, () =>
        {
            using var c = CudaContext.Create(deviceId: 0);
            using var s = CudaStream.Create();
            using var b = CudaCublasHandle.Create();
            b.SetStream(s);
        });

        _output.WriteLine(
            $"Per-cycle retention: ctx={PerCycle(ctxOnly, 20)}, ctx+stream={PerCycle(ctxStream, 10)}, "
            + $"ctx+cublas={PerCycle(ctxCublas, 10)}, trio={PerCycle(trio, 10)}");

        // Our own bookkeeping must be clean regardless of what the driver does with the memory:
        // a non-zero count means a destroy call actually failed, which IS a dotLLM bug.
        Assert.Equal(0, CudaTeardownDiagnostics.FailedDestroyCount);

        // Coarse guard on the shape #484 reported: ~30 MB x 10 cycles would be ~300 MB.
        const long TrioCeilingBytes = 64L * 1024 * 1024;
        Assert.True(trio < TrioCeilingBytes,
            $"10 create/destroy cycles of context+stream+cuBLAS retained {trio / (1024.0 * 1024):F1} MB "
            + $"(ctx-only {ctxOnly / (1024.0 * 1024):F1} MB, ctx+stream {ctxStream / (1024.0 * 1024):F1} MB, "
            + $"ctx+cuBLAS {ctxCublas / (1024.0 * 1024):F1} MB across their own cycle counts) — "
            + "this is the #484 leak, and the per-block figures name which primitive owns it.");
    }

    private long Measure(CudaContext probe, string label, int iterations, Action cycle)
    {
        // One untimed cycle first: the first context/stream/cuBLAS of a given kind pays one-off
        // driver and library initialisation that is not a per-cycle cost.
        cycle();
        long before = FreeBytes(probe);
        for (int i = 0; i < iterations; i++) cycle();
        long after = FreeBytes(probe);
        long retained = before - after;
        _output.WriteLine(
            $"{label}: {iterations} cycles, free {before / (1024 * 1024)} MB -> {after / (1024 * 1024)} MB "
            + $"(retained {retained / (1024.0 * 1024):F1} MB)");
        return retained;
    }

    private static string PerCycle(long retained, int iterations)
        => $"{retained / (1024.0 * 1024) / iterations:F2} MB";

    private static long FreeBytes(CudaContext probe)
    {
        probe.MakeCurrent();
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
        return (long)free;
    }
}
