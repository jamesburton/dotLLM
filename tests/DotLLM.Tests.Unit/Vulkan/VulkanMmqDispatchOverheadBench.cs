using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark quantifying fixed GPU-side dispatch/launch overhead
/// for the Q8_0 MMQ prefill kernel — the one structural hypothesis for the
/// dotLLM-vs-llama.cpp prefill gap that survived the closed #384-#389
/// warptile-shape/unroll-hint investigation (see <c>.docs/HANDOFF.md</c>).
/// </summary>
/// <remarks>
/// <para>
/// Two measurement modes at each shape, same batch of dispatches either way:
/// </para>
/// <list type="bullet">
///   <item><b>Pipelined</b> (no barrier between dispatches, mirrors
///     <see cref="VulkanResidualAddOverheadBench"/>'s methodology): the driver
///     is free to overlap dispatch N+1's launch with dispatch N's execution.
///     Approaches the true steady-state per-dispatch GPU cost.</item>
///   <item><b>Barrier-serialized</b> (a <c>ComputeToComputeBarrier</c> between
///     every dispatch, via <c>vkCmdWriteTimestamp</c> GPU timestamps around
///     each individual dispatch — NOT host <see cref="Stopwatch"/>, since a
///     host-side per-dispatch timer would just measure submit-call overhead,
///     not GPU execution): matches the real forward pass, where every
///     <c>RecordMatmul</c> call in <c>VulkanTransformerModel.cs</c> is
///     preceded by exactly this barrier. This is the number that actually
///     applies to production prefill.</item>
/// </list>
/// <para>
/// The DELTA between barrier-serialized and pipelined per-dispatch GPU time,
/// at a shape small enough that barrier drain dominates, estimates the fixed
/// per-dispatch+barrier launch cost independent of compute. Sweeping shape
/// size (dispatch count via varying <c>M</c>) lets that fixed cost be fit as
/// the intercept of a simple linear model
/// <c>T(workgroups) = T_fixed + T_per_workgroup * workgroups</c> — reported
/// directly via least-squares over the swept points, not eyeballed.
/// </para>
/// <para>Enable with <c>DOTLLM_MMQ_DISPATCH_OVERHEAD_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMmqDispatchOverheadBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 7;
    private const int Reps = 32; // dispatches per timed pass (batch size)

    private readonly ITestOutputHelper _output;
    public VulkanMmqDispatchOverheadBench(ITestOutputHelper output) => _output = output;

    // Single 64x64-tile-multiple shapes (K held at the minimum one 32-block,
    // so per-workgroup compute is ~fixed) sweeping workgroup count via M —
    // isolates the fixed-vs-linear split cleanly, independent of any real
    // model's actual dimensions. This alone is NOT enough to predict real
    // shapes (see SweepKBlocksAtFixedWorkgroups below) — it only samples
    // totalWork = workgroups*1 K-block, far below real shapes' totalWork.
    private static readonly int[] SweepWorkgroups = [1, 2, 4, 8, 16, 32, 64, 128];

    // Second sweep axis: fixed workgroup count, varying K-block count, so the
    // combined point set spans totalWork = workgroups*Kblocks up into the
    // range real shapes actually occupy (SmolLM K=576 -> 18 blocks, 3B
    // K=3072 -> 96 blocks). Fitting T = a + b*totalWork against BOTH sweeps
    // together (not just the workgroup sweep alone) avoids extrapolating a
    // slope measured at Kblocks=1 out to Kblocks=96, which silently assumes
    // compute doesn't scale with K — it does, so that extrapolation would
    // have badly under-predicted real-shape time and inflated the fixed-
    // overhead percentage.
    private const int KSweepWorkgroups = 64;
    private static readonly int[] SweepKBlocks = [1, 2, 4, 8, 16, 32, 64, 96];

    // Representative real-model shapes (K = full hidden/intermediate dim, so
    // these ALSO carry real compute, not just the minimal-K sweep above) to
    // translate the fitted fixed-cost model into "% of real dispatch time"
    // at the two scales this whole investigation has been comparing.
    private static readonly (string Tag, int M, int K, int N)[] RealShapes =
    [
        ("SmolLM  Q/K/V-ish  (M=576  K=576  N=512)",   576,  576, 512),
        ("SmolLM  gate/up-ish (M=1536 K=576  N=512)",  1536,  576, 512),
        ("3B      Q/K/V-ish  (M=3072 K=3072 N=512)",  3072, 3072, 512),
        ("3B      gate/up-ish (M=8192 K=3072 N=512)", 8192, 3072, 512),
    ];

    [SkippableFact]
    public unsafe void Bench_MmqDispatchOverhead()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_MMQ_DISPATCH_OVERHEAD_BENCH") == "1",
            "DOTLLM_MMQ_DISPATCH_OVERHEAD_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device lacks integer-dot-product support — MMQ unavailable.");
        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported.");

        float tsPeriodNs = device.TimestampPeriodNs;
        Skip.IfNot(tsPeriodNs > 0f, "Device does not report a usable timestamp period — cannot GPU-time individual dispatches.");

        nint queryPool = CreateQueryPool(device, Reps + 1);
        try
        {
            _output.WriteLine($"Device: {device.DeviceName}  timestampPeriod={tsPeriodNs} ns/tick");
            _output.WriteLine($"batch={Reps} dispatches/pass, {WarmupPasses} warmup + {Passes} passes (median)");
            _output.WriteLine("");
            _output.WriteLine("### Sweep 1: fixed K=32 (1 block), varying workgroup count 1..128 via M");
            _output.WriteLine("| workgroups | pipelined µs/dispatch | barrier-serialized µs/dispatch | delta |");
            _output.WriteLine("|---:|---:|---:|---:|");

            // totalWork = workgroups * Kblocks — the single regressor for the
            // combined fit below. This sweep holds Kblocks=1, so totalWork ==
            // workgroups here.
            var combinedPoints = new List<(double TotalWork, double BarrierUs)>();
            foreach (int wg in SweepWorkgroups)
            {
                int m = wg * 64; // one 64-row tile per workgroup on the M axis
                const int k = 32; // one Q8_0 block: minimal per-workgroup compute
                const int n = 64; // one N-tile, groupsY=1

                using var shape = AllocShape(device, m, k, n);
                double pipelinedUs = MeasurePipelined(device, mmq, shape, m, k, n);
                double barrierUs = MeasureBarrierSerializedGpu(device, mmq, shape, m, k, n, queryPool, tsPeriodNs);
                combinedPoints.Add((wg * 1, barrierUs));
                _output.WriteLine($"| {wg,10} | {pipelinedUs,20:F3} | {barrierUs,28:F3} | {(barrierUs - pipelinedUs),8:F3} |");
            }

            _output.WriteLine("");
            _output.WriteLine($"### Sweep 2: fixed {KSweepWorkgroups} workgroups, varying K-block count 1..96 (spans real shapes' K range)");
            _output.WriteLine("| K-blocks | pipelined µs/dispatch | barrier-serialized µs/dispatch | delta |");
            _output.WriteLine("|---:|---:|---:|---:|");
            foreach (int kb in SweepKBlocks)
            {
                const int m = KSweepWorkgroups * 64;
                int k = kb * 32;
                const int n = 64;

                using var shape = AllocShape(device, m, k, n);
                double pipelinedUs = MeasurePipelined(device, mmq, shape, m, k, n);
                double barrierUs = MeasureBarrierSerializedGpu(device, mmq, shape, m, k, n, queryPool, tsPeriodNs);
                combinedPoints.Add(((double)KSweepWorkgroups * kb, barrierUs));
                _output.WriteLine($"| {kb,8} | {pipelinedUs,20:F3} | {barrierUs,28:F3} | {(barrierUs - pipelinedUs),8:F3} |");
            }

            var (fixedUs, perTotalWorkUs) = FitLinear(combinedPoints);
            _output.WriteLine("");
            _output.WriteLine("### Combined fit (both sweeps together, regressor = workgroups x K-blocks)");
            _output.WriteLine($"T(totalWork) = {fixedUs:F3} µs (fixed per barrier-serialized dispatch) + {perTotalWorkUs:F4} µs per (workgroup x K-block)");
            _output.WriteLine("");

            _output.WriteLine("### Real-model shapes: DIRECT ground-truth measurement (not model-predicted)");
            _output.WriteLine("| shape | workgroups | K-blocks | measured barrier-serialized µs | fixed-overhead % (fitted T_fixed / measured) |");
            _output.WriteLine("|---|---:|---:|---:|---:|");
            var realMeasuredUs = new Dictionary<string, double>();
            foreach (var (tag, m, k, n) in RealShapes)
            {
                int groupsX = (m + 63) / 64;
                int groupsY = (n + 63) / 64;
                int workgroups = groupsX * groupsY;
                int kblocks = k / 32;

                using var shape = AllocShape(device, m, k, n);
                double measuredUs = MeasureBarrierSerializedGpu(device, mmq, shape, m, k, n, queryPool, tsPeriodNs);
                realMeasuredUs[tag] = measuredUs;
                double fixedPct = fixedUs / measuredUs * 100.0;
                _output.WriteLine($"| {tag} | {workgroups,10} | {kblocks,8} | {measuredUs,30:F2} | {fixedPct,15:F2}% |");
            }

            _output.WriteLine("");
            _output.WriteLine("### Real-model aggregate: SmolLM-135M prefill (30 layers, ~7 MMQ-adjacent dispatches/layer)");
            // Q, K, V, o_proj (4 square-ish dispatches) + gate, up, down (3
            // rectangular-ish dispatches) per layer, using DIRECTLY MEASURED
            // per-dispatch time from the ground-truth table above (not the
            // fitted model) — exact head/kv-head split doesn't change the
            // dispatch-count-driven overhead conclusion materially.
            double smolLmSquareUs = realMeasuredUs["SmolLM  Q/K/V-ish  (M=576  K=576  N=512)"];
            double smolLmRectUs = realMeasuredUs["SmolLM  gate/up-ish (M=1536 K=576  N=512)"];
            double perLayerUs = 4 * smolLmSquareUs + 3 * smolLmRectUs; // Q,K,V,o + gate,up,down
            double perLayerFixedUs = 7 * fixedUs;
            double totalUs = perLayerUs * 30;
            double totalFixedUs = perLayerFixedUs * 30;
            _output.WriteLine($"measured MMQ-only total (ground truth): {totalUs / 1000.0:F3} ms, of which fixed-dispatch-overhead: {totalFixedUs / 1000.0:F3} ms ({totalFixedUs / totalUs * 100.0:F2}%)");
        }
        finally
        {
            VulkanApi.vkDestroyQueryPool(device.Handle, queryPool, 0);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Measurement helpers
    // ─────────────────────────────────────────────────────────────

    private readonly record struct Shape(
        VulkanDevice.Buffer W, VulkanDevice.Buffer Xq, VulkanDevice.Buffer Xds, VulkanDevice.Buffer C) : IDisposable
    {
        public void Dispose() { W.Dispose(); Xq.Dispose(); Xds.Dispose(); C.Dispose(); }
    }

    private static Shape AllocShape(VulkanDevice device, int m, int k, int n)
    {
        int blocksPerRow = k / MatMulQ8_0MmqKernel.Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * MatMulQ8_0MmqKernel.Q8_0BlockBytes;
        long wBytes = (((long)m * rowBytes) + 3) & ~3L;
        var w = device.Allocate(wBytes);
        var xq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        var xds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        var c = device.Allocate((long)n * m * sizeof(float));

        var rng = new Random(unchecked(m * 131 + k * 7 + n));
        byte[] wBuf = new byte[wBytes];
        rng.NextBytes(wBuf);
        device.Upload(new ReadOnlySpan<byte>(wBuf), w);
        byte[] xqBuf = new byte[QuantizeQ8_1RowsKernel.PackedBytes(n, k)];
        rng.NextBytes(xqBuf);
        device.Upload(new ReadOnlySpan<byte>(xqBuf), xq);
        var xds2 = new float[(long)n * blocksPerRow * 2];
        for (int i = 0; i < xds2.Length; i += 2) { xds2[i] = 1.0f; xds2[i + 1] = 0.0f; }
        device.Upload(xds2, xds);

        return new Shape(w, xq, xds, c);
    }

    private static double MeasurePipelined(
        VulkanDevice device, MatMulQ8_0MmqKernel mmq, Shape shape, int m, int k, int n)
    {
        for (int i = 0; i < WarmupPasses; i++) RunPipelinedPass(device, mmq, shape, m, k, n);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = RunPipelinedPass(device, mmq, shape, m, k, n);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static double RunPipelinedPass(
        VulkanDevice device, MatMulQ8_0MmqKernel mmq, Shape shape, int m, int k, int n)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < Reps; i++)
            mmq.Record(ctx.CommandBuffer, shape.W, shape.Xq, shape.Xds, shape.C, m, k, n);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / Reps;
    }

    private static unsafe double MeasureBarrierSerializedGpu(
        VulkanDevice device, MatMulQ8_0MmqKernel mmq, Shape shape, int m, int k, int n,
        nint queryPool, float tsPeriodNs)
    {
        for (int i = 0; i < WarmupPasses; i++)
            RunBarrierSerializedGpuPass(device, mmq, shape, m, k, n, queryPool, tsPeriodNs);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++)
            us[p] = RunBarrierSerializedGpuPass(device, mmq, shape, m, k, n, queryPool, tsPeriodNs);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static unsafe double RunBarrierSerializedGpuPass(
        VulkanDevice device, MatMulQ8_0MmqKernel mmq, Shape shape, int m, int k, int n,
        nint queryPool, float tsPeriodNs)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        VulkanApi.vkCmdResetQueryPool(ctx.CommandBuffer, queryPool, 0, (uint)(Reps + 1));
        VulkanApi.vkCmdWriteTimestamp(ctx.CommandBuffer, VkPipelineStageFlags.BottomOfPipe, queryPool, 0);
        for (int i = 0; i < Reps; i++)
        {
            mmq.Record(ctx.CommandBuffer, shape.W, shape.Xq, shape.Xds, shape.C, m, k, n);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            VulkanApi.vkCmdWriteTimestamp(ctx.CommandBuffer, VkPipelineStageFlags.BottomOfPipe, queryPool, (uint)(i + 1));
        }
        ctx.SubmitAndWait();

        Span<ulong> ts = stackalloc ulong[Reps + 1];
        fixed (ulong* p = ts)
        {
            int rc = VulkanApi.vkGetQueryPoolResults(
                device.Handle, queryPool, 0, (uint)(Reps + 1),
                (nuint)((Reps + 1) * sizeof(ulong)), (nint)p, sizeof(ulong), flags: 0x1 | 0x2);
            if (rc < 0)
                throw new Xunit.Sdk.XunitException($"vkGetQueryPoolResults failed: {rc}");
        }
        double toUs = tsPeriodNs / 1000.0;
        double totalUs = (ts[Reps] - ts[0]) * toUs;
        return totalUs / Reps;
    }

    private static nint CreateQueryPool(VulkanDevice device, int count)
    {
        var qci = new VkQueryPoolCreateInfo
        {
            sType = 11, // VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO
            queryType = 2, // VK_QUERY_TYPE_TIMESTAMP
            queryCount = (uint)count,
        };
        int rc = VulkanApi.vkCreateQueryPool(device.Handle, qci, 0, out nint pool);
        if (rc < 0 || pool == 0)
            throw new Xunit.Sdk.XunitException($"vkCreateQueryPool failed: {rc}");
        return pool;
    }

    /// <summary>Ordinary least squares: y = a + b*x, returns (a, b).</summary>
    private static (double A, double B) FitLinear(List<(double X, double Y)> points)
    {
        int n = points.Count;
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        foreach (var (x, y) in points)
        {
            sumX += x; sumY += y; sumXY += x * y; sumXX += x * x;
        }
        double b = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        double a = (sumY - b * sumX) / n;
        return (a, b);
    }
}
