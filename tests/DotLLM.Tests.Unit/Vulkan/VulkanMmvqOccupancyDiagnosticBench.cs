using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in diagnostic micro-benchmark for the Q4_K decode-path MMVQ GEMV
/// (<see cref="MatMulQ4KMmvqKernel"/>) at realistic dense 3B/8B decode shapes.
/// </summary>
/// <remarks>
/// <para>
/// Purpose: distinguish "occupancy-starved" from "bandwidth-bound" for the current
/// dispatch geometry — <b>one workgroup per output row, one column per dispatch</b>
/// (<see cref="MatMulQ4KMmvqKernel.Record"/>: <c>vkCmdDispatch(cmdBuf, m, 1, 1)</c>,
/// 32-thread wave32 workgroups). llama.cpp's Vulkan <c>mul_mat_vec*.comp</c> instead
/// packs 1–4 rows per workgroup (quant-dependent) and batches up to
/// <c>mul_mat_vec_max_cols=8</c> columns per dispatch (<c>ggml-vulkan.cpp:4293-4308</c>,
/// <c>:271</c>) — see <c>.docs/KERNEL_MAP.md</c> §3 items 6/6b for the prior static
/// analysis this bench is meant to validate empirically.
/// </para>
/// <para>
/// Method: times <c>batch</c> back-to-back single-column GEMV dispatches behind one
/// fence (amortises submit/fence latency so the reported number approaches the
/// steady-state per-dispatch cost, not a launch-bound one) and reports effective
/// GB/s = weight bytes read / wall time. Compare against the box's known decode-GEMV
/// steady state (~48-60 GB/s, ~60% of achievable peak after the #338/#339 coalescing
/// campaign — <c>.docs/KERNEL_MAP.md</c> §3 item 18 / §6 "coalescing campaigns" notes)
/// and the UMA's nominal peak bandwidth. If measured GB/s here already sits near that
/// ~48-60 GB/s steady state at every M, the kernel is memory-bandwidth-bound and a
/// rows-per-WG / multi-column batching change (which raises occupancy, not bytes
/// moved) is unlikely to help much further; if GB/s instead scales up materially with
/// M (more rows -> more workgroups -> better SM occupancy) at the smaller M, the
/// kernel is occupancy-starved at low M and rows-per-WG batching is the higher-value
/// lever there.
/// </para>
/// <para>
/// Enable with <c>DOTLLM_MMVQ_OCCUPANCY_BENCH=1</c>. Not run as part of CI — GPU-gated
/// and opt-in, matching the project's other Vulkan micro-benches
/// (<see cref="VulkanI2SGemvBench"/>, <see cref="VulkanMmvqSharedQuantBench"/>).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMmvqOccupancyDiagnosticBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;
    private const int BatchSize = 32; // dispatches per submit, amortises fence latency

    private readonly ITestOutputHelper _output;
    public VulkanMmvqOccupancyDiagnosticBench(ITestOutputHelper output) => _output = output;

    // Dense 3B/8B decode-path output-row counts (M) at a fixed hidden/intermediate-class
    // K=4096: o_proj/QKV-class (M~4096), a wider gate/up-class shape (M~8192), and an
    // 8B-intermediate/lm_head-class shape (M~14336). K held constant so any GB/s delta
    // across rows is attributable to workgroup COUNT (occupancy), not row length.
    private static readonly (string Tag, int M, int K)[] Shapes =
    [
        ("M=4096  (o_proj/QKV-class)",  4096,  4096),
        ("M=8192  (gate/up-class)",     8192,  4096),
        ("M=14336 (8B-intermediate)",  14336,  4096),
    ];

    [SkippableFact]
    public void Bench_Q4KMmvqOccupancy()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_MMVQ_OCCUPANCY_BENCH") == "1",
            "DOTLLM_MMVQ_OCCUPANCY_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int batch = EnvInt("DOTLLM_MMVQ_OCCUPANCY_BATCH", BatchSize);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device lacks integer-dot-product; MMVQ path unavailable.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir);
        using var gemv = MatMulQ4KMmvqKernel.TryCreate(device, spvDir);
        Skip.If(quant is null || gemv is null, "Q4_K MMVQ / quantize SPVs missing.");

        _output.WriteLine("Device: " + device.DeviceName);
        _output.WriteLine($"IntegerDotProduct: {device.HasIntegerDotProduct}, SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine($"Dispatch geometry under test: 1 workgroup/row (32 threads, wave32), 1 column/dispatch"
            + $" (MatMulQ4KMmvqKernel.Record -> vkCmdDispatch(m, 1, 1)).");
        _output.WriteLine($"Schedule: {WarmupPasses} warmup + {Passes} passes, batch={batch} dispatches/submit (median).");
        _output.WriteLine(string.Empty);
        _output.WriteLine("| shape | rows (M) | workgroups | µs/GEMV | weight GB/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|");

        var rng = new Random(0x4B4D); // "KM" — Q4_K MMVQ
        foreach (var (tag, m, k) in Shapes)
        {
            int blocksPerRow = k / MatMulQ4KMmvqKernel.Q4KGroupSize;
            long rowBytes = (long)blocksPerRow * MatMulQ4KMmvqKernel.Q4KBlockBytes;
            long weightBytes = (long)m * rowBytes;

            using var bufW = device.Allocate(weightBytes);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
            using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
            using var bufY = device.Allocate((long)m * sizeof(float));

            UploadRandomBytes(device, bufW, weightBytes, rng);
            UploadRandomFloats(device, bufX, k, rng);

            // Prime the activation quant buffers once outside the timed loop — the bench
            // isolates the GEMV kernel itself, not the (separately-benched, cheap) quantize
            // dispatch. Production always precedes each MMVQ with a quantize + barrier; here
            // we quantize once and re-read the same Q8_1 buffers across the batch, matching
            // how VulkanMmvqSharedQuantBench isolates the GEMV-only cost path.
            void RecordQuantizeOnce(nint cmd) => quant.Record(cmd, bufX, bufXq, bufXds, k);
            using (var primeCtx = device.CreateSubmitContext())
            {
                primeCtx.Begin();
                RecordQuantizeOnce(primeCtx.CommandBuffer);
                primeCtx.SubmitAndWait();
            }

            void RecordGemv(nint cmd) => gemv!.Record(cmd, bufW, bufXq, bufXds, bufY, m, k);

            double us = MeasureMedian(device, RecordGemv, batch);
            double gbps = weightBytes / (us * 1e-6) / 1e9;

            _output.WriteLine($"| {tag} | {m} | {m} | {us:F2} | {gbps:F1} |");
        }

        _output.WriteLine(string.Empty);
        _output.WriteLine("Reference: decode-GEMV steady state after #338/#339 coalescing is ~48-60 GB/s"
            + " (~60% of achievable peak) per .docs/KERNEL_MAP.md §3. If the GB/s above is flat and near"
            + " that band across all three M, the kernel is memory-bandwidth-bound at this shape and"
            + " rows-per-WG/multi-column batching (raises occupancy, not bytes moved) has limited headroom;"
            + " if GB/s at M=4096 is materially lower than at M=14336, the smaller shape is occupancy-starved"
            + " (workgroup count << compute-unit count) and rows-per-WG batching is the higher-value lever there.");
    }

    private static double MeasureMedian(VulkanDevice device, Action<nint> record, int batch)
    {
        for (int w = 0; w < WarmupPasses; w++) RunPass(device, record, batch);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = RunPass(device, record, batch);
        Array.Sort(us);
        return us[Passes / 2];
    }

    // One pass = `batch` GEMV dispatches behind a single fence, no inter-dispatch barriers
    // (dispatches may overlap on the GPU — this measures steady-state throughput, matching
    // VulkanI2SGemvBench's methodology). Returns µs/dispatch.
    private static double RunPass(VulkanDevice device, Action<nint> record, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            record(ctx.CommandBuffer);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static void UploadRandomBytes(VulkanDevice device, VulkanDevice.Buffer buf, long bytes, Random rng)
    {
        var data = new byte[bytes];
        rng.NextBytes(data);
        device.Upload(data, buf);
    }

    private static void UploadRandomFloats(VulkanDevice device, VulkanDevice.Buffer buf, int count, Random rng)
    {
        var x = new float[count];
        for (int i = 0; i < count; i++) x[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        device.Upload(x, buf);
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;
}
