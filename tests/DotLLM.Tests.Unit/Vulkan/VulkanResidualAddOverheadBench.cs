using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark: measures the standalone dispatch cost of the
/// residual-add mechanism (<see cref="AddKernel"/>, <c>add.spv</c>) that
/// <c>VulkanTransformerModel</c> issues twice per layer (once after
/// <c>o_proj</c>, once after <c>down_proj</c> — see
/// <c>VulkanTransformerModel.cs</c> around the "Residual add #1"/"Residual
/// add #2" comments, e.g. lines ~3089 and ~3227 of the per-token decode
/// path), each preceded and followed by a <c>BarrierComputeToCompute</c>.
/// </summary>
/// <remarks>
/// <para>
/// Investigates whether fusing the residual add into the preceding
/// <c>o_proj</c>/<c>down_proj</c> matmul (à la llama.cpp's Vulkan
/// MUL_MAT+ADD graph fusion, <c>.docs/KERNEL_MAP.md</c> §11) is worth the
/// shader-side risk/effort: it reports the add's absolute per-dispatch cost
/// AND its cost as a fraction of a representative matmul at the same shape,
/// for both a small (SmolLM, hidden=576) and mid (3B-ish, hidden=3072)
/// model, at seqLen=1 (decode, GEMV) and seqLen=512 (prefill, GEMM).
/// </para>
/// <para>
/// Methodology mirrors <see cref="VulkanI2SGemvBench"/>: each timed pass
/// submits <c>batch</c> back-to-back dispatches (no barriers between them)
/// behind one fence, so the ~ms submit/fence latency amortizes and the
/// reported number approaches the true GPU-side dispatch cost rather than a
/// launch-bound one. This intentionally does NOT reproduce the
/// barrier-separated real forward-pass timing (that would need the barrier
/// cost folded in too) — it isolates "how expensive is the add shader
/// itself", which is the number that bounds the best case for fusing it
/// away.
/// </para>
/// <para>
/// Enable with <c>DOTLLM_RESIDUAL_ADD_BENCH=1</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanResidualAddOverheadBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;

    private readonly ITestOutputHelper _output;
    public VulkanResidualAddOverheadBench(ITestOutputHelper output) => _output = output;

    // (Tag, hiddenSize, seqLen)
    private static readonly (string Tag, int Hidden, int SeqLen)[] AddShapes =
    [
        ("SmolLM  hidden=576  decode  (seqLen=1)",    576,   1),
        ("SmolLM  hidden=576  prefill (seqLen=512)",  576, 512),
        ("3B-ish  hidden=3072 decode  (seqLen=1)",   3072,   1),
        ("3B-ish  hidden=3072 prefill (seqLen=512)", 3072, 512),
    ];

    // Representative o_proj-shaped matmul for the same models: square
    // hidden x hidden weight, Q8_0 quantized (the production decode/prefill
    // quant path). Decode uses the MMVQ GEMV kernel, prefill the MMQ GEMM
    // kernel — matching RecordMatmul's own seqLen-based routing.
    private static readonly (string Tag, int Hidden, int SeqLen)[] MatmulShapes = AddShapes;

    [SkippableFact]
    public void Bench_ResidualAddOverhead()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_RESIDUAL_ADD_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_RESIDUAL_ADD_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int batch = EnvInt("DOTLLM_RESIDUAL_ADD_BENCH_BATCH", 64);

        using var device = VulkanDevice.Create();
        using var add = AddKernel.Create(device, spvDir);

        MatMulQ8_0MmvqKernel? mmvq = device.HasIntegerDotProduct
            ? MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            : null;
        bool haveMmq = File.Exists(Path.Combine(spvDir, "matmul_q8_0_mmq.spv"));

        _output.WriteLine($"Device: {device.DeviceName}  IntegerDotProduct: {device.HasIntegerDotProduct}");
        _output.WriteLine($"batch={batch}  schedule: {WarmupPasses} warmup + {Passes} passes (median)");
        _output.WriteLine("");
        _output.WriteLine("### Residual add (AddKernel) standalone dispatch cost");
        _output.WriteLine("| shape | add µs/dispatch |");
        _output.WriteLine("|---|---:|");

        var addResults = new Dictionary<(int, int), double> ();
        foreach (var (tag, hidden, seqLen) in AddShapes)
        {
            int n = hidden * seqLen;
            using var bufA = device.Allocate((long)n * sizeof(float));
            using var bufB = device.Allocate((long)n * sizeof(float));
            using var bufC = device.Allocate((long)n * sizeof(float));

            var rng = new Random(unchecked(hidden * 31 + seqLen));
            float[] a = RandomFloats(rng, n);
            float[] b = RandomFloats(rng, n);
            device.Upload(a, bufA);
            device.Upload(b, bufB);

            double us = MeasureAdd(device, add, bufA, bufB, bufC, n, batch);
            addResults[(hidden, seqLen)] = us;
            _output.WriteLine($"| {tag} | {us:F3} |");
        }

        _output.WriteLine("");
        _output.WriteLine("### Representative o_proj-shaped Q8_0 matmul (hidden x hidden) for comparison");
        _output.WriteLine("| shape | matmul µs/dispatch | add as % of matmul |");
        _output.WriteLine("|---|---:|---:|");

        foreach (var (tag, hidden, seqLen) in MatmulShapes)
        {
            double? matmulUs = null;

            long rowBytes = (long)(hidden / 32) * MatMulQ8_0MmvqKernel.Q8_0BlockBytes;
            long wBytes = (long)hidden * rowBytes;
            using var bufW = device.Allocate(wBytes);
            var rng = new Random(unchecked(hidden * 17 + seqLen + 1));
            byte[] w = new byte[wBytes];
            rng.NextBytes(w);
            device.Upload(new ReadOnlySpan<byte>(w), bufW);

            if (seqLen == 1 && mmvq is not null)
            {
                using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(hidden));
                using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(hidden));
                using var bufY = device.Allocate((long)hidden * sizeof(float));

                // Random packed int8 activations — timing is data-independent
                // for this dp4a shader (no data-dependent branches).
                byte[] xq = new byte[QuantizeQ8_1Kernel.PackedBytes(hidden)];
                rng.NextBytes(xq);
                device.Upload(new ReadOnlySpan<byte>(xq), bufXq);
                var xds = new float[hidden / 32 * 2];
                for (int i = 0; i < xds.Length; i += 2) { xds[i] = 1.0f; xds[i + 1] = 0.0f; }
                device.Upload(xds, bufXds);

                matmulUs = MeasureMmvq(device, mmvq, bufW, bufXq, bufXds, bufY, hidden, hidden, batch);
            }
            else if (seqLen > 1 && haveMmq && device.HasIntegerDotProduct
                && MatMulQ8_0MmqKernel.TryCreate(device, spvDir) is { } mmqOrNull)
            {
                using var mmq = mmqOrNull;
                int n = seqLen;
                using var bufXq = device.Allocate((long)n * QuantizeQ8_1Kernel.PackedBytes(hidden));
                using var bufXds = device.Allocate((long)n * QuantizeQ8_1Kernel.ScaleBytes(hidden));
                using var bufC = device.Allocate((long)n * hidden * sizeof(float));

                byte[] xq = new byte[n * QuantizeQ8_1Kernel.PackedBytes(hidden)];
                rng.NextBytes(xq);
                device.Upload(new ReadOnlySpan<byte>(xq), bufXq);
                var xds = new float[n * hidden / 32 * 2];
                for (int i = 0; i < xds.Length; i += 2) { xds[i] = 1.0f; xds[i + 1] = 0.0f; }
                device.Upload(xds, bufXds);

                matmulUs = MeasureMmq(device, mmq, bufW, bufXq, bufXds, bufC, hidden, hidden, n, batch);
            }

            if (matmulUs is { } mu)
            {
                double addUs = addResults[(hidden, seqLen)];
                _output.WriteLine($"| {tag} | {mu:F3} | {(addUs / mu * 100.0):F2}% |");
            }
            else
            {
                _output.WriteLine($"| {tag} | (kernel unavailable on this device) | — |");
            }
        }

        mmvq?.Dispose();
    }

    private static double MeasureAdd(
        VulkanDevice device, AddKernel add,
        VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++) RunAddPass(device, add, a, b, c, n, batch);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = RunAddPass(device, add, a, b, c, n, batch);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static double RunAddPass(
        VulkanDevice device, AddKernel add,
        VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c, int n, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            add.Record(ctx.CommandBuffer, a, b, c, n);   // no barriers -> dispatches overlap (compute-bound)
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static double MeasureMmvq(
        VulkanDevice device, MatMulQ8_0MmvqKernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer y,
        int m, int k, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++) RunMmvqPass(device, kernel, w, xq, xds, y, m, k, batch);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = RunMmvqPass(device, kernel, w, xq, xds, y, m, k, batch);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static double RunMmvqPass(
        VulkanDevice device, MatMulQ8_0MmvqKernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer y,
        int m, int k, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            kernel.Record(ctx.CommandBuffer, w, xq, xds, y, m, k);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static double MeasureMmq(
        VulkanDevice device, MatMulQ8_0MmqKernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++) RunMmqPass(device, kernel, w, xq, xds, c, m, k, n, batch);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = RunMmqPass(device, kernel, w, xq, xds, c, m, k, n, batch);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static double RunMmqPass(
        VulkanDevice device, MatMulQ8_0MmqKernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            kernel.Record(ctx.CommandBuffer, w, xq, xds, c, m, k, n);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static float[] RandomFloats(Random rng, int n)
    {
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = rng.NextSingle() * 2f - 1f;
        return arr;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;
}
