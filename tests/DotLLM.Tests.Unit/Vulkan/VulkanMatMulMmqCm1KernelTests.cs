using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity of the int8-coopmat MMQ GEMM (#1226) against a double-precision reference over the SAME Q8_1 activations the kernel reads
/// (downloaded from the device) and the dequantized weights. Because the activation rounding is shared, the only legitimate difference
/// is F32 accumulation order, so the tolerance is tight: a wrong accumulator row mapping, scale, min term or tile bound moves outputs
/// by O(1) of the row magnitude. Shapes cover partial M and N tiles, several K steps and an L2 group boundary.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMatMulMmqCm1KernelTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(QuantizationType.Q8_0, 77, 200, 352)]
    [InlineData(QuantizationType.Q8_0, 128, 64, 128)]
    [InlineData(QuantizationType.Q8_0, 5, 2112, 1024)]
    [InlineData(QuantizationType.Q4_K, 77, 200, 512)]
    [InlineData(QuantizationType.Q4_K, 130, 320, 1280)]
    [InlineData(QuantizationType.Q4_K, 3, 2112, 256)]
    [InlineData(QuantizationType.Q5_0, 77, 200, 384)]
    public void MatchesReferenceOnSameQ8_1Activations(QuantizationType qt, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = MatMulMmqCm1Kernel.TryCreate(device, spvDir, qt);
        Skip.If(kernel is null, "int8-coopmat MMQ needs AMD + Sint8 16x16x16 coopmat + wave32 pinning.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        var rng = new Random(1226 + n * 7 + m * 3 + k);
        float[] w = new float[m * k];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        byte[] wq = qt == QuantizationType.Q4_K ? Q4KFixture.QuantizeRows(w, m, k) : QuantizeRows(w, m, k, qt);
        float[] wd = new float[m * k];
        long rowBytes = Dequantize.RowByteSize(k, qt);
        unsafe
        {
            fixed (byte* p = wq)
                for (int r = 0; r < m; r++)
                    Dequantize.ToFloat32((nint)(p + r * rowBytes), k, qt, wd.AsSpan(r * k, k));
        }

        using var bufW = device.Allocate((wq.Length + 3) & ~3);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.Upload(wq, bufW);
        device.Upload(b, bufB);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel!.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n);
            ctx.SubmitAndWait();
        }

        float[] xqF = new float[n * k / 4];
        device.Download(bufXq, xqF);
        sbyte[] xq = MemoryMarshal.Cast<float, sbyte>(xqF).ToArray();
        float[] xds = new float[n * (k / 32) * 2];
        device.Download(bufXds, xds);
        float[] c = new float[n * m];
        device.Download(bufC, c);

        double worst = 0, scale = 0;
        for (int t = 0; t < n; t++)
        {
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int i = 0; i < k; i++)
                    acc += wd[r * k + i] * (xq[t * k + i] * (double)xds[(t * (k / 32) + i / 32) * 2]);
                worst = Math.Max(worst, Math.Abs(acc - c[t * m + r]));
                scale = Math.Max(scale, Math.Abs(acc));
            }
        }
        output.WriteLine($"{qt} n={n} m={m} k={k}: max|err|={worst:E3} max|ref|={scale:E3}");
        Assert.True(worst <= 2e-4 * scale + 1e-5, $"max |ref - mmq_cm1| = {worst:E3} against max |ref| {scale:E3}");
    }

    /// <summary>
    /// The expert-grouped form over a packed bank: experts with 0, 1, few and more-than-a-tile (70 &gt; 64) rows, partial M tiles, and the
    /// same exact-reference tolerance as the dense test (shared Q8_1 activations).
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q5_0, 200, 416)]   // 13 blocks: a partial last K step (Lightning down K = 1856)
    [InlineData(QuantizationType.Q8_0, 136, 256)]
    [InlineData(QuantizationType.Q4_K, 72, 512)]
    public void Grouped_MatchesReferenceOnSameQ8_1Activations(QuantizationType qt, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = MatMulMmqCm1Kernel.TryCreate(device, spvDir, qt, grouped: true);
        Skip.If(kernel is null, "int8-coopmat MMQ needs AMD + Sint8 16x16x16 coopmat + wave32 pinning.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        int[] counts = [0, 7, 70, 1, 3];
        int e = counts.Length, rows = counts.Sum();
        uint[] offsets = new uint[e + 1];
        for (int i = 0; i < e; i++) offsets[i + 1] = offsets[i] + (uint)counts[i];
        var rng = new Random(4242 + m + k);
        float[] w = new float[e * m * k];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        float[] b = new float[rows * k];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        byte[] wq = qt == QuantizationType.Q4_K ? Q4KFixture.QuantizeRows(w, e * m, k) : QuantizeRows(w, e * m, k, qt);
        float[] wd = new float[e * m * k];
        long rowBytes = Dequantize.RowByteSize(k, qt);
        unsafe
        {
            fixed (byte* p = wq)
                for (int r = 0; r < e * m; r++)
                    Dequantize.ToFloat32((nint)(p + r * rowBytes), k, qt, wd.AsSpan(r * k, k));
        }

        using var bufW = device.Allocate((wq.Length + 3) & ~3);
        using var bufB = device.Allocate((long)rows * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(rows, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(rows, k));
        using var bufOff = device.Allocate((e + 1) * sizeof(uint));
        using var bufC = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(wq, bufW);
        device.Upload(b, bufB);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, rows, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel!.RecordGrouped(ctx.CommandBuffer, bufW, bufXq, bufXds, bufOff, bufC, m, k, rows, e, maxRowsPerExpert: counts.Max());
            ctx.SubmitAndWait();
        }

        float[] xqF = new float[rows * k / 4];
        device.Download(bufXq, xqF);
        sbyte[] xq = MemoryMarshal.Cast<float, sbyte>(xqF).ToArray();
        float[] xds = new float[rows * (k / 32) * 2];
        device.Download(bufXds, xds);
        float[] c = new float[rows * m];
        device.Download(bufC, c);

        double worst = 0, scale = 0;
        for (int ex = 0; ex < e; ex++)
        for (int t = (int)offsets[ex]; t < (int)offsets[ex + 1]; t++)
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int i = 0; i < k; i++)
                acc += wd[((long)ex * m + r) * k + i] * (xq[t * k + i] * (double)xds[(t * (k / 32) + i / 32) * 2]);
            worst = Math.Max(worst, Math.Abs(acc - c[t * m + r]));
            scale = Math.Max(scale, Math.Abs(acc));
        }
        output.WriteLine($"grouped {qt} m={m} k={k} rows={rows}: max|err|={worst:E3} max|ref|={scale:E3}");
        Assert.True(worst <= 2e-4 * scale + 1e-5, $"max |ref - grouped mmq_cm1| = {worst:E3} against max |ref| {scale:E3}");
    }

    /// <summary>Opt-in (<c>DOTLLM_GEMM_SHAPE_PROBE=1</c>): steady-state TFLOP/s of the int8-coopmat GEMM (+ its Q8_1 quantize) vs the F16 coopmat GEMM at the Qwen3.8-27B prefill shapes.</summary>
    [SkippableFact]
    public void Probe_Qwen38Shapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMM_SHAPE_PROBE") == "1", "opt-in probe");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        (QuantizationType Qt, int M, int K)[] shapes =
        [
            (QuantizationType.Q4_K, 17408, 5120), (QuantizationType.Q4_K, 5120, 17408),
            (QuantizationType.Q8_0, 10240, 5120), (QuantizationType.Q8_0, 6144, 5120), (QuantizationType.Q8_0, 5120, 6144),
        ];
        const int N = 512, Reps = 10;
        foreach (var (qt, m, k) in shapes)
        {
            using var cm1 = MatMulMmqCm1Kernel.TryCreate(device, spvDir, qt);
            Skip.If(cm1 is null, "int8-coopmat MMQ unsupported here.");
            using var f16q4 = qt == QuantizationType.Q4_K ? MatMulQ4KGemmCoopmatKernel.Create(device, spvDir) : null;
            using var f16q8 = qt == QuantizationType.Q8_0 ? MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir) : null;
            long wBytes = (Dequantize.RowByteSize(k, qt) * m + 3) & ~3L;
            using var w = device.AllocateDeviceLocal(wBytes);
            using var b = device.AllocateDeviceLocal((long)N * k * sizeof(float));
            using var xq = device.AllocateDeviceLocal(QuantizeQ8_1RowsKernel.PackedBytes(N, k));
            using var xds = device.AllocateDeviceLocal(QuantizeQ8_1RowsKernel.ScaleBytes(N, k));
            using var c = device.AllocateDeviceLocal((long)N * m * sizeof(float));
            double Time(Action<nint> rec)
            {
                double best = double.MaxValue;
                for (int trial = 0; trial < 4; trial++)
                {
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    for (int r = 0; r < Reps; r++) { rec(ctx.CommandBuffer); KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer); }
                    var sw = Stopwatch.StartNew();
                    ctx.SubmitAndWait();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / Reps);
                }
                return best;
            }
            double tMmq = Time(cb => cm1!.Record(cb, w, xq, xds, c, m, k, N));
            double tQ = Time(cb => quant.Record(cb, b, xq, xds, N, k));
            double tF16 = Time(cb =>
            {
                if (f16q4 is not null) f16q4.Record(cb, w, b, c, m, k, N);
                else f16q8!.Record(cb, w, b, c, m, k, N);
            });
            double fl = 2.0 * m * k * N / 1e9;
            output.WriteLine($"{qt} m={m,5} k={k,5}: mmq_cm1 {tMmq,6:F2} ms ({fl / tMmq,5:F1} TFLOP/s) + quant {tQ,5:F2} ms | f16 coopmat {tF16,6:F2} ms ({fl / tF16,5:F1} TFLOP/s)");
        }
    }

    private static byte[] QuantizeRows(float[] values, int rows, int cols, QuantizationType qt)
    {
        long rowBytes = Dequantize.RowByteSize(cols, qt);
        byte[] bytes = new byte[rowBytes * rows];
        for (int r = 0; r < rows; r++)
            Quantize.FromFloat32(values.AsSpan(r * cols, cols), cols, qt, bytes.AsSpan((int)(r * rowBytes), (int)rowBytes));
        return bytes;
    }
}
