using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Tests.Unit.Models.Flint;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The Q4_1 prefill GEMMs on Q8_1 activations (#1235): the int8-coopmat <see cref="MatMulMmqCm1Kernel"/> (AMD) and the dp4a
/// <see cref="MatMulQ4_1MmqKernel"/>. Two oracles: a double-precision reference over the SAME Q8_1 activations the kernel read (only
/// accumulation order may differ), and the CPU's packed Q4_1 x Q8_1 GEMM (#1159, llama.cpp-golden to &lt; 1e-4) on the same F32 input,
/// which pins the activation quantisation contract too. Weights and activations both carry an offset, so the min term
/// <c>m * s_x</c> is a large share of every output and a kernel that dropped or mis-scaled it fails by O(1) (asserted as a control).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMatMulQ4_1MmqKernelTests(ITestOutputHelper output)
{
    private interface IGemm : IDisposable
    {
        void Record(nint cmdBuf, VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer c, int m, int k, int n);
    }

    private sealed class Cm1(MatMulMmqCm1Kernel kernel) : IGemm
    {
        public void Record(nint cmdBuf, VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer c, int m, int k, int n)
            => kernel.Record(cmdBuf, w, xq, xds, c, m, k, n);

        public void Dispose() => kernel.Dispose();
    }

    private sealed class Dp4a(MatMulQ4_1MmqKernel kernel) : IGemm
    {
        public void Record(nint cmdBuf, VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer c, int m, int k, int n)
            => kernel.Record(cmdBuf, w, xq, xds, c, m, k, n);

        public void Dispose() => kernel.Dispose();
    }

    private static IGemm? Create(VulkanDevice device, string spvDir, string kind) => kind switch
    {
        "cm1" => MatMulMmqCm1Kernel.TryCreate(device, spvDir, QuantizationType.Q4_1) is { } k ? new Cm1(k) : null,
        _ => MatMulQ4_1MmqKernel.TryCreate(device, spvDir) is { } k ? new Dp4a(k) : null,
    };

    [SkippableTheory]
    [InlineData("cm1", 77, 200, 352)]     // partial M / N tiles, 11 blocks (a partial last K step)
    [InlineData("cm1", 5, 2112, 1024)]    // L2 group boundary
    [InlineData("cm1", 130, 512, 2048)]   // Flint k/v projection shape
    [InlineData("dp4a", 77, 200, 352)]
    [InlineData("dp4a", 5, 2112, 1024)]
    [InlineData("dp4a", 130, 512, 2048)]  // several FLUSH_BLOCKS windows
    public void MatchesReferenceAndCpuQ8_1Gemm(string kind, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = Create(device, spvDir, kind);
        Skip.If(kernel is null, $"Q4_1 {kind} MMQ unavailable on this device.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        var rng = new Random(1235 + n * 7 + m * 3 + k);
        float[] w = new float[m * k];
        for (int i = 0; i < w.Length; i++) w[i] = 0.05f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;   // offset rows: m != 0
        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = 0.3f + (float)(rng.NextDouble() * 2 - 1);            // offset activations: s_x != 0
        byte[] wq = SyntheticFlintGguf.EncodeQ4_1(w);
        long rowBytes = Dequantize.RowByteSize(k, QuantizationType.Q4_1);
        float[] cpu = new float[n * m];
        unsafe
        {
            fixed (byte* p = wq)
            fixed (float* pb = b)
            fixed (float* pc = cpu)
            {
                MatMul.GemmLegacyQuant(p, QuantizationType.Q4_1, pb, pc, m, k, n);
            }
        }

        using var bufW = device.Allocate(wq.Length);
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

        double worst = 0, worstCpu = 0, worstNoMin = 0, scale = 0;
        for (int t = 0; t < n; t++)
        {
            for (int r = 0; r < m; r++)
            {
                double acc = 0, minTerm = 0;
                for (int blk = 0; blk < k / 32; blk++)
                {
                    // d * d_x * dot(q, x) + m * s_x with (d_x, s_x) at fp16, as block_q8_1 stores them (CPU #1159, llama.cpp).
                    double dx = (double)(Half)xds[(t * (k / 32) + blk) * 2];
                    double sx = (double)(Half)xds[(t * (k / 32) + blk) * 2 + 1];
                    int o = (int)(r * rowBytes + blk * 20);
                    int dot = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        int x0 = t * k + blk * 32 + j;
                        dot += (wq[o + 4 + j] & 0xF) * xq[x0] + (wq[o + 4 + j] >> 4) * xq[x0 + 16];
                    }
                    double mTerm = (double)BitConverter.ToHalf(wq, o + 2) * sx;
                    acc += (double)BitConverter.ToHalf(wq, o) * dx * dot + mTerm;
                    minTerm += mTerm;
                }
                float got = c[t * m + r];
                worst = Math.Max(worst, Math.Abs(acc - got));
                worstCpu = Math.Max(worstCpu, Math.Abs(cpu[t * m + r] - got));
                worstNoMin = Math.Max(worstNoMin, Math.Abs(acc - minTerm - got));
                scale = Math.Max(scale, Math.Abs(acc));
            }
        }
        output.WriteLine($"q4_1 {kind} n={n} m={m} k={k}: max|ref-gpu|={worst:E3} max|cpu-gpu|={worstCpu:E3} (rel {worstCpu / scale:E2}) "
            + $"max|ref|={scale:E3} no-min control={worstNoMin:E3}");
        Assert.True(worst <= 2e-5 * scale + 1e-5, $"max |ref - gpu| = {worst:E3} against max |ref| {scale:E3}");
        Assert.True(worstCpu <= 1e-5 * scale, $"max |cpu - gpu| = {worstCpu:E3} against max |ref| {scale:E3}");
        Assert.True(worstNoMin > 0.05 * scale, $"the min term is too small to be checked here ({worstNoMin:E3})");
    }

    /// <summary>
    /// Opt-in (<c>DOTLLM_GEMM_SHAPE_PROBE=1</c>): steady-state time of the Q4_1 MMQs (+ the Q8_1 quantize) against the F16 coopmat GEMM at
    /// Recursive Flint's dense prefill shapes (<c>DOTLLM_GEMM_SHAPE_PROBE_N</c> rows, default 512).
    /// </summary>
    [SkippableFact]
    public void Probe_FlintShapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMM_SHAPE_PROBE") == "1", "opt-in probe");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        using var cm1 = Create(device, spvDir, "cm1");
        using var dp4a = Create(device, spvDir, "dp4a");
        using var f16 = MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spvDir, QuantizationType.Q4_1)
            ? MatMulQ4_0GemmCoopmatKernel.Create(device, spvDir, QuantizationType.Q4_1) : null;
        (int M, int K)[] shapes = [(4096, 2048), (512, 2048), (2048, 4096), (6144, 2048), (2048, 6144), (1024, 2048), (2048, 1024)];
        int nRows = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_GEMM_SHAPE_PROBE_N"), out int nn) ? nn : 512;
        const int Reps = 10;
        foreach (var (m, k) in shapes)
        {
            using var w = device.AllocateDeviceLocal(Dequantize.RowByteSize(k, QuantizationType.Q4_1) * m);
            using var b = device.AllocateDeviceLocal((long)nRows * k * sizeof(float));
            using var xq = device.AllocateDeviceLocal(QuantizeQ8_1RowsKernel.PackedBytes(nRows, k));
            using var xds = device.AllocateDeviceLocal(QuantizeQ8_1RowsKernel.ScaleBytes(nRows, k));
            using var c = device.AllocateDeviceLocal((long)nRows * m * sizeof(float));
            double Time(Action<nint> rec)
            {
                double best = double.MaxValue;
                for (int trial = 0; trial < 4; trial++)
                {
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    for (int r = 0; r < Reps; r++)
                    {
                        rec(ctx.CommandBuffer);
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    }
                    var sw = Stopwatch.StartNew();
                    ctx.SubmitAndWait();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / Reps);
                }
                return best;
            }
            string T(IGemm? g) => g is null ? "   n/a" : $"{Time(cb => g.Record(cb, w, xq, xds, c, m, k, nRows)),6:F3}";
            double tQ = Time(cb => quant.Record(cb, b, xq, xds, nRows, k));
            string tF16 = f16 is null ? "   n/a" : $"{Time(cb => f16.Record(cb, w, b, c, m, k, nRows)),6:F3}";
            output.WriteLine($"q4_1 n={nRows} m={m,5} k={k,5}: cm1 {T(cm1)} ms | dp4a {T(dp4a)} ms | + quant {tQ,6:F3} ms | f16 coopmat {tF16} ms");
        }
    }
}
