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
/// The Q4_0 prefill GEMMs on Q8_1 activations (#1447): the int8-coopmat <see cref="MatMulMmqCm1Kernel"/> (AMD) and the dp4a
/// <see cref="MatMulQ4_0MmqKernel"/>. Two oracles: a double-precision reference over the SAME Q8_1 activations the kernel read (only
/// accumulation order may differ), and the CPU's packed Q4_0 x Q8_1 GEMM (#1159, llama.cpp-golden to &lt; 1e-4) on the same Q8_1
/// blocks, which pins the GEMM contract, fp16 (d, s) included; the device's codes are also counted against the CPU quantizer's. Weights and activations both carry an offset, so the min term
/// <c>m * s_x</c> is a large share of every output and a kernel that dropped or mis-scaled it fails by O(1) (asserted as a control).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMatMulQ4_0MmqKernelTests(ITestOutputHelper output)
{
    /// <summary>ggml block_q4_0: f16 d, 16 bytes of nibbles (element j low, j + 16 high); x = (q - 8) * d.</summary>
    private static byte[] EncodeQ4_0(float[] data)
    {
        var bytes = new byte[data.Length / 32 * 18];
        for (int b = 0; b < data.Length / 32; b++)
        {
            var x = data.AsSpan(b * 32, 32);
            float amax = 0, max = 0;
            foreach (float v in x) if (MathF.Abs(v) > amax) { amax = MathF.Abs(v); max = v; }
            float d = max / -8f, id = d != 0 ? 1f / d : 0f;
            BitConverter.TryWriteBytes(bytes.AsSpan(b * 18, 2), (Half)d);
            for (int j = 0; j < 16; j++)
            {
                int lo = Math.Min(15, (int)(x[j] * id + 8.5f)), hi = Math.Min(15, (int)(x[j + 16] * id + 8.5f));
                bytes[b * 18 + 2 + j] = (byte)(lo | (hi << 4));
            }
        }
        return bytes;
    }

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

    private sealed class Dp4a(MatMulQ4_0MmqKernel kernel) : IGemm
    {
        public void Record(nint cmdBuf, VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer c, int m, int k, int n)
            => kernel.Record(cmdBuf, w, xq, xds, c, m, k, n);

        public void Dispose() => kernel.Dispose();
    }

    private static IGemm? Create(VulkanDevice device, string spvDir, string kind) => kind switch
    {
        "cm1" => MatMulMmqCm1Kernel.TryCreate(device, spvDir, QuantizationType.Q4_0) is { } k ? new Cm1(k) : null,
        _ => MatMulQ4_0MmqKernel.TryCreate(device, spvDir) is { } k ? new Dp4a(k) : null,
    };

    [SkippableTheory]
    [InlineData("cm1", 77, 200, 352)]     // partial M / N tiles, 11 blocks (a partial last K step)
    [InlineData("cm1", 5, 2112, 1024)]    // L2 group boundary
    [InlineData("cm1", 130, 512, 2048)]   // k/v-like shape; > one 128-row token tile
    [InlineData("cm1", 300, 72, 160)]     // three token tiles, the last partial; one partial weight tile
    [InlineData("dp4a", 77, 200, 352)]
    [InlineData("dp4a", 5, 2112, 1024)]
    [InlineData("dp4a", 130, 512, 2048)]  // several FLUSH_BLOCKS windows
    public void MatchesReferenceAndCpuQ8_1Gemm(string kind, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = Create(device, spvDir, kind);
        Skip.If(kernel is null, $"Q4_0 {kind} MMQ unavailable on this device.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        var rng = new Random(1235 + n * 7 + m * 3 + k);
        float[] w = new float[m * k];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = 0.3f + (float)(rng.NextDouble() * 2 - 1);
        byte[] wq = EncodeQ4_0(w);
        long rowBytes = Dequantize.RowByteSize(k, QuantizationType.Q4_0);
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

        // CPU oracle: the CPU packed Q4_0 GEMM run on the DEVICE's Q8_1 activations (block_q8_1 bytes: fp16 d, fp16 s, qs[32]).
        // Fed its own quantisation instead, a few codes differ by one where x / d sits within an ulp of .5 (AMD computes 1 / d with
        // a 1-ulp reciprocal): that is the quantizer's business, counted below, not the GEMM's.
        int blocks = n * (k / 32), qMismatch = 0;
        byte[] devQ8 = new byte[blocks * 36];
        byte[] cpuQ8 = new byte[blocks * 36];
        float[] cpu = new float[n * m];
        for (int bi = 0; bi < blocks; bi++)
        {
            BitConverter.TryWriteBytes(devQ8.AsSpan(bi * 36, 2), (Half)xds[bi * 2]);
            BitConverter.TryWriteBytes(devQ8.AsSpan(bi * 36 + 2, 2), (Half)xds[bi * 2 + 1]);
            for (int j = 0; j < 32; j++) devQ8[bi * 36 + 4 + j] = (byte)xq[bi * 32 + j];
        }
        unsafe
        {
            fixed (byte* p = wq)
            fixed (byte* pq = devQ8)
            fixed (byte* pcq = cpuQ8)
            fixed (float* pb = b)
            fixed (float* pc = cpu)
            {
                MatMul.QuantizeF32ToQ8_1(pb, pcq, n * k);
                MatMul.GemmLegacyQuant(p, QuantizationType.Q4_0, pb, pc, m, k, n, preQuantizedInput: pq);
            }
        }
        for (int bi = 0; bi < blocks; bi++)
            for (int j = 0; j < 32; j++)
                if (cpuQ8[bi * 36 + 4 + j] != devQ8[bi * 36 + 4 + j]) qMismatch++;

        double worst = 0, worstCpu = 0, worstCpuLlama = 0, worstNoMin = 0, scale = 0;
        for (int t = 0; t < n; t++)
        {
            for (int r = 0; r < m; r++)
            {
                double acc = 0, minTerm = 0, llama = 0;
                for (int blk = 0; blk < k / 32; blk++)
                {
                    // d * d_x * dot(q, x) + m * s_x with d_x at fp16, as block_q8_1 stores them (CPU #1159, llama.cpp).
                    double dx = (double)(Half)xds[(t * (k / 32) + blk) * 2];
                    double sx = (double)(Half)xds[(t * (k / 32) + blk) * 2 + 1];
                                        int o = (int)(r * rowBytes + blk * 18);
                    int dot = 0, rawDot = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        int x0 = t * k + blk * 32 + j;
                        { dot += ((wq[o + 2 + j] & 0xF) - 8) * xq[x0] + ((wq[o + 2 + j] >> 4) - 8) * xq[x0 + 16]; rawDot += (wq[o + 2 + j] & 0xF) * xq[x0] + (wq[o + 2 + j] >> 4) * xq[x0 + 16]; }
                    }
                    double mTerm = (double)BitConverter.ToHalf(wq, o) * dx * (rawDot - dot);   // the dropped recentring
                    acc += (double)BitConverter.ToHalf(wq, o) * dx * dot;
                    minTerm += mTerm;
                    // llama.cpp / CPU contract: d * (d_x * sum(q_raw * x) - 8 * s_x), s_x rounded to fp16.
                    llama += (double)BitConverter.ToHalf(wq, o) * (dx * rawDot - 8.0 * sx);
                }
                float got = c[t * m + r];
                worst = Math.Max(worst, Math.Abs(acc - got));
                worstCpu = Math.Max(worstCpu, Math.Abs(cpu[t * m + r] - got));
                worstCpuLlama = Math.Max(worstCpuLlama, Math.Abs(cpu[t * m + r] - llama));
                worstNoMin = Math.Max(worstNoMin, Math.Abs(acc + minTerm - got));
                scale = Math.Max(scale, Math.Abs(acc));
            }
        }
        output.WriteLine($"q4_0 {kind} n={n} m={m} k={k}: max|ref-gpu|={worst:E3} max|cpu-gpu|={worstCpu:E3} (rel {worstCpu / scale:E2}) max|cpu-llama contract|={worstCpuLlama:E3} "
            + $"max|ref|={scale:E3} no-recentre control={worstNoMin:E3} q8 codes != cpu quantizer: {qMismatch}/{n * k}");
        Assert.True(worst <= 2e-5 * scale + 1e-5, $"max |ref - gpu| = {worst:E3} against max |ref| {scale:E3}");
        // The CPU (and llama.cpp) fold the -8 recentring into the fp16-rounded s_x term; the kernels recentre the codes exactly, so
        // they differ by 8 * d * (s_x rounding), a few 1e-4 of the output where the activation block sums are large (offset +0.3 here).
        // The CPU is pinned to its own contract tightly, the GPU to the exact one.
        Assert.True(worstCpuLlama <= 1e-5 * scale, $"max |cpu - llama contract| = {worstCpuLlama:E3} against max |ref| {scale:E3}");
        Assert.True(worstCpu <= 3e-3 * scale, $"max |cpu - gpu| = {worstCpu:E3} against max |ref| {scale:E3}");
        Assert.True(qMismatch <= n * k / 1000, $"{qMismatch} of {n * k} Q8_1 codes differ from the CPU quantizer");
        Assert.True(worstNoMin > 0.05 * scale, $"the recentring is too small to be checked here ({worstNoMin:E3})");
    }

    /// <summary>
    /// Opt-in (<c>DOTLLM_GEMM_SHAPE_PROBE=1</c>): steady-state time of the Q4_0 MMQs (+ the Q8_1 quantize) against the F16 coopmat GEMM at
    /// Llama-3.2-1B/3B dense prefill shapes (<c>DOTLLM_GEMM_SHAPE_PROBE_N</c> rows, default 512).
    /// </summary>
    [SkippableFact]
    public void Probe_LlamaShapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMM_SHAPE_PROBE") == "1", "opt-in probe");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        using var cm1 = Create(device, spvDir, "cm1");
        using var dp4a = Create(device, spvDir, "dp4a");
        using var f16 = MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spvDir, QuantizationType.Q4_0)
            ? MatMulQ4_0GemmCoopmatKernel.Create(device, spvDir, QuantizationType.Q4_0) : null;
        using var f32 = MatMulQ4_0GemmF32Kernel.Create(device, spvDir, QuantizationType.Q4_0);
        (int M, int K)[] shapes = [(2048, 2048), (512, 2048), (8192, 2048), (2048, 8192), (3072, 3072), (1024, 3072), (8192, 3072), (3072, 8192)];
        int nRows = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_GEMM_SHAPE_PROBE_N"), out int nn) ? nn : 512;
        const int Reps = 10;
        foreach (var (m, k) in shapes)
        {
            using var w = device.AllocateDeviceLocal(Dequantize.RowByteSize(k, QuantizationType.Q4_0) * m);
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
            double tF32 = Time(cb => f32.Record(cb, w, b, c, m, k, nRows));
            output.WriteLine($"q4_0 n={nRows} m={m,5} k={k,5}: cm1 {T(cm1)} ms | dp4a {T(dp4a)} ms | + quant {tQ,6:F3} ms | f16 coopmat {tF16} ms "
                + $"| f32 gemm {tF32,6:F3} ms");
        }
    }
}
