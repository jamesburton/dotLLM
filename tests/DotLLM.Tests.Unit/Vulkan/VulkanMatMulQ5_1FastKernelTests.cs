using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The #1613 dense Q5_1 fast kernels: dp4a MMVQ decode (<see cref="MatMulQ4_0MmvqKernel"/> Q5_1 form), the int8-coopmat
/// <see cref="MatMulMmqCm1Kernel"/> and dp4a <see cref="MatMulQ4_1MmqKernel"/> prefill GEMMs on Q8_1 activations, and the F16 coopmat GEMM
/// (<see cref="MatMulQ4_0GemmCoopmatKernel"/> Q5_1 form). Oracles: for the Q8_1-activation kernels a double-precision sum over the SAME Q8_1
/// codes the kernel read (only accumulation order may differ) plus the CPU's packed Q5_1 x Q8_1 GEMM (#1159, llama.cpp-golden) on those
/// codes; for the coopmat GEMM the CPU dequant dotted in double. Every fixture carries an offset in weights and activations, so the min term
/// and the 5th bit are each a large share of every output: a kernel that drops either fails by O(1), asserted as a control
/// (<c>noMinGap</c>, <c>noHighBitGap</c>). Shapes cover partial M / N tiles, the L2 group boundary, a partial last K step and GLM-4.5-Air's
/// shared-expert down (m = 4096, k = 1408). <c>DOTLLM_GLM4MOE_GGUF</c> adds the same checks on a real Q5_1 tensor from that file.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMatMulQ5_1FastKernelTests(ITestOutputHelper output)
{
    private const int BlockBytes = 24;

    public static IEnumerable<object[]> Q8Shapes()
    {
        foreach (string kind in new[] { "cm1", "dp4a" })
        {
            yield return [kind, 77, 200, 352];     // partial M / N tiles, 11 blocks (a partial last K step for cm1's BK_STEP 4)
            yield return [kind, 5, 2112, 1024];    // L2 group boundary
            yield return [kind, 130, 512, 2048];   // > one token tile
            yield return [kind, 300, 72, 160];     // three token tiles, the last partial; one partial weight tile
            yield return [kind, 64, 4096, 1408];   // GLM-4.5-Air shared-expert down shape (64 rows)
        }
        yield return ["mmvq", 1, 37, 1408];
        yield return ["mmvq", 1, 4096, 1408];
        yield return ["mmvq", 1, 8, 64];          // 2 blocks: lanes past the row are masked
        yield return ["mmvq", 1, 21, 2080];       // 65 blocks: one U = 2 chunk + a ragged tail
    }

    [SkippableTheory]
    [MemberData(nameof(Q8Shapes))]
    public void Q8_1Kernels_MatchReferenceAndCpu(string kind, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(1613 + n * 7 + m * 3 + k);
        float[] w = new float[m * k];
        for (int i = 0; i < w.Length; i++) w[i] = 0.05f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;   // offset rows: m != 0
        byte[] wq = Quantize.FromFloat32(w, w.Length, QuantizationType.Q5_1);
        RunQ8Case(spvDir, kind, n, m, k, wq, rng, "synthetic");
    }

    [SkippableTheory]
    [InlineData("mmvq", 1)]
    [InlineData("cm1", 64)]
    [InlineData("dp4a", 64)]
    [InlineData("coopmat", 64)]
    public void RealGlm4MoeSharedDown_Q5_1(string kind, int n)
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_GLM4MOE_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var gguf = GgufFile.Open(path!);
        var t = gguf.TensorsByName["blk.6.ffn_down_shexp.weight"];
        Assert.Equal(QuantizationType.Q5_1, t.QuantizationType);
        int k = (int)t.Shape[0], m = (int)(t.Shape.ElementCount / t.Shape[0]);
        byte[] wq = new byte[Dequantize.RowByteSize(k, QuantizationType.Q5_1) * m];
        unsafe { new ReadOnlySpan<byte>((void*)gguf.TensorDataPointer(in t), wq.Length).CopyTo(wq); }
        output.WriteLine($"real blk.6.ffn_down_shexp.weight: m={m} k={k} ({wq.Length >> 20} MiB)");
        var rng = new Random(6);
        if (kind == "coopmat") RunCoopmatCase(spvDir, n, m, k, wq, rng, "real");
        else RunQ8Case(spvDir, kind, n, m, k, wq, rng, "real");
    }

    public static IEnumerable<object[]> CoopmatShapes()
    {
        yield return [77, 200, 352];
        yield return [130, 512, 2048];
        yield return [300, 72, 160];
        yield return [64, 4096, 1408];
        yield return [512, 1024, 1408];   // several 128x128 tiles
    }

    [SkippableTheory]
    [MemberData(nameof(CoopmatShapes))]
    public void CoopmatGemm_MatchesDequant(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(2613 + n * 7 + m * 3 + k);
        float[] w = new float[m * k];
        for (int i = 0; i < w.Length; i++) w[i] = 0.05f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        byte[] wq = Quantize.FromFloat32(w, w.Length, QuantizationType.Q5_1);
        RunCoopmatCase(spvDir, n, m, k, wq, rng, "synthetic");
    }

    private void RunCoopmatCase(string spvDir, int n, int m, int k, byte[] wq, Random rng, string tag)
    {
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spvDir, QuantizationType.Q5_1), "Q5_1 coopmat GEMM unavailable on this device.");
        using var kern = MatMulQ4_0GemmCoopmatKernel.Create(device, spvDir, QuantizationType.Q5_1);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = 0.3f + (float)(rng.NextDouble() * 2 - 1);
        float[] deq = new float[(long)m * k];
        unsafe { fixed (byte* p = wq) Dequantize.ToFloat32((nint)p, deq.Length, QuantizationType.Q5_1, deq); }
        using var bufW = device.Allocate(wq.Length);
        using var bufX = device.Allocate((long)x.Length * 4);
        using var bufY = device.Allocate((long)n * m * 4);
        device.Upload(wq, bufW);
        device.Upload(x, bufX);
        kern.Launch(bufW, bufX, bufY, m, k, n);
        float[] got = new float[n * m];
        device.Download(bufY, got);

        int blocks = k / 32;
        double worst = 0, scale = 0, noMinGap = 0, noHighGap = 0;
        for (int r = 0; r < n; r++)
            for (int o = 0; o < m; o++)
            {
                double acc = 0, noMin = 0, noHigh = 0;
                for (int j = 0; j < k; j++)
                {
                    int b = (o * blocks + j / 32) * BlockBytes;
                    double d = (double)BitConverter.ToHalf(wq, b), mn = (double)BitConverter.ToHalf(wq, b + 2);
                    uint qh = BitConverter.ToUInt32(wq, b + 4);
                    int jj = j % 32;
                    int nib = jj < 16 ? wq[b + 8 + jj] & 0xF : wq[b + 8 + jj - 16] >> 4;
                    int code = nib | (int)(((qh >> jj) & 1u) << 4);
                    double xv = x[r * k + j];
                    acc += (d * code + mn) * xv;
                    noMin += d * code * xv;
                    noHigh += (d * nib + mn) * xv;
                }
                double accDeq = 0;
                for (int j = 0; j < k; j++) accDeq += deq[(long)o * k + j] * (double)x[r * k + j];
                Assert.True(Math.Abs(accDeq - acc) <= 1e-6 * Math.Max(1, Math.Abs(acc)), "test-internal: hand decode != Dequantize.ToFloat32");
                float g = got[r * m + o];
                worst = Math.Max(worst, Math.Abs(acc - g));
                scale = Math.Max(scale, Math.Abs(acc));
                noMinGap = Math.Max(noMinGap, Math.Abs(acc - noMin));
                noHighGap = Math.Max(noHighGap, Math.Abs(acc - noHigh));
            }
        output.WriteLine($"q5_1 coopmat {tag} n={n} m={m} k={k}: max|ref-gpu|={worst:E3} max|ref|={scale:E3} no-min gap={noMinGap:E3} no-5th-bit gap={noHighGap:E3}");
        // F16 A operand (d * code + m rounded to fp16, ~2^-11 relative) and F16 B: the error scales with sqrt(k) terms of size ~|w||x|.
        double tol = 2.5e-3 * scale * Math.Sqrt(k / 352.0) + 1e-4;
        Assert.True(worst <= tol, $"max |ref - gpu| = {worst:E3} > {tol:E3} (max |ref| {scale:E3})");
        Assert.True(noMinGap > 0.05 * scale, $"fixture cannot see the min term ({noMinGap:E3})");
        Assert.True(noHighGap > 0.05 * scale, $"fixture cannot see the 5th bit ({noHighGap:E3})");
    }

    private void RunQ8Case(string spvDir, string kind, int n, int m, int k, byte[] wq, Random rng, string tag)
    {
        using var device = VulkanDevice.Create();
        IDisposable? kernel = kind switch
        {
            "cm1" => MatMulMmqCm1Kernel.TryCreate(device, spvDir, QuantizationType.Q5_1),
            "dp4a" => MatMulQ4_1MmqKernel.TryCreate(device, spvDir, QuantizationType.Q5_1),
            _ => MatMulQ4_0MmvqKernel.TryCreate(device, spvDir, quant: QuantizationType.Q5_1),
        };
        Skip.If(kernel is null, $"Q5_1 {kind} unavailable on this device.");
        using var _k = kernel;
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = 0.3f + (float)(rng.NextDouble() * 2 - 1);            // offset activations: s_x != 0
        long rowBytes = Dequantize.RowByteSize(k, QuantizationType.Q5_1);
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
            switch (kernel)
            {
                case MatMulMmqCm1Kernel cm1: cm1.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n); break;
                case MatMulQ4_1MmqKernel dp4a: dp4a.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n); break;
                case MatMulQ4_0MmvqKernel mmvq: Assert.Equal(1, n); mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k); break;
            }
            ctx.SubmitAndWait();
        }

        float[] xqF = new float[n * k / 4];
        device.Download(bufXq, xqF);
        sbyte[] xq = MemoryMarshal.Cast<float, sbyte>(xqF).ToArray();
        float[] xds = new float[n * (k / 32) * 2];
        device.Download(bufXds, xds);
        float[] c = new float[n * m];
        device.Download(bufC, c);

        // CPU oracle: the packed Q5_1 x Q8_1 GEMM on the DEVICE's Q8_1 codes (block_q8_1: fp16 d, fp16 s, qs[32]).
        int blocks = n * (k / 32);
        byte[] devQ8 = new byte[blocks * 36];
        for (int bi = 0; bi < blocks; bi++)
        {
            BitConverter.TryWriteBytes(devQ8.AsSpan(bi * 36, 2), (Half)xds[bi * 2]);
            BitConverter.TryWriteBytes(devQ8.AsSpan(bi * 36 + 2, 2), (Half)xds[bi * 2 + 1]);
            for (int j = 0; j < 32; j++) devQ8[bi * 36 + 4 + j] = (byte)xq[bi * 32 + j];
        }
        float[] cpu = new float[n * m];
        unsafe
        {
            fixed (byte* p = wq) fixed (byte* pq = devQ8) fixed (float* pb = b) fixed (float* pc = cpu)
                MatMul.GemmLegacyQuant(p, QuantizationType.Q5_1, pb, pc, m, k, n, preQuantizedInput: pq);
        }

        double worst = 0, worstCpu = 0, noMinGap = 0, noHighGap = 0, scale = 0;
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0, minTerm = 0, highTerm = 0;
                for (int blk = 0; blk < k / 32; blk++)
                {
                    double dx = (double)(Half)xds[(t * (k / 32) + blk) * 2];
                    double sx = (double)(Half)xds[(t * (k / 32) + blk) * 2 + 1];
                    int o = (int)(r * rowBytes + blk * BlockBytes);
                    uint qh = BitConverter.ToUInt32(wq, o + 4);
                    int dot = 0, dotHigh = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        int x0 = t * k + blk * 32 + j;
                        int lo = wq[o + 8 + j] & 0xF, hi = wq[o + 8 + j] >> 4;
                        int hb0 = (int)((qh >> j) & 1u) << 4, hb1 = (int)((qh >> (j + 16)) & 1u) << 4;
                        dot += (lo | hb0) * xq[x0] + (hi | hb1) * xq[x0 + 16];
                        dotHigh += hb0 * xq[x0] + hb1 * xq[x0 + 16];
                    }
                    double d = (double)BitConverter.ToHalf(wq, o);
                    double mTerm = (double)BitConverter.ToHalf(wq, o + 2) * sx;
                    acc += d * dx * dot + mTerm;
                    minTerm += mTerm;
                    highTerm += d * dx * dotHigh;
                }
                float got = c[t * m + r];
                worst = Math.Max(worst, Math.Abs(acc - got));
                worstCpu = Math.Max(worstCpu, Math.Abs(cpu[t * m + r] - got));
                noMinGap = Math.Max(noMinGap, Math.Abs(minTerm));
                noHighGap = Math.Max(noHighGap, Math.Abs(highTerm));
                scale = Math.Max(scale, Math.Abs(acc));
            }
        output.WriteLine($"q5_1 {kind} {tag} n={n} m={m} k={k}: max|ref-gpu|={worst:E3} max|cpu-gpu|={worstCpu:E3} (rel {worstCpu / scale:E2}) max|ref|={scale:E3} "
            + $"no-min gap={noMinGap:E3} no-5th-bit gap={noHighGap:E3}");
        Assert.True(worst <= 2e-5 * scale + 1e-5, $"max |ref - gpu| = {worst:E3} against max |ref| {scale:E3}");
        Assert.True(worstCpu <= 2e-5 * scale + 1e-5, $"max |cpu - gpu| = {worstCpu:E3} against max |ref| {scale:E3}");
        Assert.True(noMinGap > 0.05 * scale, $"the min term is too small to be checked here ({noMinGap:E3})");
        Assert.True(noHighGap > 0.05 * scale, $"the 5th-bit term is too small to be checked here ({noHighGap:E3})");
    }
}
