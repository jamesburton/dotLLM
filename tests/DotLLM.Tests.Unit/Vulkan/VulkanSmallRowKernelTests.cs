using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity for the 2..8-row kernels added for the qwen4exp small-row fast paths (#876): the F32 / F16 / Q8_0 multi-column GEMVs against
/// their single-column counterparts, and the multi-row (NR output rows per workgroup) indexed MoE MMVQs against the one-row-per-workgroup
/// kernels. Every column / cell must agree to a few ULP (same per-row accumulation order; only FMA contraction may differ).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSmallRowKernelTests
{
    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f + 3e-5f * Math.Abs(expected[i]),
                $"{what}[{i}]: expected {expected[i]:G9} actual {actual[i]:G9}");
    }

    [SkippableTheory]
    [InlineData(4, 10240, 2)]      // qwen4exp hyper-connection inject: thin M, long K
    [InlineData(4, 10240, 5)]
    [InlineData(48, 2560, 3)]      // ssm_alpha / beta
    [InlineData(512, 2560, 8)]     // router
    [InlineData(1, 2560, 4)]       // shared-expert gate logit (M = 1)
    [InlineData(17, 132, 7)]       // K not a multiple of the 128-thread stride
    public void F32Multi_EachColumnMatchesSingleColumnGemv(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x876 + m * 7 + k * 11 + n);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);

        using var device = VulkanDevice.Create();
        using var multi = MatMulF32GemvMultiKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_f32_gemv_multi.spv missing");
        using var single = MatMulF32Kernel.Create(device, spvDir);
        using var bw = device.Allocate((long)m * k * 4);
        using var bx = device.Allocate((long)n * k * 4);
        using var by = device.Allocate((long)n * m * 4);
        using var bx1 = device.Allocate((long)k * 4);
        using var by1 = device.Allocate((long)m * 4);
        device.Upload(w, bw);
        device.Upload(x, bx);
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); multi.Record(ctx.CommandBuffer, bw, bx, by, m, k, n); ctx.SubmitAndWait(); }
        float[] actual = new float[n * m];
        device.Download(by, actual);
        for (int c = 0; c < n; c++)
        {
            device.Upload(x.AsSpan(c * k, k).ToArray(), bx1);
            single.Launch(bw, bx1, by1, m, k, 1);
            float[] exp = new float[m];
            device.Download(by1, exp);
            AssertClose(exp, actual.AsSpan(c * m, m).ToArray(), $"F32 col {c}");
        }
    }

    [SkippableTheory]
    [InlineData(2560, 640, 2)]     // shared-expert down
    [InlineData(640, 2560, 5)]     // shared-expert gate / up
    [InlineData(64, 64, 8)]
    public void F16Multi_EachColumnMatchesSingleColumnGemv(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x877 + m * 7 + k * 11 + n);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] w16 = F16Bf16Fixture.QuantizeRowsF16(w, m, k);

        using var device = VulkanDevice.Create();
        using var multi = MatMulF16GemvMultiKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_f16_gemv_multi.spv missing");
        using var single = MatMulF16GemvF32Kernel.Create(device, spvDir);
        using var bw = device.Allocate(((long)w16.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var by = device.Allocate((long)n * m * 4);
        using var bx1 = device.Allocate((long)k * 4);
        using var by1 = device.Allocate((long)m * 4);
        device.Upload(new ReadOnlySpan<byte>(w16), bw);
        device.Upload(x, bx);
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); multi.Record(ctx.CommandBuffer, bw, bx, by, m, k, n); ctx.SubmitAndWait(); }
        float[] actual = new float[n * m];
        device.Download(by, actual);
        for (int c = 0; c < n; c++)
        {
            device.Upload(x.AsSpan(c * k, k).ToArray(), bx1);
            single.Launch(bw, bx1, by1, m, k);
            float[] exp = new float[m];
            device.Download(by1, exp);
            AssertClose(exp, actual.AsSpan(c * m, m).ToArray(), $"F16 col {c}");
        }
    }

    [SkippableTheory]
    [InlineData(320, 10240, 2)]    // hc down
    [InlineData(10240, 320, 5)]    // hc up (K = 10 blocks: window tail)
    [InlineData(6144, 2560, 8)]
    [InlineData(40, 96, 3)]
    public void Q8Multi_EachColumnMatchesSingleColumnMmvq(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer dot product.");
        var rng = new Random(0x878 + m * 7 + k * 11 + n);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = Quantize.FromFloat32(w, (long)m * k, QuantizationType.Q8_0);

        using var multi = MatMulQ8_0MmvqMultiKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq_multi.spv missing");
        using var single = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)!;
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        using var bw = device.Allocate(((long)wq.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var bxq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bxds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var by = device.Allocate((long)n * m * 4);
        using var bxq1 = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
        using var bxds1 = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
        using var by1 = device.Allocate((long)m * 4);
        device.Upload(new ReadOnlySpan<byte>(wq), bw);
        device.Upload(x, bx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bx, bxq, bxds, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            multi.Record(ctx.CommandBuffer, bw, bxq, bxds, by, m, k, n);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[n * m];
        device.Download(by, actual);
        for (int c = 0; c < n; c++)
        {
            using var bx1 = device.Allocate((long)k * 4);
            device.Upload(x.AsSpan(c * k, k).ToArray(), bx1);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quant.Record(ctx.CommandBuffer, bx1, bxq1, bxds1, 1, k);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                single.Record(ctx.CommandBuffer, bw, bxq1, bxds1, by1, m, k);
                ctx.SubmitAndWait();
            }
            float[] exp = new float[m];
            device.Download(by1, exp);
            AssertClose(exp, actual.AsSpan(c * m, m).ToArray(), $"Q8_0 col {c}");
        }
    }

    [SkippableTheory]
    [InlineData(20, 16, 640, 2560, 10)]    // 2 rows x top-10 gate/up shape (M = 640, K = 2560), 10 distinct experts of 16
    [InlineData(50, 16, 640, 2560, 12)]
    [InlineData(10, 9, 12, 512, 4)]
    public void Q4KMultiRow_MatchesOneRowPerWorkgroup(int n, int numExperts, int m, int k, int active)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer dot product.");
        var rng = new Random(0x879 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        byte[] bank = Quantize.FromFloat32(F16Bf16Fixture.RandomFloats(rng, numExperts * m * k, 0.25f), (long)numExperts * m * k, QuantizationType.Q4_K);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        int[] idx = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, active);

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        using var one = MoeIndexedMatmulKQuantMmvqKernel.TryCreate(device, spvDir, MoeGroupedKQuant.Q4_K)!;
        using var mr = MoeIndexedMatmulKQuantMmvqKernel.TryCreate(device, spvDir, MoeGroupedKQuant.Q4_K, rowsPerGroup: 2)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q4_k_mmvq_xdiv_mr.spv missing");
        using var bb = device.Allocate(((long)bank.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var bxq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bxds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bi = device.Allocate((long)n * 4);
        using var y1 = device.Allocate((long)n * m * 4);
        using var y2 = device.Allocate((long)n * m * 4);
        device.Upload(new ReadOnlySpan<byte>(bank), bb);
        device.Upload(x, bx);
        device.Upload(MemoryMarshal.AsBytes<int>(idx), bi);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bx, bxq, bxds, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            one.Record(ctx.CommandBuffer, bb, bxq, bxds, bi, y1, m, k, n, numExperts);
            mr.Record(ctx.CommandBuffer, bb, bxq, bxds, bi, y2, m, k, n, numExperts);
            ctx.SubmitAndWait();
        }
        float[] a = new float[n * m], b = new float[n * m];
        device.Download(y1, a);
        device.Download(y2, b);
        AssertClose(a, b, "Q4_K MR");
    }

    [SkippableTheory]
    [InlineData(20, 16, 2560, 640, 10)]    // down shape: M = hidden 2560, K = intermediate 640
    [InlineData(50, 16, 2560, 640, 12)]
    [InlineData(10, 9, 12, 96, 4)]
    public void Q5_1MultiRow_MatchesOneRowPerWorkgroup(int n, int numExperts, int m, int k, int active)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer dot product.");
        var rng = new Random(0x87A + n * 31 + numExperts * 17 + m * 11 + k * 7);
        byte[] bank = Quantize.FromFloat32(F16Bf16Fixture.RandomFloats(rng, numExperts * m * k, 0.25f), (long)numExperts * m * k, QuantizationType.Q5_1);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        int[] idx = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, active);
        float[] scale = new float[numExperts];
        for (int e = 0; e < numExperts; e++) scale[e] = 0.5f + 0.13f * e;

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        using var one = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spvDir)!;
        using var mr = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spvDir, 4)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q5_1_mmvq_mr.spv missing");
        using var bb = device.Allocate(((long)bank.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var bxq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bxds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bi = device.Allocate((long)n * 4);
        using var bs = device.Allocate((long)numExperts * 4);
        using var y1 = device.Allocate((long)n * m * 4);
        using var y2 = device.Allocate((long)n * m * 4);
        device.Upload(new ReadOnlySpan<byte>(bank), bb);
        device.Upload(x, bx);
        device.Upload(MemoryMarshal.AsBytes<int>(idx), bi);
        device.Upload(scale, bs);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bx, bxq, bxds, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            one.Record(ctx.CommandBuffer, bb, bxq, bxds, bi, y1, bs, m, k, n, numExperts);
            mr.Record(ctx.CommandBuffer, bb, bxq, bxds, bi, y2, bs, m, k, n, numExperts);
            ctx.SubmitAndWait();
        }
        float[] a = new float[n * m], b = new float[n * m];
        device.Download(y1, a);
        device.Download(y2, b);
        AssertClose(a, b, "Q5_1 MR");
    }
}
