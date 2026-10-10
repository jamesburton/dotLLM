using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The wide Q8_0 MMVQ twins (#885) against the one-wave-per-row kernel they replace for the qwen4exp gated-residual projections:
/// 4 rows per workgroup must be BIT-identical (same lane layout, same accumulation order); 4 subgroups per row changes the accumulation order
/// (fixed, deterministic) so it is held to a few ULP of the base kernel. Shapes include a row count that is not a multiple of the rows-per-group
/// and a short K (5 blocks) that leaves most lanes of the window idle.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0MmvqWideKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    [SkippableTheory]
    [InlineData(10240, 320, 4, 1)]   // gated-residual up
    [InlineData(10, 160, 4, 1)]      // rows not a multiple of 4, odd block count
    [InlineData(320, 10240, 1, 4)]   // gated-residual down
    [InlineData(7, 224, 1, 4)]       // KSPLIT > blocks/8 windows: parts with nothing to do
    [InlineData(6, 2048, 2, 2)]      // both knobs at once
    public void Wide_MatchesTheBaseKernel(int m, int k, int rows, int ksplit)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device does not advertise VK_KHR_shader_integer_dot_product.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing or unsupported.");
        using var wide = MatMulQ8_0MmvqWideKernel.TryCreate(device, spvDir, rows, ksplit);
        Skip.If(wide is null, "wide Q8_0 MMVQ needs the wave32 pin on this device.");

        var rng = new Random(0x885 + m * 3 + k);
        float[] w = new float[m * k];
        for (int i = 0; i < w.Length; i++) w[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * 1.0);
        byte[] wq = QuantizeRows(w, m, k);

        using var bufW = device.Allocate(((long)wq.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var yBase = device.Allocate((long)m * sizeof(float));
        using var yWide = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(wq), bufW);
        device.Upload(x, bufX);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, yBase, m, k);
            wide!.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, yWide, m, k);
            ctx.SubmitAndWait();
        }

        float[] a = new float[m], b = new float[m];
        device.Download(yBase, a);
        device.Download(yWide, b);

        int bitDiff = 0;
        float maxRel = 0;
        float scale = a.Max(MathF.Abs);
        for (int i = 0; i < m; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) bitDiff++;
            maxRel = MathF.Max(maxRel, MathF.Abs(a[i] - b[i]) / (scale + 1e-12f));
        }
        if (ksplit == 1)
            Assert.True(bitDiff == 0, $"rows={rows}: {bitDiff} of {m} outputs differ from the base kernel (must be bit-identical)");
        Assert.True(maxRel < 1e-5f, $"rows={rows} ksplit={ksplit}: max |wide - base| / max|base| = {maxRel:E3}");
        // a dead kernel (never dispatched) would leave the output buffer zero-initialised / stale and fail the bound above unless the answer is all zeros
        Assert.True(scale > 0, "oracle output is all zero - the comparison is vacuous");
    }

    private static unsafe byte[] QuantizeRows(float[] src, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var dst = new byte[m * rowBytes];
        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
        }
        return dst;
    }
}
