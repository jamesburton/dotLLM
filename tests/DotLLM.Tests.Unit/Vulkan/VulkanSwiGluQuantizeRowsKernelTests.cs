using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #912: the fused activation + Q8_1 row quantize used between the routed-expert gate/up and down MMVQ must be BIT-IDENTICAL to the standalone
/// activation kernel (<see cref="SwiGluF32Kernel"/> / <c>VulkanGptOssKernels.RecordSwiGluOai</c>) followed by <see cref="QuantizeQ8_1RowsKernel"/>:
/// the fusion removes a dispatch and a barrier, it must not move a logit. xq and xds are compared exactly.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanSwiGluQuantizeRowsKernelTests
{
    [SkippableTheory]
    [InlineData(1536, 4, false)]    // GLM-4.7-Flash: top-4 rows of the 1536-wide expert intermediate
    [InlineData(2880, 4, true)]     // gpt-oss: clamped SwiGLU, 90 blocks per row
    [InlineData(2880, 1, true)]
    [InlineData(256, 15, false)]    // the arm's row ceiling x topK 1
    [InlineData(2880, 15, true)]
    [InlineData(32, 1, false)]      // single block
    public void BitIdenticalToStandalonePair(int k, int rows, bool oai)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var fused = SwiGluQuantizeRowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("swiglu_quantize_q8_1_rows.spv missing.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);

        int n = k * rows;
        var rng = new Random(0x912 + k + rows * 7 + (oai ? 1 : 0));
        float[] gate = new float[n], up = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = (i % 89 == 0) ? 0f : (float)((rng.NextDouble() * 2 - 1) * (oai ? 12 : 1 + (i % 11)));   // oai: clamp region (|x| > 7) is exercised
            up[i] = (float)((rng.NextDouble() * 2 - 1) * (oai ? 12 : 0.7));
        }
        using var gBuf = device.Allocate((long)n * 4);
        using var uBuf = device.Allocate((long)n * 4);
        using var actBuf = device.Allocate((long)n * 4);
        using var xqRef = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(rows, k));
        using var xdsRef = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(rows, k));
        using var xqF = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(rows, k));
        using var xdsF = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(rows, k));
        device.Upload(gate, gBuf);
        device.Upload(up, uBuf);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            if (oai) gpt.RecordSwiGluOai(ctx.CommandBuffer, gBuf, uBuf, actBuf, n);
            else swiglu.Record(ctx.CommandBuffer, gBuf, uBuf, actBuf, n);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            quant.Record(ctx.CommandBuffer, actBuf, xqRef, xdsRef, rows, k);
            fused.Record(ctx.CommandBuffer, gBuf, uBuf, xqF, xdsF, n, oai);
            ctx.SubmitAndWait();
        }
        uint[] a = Bits(device, xqRef, n / 4), b = Bits(device, xqF, n / 4);
        for (int i = 0; i < a.Length; i++) Assert.True(a[i] == b[i], $"xq[{i}] (k={k} rows={rows} oai={oai}): pair 0x{a[i]:X8} fused 0x{b[i]:X8}");
        float[] sa = new float[n / 16], sb = new float[n / 16];
        device.Download(xdsRef, sa);
        device.Download(xdsF, sb);
        for (int i = 0; i < sa.Length; i++) Assert.True(sa[i].Equals(sb[i]), $"xds[{i}] (k={k} rows={rows} oai={oai}): pair {sa[i]:G9} fused {sb[i]:G9}");
        Assert.Contains(a, w => w != 0u);   // not a vacuous all-zero comparison
    }

    private static uint[] Bits(VulkanDevice device, VulkanDevice.Buffer buf, int count)
    {
        float[] f = new float[count];
        device.Download(buf, f);
        return Array.ConvertAll(f, BitConverter.SingleToUInt32Bits);
    }
}
