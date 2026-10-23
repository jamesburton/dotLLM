using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The GDN <c>[Q | K | V]</c> row split and the attention Q+gate per-head de-interleave replace per-token <c>vkCmdCopyBuffer</c> loops; they
/// move F32 values only, so the result must equal the host-side reference BIT-EXACTLY. Shapes keep kDim != vDim and numHeads != headDim so a
/// swapped stride or head/dim transposition lands elsewhere.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanDeinterleaveF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 6, 10)]
    [InlineData(5, 8, 20)]
    [InlineData(37, 128, 256)]
    [InlineData(1100, 64, 96)]   // > 65535 workgroups' worth of rows is not reached, but spans several 256-groups per row
    public void GdnQkvSplit_MatchesReference(int seqLen, int kDim, int vDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = DeinterleaveF32Kernel.TryCreate(device, spvDir, DeinterleaveF32Kernel.Kind.GdnQkvSplit)
            ?? throw new Xunit.Sdk.XunitException("gdn_split_qkv_f32.spv missing.");

        int convDim = 2 * kDim + vDim;
        float[] qkv = new float[seqLen * convDim];
        for (int i = 0; i < qkv.Length; i++) qkv[i] = i * 0.25f - 7f;

        using var qkvBuf = device.Allocate(qkv.Length * sizeof(float));
        using var qBuf = device.Allocate((long)seqLen * kDim * sizeof(float));
        using var kBuf = device.Allocate((long)seqLen * kDim * sizeof(float));
        using var vBuf = device.Allocate((long)seqLen * vDim * sizeof(float));
        device.Upload(qkv, qkvBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.RecordGdnQkvSplit(ctx.CommandBuffer, qkvBuf, qBuf, kBuf, vBuf, seqLen, kDim, vDim);
            ctx.SubmitAndWait();
        }
        float[] q = new float[seqLen * kDim], k = new float[seqLen * kDim], v = new float[seqLen * vDim];
        device.Download(qBuf, q); device.Download(kBuf, k); device.Download(vBuf, v);

        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < kDim; c++)
            {
                Assert.Equal(qkv[t * convDim + c], q[t * kDim + c]);
                Assert.Equal(qkv[t * convDim + kDim + c], k[t * kDim + c]);
            }
            for (int c = 0; c < vDim; c++) Assert.Equal(qkv[t * convDim + 2 * kDim + c], v[t * vDim + c]);
        }
    }

    [SkippableTheory]
    [InlineData(1, 3, 8)]
    [InlineData(7, 4, 16)]
    [InlineData(300, 16, 256)]
    public void QGateDeinterleave_MatchesReference(int seqLen, int numHeads, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = DeinterleaveF32Kernel.TryCreate(device, spvDir, DeinterleaveF32Kernel.Kind.QGateDeinterleave)
            ?? throw new Xunit.Sdk.XunitException("qgate_deinterleave_f32.spv missing.");

        int per = numHeads * headDim;
        float[] qg = new float[seqLen * 2 * per];
        for (int i = 0; i < qg.Length; i++) qg[i] = i * 0.5f + 3f;

        using var qgBuf = device.Allocate(qg.Length * sizeof(float));
        using var qBuf = device.Allocate((long)seqLen * per * sizeof(float));
        using var gBuf = device.Allocate((long)seqLen * per * sizeof(float));
        device.Upload(qg, qgBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.RecordQGateDeinterleave(ctx.CommandBuffer, qgBuf, qBuf, gBuf, seqLen, numHeads, headDim);
            ctx.SubmitAndWait();
        }
        float[] q = new float[seqLen * per], g = new float[seqLen * per];
        device.Download(qBuf, q); device.Download(gBuf, g);

        for (int t = 0; t < seqLen; t++)
        for (int h = 0; h < numHeads; h++)
        for (int d = 0; d < headDim; d++)
        {
            int src = t * 2 * per + h * 2 * headDim + d;
            Assert.Equal(qg[src], q[t * per + h * headDim + d]);
            Assert.Equal(qg[src + headDim], g[t * per + h * headDim + d]);
        }
    }
}
