using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the MLA kv_b split Vulkan kernel.
/// </summary>
/// <remarks>
/// The split is a deterministic per-element copy with no arithmetic, so the
/// reference is a scalar CPU loop — both sides should produce bit-identical
/// outputs (no F32 reduction-order noise).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMlaKvSplitF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 1, 8, 8)]                  // smallest: 1 token, 1 head
    [InlineData(1, 8, 128, 128)]              // DeepSeek-V2-Lite shape: 8 heads, qkNope=vHead=128
    [InlineData(1, 16, 128, 128)]             // DeepSeek-V2 full: 16 heads
    [InlineData(8, 8, 128, 128)]              // prefill shape, 8 tokens
    [InlineData(1, 4, 32, 16)]                // qkNope ≠ vHead
    [InlineData(2, 2, 4, 6)]                  // tiny non-power-of-2 head dims
    public void Launch_MatchesCpuReference(
        int seqLen, int numHeads, int qkNopeHeadDim, int vHeadDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xCAB + seqLen * 13 + numHeads * 7 + qkNopeHeadDim * 3 + vHeadDim);
        int perHead = qkNopeHeadDim + vHeadDim;
        float[] kvBExpanded = RandomFloats(rng, seqLen * numHeads * perHead);

        // Reference split.
        var (expectedKNope, expectedV) =
            CpuKvSplit(kvBExpanded, seqLen, numHeads, qkNopeHeadDim, vHeadDim);

        using var device = VulkanDevice.Create();
        using var kernel = MlaKvSplitF32Kernel.Create(device, spvDir);

        using var bufExp = device.Allocate((long)kvBExpanded.Length * sizeof(float));
        using var bufKNope = device.Allocate((long)expectedKNope.Length * sizeof(float));
        using var bufV = device.Allocate((long)expectedV.Length * sizeof(float));

        device.Upload(kvBExpanded, bufExp);

        kernel.Launch(bufExp, bufKNope, bufV, seqLen, numHeads, qkNopeHeadDim, vHeadDim);

        var actualKNope = new float[expectedKNope.Length];
        var actualV = new float[expectedV.Length];
        device.Download(bufKNope, actualKNope);
        device.Download(bufV, actualV);

        // Bit-identical — pure copy, no FP arithmetic.
        for (int i = 0; i < expectedKNope.Length; i++)
            Assert.Equal(expectedKNope[i], actualKNope[i]);
        for (int i = 0; i < expectedV.Length; i++)
            Assert.Equal(expectedV[i], actualV[i]);
    }

    private static (float[] kNope, float[] v) CpuKvSplit(
        float[] expanded, int seqLen, int numHeads, int qkNopeHeadDim, int vHeadDim)
    {
        int perHead = qkNopeHeadDim + vHeadDim;
        var kNope = new float[seqLen * numHeads * qkNopeHeadDim];
        var v = new float[seqLen * numHeads * vHeadDim];

        for (int t = 0; t < seqLen; t++)
        {
            int expRowBase = t * numHeads * perHead;
            int kNopeRowBase = t * numHeads * qkNopeHeadDim;
            int vRowBase = t * numHeads * vHeadDim;
            for (int h = 0; h < numHeads; h++)
            {
                int srcHeadBase = expRowBase + h * perHead;
                for (int d = 0; d < qkNopeHeadDim; d++)
                    kNope[kNopeRowBase + h * qkNopeHeadDim + d] = expanded[srcHeadBase + d];
                for (int d = 0; d < vHeadDim; d++)
                    v[vRowBase + h * vHeadDim + d] = expanded[srcHeadBase + qkNopeHeadDim + d];
            }
        }
        return (kNope, v);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
