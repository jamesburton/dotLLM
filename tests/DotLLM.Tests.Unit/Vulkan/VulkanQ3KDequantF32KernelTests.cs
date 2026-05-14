using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan Q3_K → F32 dequant kernel against the CPU
/// oracle <c>DequantizeKQuants.DequantizeQ3_KScalar</c>. Tolerance is 0 ULP —
/// both paths read the same bytes and emit the same FP32 product (no
/// reduction, single multiply per element).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ3KDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(1024)]
    public void Launch_MatchesCpuOracle(int totalBlocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalBlocks * Q3KFixture.Q3KGroupSize;
        var rng = new Random(0x3C_AF_E0 ^ (totalBlocks * 7));
        float[] srcF32 = Q3KFixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] q3kBytes = Q3KFixture.QuantizeRows(srcF32, m: totalBlocks, k: Q3KFixture.Q3KGroupSize);
        Q3KFixture.AssertFixtureRoundtrip(srcF32, q3kBytes, m: totalBlocks, k: Q3KFixture.Q3KGroupSize);

        float[] expected = Q3KFixture.CpuDequantizeQ3K(q3kBytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Q3KDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)q3kBytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(q3kBytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalBlocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        Q3KFixture.AssertClose(expected, actual, m: totalBlocks, k: Q3KFixture.Q3KGroupSize,
            absTol: 0f, relTol: 0f);
    }
}
