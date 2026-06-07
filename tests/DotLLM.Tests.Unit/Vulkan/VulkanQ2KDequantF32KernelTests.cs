using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan Q2_K → F32 dequant kernel against the CPU
/// oracle <c>DequantizeKQuants.DequantizeQ2_K</c>. Tolerance is 0 ULP — both
/// paths read the same bytes and emit the same FP32 product (no reduction,
/// single multiply per element).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ2KDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(1024)]
    public void Launch_MatchesCpuOracle(int totalBlocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalBlocks * Q2KFixture.Q2KGroupSize;
        var rng = new Random(0x2C_AF_E0 ^ (totalBlocks * 7));
        float[] srcF32 = Q2KFixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] q2kBytes = Q2KFixture.QuantizeRows(srcF32, m: totalBlocks, k: Q2KFixture.Q2KGroupSize);
        Q2KFixture.AssertFixtureRoundtrip(srcF32, q2kBytes, m: totalBlocks, k: Q2KFixture.Q2KGroupSize);

        float[] expected = Q2KFixture.CpuDequantizeQ2K(q2kBytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Q2KDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)q2kBytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(q2kBytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalBlocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        Q2KFixture.AssertClose(expected, actual, m: totalBlocks, k: Q2KFixture.Q2KGroupSize,
            absTol: 0f, relTol: 0f);
    }
}
