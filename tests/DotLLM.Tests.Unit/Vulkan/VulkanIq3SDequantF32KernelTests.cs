using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan IQ3_S → F32 dequant kernel against the CPU
/// oracle <c>Dequantize.DequantizeIQ3_S</c>. 0 ULP tolerance.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanIq3SDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void Launch_MatchesCpuOracle(int totalSuperblocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalSuperblocks * Iq3Fixture.Iq3GroupSize;
        var rng = new Random(0x33_5C_B7 ^ (totalSuperblocks * 13));
        float[] srcF32 = Iq3Fixture.RandomFloats(rng, elements, range: 0.05f);
        byte[] iqBytes = Iq3Fixture.QuantizeRowsIq3S(srcF32, m: totalSuperblocks, k: Iq3Fixture.Iq3GroupSize);
        Iq3Fixture.AssertFixtureRoundtripIq3S(srcF32, iqBytes, m: totalSuperblocks, k: Iq3Fixture.Iq3GroupSize);

        float[] expected = Iq3Fixture.CpuDequantizeIq3S(iqBytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Iq3SDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)iqBytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(iqBytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalSuperblocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        Iq3Fixture.AssertClose(expected, actual, $"iq3_s dequant superblocks={totalSuperblocks}",
            absTol: 0f, relTol: 0f);
    }
}
