using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan IQ2_S → F32 dequant kernel against the CPU
/// oracle <c>Dequantize.DequantizeIQ2_S</c>. 0 ULP tolerance.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanIq2SDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(16)]
    public void Launch_MatchesCpuOracle(int totalSuperblocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalSuperblocks * Iq2Fixture.Iq2GroupSize;
        var rng = new Random(0x3A_2B_C5 ^ (totalSuperblocks * 13));
        float[] srcF32 = Iq2Fixture.RandomFloats(rng, elements, range: 0.05f);
        byte[] iqBytes = Iq2Fixture.QuantizeRowsIq2S(srcF32, m: totalSuperblocks, k: Iq2Fixture.Iq2GroupSize);
        Iq2Fixture.AssertFixtureRoundtripIq2S(srcF32, iqBytes, m: totalSuperblocks, k: Iq2Fixture.Iq2GroupSize);

        float[] expected = Iq2Fixture.CpuDequantizeIq2S(iqBytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Iq2SDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)iqBytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(iqBytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalSuperblocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        Iq2Fixture.AssertClose(expected, actual, $"iq2_s dequant superblocks={totalSuperblocks}",
            absTol: 0f, relTol: 0f);
    }
}
