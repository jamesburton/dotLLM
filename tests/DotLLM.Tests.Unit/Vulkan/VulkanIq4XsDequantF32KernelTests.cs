using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan IQ4_XS → F32 dequant kernel against the CPU
/// oracle <c>Dequantize.DequantizeIQ4_XS</c>. 0 ULP tolerance — single mul per
/// element, no reduction.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanIq4XsDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]            // single super-block (256 elements)
    [InlineData(4)]
    [InlineData(16)]
    [InlineData(128)]          // 32k elements, exercises grid-stride
    public void Launch_MatchesCpuOracle(int totalSuperblocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalSuperblocks * Iq4Fixture.Iq4XsGroupSize;
        var rng = new Random(0xA4_C0_FE ^ (totalSuperblocks * 11));
        float[] srcF32 = Iq4Fixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] iq4Bytes = Iq4Fixture.QuantizeRowsIq4Xs(srcF32, m: totalSuperblocks, k: Iq4Fixture.Iq4XsGroupSize);
        Iq4Fixture.AssertFixtureRoundtripIq4Xs(srcF32, iq4Bytes, m: totalSuperblocks, k: Iq4Fixture.Iq4XsGroupSize);

        float[] expected = Iq4Fixture.CpuDequantizeIq4Xs(iq4Bytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Iq4XsDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)iq4Bytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(iq4Bytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalSuperblocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        Iq4Fixture.AssertClose(expected, actual, $"iq4_xs dequant superblocks={totalSuperblocks}",
            absTol: 0f, relTol: 0f);
    }
}
