using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan IQ1_S → F32 dequant kernel against the CPU
/// oracle <c>Dequantize.DequantizeIQ1_S</c>. Tolerance is 0 ULP — both paths
/// read the same bytes, expand the same 11-bit codebook index, and emit the
/// same FP32 product (no reduction; per-element <c>dl * (grid[j] + delta)</c>
/// is computed identically on CPU and GPU).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanIq1SDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]            // single super-block
    [InlineData(4)]
    [InlineData(16)]
    [InlineData(128)]          // 128 super-blocks = 32k elements — exercises grid-stride
    public void Launch_MatchesCpuOracle(int totalSuperblocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalSuperblocks * Iq1Fixture.Iq1SGroupSize;
        var rng = new Random(0xA1_BE_EF ^ (totalSuperblocks * 7));
        // Small range so the ad-hoc fixture fits the 1.5-bpw codebook.
        float[] srcF32 = Iq1Fixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] iq1Bytes = Iq1Fixture.QuantizeRowsIq1S(srcF32, m: totalSuperblocks, k: Iq1Fixture.Iq1SGroupSize);
        Iq1Fixture.AssertFixtureRoundtripIq1S(srcF32, iq1Bytes, m: totalSuperblocks, k: Iq1Fixture.Iq1SGroupSize);

        float[] expected = Iq1Fixture.CpuDequantizeIq1S(iq1Bytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Iq1SDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)iq1Bytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(iq1Bytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalSuperblocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        Iq1Fixture.AssertClose(expected, actual, $"iq1_s dequant totalSuperblocks={totalSuperblocks}",
            absTol: 0f, relTol: 0f);
    }
}
