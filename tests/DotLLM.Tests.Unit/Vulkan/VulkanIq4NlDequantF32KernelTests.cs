using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan IQ4_NL → F32 dequant kernel against the CPU
/// oracle <c>Dequantize.DequantizeIQ4_NL</c>. Tolerance is 0 ULP — both paths
/// read the same bytes and emit the same FP32 product (no reduction, single
/// multiply per element).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanIq4NlDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]            // single block
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(1024)]         // 1024 blocks = 32k elements — exercises grid-stride
    public void Launch_MatchesCpuOracle(int totalBlocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalBlocks * Iq4Fixture.Iq4NlGroupSize;
        var rng = new Random(0xA4_BE_EF ^ (totalBlocks * 7));
        float[] srcF32 = Iq4Fixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] iq4Bytes = Iq4Fixture.QuantizeRowsIq4Nl(srcF32, m: totalBlocks, k: Iq4Fixture.Iq4NlGroupSize);
        Iq4Fixture.AssertFixtureRoundtripIq4Nl(srcF32, iq4Bytes, m: totalBlocks, k: Iq4Fixture.Iq4NlGroupSize);

        float[] expected = Iq4Fixture.CpuDequantizeIq4Nl(iq4Bytes, elements);

        using var device = VulkanDevice.Create();
        using var kernel = Iq4NlDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)iq4Bytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(iq4Bytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalBlocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        // Bit-parity: 0 ULP tolerance — the dequant is a single mul per element
        // and both the CPU oracle and the shader use FP16 → FP32 cast for d
        // followed by `d * float(kv[q])`, identical numeric pipeline.
        Iq4Fixture.AssertClose(expected, actual, $"iq4_nl dequant totalBlocks={totalBlocks}",
            absTol: 0f, relTol: 0f);
    }
}
