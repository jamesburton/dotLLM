using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan IQ4_XS prefill-path GEMM kernel.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq4XsGemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 2e-3f;

    [SkippableTheory]
    [InlineData(16, 256, 4)]
    [InlineData(32, 256, 8)]
    [InlineData(17, 512, 5)]
    [InlineData(256, 256, 4)]
    [InlineData(64, 768, 32)]
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(unchecked((int)0xC0DE_F142) + m * 7 + k * 11 + n);
        float[] weightsF32 = Iq4Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Iq4Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsIq4 = Iq4Fixture.QuantizeRowsIq4Xs(weightsF32, m, k);
        Iq4Fixture.AssertFixtureRoundtripIq4Xs(weightsF32, weightsIq4, m, k);

        float[] expected = Iq4Fixture.CpuGemmIq4Xs(weightsIq4, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq4XsGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq4.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq4), bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        Iq4Fixture.AssertClose(expected, actual, $"iq4_xs GEMM m={m} k={k} n={n}", AbsTol, RelTol);
    }
}
