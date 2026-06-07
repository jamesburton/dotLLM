using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq2XsGemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 2e-3f;

    [SkippableTheory]
    [InlineData(16, 256, 4)]
    [InlineData(32, 256, 8)]
    [InlineData(17, 512, 5)]
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(unchecked((int)0xC0DE_F212) + m * 7 + k * 11 + n);
        float[] weightsF32 = Iq2Fixture.RandomFloats(rng, m * k, range: 0.05f);
        float[] inputB = Iq2Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsIq2 = Iq2Fixture.QuantizeRowsIq2Xs(weightsF32, m, k);
        Iq2Fixture.AssertFixtureRoundtripIq2Xs(weightsF32, weightsIq2, m, k);

        float[] expected = Iq2Fixture.CpuGemmIq2Xs(weightsIq2, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq2XsGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq2.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq2), bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        Iq2Fixture.AssertClose(expected, actual, $"iq2_xs GEMM m={m} k={k} n={n}", AbsTol, RelTol);
    }
}
