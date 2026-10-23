using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq2SGemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 2e-3f;

    [SkippableTheory]
    [InlineData(1, 256)]
    [InlineData(8, 256)]
    [InlineData(4, 512)]
    [InlineData(64, 256)]
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x3A_BC_F1 + m * 7 + k * 11);
        float[] weightsF32 = Iq2Fixture.RandomFloats(rng, m * k, range: 0.05f);
        float[] x = Iq2Fixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsIq2 = Iq2Fixture.QuantizeRowsIq2S(weightsF32, m, k);
        Iq2Fixture.AssertFixtureRoundtripIq2S(weightsF32, weightsIq2, m, k);

        float[] expected = Iq2Fixture.CpuGemvIq2S(weightsIq2, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq2SGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq2.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq2), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Iq2Fixture.AssertClose(expected, actual, $"iq2_s GEMV m={m} k={k}", AbsTol, RelTol);
    }
}
