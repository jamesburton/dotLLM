using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q3_K GEMV kernel against a
/// byte-identical scalar CPU reference (<see cref="Q3KFixture.CpuGemvQ3K"/>).
/// Mirrors the Q2_K / Q4_K / Q5_K / Q6_K GEMV test shape.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ3KGemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 256)]
    [InlineData(8, 256)]
    [InlineData(4, 512)]
    [InlineData(16, 768)]
    [InlineData(2048, 768)]
    [InlineData(576, 1024)]
    [InlineData(1024, 1024)]
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xCAFE_3C ^ (m * 7 + k * 11));
        float[] weightsF32 = Q3KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q3KFixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsQ3K = Q3KFixture.QuantizeRows(weightsF32, m, k);
        Q3KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ3K, m, k);

        float[] expected = Q3KFixture.CpuGemvQ3K(weightsQ3K, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ3KGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsQ3K.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ3K), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Q3KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
