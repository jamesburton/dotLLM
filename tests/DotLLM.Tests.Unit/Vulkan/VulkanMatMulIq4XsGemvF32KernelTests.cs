using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan IQ4_XS GEMV kernel against a scalar
/// CPU reference that reads the same 136-byte super-blocks the shader sees.
/// </summary>
/// <remarks>
/// Tolerance: abs 5e-3 / rel 2e-3 — IQ4_XS adds a 6-bit per-sub-block scale on
/// top of IQ4_NL's codebook, but each sub-block's drift is bounded by the same
/// codebook-rounding error so the overall tolerance is the same as IQ4_NL GEMV.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq4XsGemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 2e-3f;

    [SkippableTheory]
    [InlineData(1, 256)]                 // minimum: 1 super-block per row
    [InlineData(8, 256)]
    [InlineData(4, 512)]                 // 2 super-blocks per row
    [InlineData(16, 768)]                // 3 super-blocks per row
    [InlineData(2048, 768)]              // larger M
    [InlineData(576, 1024)]              // 4 super-blocks per row
    [InlineData(1024, 1024)]
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xC0DE_F4 + m * 7 + k * 11);
        float[] weightsF32 = Iq4Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq4Fixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsIq4 = Iq4Fixture.QuantizeRowsIq4Xs(weightsF32, m, k);
        Iq4Fixture.AssertFixtureRoundtripIq4Xs(weightsF32, weightsIq4, m, k);

        float[] expected = Iq4Fixture.CpuGemvIq4Xs(weightsIq4, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq4XsGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq4.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq4), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Iq4Fixture.AssertClose(expected, actual, $"iq4_xs GEMV m={m} k={k}", AbsTol, RelTol);
    }
}
