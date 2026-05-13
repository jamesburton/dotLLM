using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan IQ4_NL GEMV kernel against a scalar
/// CPU reference that reads the same byte buffer the shader sees and uses the
/// same per-element decode (<c>d * kvalues_iq4nl[q]</c>).
/// </summary>
/// <remarks>
/// Tolerance: abs 5e-3 / rel 2e-3 — IQ4_NL's non-linear codebook produces
/// slightly higher per-element drift than Q4_K at the same K, plus the shader
/// uses a workgroup tree reduce vs the CPU's block-sequential sum (small
/// reordering noise on top of the codebook quantisation noise).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq4NlGemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 2e-3f;

    [SkippableTheory]
    [InlineData(1, 32)]                  // minimum: 1 block per row
    [InlineData(8, 32)]                  // 8 rows, 1 block
    [InlineData(4, 64)]                  // 2 blocks per row
    [InlineData(16, 128)]                // 4 blocks per row
    [InlineData(2048, 96)]               // large M (workgroup-per-row dispatch)
    [InlineData(576, 256)]               // 8 blocks per row, SmolLM-like shape
    [InlineData(1024, 1024)]             // square, 32 blocks per row
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xC0DE_F14 + m * 7 + k * 11);
        float[] weightsF32 = Iq4Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq4Fixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsIq4 = Iq4Fixture.QuantizeRowsIq4Nl(weightsF32, m, k);
        Iq4Fixture.AssertFixtureRoundtripIq4Nl(weightsF32, weightsIq4, m, k);

        float[] expected = Iq4Fixture.CpuGemvIq4Nl(weightsIq4, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq4NlGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq4.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq4), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Iq4Fixture.AssertClose(expected, actual, $"iq4_nl GEMV m={m} k={k}", AbsTol, RelTol);
    }
}
