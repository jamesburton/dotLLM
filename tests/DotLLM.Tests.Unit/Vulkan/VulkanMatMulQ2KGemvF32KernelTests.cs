using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q2_K GEMV kernel against a
/// byte-identical scalar CPU reference (<see cref="Q2KFixture.CpuGemvQ2K"/>).
/// Mirrors the Q4_K / Q5_K / Q6_K GEMV test shape — same set of (m, k)
/// combinations to exercise small / non-power-of-2 / large dispatch geometries.
/// </summary>
/// <remarks>
/// Tolerance — Q2_K's ~2.6 bits/element makes drift inherently larger than
/// Q4_K at the same K, but because we compare GPU-Q2_K against CPU-Q2_K (not
/// against the original FP32 weights) the only differences are reduction
/// order — scalar block-sequential CPU vs workgroup tree-reduce GPU. We use
/// the same 5e-3 / 1e-3 tolerances as the higher-K-quants for consistency.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ2KGemvF32KernelTests
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

        var rng = new Random(0xCAFE_2C ^ (m * 7 + k * 11));
        float[] weightsF32 = Q2KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q2KFixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsQ2K = Q2KFixture.QuantizeRows(weightsF32, m, k);
        Q2KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ2K, m, k);

        float[] expected = Q2KFixture.CpuGemvQ2K(weightsQ2K, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ2KGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsQ2K.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ2K), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Q2KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
