using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan IQ1_S batched GEMM kernel. Same shader
/// per-element decode as the GEMV kernel; the only difference is the 16x16
/// output-cell tile + N-batched B input. Tolerance shape mirrors GEMV (abs
/// 5e-2 / rel 1e-2) — IQ1_S codebook noise dominates over per-tile reorder.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq1SGemmF32KernelTests
{
    private const float AbsTol = 5e-2f;
    private const float RelTol = 1e-2f;

    [SkippableTheory]
    [InlineData(16, 256, 4)]                  // tile-aligned, 1 super-block per row, batch=4
    [InlineData(32, 512, 8)]                  // 2 super-blocks per row, batch=8
    [InlineData(64, 256, 16)]                 // 1 super-block, batch=16 (tile boundary)
    [InlineData(48, 768, 12)]                 // 3 super-blocks, partial-tile m=48 (3 tiles)
    [InlineData(15, 256, 3)]                  // m < TILE_M, n < TILE_N — exercises partial-tile fallthrough
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xC1_DE_F25 + m * 7 + k * 11 + n * 13);
        float[] weightsF32 = Iq1Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Iq1Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsIq1 = Iq1Fixture.QuantizeRowsIq1S(weightsF32, m, k);
        Iq1Fixture.AssertFixtureRoundtripIq1S(weightsF32, weightsIq1, m, k);

        float[] expected = Iq1Fixture.CpuGemmIq1S(weightsIq1, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq1SGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq1.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq1), bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        Iq1Fixture.AssertClose(expected, actual, $"iq1_s GEMM m={m} k={k} n={n}", AbsTol, RelTol);
    }
}
