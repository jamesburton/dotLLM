using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan IQ1_S GEMV kernel against a scalar CPU
/// reference that reads the same byte buffer the shader sees and uses the same
/// per-element decode (<c>dl * (grid[j] + delta)</c>).
/// </summary>
/// <remarks>
/// Tolerance: abs 5e-2 / rel 1e-2 — IQ1_S's 1.5-bpw codebook has much higher
/// per-element drift than IQ4 at the same K, plus the shader's workgroup tree
/// reduce reorders the per-block partial sums vs the CPU's strict block-
/// sequential accumulation. The combination of FP32 reordering plus the
/// inherent scale of IQ1_S values (dl-factor up to 15) makes any tighter
/// tolerance fragile across drivers.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq1SGemvF32KernelTests
{
    private const float AbsTol = 5e-2f;
    private const float RelTol = 1e-2f;

    [SkippableTheory]
    [InlineData(1, 256)]                 // minimum: 1 super-block per row
    [InlineData(8, 256)]                 // 8 rows, 1 super-block
    [InlineData(4, 512)]                 // 2 super-blocks per row
    [InlineData(16, 1024)]               // 4 super-blocks per row
    [InlineData(2048, 256)]              // large M, single super-block per row
    [InlineData(576, 768)]               // 3 super-blocks per row, SmolLM-like shape
    [InlineData(256, 2048)]              // square-ish, 8 super-blocks per row
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xB1_DE_F14 + m * 7 + k * 11);
        // Small range so the codebook quant noise is bounded — dl scales up to
        // d * 15, and we want output to land in normal FP32 range.
        float[] weightsF32 = Iq1Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq1Fixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsIq1 = Iq1Fixture.QuantizeRowsIq1S(weightsF32, m, k);
        Iq1Fixture.AssertFixtureRoundtripIq1S(weightsF32, weightsIq1, m, k);

        float[] expected = Iq1Fixture.CpuGemvIq1S(weightsIq1, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq1SGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsIq1.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq1), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Iq1Fixture.AssertClose(expected, actual, $"iq1_s GEMV m={m} k={k}", AbsTol, RelTol);
    }
}
