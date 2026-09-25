using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan F16 native scalar tiled GEMM kernel.
/// </summary>
/// <remarks>
/// <para>
/// Same validation strategy as the F16 GEMV (<see cref="VulkanMatMulF16GemvF32KernelTests"/>)
/// extended to the prefill batched path: random FP32 weights and inputs,
/// quantise weights to F16, run a scalar CPU GEMM reference that reads the
/// same F16 bytes the GPU shader sees, compare against the GPU output.
/// </para>
/// <para>
/// Tolerance: abs 5e-3 / rel 1e-3 — same envelope as the F16 GEMV / K-quant
/// kernel parity tests. The 16x16 output tile and 32-element K-chunk match
/// the Q4_K / Q5_K / Q6_K GEMM precedents so the dispatch geometry is
/// identical and the test shape coverage mirrors those siblings.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulF16GemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 32, 1)]                   // minimum: 1 row, 1 chunk, 1 token
    [InlineData(16, 32, 1)]                  // full M tile, single token
    [InlineData(16, 32, 16)]                 // full tile both ways
    [InlineData(16, 64, 8)]                  // 2 K-chunks
    [InlineData(8, 64, 16)]                  // partial M
    [InlineData(48, 128, 17)]                // partial N (one token over 16-tile boundary)
    [InlineData(128, 256, 32)]               // 8 K-chunks
    [InlineData(256, 512, 16)]               // larger output
    [InlineData(512, 1024, 8)]               // square-ish, decode batch
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xF16B + m * 7 + k * 11 + n * 13);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = F16Bf16Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsF16 = F16Bf16Fixture.QuantizeRowsF16(weightsF32, m, k);
        Assert.Equal((long)m * k * 2, weightsF16.Length);

        float[] expected = F16Bf16Fixture.CpuGemmF16(weightsF16, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulF16GemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsF16.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsF16), bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        F16Bf16Fixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
