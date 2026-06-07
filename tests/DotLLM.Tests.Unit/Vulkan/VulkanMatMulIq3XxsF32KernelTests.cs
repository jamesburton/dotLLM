using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the IQ3_XXS GEMV + GEMM Vulkan kernels against the CPU
/// reference (dequant -> matmul). Tolerance is 5e-3 abs / 2e-3 rel — IQ3 sits
/// between IQ2 and IQ4 in bit-rate; this tracks the IQ2 tolerance roughly.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq3XxsF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 2e-3f;

    [SkippableTheory]
    [InlineData(4, 256)]
    [InlineData(8, 256)]
    public void Gemv_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x3B_CC_29 ^ (m * 13 + k));
        float[] xF32     = Iq3Fixture.RandomFloats(rng, k, 0.5f);
        float[] weightsF = Iq3Fixture.RandomFloats(rng, m * k, 0.05f);
        byte[] weightsIq = Iq3Fixture.QuantizeRowsIq3Xxs(weightsF, m, k);
        Iq3Fixture.AssertFixtureRoundtripIq3Xxs(weightsF, weightsIq, m, k);

        float[] expected = Iq3Fixture.CpuGemvIq3Xxs(weightsIq, xF32, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq3XxsGemvF32Kernel.Create(device, spvDir);

        long weightBytes = ((long)weightsIq.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq), bufW);
        device.Upload(xF32, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Iq3Fixture.AssertClose(expected, actual, $"iq3_xxs gemv m={m} k={k}",
            absTol: AbsTol, relTol: RelTol);
    }

    [SkippableTheory]
    [InlineData(4, 256, 2)]
    [InlineData(16, 256, 4)]
    public void Gemm_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x77_AC_4B ^ (m * 13 + k * 7 + n));
        float[] inputB    = Iq3Fixture.RandomFloats(rng, n * k, 0.5f);
        float[] weightsF  = Iq3Fixture.RandomFloats(rng, m * k, 0.05f);
        byte[] weightsIq  = Iq3Fixture.QuantizeRowsIq3Xxs(weightsF, m, k);
        Iq3Fixture.AssertFixtureRoundtripIq3Xxs(weightsF, weightsIq, m, k);

        float[] expected = Iq3Fixture.CpuGemmIq3Xxs(weightsIq, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulIq3XxsGemmF32Kernel.Create(device, spvDir);

        long weightBytes = ((long)weightsIq.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq), bufW);
        device.Upload(inputB, bufB);
        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        Iq3Fixture.AssertClose(expected, actual, $"iq3_xxs gemm m={m} k={k} n={n}",
            absTol: AbsTol, relTol: RelTol);
    }
}
