using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan BF16 native scalar tiled GEMM kernel.
/// </summary>
/// <remarks>
/// <para>
/// Same validation strategy as the BF16 GEMV
/// (<see cref="VulkanMatMulBf16GemvF32KernelTests"/>) extended to the prefill
/// batched path. Tolerance abs 1e-2 / rel 5e-3 — looser than F16's bar
/// because BF16 has narrower mantissa; see the GEMV-test docstring for the
/// rationale.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulBf16GemmF32KernelTests
{
    private const float AbsTol = 1e-2f;
    private const float RelTol = 5e-3f;

    [SkippableTheory]
    [InlineData(1, 32, 1)]
    [InlineData(16, 32, 1)]
    [InlineData(16, 32, 16)]
    [InlineData(16, 64, 8)]
    [InlineData(8, 64, 16)]
    [InlineData(48, 128, 17)]
    [InlineData(128, 256, 32)]
    [InlineData(256, 512, 16)]
    [InlineData(512, 1024, 8)]
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBF1B + m * 7 + k * 11 + n * 13);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = F16Bf16Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsBf16 = F16Bf16Fixture.QuantizeRowsBf16(weightsF32, m, k);
        Assert.Equal((long)m * k * 2, weightsBf16.Length);

        float[] expected = F16Bf16Fixture.CpuGemmBf16(weightsBf16, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulBf16GemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsBf16.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsBf16), bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        F16Bf16Fixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
