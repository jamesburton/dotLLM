using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan F16 native GEMV kernel.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy: generate random FP32 weights, quantise them to F16
/// via <see cref="F16Bf16Fixture.QuantizeRowsF16"/> (round-to-nearest-even
/// via <see cref="Half"/>). Reference result is from a scalar CPU GEMV that
/// reads the same F16 bytes the GPU shader sees and casts to FP32 on the fly.
/// Comparing F16-GPU against an F16-byte-identical CPU reference (rather
/// than against the original FP32 weights) catches kernel bugs in the
/// <c>unpackHalf2x16</c> read path / per-row stride computation that a
/// quantise-then-compare-to-FP32 reference would mask.
/// </para>
/// <para>
/// Tolerance: abs 5e-3 / rel 1e-3 — F16 has ~10-bit mantissa, drift is
/// dominated by the F16 round-trip noise and the GPU workgroup tree reduce
/// vs. CPU block-sequential reduction order. Same envelope as the K-quant
/// kernel parity tests.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulF16GemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 32)]                     // minimum K = K_CHUNK
    [InlineData(8, 64)]                     // 8 rows, 64 K
    [InlineData(4, 128)]
    [InlineData(16, 256)]
    [InlineData(2048, 512)]                 // larger M, exercises workgroup-per-row dispatch
    [InlineData(576, 768)]                  // non-power-of-2 K
    [InlineData(1024, 1024)]                // square shape, common attention dim
    [InlineData(4096, 4096)]                // production attention shape
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xF16A + m * 7 + k * 11);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsF16 = F16Bf16Fixture.QuantizeRowsF16(weightsF32, m, k);
        Assert.Equal((long)m * k * 2, weightsF16.Length);

        float[] expected = F16Bf16Fixture.CpuGemvF16(weightsF16, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulF16GemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsF16.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsF16), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        F16Bf16Fixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
