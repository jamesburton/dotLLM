using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan F16 native coopmat GEMM kernel.
/// </summary>
/// <remarks>
/// <para>
/// Self-skips on hosts that do not advertise <c>VK_KHR_cooperative_matrix</c>
/// — same gating pattern as <c>VulkanMatMulQ8_0GemmCoopmatKernelTests</c>.
/// On capable hardware (gfx1151 confirmed), validates the F16xF16->F32 tile
/// path against a scalar CPU GEMM reference that reads the same F16 bytes
/// the GPU shader sees.
/// </para>
/// <para>
/// Tolerance: abs 5e-3 / rel 1e-3 — F32 accumulator + F16 operand staging
/// matches what the scalar kernel already does internally for sharedW
/// dequant; drift versus the scalar GEMM is small. (B is staged through F16
/// in the coopmat path which adds the standard F32->F16 staging delta on the
/// activation side; the scalar GEMM keeps B in F32 throughout. At |B| ≈ 1
/// and K ≤ 1024 the per-output drift is ≤ 1e-3 absolute on a unit-scale
/// output.)
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulF16GemmCoopmatKernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(16, 32, 16)]                 // single coopmat tile, 1 K-chunk
    [InlineData(16, 64, 16)]                 // 2 K-chunks
    [InlineData(32, 64, 32)]                 // 2x2 tiles in M, 2x2 in N
    [InlineData(48, 128, 17)]                // partial-N tile
    [InlineData(64, 128, 16)]                // 4 M-tiles
    [InlineData(128, 256, 32)]
    [InlineData(256, 512, 16)]
    [InlineData(512, 1024, 8)]
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix,
            "Device does not advertise VK_KHR_cooperative_matrix.");

        var rng = new Random(0xF16C + m * 7 + k * 11 + n * 13);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = F16Bf16Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsF16 = F16Bf16Fixture.QuantizeRowsF16(weightsF32, m, k);

        float[] expected = F16Bf16Fixture.CpuGemmF16(weightsF16, inputB, m, k, n);

        using var kernel = MatMulF16GemmCoopmatKernel.Create(device, spvDir);

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
