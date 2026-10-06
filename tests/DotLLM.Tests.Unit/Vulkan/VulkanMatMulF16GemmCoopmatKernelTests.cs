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

        RunAndAssert(device, spvDir, F16GemmCoopmatVariant.SelectFor(device, spvDir), m, k, n);
    }

    /// <summary>
    /// Issue #240: same parity gate against the explicit
    /// <see cref="F16GemmCoopmatVariant.Coopmat32"/> variant (wave32-pinned 32-thread
    /// workgroup). Skips separately from <see cref="Launch_MatchesCpuReference"/> when either
    /// the device cannot pin <c>requiredSubgroupSize=32</c> or (expected on this machine —
    /// the shader was authored without a Vulkan SDK/<c>glslc</c> to compile it)
    /// <c>matmul_f16_gemm_coopmat32.spv</c> is not yet present in <paramref name="spvDir"/>'s
    /// directory. Once compiled via <c>native/vulkan/build.ps1</c>/<c>build.sh</c>, this test
    /// starts exercising it on any host that supports the pin.
    /// </summary>
    [SkippableTheory]
    [InlineData(16, 32, 16)]
    [InlineData(64, 128, 16)]
    [InlineData(256, 512, 16)]
    public void Launch_Coopmat32MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        var variant = F16GemmCoopmatVariant.Coopmat32;
        Skip.IfNot(variant.IsSupportedOn(device, spvDir),
            $"{variant.SpvFileName} not available (device cannot pin requiredSubgroupSize=32, " +
            "or the shader has not been compiled yet via native/vulkan/build.ps1/build.sh).");

        RunAndAssert(device, spvDir, variant, m, k, n);
    }

    /// <summary>
    /// Issue #443: the same parity gate against the 128x128 blocked coopmat tile, with the
    /// <b>ragged</b> shapes added that the legacy theory never had.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The 16x16 theory above is ragged only in N (48x128x17). That is not enough for a
    /// 128x128 tile: the boundary path here scatters <i>four</i> subgroups' staged fragments
    /// and derives the owning subgroup from a flat index, so a shape ragged in M <b>and</b> N
    /// and smaller than one tile in both is what actually discriminates a correct scatter from
    /// one that writes only the <c>warp_c == 0</c> half — the exact bug #439 hit.
    /// </para>
    /// <para>
    /// 17x?x47, 33x?x33 and 15x?x3 mirror the ragged shapes the PQ2_0 and I2_S gates use, and
    /// they are why the PQ2_0 tile could ship. K stays a multiple of 32 (the kernel's KChunk).
    /// </para>
    /// <para>
    /// Held to the <b>same</b> tolerance as the legacy kernel, deliberately: the per-output
    /// reduction order is identical (BK=32 chunks, two TK=16 <c>coopMatMulAdd</c> into one F32
    /// accumulator), so a failure here is a bug, not a tolerance question.
    /// </para>
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Contraction dim; must be a multiple of 32.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(128, 256, 128)]   // exactly one blocked tile in both dims
    [InlineData(256, 512, 256)]   // 2x2 blocked tiles
    [InlineData(16, 32, 16)]      // one coopmat fragment: 1/64th of a tile, all-boundary
    [InlineData(48, 128, 17)]     // the legacy theory's ragged-N shape
    [InlineData(17, 256, 47)]     // ragged in BOTH dims
    [InlineData(33, 128, 33)]     // one element past a full fragment in both dims
    [InlineData(15, 128, 3)]      // below a single fragment in both dims
    [InlineData(129, 256, 129)]   // one element past a full BLOCKED tile in both dims
    [InlineData(512, 1024, 8)]    // tall and thin: N=8 is 6% of a 128-wide N tile
    public void Launch_Blocked128x128x4MatchesCpuReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "Device does not advertise VK_KHR_cooperative_matrix.");

        var variant = F16GemmCoopmatVariant.Blocked128x128x4;
        Skip.IfNot(variant.IsSupportedOn(device, spvDir),
            $"{variant.SpvFileName} not available (compile it via native/vulkan/build.ps1/build.sh).");

        RunAndAssert(device, spvDir, variant, m, k, n);
    }

    /// <summary>
    /// <see cref="F16GemmCoopmatVariant.SelectFor"/> must honour
    /// <see cref="F16GemmCoopmatVariant.LegacyEnvVar"/> — the escape hatch is what makes an
    /// end-to-end A/B on a real model possible, so it is gated rather than assumed.
    /// </summary>
    [SkippableFact]
    public void SelectFor_LegacyEnvVar_RestoresCoopmat64()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        string? saved = Environment.GetEnvironmentVariable(F16GemmCoopmatVariant.LegacyEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(F16GemmCoopmatVariant.LegacyEnvVar, "1");
            Assert.Equal(F16GemmCoopmatVariant.Coopmat64, F16GemmCoopmatVariant.SelectFor(device, spvDir));
        }
        finally
        {
            Environment.SetEnvironmentVariable(F16GemmCoopmatVariant.LegacyEnvVar, saved);
        }
    }

    private static void RunAndAssert(VulkanDevice device, string spvDir, F16GemmCoopmatVariant variant, int m, int k, int n)
    {
        var rng = new Random(0xF16C + m * 7 + k * 11 + n * 13);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = F16Bf16Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsF16 = F16Bf16Fixture.QuantizeRowsF16(weightsF32, m, k);

        float[] expected = F16Bf16Fixture.CpuGemmF16(weightsF16, inputB, m, k, n);

        using var kernel = MatMulF16GemmCoopmatKernel.Create(device, spvDir, variant);

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
