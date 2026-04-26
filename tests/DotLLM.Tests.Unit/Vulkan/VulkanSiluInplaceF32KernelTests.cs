using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan in-place FP32 SiLU kernel.
/// </summary>
/// <remarks>
/// Pointwise — compared against the CPU reference <see cref="SiLu.Execute"/>.
/// Pointwise has tight tolerance (abs 1e-5 / rel 1e-4); GPU <c>exp</c> is
/// IEEE-correct on these magnitudes and no reduction is involved.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSiluInplaceF32KernelTests
{
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(17)]            // odd, less than workgroup
    [InlineData(64)]
    [InlineData(257)]           // odd, just over workgroup
    [InlineData(1024)]          // exact multiple of workgroup
    public void Launch_MatchesCpuReference(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x511 + n);
        float[] input = RandomFloats(rng, n, range: 3.0f);
        float[] expected = new float[n];
        SiLu.Execute(input, expected);

        using var device = VulkanDevice.Create();
        using var kernel = SiluInplaceF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), bufX);

        kernel.Launch(bufX, n);

        float[] actual = new float[n];
        device.Download(bufX, actual);

        AssertClose(expected, actual, n);
    }

    /// <summary>
    /// Sanity check: <c>silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0</c>.
    /// Catches obvious wiring bugs (e.g. binding the wrong buffer) without
    /// relying on the random parity sweep.
    /// </summary>
    [SkippableFact]
    public void Launch_SiluOfZeroIsZero()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 64;
        float[] input = new float[n]; // all zeros

        using var device = VulkanDevice.Create();
        using var kernel = SiluInplaceF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), bufX);

        kernel.Launch(bufX, n);

        float[] actual = new float[n];
        device.Download(bufX, actual);

        for (int i = 0; i < n; i++)
            Assert.Equal(0.0f, actual[i]);
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, int n)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"SiLU drift exceeded tolerance (n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
