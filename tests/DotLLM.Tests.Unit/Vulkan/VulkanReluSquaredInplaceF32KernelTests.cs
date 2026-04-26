using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan in-place FP32 squared-ReLU kernel.
/// </summary>
/// <remarks>
/// Pointwise — compared against the CPU reference <see cref="ReluSquared.Execute"/>.
/// Tight tolerance (abs 1e-5 / rel 1e-4) since the op is a max + multiply
/// with no transcendentals or reductions.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanReluSquaredInplaceF32KernelTests
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

        var rng = new Random(0x5132 + n);
        float[] input = RandomFloats(rng, n, range: 3.0f);
        float[] expected = new float[n];
        ReluSquared.Execute(input, expected);

        using var device = VulkanDevice.Create();
        using var kernel = ReluSquaredInplaceF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), bufX);

        kernel.Launch(bufX, n);

        float[] actual = new float[n];
        device.Download(bufX, actual);

        AssertClose(expected, actual, n);
    }

    /// <summary>
    /// Sanity check: <c>relu²(0) = max(0, 0)² = 0</c>.
    /// </summary>
    [SkippableFact]
    public void Launch_ReluSquaredOfZeroIsZero()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 64;
        float[] input = new float[n]; // all zeros

        using var device = VulkanDevice.Create();
        using var kernel = ReluSquaredInplaceF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), bufX);

        kernel.Launch(bufX, n);

        float[] actual = new float[n];
        device.Download(bufX, actual);

        for (int i = 0; i < n; i++)
            Assert.Equal(0.0f, actual[i]);
    }

    /// <summary>
    /// Sanity check: <c>relu²(-5) = max(0, -5)² = 0</c>. Catches missing
    /// zero-clamp on negative inputs (e.g. doing <c>x*x</c> only).
    /// </summary>
    [SkippableFact]
    public void Launch_ReluSquaredOfNegativeIsZero()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 64;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = -5.0f;

        using var device = VulkanDevice.Create();
        using var kernel = ReluSquaredInplaceF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), bufX);

        kernel.Launch(bufX, n);

        float[] actual = new float[n];
        device.Download(bufX, actual);

        for (int i = 0; i < n; i++)
            Assert.Equal(0.0f, actual[i]);
    }

    /// <summary>
    /// Sanity check: <c>relu²(2) = max(0, 2)² = 4</c>. Confirms the square
    /// is applied (not just a passthrough).
    /// </summary>
    [SkippableFact]
    public void Launch_ReluSquaredOfTwoIsFour()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 64;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = 2.0f;

        using var device = VulkanDevice.Create();
        using var kernel = ReluSquaredInplaceF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), bufX);

        kernel.Launch(bufX, n);

        float[] actual = new float[n];
        device.Download(bufX, actual);

        for (int i = 0; i < n; i++)
            Assert.Equal(4.0f, actual[i]);
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
            $"ReluSquared drift exceeded tolerance (n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
