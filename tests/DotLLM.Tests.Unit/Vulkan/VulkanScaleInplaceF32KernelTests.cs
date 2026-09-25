using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>scale_inplace_f32</c> — the in-place scalar multiply used
/// for Gemma's <c>sqrt(hidden)</c> embedding scale and the per-layer output
/// scale. Closes the coverage hole reported in issue #309.
/// </summary>
/// <remarks>
/// A single F32 multiply is exactly rounded on both sides, so the bar here is
/// <b>bit-exact</b> rather than a tolerance — stricter than the pointwise
/// siblings, which only need slack because they involve transcendentals. Any
/// non-zero drift would mean the shader is not doing a plain <c>x*scale</c>.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanScaleInplaceF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 2.5f)]
    [InlineData(16, 0.125f)]
    [InlineData(255, -3.75f)]        // not a multiple of the 256 workgroup; negative scale
    [InlineData(256, 1.0f)]          // exact workgroup; identity scale
    [InlineData(257, 45.254834f)]    // just over; ≈ sqrt(2048), the Gemma embedding scale
    [InlineData(4096, 0.0f)]         // zero scale must zero every slot
    [InlineData(11008, 1e-7f)]       // tiny scale — catches a dropped/defaulted push constant
    public void Launch_MatchesCpuReference_BitExact(int n, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "scale_inplace_f32.spv")),
            "scale_inplace_f32.spv not compiled (glslc / Vulkan SDK required).");

        var rng = new Random(0x5CA1 + n + (int)(scale * 97));
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 3.0);

        float[] expected = new float[n];
        for (int i = 0; i < n; i++)
            expected[i] = input[i] * scale;

        using var device = VulkanDevice.Create();
        using var kernel = ScaleInplaceF32Kernel.Create(device, spvDir);
        using var buf = device.Allocate((long)n * sizeof(float));
        device.Upload(input.AsSpan(), buf);

        kernel.Launch(buf, n, scale);

        float[] actual = new float[n];
        device.Download(buf, actual);

        for (int i = 0; i < n; i++)
            Assert.True(
                BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"scale_inplace mismatch at {i} (n={n}, scale={scale:G9}): " +
                $"cpu={expected[i]:G9} gpu={actual[i]:G9} (in={input[i]:G9}).");
    }

    /// <summary>
    /// Tail discriminator: elements past <paramref name="n"/> must be left
    /// untouched. The shader guards with <c>if (i &gt;= pc.n) return;</c>; a
    /// missing guard would scale into the neighbouring allocation, which the
    /// exact-length sweep above cannot see.
    /// </summary>
    [SkippableFact]
    public void Launch_LeavesTailBeyondNUntouched()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "scale_inplace_f32.spv")),
            "scale_inplace_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int allocated = 512;
        const int n = 300;            // leaves 212 elements inside the same workgroup span
        const float scale = 7.0f;

        float[] input = new float[allocated];
        for (int i = 0; i < allocated; i++) input[i] = i + 1;

        using var device = VulkanDevice.Create();
        using var kernel = ScaleInplaceF32Kernel.Create(device, spvDir);
        using var buf = device.Allocate((long)allocated * sizeof(float));
        device.Upload(input.AsSpan(), buf);

        kernel.Launch(buf, n, scale);

        float[] actual = new float[allocated];
        device.Download(buf, actual);

        for (int i = 0; i < n; i++)
            Assert.Equal(input[i] * scale, actual[i]);
        for (int i = n; i < allocated; i++)
            Assert.True(input[i].Equals(actual[i]),
                $"Element {i} beyond n={n} was modified: {input[i]:G9} -> {actual[i]:G9}.");
    }
}
