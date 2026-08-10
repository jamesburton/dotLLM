using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for <c>geglu_tanh_f32</c> — the Gemma 2 / Gemma 3 MLP
/// activation <c>result[i] = gelu_tanh(gate[i]) · up[i]</c>. Closes the coverage
/// hole reported in issue #309.
/// </summary>
/// <remarks>
/// Compared against <c>FusedOps.GeGLUTanhScalar</c>, the scalar CPU reference
/// (not the SIMD <c>GeGLUTanh</c>), so GPU drift is not hidden behind the
/// vectorised path — the same choice
/// <see cref="VulkanReLU2GluF32KernelTests"/> makes. The shader's op order is
/// term-for-term identical to the oracle, so the only admissible divergence is
/// GLSL <c>tanh</c>'s spec tolerance; abs 1e-4 / rel 1e-3 matches the sibling.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGeGluTanhF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(255)]    // not a multiple of the 256 workgroup
    [InlineData(256)]    // exact workgroup
    [InlineData(257)]    // just over
    [InlineData(16384)]  // Gemma-3 style intermediate width
    public void Launch_MatchesCpuReference(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "geglu_tanh_f32.spv")),
            "geglu_tanh_f32.spv not compiled (glslc / Vulkan SDK required).");

        var rng = new Random(0x6E10 + n);
        float[] gate = RandomFloats(rng, n, range: 4.0f);  // spans both GELU tails
        float[] up = RandomFloats(rng, n, range: 3.0f);
        float[] expected = new float[n];
        FusedOps.GeGLUTanhScalar(gate, up, expected);

        using var device = VulkanDevice.Create();
        using var kernel = GeGluTanhF32Kernel.Create(device, spvDir);

        using var bufGate = device.Allocate((long)n * sizeof(float));
        using var bufUp = device.Allocate((long)n * sizeof(float));
        using var bufOut = device.Allocate((long)n * sizeof(float));
        device.Upload(gate.AsSpan(), bufGate);
        device.Upload(up.AsSpan(), bufUp);

        kernel.Launch(bufGate, bufUp, bufOut, n);

        float[] actual = new float[n];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, n);
    }

    /// <summary>
    /// Argument-order discriminator: GeGLU is NOT symmetric in its two inputs
    /// (<c>gelu(g)·u ≠ gelu(u)·g</c>), so swapping the gate and up bindings must
    /// change the result. Guards against a wiring bug the random sweep would
    /// only catch by luck of the oracle also being swapped.
    /// </summary>
    [SkippableFact]
    public void Launch_IsNotSymmetricInGateAndUp()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "geglu_tanh_f32.spv")),
            "geglu_tanh_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int n = 512;
        var rng = new Random(0x5AA5);
        float[] gate = RandomFloats(rng, n, range: 4.0f);
        float[] up = RandomFloats(rng, n, range: 3.0f);

        using var device = VulkanDevice.Create();
        using var kernel = GeGluTanhF32Kernel.Create(device, spvDir);
        using var bufA = device.Allocate((long)n * sizeof(float));
        using var bufB = device.Allocate((long)n * sizeof(float));
        using var bufOut = device.Allocate((long)n * sizeof(float));
        device.Upload(gate.AsSpan(), bufA);
        device.Upload(up.AsSpan(), bufB);

        float[] forward = new float[n];
        kernel.Launch(bufA, bufB, bufOut, n);
        device.Download(bufOut, forward);

        float[] swapped = new float[n];
        kernel.Launch(bufB, bufA, bufOut, n);
        device.Download(bufOut, swapped);

        int differing = 0;
        for (int i = 0; i < n; i++)
            if (!forward[i].Equals(swapped[i])) differing++;
        Assert.True(differing > n / 2,
            $"gate/up swap changed only {differing}/{n} outputs — the kernel is not " +
            "distinguishing its two inputs.");
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
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"GeGLU-tanh drift exceeded tolerance (n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
