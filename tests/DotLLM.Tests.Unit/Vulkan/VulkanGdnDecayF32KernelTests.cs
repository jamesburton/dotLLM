using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan fused softplus + exp kernel used by
/// the Qwen3MoeHybrid GDN token-mixing path:
/// <c>g[t, vh] = exp(softplus(alpha[t, vh] + dt_bias[vh]) * A[vh])</c>.
/// </summary>
/// <remarks>
/// Mirrors the CPU oracle exactly — no x>20 numerical guard. Tolerance allows
/// up to 4 ULP transcendental drift (the standard Vulkan exp/log guarantee).
/// Larger absolute errors are tolerated on near-zero outputs (saturation
/// regime) where the relative tolerance is meaningless.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanGdnDecayF32KernelTests
{
    // Composition is `exp(log(1 + exp(a)) * A)` — chained transcendentals.
    // GLSL spec allows 4 ULP per `exp`/`log` call, so the worst-case error
    // for the composition is bounded between 4 and ~16 ULP depending on the
    // condition number at the evaluation point. Empirically across the
    // (seqLen=17, nVHead=32) sweep we see 5 ULP on a single element out of
    // 544 — the rest are within 4 ULP. An 8-ULP tolerance covers the
    // observed composition drift with a 1.6× safety margin without masking
    // any algorithmic divergence (the shader is element-wise scalar with no
    // reduction, so anything beyond transcendental ULP drift would indicate
    // a real bug).
    private const int MaxUlpDiff = 8;

    [SkippableTheory]
    [InlineData(1, 16)]
    [InlineData(8, 16)]
    [InlineData(64, 16)]
    [InlineData(17, 32)]  // odd seqLen × non-power-of-2 nVHead
    public void Launch_MatchesCpuReference(int seqLen, int nVHead)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_decay_f32.spv")),
            "gdn_decay_f32.spv not compiled (glslc / Vulkan SDK required).");

        var rng = new Random(0xDEC * (seqLen + nVHead));
        float[] alphaIn = RandomFloats(rng, seqLen * nVHead, lo: -4f, hi: 4f);
        float[] dtBias = RandomFloats(rng, nVHead, lo: -1f, hi: 1f);
        float[] aVec = RandomFloats(rng, nVHead, lo: -0.5f, hi: -0.05f);

        // CPU oracle — bit-for-bit identical to the host fallback that this
        // shader replaces.
        float[] expected = new float[alphaIn.Length];
        for (int t = 0; t < seqLen; t++)
        {
            int row = t * nVHead;
            for (int vh = 0; vh < nVHead; vh++)
            {
                float a = alphaIn[row + vh] + dtBias[vh];
                float sp = MathF.Log(1f + MathF.Exp(a));
                expected[row + vh] = MathF.Exp(sp * aVec[vh]);
            }
        }

        using var device = VulkanDevice.Create();
        using var kernel = GdnDecayF32Kernel.Create(device, spvDir);

        using var alphaBuf = device.Allocate((long)alphaIn.Length * sizeof(float));
        using var dtBuf = device.Allocate((long)dtBias.Length * sizeof(float));
        using var aBuf = device.Allocate((long)aVec.Length * sizeof(float));
        device.Upload(alphaIn.AsSpan(), alphaBuf);
        device.Upload(dtBias.AsSpan(), dtBuf);
        device.Upload(aVec.AsSpan(), aBuf);

        kernel.Launch(alphaBuf, dtBuf, aBuf, seqLen, nVHead);

        float[] actual = new float[alphaIn.Length];
        device.Download(alphaBuf, actual);

        AssertCloseUlp(expected, actual, $"seqLen={seqLen} nVHead={nVHead}");
    }

    /// <summary>
    /// Saturation sanity check: large positive <c>alpha + dt_bias</c> drives
    /// softplus into <c>+inf</c>; with negative <c>A</c> the final
    /// <c>exp(-inf)</c> must vanish to <c>0</c>. This guards the explicit
    /// "no x>20 guard" parity contract.
    /// </summary>
    [SkippableFact]
    public void Launch_SaturatesToZero_WhenAlphaIsHugeAndAIsNegative()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_decay_f32.spv")),
            "gdn_decay_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int seqLen = 4;
        const int nVHead = 8;
        float[] alpha = new float[seqLen * nVHead];
        Array.Fill(alpha, 200f);            // exp(200) → +inf
        float[] dt = new float[nVHead];
        float[] a = new float[nVHead];
        Array.Fill(a, -0.1f);               // exp(sp * a) with sp = +inf → exp(-inf) = 0

        using var device = VulkanDevice.Create();
        using var kernel = GdnDecayF32Kernel.Create(device, spvDir);

        using var alphaBuf = device.Allocate((long)alpha.Length * sizeof(float));
        using var dtBuf = device.Allocate((long)dt.Length * sizeof(float));
        using var aBuf = device.Allocate((long)a.Length * sizeof(float));
        device.Upload(alpha.AsSpan(), alphaBuf);
        device.Upload(dt.AsSpan(), dtBuf);
        device.Upload(a.AsSpan(), aBuf);

        kernel.Launch(alphaBuf, dtBuf, aBuf, seqLen, nVHead);

        float[] actual = new float[alpha.Length];
        device.Download(alphaBuf, actual);
        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(0.0f, actual[i]);
    }

    private static float[] RandomFloats(Random rng, int count, float lo, float hi)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(lo + rng.NextDouble() * (hi - lo));
        return arr;
    }

    private static unsafe int UlpDiff(float a, float b)
    {
        if (float.IsNaN(a) || float.IsNaN(b)) return int.MaxValue;
        if (a == b) return 0;
        int ai = *(int*)&a;
        int bi = *(int*)&b;
        // Two's-complement-flip the sign bit so adjacent representable floats
        // differ by exactly 1 regardless of sign.
        if (ai < 0) ai = unchecked((int)(0x80000000u - (uint)ai));
        if (bi < 0) bi = unchecked((int)(0x80000000u - (uint)bi));
        return Math.Abs(ai - bi);
    }

    private static void AssertCloseUlp(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int maxUlp = 0;
        int violations = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            int diff = UlpDiff(expected[i], actual[i]);
            // Saturation-regime outputs near 0 can have absolute differences
            // far larger than 4 ULP of the expected value (since denormals
            // span many ULPs between non-equal magnitudes). Accept those
            // when both expected and actual are effectively zero.
            float ae = MathF.Abs(expected[i]);
            if (ae < 1e-30f && MathF.Abs(actual[i]) < 1e-30f) diff = 0;
            if (diff > maxUlp) maxUlp = diff;
            if (diff > MaxUlpDiff) violations++;
        }
        Assert.True(violations == 0,
            $"GdnDecay drift exceeded {MaxUlpDiff} ULP ({label}): violations={violations}/{expected.Length}, maxUlp={maxUlp}");
    }
}
