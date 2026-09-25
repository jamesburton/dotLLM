using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity test for <c>gdn_post_scan_gate_f32</c> — the fused per-head
/// RMSNorm × silu(z) gate applied to the GDN scan output (step 6 of
/// <c>ForwardGdnBody</c>). Closes the coverage hole reported in issue #309.
/// </summary>
/// <remarks>
/// <para>
/// There is no single CPU kernel for this op — the reference is the inlined
/// two-step sequence in <c>Qwen3MoeHybridTransformerModel.ForwardGdnBody</c>
/// (RMSNorm in place against the shared <c>ssm_norm_weight</c>, then
/// <c>*= silu(z)</c>). It is reproduced here in scalar form rather than via
/// <c>RmsNorm.Execute</c> deliberately: that method reduces through
/// <c>TensorPrimitives.SumOfSquares</c> (SIMD, re-associated) and would put the
/// oracle on a different reduction path than the model actually documents.
/// </para>
/// <para>
/// Tolerance abs 1e-5 / rel 1e-4, matching the pointwise-plus-<c>exp</c>
/// siblings (<c>VulkanSiluInplaceF32KernelTests</c>): the reduction is over
/// ≤128 F32 terms (re-association error ~4e-7 relative) and the only
/// transcendental is one <c>exp</c> per element, spec'd at 4 ULP by GLSL.
/// </para>
/// <para>
/// Discriminating by construction: <c>ssm_norm_weight</c> is non-uniform (a
/// kernel that dropped it, or indexed it by the flat element instead of by the
/// <c>dState</c> position, fails), the <c>(t, vh)</c> slices carry distinct
/// magnitudes (a wrong base index fails), and <c>seqLen</c>, <c>nVHead</c> and
/// <c>dState</c> are pairwise distinct so a transposed dispatch mapping cannot
/// alias into the right answer.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGdnPostScanGateF32KernelTests
{
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    [SkippableTheory]
    [InlineData(1, 4, 16)]     // decode: single token
    [InlineData(3, 5, 32)]     // seqLen ≠ nVHead ≠ dState, all odd/prime-ish
    [InlineData(7, 6, 24)]     // dState not a power of two
    [InlineData(4, 32, 128)]   // Qwen3.6-A3B production shape
    public void Launch_MatchesCpuReference(int seqLen, int nVHead, int dState)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_post_scan_gate_f32.spv")),
            "gdn_post_scan_gate_f32.spv not compiled (glslc / Vulkan SDK required).");

        const float Eps = 1e-6f;
        var rng = new Random(0x9A7E ^ (seqLen * 1009) ^ (nVHead << 8) ^ (dState << 16));

        int total = seqLen * nVHead * dState;
        float[] gdnOut = new float[total];
        float[] z = new float[total];
        for (int t = 0; t < seqLen; t++)
        {
            for (int vh = 0; vh < nVHead; vh++)
            {
                // Distinct magnitude per (t, vh) so a wrong slice base is fatal.
                float scale = 0.25f + 0.5f * ((t * nVHead + vh) % 5);
                int b = (t * nVHead + vh) * dState;
                for (int i = 0; i < dState; i++)
                {
                    gdnOut[b + i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
                    z[b + i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 3.0);
                }
            }
        }

        // Non-uniform norm weight — a dropped or mis-indexed gain fails.
        float[] ssmNormWeight = new float[dState];
        for (int i = 0; i < dState; i++)
            ssmNormWeight[i] = 0.5f + 1.5f * (float)rng.NextDouble();

        float[] expected = CpuPostScanGate(gdnOut, z, ssmNormWeight, seqLen, nVHead, dState, Eps);

        using var device = VulkanDevice.Create();
        using var kernel = GdnPostScanGateF32Kernel.Create(device, spvDir);

        using var outBuf = device.Allocate((long)total * sizeof(float));
        using var zBuf = device.Allocate((long)total * sizeof(float));
        using var wBuf = device.Allocate((long)dState * sizeof(float));
        device.Upload(gdnOut.AsSpan(), outBuf);
        device.Upload(z.AsSpan(), zBuf);
        device.Upload(ssmNormWeight.AsSpan(), wBuf);

        kernel.Launch(outBuf, zBuf, wBuf, seqLen, nVHead, dState, Eps);

        float[] actual = new float[total];
        device.Download(outBuf, actual);

        AssertClose(expected, actual, $"seqLen={seqLen}, nVHead={nVHead}, dState={dState}");
    }

    /// <summary>
    /// Scalar reproduction of <c>ForwardGdnBody</c> step 6: per <c>(t, vh)</c>
    /// head slice, RMSNorm (<c>eps</c> INSIDE the sqrt, matching
    /// <c>RmsNorm.Execute</c>) against the shared <c>dState</c>-wide gain, then
    /// an element-wise multiply by <c>silu(z) = z·sigmoid(z)</c>.
    /// </summary>
    private static float[] CpuPostScanGate(
        float[] gdnOut, float[] z, float[] ssmNormWeight, int seqLen, int nVHead, int dState, float eps)
    {
        var result = new float[gdnOut.Length];
        for (int t = 0; t < seqLen; t++)
        {
            for (int vh = 0; vh < nVHead; vh++)
            {
                int b = (t * nVHead + vh) * dState;
                float sumSq = 0f;
                for (int i = 0; i < dState; i++)
                    sumSq += gdnOut[b + i] * gdnOut[b + i];
                float scale = 1f / MathF.Sqrt(sumSq / dState + eps);
                for (int i = 0; i < dState; i++)
                {
                    float normed = gdnOut[b + i] * scale * ssmNormWeight[i];
                    float zi = z[b + i];
                    result[b + i] = normed * (zi * (1f / (1f + MathF.Exp(-zi))));
                }
            }
        }
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
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
            $"GdnPostScanGate drift exceeded tolerance ({label}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
