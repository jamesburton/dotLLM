using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Mamba-3 QK-norm wrapper. The CPU
/// reference is <see cref="DotLLM.Cpu.Kernels.Mamba3QkNorm.Execute"/>, which
/// applies <see cref="DotLLM.Cpu.Kernels.RmsNorm.Execute"/> to each
/// <c>[d_state]</c> slice of a <c>[seqLen, n_group, d_state]</c> tensor with
/// a shared per-element weight of length <c>d_state</c>.
/// </summary>
/// <remarks>
/// <para>
/// The Vulkan wrapper (Option A) delegates straight into
/// <see cref="RmsNormF32Kernel"/> with <c>rowCount = seqLen * nGroup</c> and
/// <c>n = dState</c>, so the GPU side runs the same kernel that powers
/// every other RMSNorm in the model. The test therefore exercises the
/// wrapper-to-RmsNorm reshape (and the in-place buffer aliasing) — not a
/// new shader.
/// </para>
/// <para>
/// Tolerance is abs 1e-4 / rel 1e-3, matching the existing RMSNorm tests.
/// The GPU's workgroup tree reduction vs. the CPU's TensorPrimitives
/// accumulation drifts at the last F32 mantissa bit; per-slice widths up
/// to 128 here are well inside noise for both reduction orders.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMamba3QkNormF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;
    private const float Eps = 1e-5f;

    [SkippableTheory]
    [InlineData(1, 1, 8)]     // smallest — single slice
    [InlineData(1, 4, 64)]    // decode shape, multi-group
    [InlineData(4, 2, 64)]    // multi-token prefill
    [InlineData(3, 1, 128)]   // Mamba-3-realistic dState, SISO (n_group=1)
    [InlineData(2, 8, 17)]    // odd dState — exercises non-power-of-2 reduce + bound check
    public void Launch_MatchesCpuReference(int seqLen, int nGroup, int dState)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x3BC0 + seqLen * 31 + nGroup * 17 + dState * 11);
        int total = seqLen * nGroup * dState;

        float[] bc = RandomFloats(rng, total, range: 1.0f);
        // Real RMSNorm weights cluster around 1.0 — shift random samples into
        // that range so we exercise representative magnitudes (not micro-deltas
        // around zero).
        float[] weight = RandomFloats(rng, dState, range: 1.0f);
        for (int i = 0; i < dState; i++) weight[i] = weight[i] * 0.5f + 1.0f;

        // CPU reference — call into the actual CPU kernel we are mirroring.
        float[] expected = (float[])bc.Clone();
        Mamba3QkNorm.Execute(expected, weight, Eps, seqLen, nGroup, dState);

        using var device = VulkanDevice.Create();
        using var rmsNorm = RmsNormF32Kernel.Create(device, spvDir);
        var qkNorm = new Mamba3QkNormF32Kernel(rmsNorm);

        using var bufBc = device.Allocate((long)total * sizeof(float));
        using var bufWeight = device.Allocate((long)dState * sizeof(float));

        device.Upload(bc, bufBc);
        device.Upload(weight, bufWeight);

        qkNorm.Launch(bufBc, bufWeight, seqLen, nGroup, dState, Eps);

        float[] actual = new float[total];
        device.Download(bufBc, actual);

        AssertClose(expected, actual, seqLen, nGroup, dState);
    }

    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual,
                                     int seqLen, int nGroup, int dState)
    {
        Assert.Equal(expected.Length, actual.Length);
        int sliceStride = nGroup * dState;
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        int firstBadIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol)
            {
                if (firstBadIdx < 0) firstBadIdx = i;
                errors++;
            }
        }
        if (errors != 0)
        {
            int t = firstBadIdx / sliceStride;
            int g = (firstBadIdx % sliceStride) / dState;
            int i = (firstBadIdx % sliceStride) % dState;
            Assert.Fail(
                $"Numerical drift exceeded tolerance " +
                $"(seqLen={seqLen}, nGroup={nGroup}, dState={dState}): " +
                $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, " +
                $"first @ t={t} g={g} i={i}: cpu={expected[firstBadIdx]:G9} vs vulkan={actual[firstBadIdx]:G9}");
        }
    }
}
