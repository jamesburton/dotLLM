using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan group RMS-norm kernel
/// (Mamba-2 / NemotronH SSM <c>ssm_norm</c>). The CPU reference is the
/// per-(t, g) call into <see cref="DotLLM.Cpu.Kernels.RmsNorm.Execute"/>
/// — bit-equivalent to the reference loop in
/// <c>NemotronHTransformerModel.ForwardSsmBody</c> step 11.
/// </summary>
/// <remarks>
/// Tolerance is abs 1e-4 / rel 1e-3 — the GPU's workgroup tree reduction
/// vs. the CPU's sequential (TensorPrimitives) accumulation drifts at the
/// last F32 mantissa bit. Each group is at most 80-wide here, well inside
/// noise for both reduction orders.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanGroupRmsNormF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;
    private const float Eps = 1e-5f;

    [SkippableTheory]
    [InlineData(1, 1, 16)]   // degenerate — single group, equivalent to standard RMSNorm
    [InlineData(1, 8, 64)]   // NemotronH-realistic decode (T=1)
    [InlineData(4, 8, 64)]   // multi-token prefill
    [InlineData(1, 2, 17)]   // odd groupDim — exercises non-power-of-2 reduce + bound check
    [InlineData(2, 10, 80)]  // NemotronH-typical
    public void Launch_MatchesCpuReference(int seqLen, int nGroup, int groupDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int dInner = nGroup * groupDim;
        var rng = new Random(0x6C0017 + seqLen * 31 + nGroup * 17 + groupDim * 11);

        float[] data = RandomFloats(rng, seqLen * dInner, range: 1.0f);
        // Real ssm_norm weights cluster around 1; shift the random into that range
        // so we exercise representative magnitudes (not micro-deltas around 0).
        float[] weight = RandomFloats(rng, dInner, range: 1.0f);
        for (int i = 0; i < dInner; i++) weight[i] = weight[i] * 0.5f + 1.0f;

        float[] expected = new float[seqLen * dInner];
        CpuReference(data, weight, expected, seqLen, nGroup, groupDim, Eps);

        using var device = VulkanDevice.Create();
        using var kernel = GroupRmsNormF32Kernel.Create(device, spvDir);

        using var bufData = device.Allocate((long)seqLen * dInner * sizeof(float));
        using var bufWeight = device.Allocate((long)dInner * sizeof(float));

        device.Upload(data, bufData);
        device.Upload(weight, bufWeight);

        kernel.Launch(bufData, bufWeight, seqLen, nGroup, groupDim, Eps);

        float[] actual = new float[seqLen * dInner];
        device.Download(bufData, actual);

        AssertClose(expected, actual, seqLen, nGroup, groupDim);
    }

    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// CPU reference — the per-(t, g) loop from
    /// <c>NemotronHTransformerModel.ForwardSsmBody</c> step 11. Calls
    /// the same <c>RmsNorm.Execute</c> the model uses, sliced by group.
    /// </summary>
    private static void CpuReference(
        float[] input, float[] weight, float[] output,
        int seqLen, int nGroup, int groupDim, float eps)
    {
        int dInner = nGroup * groupDim;
        for (int t = 0; t < seqLen; t++)
        {
            for (int g = 0; g < nGroup; g++)
            {
                int dataOff = t * dInner + g * groupDim;
                int weightOff = g * groupDim;
                ReadOnlySpan<float> inSlice = input.AsSpan(dataOff, groupDim);
                ReadOnlySpan<float> wSlice = weight.AsSpan(weightOff, groupDim);
                Span<float> outSlice = output.AsSpan(dataOff, groupDim);
                RmsNorm.Execute(inSlice, wSlice, eps, outSlice);
            }
        }
    }

    private static void AssertClose(float[] expected, float[] actual,
                                     int seqLen, int nGroup, int groupDim)
    {
        Assert.Equal(expected.Length, actual.Length);
        int dInner = nGroup * groupDim;
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
            int t = firstBadIdx / dInner;
            int g = (firstBadIdx % dInner) / groupDim;
            int i = (firstBadIdx % dInner) % groupDim;
            Assert.Fail(
                $"Numerical drift exceeded tolerance " +
                $"(seqLen={seqLen}, nGroup={nGroup}, groupDim={groupDim}): " +
                $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, " +
                $"first @ t={t} g={g} i={i}: cpu={expected[firstBadIdx]:G9} vs vulkan={actual[firstBadIdx]:G9}");
        }
    }
}
