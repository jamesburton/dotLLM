using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for <c>sigmoid_gate_mul_f32</c> —
/// <c>attnOut[i] *= sigmoid(gate[i])</c>, step 8 of the Qwen3MoeHybrid /
/// Qwen3HybridDense full-attention forward pass. Closes the coverage hole
/// reported in issue #309.
/// </summary>
/// <remarks>
/// <para>
/// There is no CPU kernel class for this op (and no CUDA counterpart) — the
/// reference is the inlined loop in
/// <c>Qwen3MoeHybridTransformerModel.ForwardAttnBody</c>,
/// <c>aRow[i] *= 1f / (1f + MathF.Exp(-gRow[i]))</c>, reproduced verbatim here
/// in the same operation order.
/// </para>
/// <para>
/// Tolerance abs 1e-5 / rel 1e-4, matching
/// <see cref="VulkanSigmoidInplaceF32KernelTests"/>: pointwise, no reduction,
/// one <c>exp</c> and one reciprocal per element.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanSigmoidGateMulF32KernelTests
{
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(17)]     // odd, below the 256 workgroup
    [InlineData(255)]    // not a multiple of the workgroup
    [InlineData(256)]    // exact workgroup
    [InlineData(257)]    // just over
    [InlineData(4096)]   // seqLen × nQHead × headDim scale
    public void Launch_MatchesCpuReference(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "sigmoid_gate_mul_f32.spv")),
            "sigmoid_gate_mul_f32.spv not compiled (glslc / Vulkan SDK required).");

        var rng = new Random(0x5160 + n);
        float[] attnOut = RandomFloats(rng, n, range: 2.0f);
        float[] gate = RandomFloats(rng, n, range: 6.0f);   // spans both sigmoid tails

        float[] expected = new float[n];
        for (int i = 0; i < n; i++)
            expected[i] = attnOut[i] * (1f / (1f + MathF.Exp(-gate[i])));

        using var device = VulkanDevice.Create();
        using var kernel = SigmoidGateMulF32Kernel.Create(device, spvDir);

        using var outBuf = device.Allocate((long)n * sizeof(float));
        using var gateBuf = device.Allocate((long)n * sizeof(float));
        device.Upload(attnOut.AsSpan(), outBuf);
        device.Upload(gate.AsSpan(), gateBuf);

        kernel.Launch(outBuf, gateBuf, n);

        float[] actual = new float[n];
        device.Download(outBuf, actual);

        AssertClose(expected, actual, n);
    }

    /// <summary>
    /// Sanity + asymmetry check: with an all-zero gate the result must be
    /// exactly half the input (<c>sigmoid(0) = 0.5</c>, an exact F32 value), and
    /// the gate buffer itself must be left untouched — it is bound read-only,
    /// so a kernel that wrote the product into the wrong binding fails here.
    /// </summary>
    [SkippableFact]
    public void Launch_ZeroGateHalvesInput_AndLeavesGateUntouched()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "sigmoid_gate_mul_f32.spv")),
            "sigmoid_gate_mul_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int n = 300;
        var rng = new Random(0x0FF0);
        float[] attnOut = RandomFloats(rng, n, range: 2.0f);
        float[] gate = new float[n]; // all zeros

        using var device = VulkanDevice.Create();
        using var kernel = SigmoidGateMulF32Kernel.Create(device, spvDir);
        using var outBuf = device.Allocate((long)n * sizeof(float));
        using var gateBuf = device.Allocate((long)n * sizeof(float));
        device.Upload(attnOut.AsSpan(), outBuf);
        device.Upload(gate.AsSpan(), gateBuf);

        kernel.Launch(outBuf, gateBuf, n);

        float[] actual = new float[n];
        float[] gateAfter = new float[n];
        device.Download(outBuf, actual);
        device.Download(gateBuf, gateAfter);

        for (int i = 0; i < n; i++)
        {
            Assert.True((attnOut[i] * 0.5f).Equals(actual[i]),
                $"sigmoid(0) gate at {i}: expected {attnOut[i] * 0.5f:G9}, got {actual[i]:G9}.");
            Assert.True(gateAfter[i].Equals(0f), $"Gate buffer was written at {i}: {gateAfter[i]:G9}.");
        }
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
            $"SigmoidGateMul drift exceeded tolerance (n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
