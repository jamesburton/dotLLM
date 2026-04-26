using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan depthwise causal 1-D conv kernel
/// (Mamba2 SSM post-projection conv used by Nemotron-H and Mamba-3).
/// </summary>
/// <remarks>
/// Reference is the scalar CPU implementation
/// (<see cref="DotLLM.Cpu.Kernels.Conv1dCausal"/>). Tolerance is
/// abs 1e-4 / rel 1e-3 — this is a per-cell <c>dConv</c>-tap reduction
/// (so ≤ 5 FMAs in any covered case), tight enough to catch a swapped
/// weight stride.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanConv1dCausalF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(4, 8, 1)]      // decode step (T=1)
    [InlineData(4, 8, 16)]     // small prefill
    [InlineData(4, 128, 4)]    // NemotronH-ish channel count
    [InlineData(3, 16, 8)]     // different dConv
    [InlineData(5, 17, 3)]     // odd channels — exercises stride past WG boundary
    public void Launch_MatchesCpuReference(int dConv, int channels, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xC0117D + dConv * 31 + channels * 17 + seqLen * 11);

        int inputRows = dConv - 1 + seqLen;
        float[] input = RandomFloats(rng, inputRows * channels);
        float[] weight = RandomFloats(rng, dConv * channels);
        float[] bias = RandomFloats(rng, channels);

        float[] expected = new float[seqLen * channels];
        Conv1dCausal.Execute(input, weight, bias, expected, dConv, channels, seqLen);

        using var device = VulkanDevice.Create();
        using var kernel = Conv1dCausalF32Kernel.Create(device, spvDir);

        using var bufIn = device.Allocate((long)input.Length * sizeof(float));
        using var bufWeight = device.Allocate((long)weight.Length * sizeof(float));
        using var bufBias = device.Allocate((long)bias.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(input, bufIn);
        device.Upload(weight, bufWeight);
        device.Upload(bias, bufBias);

        kernel.Launch(bufIn, bufWeight, bufBias, bufOut, dConv, channels, seqLen);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            int t = i / channels;
            int c = i % channels;
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"t={t}, c={c}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void Launch_DConvEqualsOne_DegeneratesToBiasedScale()
    {
        // Sanity check: dConv=1 ⇒ no temporal context, the kernel should
        // collapse to y[t, c] = bias[c] + input[t, c] * weight[c]. With
        // dConv=1 there is no conv_state, so input has shape [seqLen, channels].
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int dConv = 1;
        const int channels = 32;
        const int seqLen = 5;

        var rng = new Random(unchecked((int)0xC0117D01));
        float[] input = RandomFloats(rng, seqLen * channels);
        float[] weight = RandomFloats(rng, dConv * channels);
        float[] bias = RandomFloats(rng, channels);

        // Hand-rolled reference for dConv=1 — no inner sum, just bias + scale.
        float[] expected = new float[seqLen * channels];
        for (int t = 0; t < seqLen; t++)
            for (int c = 0; c < channels; c++)
                expected[t * channels + c] = bias[c] + input[t * channels + c] * weight[c];

        // Cross-check against the CPU reference itself for this degenerate case.
        float[] cpuRef = new float[seqLen * channels];
        Conv1dCausal.Execute(input, weight, bias, cpuRef, dConv, channels, seqLen);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], cpuRef[i]);

        using var device = VulkanDevice.Create();
        using var kernel = Conv1dCausalF32Kernel.Create(device, spvDir);

        using var bufIn = device.Allocate((long)input.Length * sizeof(float));
        using var bufWeight = device.Allocate((long)weight.Length * sizeof(float));
        using var bufBias = device.Allocate((long)bias.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(input, bufIn);
        device.Upload(weight, bufWeight);
        device.Upload(bias, bufBias);

        kernel.Launch(bufIn, bufWeight, bufBias, bufOut, dConv, channels, seqLen);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"i={i}: expected={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
