using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the fused MoE sigmoid-gated add kernel.
/// </summary>
/// <remarks>
/// Reference: scalar CPU loop computing
/// <c>out[t, h] += sigmoid(gateLogits[t]) * b[t, h]</c>. Tolerance abs 1e-4 /
/// rel 1e-3 — the only F32 deviation between CPU and GPU is the order of
/// the FMA, which adds at most a few ULPs at these shapes.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeSigmoidGatedAddF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 16)]    // smallest: 1 token
    [InlineData(3, 16)]    // 3-token prefill
    [InlineData(8, 64)]    // batched, larger hidden
    [InlineData(1, 128)]   // single token, Qwen1.5-MoE-ish hidden
    [InlineData(5, 17)]    // odd shapes — ensures stride loop bounds correctly
    public void Launch_MatchesCpuReference(int seqLen, int hiddenSize)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(unchecked((int)0xCAFEFACE) + seqLen * 13 + hiddenSize * 7);
        float[] outBuf = RandomFloats(rng, seqLen * hiddenSize);
        float[] b = RandomFloats(rng, seqLen * hiddenSize);
        float[] gateLogits = RandomFloats(rng, seqLen);

        float[] expected = (float[])outBuf.Clone();
        for (int t = 0; t < seqLen; t++)
        {
            float scale = 1.0f / (1.0f + MathF.Exp(-gateLogits[t]));
            for (int h = 0; h < hiddenSize; h++)
                expected[t * hiddenSize + h] += scale * b[t * hiddenSize + h];
        }

        using var device = VulkanDevice.Create();
        using var kernel = MoeSigmoidGatedAddF32Kernel.Create(device, spvDir);

        using var bufOut = device.Allocate((long)outBuf.Length * sizeof(float));
        using var bufB = device.Allocate((long)b.Length * sizeof(float));
        using var bufLogit = device.Allocate((long)gateLogits.Length * sizeof(float));

        device.Upload(outBuf, bufOut);
        device.Upload(b, bufB);
        device.Upload(gateLogits, bufLogit);

        kernel.Launch(bufOut, bufB, bufLogit, seqLen, hiddenSize);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"t={i / hiddenSize}, h={i % hiddenSize}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void Launch_ZeroLogits_AddsHalfOfB()
    {
        // sigmoid(0) = 0.5 exactly — sanity check the math isn't flipped.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 2, hidden = 4;
        float[] outBuf = { 1f, 2f, 3f, 4f, 10f, 20f, 30f, 40f };
        float[] b = { 2f, 4f, 6f, 8f, 100f, 200f, 300f, 400f };
        float[] gateLogits = { 0f, 0f };

        float[] expected = new float[seqLen * hidden];
        for (int i = 0; i < expected.Length; i++) expected[i] = outBuf[i] + 0.5f * b[i];

        using var device = VulkanDevice.Create();
        using var kernel = MoeSigmoidGatedAddF32Kernel.Create(device, spvDir);
        using var bufOut = device.Allocate(outBuf.Length * sizeof(float));
        using var bufB = device.Allocate(b.Length * sizeof(float));
        using var bufLogit = device.Allocate(gateLogits.Length * sizeof(float));
        device.Upload(outBuf, bufOut);
        device.Upload(b, bufB);
        device.Upload(gateLogits, bufLogit);
        kernel.Launch(bufOut, bufB, bufLogit, seqLen, hidden);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 5);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
