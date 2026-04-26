using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the MoE weighted-scatter Vulkan kernel —
/// the final combine step that collapses per-(token, slot) expert outputs
/// into per-token outputs via a weighted sum.
/// </summary>
/// <remarks>
/// The arithmetic is a small fixed-size weighted sum per output element
/// (typically topK = 2 or 8). Tolerance abs 1e-4 / rel 1e-3 — matches the
/// other F32 pointwise/reduce kernels.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeWeightedScatterF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 2, 16)]      // smallest: 1 token, k=2 (Mixtral 8/2)
    [InlineData(1, 8, 16)]      // k=8 (DeepSeek-V2 256/8 / Qwen3-MoE 128/8)
    [InlineData(8, 2, 64)]      // 8-token prefill, k=2
    [InlineData(4, 4, 128)]     // 4-token prefill, k=4
    [InlineData(1, 1, 32)]      // degenerate k=1 (passthrough * weight)
    [InlineData(3, 6, 17)]      // odd shapes
    public void Launch_MatchesCpuReference(int seqLen, int topK, int hiddenSize)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xC0FFEE + seqLen * 13 + topK * 7 + hiddenSize);
        float[] x = RandomFloats(rng, seqLen * topK * hiddenSize);
        float[] weights = RandomFloats(rng, seqLen * topK);

        float[] expected = CpuWeightedScatter(x, weights, seqLen, topK, hiddenSize);

        using var device = VulkanDevice.Create();
        using var kernel = MoeWeightedScatterF32Kernel.Create(device, spvDir);

        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufW = device.Allocate((long)weights.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(x, bufX);
        device.Upload(weights, bufW);

        kernel.Launch(bufX, bufW, bufOut, seqLen, topK, hiddenSize);

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
    public void Launch_TopKOne_PassesScaledRowThrough()
    {
        // With topK=1, output is just weights[t] * x[t, :]. Sanity-check the
        // shape contract (no accidental cross-row mixing).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 3, topK = 1, hiddenSize = 8;
        var rng = new Random(0xFEED);
        float[] x = RandomFloats(rng, seqLen * topK * hiddenSize);
        float[] weights = { 0.5f, -1.25f, 2.0f };

        var expected = new float[seqLen * hiddenSize];
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < hiddenSize; h++)
                expected[t * hiddenSize + h] = weights[t] * x[t * hiddenSize + h];

        using var device = VulkanDevice.Create();
        using var kernel = MoeWeightedScatterF32Kernel.Create(device, spvDir);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufW = device.Allocate((long)weights.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(x, bufX);
        device.Upload(weights, bufW);

        kernel.Launch(bufX, bufW, bufOut, seqLen, topK, hiddenSize);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= AbsTol + RelTol * MathF.Abs(expected[i]),
                $"i={i}: expected={expected[i]:F6} vs actual={actual[i]:F6}");
    }

    private static float[] CpuWeightedScatter(float[] x, float[] weights, int seqLen, int topK, int hidden)
    {
        var output = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++)
        {
            int rowBase = t * topK;
            int xBase = rowBase * hidden;
            for (int h = 0; h < hidden; h++)
            {
                float acc = 0f;
                for (int slot = 0; slot < topK; slot++)
                    acc += weights[rowBase + slot] * x[xBase + slot * hidden + h];
                output[t * hidden + h] = acc;
            }
        }
        return output;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
