using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan MoE router top-k softmax kernel.
/// </summary>
/// <remarks>
/// Reference is a self-contained scalar CPU implementation matching the
/// CPU MoE routing path: full softmax over expert logits, sequential top-k
/// argmax (lower index wins on ties), optional renormalisation. Validates
/// indices match exactly (deterministic) and weights match within F32
/// noise. Covers Mixtral / Qwen-MoE / Phi-3.5-MoE / DeepSeek-V2-MoE
/// configurations.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeTopKSoftmaxF32KernelTests
{
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 8, 2, true)]                  // Mixtral / Phi-3.5-MoE: 8 experts top-2
    [InlineData(1, 8, 2, false)]                 // Qwen1.5-MoE: 8 experts top-2 raw probs
    [InlineData(1, 64, 4, true)]                 // larger expert count, top-4
    [InlineData(4, 8, 2, true)]                  // multi-token batch
    [InlineData(8, 16, 4, true)]                 // 8 tokens, 16 experts, top-4
    [InlineData(1, 16, 1, true)]                 // top-1 — degenerate (norm = identity)
    [InlineData(1, 256, 8, true)]                // MAX_EXPERTS, top-8
    public void Launch_MatchesCpuReference(
        int seqLen, int numExperts, int k, bool normTopKProb)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEAD + seqLen * 13 + numExperts * 7 + k * 3 + (normTopKProb ? 1 : 0));
        float[] logits = RandomFloats(rng, seqLen * numExperts);

        // Reference
        var (expectedIdx, expectedWt) = CpuTopKSoftmax(logits, seqLen, numExperts, k, normTopKProb);

        using var device = VulkanDevice.Create();
        using var kernel = MoeTopKSoftmaxF32Kernel.Create(device, spvDir);

        using var bufLogits = device.Allocate((long)logits.Length * sizeof(float));
        using var bufIdx = device.Allocate((long)seqLen * k * sizeof(int));
        using var bufWeights = device.Allocate((long)seqLen * k * sizeof(float));

        device.Upload(logits, bufLogits);

        kernel.Launch(bufLogits, bufIdx, bufWeights, seqLen, numExperts, k, normTopKProb);

        var actualIdx = new int[seqLen * k];
        var actualWt = new float[seqLen * k];
        // Download API only exposes Span<float>; int and float are both 4 bytes
        // so the bit pattern survives the cast unchanged.
        device.Download(bufIdx, MemoryMarshal.Cast<int, float>(actualIdx.AsSpan()));
        device.Download(bufWeights, actualWt);

        // Indices: must match exactly (deterministic tie-break).
        for (int i = 0; i < expectedIdx.Length; i++)
            Assert.True(
                expectedIdx[i] == actualIdx[i],
                $"Index mismatch at token {i / k} slot {i % k}: cpu={expectedIdx[i]}, gpu={actualIdx[i]}");

        // Weights: F32 noise tolerance.
        AssertClose(expectedWt, actualWt, "weights");
    }

    /// <summary>CPU reference: softmax over experts, then sequential argmax with stable tie-break.</summary>
    private static (int[] idx, float[] wt) CpuTopKSoftmax(
        float[] logits, int seqLen, int numExperts, int k, bool normTopKProb)
    {
        var idx = new int[seqLen * k];
        var wt = new float[seqLen * k];
        var probs = new float[numExperts];

        for (int t = 0; t < seqLen; t++)
        {
            // Softmax (numerically stable).
            float max = float.NegativeInfinity;
            for (int e = 0; e < numExperts; e++)
                if (logits[t * numExperts + e] > max) max = logits[t * numExperts + e];
            float sum = 0;
            for (int e = 0; e < numExperts; e++)
            {
                probs[e] = MathF.Exp(logits[t * numExperts + e] - max);
                sum += probs[e];
            }
            float invSum = 1.0f / sum;
            for (int e = 0; e < numExperts; e++) probs[e] *= invSum;

            // Top-k via repeated argmax with masking.
            for (int slot = 0; slot < k; slot++)
            {
                int bestIdx = -1;
                float bestVal = float.NegativeInfinity;
                for (int e = 0; e < numExperts; e++)
                {
                    if (probs[e] > bestVal)
                    {
                        bestVal = probs[e];
                        bestIdx = e;
                    }
                }
                idx[t * k + slot] = bestIdx;
                wt[t * k + slot] = bestVal;
                probs[bestIdx] = float.NegativeInfinity;
            }

            // Optional renormalise.
            if (normTopKProb)
            {
                float swSum = 0;
                for (int slot = 0; slot < k; slot++) swSum += wt[t * k + slot];
                float invSw = 1.0f / swSum;
                for (int slot = 0; slot < k; slot++) wt[t * k + slot] *= invSw;
            }
        }
        return (idx, wt);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"MoE top-k softmax {label} drift exceeded tolerance: " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
