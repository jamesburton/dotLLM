using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity test (#742) for the Vulkan sigmoid + selection-bias MoE router (DeepSeek-V3 / GLM-4.7-Flash).
/// The reference mirrors <c>MoeSwiGluMlp.Route(sigmoidGating: true)</c>: select on sigmoid(logit)+bias,
/// weight with the UNBIASED sigmoid, optional renorm (clamped 6.1035e-5), then scale.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeTopKSigmoidBiasF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 8, 2, true, 1.0f)]
    [InlineData(1, 8, 2, false, 1.0f)]
    [InlineData(3, 64, 4, true, 1.8f)]       // scale applied after renorm
    [InlineData(2, 64, 6, false, 2.5f)]      // scale without renorm
    [InlineData(1, 16, 1, true, 1.0f)]
    [InlineData(2, 256, 8, true, 1.8f)]      // MAX_EXPERTS
    public void Launch_MatchesCpuReference_AndBiasChangesSelection(
        int seqLen, int numExperts, int k, bool norm, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x742 + seqLen * 13 + numExperts * 7 + k);
        float[] logits = new float[seqLen * numExperts];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() * 6 - 3);
        float[] bias = new float[numExperts];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 1.5 - 0.5);

        var (expIdx, expW) = Reference(logits, bias, seqLen, numExperts, k, norm, scale);
        var (zeroIdx, _) = Reference(logits, new float[numExperts], seqLen, numExperts, k, norm, scale);
        // Sensitivity control: the bias must actually change the chosen experts, otherwise this
        // test could not tell a bias-ignoring kernel from a correct one.
        Assert.NotEqual(zeroIdx, expIdx);

        var (gpuIdx, gpuW) = Run(spvDir, logits, bias, seqLen, numExperts, k, norm, scale);
        Assert.Equal(expIdx, gpuIdx);
        for (int i = 0; i < expW.Length; i++)
            Assert.True(MathF.Abs(expW[i] - gpuW[i]) <= 1e-5f + 1e-3f * MathF.Abs(expW[i]),
                $"weight[{i}] cpu={expW[i]:G9} gpu={gpuW[i]:G9}");
    }

    [SkippableFact]
    public void Launch_WeightsAreUnbiasedSigmoid()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // logits [2,1,0,-1], bias promotes expert 2 over expert 1 (same case as the CPU Route test).
        float[] logits = [2f, 1f, 0f, -1f];
        float[] bias = [0f, 0f, 0.5f, 0.5f];
        var (idx, w) = Run(spvDir, logits, bias, 1, 4, 2, norm: true, scale: 2.0f);
        Assert.Equal([2, 0], idx);
        float s2 = Sig(0f), s0 = Sig(2f), sum = s2 + s0;
        Assert.Equal(s2 / sum * 2f, w[0], 5);
        Assert.Equal(s0 / sum * 2f, w[1], 5);
    }

    private static float Sig(float x) => 1f / (1f + MathF.Exp(-x));

    private static (int[] idx, float[] w) Run(string spvDir, float[] logits, float[] bias,
        int seqLen, int numExperts, int k, bool norm, float scale)
    {
        using var device = VulkanDevice.Create();
        using var kernel = MoeTopKSigmoidBiasF32Kernel.Create(device, spvDir);
        using var bL = device.Allocate((long)logits.Length * 4);
        using var bB = device.Allocate((long)bias.Length * 4);
        using var bI = device.Allocate((long)seqLen * k * 4);
        using var bW = device.Allocate((long)seqLen * k * 4);
        device.Upload(logits, bL);
        device.Upload(bias, bB);
        kernel.Launch(bL, bB, bI, bW, seqLen, numExperts, k, norm, scale);
        var idx = new int[seqLen * k];
        var w = new float[seqLen * k];
        device.Download(bI, MemoryMarshal.Cast<int, float>(idx.AsSpan()));
        device.Download(bW, w);
        return (idx, w);
    }

    private static (int[] idx, float[] w) Reference(float[] logits, float[] bias,
        int seqLen, int numExperts, int k, bool norm, float scale)
    {
        var idx = new int[seqLen * k];
        var w = new float[seqLen * k];
        var p = new float[numExperts];
        var sel = new float[numExperts];
        for (int t = 0; t < seqLen; t++)
        {
            for (int e = 0; e < numExperts; e++)
            {
                p[e] = Sig(logits[t * numExperts + e]);
                sel[e] = p[e] + bias[e];
            }
            float sum = 0;
            for (int s = 0; s < k; s++)
            {
                int best = 0; float bv = float.NegativeInfinity;
                for (int e = 0; e < numExperts; e++) if (sel[e] > bv) { bv = sel[e]; best = e; }
                idx[t * k + s] = best; w[t * k + s] = p[best]; sum += p[best];
                sel[best] = float.NegativeInfinity;
            }
            float inv = norm ? 1f / MathF.Max(sum, 6.103515625e-5f) : 1f;
            for (int s = 0; s < k; s++) w[t * k + s] *= inv * scale;
        }
        return (idx, w);
    }
}
