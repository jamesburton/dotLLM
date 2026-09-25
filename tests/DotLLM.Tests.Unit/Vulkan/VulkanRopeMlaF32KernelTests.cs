using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the MLA RoPE Vulkan kernel.
/// </summary>
/// <remarks>
/// Reference is a self-contained scalar CPU implementation that mirrors
/// the Norm/interleaved RoPE pairing used by DeepSeek-V2/V3 (and the CPU
/// MLA path's <c>ApplyRopeNormInPlace</c>): the q_pe tail of each Q head
/// is rotated, and the MQA-shared K_pe is rotated once per token.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanRopeMlaF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 1, 0, 16, 10000.0f)]                  // smallest: 1 head, qkNope=0
    [InlineData(1, 1, 16, 16, 10000.0f)]                 // qkNope==qkRope, 1 head
    [InlineData(4, 2, 32, 16, 10000.0f)]                 // multi-token, multi-head
    [InlineData(1, 8, 128, 64, 10000.0f)]                // DeepSeek-V2-Lite shape
    [InlineData(1, 16, 128, 64, 10000.0f)]               // DeepSeek-V2 full shape
    [InlineData(8, 16, 128, 64, 10000.0f)]               // DeepSeek-V2 prefill
    public void Launch_MatchesCpuReference(
        int seqLen, int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, float theta)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xD00D + seqLen * 13 + numHeads * 7 + qkNopeHeadDim + qkRopeHeadDim);
        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;

        float[] q = RandomFloats(rng, seqLen * numHeads * qkHeadDim);
        float[] kPe = RandomFloats(rng, seqLen * qkRopeHeadDim);
        int[] positions = new int[seqLen];
        for (int t = 0; t < seqLen; t++) positions[t] = t;

        // Reference: in-place rotation on copies.
        float[] expectedQ = (float[])q.Clone();
        float[] expectedKPe = (float[])kPe.Clone();
        CpuRopeMla(expectedQ, expectedKPe, positions,
            seqLen, numHeads, qkNopeHeadDim, qkRopeHeadDim, theta);

        using var device = VulkanDevice.Create();
        using var kernel = RopeMlaF32Kernel.Create(device, spvDir);

        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufKPe = device.Allocate((long)kPe.Length * sizeof(float));
        using var bufPos = device.Allocate((long)positions.Length * sizeof(int));

        device.Upload(q, bufQ);
        device.Upload(kPe, bufKPe);
        device.Upload(MemoryMarshal.AsBytes<int>(positions), bufPos);

        kernel.Launch(bufQ, bufKPe, bufPos, seqLen, numHeads, qkNopeHeadDim, qkRopeHeadDim, theta);

        float[] actualQ = new float[q.Length];
        float[] actualKPe = new float[kPe.Length];
        device.Download(bufQ, actualQ);
        device.Download(bufKPe, actualKPe);

        AssertClose(expectedQ, actualQ, "Q (q_nope passthrough + q_pe rotation)");
        AssertClose(expectedKPe, actualKPe, "K_pe");
    }

    /// <summary>
    /// CPU reference: rotates q_pe slices in place and the shared K_pe in
    /// place. Uses Norm/interleaved pairing — pair <c>(2i, 2i+1)</c>
    /// within each rope sub-dim. Matches <c>MlaAttention.ApplyRopeNormInPlace</c>.
    /// </summary>
    private static void CpuRopeMla(
        float[] q, float[] kPe, int[] positions,
        int seqLen, int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, float theta)
    {
        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        int halfRope = qkRopeHeadDim / 2;

        for (int t = 0; t < seqLen; t++)
        {
            int pos = positions[t];
            for (int pair = 0; pair < halfRope; pair++)
            {
                float expn = (float)(2 * pair) / qkRopeHeadDim;
                float freq = 1.0f / MathF.Pow(theta, expn);
                float angle = pos * freq;
                float c = MathF.Cos(angle);
                float s = MathF.Sin(angle);

                // Rotate q_pe within each head.
                for (int h = 0; h < numHeads; h++)
                {
                    int headBase = t * numHeads * qkHeadDim + h * qkHeadDim + qkNopeHeadDim;
                    int i0 = headBase + 2 * pair;
                    int i1 = i0 + 1;
                    float v0 = q[i0];
                    float v1 = q[i1];
                    q[i0] = v0 * c - v1 * s;
                    q[i1] = v0 * s + v1 * c;
                }

                // Rotate shared K_pe (one rotation per token, no head dim).
                int kBase = t * qkRopeHeadDim;
                int k0 = kBase + 2 * pair;
                int k1 = k0 + 1;
                float kv0 = kPe[k0];
                float kv1 = kPe[k1];
                kPe[k0] = kv0 * c - kv1 * s;
                kPe[k1] = kv0 * s + kv1 * c;
            }
        }
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
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
            $"MLA RoPE drift exceeded tolerance ({label}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
