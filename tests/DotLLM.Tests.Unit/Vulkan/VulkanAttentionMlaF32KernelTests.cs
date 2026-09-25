using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the MLA attention Vulkan kernel.
/// </summary>
/// <remarks>
/// <para>
/// Reference is a self-contained scalar CPU implementation that mirrors
/// the post-projection attention loop of <c>MlaAttention.Execute</c>: per
/// head, score = (Q_nope · K_nope_h + Q_pe · K_pe_shared) * scale, causal
/// mask, softmax, sum over V_h. We feed the Vulkan kernel the same already-
/// projected Q / K_nope / V / K_pe arrays so the comparison isolates the
/// attention math from the projection pipeline.
/// </para>
/// <para>
/// Tolerance: relative 1e-3 / absolute 1e-4 — same as the F32 attention
/// parity tests. The reduction order differs between CPU (per-row scalar)
/// and GPU (online tile softmax + workgroup tree reduce) but the absolute
/// drift on uniform-random inputs at the tested shapes stays well below
/// this bar.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanAttentionMlaF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 1, 1, 16, 16, 16)]                  // smallest workable: 1 head, decode-1
    [InlineData(1, 4, 1, 16, 16, 16)]                  // decode-4 cached
    [InlineData(4, 4, 2, 32, 16, 24)]                  // multi-head, prefill-style
    [InlineData(1, 8, 4, 64, 32, 48)]                  // larger head count, asymmetric v_head_dim
    [InlineData(1, 32, 8, 128, 64, 128)]               // DeepSeek-V2-Lite shape (qk_nope=128, qk_rope=64, v=128, 8 heads)
    [InlineData(2, 2, 2, 128, 64, 128)]                // 2-token prefill at V2-Lite head dims
    public void Launch_MatchesCpuReference(
        int seqQ, int seqKv, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        // positionOffset chosen so that all queries can attend to all KV positions
        // (s ≤ positionOffset + tq for every t). Using positionOffset = seqKv - seqQ
        // mirrors the typical decode case where seqKv = cachedLength + seqQ.
        int positionOffset = seqKv - seqQ;

        var rng = new Random(0xCAFE + seqQ * 31 + seqKv * 17 + numHeads * 11
                              + qkNopeHeadDim + qkRopeHeadDim + vHeadDim);
        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;

        float[] q = RandomFloats(rng, seqQ * numHeads * qkHeadDim);
        float[] kNope = RandomFloats(rng, seqKv * numHeads * qkNopeHeadDim);
        float[] v = RandomFloats(rng, seqKv * numHeads * vHeadDim);
        float[] kPe = RandomFloats(rng, seqKv * qkRopeHeadDim);

        float scale = 1.0f / MathF.Sqrt(qkHeadDim);

        float[] expected = CpuMlaAttention(
            q, kNope, v, kPe,
            seqQ, seqKv, numHeads,
            qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
            positionOffset, scale);

        using var device = VulkanDevice.Create();
        using var kernel = AttentionMlaF32Kernel.Create(device, spvDir);

        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufKNope = device.Allocate((long)kNope.Length * sizeof(float));
        using var bufV = device.Allocate((long)v.Length * sizeof(float));
        using var bufKPe = device.Allocate((long)kPe.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(q, bufQ);
        device.Upload(kNope, bufKNope);
        device.Upload(v, bufV);
        device.Upload(kPe, bufKPe);

        kernel.Launch(bufQ, bufKNope, bufV, bufKPe, bufOut,
            seqQ, seqKv, numHeads,
            qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
            positionOffset, scale);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, seqQ, seqKv, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim);
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    /// <summary>
    /// Self-contained scalar CPU reference for the MLA attention loop.
    /// Mirrors the post-projection loop of <c>MlaAttention.Execute</c>.
    /// </summary>
    private static float[] CpuMlaAttention(
        float[] q, float[] kNope, float[] v, float[] kPe,
        int seqQ, int seqKv, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int positionOffset, float scale)
    {
        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        int qStride = numHeads * qkHeadDim;
        int kNopeStride = numHeads * qkNopeHeadDim;
        int vStride = numHeads * vHeadDim;

        var output = new float[seqQ * numHeads * vHeadDim];

        for (int h = 0; h < numHeads; h++)
        {
            for (int tq = 0; tq < seqQ; tq++)
            {
                int posQ = positionOffset + tq;
                int qBase = tq * qStride + h * qkHeadDim;

                // 1. Compute scores for all KV positions, applying causal mask.
                var scores = new float[seqKv];
                for (int s = 0; s < seqKv; s++)
                {
                    if (s > posQ)
                    {
                        scores[s] = float.NegativeInfinity;
                        continue;
                    }
                    float dot = 0;
                    int kNopeBase = s * kNopeStride + h * qkNopeHeadDim;
                    int kPeBase = s * qkRopeHeadDim;
                    for (int d = 0; d < qkNopeHeadDim; d++)
                        dot += q[qBase + d] * kNope[kNopeBase + d];
                    for (int d = 0; d < qkRopeHeadDim; d++)
                        dot += q[qBase + qkNopeHeadDim + d] * kPe[kPeBase + d];
                    scores[s] = dot * scale;
                }

                // 2. Numerically stable softmax in place.
                float max = float.NegativeInfinity;
                for (int s = 0; s < seqKv; s++) if (scores[s] > max) max = scores[s];
                float sum = 0f;
                for (int s = 0; s < seqKv; s++)
                {
                    if (float.IsNegativeInfinity(scores[s])) { scores[s] = 0; continue; }
                    scores[s] = MathF.Exp(scores[s] - max);
                    sum += scores[s];
                }
                if (sum > 0)
                {
                    float inv = 1f / sum;
                    for (int s = 0; s < seqKv; s++) scores[s] *= inv;
                }

                // 3. Weighted sum over V_h.
                int outBase = tq * vStride + h * vHeadDim;
                for (int d = 0; d < vHeadDim; d++) output[outBase + d] = 0;
                for (int s = 0; s < seqKv; s++)
                {
                    float w = scores[s];
                    if (w == 0f) continue;
                    int vBase = s * vStride + h * vHeadDim;
                    for (int d = 0; d < vHeadDim; d++)
                        output[outBase + d] += w * v[vBase + d];
                }
            }
        }
        return output;
    }

    private static void AssertClose(
        float[] expected, float[] actual,
        int seqQ, int seqKv, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim)
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
            $"MLA attention drift exceeded tolerance " +
            $"(seqQ={seqQ}, seqKv={seqKv}, heads={numHeads}, qkNope={qkNopeHeadDim}, qkRope={qkRopeHeadDim}, v={vHeadDim}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
