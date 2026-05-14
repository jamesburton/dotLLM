using DotLLM.Cpu.Kernels;
using DotLLM.Core.PositionEncoding;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan Flash-Attention F32 kernel.
/// Validates against the scalar CPU reference <see cref="Attention.ExecuteScalar"/>
/// across MHA / GQA / sliding-window / soft-cap / ALiBi configurations and
/// prompt lengths up to 2048 tokens.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanFlashAttentionF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void Launch_Mha_ShortPrefill()
    {
        // 4 queries × 4 keys, single head — exercises the partial-Q-tile
        // branch (rowsInTile == 4 < BR).
        RunOne(seqQ: 4, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_Mha_HeadDim128()
    {
        // Llama-style head dim 128 — exercises the wider qTile / outAccum
        // shared-memory footprint.
        RunOne(seqQ: 16, seqKv: 16, numHeads: 8, numKvHeads: 8, headDim: 128, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_Gqa4_Prefill_128()
    {
        // GQA-4 (numHeads / numKvHeads == 4), prompt length 128 — exercises
        // the GQA broadcast (hkv = hq / 4) for multiple Q-tiles.
        RunOne(seqQ: 128, seqKv: 128, numHeads: 8, numKvHeads: 2, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_Gqa8_Prefill_512()
    {
        // GQA-8, Llama-3.2-1B-ish shape (32 heads / 8 kv heads on a head_dim
        // of 64 to keep MAX_HEAD_DIM=128 happy; production Llama-3 has
        // head_dim 64 too). 512 queries × 512 keys — exercises the multi-KV-tile
        // outer loop.
        RunOne(seqQ: 512, seqKv: 512, numHeads: 32, numKvHeads: 4, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_Gqa8_Prefill_2048()
    {
        // Long-context prefill — 2048 queries × 2048 keys. Exercises 32
        // KV tiles per Q-tile and 128 Q-tiles per head.
        RunOne(seqQ: 2048, seqKv: 2048, numHeads: 8, numKvHeads: 2, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_SlidingWindow_4()
    {
        // Mistral-style sliding window: each query attends only to the most
        // recent 4 keys.
        RunOne(seqQ: 16, seqKv: 32, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, slidingWindow: 4);
    }

    [SkippableFact]
    public void Launch_SoftCap_50()
    {
        // Gemma-2 style attention soft-cap. Choose a scenario where raw
        // scores comfortably exceed the cap so the tanh squashing exercises.
        RunOne(seqQ: 32, seqKv: 32, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, softCap: 50.0f);
    }

    [SkippableFact]
    public void Launch_Alibi_Mha()
    {
        RunOne(seqQ: 16, seqKv: 16, numHeads: 6, numKvHeads: 2, headDim: 64,
            positionOffset: 0, useAlibi: true);
    }

    [SkippableFact]
    public void Launch_PartialKvTile()
    {
        // seqKv = 33 → final KV tile is partial (33 mod 64 = 33). Validates
        // the tileLen clamp inside the KV loop.
        RunOne(seqQ: 16, seqKv: 33, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_PartialQTile()
    {
        // seqQ = 33 → final Q-tile rows = 33 - 16*2 = 1. Validates the
        // partial-row branch (rowsInTile == 1).
        RunOne(seqQ: 33, seqKv: 64, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0);
    }

    // ─────────────────────────────────────────────────────────────

    private static void RunOne(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow = 0, float softCap = 0.0f, bool useAlibi = false)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xF1A54 + seqQ * 41 + seqKv * 17 + numHeads * 7 + headDim);
        float[] qh = RandomFloats(rng, seqQ * numHeads * headDim);
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] expected = new float[seqQ * numHeads * headDim];

        ComputeExpected(qh, kh, vh, expected,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow, softCap, useAlibi);

        using var device = VulkanDevice.Create();
        using var kernel = VulkanFlashAttentionF32Kernel.Create(device, spvDir);

        using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
        using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
        using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(qh.AsSpan(), bufQ);
        device.Upload(kh.AsSpan(), bufK);
        device.Upload(vh.AsSpan(), bufV);

        kernel.Launch(bufQ, bufK, bufV, bufOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim,
            positionOffset: positionOffset, slidingWindow: slidingWindow,
            useAlibi: useAlibi, softCap: softCap);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, seqQ, seqKv, numHeads, numKvHeads, headDim);
    }

    /// <summary>
    /// CPU reference. Uses <see cref="Attention.ExecuteScalar"/> when no
    /// soft-cap is present (the public scalar path matches the shader's
    /// pre-cap math exactly); when soft-cap is non-zero, applies the
    /// tanh squash in-line — there is no soft-cap overload on the CPU
    /// kernel today.
    /// </summary>
    private static void ComputeExpected(
        float[] q, float[] k, float[] v, float[] output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow, float softCap, bool useAlibi)
    {
        if (softCap <= 0f)
        {
            int? swArg = slidingWindow > 0 ? slidingWindow : null;
            if (useAlibi)
                Attention.ExecuteScalar(q, k, v, output,
                    seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                    AlibiPositionEncoding.CreateSlopes(numHeads), swArg);
            else
                Attention.ExecuteScalar(q, k, v, output,
                    seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, swArg);
            return;
        }

        // Soft-cap reference: replicate the scalar path with tanh-capped
        // scores. Matches the shader's `softCap * tanh(s / softCap)` step.
        SoftCapReference(q, k, v, output,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow, softCap, useAlibi);
    }

    private static void SoftCapReference(
        float[] q, float[] k, float[] v, float[] output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow, float softCap, bool useAlibi)
    {
        int groupSize = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);
        float[] slopes = useAlibi ? AlibiPositionEncoding.CreateSlopes(numHeads) : Array.Empty<float>();

        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / groupSize;
            float slope = slopes.Length > 0 ? slopes[h] : 0f;

            for (int i = 0; i < seqQ; i++)
            {
                int posQ = positionOffset + i;
                float[] scores = new float[seqKv];
                for (int j = 0; j < seqKv; j++)
                {
                    bool masked =
                        j > posQ ||
                        (slidingWindow > 0 && posQ - j >= slidingWindow);
                    if (masked) { scores[j] = float.NegativeInfinity; continue; }

                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * qStride + h * headDim + d] * k[j * kvStride + kvH * headDim + d];
                    float s = dot * scale - slope * (posQ - j);
                    scores[j] = softCap * MathF.Tanh(s / softCap);
                }

                float maxS = float.NegativeInfinity;
                for (int j = 0; j < seqKv; j++) if (scores[j] > maxS) maxS = scores[j];
                float sum = 0;
                for (int j = 0; j < seqKv; j++)
                {
                    scores[j] = (scores[j] > float.NegativeInfinity * 0.5f) ? MathF.Exp(scores[j] - maxS) : 0f;
                    sum += scores[j];
                }
                float inv = sum > 0 ? 1f / sum : 0f;

                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0;
                    for (int j = 0; j < seqKv; j++)
                        acc += scores[j] * v[j * kvStride + kvH * headDim + d];
                    output[i * qStride + h * headDim + d] = acc * inv;
                }
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

    private static void AssertClose(float[] expected, float[] actual,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
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
            $"FlashAttention drift exceeded tolerance " +
            $"(seqQ={seqQ},seqKv={seqKv},nh={numHeads},nkv={numKvHeads},hd={headDim}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
