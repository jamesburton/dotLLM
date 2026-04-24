using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class RoPETests
{
    [Fact]
    public void PrecomputeFrequencyTable_KnownTheta_MatchesHandCalculated()
    {
        // headDim=4 → halfDim=2, theta=10000, pos=1
        // freq[0] = 1 / 10000^(0/4) = 1.0
        // freq[1] = 1 / 10000^(2/4) = 1/100 = 0.01
        // angle[0] = 1 * 1.0 = 1.0, angle[1] = 1 * 0.01 = 0.01
        const int headDim = 4;
        const float theta = 10000f;
        const int maxSeqLen = 2;
        int halfDim = headDim / 2;

        float[] cos = new float[maxSeqLen * halfDim];
        float[] sin = new float[maxSeqLen * halfDim];

        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cos, sin);

        // pos=1, i=0: angle = 1.0
        Assert.Equal(MathF.Cos(1.0f), cos[1 * halfDim + 0], 1e-5f);
        Assert.Equal(MathF.Sin(1.0f), sin[1 * halfDim + 0], 1e-5f);

        // pos=1, i=1: angle = 0.01
        Assert.Equal(MathF.Cos(0.01f), cos[1 * halfDim + 1], 1e-5f);
        Assert.Equal(MathF.Sin(0.01f), sin[1 * halfDim + 1], 1e-5f);
    }

    [Fact]
    public void PrecomputeFrequencyTable_PositionZero_CosOnesSinZeros()
    {
        const int headDim = 128;
        const float theta = 10000f;
        const int maxSeqLen = 1;
        int halfDim = headDim / 2;

        float[] cos = new float[maxSeqLen * halfDim];
        float[] sin = new float[maxSeqLen * halfDim];

        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cos, sin);

        // pos=0: angle = 0 for all dimensions → cos=1, sin=0
        for (int i = 0; i < halfDim; i++)
        {
            Assert.Equal(1.0f, cos[i], 1e-6f);
            Assert.Equal(0.0f, sin[i], 1e-6f);
        }
    }

    [Fact]
    public void PrecomputeFrequencyTable_CustomTheta_DifferentFrequencies()
    {
        const int headDim = 8;
        const int maxSeqLen = 4;
        int halfDim = headDim / 2;

        float[] cos1 = new float[maxSeqLen * halfDim];
        float[] sin1 = new float[maxSeqLen * halfDim];
        float[] cos2 = new float[maxSeqLen * halfDim];
        float[] sin2 = new float[maxSeqLen * halfDim];

        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, 10000f, cos1, sin1);
        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, 500000f, cos2, sin2);

        // Different theta should produce different values at pos > 0
        bool anyDifferent = false;
        for (int i = 0; i < maxSeqLen * halfDim; i++)
        {
            if (MathF.Abs(cos1[i] - cos2[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Different theta values should produce different frequency tables");
    }

    [Fact]
    public void ApplyRotation_PositionZero_NoChange()
    {
        // cos=1, sin=0 → identity: vec' = vec * 1 - vec_odd * 0 = vec
        const int headDim = 8;
        int halfDim = headDim / 2;

        float[] vec = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] original = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] cos = new float[halfDim];
        float[] sin = new float[halfDim];
        Array.Fill(cos, 1.0f);
        // sin already zero

        RoPE.ApplyRotation(vec, cos, sin, headDim);

        for (int i = 0; i < headDim; i++)
            Assert.Equal(original[i], vec[i], 1e-6f);
    }

    [Fact]
    public void ApplyRotation_KnownRotation_HandCalculated()
    {
        // headDim=4, halfDim=2
        // vec = [1, 2, 3, 4]
        // cos = [cos(1.0), cos(0.01)]
        // sin = [sin(1.0), sin(0.01)]
        //
        // vec'[0] = 1 * cos(1.0) - 2 * sin(1.0)
        // vec'[1] = 1 * sin(1.0) + 2 * cos(1.0)
        // vec'[2] = 3 * cos(0.01) - 4 * sin(0.01)
        // vec'[3] = 3 * sin(0.01) + 4 * cos(0.01)
        const int headDim = 4;
        float[] vec = [1f, 2f, 3f, 4f];
        float[] cos = [MathF.Cos(1.0f), MathF.Cos(0.01f)];
        float[] sin = [MathF.Sin(1.0f), MathF.Sin(0.01f)];

        RoPE.ApplyRotation(vec, cos, sin, headDim);

        Assert.Equal(1f * MathF.Cos(1.0f) - 2f * MathF.Sin(1.0f), vec[0], 1e-5f);
        Assert.Equal(1f * MathF.Sin(1.0f) + 2f * MathF.Cos(1.0f), vec[1], 1e-5f);
        Assert.Equal(3f * MathF.Cos(0.01f) - 4f * MathF.Sin(0.01f), vec[2], 1e-5f);
        Assert.Equal(3f * MathF.Sin(0.01f) + 4f * MathF.Cos(0.01f), vec[3], 1e-5f);
    }

    [Fact]
    public void ApplyRotation_ScalarMatchesSIMD()
    {
        if (!Avx2.IsSupported)
            return; // SIMD path not available

        const int headDim = 128;
        int halfDim = headDim / 2;
        var rng = new Random(42);

        float[] vecSimd = new float[headDim];
        float[] vecScalar = new float[headDim];
        float[] cos = new float[halfDim];
        float[] sin = new float[halfDim];

        for (int i = 0; i < headDim; i++)
        {
            float v = rng.NextSingle() * 2f - 1f;
            vecSimd[i] = v;
            vecScalar[i] = v;
        }

        for (int i = 0; i < halfDim; i++)
        {
            float angle = rng.NextSingle() * MathF.PI * 2f;
            cos[i] = MathF.Cos(angle);
            sin[i] = MathF.Sin(angle);
        }

        RoPE.ApplyRotation(vecSimd, cos, sin, headDim);
        RoPE.ApplyRotationScalar(vecScalar, cos, sin, headDim);

        for (int i = 0; i < headDim; i++)
            Assert.Equal(vecScalar[i], vecSimd[i], 1e-5f);
    }

    [Fact]
    public void Execute_MultipleHeads_EachRotatedIndependently()
    {
        const int headDim = 4;
        const int numHeads = 2;
        const int numKvHeads = 2;
        const float theta = 10000f;
        int halfDim = headDim / 2;
        int maxSeqLen = 2;

        float[] cosTable = new float[maxSeqLen * halfDim];
        float[] sinTable = new float[maxSeqLen * halfDim];
        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cosTable, sinTable);

        // Q: 2 heads, each headDim=4 → 8 floats
        float[] q = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] k = [9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f];
        int[] positions = [1];

        // Apply individually to verify
        float[] qHead0 = [1f, 2f, 3f, 4f];
        float[] qHead1 = [5f, 6f, 7f, 8f];
        var cos1 = cosTable.AsSpan(1 * halfDim, halfDim);
        var sin1 = sinTable.AsSpan(1 * halfDim, halfDim);
        RoPE.ApplyRotationScalar(qHead0, cos1, sin1, headDim);
        RoPE.ApplyRotationScalar(qHead1, cos1, sin1, headDim);

        RoPE.Execute(q, k, positions, numHeads, numKvHeads, headDim, cosTable, sinTable);

        // Each head rotated with same position angles
        for (int i = 0; i < headDim; i++)
        {
            Assert.Equal(qHead0[i], q[i], 1e-5f);
            Assert.Equal(qHead1[i], q[headDim + i], 1e-5f);
        }
    }

    [Fact]
    public void Execute_DifferentPositions_DifferentRotations()
    {
        const int headDim = 4;
        const int numHeads = 1;
        const int numKvHeads = 1;
        const float theta = 10000f;
        int halfDim = headDim / 2;
        int maxSeqLen = 4;

        float[] cosTable = new float[maxSeqLen * halfDim];
        float[] sinTable = new float[maxSeqLen * halfDim];
        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cosTable, sinTable);

        // 3 tokens with positions [0, 1, 2]
        float[] q = [1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f];
        float[] k = [1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f];
        int[] positions = [0, 1, 2];

        RoPE.Execute(q, k, positions, numHeads, numKvHeads, headDim, cosTable, sinTable);

        // Position 0: cos=1, sin=0 → no change → q[0..3] = [1, 0, 0, 0]
        Assert.Equal(1.0f, q[0], 1e-6f);
        Assert.Equal(0.0f, q[1], 1e-6f);

        // Position 1 and 2 should differ from position 0 and from each other
        Assert.NotEqual(q[0], q[4], 1e-3f);
        Assert.NotEqual(q[4], q[8], 1e-3f);
    }

    [Fact]
    public void Execute_ScalarMatchesSIMD_LargeInput()
    {
        if (!Avx2.IsSupported)
            return;

        const int headDim = 128;
        const int numHeads = 32;
        const int numKvHeads = 8;
        const int maxSeqLen = 4;
        const float theta = 10000f;
        int halfDim = headDim / 2;
        var rng = new Random(123);

        float[] cosTable = new float[maxSeqLen * halfDim];
        float[] sinTable = new float[maxSeqLen * halfDim];
        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cosTable, sinTable);

        int qLen = maxSeqLen * numHeads * headDim;
        int kLen = maxSeqLen * numKvHeads * headDim;

        float[] qSimd = new float[qLen];
        float[] kSimd = new float[kLen];
        float[] qScalar = new float[qLen];
        float[] kScalar = new float[kLen];
        int[] positions = new int[maxSeqLen];

        for (int i = 0; i < qLen; i++)
        {
            float v = rng.NextSingle() * 2f - 1f;
            qSimd[i] = v;
            qScalar[i] = v;
        }

        for (int i = 0; i < kLen; i++)
        {
            float v = rng.NextSingle() * 2f - 1f;
            kSimd[i] = v;
            kScalar[i] = v;
        }

        for (int i = 0; i < maxSeqLen; i++)
            positions[i] = i;

        RoPE.Execute(qSimd, kSimd, positions, numHeads, numKvHeads, headDim, cosTable, sinTable);
        RoPE.ExecuteScalar(qScalar, kScalar, positions, numHeads, numKvHeads, headDim, cosTable, sinTable);

        for (int i = 0; i < qLen; i++)
            Assert.Equal(qScalar[i], qSimd[i], 1e-5f);

        for (int i = 0; i < kLen; i++)
            Assert.Equal(kScalar[i], kSimd[i], 1e-5f);
    }

    [Fact]
    public void ApplyRotation_HeadDim128_PairRotation()
    {
        // Verify canonical pair-wise rotation: each (2i, 2i+1) pair is rotated
        // by the corresponding angle, independent of other pairs.
        const int headDim = 128;
        int halfDim = headDim / 2;

        float[] vec = new float[headDim];
        float[] cos = new float[halfDim];
        float[] sin = new float[halfDim];

        // Set one pair to (1, 0), rest zeros. Only that pair should change.
        int testPair = 17;
        vec[2 * testPair] = 1.0f;
        vec[2 * testPair + 1] = 0.0f;

        float angle = 0.5f;
        cos[testPair] = MathF.Cos(angle);
        sin[testPair] = MathF.Sin(angle);
        // Other cos/sin = (1, 0) → identity
        for (int i = 0; i < halfDim; i++)
        {
            if (i != testPair)
            {
                cos[i] = 1.0f;
                sin[i] = 0.0f;
            }
        }

        RoPE.ApplyRotation(vec, cos, sin, headDim);

        // The test pair should be rotated
        Assert.Equal(MathF.Cos(angle), vec[2 * testPair], 1e-5f);
        Assert.Equal(MathF.Sin(angle), vec[2 * testPair + 1], 1e-5f);

        // Other pairs should remain zero
        for (int i = 0; i < halfDim; i++)
        {
            if (i != testPair)
            {
                Assert.Equal(0.0f, vec[2 * i], 1e-6f);
                Assert.Equal(0.0f, vec[2 * i + 1], 1e-6f);
            }
        }
    }

    [Fact]
    public void Execute_PartialRotation_OnlyFirstRopeDimRotated()
    {
        // headDim=8, ropeDim=4: first 4 dims rotated, last 4 unchanged.
        const int headDim = 8;
        const int ropeDim = 4;
        const int numHeads = 2;
        const int numKvHeads = 1;
        const float theta = 10000f;
        const int maxSeqLen = 2;
        int halfRopeDim = ropeDim / 2;

        float[] cosTable = new float[maxSeqLen * halfRopeDim];
        float[] sinTable = new float[maxSeqLen * halfRopeDim];
        RoPE.PrecomputeFrequencyTable(maxSeqLen, ropeDim, theta, cosTable, sinTable);

        // Q: 2 heads × 8 dims = 16 floats per token, 1 token at position 1
        float[] q = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,   // head 0
                     9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f]; // head 1
        float[] k = [17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f]; // 1 KV head
        float[] qOrig = (float[])q.Clone();
        float[] kOrig = (float[])k.Clone();
        int[] positions = [1];

        RoPE.Execute(q, k, positions, numHeads, numKvHeads, headDim, ropeDim, cosTable, sinTable);

        var cos1 = cosTable.AsSpan(1 * halfRopeDim, halfRopeDim);
        var sin1 = sinTable.AsSpan(1 * halfRopeDim, halfRopeDim);

        // Verify first ropeDim (4) dimensions of each head ARE rotated.
        for (int h = 0; h < numHeads; h++)
        {
            int offset = h * headDim;
            float[] expected = new float[ropeDim];
            Array.Copy(qOrig, offset, expected, 0, ropeDim);
            RoPE.ApplyRotationScalar(expected, cos1, sin1, ropeDim);

            for (int d = 0; d < ropeDim; d++)
                Assert.Equal(expected[d], q[offset + d], 1e-5f);
        }

        // Verify last (headDim - ropeDim) dimensions of each head are UNCHANGED.
        for (int h = 0; h < numHeads; h++)
        {
            int offset = h * headDim;
            for (int d = ropeDim; d < headDim; d++)
                Assert.Equal(qOrig[offset + d], q[offset + d], 1e-6f);
        }

        // Same for K.
        {
            float[] expectedK = new float[ropeDim];
            Array.Copy(kOrig, 0, expectedK, 0, ropeDim);
            RoPE.ApplyRotationScalar(expectedK, cos1, sin1, ropeDim);
            for (int d = 0; d < ropeDim; d++)
                Assert.Equal(expectedK[d], k[d], 1e-5f);
            for (int d = ropeDim; d < headDim; d++)
                Assert.Equal(kOrig[d], k[d], 1e-6f);
        }
    }

    [Fact]
    public void PrecomputeFrequencyTable_MatchesReference()
    {
        // Llama-like config: headDim=128, theta=10000
        // Verify specific values at known positions.
        const int headDim = 128;
        const float theta = 10000f;
        const int maxSeqLen = 8;
        int halfDim = headDim / 2;

        float[] cos = new float[maxSeqLen * halfDim];
        float[] sin = new float[maxSeqLen * halfDim];

        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cos, sin);

        // pos=5, i=0: freq = 1.0, angle = 5.0
        float expectedCos = MathF.Cos(5.0f);
        float expectedSin = MathF.Sin(5.0f);
        Assert.Equal(expectedCos, cos[5 * halfDim + 0], 1e-5f);
        Assert.Equal(expectedSin, sin[5 * halfDim + 0], 1e-5f);

        // pos=5, i=32: freq = 1/10000^(64/128) = 0.01, angle = 0.05
        float freq32 = 1.0f / MathF.Pow(theta, 64.0f / 128.0f);
        float angle32 = 5.0f * freq32;
        Assert.Equal(MathF.Cos(angle32), cos[5 * halfDim + 32], 1e-5f);
        Assert.Equal(MathF.Sin(angle32), sin[5 * halfDim + 32], 1e-5f);

        // pos=5, i=63 (last): freq = 1/10000^(126/128), angle = 5 * freq
        float freq63 = 1.0f / MathF.Pow(theta, 126.0f / 128.0f);
        float angle63 = 5.0f * freq63;
        Assert.Equal(MathF.Cos(angle63), cos[5 * halfDim + 63], 1e-5f);
        Assert.Equal(MathF.Sin(angle63), sin[5 * halfDim + 63], 1e-5f);
    }

    /// <summary>
    /// Locks the HF-Llama RoPE convention. HuggingFace <c>modeling_llama.apply_rotary_pos_emb</c>
    /// computes <c>q' = (q * cos) + (rotate_half(q) * sin)</c> where
    /// <c>rotate_half([a, b, c, d]) = [-c, -d, a, b]</c> — i.e. it rotates HALVES,
    /// not adjacent-pairs. This matches dotLLM's <see cref="RoPE.ApplyRotationNeoX"/>
    /// (pairs <c>(i, i+halfDim)</c>), NOT <see cref="RoPE.ApplyRotation"/>
    /// (pairs <c>(2i, 2i+1)</c>). This test fixes that fact as a regression lock.
    /// </summary>
    [Fact]
    public void NeoXRotation_MatchesHuggingFaceRotateHalfConvention_HeadDim4()
    {
        // Fixed input vector and cos/sin pair. Cos/sin length = halfDim = 2 because
        // HF stores the first-half frequencies once; the second half is the same
        // freqs duplicated (torch.cat([freqs, freqs], dim=-1)), so for our kernels'
        // halved tables the two conventions see identical cos/sin values.
        const int headDim = 4;
        Span<float> dotllmNeoX = stackalloc float[headDim] { 1.0f, 2.0f, 3.0f, 4.0f };
        Span<float> dotllmNorm = stackalloc float[headDim] { 1.0f, 2.0f, 3.0f, 4.0f };
        Span<float> hfRef = stackalloc float[headDim] { 1.0f, 2.0f, 3.0f, 4.0f };

        ReadOnlySpan<float> cos = stackalloc float[2] { 0.5f, 0.8f };
        ReadOnlySpan<float> sin = stackalloc float[2] { 0.866025f, 0.6f };

        // dotLLM NeoX rotation: pairs (i, i+halfDim).
        RoPE.ApplyRotationNeoXScalar(dotllmNeoX, cos, sin, headDim);

        // dotLLM Norm rotation: pairs (2i, 2i+1).
        RoPE.ApplyRotationScalar(dotllmNorm, cos, sin, headDim);

        // HF reference — translate rotate_half directly:
        //   rotate_half([a,b,c,d]) = [-c, -d, a, b]
        //   cos_full = [cos[0], cos[1], cos[0], cos[1]]   (torch.cat([cos, cos], -1))
        //   sin_full = [sin[0], sin[1], sin[0], sin[1]]
        //   q' = q * cos_full + rotate_half(q) * sin_full
        float[] cosFull = [cos[0], cos[1], cos[0], cos[1]];
        float[] sinFull = [sin[0], sin[1], sin[0], sin[1]];
        float[] rotatedHalf = [-hfRef[2], -hfRef[3], hfRef[0], hfRef[1]];
        Span<float> hfExpected = stackalloc float[headDim];
        for (int i = 0; i < headDim; i++)
            hfExpected[i] = hfRef[i] * cosFull[i] + rotatedHalf[i] * sinFull[i];

        // dotLLM NeoX must match HF exactly.
        for (int i = 0; i < headDim; i++)
            Assert.Equal(hfExpected[i], dotllmNeoX[i], 1e-5f);

        // dotLLM Norm must differ from HF — this proves the two conventions are
        // materially different, not just near-equal noise.
        bool anyDiffers = false;
        for (int i = 0; i < headDim; i++)
        {
            if (MathF.Abs(dotllmNorm[i] - hfExpected[i]) > 0.05f)
            {
                anyDiffers = true;
                break;
            }
        }
        Assert.True(anyDiffers,
            "RoPEType.Norm (interleaved pairs) must produce different output than HF "
            + "rotate_half (halves). If this assertion fires, either the conventions "
            + "converged by coincidence for this input or the Norm kernel regressed.");
    }

    /// <summary>
    /// DeepSeek-V2-Lite YaRN config:
    ///   rope_theta=10000, qk_rope_head_dim=64, factor=40,
    ///   original_max_position_embeddings=4096, beta_fast=32, beta_slow=1,
    ///   mscale=0.707, mscale_all_dim=0.707 (ratio → 1.0, cos/sin unscaled).
    /// Verifies inv_freq[i] = freq_inter * mask + freq_extra * (1 - mask) against
    /// a hand-computed reference that mirrors HF's DeepseekV2YarnRotaryEmbedding.
    /// Also asserts YaRN is a no-op within the original context window (pos &lt;
    /// original_max_position_embeddings) for fast-rotation dims where mask == 0
    /// — i.e. the plain-RoPE path and YaRN path agree on the first few tokens
    /// for high-frequency dims (dims near 0).
    /// </summary>
    [Fact]
    public void YarnFrequencyRescale_MatchesDeepSeekV2LiteFormula()
    {
        const int headDim = 64;
        const float theta = 10000f;
        const float factor = 40f;
        const int origMaxPos = 4096;
        const float betaFast = 32f;
        const float betaSlow = 1f;
        const float mscaleMultiplier = 1.0f; // V2-Lite: mscale == mscale_all_dim
        const int maxSeqLen = 8;
        int halfDim = headDim / 2;

        float[] cos = new float[maxSeqLen * halfDim];
        float[] sin = new float[maxSeqLen * halfDim];

        RoPE.PrecomputeFrequencyTableYarn(
            maxSeqLen, headDim, theta, factor, origMaxPos, betaFast, betaSlow,
            mscaleMultiplier, cos, sin);

        // Hand-compute the ramp bounds per HF yarn_find_correction_range.
        // yarn_find_correction_dim(r, dim, base, maxPos) =
        //     dim * log(maxPos / (r * 2 * pi)) / (2 * log(base))
        double lowReal = headDim * Math.Log(origMaxPos / (betaFast * 2.0 * Math.PI)) / (2.0 * Math.Log(theta));
        double highReal = headDim * Math.Log(origMaxPos / (betaSlow * 2.0 * Math.PI)) / (2.0 * Math.Log(theta));
        float low = MathF.Max((float)Math.Floor(lowReal), 0f);
        float high = MathF.Min((float)Math.Ceiling(highReal), headDim - 1);
        if (low == high) high += 0.001f;

        // Build reference inv_freq and cos/sin.
        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                float exponent = 2.0f * i / headDim;
                float freqExtra = 1.0f / MathF.Pow(theta, exponent);
                float freqInter = 1.0f / (factor * MathF.Pow(theta, exponent));
                float linear = (i - low) / (high - low);
                float mask = linear < 0f ? 0f : (linear > 1f ? 1f : linear);
                float invFreq = freqInter * mask + freqExtra * (1.0f - mask);
                float angle = pos * invFreq;

                Assert.Equal(MathF.Cos(angle), cos[pos * halfDim + i], 1e-5f);
                Assert.Equal(MathF.Sin(angle), sin[pos * halfDim + i], 1e-5f);
            }
        }
    }

    /// <summary>
    /// For the fast-rotation dims (mask == 0, i.e. i &lt; low), YaRN uses
    /// freq_extra exactly — so plain RoPE and YaRN must agree bit-for-bit on
    /// those dims. This guards against accidental rescaling of the original
    /// context window.
    /// </summary>
    [Fact]
    public void YarnFrequencyRescale_FastRotationDims_AgreeWithPlainRope()
    {
        const int headDim = 64;
        const float theta = 10000f;
        const float factor = 40f;
        const int origMaxPos = 4096;
        const int maxSeqLen = 16;
        int halfDim = headDim / 2;

        float[] cosYarn = new float[maxSeqLen * halfDim];
        float[] sinYarn = new float[maxSeqLen * halfDim];
        float[] cosPlain = new float[maxSeqLen * halfDim];
        float[] sinPlain = new float[maxSeqLen * halfDim];

        RoPE.PrecomputeFrequencyTableYarn(
            maxSeqLen, headDim, theta, factor, origMaxPos, 32f, 1f,
            mscaleMultiplier: 1.0f, cosYarn, sinYarn);
        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cosPlain, sinPlain);

        // Find 'low' per HF yarn_find_correction_range(beta_fast=32, ...).
        double lowReal = headDim * Math.Log(origMaxPos / (32.0 * 2.0 * Math.PI)) / (2.0 * Math.Log(theta));
        int low = Math.Max((int)Math.Floor(lowReal), 0);

        // Dims 0..low-1 should match plain exactly; dims above 'high' should
        // use freq_inter (i.e. differ). We only assert the agreement side.
        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < low; i++)
            {
                Assert.Equal(cosPlain[pos * halfDim + i], cosYarn[pos * halfDim + i], 1e-6f);
                Assert.Equal(sinPlain[pos * halfDim + i], sinYarn[pos * halfDim + i], 1e-6f);
            }
        }
    }
}
