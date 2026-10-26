using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// PTQ1_0 (PrismML Bonsai dense-trit ternary, GGUF type 143) CPU kernel tests — issue #436.
///
/// <para>These tests are the <i>structural</i> half of the correctness story: they build packed
/// blocks with a local port of the fork's <c>quantize_row_ptq1_0_ref</c> and check that the
/// dequant, the GEMV and the GEMM all agree with what was packed. That catches ordering and
/// scale-placement errors independently of any external artifact. The <i>authoritative</i> half is
/// <see cref="PTQ1_0ForkOracleTests"/>, which compares raw bit patterns against the fork's own
/// compiled routine on real model bytes — because a self-consistent pack/unpack pair can be
/// self-consistently wrong.</para>
/// </summary>
public sealed class PTQ1_0Tests
{
    private const int G = QuantFormat.TernaryGroupSize;      // 128
    private const int B = QuantFormat.PTQ1_0BlockBytes;      // 28
    private const int QsBytes = 24;
    private const int QhBytes = 2;

    [Theory]
    [InlineData(128, 28)]
    [InlineData(5120, 5120 / 128 * 28)]
    [InlineData(17408, 17408 / 128 * 28)]
    public void RowByteSize_IsTwentyEightBytesPerHundredTwentyEightWeights(long elementCount, long expected)
    {
        Assert.Equal(expected, Dequantize.RowByteSize(elementCount, QuantizationType.PTQ1_0));
        Assert.Equal(expected, QuantizationType.PTQ1_0.ComputeByteCount(elementCount));
        Assert.Equal(expected, QuantFormat.TryGetInfo(QuantizationType.PTQ1_0)!.Value.RowByteSize(elementCount));
    }

    /// <summary>
    /// The first two blocks of <c>blk.0.attn_qkv.weight</c> in <c>Ternary-Bonsai-2-2B-PTQ1_0.gguf</c> (#1280). The fork's
    /// <c>dequantize_row_ptq1_0</c> output for them is <see cref="ForkDequantBits"/>.
    /// </summary>
    private static ReadOnlySpan<byte> ForkBlocks =>
    [
        0x8C, 0x15, 0x89, 0xBD, 0xE3, 0xF3, 0x93, 0x55, 0x04, 0x73, 0xE8, 0x3B, 0x9A, 0x60,
        0xB5, 0x06, 0x6D, 0xD7, 0x9A, 0x6D, 0x9F, 0x16, 0xCF, 0x1D, 0x39, 0xF7, 0xA8, 0x24,
        0xFA, 0xE1, 0x38, 0x3B, 0x02, 0xE1, 0x83, 0xEE, 0x85, 0x43, 0x11, 0x91, 0x2D, 0x50,
        0xBC, 0x29, 0xC9, 0xE6, 0x7D, 0x2E, 0xA8, 0x20, 0xCE, 0xDA, 0x92, 0x9F, 0x6A, 0x24,
    ];

    /// <summary>
    /// The fork's <c>dequantize_row_ptq1_0</c> output for <see cref="ForkBlocks"/> (256 floats as raw IEEE-754 bits),
    /// captured from PrismML-Eng/llama.cpp <c>prism</c> at <c>c96baf4a</c> (CPU build, <c>ggml-base.dll</c> via ctypes).
    /// Unlike <see cref="PTQ1_0ForkOracleTests"/> this needs neither the DLL nor the model, so it always runs.
    /// </summary>
    private static readonly uint[] ForkDequantBits =
    [
        0x00000000, 0xBC950000, 0x00000000, 0x3C950000, 0x3C950000, 0x3C950000, 0x00000000, 0xBC950000,
        0xBC950000, 0x00000000, 0x3C950000, 0xBC950000, 0x00000000, 0x00000000, 0x3C950000, 0xBC950000,
        0x00000000, 0xBC950000, 0x00000000, 0xBC950000, 0x00000000, 0x3C950000, 0x3C950000, 0x3C950000,
        0xBC950000, 0x00000000, 0x3C950000, 0x3C950000, 0x3C950000, 0xBC950000, 0xBC950000, 0xBC950000,
        0x3C950000, 0x3C950000, 0x3C950000, 0x00000000, 0x3C950000, 0x00000000, 0xBC950000, 0x3C950000,
        0xBC950000, 0xBC950000, 0xBC950000, 0xBC950000, 0x00000000, 0x00000000, 0x00000000, 0xBC950000,
        0x3C950000, 0xBC950000, 0x00000000, 0x3C950000, 0x3C950000, 0x00000000, 0x00000000, 0x3C950000,
        0x00000000, 0xBC950000, 0x00000000, 0xBC950000, 0xBC950000, 0xBC950000, 0xBC950000, 0x00000000,
        0xBC950000, 0x00000000, 0x00000000, 0x3C950000, 0x3C950000, 0x3C950000, 0x00000000, 0x3C950000,
        0xBC950000, 0x00000000, 0x00000000, 0x3C950000, 0x3C950000, 0x00000000, 0xBC950000, 0x3C950000,
        0x00000000, 0x3C950000, 0x00000000, 0x00000000, 0x00000000, 0xBC950000, 0x3C950000, 0xBC950000,
        0xBC950000, 0x00000000, 0x3C950000, 0xBC950000, 0x3C950000, 0xBC950000, 0x00000000, 0x00000000,
        0x3C950000, 0x00000000, 0x00000000, 0x3C950000, 0x00000000, 0x3C950000, 0xBC950000, 0xBC950000,
        0x00000000, 0x3C950000, 0xBC950000, 0x00000000, 0x3C950000, 0xBC950000, 0x3C950000, 0xBC950000,
        0x00000000, 0xBC950000, 0x3C950000, 0x00000000, 0xBC950000, 0x3C950000, 0x00000000, 0xBC950000,
        0xBC950000, 0x3C950000, 0x3C950000, 0x3C950000, 0xBC950000, 0x3C950000, 0xBC950000, 0xBC950000,
        0x3C8D4000, 0x3C8D4000, 0xBC8D4000, 0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0x00000000, 0x3C8D4000,
        0x00000000, 0xBC8D4000, 0xBC8D4000, 0x00000000, 0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0xBC8D4000,
        0x3C8D4000, 0x00000000, 0x00000000, 0x3C8D4000, 0xBC8D4000, 0x00000000, 0x00000000, 0x3C8D4000,
        0x00000000, 0x3C8D4000, 0xBC8D4000, 0x3C8D4000, 0x00000000, 0x3C8D4000, 0xBC8D4000, 0x00000000,
        0x3C8D4000, 0x3C8D4000, 0x3C8D4000, 0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0x00000000, 0x00000000,
        0x3C8D4000, 0x00000000, 0x00000000, 0xBC8D4000, 0x00000000, 0x3C8D4000, 0x00000000, 0x00000000,
        0x00000000, 0x3C8D4000, 0x3C8D4000, 0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0x3C8D4000, 0xBC8D4000,
        0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0xBC8D4000, 0x3C8D4000, 0x00000000, 0x3C8D4000, 0xBC8D4000,
        0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0x3C8D4000, 0x00000000, 0xBC8D4000, 0x00000000, 0xBC8D4000,
        0xBC8D4000, 0xBC8D4000, 0x00000000, 0x3C8D4000, 0xBC8D4000, 0xBC8D4000, 0x00000000, 0x3C8D4000,
        0x3C8D4000, 0x3C8D4000, 0x00000000, 0xBC8D4000, 0x00000000, 0xBC8D4000, 0x3C8D4000, 0x3C8D4000,
        0x00000000, 0x3C8D4000, 0x00000000, 0x00000000, 0x3C8D4000, 0x00000000, 0x00000000, 0x00000000,
        0xBC8D4000, 0xBC8D4000, 0x00000000, 0x00000000, 0x3C8D4000, 0xBC8D4000, 0xBC8D4000, 0x00000000,
        0xBC8D4000, 0xBC8D4000, 0xBC8D4000, 0x3C8D4000, 0x3C8D4000, 0x00000000, 0x3C8D4000, 0x3C8D4000,
        0x00000000, 0x3C8D4000, 0x00000000, 0x00000000, 0xBC8D4000, 0xBC8D4000, 0xBC8D4000, 0x3C8D4000,
        0x00000000, 0x00000000, 0x3C8D4000, 0x3C8D4000, 0xBC8D4000, 0x00000000, 0x00000000, 0x3C8D4000,
    ];

    [Fact]
    public unsafe void Dequantize_MatchesForkReferenceBits()
    {
        var ours = new float[2 * G];
        fixed (byte* p = ForkBlocks)
            Dequantize.ToFloat32((nint)p, ours.Length, QuantizationType.PTQ1_0, ours);

        for (int i = 0; i < ours.Length; i++)
            Assert.True(BitConverter.SingleToUInt32Bits(ours[i]) == ForkDequantBits[i],
                $"element {i}: ours=0x{BitConverter.SingleToUInt32Bits(ours[i]):X8}, fork=0x{ForkDequantBits[i]:X8}");
    }

    /// <summary>
    /// The SIMD group unpack (#1280) against the scalar transcription of the fork loop, over random blocks plus blocks
    /// holding every byte value, so each of the 256 codes passes through every trit position.
    /// </summary>
    [Fact]
    public unsafe void GroupUnpackI8_Simd_MatchesScalar()
    {
        var rng = new Random(1280);
        var block = new byte[B];
        var simd = new sbyte[G];
        var scalar = new sbyte[G];
        for (int iter = 0; iter < 64; iter++)
        {
            if (iter < 11)
                for (int i = 0; i < QsBytes + QhBytes; i++) block[i] = (byte)(iter * 26 + i); // covers 0..255
            else
                rng.NextBytes(block);

            fixed (byte* p = block)
            fixed (sbyte* a = simd)
            fixed (sbyte* s = scalar)
            {
                MatMul.UnpackPTQ1_0GroupI8ForTest(p, a);
                MatMul.UnpackPTQ1_0GroupI8Scalar(p, s);
            }

            Assert.Equal(scalar, simd);
        }
    }

    [Fact]
    public void BitsPerWeight_Is1Point75()
    {
        Assert.Equal(1.75, QuantFormat.TryGetInfo(QuantizationType.PTQ1_0)!.Value.BitsPerWeight);
    }

    /// <summary>
    /// The scale lives at the END of the block. Packing a group whose weights are all +1 and then
    /// moving the scale to the front (PQ2_0's position) must change the decode — a test that only
    /// checked "values are ternary times something" would pass either way.
    /// </summary>
    [Fact]
    public unsafe void Scale_IsReadFromTheEndOfTheBlock_NotTheStart()
    {
        var ternary = new sbyte[G];
        Array.Fill(ternary, (sbyte)1);
        const float Scale = 0.25f;

        byte[] block = PackGroup(ternary, Scale);
        Assert.Equal(B, block.Length);

        var dest = new float[G];
        fixed (byte* p = block)
            Dequantize.ToFloat32((nint)p, G, QuantizationType.PTQ1_0, dest);

        for (int i = 0; i < G; i++)
            Assert.Equal(Scale, dest[i], 1e-4f);

        // Reading the Half from offset 0 (PQ2_0's convention) would pick up qs[0..1], which for an
        // all-+1 group is a large-magnitude payload byte pair, not 0.25.
        float wrong = (float)BitConverter.ToHalf(block.AsSpan(0, 2));
        Assert.NotEqual(Scale, wrong, 1e-4f);
    }

    /// <summary>
    /// Round-trips a random ternary vector through the fork's packing and back. This is what
    /// pins element ORDER: the base-3 layout is trit-major within a byte chunk, so any transposed
    /// or strided traversal scrambles positions while leaving the multiset of values untouched —
    /// exactly the failure mode that let a wrong PQ2_0 bit convention (#269) and a transposed Q3_K
    /// dequant both survive statistical checks.
    /// </summary>
    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(1280)]
    public unsafe void Dequantize_RoundTripsTheForkPacking_Exactly(int k)
    {
        var rng = new Random(20260919);
        var ternary = new sbyte[k];
        for (int i = 0; i < k; i++)
            ternary[i] = (sbyte)(rng.Next(3) - 1);

        float[] scales = new float[k / G];
        for (int g = 0; g < scales.Length; g++)
            scales[g] = (float)(Half)(0.05f + 0.01f * g);

        byte[] packed = PackRow(ternary, scales);
        var dest = new float[k];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, k, QuantizationType.PTQ1_0, dest);

        for (int i = 0; i < k; i++)
            Assert.Equal(ternary[i] * scales[i / G], dest[i], 1e-6f);
    }

    /// <summary>
    /// GEMV through the public entry point, which takes the W2A8 int8-activation tier on AVX2
    /// hardware — so this covers that tier's per-128-group scale folding, the part shared with
    /// PQ2_0's <c>VecDotPQ2_0Q8</c>.
    ///
    /// <para>The bound is derived rather than guessed. W2A8 quantizes the <i>activation</i> to
    /// Q8_0, so each product carries an error of about <c>|w| · xAbsMax / 254</c>, and over K
    /// independent terms those accumulate as a random walk: <c>√K · wRms · xAbsMax / 254</c>.
    /// Four of those is the tolerance. It is not lax: a scrambled element order would land the
    /// result a full <c>√K · wRms · xRms</c> away — on the K=1280 shape roughly 35× the bound —
    /// so this still discriminates the failure mode that matters. A fixed absolute tolerance was
    /// tried first and is simply wrong here, because the error grows with both K and the spread
    /// of the per-group scales.</para>
    /// </summary>
    [Theory]
    [InlineData(4, 128)]
    [InlineData(16, 512)]
    [InlineData(33, 1280)]
    public unsafe void GemvPTQ1_0_MatchesDequantizedDenseReference(int m, int k)
    {
        var rng = new Random(4361);
        (byte[] packed, float[] dense) = BuildRandomMatrix(rng, m, k);

        var x = new float[k];
        float xAbsMax = 0f;
        for (int i = 0; i < k; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            xAbsMax = MathF.Max(xAbsMax, MathF.Abs(x[i]));
        }

        var got = new float[m];
        fixed (byte* w = packed)
        fixed (float* xp = x)
        fixed (float* rp = got)
            MatMul.GemvPTQ1_0(w, xp, rp, m, k, null);

        for (int r = 0; r < m; r++)
        {
            float expected = 0f;
            double sumSq = 0.0;
            for (int i = 0; i < k; i++)
            {
                float wv = dense[(long)r * k + i];
                expected += wv * x[i];
                sumSq += (double)wv * wv;
            }

            float wRms = (float)Math.Sqrt(sumSq / k);
            float tol = 1e-3f + 4f * MathF.Sqrt(k) * wRms * (xAbsMax / 254f);
            Assert.True(MathF.Abs(got[r] - expected) <= tol,
                $"row {r}: got {got[r]}, expected {expected} (tol {tol})");
        }
    }

    /// <summary>
    /// The scalar float tier alone, held to a much tighter bound than
    /// <see cref="GemvPTQ1_0_MatchesDequantizedDenseReference"/>: it does no activation
    /// quantization, so it should differ from the dense reference only by summation order.
    /// </summary>
    [Fact]
    public unsafe void GemvPTQ1_0Scalar_MatchesDequantizedDenseReference_Tightly()
    {
        const int M = 8, K = 512;
        var rng = new Random(9114);
        (byte[] packed, float[] dense) = BuildRandomMatrix(rng, M, K);

        var x = new float[K];
        for (int i = 0; i < K; i++)
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var got = new float[M];
        fixed (byte* w = packed)
        fixed (float* xp = x)
        fixed (float* rp = got)
            MatMul.GemvPTQ1_0Scalar(w, xp, rp, M, K);

        for (int r = 0; r < M; r++)
        {
            float expected = 0f;
            for (int i = 0; i < K; i++)
                expected += dense[(long)r * K + i] * x[i];

            Assert.Equal(expected, got[r], 1e-3f);
        }
    }

    /// <summary>GEMM (multi-token) against the same dense reference.</summary>
    [Theory]
    [InlineData(6, 256, 3)]
    [InlineData(20, 384, 5)]
    public unsafe void GemmPTQ1_0_MatchesDequantizedDenseReference(int m, int k, int n)
    {
        var rng = new Random(771);
        (byte[] packed, float[] dense) = BuildRandomMatrix(rng, m, k);

        var b = new float[(long)n * k];
        for (int i = 0; i < b.Length; i++)
            b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var got = new float[(long)n * m];
        fixed (byte* w = packed)
        fixed (float* bp = b)
        fixed (float* cp = got)
            MatMul.GemmPTQ1_0(w, bp, cp, m, k, n, null);

        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                float expected = 0f;
                for (int i = 0; i < k; i++)
                    expected += dense[(long)r * k + i] * b[(long)t * k + i];

                Assert.Equal(expected, got[(long)t * m + r], 1e-3f);
            }
    }

    // ── fixtures: a local port of the fork's quantize_row_ptq1_0_ref ──

    private static (byte[] Packed, float[] Dense) BuildRandomMatrix(Random rng, int m, int k)
    {
        int groupsPerRow = k / G;
        var packed = new byte[(long)m * groupsPerRow * B];
        var dense = new float[(long)m * k];

        var ternary = new sbyte[k];
        var scales = new float[groupsPerRow];

        for (int r = 0; r < m; r++)
        {
            for (int i = 0; i < k; i++)
                ternary[i] = (sbyte)(rng.Next(3) - 1);
            for (int g = 0; g < groupsPerRow; g++)
                scales[g] = (float)(Half)(0.02f + 0.3f * (float)rng.NextDouble());

            byte[] row = PackRow(ternary, scales);
            row.CopyTo(packed, (long)r * groupsPerRow * B);

            for (int i = 0; i < k; i++)
                dense[(long)r * k + i] = ternary[i] * scales[i / G];
        }

        return (packed, dense);
    }

    private static byte[] PackRow(ReadOnlySpan<sbyte> ternary, ReadOnlySpan<float> scales)
    {
        int groups = ternary.Length / G;
        var result = new byte[groups * B];
        for (int g = 0; g < groups; g++)
            PackGroup(ternary.Slice(g * G, G), scales[g]).CopyTo(result, g * B);
        return result;
    }

    /// <summary>
    /// Transcription of <c>quantize_row_ptq1_0_ref</c> (fork, <c>ggml/src/ggml-quants.c</c>) for a
    /// single 128-weight group, with the scale supplied rather than derived from an absmax. Note
    /// the ceiling-division encode (<c>(q·256 + 242) / 243</c>) and, for <c>qh</c>, the extra
    /// <c>q *= 3</c> that shifts four trits into the five-trit field.
    /// </summary>
    private static byte[] PackGroup(ReadOnlySpan<sbyte> ternary, float scale)
    {
        var block = new byte[B];
        int[] stages = [32, 16, 8];

        int src = 0;
        int j = 0;
        foreach (int c in stages)
        {
            while (j + c <= QsBytes)
            {
                for (int mm = 0; mm < c; mm++)
                {
                    int q = 0;
                    for (int nn = 0; nn < 5; nn++)
                        q = q * 3 + (ternary[src + mm + nn * c] + 1);

                    block[j + mm] = (byte)((q * 256 + 242) / 243);
                }

                src += 5 * c;
                j += c;
            }
        }

        for (int h = 0; h < QhBytes; h++)
        {
            int q = 0;
            for (int mm = 0; mm < 4; mm++)
                q = q * 3 + (ternary[src + h + mm * QhBytes] + 1);

            q *= 3;
            block[QsBytes + h] = (byte)((q * 256 + 242) / 243);
        }

        BitConverter.TryWriteBytes(block.AsSpan(QsBytes + QhBytes, 2), (Half)scale);
        return block;
    }
}
