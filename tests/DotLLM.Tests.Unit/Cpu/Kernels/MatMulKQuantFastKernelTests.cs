using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #933. The Q4_K / Q6_K x Q8_K fast kernels (hoisted scale unpack, vectorised min/bias correction,
/// 4-token weight reuse) must be bit-identical to the original 4-row kernels
/// (<c>VecDotQ*_K_Q8_K_4RowsReference</c>) - the integer reduction is exact and the per-element float
/// accumulation order is unchanged, so no tolerance is allowed. The 4-token form is checked token by token
/// against the single-token reference, which is what makes a token's logits independent of its batch-mates (#530).
/// </summary>
public sealed unsafe class MatMulKQuantFastKernelTests
{
    private const int Q8KBytes = 292;

    public static TheoryData<bool, int> Cases()
    {
        var d = new TheoryData<bool, int>();
        foreach (bool q6 in new[] { false, true })
            foreach (int sb in new[] { 1, 2, 6, 7 })
                d.Add(q6, sb);
        return d;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FastKernels_MatchReference_BitExact(bool q6, int sbCount)
    {
        int blockBytes = q6 ? 210 : 144;
        int k = sbCount * 256;
        const int tokens = 4;
        var rng = new Random(933 + sbCount + (q6 ? 100 : 0));

        // One R4 group (4 rows): super-block s of row r lives at (s * 4 + r) * blockBytes.
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(4L * sbCount * blockBytes), 64);
        byte* x = (byte*)NativeMemory.AlignedAlloc((nuint)((long)tokens * sbCount * Q8KBytes), 64);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        float* expected = (float*)NativeMemory.AlignedAlloc((nuint)(tokens * 4 * 4), 64);
        float* actual = (float*)NativeMemory.AlignedAlloc((nuint)(tokens * 4 * 4), 64);
        try
        {
            for (long i = 0; i < 4L * sbCount * blockBytes; i++)
                w[i] = (byte)rng.Next(256);
            for (int b = 0; b < 4 * sbCount; b++)
            {
                byte* blk = w + (long)b * blockBytes;
                if (q6)
                {
                    *(Half*)(blk + 208) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                    for (int s = 0; s < 16; s++)
                        ((sbyte*)(blk + 192))[s] = (sbyte)rng.Next(-128, 128);
                }
                else
                {
                    *(Half*)blk = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                    *(Half*)(blk + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                }
            }

            for (int t = 0; t < tokens; t++)
            {
                for (int i = 0; i < k; i++)
                    src[i] = (float)((rng.NextDouble() - 0.5) * (t == 3 ? 40 : 4));
                MatMul.QuantizeF32ToQ8_K(src, x + (long)t * sbCount * Q8KBytes, k);
            }

            int wStride = 4 * blockBytes;
            int q8Row = sbCount * Q8KBytes;
            for (int t = 0; t < tokens; t++)
            {
                byte* g = w;
                if (q6)
                    MatMul.VecDotQ6_K_Q8_K_4RowsReference(g, g + 210, g + 420, g + 630,
                        x + (long)t * q8Row, sbCount, wStride, expected + t * 4);
                else
                    MatMul.VecDotQ4_K_Q8_K_4RowsReference(g, g + 144, g + 288, g + 432,
                        x + (long)t * q8Row, sbCount, wStride, expected + t * 4);
            }

            // (1) 4 rows x 1 token.
            for (int t = 0; t < tokens; t++)
            {
                byte* g = w;
                if (q6)
                    MatMul.Q6KRows4Tokens1(g, g + 210, g + 420, g + 630, x + (long)t * q8Row, sbCount, wStride, actual + t * 4);
                else
                    MatMul.Q4KRows4Tokens1(g, g + 144, g + 288, g + 432, x + (long)t * q8Row, sbCount, wStride, actual + t * 4);
            }
            AssertBits(expected, actual, tokens * 4, $"{(q6 ? "Q6_K" : "Q4_K")} sb={sbCount} 1-token kernel");

            // (2) 4 rows x 4 tokens; output laid out [token][row] with ldc = 4.
            new Span<float>(actual, tokens * 4).Clear();
            if (q6)
                MatMul.Q6KRows4Tokens4(w, sbCount, x, q8Row, actual, 4);
            else
                MatMul.Rows4Tokens4(w, sbCount, x, q8Row, actual, 4);
            AssertBits(expected, actual, tokens * 4, $"{(q6 ? "Q6_K" : "Q4_K")} sb={sbCount} 4-token kernel");
        }
        finally
        {
            NativeMemory.AlignedFree(w);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(src);
            NativeMemory.AlignedFree(expected);
            NativeMemory.AlignedFree(actual);
        }
    }

    /// <summary>
    /// The row-major pooled tiled GEMM (no repack: the default when RAM is tight) takes the 4-token path for
    /// whole 4-row groups. M = 250 forces three tiles and a 2-row remainder; n = 9 forces two 4-token blocks
    /// plus a single-token tail. Must equal the serial per-token result bit for bit.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RowMajorPooledGemm_FourTokenPath_MatchesSerial_BitExact(bool q6)
    {
        const int m = 250, k = 4096, n = 9;
        int sb = k / 256;
        int blockBytes = q6 ? 210 : 144;
        var rng = new Random(q6 ? 61 : 41);
        long wBytes = (long)m * sb * blockBytes;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * 4), 64);
        float* serial = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        float* pooled = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        try
        {
            for (long i = 0; i < wBytes; i++) w[i] = (byte)rng.Next(256);
            for (int b = 0; b < m * sb; b++)
            {
                byte* blk = w + (long)b * blockBytes;
                if (q6) { *(Half*)(blk + 208) = (Half)0.01f; }
                else { *(Half*)blk = (Half)0.01f; *(Half*)(blk + 2) = (Half)0.005f; }
            }
            for (int i = 0; i < n * k; i++) x[i] = (float)(rng.NextDouble() - 0.5);

            using var pool = new DotLLM.Cpu.Threading.ComputeThreadPool(4);
            if (q6)
            {
                MatMul.GemmQ6_K(w, x, serial, m, k, n, null, null);
                MatMul.GemmQ6_K(w, x, pooled, m, k, n, pool, null);
            }
            else
            {
                MatMul.GemmQ4_K(w, x, serial, m, k, n, null, null);
                MatMul.GemmQ4_K(w, x, pooled, m, k, n, pool, null);
            }

            AssertBits(serial, pooled, n * m, q6 ? "Q6_K pooled row-major GEMM" : "Q4_K pooled row-major GEMM");
        }
        finally
        {
            NativeMemory.AlignedFree(w);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(serial);
            NativeMemory.AlignedFree(pooled);
        }
    }

    private static void AssertBits(float* expected, float* actual, int count, string what)
    {
        for (int i = 0; i < count; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}: element {i} expected {expected[i]:R} actual {actual[i]:R}");
    }
}
