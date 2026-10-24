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

    public static TheoryData<int, int> Cases()
    {
        var d = new TheoryData<int, int>();
        foreach (int fmt in new[] { 4, 5, 6 })
            foreach (int sb in new[] { 1, 2, 6, 7 })
                d.Add(fmt, sb);
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void FastKernels_MatchReference_BitExact(int fmt, int sbCount)
    {
        Skip.IfNot(MatMul.FastKQuantIsa, "fast K-quant kernels need AVX2 + SSE4.1");
        bool q6 = fmt == 6, q5 = fmt == 5;
        int blockBytes = q6 ? 210 : q5 ? 176 : 144;
        int k = sbCount * 256;
        const int tokens = 4;
        var rng = new Random(933 + sbCount + fmt * 100);

        // One 4-row group over row-major rows (#1156): super-block s of row r lives at (r * sbCount + s) * blockBytes.
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

            int wStride = blockBytes;
            long rs = (long)sbCount * blockBytes;
            int q8Row = sbCount * Q8KBytes;
            for (int t = 0; t < tokens; t++)
            {
                byte* g = w;
                if (q6)
                    MatMul.VecDotQ6_K_Q8_K_4RowsReference(g, g + rs, g + 2 * rs, g + 3 * rs,
                        x + (long)t * q8Row, sbCount, wStride, expected + t * 4);
                else if (q5)
                    MatMul.VecDotQ5_K_Q8_K_4RowsReference(g, g + rs, g + 2 * rs, g + 3 * rs,
                        x + (long)t * q8Row, sbCount, wStride, expected + t * 4);
                else
                    MatMul.VecDotQ4_K_Q8_K_4RowsReference(g, g + rs, g + 2 * rs, g + 3 * rs,
                        x + (long)t * q8Row, sbCount, wStride, expected + t * 4);
            }

            // (1) 4 rows x 1 token.
            for (int t = 0; t < tokens; t++)
            {
                byte* g = w;
                if (q6)
                    MatMul.Q6KRows4Tokens1(g, g + rs, g + 2 * rs, g + 3 * rs, x + (long)t * q8Row, sbCount, wStride, actual + t * 4);
                else if (q5)
                    MatMul.Q5KRows4Tokens1(g, g + rs, g + 2 * rs, g + 3 * rs, x + (long)t * q8Row, sbCount, wStride, actual + t * 4);
                else
                    MatMul.Q4KRows4Tokens1(g, g + rs, g + 2 * rs, g + 3 * rs, x + (long)t * q8Row, sbCount, wStride, actual + t * 4);
            }
            AssertBits(expected, actual, tokens * 4, $"Q{fmt}_K sb={sbCount} 1-token kernel");

            // (2) 4 rows x 4 tokens; output laid out [token][row] with ldc = 4.
            new Span<float>(actual, tokens * 4).Clear();
            if (q6)
                MatMul.Q6KRows4Tokens4(w, sbCount, x, q8Row, actual, 4);
            else if (q5)
                MatMul.Q5KRows4Tokens4(w, sbCount, x, q8Row, actual, 4);
            else
                MatMul.Rows4Tokens4(w, sbCount, x, q8Row, actual, 4);
            AssertBits(expected, actual, tokens * 4, $"Q{fmt}_K sb={sbCount} 4-token kernel");
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
    [SkippableTheory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void RowMajorPooledGemm_FourTokenPath_MatchesSerial_BitExact(int fmt)
    {
        Skip.IfNot(MatMul.FastKQuantIsa, "fast K-quant kernels need AVX2 + SSE4.1");
        bool q6 = fmt == 6, q5 = fmt == 5;
        const int m = 250, k = 4096, n = 9;
        int sb = k / 256;
        int blockBytes = q6 ? 210 : q5 ? 176 : 144;
        var rng = new Random(41 + fmt);
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
            else if (q5)
            {
                MatMul.GemmQ5_K(w, x, serial, m, k, n, null, null);
                MatMul.GemmQ5_K(w, x, pooled, m, k, n, pool, null);
            }
            else
            {
                MatMul.GemmQ4_K(w, x, serial, m, k, n, null, null);
                MatMul.GemmQ4_K(w, x, pooled, m, k, n, pool, null);
            }

            AssertBits(serial, pooled, n * m, $"Q{fmt}_K pooled row-major GEMM");
        }
        finally
        {
            NativeMemory.AlignedFree(w);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(serial);
            NativeMemory.AlignedFree(pooled);
        }
    }

    /// <summary>
    /// With the fast ISA forced off (what a non-AVX2 / non-x86 host sees) every entry point and dispatcher must
    /// fall back to the reference kernels and never throw PlatformNotSupportedException. Runs on every host.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void FastIsaForcedOff_EntryPointsFallBackToReference(int fmt)
    {
        bool q6 = fmt == 6, q5 = fmt == 5;
        int bb = q6 ? 210 : q5 ? 176 : 144;
        const int sb = 2, tokens = 4;
        int k = sb * 256, q8Row = sb * Q8KBytes;
        var rng = new Random(77 + fmt);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(4L * sb * bb), 64);
        byte* x = (byte*)NativeMemory.AlignedAlloc((nuint)((long)tokens * q8Row), 64);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        float* exp = (float*)NativeMemory.AlignedAlloc(tokens * 4 * 4, 64);
        float* act = (float*)NativeMemory.AlignedAlloc(tokens * 4 * 4, 64);
        bool prev = MatMul.ForceDisableFastKQuant;
        try
        {
            for (int i = 0; i < 4 * sb * bb; i++) w[i] = (byte)rng.Next(256);
            for (int b = 0; b < 4 * sb; b++)
            {
                byte* blk = w + (long)b * bb;
                if (q6) *(Half*)(blk + 208) = (Half)0.01f;
                else { *(Half*)blk = (Half)0.01f; *(Half*)(blk + 2) = (Half)0.005f; }
            }
            for (int t = 0; t < tokens; t++)
            {
                for (int i = 0; i < k; i++) src[i] = (float)(rng.NextDouble() - 0.5);
                MatMul.QuantizeF32ToQ8_K(src, x + (long)t * q8Row, k);
            }

            int stride = bb;
            long rs = (long)sb * bb;
            for (int t = 0; t < tokens; t++)
            {
                byte* q = x + (long)t * q8Row;
                if (q6) MatMul.VecDotQ6_K_Q8_K_4RowsReference(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, exp + t * 4);
                else if (q5) MatMul.VecDotQ5_K_Q8_K_4RowsReference(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, exp + t * 4);
                else MatMul.VecDotQ4_K_Q8_K_4RowsReference(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, exp + t * 4);
            }

            MatMul.ForceDisableFastKQuant = true;
            Assert.False(MatMul.FastKQuantIsa);

            // 4-token entry point.
            new Span<float>(act, tokens * 4).Clear();
            if (q6) MatMul.Q6KRows4Tokens4(w, sb, x, q8Row, act, 4);
            else if (q5) MatMul.Q5KRows4Tokens4(w, sb, x, q8Row, act, 4);
            else MatMul.Rows4Tokens4(w, sb, x, q8Row, act, 4);
            AssertBits(exp, act, tokens * 4, $"Q{fmt}_K forced-off 4-token entry point");

            // 1-token entry point and the public dispatcher.
            for (int pass = 0; pass < 2; pass++)
            {
                new Span<float>(act, tokens * 4).Clear();
                for (int t = 0; t < tokens; t++)
                {
                    byte* q = x + (long)t * q8Row;
                    if (pass == 0)
                    {
                        if (q6) MatMul.Q6KRows4Tokens1(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, act + t * 4);
                        else if (q5) MatMul.Q5KRows4Tokens1(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, act + t * 4);
                        else MatMul.Q4KRows4Tokens1(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, act + t * 4);
                    }
                    else
                    {
                        if (q6) MatMul.VecDotQ6_K_Q8_K_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, act + t * 4);
                        else if (q5) MatMul.VecDotQ5_K_Q8_K_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, act + t * 4);
                        else MatMul.VecDotQ4_K_Q8_K_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, q, sb, stride, act + t * 4);
                    }
                }
                AssertBits(exp, act, tokens * 4, $"Q{fmt}_K forced-off {(pass == 0 ? "1-token entry point" : "dispatcher")}");
            }
        }
        finally
        {
            MatMul.ForceDisableFastKQuant = prev;
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(src);
            NativeMemory.AlignedFree(exp); NativeMemory.AlignedFree(act);
        }
    }

    private static void AssertBits(float* expected, float* actual, int count, string what)
    {
        for (int i = 0; i < count; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}: element {i} expected {expected[i]:R} actual {actual[i]:R}");
    }
}
