using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477: parity of the 128-bit (SSE2/SSSE3) ternary W2A8 tier (<c>MatMul.W2A8Sse.cs</c>) —
/// the path PQ2_0 and I2_S take on Westmere-class hardware — against scalar references, run on
/// whatever box executes the suite. SSSE3 is a strict subset of every AVX2 machine's ISA, so the
/// 128-bit kernels are called directly here and execute natively even on an AVX-512 dev box; no
/// ISA emulation is needed to verify them.
///
/// <para>Every comparison is <b>tight</b> (byte-exact for unpacks; ~1e-6 relative for dots, whose
/// integer block sums are exact and only the float cross-block summation order can differ). The
/// existing quant-tolerance tests (2% relative vs a float reference) are far too loose to see a
/// lane-order, sign-operand or group-scale-index bug; these are the discriminating tests.</para>
///
/// <para>Unpack inputs are <b>random packed bytes</b>, not the ternary test packer, so code 3
/// (which PQ2_0 decodes to +2 and the packer never emits) is exercised too.</para>
///
/// <para>To exercise the whole dispatch (drivers + 128-bit tier) rather than just the kernels, run
/// the suite with <c>DOTNET_EnableAVX=0</c>: the runtime then reports AVX/AVX2/FMA/AVX-512 as
/// unsupported while SSE2..SSE4.2 stay on, which is exactly Westmere's ISA, and the
/// <c>*_Dispatch_*</c> tests below take the 128-bit tier.</para>
/// </summary>
public sealed unsafe class W2A8SseTierTests
{
    private const int Q8Block = 32;
    private const int Q8BlockBytes = 34;

    // ──────────────────── Unpacks (byte-exact) ────────────────────

    [SkippableTheory]
    [InlineData(128)]
    [InlineData(384)]
    [InlineData(640)]
    [InlineData(5120)]
    public void UnpackPQ2_0RowI8_Sse_MatchesScalar_ByteExact(int k)
    {
        Skip.IfNot(Sse2.IsSupported, "needs SSE2");
        var rng = new Random(477_000 + k);
        int groups = k / 128;
        byte* row = RandomPQ2_0Row(rng, groups);
        sbyte* expected = (sbyte*)NativeMemory.Alloc((nuint)k);
        sbyte* actual = (sbyte*)NativeMemory.Alloc((nuint)k);
        float* gsExpected = (float*)NativeMemory.Alloc((nuint)(groups * sizeof(float)));
        float* gsActual = (float*)NativeMemory.Alloc((nuint)(groups * sizeof(float)));
        try
        {
            MatMul.UnpackPQ2_0RowI8Scalar(row, expected, gsExpected, k);
            MatMul.UnpackPQ2_0RowI8Sse(row, actual, gsActual, k);
            AssertBytesEqual(expected, actual, k);
            for (int g = 0; g < groups; g++) Assert.Equal(gsExpected[g], gsActual[g]);

            if (Avx2.IsSupported)
            {
                // The pre-existing AVX2 unpack must agree with the same scalar reference.
                MatMul.UnpackPQ2_0RowI8Avx2(row, actual, gsActual, k);
                AssertBytesEqual(expected, actual, k);
            }
        }
        finally
        {
            NativeMemory.Free(row); NativeMemory.Free(expected); NativeMemory.Free(actual);
            NativeMemory.Free(gsExpected); NativeMemory.Free(gsActual);
        }
    }

    [SkippableTheory]
    [InlineData(128)]
    [InlineData(384)]
    [InlineData(2560)]
    [InlineData(6912)]
    public void UnpackI2SRowI8_Sse_MatchesScalar_ByteExact(int k)
    {
        Skip.IfNot(Sse2.IsSupported, "needs SSE2");
        var rng = new Random(477_100 + k);
        int bytes = k / 4;
        byte* row = (byte*)NativeMemory.Alloc((nuint)bytes);
        sbyte* expected = (sbyte*)NativeMemory.Alloc((nuint)k);
        sbyte* actual = (sbyte*)NativeMemory.Alloc((nuint)k);
        try
        {
            for (int i = 0; i < bytes; i++) row[i] = (byte)rng.Next(256);
            MatMul.UnpackRowI8Scalar(row, expected, k);
            MatMul.UnpackRowI8Sse(row, actual, k);
            AssertBytesEqual(expected, actual, k);

            if (Avx2.IsSupported)
            {
                MatMul.UnpackRowI8Avx2(row, actual, k);
                AssertBytesEqual(expected, actual, k);
            }
        }
        finally { NativeMemory.Free(row); NativeMemory.Free(expected); NativeMemory.Free(actual); }
    }

    // ──────────────────── Dots (exact-integer reference) ────────────────────

    [SkippableTheory]
    [InlineData(1)]     // one 128-group
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(40)]    // Bonsai-27B hidden (5120)
    [InlineData(136)]   // Bonsai-27B ffn (17408)
    public void VecDotPQ2_0Q8_Sse_MatchesExactReference(int groups)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(477_200 + groups);
        int k = groups * 128;
        int blocks = k / Q8Block;

        sbyte* w = RandomCodes(rng, k, maxCode: 2);        // {-1,0,+1,+2}
        float* gs = (float*)NativeMemory.Alloc((nuint)(groups * sizeof(float)));
        byte* xQ8 = RandomQ8(rng, k);
        float* xs = ScalesOf(xQ8, blocks);
        try
        {
            for (int g = 0; g < groups; g++) gs[g] = 0.005f + rng.NextSingle() * 0.08f;

            (double exact, double mag) = ReferencePQ2_0(w, gs, xQ8, blocks);
            float sse = MatMul.VecDotPQ2_0Q8Sse(w, gs, xQ8, xs, blocks);
            AssertTight(exact, mag, sse, "SSE vs exact");

            if (Avx2.IsSupported)
            {
                float avx2 = MatMul.VecDotPQ2_0Q8Avx2(w, gs, xQ8, xs, blocks);
                AssertTight(exact, mag, avx2, "AVX2 vs exact");
                AssertTight(avx2, mag, sse, "SSE vs AVX2");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(gs); NativeMemory.Free(xQ8); NativeMemory.Free(xs); }
    }

    [SkippableTheory]
    [InlineData(1)]     // single Q8_0 block — odd, below any 4-block unroll
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(80)]    // BitNet-2B hidden (2560)
    [InlineData(216)]   // BitNet-2B ffn (6912)
    public void VecDotI2SQ8_Sse_MatchesExactReference(int blocks)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(477_300 + blocks);
        int k = blocks * Q8Block;

        sbyte* w = RandomCodes(rng, k, maxCode: 1);        // {-1,0,+1}
        byte* xQ8 = RandomQ8(rng, k);
        float* xs = ScalesOf(xQ8, blocks);
        try
        {
            (double exact, double mag) = ReferenceI2S(w, xQ8, blocks);
            float sse = MatMul.VecDotI2SQ8Sse(w, xQ8, xs, blocks);
            AssertTight(exact, mag, sse, "SSE vs exact");

            if (Avx2.IsSupported)
            {
                float avx2 = MatMul.VecDotI2SQ8Avx2(w, xQ8, xs, blocks);
                AssertTight(exact, mag, avx2, "AVX2 vs exact");
                AssertTight(avx2, mag, sse, "SSE vs AVX2");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(xQ8); NativeMemory.Free(xs); }
    }

    /// <summary>
    /// Saturation guard: all-+2 PQ2_0 weights against all-±127 activations is the largest possible
    /// PMADDUBSW pair sum (2·127·2 = 508), well inside int16. Must still be exact.
    /// </summary>
    [SkippableFact]
    public void VecDotPQ2_0Q8_Sse_ExtremeOperands_Exact()
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        const int groups = 2;
        const int k = groups * 128;
        const int blocks = k / Q8Block;
        sbyte* w = (sbyte*)NativeMemory.Alloc(k);
        float* gs = (float*)NativeMemory.Alloc(groups * sizeof(float));
        byte* xQ8 = (byte*)NativeMemory.Alloc(blocks * Q8BlockBytes);
        try
        {
            for (int i = 0; i < k; i++) w[i] = (sbyte)(i % 3 == 0 ? -1 : 2);
            gs[0] = 1f; gs[1] = 0.5f;
            for (int b = 0; b < blocks; b++)
            {
                *(Half*)(xQ8 + b * Q8BlockBytes) = (Half)1f;
                sbyte* q = (sbyte*)(xQ8 + b * Q8BlockBytes + 2);
                for (int i = 0; i < Q8Block; i++) q[i] = (sbyte)(i % 2 == 0 ? 127 : -127);
            }
            float* xs = ScalesOf(xQ8, blocks);
            (double exact, double mag) = ReferencePQ2_0(w, gs, xQ8, blocks);
            AssertTight(exact, mag, MatMul.VecDotPQ2_0Q8Sse(w, gs, xQ8, xs, blocks), "extreme");
            NativeMemory.Free(xs);
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(gs); NativeMemory.Free(xQ8); }
    }

    // ──────────────────── Whole dispatch (tier-agnostic) ────────────────────

    /// <summary>
    /// The public GEMV/GEMM entries vs an exact W2A8 reference built from the <b>same</b> Q8_0
    /// activation quantization. Validates whichever SIMD tier the process dispatches to: 256-bit
    /// normally, 128-bit under <c>DOTNET_EnableAVX=0</c> (or on real pre-AVX2 hardware).
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 128, 1)]
    [InlineData(3, 384, 1)]
    [InlineData(7, 640, 1)]
    [InlineData(17, 1280, 1)]   // >= ParallelMinRows-style row counts with an odd remainder
    [InlineData(5, 384, 3)]     // GEMM via the test-only W2A8 GEMM entry
    [InlineData(9, 256, 4)]
    public void PQ2_0_Dispatch_MatchesExactW2A8Reference(int m, int k, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_PQ2_W2A8") is "0" or "false" or "off",
            "W2A8 disabled by DOTLLM_PQ2_W2A8");
        var rng = new Random(477_400 + m * 31 + k + n);
        int groups = k / 128, rowBytes = groups * 34, blocks = k / Q8Block;

        byte* weights = (byte*)NativeMemory.Alloc((nuint)(m * rowBytes));
        float* b = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* c = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        byte* bQ8 = (byte*)NativeMemory.Alloc((nuint)(n * blocks * Q8BlockBytes));
        sbyte* wI8 = (sbyte*)NativeMemory.Alloc((nuint)k);
        float* gs = (float*)NativeMemory.Alloc((nuint)(groups * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = RandomPQ2_0Row(rng, groups);
                Buffer.MemoryCopy(row, weights + r * rowBytes, rowBytes, rowBytes);
                NativeMemory.Free(row);
            }
            for (int i = 0; i < n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;
            for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_0(b + t * k, bQ8 + t * blocks * Q8BlockBytes, k);

            if (n == 1) MatMul.GemvPQ2_0(weights, b, c, m, k, null);
            else MatMul.GemmPQ2_0W2A8ForTest(weights, b, c, m, k, n);

            for (int r = 0; r < m; r++)
            {
                MatMul.UnpackPQ2_0RowI8Scalar(weights + r * rowBytes, wI8, gs, k);
                for (int t = 0; t < n; t++)
                {
                    (double exact, double mag) = ReferencePQ2_0(wI8, gs, bQ8 + t * blocks * Q8BlockBytes, blocks);
                    AssertTight(exact, mag, c[t * m + r], $"row {r} token {t}");
                }
            }
        }
        finally
        {
            NativeMemory.Free(weights); NativeMemory.Free(b); NativeMemory.Free(c);
            NativeMemory.Free(bQ8); NativeMemory.Free(wI8); NativeMemory.Free(gs);
        }
    }

    /// <summary>I2_S analog of <see cref="PQ2_0_Dispatch_MatchesExactW2A8Reference"/>.</summary>
    [SkippableTheory]
    [InlineData(1, 128, 1)]
    [InlineData(5, 384, 1)]
    [InlineData(17, 1280, 1)]
    [InlineData(6, 256, 3)]
    [InlineData(9, 384, 5)]     // exercises the 4x4 tile + remainders on AVX2, per-cell on SSE
    public void I2S_Dispatch_MatchesExactW2A8Reference(int m, int k, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_I2S_W2A8") is "0" or "false" or "off",
            "W2A8 disabled by DOTLLM_I2S_W2A8");
        var rng = new Random(477_500 + m * 31 + k + n);
        int rowBytes = k / 4, blocks = k / Q8Block;
        const float scale = 0.037f;

        byte* weights = (byte*)NativeMemory.Alloc((nuint)(m * rowBytes + 4));
        float* b = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* c = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        byte* bQ8 = (byte*)NativeMemory.Alloc((nuint)(n * blocks * Q8BlockBytes));
        sbyte* wI8 = (sbyte*)NativeMemory.Alloc((nuint)k);
        try
        {
            // Codes 0..2 only (I2_S is strictly ternary); per-tensor scale at the tail.
            for (int i = 0; i < m * rowBytes; i++)
                weights[i] = (byte)(rng.Next(3) | (rng.Next(3) << 2) | (rng.Next(3) << 4) | (rng.Next(3) << 6));
            *(float*)(weights + m * rowBytes) = scale;
            for (int i = 0; i < n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;
            for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_0(b + t * k, bQ8 + t * blocks * Q8BlockBytes, k);

            MatMul.GemmI2_S(weights, b, c, m, k, n, null);

            for (int r = 0; r < m; r++)
            {
                MatMul.UnpackRowI8Scalar(weights + r * rowBytes, wI8, k);
                for (int t = 0; t < n; t++)
                {
                    (double exact, double mag) = ReferenceI2S(wI8, bQ8 + t * blocks * Q8BlockBytes, blocks);
                    AssertTight(exact * scale, mag * scale, c[t * m + r], $"row {r} token {t}");
                }
            }
        }
        finally
        {
            NativeMemory.Free(weights); NativeMemory.Free(b); NativeMemory.Free(c);
            NativeMemory.Free(bQ8); NativeMemory.Free(wI8);
        }
    }

    /// <summary>
    /// The pooled W2A8 drivers (row-partitioned workers reading the context structs, including the
    /// pre-converted activation scales) must be bit-identical to the single-threaded drivers: each
    /// row runs the same kernels in the same order. m is above ParallelMinRows (32) and not a
    /// multiple of the thread count, so the pool path engages with uneven partitions.
    /// </summary>
    [SkippableTheory]
    [InlineData("pq2", 97, 640, 1)]
    [InlineData("i2s", 97, 384, 1)]
    [InlineData("i2s", 97, 384, 3)]
    [InlineData("i2s", 101, 256, 6)]   // 4x4 tile + token remainder on AVX2; per-cell on SSE
    public void Pooled_MatchesSingleThreaded_BitExact(string format, int m, int k, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(477_600 + m + k + n);
        bool pq2 = format == "pq2";
        long wBytes = pq2 ? (long)m * (k / 128) * 34 : (long)m * k / 4 + 4;
        byte* weights = (byte*)NativeMemory.Alloc((nuint)wBytes);
        float* b = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* c1 = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* cP = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        using var pool = new DotLLM.Cpu.Threading.ComputeThreadPool(5);
        try
        {
            if (pq2)
            {
                int rowBytes = (k / 128) * 34;
                for (int r = 0; r < m; r++)
                {
                    byte* row = RandomPQ2_0Row(rng, k / 128);
                    Buffer.MemoryCopy(row, weights + r * rowBytes, rowBytes, rowBytes);
                    NativeMemory.Free(row);
                }
            }
            else
            {
                for (long i = 0; i < wBytes - 4; i++)
                    weights[i] = (byte)(rng.Next(3) | (rng.Next(3) << 2) | (rng.Next(3) << 4) | (rng.Next(3) << 6));
                *(float*)(weights + wBytes - 4) = 0.041f;
            }
            for (int i = 0; i < n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;

            if (pq2)
            {
                MatMul.GemvPQ2_0(weights, b, c1, m, k, null);
                MatMul.GemvPQ2_0(weights, b, cP, m, k, pool);
            }
            else
            {
                MatMul.GemmI2_S(weights, b, c1, m, k, n, null);
                MatMul.GemmI2_S(weights, b, cP, m, k, n, pool);
            }

            for (int i = 0; i < n * m; i++)
                Assert.True(BitConverter.SingleToInt32Bits(c1[i]) == BitConverter.SingleToInt32Bits(cP[i]),
                    $"{format} element {i}: single-threaded {c1[i]} vs pooled {cP[i]}");
        }
        finally
        {
            NativeMemory.Free(weights); NativeMemory.Free(b); NativeMemory.Free(c1); NativeMemory.Free(cP);
        }
    }

    // ──────────────────── helpers ────────────────────

    /// <summary>Σ_b (float)(d_b·g_{b/4}) · Σ w·q, with the integer block sum exact and the float
    /// combination done in double. Also returns Σ|term| as the rounding-magnitude scale.</summary>
    private static (double Exact, double Magnitude) ReferencePQ2_0(sbyte* w, float* gs, byte* xQ8, int blocks)
    {
        double sum = 0, mag = 0;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* xb = xQ8 + blk * Q8BlockBytes;
            float dCombined = (float)*(Half*)xb * gs[blk / 4];
            sbyte* q = (sbyte*)(xb + 2);
            int isum = 0;
            for (int i = 0; i < Q8Block; i++) isum += w[blk * Q8Block + i] * q[i];
            double term = (double)dCombined * isum;
            sum += term;
            mag += Math.Abs(term);
        }
        return (sum, mag);
    }

    private static (double Exact, double Magnitude) ReferenceI2S(sbyte* w, byte* xQ8, int blocks)
    {
        double sum = 0, mag = 0;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* xb = xQ8 + blk * Q8BlockBytes;
            float dx = (float)*(Half*)xb;
            sbyte* q = (sbyte*)(xb + 2);
            int isum = 0;
            for (int i = 0; i < Q8Block; i++) isum += w[blk * Q8Block + i] * q[i];
            double term = (double)dx * isum;
            sum += term;
            mag += Math.Abs(term);
        }
        return (sum, mag);
    }

    /// <summary>fp32 cross-block summation envelope: a few ulps of Σ|term|. Any real kernel bug
    /// (wrong lane, sign, scale index, field) moves the result by O(term), orders of magnitude more.</summary>
    private static void AssertTight(double expected, double magnitude, float actual, string what)
    {
        double tol = 4e-6 * magnitude + 1e-7;
        double diff = Math.Abs(expected - actual);
        Assert.True(diff <= tol, $"{what}: expected {expected}, got {actual}, |Δ|={diff} > tol {tol}");
    }

    private static void AssertBytesEqual(sbyte* expected, sbyte* actual, int k)
    {
        for (int i = 0; i < k; i++)
            if (expected[i] != actual[i])
                Assert.Fail($"mismatch at element {i}: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>A PQ2_0 row of <paramref name="groups"/> groups with random scales and fully
    /// random code bytes (all four 2-bit codes, including 3 → +2).</summary>
    private static byte* RandomPQ2_0Row(Random rng, int groups)
    {
        byte* row = (byte*)NativeMemory.Alloc((nuint)(groups * 34));
        for (int g = 0; g < groups; g++)
        {
            byte* gb = row + g * 34;
            *(Half*)gb = (Half)(0.005f + rng.NextSingle() * 0.08f);
            for (int i = 0; i < 32; i++) gb[2 + i] = (byte)rng.Next(256);
        }
        return row;
    }

    /// <summary>Pre-converted Q8_0 block scales, as the W2A8 drivers pass to the dots.</summary>
    private static float* ScalesOf(byte* xQ8, int blocks)
    {
        float* xs = (float*)NativeMemory.Alloc((nuint)(blocks * sizeof(float)));
        MatMul.ConvertQ8_0Scales(xQ8, xs, blocks);
        return xs;
    }

    private static sbyte* RandomCodes(Random rng, int k, int maxCode)
    {
        sbyte* w = (sbyte*)NativeMemory.Alloc((nuint)k);
        for (int i = 0; i < k; i++) w[i] = (sbyte)(rng.Next(maxCode + 2) - 1);
        return w;
    }

    private static byte* RandomQ8(Random rng, int k)
    {
        float* x = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        byte* xQ8 = (byte*)NativeMemory.Alloc((nuint)(k / Q8Block * Q8BlockBytes));
        try
        {
            // Vary block magnitudes so the per-block scales differ meaningfully.
            for (int i = 0; i < k; i++) x[i] = (rng.NextSingle() * 2f - 1f) * (1 + (i / Q8Block) % 5);
            MatMul.QuantizeF32ToQ8_0Scalar(x, xQ8, k);
        }
        finally { NativeMemory.Free(x); }
        return xQ8;
    }
}
