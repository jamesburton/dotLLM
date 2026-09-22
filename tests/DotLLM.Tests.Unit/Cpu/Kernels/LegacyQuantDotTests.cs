using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #489: the packed <c>weight × Q8_1</c> dots for Q4_0, Q4_1, Q5_1 and IQ4_NL
/// (<c>MatMulLegacyQuants.cs</c> + <c>MatMulLegacyQuants.Sse.cs</c>). Before #489 these four went
/// through <c>Dequantize*Scalar</c> + <c>GemmDequantRows</c> on every ISA.
///
/// <para>Three independent oracles, strongest first:</para>
/// <list type="number">
///   <item><b>Exact integer reference</b> with unit scales — no kernel involved. Pins the nibble
///         order, Q4_0's −8 zero point, Q4_1/Q5_1's minimum term, Q5_1's <c>qh</c> bit spread and
///         the IQ4_NL codebook. A wrong constant fails here by construction.</item>
///   <item><b>128-bit tier bit-exact with scalar</b> — float bit patterns, including raw Q8_1
///         bytes (−128 included) that no quantizer would emit.</item>
///   <item><b>Dispatch parity</b> — the public GEMV/GEMM drivers against the scalar dot per row,
///         and against the <c>GemmDequantRows</c> path the kernels replace, on odd shapes.</item>
/// </list>
///
/// <para>The AVX2 tier folds eight per-lane float partials once per row (like the Q5_0 AVX2 tier)
/// so it is compared by tolerance, not bits; the 128-bit tier is bit-exact and is asserted as such
/// whenever it is the dispatched tier (<c>DOTNET_EnableAVX=0</c>, or real pre-AVX2 hardware).</para>
/// </summary>
public sealed unsafe class LegacyQuantDotTests
{
    private const int Q81Bytes = 36;
    private readonly ITestOutputHelper _output;
    public LegacyQuantDotTests(ITestOutputHelper output) => _output = output;

    private static bool SseTierDispatched => Ssse3.IsSupported && !Avx2.IsSupported;

    public static TheoryData<QuantizationType> Formats
    {
        get
        {
            var data = new TheoryData<QuantizationType>();
            foreach (var qt in PackedFormats) data.Add(qt);
            return data;
        }
    }

    private static QuantizationType[] PackedFormats =>
    [
        QuantizationType.Q4_0, QuantizationType.Q4_1,
        QuantizationType.Q5_1, QuantizationType.IQ4_NL
    ];

    public static TheoryData<QuantizationType, int> FormatsAndBlockCounts()
    {
        var data = new TheoryData<QuantizationType, int>();
        foreach (var qt in PackedFormats)
            foreach (int bc in new[] { 1, 3, 4, 5, 18, 48, 137 })
                data.Add(qt, bc);
        return data;
    }

    // ──────────────────── 1. Exact integer reference ────────────────────

    /// <summary>
    /// Unit scales (<c>d = d8 = 1</c>, <c>m = 0</c>) and activations in [−32, 31] so every block's
    /// <c>s8 = (Half)Σq8</c> is exact (|Σ| ≤ 1024 ≤ 2048). The dot is then the plain integer
    /// <c>Σ w_i·q8_i</c> with <c>w</c> read straight out of the block layout — an oracle that
    /// shares no code with any kernel — and stays well below 2^24 so float holds it exactly.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Formats))]
    public void UnitScales_AllTiers_EqualExactIntegerDot(QuantizationType qt)
    {
        const int blockCount = 18;        // k = 576, the SmolLM-135M projection width
        const int m = 5;                  // 4-row group + 1 tail row
        int blockBytes = BlockBytes(qt);
        int rowBytes = blockCount * blockBytes;
        var rng = new Random(4890 + (int)qt);

        byte* w = (byte*)NativeMemory.Alloc((nuint)(m * rowBytes));
        byte* x = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        float* res = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++) FillWeightRow(rng, qt, w + r * rowBytes, blockCount, unitScales: true);
            PinCornerQuants(qt, w);
            for (int b = 0; b < blockCount; b++)
            {
                byte* blk = x + b * Q81Bytes;
                int sum = 0;
                for (int i = 0; i < 32; i++)
                {
                    int v = rng.Next(-32, 32);
                    blk[4 + i] = unchecked((byte)(sbyte)v);
                    sum += v;
                }
                *(Half*)blk = (Half)1f;
                *(Half*)(blk + 2) = (Half)sum;
                Assert.Equal((float)sum, (float)*(Half*)(blk + 2));   // exact Half round-trip
            }

            for (int r = 0; r < m; r++)
            {
                byte* row = w + r * rowBytes;
                long exact = ExactIntegerDot(qt, row, x, blockCount);
                Assert.True(Math.Abs(exact) < (1 << 24), $"oracle {exact} outside exact-float range");

                Assert.Equal((float)exact, MatMul.VecDotLegacyQuantScalar(qt, row, x, blockCount));
                if (Ssse3.IsSupported)
                    Assert.Equal((float)exact, SseDot(qt, row, x, blockCount));
                if (Avx2.IsSupported)
                    Assert.Equal((float)exact, Avx2Dot(qt, row, x, blockCount));
            }

            // Both dispatch bodies (single-column and multi-column) see the same oracle.
            MatMul.ComputeRowsLegacyQuant(qt, w, x, res, m, blockCount);
            for (int r = 0; r < m; r++)
                Assert.Equal((float)ExactIntegerDot(qt, w + r * rowBytes, x, blockCount), res[r]);

            new Span<float>(res, m).Clear();
            MatMul.ComputeRowsLegacyQuantMulti(qt, w, x, blockCount * Q81Bytes, res, m, m, 1, blockCount);
            for (int r = 0; r < m; r++)
                Assert.Equal((float)ExactIntegerDot(qt, w + r * rowBytes, x, blockCount), res[r]);
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); NativeMemory.Free(res); }
    }

    // ──────────────────── 2. 128-bit tier is bit-exact with scalar ────────────────────

    [SkippableTheory]
    [MemberData(nameof(FormatsAndBlockCounts))]
    public void Sse128Tier_BitExactWithScalar(QuantizationType qt, int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4900 + (int)qt * 31 + blockCount);
        int blockBytes = BlockBytes(qt);
        byte* w = (byte*)NativeMemory.Alloc((nuint)(blockCount * blockBytes));
        byte* raw = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        byte* xq = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        try
        {
            FillWeightRow(rng, qt, w, blockCount, unitScales: false);
            FillRawQ8_1Row(rng, raw, blockCount);
            QuantizedQ8_1Row(rng, xq, blockCount);

            // IQ4_NL's sign trick has a documented precondition (no -128 activation byte), so it
            // only claims bit-exactness on quantizer output; the other three have none.
            if (qt != QuantizationType.IQ4_NL)
                AssertBitEqual(MatMul.VecDotLegacyQuantScalar(qt, w, raw, blockCount),
                    SseDot(qt, w, raw, blockCount), $"{qt} raw bc={blockCount}");

            AssertBitEqual(MatMul.VecDotLegacyQuantScalar(qt, w, xq, blockCount),
                SseDot(qt, w, xq, blockCount), $"{qt} quantized bc={blockCount}");

            if (Avx2.IsSupported)
            {
                float s = SseDot(qt, w, xq, blockCount);
                float a = Avx2Dot(qt, w, xq, blockCount);
                float mag = AbsTermSum(qt, w, xq, blockCount);
                Assert.True(Math.Abs(s - a) <= 1e-5f * mag + 1e-30f, $"{qt}: SSE {s} vs AVX2 {a} (mag {mag})");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(raw); NativeMemory.Free(xq); }
    }

    /// <summary>
    /// The 4-rows-per-lane 128-bit kernel against the scalar dot per row, bit for bit. A broadcast
    /// or lane-ordering bug shows here and not in the single-row kernel.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(FormatsAndBlockCounts))]
    public void Sse128Tier_4Rows_BitExactWithScalarPerRow(QuantizationType qt, int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4910 + (int)qt * 31 + blockCount);
        int rowBytes = blockCount * BlockBytes(qt);
        byte* w = (byte*)NativeMemory.Alloc((nuint)(4 * rowBytes));
        byte* x = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        float* res = stackalloc float[4];
        try
        {
            for (int r = 0; r < 4; r++) FillWeightRow(rng, qt, w + r * rowBytes, blockCount, unitScales: false);
            QuantizedQ8_1Row(rng, x, blockCount);

            MatMul.VecDotLegacyQuantSse_4Rows(qt, w, rowBytes, x, blockCount, res);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.VecDotLegacyQuantScalar(qt, w + r * rowBytes, x, blockCount), res[r],
                    $"{qt} 4-rows r={r} bc={blockCount}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>
    /// The IQ4_NL sign trick needs PSIGN's operand never to be −128. That is a property of
    /// <see cref="MatMul.QuantizeF32ToQ8_1"/>, not of the kernel, so it is pinned here — including
    /// the inputs most likely to round to −128 (an exact negative maximum).
    /// </summary>
    [Fact]
    public void QuantizeF32ToQ8_1_NeverEmitsMinus128()
    {
        const int k = 32 * 64;
        var rng = new Random(4920);
        float* xf = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        byte* q = (byte*)NativeMemory.Alloc((nuint)(k / 32 * Q81Bytes));
        try
        {
            for (int i = 0; i < k; i++) xf[i] = (rng.NextSingle() * 2f - 1f) * 7f;
            for (int b = 0; b < k / 32; b++) xf[b * 32 + (b % 32)] = -(1f + b);  // the block's negative max
            xf[0] = -1e30f; xf[1] = 1e-30f;

            MatMul.QuantizeF32ToQ8_1(xf, q, k);
            for (int b = 0; b < k / 32; b++)
                for (int i = 0; i < 32; i++)
                    Assert.NotEqual(-128, (sbyte)q[b * Q81Bytes + 4 + i]);

            MatMul.QuantizeF32ToQ8_1Scalar(xf, q, k);
            for (int b = 0; b < k / 32; b++)
                for (int i = 0; i < 32; i++)
                    Assert.NotEqual(-128, (sbyte)q[b * Q81Bytes + 4 + i]);
        }
        finally { NativeMemory.Free(xf); NativeMemory.Free(q); }
    }

    // ──────────────────── 3. Dispatch parity ────────────────────

    /// <summary>
    /// Every public driver against the scalar dot per row, at odd row counts that straddle the
    /// 4-row group, the pool threshold and the column tile. Bit-exact when the 128-bit tier is the
    /// dispatched one; tolerance otherwise (AVX2 reassociation).
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 7, 5)]
    [InlineData(QuantizationType.Q4_0, 67, 18)]
    [InlineData(QuantizationType.Q4_1, 13, 18)]
    [InlineData(QuantizationType.Q4_1, 39, 5)]
    [InlineData(QuantizationType.Q5_1, 39, 18)]
    [InlineData(QuantizationType.Q5_1, 7, 48)]
    [InlineData(QuantizationType.IQ4_NL, 67, 18)]
    [InlineData(QuantizationType.IQ4_NL, 13, 5)]
    public void Dispatch_MatchesScalarPerRow(QuantizationType qt, int m, int blockCount)
    {
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"-> SSE tier dispatched: {SseTierDispatched}");
        var rng = new Random(4930 + (int)qt * 131 + m);
        int k = blockCount * 32, rowBytes = blockCount * BlockBytes(qt), xRowBytes = blockCount * Q81Bytes;
        const int n = 6;   // > the 4-column multi tile, so the remainder columns run too

        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        float* xf = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        byte* xs = (byte*)NativeMemory.Alloc((nuint)(n * xRowBytes));
        float* expected = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* actual = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* scale = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++) FillWeightRow(rng, qt, w + r * rowBytes, blockCount, unitScales: false);
            for (int i = 0; i < n * k; i++) xf[i] = (rng.NextSingle() * 2f - 1f) * 3f;
            for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_1(xf + t * k, xs + t * xRowBytes, k);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                {
                    expected[t * m + r] = MatMul.VecDotLegacyQuantScalar(qt, w + r * rowBytes, xs + t * xRowBytes, blockCount);
                    scale[t * m + r] = AbsTermSum(qt, w + r * rowBytes, xs + t * xRowBytes, blockCount);
                }

            MatMul.ComputeRowsLegacyQuant(qt, w, xs, actual, m, blockCount);
            AssertRows(expected, actual, scale, m, $"{qt} ComputeRows");

            new Span<float>(actual, n * m).Clear();
            MatMul.ComputeRowsLegacyQuantMulti(qt, w, xs, xRowBytes, actual, m, m, n, blockCount);
            AssertRows(expected, actual, scale, n * m, $"{qt} ComputeRowsMulti");

            new Span<float>(actual, m).Clear();
            MatMul.GemvLegacyQuant(w, qt, xf, actual, m, k);
            AssertRows(expected, actual, scale, m, $"{qt} Gemv");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmLegacyQuant(w, qt, xf, actual, m, k, n);
            AssertRows(expected, actual, scale, n * m, $"{qt} Gemm");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmLegacyQuant(w, qt, xf, actual, m, k, n, xs);
            AssertRows(expected, actual, scale, n * m, $"{qt} Gemm pre-quantized");

            new Span<float>(actual, m).Clear();
            MatMul.GemmLegacyQuant(w, qt, xf, actual, m, k, 1);
            AssertRows(expected, actual, scale, m, $"{qt} Gemm n=1");

            using var pool = new ComputeThreadPool(4);
            new Span<float>(actual, m).Clear();
            MatMul.GemvLegacyQuant(w, qt, xf, actual, m, k, pool);
            AssertRows(expected, actual, scale, m, $"{qt} Gemv pooled");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmLegacyQuant(w, qt, xf, actual, m, k, n, pool);
            AssertRows(expected, actual, scale, n * m, $"{qt} Gemm pooled");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmLegacyQuant(w, qt, xf, actual, m, k, n, pool, xs);
            AssertRows(expected, actual, scale, n * m, $"{qt} Gemm pooled pre-quantized");

            new Span<float>(actual, m).Clear();
            MatMul.GemmLegacyQuant(w, qt, xf, actual, m, k, 1, pool, xs);
            AssertRows(expected, actual, scale, m, $"{qt} Gemm pooled n=1 pre-quantized");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xf); NativeMemory.Free(xs);
            NativeMemory.Free(expected); NativeMemory.Free(actual); NativeMemory.Free(scale);
        }
    }

    /// <summary>
    /// Parity with the <c>GemmDequantRows</c> path these kernels replace. That path is F32-exact
    /// while this one quantizes the activations to Q8_1, so the comparison is a relative one
    /// against <c>Σ|w_i·x_i|</c>: a real difference (wrong nibble order, wrong zero point, missing
    /// <c>qh</c> bit, shifted codebook) moves the result by far more than Q8_1's ~1/127 per-element
    /// rounding, which is what the mutant table in the issue thread shows.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.Q4_0, 37, 576)]
    [InlineData(QuantizationType.Q4_1, 37, 576)]
    [InlineData(QuantizationType.Q5_1, 37, 576)]
    [InlineData(QuantizationType.IQ4_NL, 37, 576)]
    [InlineData(QuantizationType.Q4_0, 13, 160)]
    [InlineData(QuantizationType.IQ4_NL, 13, 160)]
    public void MatchesDequantizeThenGemm(QuantizationType qt, int m, int k)
    {
        var rng = new Random(4940 + (int)qt * 17 + m);
        int blockCount = k / 32, rowBytes = blockCount * BlockBytes(qt);
        const int n = 3;

        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        float* xf = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* packed = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* dequant = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float[] row = new float[k];
        try
        {
            for (int r = 0; r < m; r++) FillWeightRow(rng, qt, w + r * rowBytes, blockCount, unitScales: false);
            for (int i = 0; i < n * k; i++) xf[i] = (rng.NextSingle() * 2f - 1f) * 3f;

            MatMul.GemmDequantRows(w, qt, xf, dequant, m, k, n, pool: null);
            MatMul.GemmLegacyQuant(w, qt, xf, packed, m, k, n);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                {
                    Dequantize.ToFloat32((nint)(w + (long)r * rowBytes), k, qt, row);
                    double mag = 0;
                    for (int i = 0; i < k; i++) mag += Math.Abs(row[i] * xf[t * k + i]);
                    double diff = Math.Abs(packed[t * m + r] - dequant[t * m + r]);
                    Assert.True(diff <= 0.02 * mag + 1e-6,
                        $"{qt} [t={t},r={r}]: packed {packed[t * m + r]} vs dequant {dequant[t * m + r]} (mag {mag})");
                }

            // And the policy helper: n == 1 must take the packed path, n > 1 the dequant one.
            Assert.True(MatMul.UsesPackedLegacyDot(qt, 1));
            Assert.False(MatMul.UsesPackedLegacyDot(qt, 2));
            Assert.False(MatMul.UsesPackedLegacyDot(QuantizationType.Q8_0, 1));

            new Span<float>(packed, n * m).Clear();
            MatMul.GemmLegacyQuantOrDequant(w, qt, xf, packed, m, k, n, pool: null);
            for (int i = 0; i < n * m; i++) AssertBitEqual(dequant[i], packed[i], $"{qt} OrDequant n={n} [{i}]");

            new Span<float>(packed, m).Clear();
            float* gemv = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
            try
            {
                MatMul.GemmLegacyQuantOrDequant(w, qt, xf, packed, m, k, 1, pool: null);
                MatMul.GemvLegacyQuant(w, qt, xf, gemv, m, k);
                for (int i = 0; i < m; i++) AssertBitEqual(gemv[i], packed[i], $"{qt} OrDequant n=1 [{i}]");
            }
            finally { NativeMemory.Free(gemv); }
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xf);
            NativeMemory.Free(packed); NativeMemory.Free(dequant);
        }
    }

    [Fact]
    public void HasPackedLegacyDot_CoversExactlyTheFourFormats()
    {
        foreach (var qt in Enum.GetValues<QuantizationType>())
        {
            bool expected = qt is QuantizationType.Q4_0 or QuantizationType.Q4_1
                              or QuantizationType.Q5_1 or QuantizationType.IQ4_NL;
            Assert.Equal(expected, MatMul.HasPackedLegacyDot(qt));
        }
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void Gemv_RejectsNonMultipleOf32(QuantizationType qt)
    {
        byte* w = (byte*)NativeMemory.Alloc(1024);
        float* x = (float*)NativeMemory.Alloc(1024);
        float* y = (float*)NativeMemory.Alloc(1024);
        try
        {
            Assert.Throws<ArgumentException>(() => MatMul.GemvLegacyQuant(w, qt, x, y, 1, 33));
            Assert.Throws<ArgumentException>(() => MatMul.GemmLegacyQuant(w, qt, x, y, 1, 33, 2));
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); NativeMemory.Free(y); }
    }

    // ──────────────────── oracles and helpers ────────────────────

    private static int BlockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_0 => 18,
        QuantizationType.Q4_1 => 20,
        QuantizationType.Q5_1 => 24,
        _ => 18
    };

    /// <summary>Byte offset of <c>qs</c> within a block — independent of the kernel's own table.</summary>
    private static int QsOffset(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_0 => 2,
        QuantizationType.Q4_1 => 4,
        QuantizationType.Q5_1 => 8,
        _ => 2
    };

    private static ReadOnlySpan<sbyte> Iq4NlTable =>
    [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    ];

    /// <summary>
    /// <c>Σ w_i·q8_i</c> from the raw block bytes, assuming unit scales and a zero minimum —
    /// written from the GGUF layout, sharing nothing with the kernels under test.
    /// </summary>
    private static long ExactIntegerDot(QuantizationType qt, byte* row, byte* x, int blockCount)
    {
        int blockBytes = BlockBytes(qt), qsOff = QsOffset(qt);
        long total = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = row + b * blockBytes;
            byte* qs = blk + qsOff;
            sbyte* q8 = (sbyte*)(x + b * Q81Bytes + 4);
            uint qh = qt == QuantizationType.Q5_1 ? *(uint*)(blk + 4) : 0;
            for (int j = 0; j < 16; j++)
            {
                int lo = qs[j] & 0xF, hi = qs[j] >> 4;
                int v0, v1;
                switch (qt)
                {
                    case QuantizationType.Q4_0: v0 = lo - 8; v1 = hi - 8; break;
                    case QuantizationType.Q4_1: v0 = lo; v1 = hi; break;
                    case QuantizationType.Q5_1:
                        v0 = lo | (int)(((qh >> j) & 1) << 4);
                        v1 = hi | (int)(((qh >> (j + 16)) & 1) << 4);
                        break;
                    default: v0 = Iq4NlTable[lo]; v1 = Iq4NlTable[hi]; break;
                }
                total += (long)v0 * q8[j] + (long)v1 * q8[j + 16];
            }
        }
        return total;
    }

    /// <summary>Σ over blocks of |d·d8·sumi| plus the offset magnitude — bounds float reassociation.</summary>
    private static float AbsTermSum(QuantizationType qt, byte* w, byte* x, int blockCount)
    {
        int blockBytes = BlockBytes(qt), qsOff = QsOffset(qt);
        double s = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = w + b * blockBytes;
            byte* xb = x + b * Q81Bytes;
            byte* qs = blk + qsOff;
            sbyte* q8 = (sbyte*)(xb + 4);
            uint qh = qt == QuantizationType.Q5_1 ? *(uint*)(blk + 4) : 0;
            long sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                int lo = qs[j] & 0xF, hi = qs[j] >> 4;
                if (qt == QuantizationType.Q5_1)
                {
                    lo |= (int)(((qh >> j) & 1) << 4);
                    hi |= (int)(((qh >> (j + 16)) & 1) << 4);
                }
                else if (qt == QuantizationType.IQ4_NL)
                {
                    lo = Iq4NlTable[lo]; hi = Iq4NlTable[hi];
                }
                sumi += (long)lo * q8[j] + (long)hi * q8[j + 16];
            }
            double d = (float)*(Half*)blk;
            s += Math.Abs(d * (float)*(Half*)xb * sumi);
            if (qt == QuantizationType.Q4_0) s += 8 * Math.Abs(d * (float)*(Half*)(xb + 2));
            else if (qt != QuantizationType.IQ4_NL) s += Math.Abs((float)*(Half*)(blk + 2) * (float)*(Half*)(xb + 2));
        }
        return (float)s;
    }

    private static float SseDot(QuantizationType qt, byte* w, byte* x, int blockCount) => qt switch
    {
        QuantizationType.Q4_0 => MatMul.VecDotQ4_0Q8_1Sse(w, x, blockCount),
        QuantizationType.Q4_1 => MatMul.VecDotQ4_1Q8_1Sse(w, x, blockCount),
        QuantizationType.Q5_1 => MatMul.VecDotQ5_1Q8_1Sse(w, x, blockCount),
        _ => MatMul.VecDotIQ4_NLQ8_1Sse(w, x, blockCount)
    };

    private static float Avx2Dot(QuantizationType qt, byte* w, byte* x, int blockCount) => qt switch
    {
        QuantizationType.Q4_0 => MatMul.VecDotQ4_0Q8_1Avx2(w, x, blockCount),
        QuantizationType.Q4_1 => MatMul.VecDotQ4_1Q8_1Avx2(w, x, blockCount),
        QuantizationType.Q5_1 => MatMul.VecDotQ5_1Q8_1Avx2(w, x, blockCount),
        _ => MatMul.VecDotIQ4_NLQ8_1Avx2(w, x, blockCount)
    };

    /// <summary>
    /// Random weight row. <paramref name="unitScales"/> writes <c>d = 1</c> and <c>m = 0</c> for
    /// the integer oracle; otherwise scales span normals, both signs and Half subnormals.
    /// </summary>
    private static void FillWeightRow(Random rng, QuantizationType qt, byte* p, int blockCount, bool unitScales)
    {
        int blockBytes = BlockBytes(qt), qsOff = QsOffset(qt);
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = p + b * blockBytes;
            for (int i = 0; i < blockBytes; i++) blk[i] = (byte)rng.Next(256);

            if (unitScales)
            {
                *(Half*)blk = (Half)1f;
                if (qt is QuantizationType.Q4_1 or QuantizationType.Q5_1) *(Half*)(blk + 2) = (Half)0f;
            }
            else
            {
                float d = rng.Next(8) == 0 ? (rng.NextSingle() - 0.5f) * 1e-5f
                                           : (rng.NextSingle() * 2f - 1f) * 0.05f;
                *(Half*)blk = (Half)d;
                if (qt is QuantizationType.Q4_1 or QuantizationType.Q5_1)
                    *(Half*)(blk + 2) = (Half)((rng.NextSingle() * 2f - 1f) * 0.2f);
            }
            _ = qsOff;
        }
    }

    /// <summary>Pins the quant corners in row 0 block 0: element 0 maximal, element 31 minimal.</summary>
    private static void PinCornerQuants(QuantizationType qt, byte* w)
    {
        int qsOff = QsOffset(qt);
        byte* qs = w + qsOff;
        qs[0] = (byte)((qs[0] & 0xF0) | 0x0F);      // element 0 -> nibble 15
        qs[15] = (byte)(qs[15] & 0x0F);             // element 31 -> nibble 0
        if (qt == QuantizationType.Q5_1)
        {
            uint* qh = (uint*)(w + 4);
            *qh = (*qh | 1u) & 0x7FFFFFFFu;         // element 0 bit set, element 31 bit clear
        }
    }

    private static void FillRawQ8_1Row(Random rng, byte* p, int blockCount)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = p + b * Q81Bytes;
            *(Half*)blk = (Half)((rng.NextSingle() * 2f - 1f) * 0.05f);
            *(Half*)(blk + 2) = (Half)((rng.NextSingle() * 2f - 1f) * 20f);
            for (int i = 0; i < 32; i++) blk[4 + i] = (byte)rng.Next(256);
        }
    }

    private static void QuantizedQ8_1Row(Random rng, byte* p, int blockCount)
    {
        int k = blockCount * 32;
        float* xf = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        try
        {
            for (int i = 0; i < k; i++) xf[i] = (rng.NextSingle() * 2f - 1f) * 3f;
            MatMul.QuantizeF32ToQ8_1(xf, p, k);
        }
        finally { NativeMemory.Free(xf); }
    }

    private static void AssertRows(float* expected, float* actual, float* scale, int count, string what)
    {
        for (int i = 0; i < count; i++)
        {
            if (SseTierDispatched)
                AssertBitEqual(expected[i], actual[i], $"{what} [{i}]");
            else
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * scale[i] + 1e-30f,
                    $"{what} [{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    private static void AssertBitEqual(float expected, float actual, string what) =>
        Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
            $"{what}: expected {expected:R}, got {actual:R}");
}
