using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Packed <c>weight × Q8_1</c> dot kernels for the four 32-element "legacy" block formats that
/// had no quantized CPU matmul on any ISA before issue #489 — <see cref="QuantizationType.Q4_0"/>,
/// <see cref="QuantizationType.Q4_1"/>, <see cref="QuantizationType.Q5_1"/> and
/// <see cref="QuantizationType.IQ4_NL"/>. Every GEMV/GEMM for these used to go through
/// <c>Dequantize*Scalar</c> + <see cref="GemmDequantRows"/>: each decode step expanded the whole
/// weight matrix to F32 first, which costs 4–8× the bytes the packed rows occupy plus the unpack
/// ALU work. Measured on SmolLM-135M (pure ladder, Zen 5, 32 threads): Q8_0 decodes at 173 tok/s
/// while Q4_0/Q4_1/IQ4_NL sit at ~100 and Q5_1 at 74 — slower than a format with <em>twice</em>
/// the weight bytes, which is the signature of a dequantize-bound path.
///
/// <para><b>Structure</b> mirrors <c>MatMulQ5_0.cs</c>/<c>MatMul.Q5_0Sse.cs</c>: a scalar
/// reference, an AVX2 tier, a 128-bit SSSE3 tier (<c>MatMul.LegacyQuantsSse.cs</c>, for pre-AVX2
/// hardware — issue #477's Westmere target), and one shared set of GEMV/GEMM/thread-pool outer
/// loops parameterized by <see cref="QuantizationType"/>.</para>
///
/// <para><b>Activations</b> are quantized to Q8_1, matching the reference <c>vec_dot</c> ggml
/// uses for each of these formats (<c>ggml-cpu/arch/x86/quants.c</c>: <c>vec_dot_q4_0_q8_1</c>,
/// <c>vec_dot_q4_1_q8_1</c>, <c>vec_dot_q5_1_q8_1</c>, <c>vec_dot_iq4_nl_q8_1</c>). The Q8_1
/// block carries both <c>d</c> and the precomputed sum <c>s = d·Σq8</c>, which is what makes the
/// zero-point (Q4_0's −8) and minimum (Q4_1/Q5_1's <c>m</c>) terms free.</para>
///
/// <para><b>Per-format block math</b>, with <c>sumi = Σ q_unsigned[i]·q8[i]</c> over one block:
/// <list type="bullet">
///   <item>Q4_0 (18 B: <c>d</c>@0, <c>qs[16]</c>@2) — <c>Σ d·d8·sumi − 8·Σ d·s8</c></item>
///   <item>Q4_1 (20 B: <c>d</c>@0, <c>m</c>@2, <c>qs[16]</c>@4) — <c>Σ d·d8·sumi + Σ m·s8</c></item>
///   <item>Q5_1 (24 B: <c>d</c>@0, <c>m</c>@2, <c>qh</c>@4, <c>qs[16]</c>@8) — as Q4_1 with the
///         5th bit from <c>qh</c></item>
///   <item>IQ4_NL (18 B: <c>d</c>@0, <c>qs[16]</c>@2) — <c>Σ d·d8·sumiLut</c>, where the nibble
///         indexes ggml's signed <c>kvalues_iq4nl</c> table; there is no offset term</item>
/// </list>
/// Nibble order is llama.cpp's: within <c>qs[j]</c> the low nibble is element <c>j</c> and the
/// high nibble is element <c>j + 16</c>.</para>
///
/// <para><b>Saturation.</b> For Q4_0/Q4_1/Q5_1 the weight operand is the unsigned 0..15 / 0..31
/// value, so it is PMADDUBSW's unsigned operand directly and a pair sum is at most
/// <c>2·31·128 = 7936</c> — nothing saturates for any activation byte. IQ4_NL's table is signed
/// (−127..113), so it uses ggml's sign trick, <c>maddubs(|lut|, sign(q8, lut))</c>: a pair sum is
/// at most <c>2·127·127 = 32258 &lt; 32767</c>. That trick requires the activation byte never to
/// be −128 (PSIGN maps −128 to itself), which <see cref="QuantizeF32ToQ8_1"/> guarantees — it
/// clamps to [−127, 127].</para>
///
/// <para><b>Numerics.</b> Every tier computes each block's <c>sumi</c> exactly in int32; the tiers
/// differ only in how the per-block float products are summed. The <b>128-bit tier is bit-exact</b>
/// with the scalar reference — it reduces each block horizontally and runs the scalar recurrence in
/// block order. The <b>AVX2 tier accumulates eight per-lane float partials</b> and folds them once
/// per row, exactly as the pre-existing Q5_0 AVX2 tier does, so it differs from scalar in the last
/// ulps. That is deliberate: a horizontal reduction inside the block loop is ~10 cycles of latency
/// each and measured ~5x slower at prefill on Zen 5.</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>Q4_0 block: 2 (Half d) + 16 (qs) = 18 bytes, 32 elements.</summary>
    internal const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    /// <summary>Q4_1 block: 2 (Half d) + 2 (Half m) + 16 (qs) = 20 bytes, 32 elements.</summary>
    internal const int Q4_1BlockBytes = QuantFormat.Q4_1BlockBytes;

    /// <summary>Q5_1 block: 2 (Half d) + 2 (Half m) + 4 (qh) + 16 (qs) = 24 bytes, 32 elements.</summary>
    internal const int Q5_1BlockBytes = QuantFormat.Q5_1BlockBytes;

    /// <summary>IQ4_NL block: 2 (Half d) + 16 (qs) = 18 bytes, 32 elements.</summary>
    internal const int IQ4_NLBlockBytes = QuantFormat.IQ4_NLBlockBytes;

    /// <summary>ggml's <c>kvalues_iq4nl</c> — the signed non-linear IQ4_NL codebook.</summary>
    internal static ReadOnlySpan<sbyte> KValuesIq4Nl =>
    [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    ];

    /// <summary>
    /// True when this quantization type has a packed <c>× Q8_1</c> dot in this file — i.e. when
    /// <c>GemvLegacyQuant</c>/<c>GemmLegacyQuant</c> may be called for it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool HasPackedLegacyDot(QuantizationType qt) =>
        qt is QuantizationType.Q4_0 or QuantizationType.Q4_1
           or QuantizationType.Q5_1 or QuantizationType.IQ4_NL;

    /// <summary>Block byte stride for a format accepted by <see cref="HasPackedLegacyDot"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int LegacyBlockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_0 => Q4_0BlockBytes,
        QuantizationType.Q4_1 => Q4_1BlockBytes,
        QuantizationType.Q5_1 => Q5_1BlockBytes,
        QuantizationType.IQ4_NL => IQ4_NLBlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(qt), qt, "not a packed legacy quant")
    };

    // ──────────────────── Scalar references ────────────────────

    /// <summary>
    /// Scalar Q4_0 × Q8_1 dot. <c>Σ d·d8·sumi − 8·Σ d·s8</c>; this accumulation order is what
    /// every SIMD tier reproduces bit for bit.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ4_0Q8_1Scalar(byte* w, byte* q8, int blockCount)
    {
        float sumf = 0;
        float offsetSum = 0;
        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * Q4_0BlockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(xb);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(xb + 2);
            sbyte* q8s = (sbyte*)(xb + 4);
            byte* qs = wb + 2;

            int sumi = 0;
            for (int j = 0; j < 16; j++)
                sumi += (qs[j] & 0xF) * q8s[j] + (qs[j] >> 4) * q8s[j + 16];

            sumf += d * d8 * sumi;
            offsetSum += d * s8;
        }
        return sumf - 8.0f * offsetSum;
    }

    /// <summary>Scalar Q4_1 × Q8_1 dot. <c>Σ d·d8·sumi + Σ m·s8</c>.</summary>
    [SkipLocalsInit]
    internal static float VecDotQ4_1Q8_1Scalar(byte* w, byte* q8, int blockCount)
    {
        float sumf = 0;
        float minSum = 0;
        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * Q4_1BlockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            float m = (float)Unsafe.ReadUnaligned<Half>(wb + 2);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(xb);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(xb + 2);
            sbyte* q8s = (sbyte*)(xb + 4);
            byte* qs = wb + 4;

            int sumi = 0;
            for (int j = 0; j < 16; j++)
                sumi += (qs[j] & 0xF) * q8s[j] + (qs[j] >> 4) * q8s[j + 16];

            sumf += d * d8 * sumi;
            minSum += m * s8;
        }
        return sumf + minSum;
    }

    /// <summary>Scalar Q5_1 × Q8_1 dot. As Q4_1, with the 5th bit of element <c>j</c> in bit <c>j</c> of <c>qh</c>.</summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_1Q8_1Scalar(byte* w, byte* q8, int blockCount)
    {
        float sumf = 0;
        float minSum = 0;
        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * Q5_1BlockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            float m = (float)Unsafe.ReadUnaligned<Half>(wb + 2);
            uint qh = Unsafe.ReadUnaligned<uint>(wb + 4);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(xb);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(xb + 2);
            sbyte* q8s = (sbyte*)(xb + 4);
            byte* qs = wb + 8;

            int sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                int lo = (qs[j] & 0xF) | (int)(((qh >> j) & 1) << 4);
                int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
                sumi += lo * q8s[j] + hi * q8s[j + 16];
            }

            sumf += d * d8 * sumi;
            minSum += m * s8;
        }
        return sumf + minSum;
    }

    /// <summary>Scalar IQ4_NL × Q8_1 dot. <c>Σ d·d8·Σ kvalues[q]·q8</c> — no offset term.</summary>
    [SkipLocalsInit]
    internal static float VecDotIQ4_NLQ8_1Scalar(byte* w, byte* q8, int blockCount)
    {
        float sumf = 0;
        ReadOnlySpan<sbyte> lut = KValuesIq4Nl;
        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * IQ4_NLBlockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(xb);
            sbyte* q8s = (sbyte*)(xb + 4);
            byte* qs = wb + 2;

            int sumi = 0;
            for (int j = 0; j < 16; j++)
                sumi += lut[qs[j] & 0xF] * q8s[j] + lut[qs[j] >> 4] * q8s[j + 16];

            sumf += d * d8 * sumi;
        }
        return sumf;
    }

    // ──────────────────── AVX2 block dots ────────────────────

    /// <summary>Low nibbles in the low 128-bit lane, high nibbles in the high lane — the ggml element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> UnpackNibblesAvx2(byte* qs)
    {
        Vector128<byte> raw = Unsafe.ReadUnaligned<Vector128<byte>>(qs);
        Vector128<byte> nib = Vector128.Create((byte)0x0F);
        Vector128<byte> lo = Sse2.And(raw, nib);
        Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), nib);
        return Vector256.Create(lo, hi);
    }

    /// <summary>Exact int32 <c>Σ q·q8</c> for one 32-element block of unsigned quants.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int UnsignedBlockSumAvx2(Vector256<byte> q, Vector256<sbyte> q8, Vector256<short> ones)
        => HorizontalSumInt(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q, q8), ones));

    /// <summary>Exact int32 horizontal sum of eight lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HorizontalSumInt(Vector256<int> v)
    {
        Vector128<int> s = Sse2.Add(v.GetLower(), v.GetUpper());
        s = Ssse3.HorizontalAdd(s, s);
        s = Ssse3.HorizontalAdd(s, s);
        return s.ToScalar();
    }

    /// <summary>
    /// IQ4_NL codebook broadcast into both 128-bit lanes, so <c>vpshufb</c> maps a 0..15 nibble
    /// to its signed table value within each lane.
    /// </summary>
    private static Vector256<sbyte> Iq4NlLutAvx2 => Vector256.Create(
        (sbyte)-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

    // ──────────────────── AVX2 single-row dots (bit-exact with scalar) ────────────────────

    /// <summary>
    /// AVX2 single-row packed legacy-quant x Q8_1 dot. Each block's <c>sumi</c> is exact in int32,
    /// but the eight int lanes are converted to float and accumulated per lane — one vcvtdq2ps +
    /// one multiply-add per block, with no horizontal reduction inside the loop. That is the same
    /// shape as the Q5_0 AVX2 tier and, like it, differs from the scalar reference in the last
    /// ulps (the 128-bit tier below is the bit-exact one). Folding the block sums horizontally
    /// per block instead costs ~10 cycles of latency each and measured 5x slower at prefill.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float VecDotLegacyAvx2(QuantizationType qt, byte* w, byte* q8, int blockCount)
    {
        int blockBytes = LegacyBlockBytes(qt);
        bool hasMin = qt is QuantizationType.Q4_1 or QuantizationType.Q5_1;
        bool isIq4 = qt == QuantizationType.IQ4_NL;
        Vector256<short> ones = Vector256.Create((short)1);
        Vector256<sbyte> lut = Iq4NlLutAvx2;

        Vector256<float> acc = Vector256<float>.Zero;
        float offsetSum = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * blockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            Vector256<byte> q = UnpackLegacyAvx2(qt, wb, lut, out Vector256<sbyte> signSrc);
            Vector256<sbyte> q8v = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 4);
            if (isIq4) q8v = Avx2.Sign(q8v, signSrc);
            Vector256<int> p = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q, q8v), ones);

            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            Vector256<float> scale = Vector256.Create(d * (float)Unsafe.ReadUnaligned<Half>(xb));
            acc = Avx.Add(acc, Avx.Multiply(scale, Avx.ConvertToVector256Single(p)));

            if (isIq4) continue;
            float offd = hasMin ? (float)Unsafe.ReadUnaligned<Half>(wb + 2) : d;
            offsetSum += offd * (float)Unsafe.ReadUnaligned<Half>(xb + 2);
        }

        float sumf = HorizontalSumAvx2Float(acc);
        if (isIq4) return sumf;
        return hasMin ? sumf + offsetSum : sumf - 8.0f * offsetSum;
    }

    /// <inheritdoc cref="VecDotQ4_0Q8_1Scalar"/>
    internal static float VecDotQ4_0Q8_1Avx2(byte* w, byte* q8, int blockCount)
        => VecDotLegacyAvx2(QuantizationType.Q4_0, w, q8, blockCount);

    /// <inheritdoc cref="VecDotQ4_1Q8_1Scalar"/>
    internal static float VecDotQ4_1Q8_1Avx2(byte* w, byte* q8, int blockCount)
        => VecDotLegacyAvx2(QuantizationType.Q4_1, w, q8, blockCount);

    /// <inheritdoc cref="VecDotQ5_1Q8_1Scalar"/>
    internal static float VecDotQ5_1Q8_1Avx2(byte* w, byte* q8, int blockCount)
        => VecDotLegacyAvx2(QuantizationType.Q5_1, w, q8, blockCount);

    /// <inheritdoc cref="VecDotIQ4_NLQ8_1Scalar"/>
    internal static float VecDotIQ4_NLQ8_1Avx2(byte* w, byte* q8, int blockCount)
        => VecDotLegacyAvx2(QuantizationType.IQ4_NL, w, q8, blockCount);


    /// <summary>
    /// Folds four 8-lane int vectors into one 4-lane vector holding their four totals — exact,
    /// and the AVX2 counterpart of <see cref="Reduce4"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Reduce4Avx2(Vector256<int> a, Vector256<int> b,
        Vector256<int> c, Vector256<int> d)
    {
        Vector256<int> ab = Avx2.HorizontalAdd(a, b);
        Vector256<int> cd = Avx2.HorizontalAdd(c, d);
        Vector256<int> abcd = Avx2.HorizontalAdd(ab, cd);
        return Sse2.Add(abcd.GetLower(), abcd.GetUpper());
    }

    /// <summary>
    /// Four rows at once with one <em>row</em> per 32-bit lane: the Q8_1 block is loaded once for
    /// all four, the four block sums are folded into one vector, and every float operation runs
    /// four-wide. That amortizes the three per-block Half conversions and breaks the serial
    /// offset-accumulator chain that make the single-row kernel 3–4x slower per block — measured
    /// on Zen 5 at the SmolLM-135M ffn_gate shape.
    ///
    /// <para>Bit-exact per row with <see cref="VecDotLegacyQuantScalar"/>, like the 128-bit
    /// 4-row kernel and unlike the single-row AVX2 kernel.</para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotLegacyQuantAvx2_4Rows(QuantizationType qt, byte* w, nint rowStride,
        byte* q8, int blockCount, float* results)
    {
        int blockBytes = LegacyBlockBytes(qt);
        bool hasMin = qt is QuantizationType.Q4_1 or QuantizationType.Q5_1;
        bool isIq4 = qt == QuantizationType.IQ4_NL;
        Vector256<short> ones = Vector256.Create((short)1);
        Vector256<sbyte> lut = Iq4NlLutAvx2;

        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<float> off = Vector128<float>.Zero;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * blockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            Vector256<sbyte> q8v = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 4);

            Vector128<int> s = Reduce4Avx2(
                LegacyBlockDotAvx2(qt, wb, q8v, lut, ones, isIq4),
                LegacyBlockDotAvx2(qt, wb + rowStride, q8v, lut, ones, isIq4),
                LegacyBlockDotAvx2(qt, wb + 2 * rowStride, q8v, lut, ones, isIq4),
                LegacyBlockDotAvx2(qt, wb + 3 * rowStride, q8v, lut, ones, isIq4));

            Vector128<float> d = HalfToSingleSse2(LoadHalf4(wb, rowStride));
            Vector128<float> d8 = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(xb)));
            acc = Sse.Add(acc, Sse.Multiply(Sse.Multiply(d, d8), Sse2.ConvertToVector128Single(s)));

            if (isIq4) continue;
            Vector128<float> offd = hasMin ? HalfToSingleSse2(LoadHalf4(wb + 2, rowStride)) : d;
            Vector128<float> s8 = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(xb + 2)));
            off = Sse.Add(off, Sse.Multiply(offd, s8));
        }

        Vector128<float> result = isIq4 ? acc
            : hasMin ? Sse.Add(acc, off)
            : Sse.Subtract(acc, Sse.Multiply(Vector128.Create(8.0f), off));
        Unsafe.WriteUnaligned(results, result);
    }

    /// <summary>Exact int32 lane partials of one block's <c>Σ q·q8</c> for <paramref name="qt"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> LegacyBlockDotAvx2(QuantizationType qt, byte* wb, Vector256<sbyte> q8v,
        Vector256<sbyte> lut, Vector256<short> ones, bool isIq4)
    {
        Vector256<byte> q = UnpackLegacyAvx2(qt, wb, lut, out Vector256<sbyte> signSrc);
        if (isIq4) q8v = Avx2.Sign(q8v, signSrc);
        return Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q, q8v), ones);
    }

    // ──────────────────── Tier selection ────────────────────

    /// <summary>Best available single-row dot for <paramref name="qt"/> (AVX2, then SSSE3, then scalar).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotLegacyQuantRow(QuantizationType qt, byte* w, byte* q8, int blockCount)
    {
        if (Avx2.IsSupported)
            return qt switch
            {
                QuantizationType.Q4_0 => VecDotQ4_0Q8_1Avx2(w, q8, blockCount),
                QuantizationType.Q4_1 => VecDotQ4_1Q8_1Avx2(w, q8, blockCount),
                QuantizationType.Q5_1 => VecDotQ5_1Q8_1Avx2(w, q8, blockCount),
                _ => VecDotIQ4_NLQ8_1Avx2(w, q8, blockCount)
            };
        if (Ssse3.IsSupported)
            return qt switch
            {
                QuantizationType.Q4_0 => VecDotQ4_0Q8_1Sse(w, q8, blockCount),
                QuantizationType.Q4_1 => VecDotQ4_1Q8_1Sse(w, q8, blockCount),
                QuantizationType.Q5_1 => VecDotQ5_1Q8_1Sse(w, q8, blockCount),
                _ => VecDotIQ4_NLQ8_1Sse(w, q8, blockCount)
            };
        return VecDotLegacyQuantScalar(qt, w, q8, blockCount);
    }

    /// <summary>Scalar reference dot for <paramref name="qt"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotLegacyQuantScalar(QuantizationType qt, byte* w, byte* q8, int blockCount)
        => qt switch
        {
            QuantizationType.Q4_0 => VecDotQ4_0Q8_1Scalar(w, q8, blockCount),
            QuantizationType.Q4_1 => VecDotQ4_1Q8_1Scalar(w, q8, blockCount),
            QuantizationType.Q5_1 => VecDotQ5_1Q8_1Scalar(w, q8, blockCount),
            QuantizationType.IQ4_NL => VecDotIQ4_NLQ8_1Scalar(w, q8, blockCount),
            _ => throw new ArgumentOutOfRangeException(nameof(qt), qt, "not a packed legacy quant")
        };

    /// <summary>
    /// Row-major <c>result[row] = dot(weights[row], xQ8)</c> for one packed legacy quant.
    /// Picks the SSSE3 4-rows-per-lane kernel when AVX2 is absent; AVX2 hardware runs the
    /// single-row kernel, which is already bit-exact with the scalar reference.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsLegacyQuant(QuantizationType qt, byte* weights, byte* xQ8,
        float* result, int m, int blockCount)
    {
        nint rowBytes = (nint)blockCount * LegacyBlockBytes(qt);

        if (Avx2.IsSupported)
        {
            int row = 0;
            for (; row + 3 < m; row += 4)
                VecDotLegacyQuantAvx2_4Rows(qt, weights + row * rowBytes, rowBytes, xQ8, blockCount, result + row);
            for (; row < m; row++)
                result[row] = VecDotLegacyAvx2(qt, weights + row * rowBytes, xQ8, blockCount);
            return;
        }

        if (Ssse3.IsSupported)
        {
            ComputeRowsLegacyQuantSse(qt, weights, xQ8, result, m, blockCount);
            return;
        }

        for (int row = 0; row < m; row++)
            result[row] = VecDotLegacyQuantScalar(qt, weights + row * rowBytes, xQ8, blockCount);
    }

    /// <summary>Token columns a multi-column step folds together — one weight unpack feeds all of them.</summary>
    private const int LegacyMultiTokenTile = 4;

    /// <summary>
    /// Multi-column variant of <see cref="ComputeRowsLegacyQuant"/>:
    /// <c>c[t·mTotal + row] = dot(weights[row], xQ8[t])</c> for <c>t &lt; n</c>.
    ///
    /// <para>A per-token loop around the single-column kernel would unpack every weight block
    /// <c>n</c> times, which is exactly why the dequantize-then-GEMM path it replaces was
    /// competitive at prefill — that path pays the unpack once per row and reuses the F32 row
    /// across all columns. Here each block is unpacked once and dotted against
    /// <see cref="LegacyMultiTokenTile"/> Q8_1 columns, so the unpack cost is amortized without
    /// ever materializing an F32 row. Per-column results stay bit-identical to the single-column
    /// kernels: the block sums are exact ints and the float recurrence runs in block order.</para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsLegacyQuantMulti(QuantizationType qt, byte* weights, byte* xQ8,
        int q8RowBytes, float* c, int mTotal, int m, int n, int blockCount)
    {
        if (n == 1)
        {
            ComputeRowsLegacyQuant(qt, weights, xQ8, c, m, blockCount);
            return;
        }

        if (Avx2.IsSupported)
        {
            ComputeRowsLegacyMultiAvx2(qt, weights, xQ8, q8RowBytes, c, mTotal, m, n, blockCount);
            return;
        }
        if (Ssse3.IsSupported)
        {
            ComputeRowsLegacyMultiSse(qt, weights, xQ8, q8RowBytes, c, mTotal, m, n, blockCount);
            return;
        }

        nint rowBytes = (nint)blockCount * LegacyBlockBytes(qt);
        for (int t = 0; t < n; t++)
            for (int row = 0; row < m; row++)
                c[(long)t * mTotal + row] =
                    VecDotLegacyQuantScalar(qt, weights + row * rowBytes, xQ8 + (long)t * q8RowBytes, blockCount);
    }

    /// <summary>The unsigned PMADDUBSW operand for one block, plus the sign source IQ4_NL needs.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> UnpackLegacyAvx2(QuantizationType qt, byte* wb, Vector256<sbyte> lut,
        out Vector256<sbyte> signSrc)
    {
        switch (qt)
        {
            case QuantizationType.Q4_0:
                signSrc = default;
                return UnpackNibblesAvx2(wb + 2);
            case QuantizationType.Q4_1:
                signSrc = default;
                return UnpackNibblesAvx2(wb + 4);
            case QuantizationType.Q5_1:
                signSrc = default;
                return Avx2.Or(UnpackNibblesAvx2(wb + 8), ExtractQ5HighBits(Unsafe.ReadUnaligned<uint>(wb + 4)));
            default:
                signSrc = Avx2.Shuffle(lut, UnpackNibblesAvx2(wb + 2).AsSByte());
                return Avx2.Abs(signSrc);
        }
    }

    /// <inheritdoc cref="ComputeRowsLegacyQuantMulti"/>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeRowsLegacyMultiAvx2(QuantizationType qt, byte* weights, byte* xQ8,
        int q8RowBytes, float* c, int mTotal, int m, int n, int blockCount)
    {
        int blockBytes = LegacyBlockBytes(qt);
        bool hasMin = qt is QuantizationType.Q4_1 or QuantizationType.Q5_1;
        bool isIq4 = qt == QuantizationType.IQ4_NL;
        nint rowBytes = (nint)blockCount * blockBytes;
        Vector256<short> ones = Vector256.Create((short)1);
        Vector256<sbyte> lut = Iq4NlLutAvx2;

        Vector256<float>* acc = stackalloc Vector256<float>[LegacyMultiTokenTile];
        float* off = stackalloc float[LegacyMultiTokenTile];

        for (int row = 0; row < m; row++)
        {
            byte* wRow = weights + row * rowBytes;
            for (int t0 = 0; t0 < n; t0 += LegacyMultiTokenTile)
            {
                int nt = Math.Min(LegacyMultiTokenTile, n - t0);
                for (int j = 0; j < nt; j++) { acc[j] = Vector256<float>.Zero; off[j] = 0; }

                for (int block = 0; block < blockCount; block++)
                {
                    byte* wb = wRow + block * blockBytes;
                    Vector256<byte> q = UnpackLegacyAvx2(qt, wb, lut, out Vector256<sbyte> signSrc);
                    float d = (float)Unsafe.ReadUnaligned<Half>(wb);
                    float offd = hasMin ? (float)Unsafe.ReadUnaligned<Half>(wb + 2) : d;

                    for (int j = 0; j < nt; j++)
                    {
                        byte* xb = xQ8 + (long)(t0 + j) * q8RowBytes + block * Q8_1BlockBytes;
                        Vector256<sbyte> q8v = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 4);
                        if (isIq4) q8v = Avx2.Sign(q8v, signSrc);
                        Vector256<int> p = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q, q8v), ones);
                        Vector256<float> scale = Vector256.Create(d * (float)Unsafe.ReadUnaligned<Half>(xb));
                        acc[j] = Avx.Add(acc[j], Avx.Multiply(scale, Avx.ConvertToVector256Single(p)));
                        if (!isIq4) off[j] += offd * (float)Unsafe.ReadUnaligned<Half>(xb + 2);
                    }
                }

                for (int j = 0; j < nt; j++)
                {
                    float sumf = HorizontalSumAvx2Float(acc[j]);
                    c[(long)(t0 + j) * mTotal + row] = isIq4 ? sumf
                        : hasMin ? sumf + off[j]
                        : sumf - 8.0f * off[j];
                }
            }
        }
    }

    /// <summary>
    /// Largest column count for which the packed dot beats dequantize-then-GEMM.
    ///
    /// <para>At <c>n == 1</c> (decode) the packed dot is the whole win: nothing amortizes the
    /// dequantize, so <c>GemvDequantRows</c> pays a full F32 expansion of the weight matrix per
    /// token. Measured on SmolLM-135M pure fixtures (Zen 5, 32 threads, <c>bench -n 32</c>):
    /// Q4_0 100 → 272 tok/s, Q4_1 101 → 240, Q5_1 74 → 209, IQ4_NL 100 → 236.</para>
    ///
    /// <para>At prefill it inverts. <c>GemmDequantRows</c> decodes a row <em>once</em> and then
    /// runs <c>TensorPrimitives.Dot</c> — 16-wide AVX-512 FMA, ~36 instructions for a k = 576 row —
    /// against every column, so its per-column cost is far below the ~5 integer ops per 32
    /// elements a packed dot needs, however well the unpack is amortized. Measured at
    /// <c>-p 32</c>: Q4_0 prefill 808 → 197 tok/s with the packed GEMM, i.e. a 4x regression.
    /// So the packed path is taken for decode only and prefill keeps the existing kernel.</para>
    ///
    /// <para>Beating dequantize-then-GEMM at prefill needs an MMQ-style repacked-weight GEMM
    /// (llama.cpp's approach) rather than a column loop over a row kernel — that is a separate
    /// piece of work, not a threshold to tune.</para>
    /// </summary>
    private const int PackedLegacyMaxColumns = 1;

    /// <summary>
    /// True when a GEMM of <paramref name="n"/> columns in <paramref name="qt"/> will take the
    /// packed path. Input pre-quantization is only worth doing when it will — quantizing to Q8_1
    /// for a prefill that then runs <c>GemmDequantRows</c> off the F32 input is pure waste, and
    /// measured a ~25% prefill loss on the SmolLM-135M pure fixtures.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool UsesPackedLegacyDot(QuantizationType qt, int n)
        => HasPackedLegacyDot(qt) && n <= PackedLegacyMaxColumns;

    /// <summary>
    /// Dispatches a packed legacy quant to whichever kernel measurement favours at this
    /// <paramref name="n"/> — see <see cref="PackedLegacyMaxColumns"/>. Callers that used to call
    /// <c>GemmDequantRows</c> unconditionally for these formats should call this instead.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void GemmLegacyQuantOrDequant(byte* weights, QuantizationType qt, float* b, float* c,
        int m, int k, int n, ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (n <= PackedLegacyMaxColumns)
            GemmLegacyQuant(weights, qt, b, c, m, k, n, pool, preQuantizedInput);
        else
            GemmDequantRows(weights, qt, b, c, m, k, n, pool);
    }

    // ──────────────────── GEMV ────────────────────

    /// <summary>
    /// Packed GEMV for Q4_0/Q4_1/Q5_1/IQ4_NL: <c>result[m] = W[m,k] · x[k]</c> with <c>x</c>
    /// quantized to Q8_1. Replaces the dequantize-to-F32 path in <c>GemvDequantRows</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvLegacyQuant(byte* weights, QuantizationType qt, float* x, float* result,
                                       int m, int k)
    {
        if (k % Q8_1GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_1GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;
        int xQ8Bytes = blockCount * Q8_1BlockBytes;

        if (xQ8Bytes <= StackAllocThreshold)
        {
            byte* xQ8 = stackalloc byte[xQ8Bytes];
            QuantizeF32ToQ8_1(x, xQ8, k);
            ComputeRowsLegacyQuant(qt, weights, xQ8, result, m, blockCount);
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
        fixed (byte* xQ8 = rented)
        {
            QuantizeF32ToQ8_1(x, xQ8, k);
            ComputeRowsLegacyQuant(qt, weights, xQ8, result, m, blockCount);
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    /// <summary>Packed legacy-quant GEMV with optional row parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvLegacyQuant(byte* weights, QuantizationType qt, float* x, float* result,
                                       int m, int k, ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvLegacyQuant(weights, qt, x, result, m, k);
            return;
        }

        if (k % Q8_1GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_1GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;
        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, blockCount * Q8_1BlockBytes);
        QuantizeF32ToQ8_1(x, xQ8, k);

        var ctx = new LegacyQuantRowsCtx
        {
            Weights = weights, XQ8 = xQ8, Result = result,
            M = m, BlockCount = blockCount, Qt = qt
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsLegacyQuantWorker);
    }

    // ──────────────────── GEMM ────────────────────

    /// <summary>
    /// Packed legacy-quant GEMM: <c>C[n,m] = B[n,k] × W[m,k]^T</c>. <paramref name="preQuantizedInput"/>,
    /// when non-null, is the already-Q8_1-quantized <c>B</c> (see <c>TransformerModel.QuantizeInput</c>).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmLegacyQuant(byte* weights, QuantizationType qt, float* b, float* c,
                                       int m, int k, int n, byte* preQuantizedInput = null)
    {
        if (k % Q8_1GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_1GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;
        int q8RowBytes = blockCount * Q8_1BlockBytes;
        int wRowBytes = blockCount * LegacyBlockBytes(qt);

        if (n == 1)
        {
            if (preQuantizedInput != null)
                ComputeRowsLegacyQuant(qt, weights, preQuantizedInput, c, m, blockCount);
            else
                GemvLegacyQuant(weights, qt, b, c, m, k);
            return;
        }

        if (preQuantizedInput != null)
        {
            GemmLegacyQuantTiles(qt, weights, preQuantizedInput, c, 0, m, m, n, blockCount,
                ComputeTileM(wRowBytes), wRowBytes, q8RowBytes);
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(n * q8RowBytes);
        fixed (byte* inputQ8 = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_1(b + t * k, inputQ8 + t * q8RowBytes, k);

            GemmLegacyQuantTiles(qt, weights, inputQ8, c, 0, m, m, n, blockCount,
                ComputeTileM(wRowBytes), wRowBytes, q8RowBytes);
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    /// <summary>Packed legacy-quant GEMM with optional tile parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmLegacyQuant(byte* weights, QuantizationType qt, float* b, float* c,
                                       int m, int k, int n, ComputeThreadPool? pool,
                                       byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmLegacyQuant(weights, qt, b, c, m, k, n, preQuantizedInput);
            return;
        }

        if (k % Q8_1GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_1GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                var gemvCtx = new LegacyQuantRowsCtx
                {
                    Weights = weights, XQ8 = preQuantizedInput, Result = c,
                    M = m, BlockCount = blockCount, Qt = qt
                };
                pool.Dispatch((nint)(&gemvCtx), &ComputeRowsLegacyQuantWorker);
            }
            else
            {
                GemvLegacyQuant(weights, qt, b, c, m, k, pool);
            }
            return;
        }

        int q8RowBytes = blockCount * Q8_1BlockBytes;
        int wRowBytes = blockCount * LegacyBlockBytes(qt);
        int tileM = ComputeTileM(wRowBytes);

        if (preQuantizedInput != null)
        {
            if (m < ParallelMinRows)
            {
                GemmLegacyQuant(weights, qt, b, c, m, k, n, preQuantizedInput);
                return;
            }

            var ctx = new LegacyQuantGemmCtx
            {
                Weights = weights, InputQ8 = preQuantizedInput, C = c,
                M = m, N = n, BlockCount = blockCount, TileM = tileM,
                WRowBytes = wRowBytes, Q8RowBytes = q8RowBytes, Qt = qt
            };
            pool.Dispatch((nint)(&ctx), &GemmLegacyQuantWorker);
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(n * q8RowBytes);
        fixed (byte* inputQ8 = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_1(b + t * k, inputQ8 + t * q8RowBytes, k);

            if (m < ParallelMinRows)
            {
                GemmLegacyQuantTiles(qt, weights, inputQ8, c, 0, m, m, n, blockCount,
                    tileM, wRowBytes, q8RowBytes);
            }
            else
            {
                var ctx = new LegacyQuantGemmCtx
                {
                    Weights = weights, InputQ8 = inputQ8, C = c,
                    M = m, N = n, BlockCount = blockCount, TileM = tileM,
                    WRowBytes = wRowBytes, Q8RowBytes = q8RowBytes, Qt = qt
                };
                pool.Dispatch((nint)(&ctx), &GemmLegacyQuantWorker);
            }
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    /// <summary>
    /// Runs the M-tiles <c>[rowStart, rowStart + rowCount)</c> of a packed legacy-quant GEMM.
    /// <paramref name="mTotal"/> is the full row count — it is the column stride of <c>C</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GemmLegacyQuantTiles(QuantizationType qt, byte* weights, byte* inputQ8, float* c,
        int rowStart, int rowCount, int mTotal, int n, int blockCount,
        int tileM, int wRowBytes, int q8RowBytes)
    {
        int end = rowStart + rowCount;
        for (int mStart = rowStart; mStart < end; mStart += tileM)
        {
            int tileRows = Math.Min(tileM, end - mStart);
            byte* tileWeights = weights + (long)mStart * wRowBytes;
            ComputeRowsLegacyQuantMulti(qt, tileWeights, inputQ8, q8RowBytes,
                c + mStart, mTotal, tileRows, n, blockCount);
        }
    }

    // ──────────────────── Thread-pool contexts and workers ────────────────────

    private struct LegacyQuantRowsCtx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* Result;
        public int M;
        public int BlockCount;
        public QuantizationType Qt;
    }

    private struct LegacyQuantGemmCtx
    {
        public byte* Weights;
        public byte* InputQ8;
        public float* C;
        public int M;
        public int N;
        public int BlockCount;
        public int TileM;
        public int WRowBytes;
        public int Q8RowBytes;
        public QuantizationType Qt;
    }

    private static void ComputeRowsLegacyQuantWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<LegacyQuantRowsCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        nint rowBytes = (nint)ctx.BlockCount * LegacyBlockBytes(ctx.Qt);
        ComputeRowsLegacyQuant(ctx.Qt, ctx.Weights + start * rowBytes, ctx.XQ8,
            ctx.Result + start, count, ctx.BlockCount);
    }

    /// <summary>
    /// Partitions the GEMM over <em>rows</em>, not over M-tiles. Tiling alone does not parallelize:
    /// <c>ComputeTileM</c> sizes a tile to fit cache, so a small model's whole weight matrix is one
    /// tile and a tile-partitioned dispatch would leave 31 of 32 threads idle. Rows are independent
    /// and each thread still tiles its own range, so results are bit-identical to the serial path.
    /// </summary>
    private static void GemmLegacyQuantWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<LegacyQuantGemmCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmLegacyQuantTiles(ctx.Qt, ctx.Weights, ctx.InputQ8, ctx.C, start, count,
            ctx.M, ctx.N, ctx.BlockCount, ctx.TileM, ctx.WRowBytes, ctx.Q8RowBytes);
    }
}
