using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Faster Q4_K x Q8_K dot kernels (R4 / strided 4-row form), bit-identical to
/// <see cref="MatMul.VecDotQ4_K_Q8_K_4Rows(byte*, byte*, byte*, byte*, byte*, int, int, float*)"/>.
/// </summary>
/// <remarks>
/// <para>
/// Profiling Gemma-4 E2B on CPU (issue #933) showed the 4-row kernel spending ~65 cycles per
/// (row, super-block, token) where the 8 <c>vpmaddubsw</c>/<c>vpmaddwd</c> pairs need ~8: the rest is
/// weight-only bookkeeping redone for every token and every call - the scalar 12-byte scale unpack
/// (<c>UnpackQ4Q5Scales</c>, two byte-store loops), two Half-to-float conversions, and a scalar
/// 8-iteration min-correction loop.
/// </para>
/// <para>
/// Here that work is (a) done with the 32-bit mask trick instead of byte loops, (b) the min
/// correction becomes one <c>vpmaddwd</c> plus a horizontal int sum, and (c) for the multi-token
/// form the weight-side work (scale unpack, nibble split, scale broadcasts) is hoisted out of the
/// token loop: <see cref="Rows4Tokens4"/> decodes a row's super-block once and applies it to four
/// activation rows. The integer arithmetic is exact and the float accumulation order per output
/// element is unchanged, so results are bit-identical to the old kernel for any batch size (#530).
/// </para>
/// </remarks>
public static unsafe partial class MatMul
{
    /// <summary>
    /// 4 rows x 1 token. <paramref name="w0"/> is the first row's first super-block, rows are
    /// <see cref="Q4_K_BlockBytes"/> apart within a group, super-blocks of one row are
    /// <paramref name="wStride"/> apart.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4KRows4Tokens1(byte* w0, byte* w1, byte* w2, byte* w3, byte* q8k,
        int superBlockCount, int wStride, float* results)
    {
        results[0] = Q4KRowTok1(w0, q8k, superBlockCount, wStride);
        results[1] = Q4KRowTok1(w1, q8k, superBlockCount, wStride);
        results[2] = Q4KRowTok1(w2, q8k, superBlockCount, wStride);
        results[3] = Q4KRowTok1(w3, q8k, superBlockCount, wStride);
    }

    /// <summary>
    /// 4 rows x 4 tokens: <paramref name="groupBase"/> is an R4 group (4 adjacent rows' super-blocks,
    /// row stride <see cref="Q4_K_BlockBytes"/>, super-block stride 4x that). Writes
    /// <c>results[t * ldc + r]</c> for <c>t &lt; 4</c>, <c>r &lt; 4</c>. Token rows of the Q8_K input are
    /// <paramref name="q8RowBytes"/> apart.
    /// </summary>
    [SkipLocalsInit]
    internal static void Rows4Tokens4(byte* groupBase, int superBlockCount, byte* q8k, int q8RowBytes,
        float* results, int ldc)
    {
        int wStride = 4 * Q4_K_BlockBytes;
        for (int r = 0; r < 4; r++)
        {
            Q4KRowTok4(groupBase + r * Q4_K_BlockBytes, q8k, q8RowBytes, superBlockCount, wStride,
                results + r, ldc);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackQ4KScalesFast(byte* s12, out Vector128<short> sc16, out Vector256<short> mnDup)
    {
        uint u0 = Unsafe.ReadUnaligned<uint>(s12);
        uint u1 = Unsafe.ReadUnaligned<uint>(s12 + 4);
        uint u2 = Unsafe.ReadUnaligned<uint>(s12 + 8);
        uint ut3 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
        uint ut2 = u1 & 0x3f3f3f3fu;
        uint ut1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
        uint ut0 = u0 & 0x3f3f3f3fu;

        Vector128<byte> scB = Vector128.CreateScalarUnsafe(((ulong)ut1 << 32) | ut0).AsByte();
        Vector128<byte> mnB = Vector128.CreateScalarUnsafe(((ulong)ut3 << 32) | ut2).AsByte();
        sc16 = Sse41.ConvertToVector128Int16(scB);
        Vector128<short> mn16 = Sse41.ConvertToVector128Int16(mnB);
        mnDup = Vector256.Create(Sse2.UnpackLow(mn16, mn16), Sse2.UnpackHigh(mn16, mn16));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HorizontalSumInt(Vector256<int> v)
    {
        Vector128<int> a = Sse2.Add(v.GetLower(), v.GetUpper());
        a = Sse2.Add(a, Sse2.Shuffle(a, 0x4E));
        a = Sse2.Add(a, Sse2.Shuffle(a, 0xB1));
        return a.ToScalar();
    }

    /// <summary>One sub-block pair: sumi += scLo*maddubs(lo,q8Lo) + scHi*maddubs(hi,q8Hi) (exact int math).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Q4KAccum(Vector256<int> sumi, Vector256<byte> lo, Vector256<byte> hi,
        byte* q8, Vector256<short> scLo, Vector256<short> scHi)
    {
        Vector256<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8);
        Vector256<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8 + 32);
        Vector256<short> pLo = Avx2.MultiplyAddAdjacent(lo, q8Lo);
        Vector256<short> pHi = Avx2.MultiplyAddAdjacent(hi, q8Hi);
        if (AvxVnni.IsSupported)
        {
            sumi = AvxVnni.MultiplyWideningAndAdd(sumi, pLo, scLo);
            return AvxVnni.MultiplyWideningAndAdd(sumi, pHi, scHi);
        }
        return Avx2.Add(sumi, Avx2.Add(Avx2.MultiplyAddAdjacent(pLo, scLo), Avx2.MultiplyAddAdjacent(pHi, scHi)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Q4KFma(float scale, Vector256<int> sumi, Vector256<float> acc)
        => Fma.IsSupported
            ? Fma.MultiplyAdd(Vector256.Create(scale), Avx.ConvertToVector256Single(sumi), acc)
            : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), Avx.ConvertToVector256Single(sumi)));

    [SkipLocalsInit]
    private static float Q4KRowTok1(byte* w, byte* q8k, int superBlockCount, int wStride)
    {
        if (!Avx2.IsSupported || !Sse41.IsSupported)
            return VecDotQ4_K_Q8_KPortable(w, q8k, superBlockCount, wStride);

        Vector256<float> acc = Vector256<float>.Zero;
        float sumMin = 0;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            float d4 = (float)Unsafe.ReadUnaligned<Half>(w);
            float dm = (float)Unsafe.ReadUnaligned<Half>(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);

            Vector256<short> bs = Unsafe.ReadUnaligned<Vector256<short>>(q8k + 260);
            int mc = HorizontalSumInt(Avx2.MultiplyAddAdjacent(bs, mnDup));
            sumMin += dm * d8 * mc;

            byte* qs = w + 16;
            byte* q8 = q8k + 4;
            Vector256<int> sumi = Vector256<int>.Zero;
            for (int j = 0; j < 4; j++)
            {
                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + j * 32);
                Vector256<byte> lo = Avx2.And(raw, mask0F);
                Vector256<byte> hi = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                sumi = Q4KAccum(sumi, lo, hi, q8 + j * 64,
                    Avx2.BroadcastScalarToVector256(sc + j * 2), Avx2.BroadcastScalarToVector256(sc + j * 2 + 1));
            }

            acc = Q4KFma(d4 * d8, sumi, acc);
            w += wStride;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc) - sumMin;
    }

    [SkipLocalsInit]
    private static void Q4KRowTok4(byte* w, byte* q8k, int q8RowBytes, int superBlockCount, int wStride,
        float* results, int ldc)
    {
        byte* qa0 = q8k, qa1 = q8k + q8RowBytes, qa2 = q8k + 2L * q8RowBytes, qa3 = q8k + 3L * q8RowBytes;

        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        Vector128<float> sm = Vector128<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d4 = (float)Unsafe.ReadUnaligned<Half>(w);
            float dm = (float)Unsafe.ReadUnaligned<Half>(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);

            float d80 = Unsafe.ReadUnaligned<float>(qa0), d81 = Unsafe.ReadUnaligned<float>(qa1);
            float d82 = Unsafe.ReadUnaligned<float>(qa2), d83 = Unsafe.ReadUnaligned<float>(qa3);

            // Min corrections for the 4 tokens at once: per-token int sums via a hadd tree (exact),
            // then (dm * d8) * (float)mc in the same operand order as the scalar kernel.
            Vector256<int> m0 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa0 + 260), mnDup);
            Vector256<int> m1 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa1 + 260), mnDup);
            Vector256<int> m2 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa2 + 260), mnDup);
            Vector256<int> m3 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa3 + 260), mnDup);
            Vector256<int> h = Avx2.HorizontalAdd(Avx2.HorizontalAdd(m0, m1), Avx2.HorizontalAdd(m2, m3));
            Vector128<int> mc4 = Sse2.Add(h.GetLower(), h.GetUpper());
            Vector128<float> dmd8 = Sse.Multiply(Vector128.Create(dm), Vector128.Create(d80, d81, d82, d83));
            sm = Sse.Add(sm, Sse.Multiply(dmd8, Sse2.ConvertToVector128Single(mc4)));

            byte* qs = w + 16;
            Vector256<int> s0 = Vector256<int>.Zero, s1 = s0, s2 = s0, s3 = s0;
            for (int j = 0; j < 4; j++)
            {
                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + j * 32);
                Vector256<byte> lo = Avx2.And(raw, mask0F);
                Vector256<byte> hi = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                Vector256<short> scLo = Avx2.BroadcastScalarToVector256(sc + j * 2);
                Vector256<short> scHi = Avx2.BroadcastScalarToVector256(sc + j * 2 + 1);
                s0 = Q4KAccum(s0, lo, hi, qa0 + 4 + j * 64, scLo, scHi);
                s1 = Q4KAccum(s1, lo, hi, qa1 + 4 + j * 64, scLo, scHi);
                s2 = Q4KAccum(s2, lo, hi, qa2 + 4 + j * 64, scLo, scHi);
                s3 = Q4KAccum(s3, lo, hi, qa3 + 4 + j * 64, scLo, scHi);
            }

            acc0 = Q4KFma(d4 * d80, s0, acc0);
            acc1 = Q4KFma(d4 * d81, s1, acc1);
            acc2 = Q4KFma(d4 * d82, s2, acc2);
            acc3 = Q4KFma(d4 * d83, s3, acc3);

            w += wStride;
            qa0 += Q8_K_BlockBytes; qa1 += Q8_K_BlockBytes; qa2 += Q8_K_BlockBytes; qa3 += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0) - sm.GetElement(0);
        results[ldc] = HorizontalSumAvx2Float(acc1) - sm.GetElement(1);
        results[2L * ldc] = HorizontalSumAvx2Float(acc2) - sm.GetElement(2);
        results[3L * ldc] = HorizontalSumAvx2Float(acc3) - sm.GetElement(3);
    }
}
