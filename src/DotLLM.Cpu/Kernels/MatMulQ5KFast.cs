using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Faster Q5_K x Q8_K dot kernels (strided 4-row form), bit-identical to <c>VecDotQ5_K_Q8_K_4Rows</c>.
/// Same restructuring as <c>MatMulQ4KFast</c> (#933): scale/min handling hoisted and vectorised, the
/// 4-token form decodes a row's 5-bit values once for four activation rows.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>4 rows x 1 token Q5_K x Q8_K.</summary>
    [SkipLocalsInit]
    internal static void Q5KRows4Tokens1(byte* w0, byte* w1, byte* w2, byte* w3, byte* q8k,
        int superBlockCount, int wStride, float* results)
    {
        results[0] = Q5KRowTok1(w0, q8k, superBlockCount, wStride);
        results[1] = Q5KRowTok1(w1, q8k, superBlockCount, wStride);
        results[2] = Q5KRowTok1(w2, q8k, superBlockCount, wStride);
        results[3] = Q5KRowTok1(w3, q8k, superBlockCount, wStride);
    }

    /// <summary>Q5_K counterpart of <see cref="Rows4Tokens4"/>.</summary>
    [SkipLocalsInit]
    internal static void Q5KRows4Tokens4(byte* groupBase, int superBlockCount, byte* q8k, int q8RowBytes,
        float* results, int ldc)
    {
        int wStride = 4 * Q5_K_BlockBytes;
        for (int r = 0; r < 4; r++)
            Q5KRowTok4(groupBase + r * Q5_K_BlockBytes, q8k, q8RowBytes, superBlockCount, wStride,
                results + r, ldc);
    }

    /// <summary>Decodes the lo/hi 32-element 5-bit vectors of sub-block pair <paramref name="j"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DecodeQ5K(byte* qs, Vector256<byte> qhV, int j, Vector256<byte> m0F,
        out Vector256<byte> lo5, out Vector256<byte> hi5)
    {
        byte loBm = (byte)(1 << (j * 2));
        byte hiBm = (byte)(1 << (j * 2 + 1));
        Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + j * 32);
        Vector256<byte> loNib = Avx2.And(raw, m0F);
        Vector256<byte> hiNib = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), m0F);
        Vector256<byte> loMask = Vector256.Create(loBm), hiMask = Vector256.Create(hiBm);
        Vector256<byte> sixteen = Vector256.Create((byte)16);
        Vector256<byte> loHas = Avx2.CompareEqual(Avx2.And(qhV, loMask), loMask);
        Vector256<byte> hiHas = Avx2.CompareEqual(Avx2.And(qhV, hiMask), hiMask);
        lo5 = Avx2.Or(loNib, Avx2.And(loHas, sixteen));
        hi5 = Avx2.Or(hiNib, Avx2.And(hiHas, sixteen));
    }

    [SkipLocalsInit]
    private static float Q5KRowTok1(byte* w, byte* q8k, int superBlockCount, int wStride)
    {
        if (!Avx2.IsSupported || !Sse41.IsSupported)
            return VecDotQ5_K_Q8_KPortable(w, q8k, superBlockCount, wStride);

        Vector256<float> acc = Vector256<float>.Zero;
        float sumMin = 0;
        Vector256<byte> m0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            float d5 = (float)Unsafe.ReadUnaligned<Half>(w);
            float dm = (float)Unsafe.ReadUnaligned<Half>(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);

            int mc = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(q8k + 260), mnDup));
            sumMin += dm * d8 * mc;

            byte* qs = w + 48;
            byte* q8 = q8k + 4;
            Vector256<byte> qhV = Unsafe.ReadUnaligned<Vector256<byte>>(w + 16);
            Vector256<int> sumi = Vector256<int>.Zero;
            for (int j = 0; j < 4; j++)
            {
                DecodeQ5K(qs, qhV, j, m0F, out Vector256<byte> lo5, out Vector256<byte> hi5);
                sumi = Q4KAccum(sumi, lo5, hi5, q8 + j * 64,
                    Avx2.BroadcastScalarToVector256(sc + j * 2), Avx2.BroadcastScalarToVector256(sc + j * 2 + 1));
            }

            acc = Q4KFma(d5 * d8, sumi, acc);
            w += wStride;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc) - sumMin;
    }

    [SkipLocalsInit]
    private static void Q5KRowTok4(byte* w, byte* q8k, int q8RowBytes, int superBlockCount, int wStride,
        float* results, int ldc)
    {
        byte* qa0 = q8k, qa1 = q8k + q8RowBytes, qa2 = q8k + 2L * q8RowBytes, qa3 = q8k + 3L * q8RowBytes;
        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        Vector128<float> sm = Vector128<float>.Zero;
        Vector256<byte> m0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d5 = (float)Unsafe.ReadUnaligned<Half>(w);
            float dm = (float)Unsafe.ReadUnaligned<Half>(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);

            float d80 = Unsafe.ReadUnaligned<float>(qa0), d81 = Unsafe.ReadUnaligned<float>(qa1);
            float d82 = Unsafe.ReadUnaligned<float>(qa2), d83 = Unsafe.ReadUnaligned<float>(qa3);

            Vector256<int> m0 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa0 + 260), mnDup);
            Vector256<int> m1 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa1 + 260), mnDup);
            Vector256<int> m2 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa2 + 260), mnDup);
            Vector256<int> m3 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa3 + 260), mnDup);
            Vector256<int> h = Avx2.HorizontalAdd(Avx2.HorizontalAdd(m0, m1), Avx2.HorizontalAdd(m2, m3));
            Vector128<int> mc4 = Sse2.Add(h.GetLower(), h.GetUpper());
            Vector128<float> dmd8 = Sse.Multiply(Vector128.Create(dm), Vector128.Create(d80, d81, d82, d83));
            sm = Sse.Add(sm, Sse.Multiply(dmd8, Sse2.ConvertToVector128Single(mc4)));

            byte* qs = w + 48;
            Vector256<byte> qhV = Unsafe.ReadUnaligned<Vector256<byte>>(w + 16);
            Vector256<int> s0 = Vector256<int>.Zero, s1 = s0, s2 = s0, s3 = s0;
            for (int j = 0; j < 4; j++)
            {
                DecodeQ5K(qs, qhV, j, m0F, out Vector256<byte> lo5, out Vector256<byte> hi5);
                Vector256<short> scLo = Avx2.BroadcastScalarToVector256(sc + j * 2);
                Vector256<short> scHi = Avx2.BroadcastScalarToVector256(sc + j * 2 + 1);
                s0 = Q4KAccum(s0, lo5, hi5, qa0 + 4 + j * 64, scLo, scHi);
                s1 = Q4KAccum(s1, lo5, hi5, qa1 + 4 + j * 64, scLo, scHi);
                s2 = Q4KAccum(s2, lo5, hi5, qa2 + 4 + j * 64, scLo, scHi);
                s3 = Q4KAccum(s3, lo5, hi5, qa3 + 4 + j * 64, scLo, scHi);
            }

            acc0 = Q4KFma(d5 * d80, s0, acc0);
            acc1 = Q4KFma(d5 * d81, s1, acc1);
            acc2 = Q4KFma(d5 * d82, s2, acc2);
            acc3 = Q4KFma(d5 * d83, s3, acc3);

            w += wStride;
            qa0 += Q8_K_BlockBytes; qa1 += Q8_K_BlockBytes; qa2 += Q8_K_BlockBytes; qa3 += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0) - sm.GetElement(0);
        results[ldc] = HorizontalSumAvx2Float(acc1) - sm.GetElement(1);
        results[2L * ldc] = HorizontalSumAvx2Float(acc2) - sm.GetElement(2);
        results[3L * ldc] = HorizontalSumAvx2Float(acc3) - sm.GetElement(3);
    }
}
