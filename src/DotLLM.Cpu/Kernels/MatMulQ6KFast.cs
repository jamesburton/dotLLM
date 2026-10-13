using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Faster Q6_K x Q8_K dot kernels (strided 4-row form), bit-identical to
/// <c>VecDotQ6_K_Q8_K_4Rows</c>. Same restructuring as <c>MatMulQ4KFast</c> (#933): weight-side decode
/// and scale handling hoisted out of the token loop for the 4-token form, vectorised bias correction,
/// <c>vpdpwssd</c> accumulation where AVX-VNNI is present (exact integer math either way).
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>4 rows x 1 token Q6_K x Q8_K.</summary>
    [SkipLocalsInit]
    internal static void Q6KRows4Tokens1(byte* w0, byte* w1, byte* w2, byte* w3, byte* q8k,
        int superBlockCount, int wStride, float* results)
    {
        if (!FastKQuantIsa)
        {
            VecDotQ6_K_Q8_K_4RowsReference(w0, w1, w2, w3, q8k, superBlockCount, wStride, results);
            return;
        }

        results[0] = Q6KRowTok1(w0, q8k, superBlockCount, wStride);
        results[1] = Q6KRowTok1(w1, q8k, superBlockCount, wStride);
        results[2] = Q6KRowTok1(w2, q8k, superBlockCount, wStride);
        results[3] = Q6KRowTok1(w3, q8k, superBlockCount, wStride);
    }

    /// <summary>Q6_K counterpart of <see cref="Rows4Tokens4"/>.</summary>
    [SkipLocalsInit]
    internal static void Q6KRows4Tokens4(byte* groupBase, int superBlockCount, byte* q8k, int q8RowBytes,
        float* results, int ldc)
    {
        int wStride = 4 * Q6_K_BlockBytes;
        if (!FastKQuantIsa)
        {
            float* tmp = stackalloc float[4];
            for (int t = 0; t < 4; t++)
            {
                VecDotQ6_K_Q8_K_4RowsReference(groupBase, groupBase + Q6_K_BlockBytes, groupBase + 2 * Q6_K_BlockBytes,
                    groupBase + 3 * Q6_K_BlockBytes, q8k + (long)t * q8RowBytes, superBlockCount, wStride, tmp);
                for (int r = 0; r < 4; r++) results[(long)t * ldc + r] = tmp[r];
            }
            return;
        }

        for (int r = 0; r < 4; r++)
            Q6KRowTok4(groupBase + r * Q6_K_BlockBytes, q8k, q8RowBytes, superBlockCount, wStride,
                results + r, ldc);
    }

    /// <summary>
    /// Decodes one 32-element Q6_K vector (unsigned 6-bit, 0..63) for iteration <paramref name="it"/>
    /// (0..3) of a 128-element half: ql chunk (it &amp; 1), low nibble for it &lt; 2 else high nibble,
    /// OR the 2-bit field of qh at shift 2*it placed in bits 4..5.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> DecodeQ6K(Vector256<byte> qlA, Vector256<byte> qlB, Vector256<byte> qhV,
        int it, Vector256<byte> m0F, Vector256<byte> m03)
    {
        Vector256<byte> ql = (it & 1) == 0 ? qlA : qlB;
        Vector256<byte> nib = it < 2
            ? Avx2.And(ql, m0F)
            : Avx2.And(Avx2.ShiftRightLogical(ql.AsUInt16(), 4).AsByte(), m0F);
        Vector256<byte> qhB = it switch
        {
            0 => Avx2.And(qhV, m03),
            1 => Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 2).AsByte(), m03),
            2 => Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 4).AsByte(), m03),
            _ => Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 6).AsByte(), m03),
        };
        return Avx2.Or(nib, Avx2.ShiftLeftLogical(qhB.AsUInt16(), 4).AsByte());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Q6KAccum(Vector256<int> sumi, Vector256<byte> q6u, byte* q8, Vector256<short> sv)
    {
        Vector256<short> prod = Avx2.MultiplyAddAdjacent(q6u, Unsafe.ReadUnaligned<Vector256<sbyte>>(q8));
        return AvxVnni.IsSupported
            ? AvxVnni.MultiplyWideningAndAdd(sumi, prod, sv)
            : Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(prod, sv));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Q6KScaleVec(sbyte* s, int sub)
        => Vector256.Create(Vector128.Create((short)s[sub]), Vector128.Create((short)s[sub + 1]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Q6KStep1(Vector256<int> sumi, Vector256<byte> qlA, Vector256<byte> qlB,
        Vector256<byte> qhV, int it, int half, byte* q8, sbyte* s, Vector256<byte> m0F, Vector256<byte> m03)
    {
        int g = half * 4 + it;
        return Q6KAccum(sumi, DecodeQ6K(qlA, qlB, qhV, it, m0F, m03), q8 + g * 32, Q6KScaleVec(s, 2 * g));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q6KStep4(ref Vector256<int> s0, ref Vector256<int> s1, ref Vector256<int> s2,
        ref Vector256<int> s3, Vector256<byte> qlA, Vector256<byte> qlB, Vector256<byte> qhV, int it, int half,
        byte* qa0, byte* qa1, byte* qa2, byte* qa3, sbyte* s, Vector256<byte> m0F, Vector256<byte> m03)
    {
        int g = half * 4 + it;
        Vector256<byte> q6u = DecodeQ6K(qlA, qlB, qhV, it, m0F, m03);
        Vector256<short> sv = Q6KScaleVec(s, 2 * g);
        int o = 4 + g * 32;
        s0 = Q6KAccum(s0, q6u, qa0 + o, sv);
        s1 = Q6KAccum(s1, q6u, qa1 + o, sv);
        s2 = Q6KAccum(s2, q6u, qa2 + o, sv);
        s3 = Q6KAccum(s3, q6u, qa3 + o, sv);
    }

    [SkipLocalsInit]
    private static float Q6KRowTok1(byte* w, byte* q8k, int superBlockCount, int wStride)
    {
        if (!Avx2.IsSupported)
            return VecDotQ6_K_Q8_KPortable(w, q8k, superBlockCount, wStride);

        Vector256<float> acc = Vector256<float>.Zero;
        float sumBias = 0;
        Vector256<byte> m0F = Vector256.Create((byte)0x0F), m03 = Vector256.Create((byte)0x03);

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            float d6 = (float)Unsafe.ReadUnaligned<Half>(w + 208);
            sbyte* s = (sbyte*)(w + 192);

            Vector256<short> s16 = Avx2.ConvertToVector256Int16(Unsafe.ReadUnaligned<Vector128<sbyte>>(s));
            int bc = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(q8k + 260), s16));
            sumBias += 32.0f * d6 * d8 * bc;

            byte* q8 = q8k + 4;
            Vector256<int> sumi = Vector256<int>.Zero;
            for (int half = 0; half < 2; half++)
            {
                byte* ql = w + half * 64;
                Vector256<byte> qlA = Unsafe.ReadUnaligned<Vector256<byte>>(ql);
                Vector256<byte> qlB = Unsafe.ReadUnaligned<Vector256<byte>>(ql + 32);
                Vector256<byte> qhV = Unsafe.ReadUnaligned<Vector256<byte>>(w + 128 + half * 32);
                sumi = Q6KStep1(sumi, qlA, qlB, qhV, 0, half, q8, s, m0F, m03);
                sumi = Q6KStep1(sumi, qlA, qlB, qhV, 1, half, q8, s, m0F, m03);
                sumi = Q6KStep1(sumi, qlA, qlB, qhV, 2, half, q8, s, m0F, m03);
                sumi = Q6KStep1(sumi, qlA, qlB, qhV, 3, half, q8, s, m0F, m03);
            }

            acc = Q4KFma(d6 * d8, sumi, acc);
            w += wStride;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc) - sumBias;
    }

    [SkipLocalsInit]
    private static void Q6KRowTok4(byte* w, byte* q8k, int q8RowBytes, int superBlockCount, int wStride,
        float* results, int ldc)
    {
        byte* qa0 = q8k, qa1 = q8k + q8RowBytes, qa2 = q8k + 2L * q8RowBytes, qa3 = q8k + 3L * q8RowBytes;
        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        Vector128<float> sb4 = Vector128<float>.Zero;
        Vector256<byte> m0F = Vector256.Create((byte)0x0F), m03 = Vector256.Create((byte)0x03);

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d6 = (float)Unsafe.ReadUnaligned<Half>(w + 208);
            sbyte* s = (sbyte*)(w + 192);
            float d80 = Unsafe.ReadUnaligned<float>(qa0), d81 = Unsafe.ReadUnaligned<float>(qa1);
            float d82 = Unsafe.ReadUnaligned<float>(qa2), d83 = Unsafe.ReadUnaligned<float>(qa3);

            Vector256<short> s16 = Avx2.ConvertToVector256Int16(Unsafe.ReadUnaligned<Vector128<sbyte>>(s));
            Vector256<int> b0 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa0 + 260), s16);
            Vector256<int> b1 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa1 + 260), s16);
            Vector256<int> b2 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa2 + 260), s16);
            Vector256<int> b3 = Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa3 + 260), s16);
            Vector256<int> h = Avx2.HorizontalAdd(Avx2.HorizontalAdd(b0, b1), Avx2.HorizontalAdd(b2, b3));
            Vector128<int> bc4 = Sse2.Add(h.GetLower(), h.GetUpper());
            // (32 * d6) * d8 * bc, same operand order as the scalar kernel.
            Vector128<float> k = Sse.Multiply(Vector128.Create(32.0f * d6), Vector128.Create(d80, d81, d82, d83));
            sb4 = Sse.Add(sb4, Sse.Multiply(k, Sse2.ConvertToVector128Single(bc4)));

            Vector256<int> s0 = Vector256<int>.Zero, s1 = s0, s2 = s0, s3 = s0;
            for (int half = 0; half < 2; half++)
            {
                byte* ql = w + half * 64;
                Vector256<byte> qlA = Unsafe.ReadUnaligned<Vector256<byte>>(ql);
                Vector256<byte> qlB = Unsafe.ReadUnaligned<Vector256<byte>>(ql + 32);
                Vector256<byte> qhV = Unsafe.ReadUnaligned<Vector256<byte>>(w + 128 + half * 32);
                Q6KStep4(ref s0, ref s1, ref s2, ref s3, qlA, qlB, qhV, 0, half, qa0, qa1, qa2, qa3, s, m0F, m03);
                Q6KStep4(ref s0, ref s1, ref s2, ref s3, qlA, qlB, qhV, 1, half, qa0, qa1, qa2, qa3, s, m0F, m03);
                Q6KStep4(ref s0, ref s1, ref s2, ref s3, qlA, qlB, qhV, 2, half, qa0, qa1, qa2, qa3, s, m0F, m03);
                Q6KStep4(ref s0, ref s1, ref s2, ref s3, qlA, qlB, qhV, 3, half, qa0, qa1, qa2, qa3, s, m0F, m03);
            }

            acc0 = Q4KFma(d6 * d80, s0, acc0);
            acc1 = Q4KFma(d6 * d81, s1, acc1);
            acc2 = Q4KFma(d6 * d82, s2, acc2);
            acc3 = Q4KFma(d6 * d83, s3, acc3);

            w += wStride;
            qa0 += Q8_K_BlockBytes; qa1 += Q8_K_BlockBytes; qa2 += Q8_K_BlockBytes; qa3 += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0) - sb4.GetElement(0);
        results[ldc] = HorizontalSumAvx2Float(acc1) - sb4.GetElement(1);
        results[2L * ldc] = HorizontalSumAvx2Float(acc2) - sb4.GetElement(2);
        results[3L * ldc] = HorizontalSumAvx2Float(acc3) - sb4.GetElement(3);
    }
}
