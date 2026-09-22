using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// 128-bit (SSE2/SSE3/SSSE3) tier of the K-quant x Q8_K dots — Q4_K, Q5_K, Q6_K (issue #477,
/// Westmere / pre-AVX2 hardware). Before this tier every K-quant GEMV/GEMM on a box without AVX2
/// ran the scalar per-element loop.
///
/// <para><b>Math mirrors the AVX2 tier, not the scalar one.</b> Each super-block is reduced to an
/// exact int32 "scale-in-madd" sum <c>Σ_sub sc_sub · Σ_i q_i · q8_i</c> (PMADDUBSW with the
/// unsigned weight code first, then PMADDWD by the broadcast sub-block scale), converted to float
/// once and multiplied by <c>d·d8</c>; the min / bias term comes from the Q8_K <c>bsums</c>
/// exactly as in the AVX2 kernels. The scalar tier instead rounds to float per sub-block, so the
/// two differ by float rounding only. The 128-bit and AVX2 tiers differ only in float summation
/// order across super-blocks (4 lanes vs 8).</para>
///
/// <para><b>No saturation.</b> PMADDUBSW pair sums are at most <c>2·63·128 = 16128</c> (Q6_K,
/// 6-bit code) and PMADDWD by a scale of at most 63 (Q4_K/Q5_K) or |128| (Q6_K) stays far inside
/// int32 over a 256-element super-block. Any Q8_K byte, including <c>-128</c>, is handled.</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Converts the two adjacent Half scales at <paramref name="p"/> (<c>d</c>, <c>dmin</c>) to
    /// Single in lanes 0 and 1, exactly as <c>(float)Half</c> does, without F16C: the software
    /// <c>(float)Half</c> costs ~1.2 ns each, ~12% of a 19 ns Q4_K super-block on the target.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> LoadHalfPairSse2(byte* p) =>
        HalfToSingleSse2(Sse2.UnpackLow(Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(p)).AsUInt16(),
            Vector128<ushort>.Zero).AsInt32());

    /// <summary>128-bit twin of <see cref="VecDotQ4_K_Q8_KAvx2"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ4_K_Q8_KSse(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<byte> mask0F = Vector128.Create((byte)0x0F);
        float sumMin = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            Vector128<float> dd = LoadHalfPairSse2(qk);
            float d4 = dd.ToScalar();
            float dmin = dd.GetElement(1);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qs = qk + 16;

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            int minCorr = 0;
            for (int j = 0; j < 8; j++)
                minCorr += mnBuf[j] * (bsums[j * 2] + bsums[j * 2 + 1]);
            sumMin += dmin * d8 * minCorr;

            Vector128<int> sumi = Vector128<int>.Zero;
            for (int j = 0; j < 4; j++)
            {
                Vector128<short> scLo = Vector128.Create((short)scBuf[j * 2]);
                Vector128<short> scHi = Vector128.Create((short)scBuf[j * 2 + 1]);
                for (int h = 0; h < 2; h++)
                {
                    Vector128<byte> raw = Unsafe.ReadUnaligned<Vector128<byte>>(qs + j * 32 + h * 16);
                    Vector128<byte> lo = Sse2.And(raw, mask0F);
                    Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                    Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8qs + j * 64 + h * 16);
                    Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8qs + j * 64 + 32 + h * 16);
                    sumi = Sse2.Add(sumi, Sse2.Add(
                        Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(lo, q8Lo), scLo),
                        Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(hi, q8Hi), scHi)));
                }
            }

            acc = Sse.Add(acc, Sse.Multiply(Vector128.Create(d4 * d8), Sse2.ConvertToVector128Single(sumi)));

            qk += Q4_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumSse(acc) - sumMin;
    }

    /// <summary>128-bit twin of <see cref="VecDotQ5_K_Q8_KAvx2"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ5_K_Q8_KSse(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<byte> mask0F = Vector128.Create((byte)0x0F);
        Vector128<byte> sixteen = Vector128.Create((byte)16);
        float sumMin = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            Vector128<float> dd = LoadHalfPairSse2(qk);
            float d5 = dd.ToScalar();
            float dmin = dd.GetElement(1);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qh = qk + 16;
            byte* qs = qk + 48;

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            int minCorr = 0;
            for (int j = 0; j < 8; j++)
                minCorr += mnBuf[j] * (bsums[j * 2] + bsums[j * 2 + 1]);
            sumMin += dmin * d8 * minCorr;

            Vector128<byte> qh0 = Unsafe.ReadUnaligned<Vector128<byte>>(qh);
            Vector128<byte> qh1 = Unsafe.ReadUnaligned<Vector128<byte>>(qh + 16);

            Vector128<int> sumi = Vector128<int>.Zero;
            for (int j = 0; j < 4; j++)
            {
                Vector128<short> scLo = Vector128.Create((short)scBuf[j * 2]);
                Vector128<short> scHi = Vector128.Create((short)scBuf[j * 2 + 1]);
                Vector128<byte> loBit = Vector128.Create((byte)(1 << (j * 2)));
                Vector128<byte> hiBit = Vector128.Create((byte)(1 << (j * 2 + 1)));
                for (int h = 0; h < 2; h++)
                {
                    Vector128<byte> qhv = h == 0 ? qh0 : qh1;
                    Vector128<byte> raw = Unsafe.ReadUnaligned<Vector128<byte>>(qs + j * 32 + h * 16);
                    Vector128<byte> lo = Sse2.Or(Sse2.And(raw, mask0F),
                        Sse2.And(Sse2.CompareEqual(Sse2.And(qhv, loBit), loBit), sixteen));
                    Vector128<byte> hi = Sse2.Or(Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F),
                        Sse2.And(Sse2.CompareEqual(Sse2.And(qhv, hiBit), hiBit), sixteen));
                    Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8qs + j * 64 + h * 16);
                    Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8qs + j * 64 + 32 + h * 16);
                    sumi = Sse2.Add(sumi, Sse2.Add(
                        Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(lo, q8Lo), scLo),
                        Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(hi, q8Hi), scHi)));
                }
            }

            acc = Sse.Add(acc, Sse.Multiply(Vector128.Create(d5 * d8), Sse2.ConvertToVector128Single(sumi)));

            qk += Q5_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumSse(acc) - sumMin;
    }

    /// <summary>128-bit twin of <see cref="VecDotQ6_K_Q8_KAvx2"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ6_K_Q8_KSse(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<byte> mask0F = Vector128.Create((byte)0x0F);
        Vector128<byte> mask03 = Vector128.Create((byte)0x03);
        float sumBias = 0;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* ql = qk;
            byte* qh = qk + 128;
            sbyte* scales = (sbyte*)(qk + 192);
            float d6 = (float)Unsafe.ReadUnaligned<Half>(qk + 208); // one scale per super-block: SSE conversion measured neutral

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            // Bias (the -32 offset) via bsums: 32 · d6 · d8 · Σ scales[s] · bsums[s].
            int biasCorr = 0;
            for (int s = 0; s < 16; s++)
                biasCorr += scales[s] * bsums[s];
            sumBias += 32.0f * d6 * d8 * biasCorr;

            Vector128<int> sumi = Vector128<int>.Zero;
            for (int sub = 0; sub < 16; sub++)
            {
                // Same element ordering as VecDotQ6_K_Q8_KScalar / llama.cpp.
                int half = sub / 8;
                int sh = sub % 8;
                int qlOff = half * 64 + (sh % 4) * 16;
                int qhOff = half * 32 + (sh % 2) * 16;
                int qhShift = (sh / 2) * 2;

                Vector128<byte> qlRaw = Unsafe.ReadUnaligned<Vector128<byte>>(ql + qlOff);
                Vector128<byte> nib = sh >= 4
                    ? Sse2.And(Sse2.ShiftRightLogical(qlRaw.AsUInt16(), 4).AsByte(), mask0F)
                    : Sse2.And(qlRaw, mask0F);
                Vector128<ushort> qhRaw = Unsafe.ReadUnaligned<Vector128<ushort>>(qh + qhOff);
                Vector128<byte> hi2 = Sse2.And(Sse2.ShiftRightLogical(qhRaw, Vector128.CreateScalar((ulong)qhShift).AsUInt16()).AsByte(), mask03);
                Vector128<byte> q6u = Sse2.Or(nib, Sse2.ShiftLeftLogical(hi2.AsUInt16(), 4).AsByte());

                Vector128<sbyte> q8v = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8qs + sub * 16);
                sumi = Sse2.Add(sumi, Sse2.MultiplyAddAdjacent(
                    Ssse3.MultiplyAddAdjacent(q6u, q8v), Vector128.Create((short)scales[sub])));
            }

            acc = Sse.Add(acc, Sse.Multiply(Vector128.Create(d6 * d8), Sse2.ConvertToVector128Single(sumi)));

            qk += Q6_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumSse(acc) - sumBias;
    }

    // Best non-AVX2 single-row K-quant dots: the 128-bit tier on SSSE3 hardware, scalar otherwise.
    // Used by every dispatch site's pre-AVX2 branch (row-major 4-row fallback, tails, R4).

    /// <summary>Best non-AVX2 Q4_K dot (SSSE3, else scalar).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotQ4_K_Q8_KPortable(byte* qk, byte* q8k, int superBlockCount) =>
        Ssse3.IsSupported ? VecDotQ4_K_Q8_KSse(qk, q8k, superBlockCount) : VecDotQ4_K_Q8_KScalar(qk, q8k, superBlockCount);

    /// <summary>Best non-AVX2 Q5_K dot (SSSE3, else scalar).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotQ5_K_Q8_KPortable(byte* qk, byte* q8k, int superBlockCount) =>
        Ssse3.IsSupported ? VecDotQ5_K_Q8_KSse(qk, q8k, superBlockCount) : VecDotQ5_K_Q8_KScalar(qk, q8k, superBlockCount);

    /// <summary>Best non-AVX2 Q6_K dot (SSSE3, else scalar).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotQ6_K_Q8_KPortable(byte* qk, byte* q8k, int superBlockCount) =>
        Ssse3.IsSupported ? VecDotQ6_K_Q8_KSse(qk, q8k, superBlockCount) : VecDotQ6_K_Q8_KScalar(qk, q8k, superBlockCount);
}
