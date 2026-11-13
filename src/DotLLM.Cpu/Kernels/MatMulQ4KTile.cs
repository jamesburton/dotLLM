using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Eight-token K-quant prefill microkernel (#1675), bit-identical to <see cref="Rows4Tokens4"/> and so to the per-token
/// reference.
/// </summary>
/// <remarks>
/// <para>
/// Profiling the R4 Q4_K GEMM on the TwIL-8B shapes (single thread, about 84 GMAC/s) showed it is not bound by the
/// multiplies: replacing every <c>vpmaddubsw</c>/<c>vpdpwssd</c> with an add of the same operands moved the time by
/// under 15 %, and sharing the four tokens' activation loads across four rows gained nothing. It is bound by instruction
/// count. Per (row, super-block) <see cref="Q4KRowTok4"/> spends about 100 instructions on weight-side bookkeeping
/// (two software Half conversions, the 12-byte scale unpack, the min-correction tree, the FMA tail) plus about 12
/// instructions of JIT address arithmetic per 64-element chunk, against 16 multiply instructions per chunk.
/// </para>
/// <para>
/// This kernel decodes a row's super-block once for eight tokens instead of four (halving the per-token share of the
/// weight-side work) and spells out the four 64-element chunks with constant offsets, so the inner body is just the
/// multiplies and their memory operands.
/// </para>
/// <para>
/// Bit-identity: each (row, token) integer accumulator receives exactly the same products in the same lanes as
/// <see cref="Q4KRowTok4"/> (integer adds are exact, so their order is free); the per-super-block float update is the same
/// <c>fma(d4 * d8, float(sumi), acc)</c> in the same super-block order; the min correction is the same
/// <c>(dm * d8) * float(mc)</c> running sum; and the final horizontal sum is the same helper.
/// </para>
/// </remarks>
public static unsafe partial class MatMul
{
    private static readonly bool s_kqTileEnvOff = Environment.GetEnvironmentVariable("DOTLLM_CPU_KQ_TILE") == "0";

    /// <summary>
    /// Runtime switch for the eight-token K-quant kernels (#1675). <c>DOTLLM_CPU_KQ_TILE=0</c> turns them off.
    /// Settable for in-process A/B. Results are bit-identical either way.
    /// </summary>
    internal static volatile bool KQuantTile = !s_kqTileEnvOff;

    /// <summary>True when the eight-token kernel exists for this quant and the ISA can run it.</summary>
    internal static bool KQuantTileAvailable(bool q6) => !q6 && FastKQuantIsa;

    /// <summary>
    /// Q4_K 4 rows x 8 tokens. Same contract as <see cref="Rows4Tokens4"/> with eight tokens: rows are
    /// <c>superBlockCount * Q4_K_BlockBytes</c> apart from <paramref name="groupBase"/>, writes <c>results[t * ldc + r]</c>
    /// for <c>t &lt; 8</c>, <c>r &lt; 4</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4KRows4Tokens8(byte* groupBase, int superBlockCount, byte* q8k, int q8RowBytes,
        float* results, int ldc)
    {
        if (!FastKQuantIsa)
        {
            Rows4Tokens4(groupBase, superBlockCount, q8k, q8RowBytes, results, ldc);
            Rows4Tokens4(groupBase, superBlockCount, q8k + 4L * q8RowBytes, q8RowBytes, results + 4L * ldc, ldc);
            return;
        }

        nint rowStride = (nint)superBlockCount * Q4_K_BlockBytes;
        for (int r = 0; r < 4; r++)
            Q4KRowTok8(groupBase + r * rowStride, q8k, q8RowBytes, superBlockCount, results + r, ldc);
    }

    /// <summary>One sub-block pair against one token: <c>sumi += scLo*maddubs(lo,q8Lo) + scHi*maddubs(hi,q8Hi)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Q4KAcc8(Vector256<int> sumi, Vector256<byte> lo, Vector256<byte> hi, byte* q8,
        Vector256<short> scLo, Vector256<short> scHi)
    {
        Vector256<short> pLo = Avx2.MultiplyAddAdjacent(lo, Unsafe.ReadUnaligned<Vector256<sbyte>>(q8));
        Vector256<short> pHi = Avx2.MultiplyAddAdjacent(hi, Unsafe.ReadUnaligned<Vector256<sbyte>>(q8 + 32));
        if (CpuFeatures.AvxVnni)
        {
            sumi = AvxVnni.MultiplyWideningAndAdd(sumi, pLo, scLo);
            return AvxVnni.MultiplyWideningAndAdd(sumi, pHi, scHi);
        }
        return Avx2.Add(sumi, Avx2.Add(Avx2.MultiplyAddAdjacent(pLo, scLo), Avx2.MultiplyAddAdjacent(pHi, scHi)));
    }

    /// <summary>One 64-element chunk (<paramref name="j"/> = 0..3) of one row against eight tokens.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q4KChunk8(byte* qs, int j, byte* qa, nint q8RowBytes, Vector256<byte> mask0F,
        Vector256<short> scLo, Vector256<short> scHi,
        ref Vector256<int> s0, ref Vector256<int> s1, ref Vector256<int> s2, ref Vector256<int> s3,
        ref Vector256<int> s4, ref Vector256<int> s5, ref Vector256<int> s6, ref Vector256<int> s7)
    {
        Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + j * 32);
        Vector256<byte> lo = Avx2.And(raw, mask0F);
        Vector256<byte> hi = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
        byte* q = qa + 4 + j * 64;
        s0 = Q4KAcc8(s0, lo, hi, q, scLo, scHi);
        s1 = Q4KAcc8(s1, lo, hi, q + q8RowBytes, scLo, scHi);
        s2 = Q4KAcc8(s2, lo, hi, q + 2 * q8RowBytes, scLo, scHi);
        s3 = Q4KAcc8(s3, lo, hi, q + 3 * q8RowBytes, scLo, scHi);
        s4 = Q4KAcc8(s4, lo, hi, q + 4 * q8RowBytes, scLo, scHi);
        s5 = Q4KAcc8(s5, lo, hi, q + 5 * q8RowBytes, scLo, scHi);
        s6 = Q4KAcc8(s6, lo, hi, q + 6 * q8RowBytes, scLo, scHi);
        s7 = Q4KAcc8(s7, lo, hi, q + 7 * q8RowBytes, scLo, scHi);
    }

    [SkipLocalsInit]
    private static void Q4KRowTok8(byte* w, byte* q8k, int q8RowBytes, int superBlockCount, float* results, int ldc)
    {
        nint rb = q8RowBytes;
        byte* qa = q8k;
        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        Vector256<float> acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
        Vector128<float> smA = Vector128<float>.Zero, smB = smA;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d4 = HalfToSingle(w);
            float dm = HalfToSingle(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);

            float d80 = Unsafe.ReadUnaligned<float>(qa), d81 = Unsafe.ReadUnaligned<float>(qa + rb);
            float d82 = Unsafe.ReadUnaligned<float>(qa + 2 * rb), d83 = Unsafe.ReadUnaligned<float>(qa + 3 * rb);
            float d84 = Unsafe.ReadUnaligned<float>(qa + 4 * rb), d85 = Unsafe.ReadUnaligned<float>(qa + 5 * rb);
            float d86 = Unsafe.ReadUnaligned<float>(qa + 6 * rb), d87 = Unsafe.ReadUnaligned<float>(qa + 7 * rb);

            // Min corrections, four tokens per hadd tree: exact int sums, then (dm * d8) * float(mc) as in Q4KRowTok4.
            Vector256<int> h = Avx2.HorizontalAdd(
                Avx2.HorizontalAdd(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 260), mnDup),
                                   Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + rb + 260), mnDup)),
                Avx2.HorizontalAdd(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 2 * rb + 260), mnDup),
                                   Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 3 * rb + 260), mnDup)));
            Vector256<int> h2 = Avx2.HorizontalAdd(
                Avx2.HorizontalAdd(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 4 * rb + 260), mnDup),
                                   Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 5 * rb + 260), mnDup)),
                Avx2.HorizontalAdd(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 6 * rb + 260), mnDup),
                                   Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa + 7 * rb + 260), mnDup)));
            Vector128<float> dmV = Vector128.Create(dm);
            smA = Sse.Add(smA, Sse.Multiply(Sse.Multiply(dmV, Vector128.Create(d80, d81, d82, d83)),
                Sse2.ConvertToVector128Single(Sse2.Add(h.GetLower(), h.GetUpper()))));
            smB = Sse.Add(smB, Sse.Multiply(Sse.Multiply(dmV, Vector128.Create(d84, d85, d86, d87)),
                Sse2.ConvertToVector128Single(Sse2.Add(h2.GetLower(), h2.GetUpper()))));

            byte* qs = w + 16;
            Vector256<int> s0 = Vector256<int>.Zero, s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
            Q4KChunk8(qs, 0, qa, rb, mask0F, Avx2.BroadcastScalarToVector256(sc), Avx2.BroadcastScalarToVector256(sc + 1),
                ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7);
            Q4KChunk8(qs, 1, qa, rb, mask0F, Avx2.BroadcastScalarToVector256(sc + 2), Avx2.BroadcastScalarToVector256(sc + 3),
                ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7);
            Q4KChunk8(qs, 2, qa, rb, mask0F, Avx2.BroadcastScalarToVector256(sc + 4), Avx2.BroadcastScalarToVector256(sc + 5),
                ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7);
            Q4KChunk8(qs, 3, qa, rb, mask0F, Avx2.BroadcastScalarToVector256(sc + 6), Avx2.BroadcastScalarToVector256(sc + 7),
                ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7);

            acc0 = Q4KFma(d4 * d80, s0, acc0); acc1 = Q4KFma(d4 * d81, s1, acc1);
            acc2 = Q4KFma(d4 * d82, s2, acc2); acc3 = Q4KFma(d4 * d83, s3, acc3);
            acc4 = Q4KFma(d4 * d84, s4, acc4); acc5 = Q4KFma(d4 * d85, s5, acc5);
            acc6 = Q4KFma(d4 * d86, s6, acc6); acc7 = Q4KFma(d4 * d87, s7, acc7);

            w += Q4_K_BlockBytes;
            qa += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0) - smA.GetElement(0);
        results[ldc] = HorizontalSumAvx2Float(acc1) - smA.GetElement(1);
        results[2L * ldc] = HorizontalSumAvx2Float(acc2) - smA.GetElement(2);
        results[3L * ldc] = HorizontalSumAvx2Float(acc3) - smA.GetElement(3);
        results[4L * ldc] = HorizontalSumAvx2Float(acc4) - smB.GetElement(0);
        results[5L * ldc] = HorizontalSumAvx2Float(acc5) - smB.GetElement(1);
        results[6L * ldc] = HorizontalSumAvx2Float(acc6) - smB.GetElement(2);
        results[7L * ldc] = HorizontalSumAvx2Float(acc7) - smB.GetElement(3);
    }
}
