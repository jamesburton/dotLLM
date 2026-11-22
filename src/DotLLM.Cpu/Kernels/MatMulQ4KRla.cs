using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Q4_K x Q8_K "row-lane arithmetic" (RLA, #1927): the per-row accumulation order of the row-lane GEMM
/// (<see cref="MatMul.GemmQ4KRowLane8"/>), reproduced by the single-row, 4-row and 4-token kernels so that EVERY Q4_K path (decode
/// GEMV, fused decode, short and long GEMMs, any batch composition) gives a row the same bits.
/// <para>
/// Per (row, super-block): the integer dot <c>S = Σ_sub sc_sub · Σ w·q</c> and the min term <c>mc = Σ_sub mn_sub · (bsum pair)</c>
/// are formed EXACTLY in int32 (any instruction sequence gives the same integer), then
/// <c>acc = fma(d4 · d8, (float)S, acc)</c> and <c>sm = sm + (dmin · d8) · (float)mc</c> in super-block order; the result is
/// <c>acc - sm</c>. The R4 kernels instead convert eight K-direction int32 lanes and sum them in float at the end. RLA is closer to
/// float64 (see <c>docs/PRECISION.md</c> E19) and, because nothing in it depends on the batch, batch-invariant by construction.
/// </para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// True when Q4_K kernels use the row-lane arithmetic: the switch is on (the default; <c>DOTLLM_CPU_Q4K_ROWLANE=0</c> restores the
    /// R4 arithmetic everywhere) and the row-lane ISA is present (AVX2 + SSE4.1 + FMA, not force-disabled). Hosts without it keep the
    /// R4 kernels at every batch size, so they are batch-invariant too. Read inside the kernels on every worker thread, so it is
    /// process-wide by design: a thread-scoped switch could not reach pool workers.
    /// </summary>
    internal static bool Q4KRlaActive => Q4KRowLaneEnabled && Q4KRowLaneIsa && !ForceDisableFastKQuant;

    /// <summary>One Q4_K row against one Q8_K row in the row-lane arithmetic. <paramref name="wStride"/>: bytes between super-blocks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Q4KRlaDot1(byte* w, byte* q8k, int superBlockCount, int wStride)
        => CpuFeatures.Avx2 && CpuFeatures.Sse41 ? Q4KRlaDot1Avx2(w, q8k, superBlockCount, wStride)   // always true when Q4KRlaActive
                                                 : Q4KRlaDot1Scalar(w, q8k, superBlockCount, wStride);

    /// <summary>Portable RLA reference: plain integer loops, the same float recurrence. The oracle of the vector forms.</summary>
    [SkipLocalsInit]
    internal static float Q4KRlaDot1Scalar(byte* w, byte* q8k, int superBlockCount, int wStride)
    {
        byte* sc = stackalloc byte[8];
        byte* mn = stackalloc byte[8];
        float acc = 0f, sm = 0f;
        for (int sb = 0; sb < superBlockCount; sb++, w += wStride, q8k += Q8_K_BlockBytes)
        {
            float d4 = HalfToSingle(w), dm = HalfToSingle(w + 2);
            Dequantize.UnpackQ4Q5Scales(w + 4, sc, mn);
            byte* qs = w + 16;
            sbyte* q = (sbyte*)(q8k + 4);
            short* bs = (short*)(q8k + 260);
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            int s = 0, mc = 0;
            for (int j = 0; j < 8; j++)
            {
                int dot = 0;
                byte* src = qs + (j >> 1) * 32;
                sbyte* qq = q + j * 32;
                for (int i = 0; i < 32; i++)
                    dot += ((j & 1) == 0 ? src[i] & 0x0F : src[i] >> 4) * qq[i];
                s += sc[j] * dot;
                mc += mn[j] * (bs[2 * j] + bs[2 * j + 1]);
            }
            acc = MathF.FusedMultiplyAdd(d4 * d8, (float)s, acc);
            sm += (dm * d8) * (float)mc;
        }
        return acc - sm;
    }

    [SkipLocalsInit]
    private static float Q4KRlaDot1Avx2(byte* w, byte* q8k, int superBlockCount, int wStride)
    {
        float acc = 0f, sm = 0f;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];
        for (int sb = 0; sb < superBlockCount; sb++, w += wStride, q8k += Q8_K_BlockBytes)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            float d4 = HalfToSingle(w), dm = HalfToSingle(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);
            int mc = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(q8k + 260), mnDup));

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
            acc = MathF.FusedMultiplyAdd(d4 * d8, (float)HorizontalSumInt(sumi), acc);
            sm += (dm * d8) * (float)mc;
        }
        return acc - sm;
    }

    /// <summary>RLA form of <see cref="Q4KRowTok4"/>: one row's super-block decoded once for four tokens, each token's RLA recurrence.</summary>
    [SkipLocalsInit]
    private static void Q4KRlaRowTok4(byte* w, byte* q8k, int q8RowBytes, int superBlockCount, int wStride, float* results, int ldc)
    {
        byte* qa0 = q8k, qa1 = q8k + q8RowBytes, qa2 = q8k + 2L * q8RowBytes, qa3 = q8k + 3L * q8RowBytes;
        float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, m0 = 0f, m1 = 0f, m2 = 0f, m3 = 0f;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        short* sc = stackalloc short[8];
        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d4 = HalfToSingle(w), dm = HalfToSingle(w + 2);
            UnpackQ4KScalesFast(w + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
            Sse2.Store(sc, sc16);
            float d80 = Unsafe.ReadUnaligned<float>(qa0), d81 = Unsafe.ReadUnaligned<float>(qa1);
            float d82 = Unsafe.ReadUnaligned<float>(qa2), d83 = Unsafe.ReadUnaligned<float>(qa3);
            int mc0 = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa0 + 260), mnDup));
            int mc1 = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa1 + 260), mnDup));
            int mc2 = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa2 + 260), mnDup));
            int mc3 = HorizontalSumInt(Avx2.MultiplyAddAdjacent(Unsafe.ReadUnaligned<Vector256<short>>(qa3 + 260), mnDup));

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
            a0 = MathF.FusedMultiplyAdd(d4 * d80, (float)HorizontalSumInt(s0), a0);
            a1 = MathF.FusedMultiplyAdd(d4 * d81, (float)HorizontalSumInt(s1), a1);
            a2 = MathF.FusedMultiplyAdd(d4 * d82, (float)HorizontalSumInt(s2), a2);
            a3 = MathF.FusedMultiplyAdd(d4 * d83, (float)HorizontalSumInt(s3), a3);
            m0 += (dm * d80) * (float)mc0;
            m1 += (dm * d81) * (float)mc1;
            m2 += (dm * d82) * (float)mc2;
            m3 += (dm * d83) * (float)mc3;

            w += wStride;
            qa0 += Q8_K_BlockBytes; qa1 += Q8_K_BlockBytes; qa2 += Q8_K_BlockBytes; qa3 += Q8_K_BlockBytes;
        }
        results[0] = a0 - m0;
        results[ldc] = a1 - m1;
        results[2L * ldc] = a2 - m2;
        results[3L * ldc] = a3 - m3;
    }
}
