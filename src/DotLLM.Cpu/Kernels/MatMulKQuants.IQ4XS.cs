using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// <b>IQ4_XS</b> × Q8_K packed dot (issue #605). Before this file every IQ4_XS matmul fell through
/// the dequantize-per-row fallback (decode 1.1–3.7 tok/s on any CPU, 5–15x below Q4_K).
/// </summary>
/// <remarks>
/// <para>IQ4_XS layout per 256-value super-block: d(Half)@0, scales_h(u16)@2, scales_l[4]@4,
/// qs[128]@8. Each 32-value sub-block has a 6-bit scale (<c>ls − 32</c>) and 16 bytes of nibbles:
/// the low nibbles are values 0..15, the high nibbles values 16..31, each mapped through the
/// signed non-linear <c>kvalues_iq4nl</c> table. There is no per-block minimum, so unlike Q3_K/Q6_K
/// no <c>bsums</c> correction is needed: <c>sum = d·d8·Σ_sub (ls−32)·Σ_j kv[nib_j]·q8_j</c>.</para>
/// <para><b>Sign trick.</b> Both operands are signed, so PMADDUBSW takes <c>|w|</c> (≤127) and
/// <c>sign(q8, w)</c>. The table has no zero, and production Q8_K is clamped to ±127 (scale =
/// maxAbs/127), so the negation never wraps. Pair sums ≤ 2·127·127 = 32258 &lt; 32767: no
/// saturation. PMADDWD by |ls−32| ≤ 32 keeps the int32 accumulator far inside range.</para>
/// </remarks>
public static unsafe partial class MatMul
{
    private const int IQ4_XS_BlockBytes = QuantFormat.IQ4_XSBlockBytes;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int IQ4XsSubScale(byte* qk, int ib)
    {
        ushort scalesH = Unsafe.ReadUnaligned<ushort>(qk + 2);
        int low = (qk[4 + ib / 2] >> (4 * (ib % 2))) & 0x0F;
        int high = (scalesH >> (2 * ib)) & 0x03;
        return (low | (high << 4)) - 32;
    }

    /// <summary>
    /// Scalar IQ4_XS × Q8_K dot — a transcription of llama.cpp's
    /// <c>ggml_vec_dot_iq4_xs_q8_K_generic</c>, written from that function rather than from
    /// <see cref="Dequantize.DequantizeIQ4_XS"/> so the two stay independent oracles.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotIQ4_XS_Q8_KScalar(byte* qk, byte* q8k, int superBlockCount)
    {
        ReadOnlySpan<sbyte> values = Dequantize.KValuesIq4Nl;
        float sumf = 0;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(qk) * Unsafe.ReadUnaligned<float>(q8k);
            byte* qs = qk + 8;
            sbyte* q8 = (sbyte*)(q8k + 4);

            int sumi = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                int ls = IQ4XsSubScale(qk, ib);
                int s = 0;
                for (int j = 0; j < 16; j++)
                {
                    s += q8[j] * values[qs[j] & 0x0F];
                    s += q8[j + 16] * values[qs[j] >> 4];
                }
                sumi += ls * s;
                qs += 16;
                q8 += 32;
            }

            sumf += d * sumi;
            qk += IQ4_XS_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return sumf;
    }

    /// <summary>
    /// AVX2 IQ4_XS × Q8_K dot. Per 32-value sub-block: PSHUFB the 16 nibble bytes through the
    /// 16-entry table (low nibbles → lane 0, high nibbles → lane 1), then the sign-trick
    /// PMADDUBSW × PMADDWD against the sub-block scale.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotIQ4_XS_Q8_KAvx2(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector128<sbyte> table = Vector128.Create(Dequantize.KValuesIq4Nl);
        Vector256<sbyte> table2 = Vector256.Create(table, table);
        Vector128<byte> mask0F = Vector128.Create((byte)0x0F);

        Vector256<float> acc = Vector256<float>.Zero;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float dd = (float)Unsafe.ReadUnaligned<Half>(qk) * Unsafe.ReadUnaligned<float>(q8k);
            byte* qs = qk + 8;
            sbyte* q8 = (sbyte*)(q8k + 4);

            Vector256<int> sumi = Vector256<int>.Zero;
            for (int ib = 0; ib < 8; ib++)
            {
                Vector128<byte> raw = Unsafe.ReadUnaligned<Vector128<byte>>(qs + ib * 16);
                Vector128<byte> lo = Sse2.And(raw, mask0F);
                Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                Vector256<sbyte> w = Avx2.Shuffle(table2, Vector256.Create(lo, hi).AsSByte());
                Vector256<sbyte> q8v = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8 + ib * 32);

                Vector256<short> prod = Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(q8v, w));
                sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(prod, Vector256.Create((short)IQ4XsSubScale(qk, ib))));
            }

            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(Vector256.Create(dd), Avx.ConvertToVector256Single(sumi), acc)
                : Avx.Add(acc, Avx.Multiply(Vector256.Create(dd), Avx.ConvertToVector256Single(sumi)));

            qk += IQ4_XS_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>128-bit (SSSE3) twin of <see cref="VecDotIQ4_XS_Q8_KAvx2"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotIQ4_XS_Q8_KSse(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector128<sbyte> table = Vector128.Create(Dequantize.KValuesIq4Nl);
        Vector128<byte> mask0F = Vector128.Create((byte)0x0F);

        Vector128<float> acc = Vector128<float>.Zero;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float dd = (float)Unsafe.ReadUnaligned<Half>(qk) * Unsafe.ReadUnaligned<float>(q8k);
            byte* qs = qk + 8;
            sbyte* q8 = (sbyte*)(q8k + 4);

            Vector128<int> sumi = Vector128<int>.Zero;
            for (int ib = 0; ib < 8; ib++)
            {
                Vector128<byte> raw = Unsafe.ReadUnaligned<Vector128<byte>>(qs + ib * 16);
                Vector128<byte> lo = Sse2.And(raw, mask0F);
                Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                Vector128<sbyte> wLo = Ssse3.Shuffle(table.AsByte(), lo).AsSByte();
                Vector128<sbyte> wHi = Ssse3.Shuffle(table.AsByte(), hi).AsSByte();
                Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8 + ib * 32);
                Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8 + ib * 32 + 16);

                // Scale each 16-value half separately: two pair-sum vectors (each up to 32258 per
                // lane) must not be added in int16.
                Vector128<short> scale = Vector128.Create((short)IQ4XsSubScale(qk, ib));
                sumi = Sse2.Add(sumi, Sse2.Add(
                    Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(Ssse3.Abs(wLo), Ssse3.Sign(q8Lo, wLo)), scale),
                    Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(Ssse3.Abs(wHi), Ssse3.Sign(q8Hi, wHi)), scale)));
            }

            acc = Sse.Add(acc, Sse.Multiply(Vector128.Create(dd), Sse2.ConvertToVector128Single(sumi)));

            qk += IQ4_XS_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumSse(acc);
    }

    /// <summary>Best non-AVX2 IQ4_XS dot (SSSE3, else scalar).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotIQ4_XS_Q8_KPortable(byte* qk, byte* q8k, int superBlockCount) =>
        Ssse3.IsSupported ? VecDotIQ4_XS_Q8_KSse(qk, q8k, superBlockCount) : VecDotIQ4_XS_Q8_KScalar(qk, q8k, superBlockCount);

    /// <summary>Row-major IQ4_XS × Q8_K: one dot per output row.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsIQ4_XS(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * IQ4_XS_BlockBytes;
        if (Avx2.IsSupported)
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotIQ4_XS_Q8_KAvx2(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotIQ4_XS_Q8_KPortable(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
    }

    /// <summary>IQ4_XS GEMV: quantizes the input to Q8_K once, then one packed dot per row.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvIQ4_XS(byte* weights, float* x, float* result, int m, int k)
        => GemvLowBitKQuant(weights, x, result, m, k, &ComputeRowsIQ4_XS);

    /// <summary>IQ4_XS GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvIQ4_XS(byte* weights, float* x, float* result, int m, int k,
                                  ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvIQ4_XS(weights, x, result, m, k);
            return;
        }

        GemvKQuantParallel(weights, x, result, m, k, IQ4_XS_BlockBytes, &ComputeRowsIQ4_XS, pool);
    }

    /// <summary>IQ4_XS GEMM: C[N,M] = B[N,K] × A[M,K]^T where A is IQ4_XS weights.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmIQ4_XS(byte* weights, float* b, float* c, int m, int k, int n,
                                  byte* preQuantizedInput = null)
        => GemmKQuant(weights, b, c, m, k, n, IQ4_XS_BlockBytes, &ComputeRowsIQ4_XS, preQuantizedInput);

    /// <summary>IQ4_XS GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmIQ4_XS(byte* weights, float* b, float* c, int m, int k, int n,
                                  ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmIQ4_XS(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        GemmKQuantParallel(weights, b, c, m, k, n, IQ4_XS_BlockBytes,
            &ComputeRowsIQ4_XS, pool, preQuantizedInput);
    }
}
