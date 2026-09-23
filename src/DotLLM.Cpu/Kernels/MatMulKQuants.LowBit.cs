using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// The low-bit half of the K-quant × Q8_K family: <b>Q2_K</b> and <b>Q3_K</b> (issue #497).
/// Before this file every Q2_K/Q3_K matmul fell through <see cref="GemmDequantRows"/>, expanding
/// the whole weight matrix to F32 — issue #489 packed the 32-element/Q8_1 legacy quants and left
/// these two open because they are 256-element super-blocks scored against Q8_K.
///
/// <para><b>Shape.</b> Both formats are 16 sub-blocks of 16 elements, so Q8_K's
/// <c>bsums[16]</c> indexes sub-blocks 1:1 — this is the <see cref="VecDotQ6_K_Q8_KAvx2"/> shape,
/// not the Q4_K one (no <c>bsums[2j]+bsums[2j+1]</c> pairing). Each super-block reduces to an
/// exact int32 "scale-in-madd" sum (PMADDUBSW with the <i>unsigned</i> weight code first, then
/// PMADDWD by the broadcast sub-block scale), converted to float once and multiplied by
/// <c>d·d8</c>; the affine term (Q2_K's <c>dmin</c>, Q3_K's <c>-4</c>) comes from <c>bsums</c>.</para>
///
/// <para><b>Element ordering is transposed in both formats</b> — see
/// <see cref="Dequantize.DequantizeQ2_K"/> and <see cref="Dequantize.DequantizeQ3_KScalar"/>.
/// Each 128-element half consumes 32 <c>qs</c> bytes and each byte carries four elements 32 apart.
/// That is what makes these kernels simple: one 32-byte load per (half, shift) pair covers two
/// whole sub-blocks, low 16 lanes for the even one and high 16 for the odd one, exactly the
/// <c>scaleVec</c> trick Q6_K uses.</para>
///
/// <para><b>No saturation.</b> PMADDUBSW pair sums are at most <c>2·3·127 = 762</c> (Q2_K) and
/// <c>2·7·128 = 1792</c> (Q3_K); PMADDWD by a scale of at most 15 (Q2_K) or |−32| (Q3_K) leaves
/// the int32 super-block accumulator three orders of magnitude inside range. Any Q8_K byte,
/// including <c>−128</c>, is handled.</para>
/// </summary>
public static unsafe partial class MatMul
{
    private const int Q2_K_BlockBytes = QuantFormat.Q2_KBlockBytes;
    private const int Q3_K_BlockBytes = QuantFormat.Q3_KBlockBytes;

    /// <summary>
    /// Reports whether <paramref name="qt"/> has a packed <c>vec_dot</c> against <b>Q8_K</b>
    /// activations — i.e. whether a Q8_K pre-quantized input buffer is the right thing to hand it.
    /// </summary>
    /// <remarks>
    /// Since #497 this is the whole K-quant family: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. Callers that
    /// used to spell the Q4_K/Q5_K/Q6_K triple out by hand should use this instead — the
    /// hand-written lists are what left Q2_K and Q3_K on the dequantize-per-row fallback after
    /// their kernels existed.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool UsesQ8KDot(QuantizationType qt) =>
        qt is QuantizationType.Q2_K or QuantizationType.Q3_K or QuantizationType.Q4_K
            or QuantizationType.Q5_K or QuantizationType.Q6_K;

    // ──────────────────── Q2_K × Q8_K scalar ────────────────────

    /// <summary>
    /// Scalar Q2_K × Q8_K dot product — a transcription of llama.cpp's
    /// <c>ggml_vec_dot_q2_K_q8_K_generic</c>, deliberately written from that function rather than
    /// derived from <see cref="Dequantize.DequantizeQ2_K"/> so the two stay independent oracles.
    /// Q2_K layout: scales[16]@0, qs[64]@16, d(Half)@80, dmin(Half)@82.
    /// Q8_K layout: d(float)@0, qs[256]@4, bsums[16]@260.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ2_K_Q8_KScalar(byte* qk, byte* q8k, int superBlockCount)
    {
        float sumf = 0;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* sc = qk;
            byte* q2 = qk + 16;
            float d2 = (float)Unsafe.ReadUnaligned<Half>(qk + 80);
            float dmin2 = (float)Unsafe.ReadUnaligned<Half>(qk + 82);

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8 = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            int summs = 0;
            for (int j = 0; j < 16; j++)
                summs += bsums[j] * (sc[j] >> 4);

            int isum = 0;
            int isIdx = 0;
            int qOff = 0;
            int q8Off = 0;
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int scale = sc[isIdx++] & 0xF;
                    int isuml = 0;
                    for (int l = 0; l < 16; l++)
                        isuml += q8[q8Off + l] * ((q2[qOff + l] >> shift) & 3);
                    isum += scale * isuml;

                    scale = sc[isIdx++] & 0xF;
                    isuml = 0;
                    for (int l = 16; l < 32; l++)
                        isuml += q8[q8Off + l] * ((q2[qOff + l] >> shift) & 3);
                    isum += scale * isuml;

                    shift += 2;
                    q8Off += 32;
                }
                qOff += 32;
            }

            sumf += d2 * d8 * isum - dmin2 * d8 * summs;

            qk += Q2_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── Q3_K × Q8_K scalar ────────────────────

    /// <summary>
    /// Scalar Q3_K × Q8_K dot product — a transcription of llama.cpp's
    /// <c>ggml_vec_dot_q3_K_q8_K_generic</c>.
    /// Q3_K layout: hmask[32]@0, qs[64]@32, scales[12]@96, d(Half)@108.
    /// The 3-bit code is <c>(hbit &lt;&lt; 2) | qbits</c> biased by <c>−4</c>; the high bit for
    /// half <c>h</c>, shift-step <c>j</c> is bit <c>4h + j</c> of <c>hmask[l]</c> — the counter
    /// does <b>not</b> reset between halves.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ3_K_Q8_KScalar(byte* qk, byte* q8k, int superBlockCount)
    {
        float sumf = 0;
        byte* scales = stackalloc byte[16];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* hm = qk;
            byte* q3 = qk + 32;
            Dequantize.UnpackQ3KScales(qk + 96, scales);
            float d3 = (float)Unsafe.ReadUnaligned<Half>(qk + 108);

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8 = (sbyte*)(q8k + 4);

            int isum = 0;
            int isIdx = 0;
            int qOff = 0;
            int q8Off = 0;
            int m = 0;
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int scale = scales[isIdx++] - 32;
                    int isuml = 0;
                    for (int l = 0; l < 16; l++)
                        isuml += q8[q8Off + l]
                            * (((q3[qOff + l] >> shift) & 3) - (((hm[l] >> m) & 1) != 0 ? 0 : 4));
                    isum += scale * isuml;

                    scale = scales[isIdx++] - 32;
                    isuml = 0;
                    for (int l = 16; l < 32; l++)
                        isuml += q8[q8Off + l]
                            * (((q3[qOff + l] >> shift) & 3) - (((hm[l] >> m) & 1) != 0 ? 0 : 4));
                    isum += scale * isuml;

                    shift += 2;
                    m++;
                    q8Off += 32;
                }
                qOff += 32;
            }

            sumf += d3 * d8 * isum;

            qk += Q3_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── Q2_K × Q8_K AVX2 ────────────────────

    /// <summary>
    /// AVX2 Q2_K × Q8_K dot product. Integer accumulation with scale-in-madd; the <c>dmin</c>
    /// term is folded through the Q8_K <c>bsums</c> once per super-block. 8 iterations of
    /// 32 values (2 halves × 4 shifts), each covering two whole sub-blocks.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ2_K_Q8_KAvx2(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask03 = Vector256.Create((byte)0x03);
        float sumMin = 0;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* sc = qk;
            byte* q2 = qk + 16;
            float d2 = (float)Unsafe.ReadUnaligned<Half>(qk + 80);
            float dmin2 = (float)Unsafe.ReadUnaligned<Half>(qk + 82);

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            int minCorr = 0;
            for (int j = 0; j < 16; j++)
                minCorr += (sc[j] >> 4) * bsums[j];
            sumMin += dmin2 * d8 * minCorr;

            Vector256<int> sumi = Vector256<int>.Zero;
            for (int half = 0; half < 2; half++)
            {
                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(q2 + half * 32);
                for (int j = 0; j < 4; j++)
                {
                    // A 16-bit right shift is safe here: the 0x03 mask keeps only bits 0-1 of
                    // each byte, which after >>2j are that byte's own bits 2j..2j+1.
                    Vector256<byte> bits = j == 0
                        ? Avx2.And(raw, mask03)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(),
                            Vector128.CreateScalar((ulong)(j * 2)).AsUInt16()).AsByte(), mask03);

                    int sub = half * 8 + j * 2;
                    Vector256<sbyte> q8Vals =
                        Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + (half * 128 + j * 32));

                    // Lanes 0..15 are sub-block `sub`, lanes 16..31 sub-block `sub+1`.
                    Vector256<short> scaleVec = Vector256.Create(
                        Vector128.Create((short)(sc[sub] & 0xF)),
                        Vector128.Create((short)(sc[sub + 1] & 0xF)));

                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(
                        Avx2.MultiplyAddAdjacent(bits, q8Vals), scaleVec));
                }
            }

            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(Vector256.Create(d2 * d8), Avx.ConvertToVector256Single(sumi), acc)
                : Avx.Add(acc, Avx.Multiply(Vector256.Create(d2 * d8), Avx.ConvertToVector256Single(sumi)));

            qk += Q2_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc) - sumMin;
    }

    // ──────────────────── Q3_K × Q8_K AVX2 ────────────────────

    /// <summary>
    /// AVX2 Q3_K × Q8_K dot product. The 3-bit code is assembled unsigned
    /// (<c>(hbit &lt;&lt; 2) | qbits ∈ [0,7]</c>) so it can be the first PMADDUBSW operand; the
    /// <c>−4</c> bias is folded through <c>bsums</c> as
    /// <c>4·d·d8·Σ (scale_sub − 32)·bsums[sub]</c>, exactly as Q6_K folds its <c>−32</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ3_K_Q8_KAvx2(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask03 = Vector256.Create((byte)0x03);
        Vector256<byte> mask01 = Vector256.Create((byte)0x01);
        float sumBias = 0;
        byte* scales = stackalloc byte[16];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* hm = qk;
            byte* q3 = qk + 32;
            Dequantize.UnpackQ3KScales(qk + 96, scales);
            float d3 = (float)Unsafe.ReadUnaligned<Half>(qk + 108);

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            int biasCorr = 0;
            for (int s = 0; s < 16; s++)
                biasCorr += (scales[s] - 32) * bsums[s];
            sumBias += 4.0f * d3 * d8 * biasCorr;

            Vector256<byte> hmVec = Unsafe.ReadUnaligned<Vector256<byte>>(hm);

            Vector256<int> sumi = Vector256<int>.Zero;
            int m = 0;
            for (int half = 0; half < 2; half++)
            {
                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(q3 + half * 32);
                for (int j = 0; j < 4; j++, m++)
                {
                    Vector256<byte> bits = j == 0
                        ? Avx2.And(raw, mask03)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(),
                            Vector128.CreateScalar((ulong)(j * 2)).AsUInt16()).AsByte(), mask03);

                    // hbit for this (half, j) is bit `m` of every hmask byte; m runs 0..7
                    // across BOTH halves. Masking to 0x01 after a 16-bit shift keeps each
                    // byte's own bit m.
                    Vector256<byte> hbit = m == 0
                        ? Avx2.And(hmVec, mask01)
                        : Avx2.And(Avx2.ShiftRightLogical(hmVec.AsUInt16(),
                            Vector128.CreateScalar((ulong)m).AsUInt16()).AsByte(), mask01);

                    // 0/1 → 0/4 without crossing byte boundaries (max value 1 << 2 == 4).
                    Vector256<byte> code = Avx2.Or(bits,
                        Avx2.ShiftLeftLogical(hbit.AsUInt16(), 2).AsByte());

                    int sub = half * 8 + j * 2;
                    Vector256<sbyte> q8Vals =
                        Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + (half * 128 + j * 32));

                    Vector256<short> scaleVec = Vector256.Create(
                        Vector128.Create((short)(scales[sub] - 32)),
                        Vector128.Create((short)(scales[sub + 1] - 32)));

                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(
                        Avx2.MultiplyAddAdjacent(code, q8Vals), scaleVec));
                }
            }

            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(Vector256.Create(d3 * d8), Avx.ConvertToVector256Single(sumi), acc)
                : Avx.Add(acc, Avx.Multiply(Vector256.Create(d3 * d8), Avx.ConvertToVector256Single(sumi)));

            qk += Q3_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc) - sumBias;
    }

    // ──────────────────── ComputeRows ────────────────────

    /// <summary>Row-major Q2_K × Q8_K: one dot per output row.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ2_K(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q2_K_BlockBytes;
        if (Avx2.IsSupported)
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ2_K_Q8_KAvx2(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ2_K_Q8_KPortable(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
    }

    /// <summary>Row-major Q3_K × Q8_K: one dot per output row.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ3_K(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q3_K_BlockBytes;
        if (Avx2.IsSupported)
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ3_K_Q8_KAvx2(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ3_K_Q8_KPortable(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
    }

    // ──────────────────── Gemv / Gemm entry points ────────────────────

    /// <summary>
    /// Q2_K GEMV: weights[M,K] in Q2_K × f32 input[K] → f32 output[M].
    /// Quantizes the input to Q8_K once, then runs the fused Q2_K × Q8_K dot per row.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ2_K(byte* weights, float* x, float* result, int m, int k)
        => GemvLowBitKQuant(weights, x, result, m, k, &ComputeRowsQ2_K);

    /// <summary>Q3_K GEMV.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ3_K(byte* weights, float* x, float* result, int m, int k)
        => GemvLowBitKQuant(weights, x, result, m, k, &ComputeRowsQ3_K);

    [SkipLocalsInit]
    private static void GemvLowBitKQuant(byte* weights, float* x, float* result, int m, int k,
        delegate*<byte*, byte*, float*, int, int, void> computeRows)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = (k / Q8_K_GroupSize) * Q8_K_BlockBytes;

        if (xQ8Bytes <= StackAllocThreshold)
        {
            byte* xQ8 = stackalloc byte[xQ8Bytes];
            QuantizeF32ToQ8_K(x, xQ8, k);
            computeRows(weights, xQ8, result, m, superBlockCount);
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
        fixed (byte* xQ8 = rented)
        {
            QuantizeF32ToQ8_K(x, xQ8, k);
            computeRows(weights, xQ8, result, m, superBlockCount);
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    /// <summary>Q2_K GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ2_K(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ2_K(weights, x, result, m, k);
            return;
        }

        GemvKQuantParallel(weights, x, result, m, k, Q2_K_BlockBytes, &ComputeRowsQ2_K, pool);
    }

    /// <summary>Q3_K GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ3_K(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ3_K(weights, x, result, m, k);
            return;
        }

        GemvKQuantParallel(weights, x, result, m, k, Q3_K_BlockBytes, &ComputeRowsQ3_K, pool);
    }

    /// <summary>Q2_K GEMM: C[N,M] = B[N,K] × A[M,K]^T where A is Q2_K weights.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ2_K(byte* weights, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
        => GemmKQuant(weights, b, c, m, k, n, Q2_K_BlockBytes, &ComputeRowsQ2_K, preQuantizedInput);

    /// <summary>Q3_K GEMM.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ3_K(byte* weights, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
        => GemmKQuant(weights, b, c, m, k, n, Q3_K_BlockBytes, &ComputeRowsQ3_K, preQuantizedInput);

    /// <summary>Q2_K GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ2_K(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ2_K(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        GemmKQuantParallel(weights, b, c, m, k, n, Q2_K_BlockBytes,
            &ComputeRowsQ2_K, pool, preQuantizedInput);
    }

    /// <summary>Q3_K GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ3_K(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ3_K(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        GemmKQuantParallel(weights, b, c, m, k, n, Q3_K_BlockBytes,
            &ComputeRowsQ3_K, pool, preQuantizedInput);
    }
}
