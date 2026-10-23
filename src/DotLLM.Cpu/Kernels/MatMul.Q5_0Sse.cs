using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// 128-bit (SSE2/SSSE3) tier of the Q5_0 x Q8_1 dot kernels (issue #477 — Westmere /
/// pre-AVX2 hardware). Before this tier every Q5_0 GEMV/GEMM on a box without AVX2 ran the scalar
/// per-element loop: the row-major path (<see cref="ComputeRowsQ5_0"/>, used by prefill through
/// <see cref="GemmQ5_0(byte*, float*, float*, int, int, int, byte*)"/>) and the R4-interleaved
/// decode path. Q5_0 is common in llama.cpp "Q4_K_M" files wherever a tensor's row length is not
/// a multiple of 256 (SmolLM-135M: every k = 576 projection).
///
/// <para><b>Block dot.</b> The 16 <c>qs</c> bytes give elements 0–15 (low nibbles) and 16–31
/// (high nibbles); bit <c>j</c> of <c>qh</c> is bit 4 of element <c>j</c>, spread with PSHUFB +
/// bit-mask compare exactly like the AVX2 <see cref="ExtractQ5HighBits"/>. The weight value is
/// the <em>unsigned</em> 0..31 (the −16 is applied through the Q8_1 block sum <c>s</c>), so it is
/// PMADDUBSW's unsigned operand directly — no sign trick. A pair sum is at most
/// <c>2·31·128 = 7936</c>, so nothing saturates for any activation byte, including −128: unlike
/// the Q8_0 tier there is no precondition on the Q8_1 operand.</para>
///
/// <para><b>Bit-exact with the scalar tier.</b> The per-block int32 sum is exact, and each output
/// lane performs exactly the scalar recurrence
/// <c>sumf += (d5 * d8) * (float)sumi; offsetSum += d5 * s8</c> in block order, finishing with
/// <c>sumf - 16f * offsetSum</c>: the 4-row kernel puts one <em>row</em> in each lane, the
/// single-row kernel adds its four per-block products in block order. Half scales go through the
/// exact <see cref="HalfToSingleSse2"/>. So results equal <see cref="VecDotQ5_0Q8_1Scalar"/> /
/// <see cref="VecDotQ5_0Q8_1ScalarR4"/> bit for bit (the AVX2 tier sums per-lane partials and
/// differs in the last ulps).</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>PSHUFB mask: qh byte 0 → lanes 0–7, byte 1 → lanes 8–15 (elements 0–15).</summary>
    private static readonly Vector128<byte> Q5_0_QhShuffleLo = Vector128.Create(
        (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);

    /// <summary>PSHUFB mask: qh byte 2 → lanes 0–7, byte 3 → lanes 8–15 (elements 16–31).</summary>
    private static readonly Vector128<byte> Q5_0_QhShuffleHi = Vector128.Create(
        (byte)2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);

    /// <summary>Single-bit masks <c>[1, 2, 4, …, 128]</c> twice.</summary>
    private static readonly Vector128<byte> Q5_0_BitMask128 = Vector128.Create(
        (byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);

    /// <summary>
    /// Exact int32 lane partials of <c>Σ q5u[i]·q8[i]</c> over one 32-element Q5_0 block, where
    /// <c>q5u</c> is the unsigned 5-bit value (0..31). The horizontal total is the block sum.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Q5_0BlockDotSsse3(byte* q5Block, Vector128<sbyte> q8Lo, Vector128<sbyte> q8Hi,
        Vector128<short> ones)
    {
        Vector128<byte> qhVec = Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(q5Block + 2)).AsByte();
        Vector128<byte> qs = Unsafe.ReadUnaligned<Vector128<byte>>(q5Block + 6);
        Vector128<byte> nib = Vector128.Create((byte)0x0F);
        Vector128<byte> bit4 = Vector128.Create((byte)0x10);
        Vector128<byte> bitMask = Q5_0_BitMask128;

        Vector128<byte> hbLo = Sse2.And(Sse2.CompareEqual(Sse2.And(Ssse3.Shuffle(qhVec, Q5_0_QhShuffleLo), bitMask), bitMask), bit4);
        Vector128<byte> hbHi = Sse2.And(Sse2.CompareEqual(Sse2.And(Ssse3.Shuffle(qhVec, Q5_0_QhShuffleHi), bitMask), bitMask), bit4);

        Vector128<byte> lo = Sse2.Or(Sse2.And(qs, nib), hbLo);
        Vector128<byte> hi = Sse2.Or(Sse2.And(Sse2.ShiftRightLogical(qs.AsUInt16(), 4).AsByte(), nib), hbHi);

        Vector128<short> p0 = Ssse3.MultiplyAddAdjacent(lo, q8Lo);
        Vector128<short> p1 = Ssse3.MultiplyAddAdjacent(hi, q8Hi);
        return Sse2.Add(Sse2.MultiplyAddAdjacent(p0, ones), Sse2.MultiplyAddAdjacent(p1, ones));
    }

    /// <summary>
    /// 4-row Q5_0 x Q8_1 dot with one row per lane. Row <c>r</c>'s block <c>b</c> is at
    /// <c>w + r·rowStride + b·blockStride</c>: row-major is <c>(rowBytes, 22)</c>, R4-interleaved
    /// is <c>(22, 88)</c>. Bit-exact with <see cref="VecDotQ5_0Q8_1Scalar"/> per row.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotQ5_0Q8_1Sse_4Rows(byte* w, nint rowStride, nint blockStride,
        byte* q8, int blockCount, float* results)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<float> off = Vector128<float>.Zero;
        Vector128<short> ones = Vector128.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            byte* q8Block = q8 + block * Q8_1BlockBytes;
            byte* wBlock = w + block * blockStride;
            Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8Block + 4);
            Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8Block + 20);

            Vector128<int> s = Reduce4(
                Q5_0BlockDotSsse3(wBlock, q8Lo, q8Hi, ones),
                Q5_0BlockDotSsse3(wBlock + rowStride, q8Lo, q8Hi, ones),
                Q5_0BlockDotSsse3(wBlock + 2 * rowStride, q8Lo, q8Hi, ones),
                Q5_0BlockDotSsse3(wBlock + 3 * rowStride, q8Lo, q8Hi, ones));

            // Lane r: sumf += (d5_r * d8) * (float)sumi_r; offsetSum += d5_r * s8.
            Vector128<float> d5 = HalfToSingleSse2(LoadHalf4(wBlock, rowStride));
            Vector128<float> d8 = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(q8Block)));
            Vector128<float> s8 = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(q8Block + 2)));
            acc = Sse.Add(acc, Sse.Multiply(Sse.Multiply(d5, d8), Sse2.ConvertToVector128Single(s)));
            off = Sse.Add(off, Sse.Multiply(d5, s8));
        }

        Unsafe.WriteUnaligned(results, Sse.Subtract(acc, Sse.Multiply(Vector128.Create(16.0f), off)));
    }

    /// <summary>
    /// Single-row Q5_0 x Q8_1 dot, four blocks per step. Bit-exact with
    /// <see cref="VecDotQ5_0Q8_1Scalar"/>: per-block products and offsets are added in block order.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ5_0Q8_1Sse(byte* q5, byte* q8, int blockCount)
    {
        float sumf = 0;
        float offsetSum = 0;
        Vector128<short> ones = Vector128.Create((short)1);
        int block = 0;

        for (; block + 3 < blockCount; block += 4)
        {
            byte* wb = q5 + block * Q5_0BlockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;

            Vector128<int> s = Reduce4(
                Q5_0BlockDotSsse3(wb,
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 4), Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 20), ones),
                Q5_0BlockDotSsse3(wb + Q5_0BlockBytes,
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + Q8_1BlockBytes + 4),
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + Q8_1BlockBytes + 20), ones),
                Q5_0BlockDotSsse3(wb + 2 * Q5_0BlockBytes,
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 2 * Q8_1BlockBytes + 4),
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 2 * Q8_1BlockBytes + 20), ones),
                Q5_0BlockDotSsse3(wb + 3 * Q5_0BlockBytes,
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 3 * Q8_1BlockBytes + 4),
                    Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 3 * Q8_1BlockBytes + 20), ones));

            Vector128<float> d5 = HalfToSingleSse2(LoadHalf4(wb, Q5_0BlockBytes));
            Vector128<float> d8 = HalfToSingleSse2(LoadHalf4(xb, Q8_1BlockBytes));
            Vector128<float> s8 = HalfToSingleSse2(LoadHalf4(xb + 2, Q8_1BlockBytes));
            Vector128<float> p = Sse.Multiply(Sse.Multiply(d5, d8), Sse2.ConvertToVector128Single(s));
            Vector128<float> o = Sse.Multiply(d5, s8);

            sumf += p.ToScalar();
            sumf += p.GetElement(1);
            sumf += p.GetElement(2);
            sumf += p.GetElement(3);
            offsetSum += o.ToScalar();
            offsetSum += o.GetElement(1);
            offsetSum += o.GetElement(2);
            offsetSum += o.GetElement(3);
        }

        for (; block < blockCount; block++)
        {
            byte* wb = q5 + block * Q5_0BlockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            Vector128<int> v = Q5_0BlockDotSsse3(wb,
                Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 4), Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 20), ones);
            v = Ssse3.HorizontalAdd(v, v);
            v = Ssse3.HorizontalAdd(v, v);
            float d5 = (float)Unsafe.ReadUnaligned<Half>(wb);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(xb);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(xb + 2);
            sumf += d5 * d8 * v.ToScalar();
            offsetSum += d5 * s8;
        }

        return sumf - 16.0f * offsetSum;
    }

    /// <summary>Row-major <see cref="ComputeRowsQ5_0"/> body for the SSSE3 tier.</summary>
    [SkipLocalsInit]
    internal static void ComputeRowsQ5_0Sse(byte* weights, byte* xQ8, float* result, int m, int blockCount)
    {
        nint rowBytes = (nint)blockCount * Q5_0BlockBytes;
        int row = 0;
        for (; row + 3 < m; row += 4)
            VecDotQ5_0Q8_1Sse_4Rows(weights + row * rowBytes, rowBytes, Q5_0BlockBytes, xQ8, blockCount, result + row);
        for (; row < m; row++)
            result[row] = VecDotQ5_0Q8_1Sse(weights + row * rowBytes, xQ8, blockCount);
    }

    /// <summary>R4-interleaved 4-row group for the SSSE3 tier (blocks of the 4 rows are adjacent).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void VecDotQ5_0Q8_1Sse_4RowsR4(byte* groupBase, byte* xQ8, int blockCount, float* results)
        => VecDotQ5_0Q8_1Sse_4Rows(groupBase, Q5_0BlockBytes, 4 * Q5_0BlockBytes, xQ8, blockCount, results);

    /// <summary>Best available single-row Q5_0 dot (AVX2, then SSSE3, then scalar) for tail rows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotQ5_0Q8_1Row(byte* w, byte* xQ8, int blockCount)
    {
        if (Avx2.IsSupported) return VecDotQ5_0Q8_1Avx2(w, xQ8, blockCount);
        if (Ssse3.IsSupported) return VecDotQ5_0Q8_1Sse(w, xQ8, blockCount);
        return VecDotQ5_0Q8_1Scalar(w, xQ8, blockCount);
    }

    /// <summary>Best available R4 4-row Q5_0 group dot (AVX2, then SSSE3, then scalar).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void VecDotQ5_0Q8_1Group4R4(byte* groupBase, byte* xQ8, int blockCount, float* results)
    {
        if (Avx2.IsSupported)
            VecDotQ5_0Q8_1Avx2_4RowsR4(groupBase, xQ8, blockCount, results);
        else if (Ssse3.IsSupported)
            VecDotQ5_0Q8_1Sse_4RowsR4(groupBase, xQ8, blockCount, results);
        else
            for (int r = 0; r < 4; r++)
                results[r] = VecDotQ5_0Q8_1ScalarR4(groupBase, r, xQ8, blockCount);
    }
}
