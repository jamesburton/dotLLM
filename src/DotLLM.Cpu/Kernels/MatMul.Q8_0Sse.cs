using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// 128-bit (SSE2/SSE3/SSSE3) tier of the Q8_0 x Q8_0 dot kernels (issue #477 — Westmere /
/// pre-AVX2 hardware). Before this tier, every Q8_0 GEMV/GEMM on a box without AVX2 ran the
/// scalar per-element loop, including the R4-interleaved decode path that the transformer uses
/// for every repacked projection.
///
/// <para><b>Bit-exact with the scalar tier.</b> Each 32-element block is reduced to its exact
/// int32 sum (sign trick + PMADDUBSW + PMADDWD, then PHADDD), and the float accumulation is laid
/// out so that each output lane performs exactly the scalar recurrence
/// <c>sumf += (dw * dx) * (float)sumi</c> in block order: the 4-row kernels put one <em>row</em>
/// in each lane, the single-row kernel adds its four per-block products to the scalar sum in
/// block order. No FMA exists on the target and the JIT does not contract, so the results equal
/// <see cref="VecDotQ8_0Scalar"/> / <see cref="VecDotQ8_0ScalarR4"/> bit for bit. (The AVX2 tier
/// instead accumulates per-lane partial sums and differs from both in the last ulps.)</para>
///
/// <para><b>Saturation / -128.</b> Via <see cref="BlockDotSsse3"/> the unsigned PMADDUBSW operand
/// is the <em>weight</em> magnitude <c>|w|</c> (a weight byte of <c>-128</c> becomes the unsigned
/// 128, which is correct) and the signed one is <c>Sign(x, w)</c> with the activation from our
/// quantizer (<c>|x| ≤ 127</c>, never <c>-128</c>). A pair sum is at most <c>2·128·127 = 32512 &lt;
/// 32767</c>, so nothing saturates and every weight byte, including <c>-128</c>, matches the scalar
/// tier. (The AVX2 tier puts <c>|x|</c> unsigned and <c>Sign(w, x)</c> signed, which wraps a
/// <c>-128</c> weight against a negative activation; llama.cpp never emits <c>-128</c>.)</para>
///
/// <para><b>Half scales without F16C.</b> <c>(float)Half</c> is a managed software routine on
/// .NET 10 (~1.2 ns) and Westmere has no F16C, so converting two scales per block would cost more
/// than the SIMD block dot. <see cref="HalfToSingleSse2"/> converts four at once with integer ops
/// and one multiply, and is exact for every Half bit pattern (checked exhaustively in the tests).</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>2^112: rebias from the Half exponent (bias 15) to the Single exponent (bias 127).</summary>
    private const float HalfToSingleRebias = 5.192296858534828e33f; // 0x77800000

    /// <summary>
    /// Converts four Half bit patterns (low 16 bits of each int32 lane, upper bits zero) to Single,
    /// exactly as <c>(float)Half</c> does for every input including subnormals, ±0, ±Inf and NaN.
    /// </summary>
    /// <remarks>
    /// Shifting the 15 magnitude bits left by 13 places the Half exponent/mantissa in the Single
    /// exponent/mantissa fields with the Half bias; multiplying by 2^112 rebiases it. For a Half
    /// subnormal the shifted pattern is a Single subnormal whose value times 2^112 is exact. For
    /// exponent 31 (Inf/NaN) the product has Single exponent 143 (<c>0b10001111</c>), so OR-ing the
    /// all-ones exponent yields 255 with the mantissa (NaN payload) preserved.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> HalfToSingleSse2(Vector128<int> h)
    {
        Vector128<int> em = Sse2.ShiftLeftLogical(Sse2.And(h, Vector128.Create(0x7FFF)), 13);
        Vector128<float> f = Sse.Multiply(em.AsSingle(), Vector128.Create(HalfToSingleRebias));
        Vector128<int> infNan = Sse2.CompareGreaterThan(em, Vector128.Create(0x0F7FFFFF));
        Vector128<int> sign = Sse2.ShiftLeftLogical(Sse2.And(h, Vector128.Create(0x8000)), 16);
        return Sse.Or(f, Sse2.Or(Sse2.And(infNan, Vector128.Create(0x7F800000)), sign).AsSingle());
    }

    /// <summary>Loads four Half scales at byte offsets <c>0, stride, 2·stride, 3·stride</c> as int32 lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> LoadHalf4(byte* p, nint stride) => Vector128.Create(
        (int)Unsafe.ReadUnaligned<ushort>(p),
        Unsafe.ReadUnaligned<ushort>(p + stride),
        Unsafe.ReadUnaligned<ushort>(p + 2 * stride),
        Unsafe.ReadUnaligned<ushort>(p + 3 * stride));

    /// <summary>
    /// Reduces four blocks' int32 lane partials to <c>[Σa, Σb, Σc, Σd]</c> (exact).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Reduce4(Vector128<int> a, Vector128<int> b, Vector128<int> c, Vector128<int> d)
        => Ssse3.HorizontalAdd(Ssse3.HorizontalAdd(a, b), Ssse3.HorizontalAdd(c, d));

    /// <summary>
    /// 4-row Q8_0 dot with one row per lane. Row <c>r</c>'s block <c>b</c> is at
    /// <c>w + r·rowStride + b·blockStride</c>, which covers both the row-major layout
    /// (<c>rowStride = rowBytes, blockStride = 34</c>) and the R4-interleaved one
    /// (<c>rowStride = 34, blockStride = 136</c>). Bit-exact with <see cref="VecDotQ8_0Scalar"/>
    /// per row.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotQ8_0Sse_4Rows(byte* w, nint rowStride, nint blockStride,
        byte* x, int blockCount, float* results)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<short> ones = Vector128.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            byte* wBlock = w + block * blockStride;

            Vector128<int> s = Reduce4(
                BlockDotSsse3((sbyte*)(wBlock + 2), xBlock + 2, ones),
                BlockDotSsse3((sbyte*)(wBlock + rowStride + 2), xBlock + 2, ones),
                BlockDotSsse3((sbyte*)(wBlock + 2 * rowStride + 2), xBlock + 2, ones),
                BlockDotSsse3((sbyte*)(wBlock + 3 * rowStride + 2), xBlock + 2, ones));

            // Lane r: (dw_r * dx) * (float)sumi_r — the scalar recurrence, one row per lane.
            Vector128<float> dw = HalfToSingleSse2(LoadHalf4(wBlock, rowStride));
            Vector128<float> dx = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(xBlock)));
            acc = Sse.Add(acc, Sse.Multiply(Sse.Multiply(dw, dx), Sse2.ConvertToVector128Single(s)));
        }

        Unsafe.WriteUnaligned(results, acc);
    }

    /// <summary>
    /// Single-row Q8_0 dot, four blocks per step. Bit-exact with <see cref="VecDotQ8_0Scalar"/>:
    /// the four per-block products are added to the scalar sum in block order.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotQ8_0Sse(byte* a, byte* b, int blockCount)
    {
        float sumf = 0;
        Vector128<short> ones = Vector128.Create((short)1);
        int block = 0;

        for (; block + 3 < blockCount; block += 4)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            Vector128<int> s = Reduce4(
                BlockDotSsse3((sbyte*)(aBlock + 2), bBlock + 2, ones),
                BlockDotSsse3((sbyte*)(aBlock + Q8_0BlockBytes + 2), bBlock + Q8_0BlockBytes + 2, ones),
                BlockDotSsse3((sbyte*)(aBlock + 2 * Q8_0BlockBytes + 2), bBlock + 2 * Q8_0BlockBytes + 2, ones),
                BlockDotSsse3((sbyte*)(aBlock + 3 * Q8_0BlockBytes + 2), bBlock + 3 * Q8_0BlockBytes + 2, ones));

            Vector128<float> da = HalfToSingleSse2(LoadHalf4(aBlock, Q8_0BlockBytes));
            Vector128<float> db = HalfToSingleSse2(LoadHalf4(bBlock, Q8_0BlockBytes));
            Vector128<float> p = Sse.Multiply(Sse.Multiply(da, db), Sse2.ConvertToVector128Single(s));

            sumf += p.ToScalar();
            sumf += p.GetElement(1);
            sumf += p.GetElement(2);
            sumf += p.GetElement(3);
        }

        for (; block < blockCount; block++)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;
            Vector128<int> v = BlockDotSsse3((sbyte*)(aBlock + 2), bBlock + 2, ones);
            v = Ssse3.HorizontalAdd(v, v);
            v = Ssse3.HorizontalAdd(v, v);
            float da = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float db = (float)Unsafe.ReadUnaligned<Half>(bBlock);
            sumf += da * db * v.ToScalar();
        }

        return sumf;
    }

    /// <summary>Row-major <see cref="ComputeRows(byte*, byte*, float*, int, int)"/> body for the SSSE3 tier.</summary>
    [SkipLocalsInit]
    internal static void ComputeRowsQ8_0Sse(byte* weightsQ8, byte* xQ8, float* result, int m, int blockCount)
    {
        nint rowBytes = (nint)blockCount * Q8_0BlockBytes;
        int row = 0;
        for (; row + 3 < m; row += 4)
            VecDotQ8_0Sse_4Rows(weightsQ8 + row * rowBytes, rowBytes, Q8_0BlockBytes, xQ8, blockCount, result + row);
        for (; row < m; row++)
            result[row] = VecDotQ8_0Sse(weightsQ8 + row * rowBytes, xQ8, blockCount);
    }

    /// <summary>R4-interleaved 4-row group for the SSSE3 tier (blocks of the 4 rows are adjacent).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void VecDotQ8_0Sse_4RowsR4(byte* groupBase, byte* xQ8, int blockCount, float* results)
        => VecDotQ8_0Sse_4Rows(groupBase, Q8_0BlockBytes, 4 * Q8_0BlockBytes, xQ8, blockCount, results);

    /// <summary>Best available single-row Q8_0 dot (AVX2, then SSSE3, then scalar) for tail rows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotQ8_0Row(byte* w, byte* xQ8, int blockCount)
    {
        if (Avx2.IsSupported) return VecDotQ8_0Avx2(w, xQ8, blockCount);
        if (Ssse3.IsSupported) return VecDotQ8_0Sse(w, xQ8, blockCount);
        return VecDotQ8_0Scalar(w, xQ8, blockCount);
    }
}
