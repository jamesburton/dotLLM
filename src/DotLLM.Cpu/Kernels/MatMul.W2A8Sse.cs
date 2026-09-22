using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// 128-bit (SSE2/SSSE3) tier of the ternary W2A8 (int8-activation) kernels for PQ2_0 and I2_S
/// (issue #477 — Westmere / pre-AVX hardware). Before this tier existed, both formats fell all the
/// way back to the float (W2A16) scalar path on any box without AVX2, because the W2A8 dot was
/// written against <see cref="Avx2"/> only.
///
/// <para><b>Math is identical to the AVX2 tier.</b> Each 32-element Q8_0 block is processed as two
/// 16-byte halves using the same sign trick (<c>absW = Sign(w,w)</c>, <c>adjQ = Sign(q,w)</c>),
/// <c>Ssse3.MultiplyAddAdjacent</c> (PMADDUBSW) then <c>Sse2.MultiplyAddAdjacent</c> (PMADDWD) to
/// int32. The two halves' int32 vectors are added in <b>integer</b> (exact) before the single
/// int32→float conversion, so per block the value fed into the float accumulator is the exact
/// integer block sum, just as in the AVX2 tier. The only difference is float summation order
/// across blocks (4 lanes here vs 8 there) — a last-ulp effect. PMADDUBSW cannot saturate:
/// <c>|w| ≤ 2</c> (PQ2_0 code 3 decodes to +2) and <c>|q| ≤ 127</c>, so a pair sum is ≤ 508.</para>
///
/// <para><b>No FMA on the target</b> (Westmere has none), so the accumulate is an explicit
/// multiply then add, which the JIT does not contract.</para>
///
/// <para><b>Unpacks.</b> Both formats unpack without widening: a 16-bit logical shift right by
/// the field offset followed by a byte mask of <c>0x03</c> isolates each 2-bit field in every byte
/// lane (bits leaking in from the neighbouring byte land above bit 1 and are masked away). PQ2_0's
/// fields are consecutive elements, so the four field vectors are re-interleaved with
/// <c>UnpackLow/High</c> at byte then 16-bit granularity. I2_S's fields already land on contiguous
/// 32-element slices, so they are stored directly.</para>
/// </summary>
public static unsafe partial class MatMul
{
    // ─────────────────────────── PQ2_0 ───────────────────────────

    /// <summary>
    /// 128-bit (SSSE3) twin of <see cref="VecDotPQ2_0Q8Avx2"/>:
    /// <c>Σ_blocks (d_b · g_{block/4}) · Σ_{i∈block} w[i]·q[i]</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotPQ2_0Q8Sse(sbyte* wI8, float* groupScales, byte* xQ8, int blockCount)
    {
        const int blocksPerGroup = PQ2_0GroupSize / Q8_0GroupSize; // 128 / 32 = 4

        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<short> ones = Vector128.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
            float dCombined = dx * groupScales[block / blocksPerGroup];

            Vector128<int> isum = BlockDotSsse3(wI8 + block * Q8_0GroupSize, xBlock + 2, ones);

            Vector128<float> fsum = Sse2.ConvertToVector128Single(isum);
            acc = Sse.Add(acc, Sse.Multiply(fsum, Vector128.Create(dCombined)));
        }

        return HorizontalSumSse(acc);
    }

    /// <summary>
    /// 128-bit (SSE2) twin of the AVX2 branch of <see cref="UnpackPQ2_0RowI8"/>: unpacks one PQ2_0
    /// row to int8 codes <c>{-1,0,+1,+2}</c> in true element order (byte <c>b</c> of a group holds
    /// elements <c>4b..4b+3</c> at bit offsets 0,2,4,6) plus its per-128-group float scales.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UnpackPQ2_0RowI8Sse(byte* rowPtr, sbyte* dest, float* groupScales, int k)
    {
        int groups = k / PQ2_0GroupSize;
        Vector128<byte> mask = Vector128.Create((byte)3);
        Vector128<sbyte> one = Vector128.Create((sbyte)1);

        for (int g = 0; g < groups; g++)
        {
            byte* groupBase = rowPtr + g * PQ2_0GroupBytes;
            groupScales[g] = (float)Unsafe.ReadUnaligned<Half>(groupBase);
            byte* bp = groupBase + 2;
            sbyte* outp = dest + g * PQ2_0GroupSize;

            // Two 16-byte halves; each expands to 64 consecutive elements.
            for (int h = 0; h < 2; h++)
            {
                Vector128<ushort> packed = Unsafe.ReadUnaligned<Vector128<ushort>>(bp + h * 16);

                Vector128<sbyte> f0 = Sse2.Subtract(Sse2.And(packed.AsByte(), mask).AsSByte(), one);
                Vector128<sbyte> f2 = Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(packed, 2).AsByte(), mask).AsSByte(), one);
                Vector128<sbyte> f4 = Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(packed, 4).AsByte(), mask).AsSByte(), one);
                Vector128<sbyte> f6 = Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(packed, 6).AsByte(), mask).AsSByte(), one);

                // Byte interleave: (e0,e1) pairs and (e2,e3) pairs.
                Vector128<short> loA = Sse2.UnpackLow(f0, f2).AsInt16();
                Vector128<short> loB = Sse2.UnpackLow(f4, f6).AsInt16();
                Vector128<short> hiA = Sse2.UnpackHigh(f0, f2).AsInt16();
                Vector128<short> hiB = Sse2.UnpackHigh(f4, f6).AsInt16();

                // 16-bit interleave: (e0,e1,e2,e3) quads in byte order.
                sbyte* o = outp + h * 64;
                Unsafe.WriteUnaligned(o, Sse2.UnpackLow(loA, loB).AsSByte());
                Unsafe.WriteUnaligned(o + 16, Sse2.UnpackHigh(loA, loB).AsSByte());
                Unsafe.WriteUnaligned(o + 32, Sse2.UnpackLow(hiA, hiB).AsSByte());
                Unsafe.WriteUnaligned(o + 48, Sse2.UnpackHigh(hiA, hiB).AsSByte());
            }
        }
    }

    // ─────────────────────────── I2_S ───────────────────────────

    /// <summary>
    /// 128-bit (SSSE3) twin of <see cref="VecDotI2SQ8Avx2"/>: <c>Σ_blocks d_b · Σ w[i]·q[i]</c>
    /// (the per-tensor scale is applied by the caller).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotI2SQ8Sse(sbyte* wI8, byte* xQ8, int blockCount)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<short> ones = Vector128.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            Vector128<int> isum = BlockDotSsse3(wI8 + block * Q8_0GroupSize, xBlock + 2, ones);

            Vector128<float> fsum = Sse2.ConvertToVector128Single(isum);
            acc = Sse.Add(acc, Sse.Multiply(fsum, Vector128.Create(dx)));
        }

        return HorizontalSumSse(acc);
    }

    /// <summary>
    /// 128-bit (SSE2) twin of the AVX2 branch of <see cref="UnpackRowI8"/>: within a 128-element
    /// block, byte <c>gp</c> holds elements <c>{gp, +32, +64, +96}</c> at bit offsets
    /// <c>{6, 4, 2, 0}</c>; ternary = <c>code − 1</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UnpackRowI8Sse(byte* rowPtr, sbyte* dest, int k)
    {
        int blocks = k / I2SBlockSize;
        Vector128<byte> mask = Vector128.Create((byte)3);
        Vector128<sbyte> one = Vector128.Create((sbyte)1);

        for (int blk = 0; blk < blocks; blk++)
        {
            byte* bp = rowPtr + blk * 32;
            sbyte* outp = dest + blk * I2SBlockSize;

            for (int h = 0; h < 2; h++)
            {
                Vector128<ushort> packed = Unsafe.ReadUnaligned<Vector128<ushort>>(bp + h * 16);
                sbyte* o = outp + h * 16;

                Unsafe.WriteUnaligned(o, Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(packed, 6).AsByte(), mask).AsSByte(), one));
                Unsafe.WriteUnaligned(o + 32, Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(packed, 4).AsByte(), mask).AsSByte(), one));
                Unsafe.WriteUnaligned(o + 64, Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(packed, 2).AsByte(), mask).AsSByte(), one));
                Unsafe.WriteUnaligned(o + 96, Sse2.Subtract(Sse2.And(packed.AsByte(), mask).AsSByte(), one));
            }
        }
    }

    // ─────────────────────────── bench entries ───────────────────────────

    /// <summary>
    /// Benchmark-only single-threaded PQ2_0 GEMV that always takes the 128-bit tier, regardless of
    /// the host's ISA, so it can be timed head-to-head with <see cref="GemvPQ2_0Scalar"/> on an
    /// AVX2 box. The activation is quantized with the scalar Q8_0 quantizer, as on Westmere.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemvPQ2_0Sse128ForBench(byte* weights, float* x, float* result, int m, int k,
                                                 byte* xQ8Scratch, sbyte* rowScratch, float* groupScratch)
    {
        int rowBytes = (k / PQ2_0GroupSize) * PQ2_0GroupBytes;
        int blockCount = k / Q8_0GroupSize;
        QuantizeF32ToQ8_0Scalar(x, xQ8Scratch, k);
        for (int r = 0; r < m; r++)
        {
            UnpackPQ2_0RowI8Sse(weights + (long)r * rowBytes, rowScratch, groupScratch, k);
            result[r] = VecDotPQ2_0Q8Sse(rowScratch, groupScratch, xQ8Scratch, blockCount);
        }
    }

    /// <summary>I2_S analog of <see cref="GemvPQ2_0Sse128ForBench"/> (per-tensor scale from the tail).</summary>
    [SkipLocalsInit]
    internal static void GemvI2_SSse128ForBench(byte* weights, float* x, float* result, int m, int k,
                                                byte* xQ8Scratch, sbyte* rowScratch)
    {
        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);
        int rowBytes = k / 4;
        int blockCount = k / Q8_0GroupSize;
        QuantizeF32ToQ8_0Scalar(x, xQ8Scratch, k);
        for (int r = 0; r < m; r++)
        {
            UnpackRowI8Sse(weights + (long)r * rowBytes, rowScratch, k);
            result[r] = VecDotI2SQ8Sse(rowScratch, xQ8Scratch, blockCount) * scale;
        }
    }

    // ─────────────────────────── shared ───────────────────────────

    /// <summary>
    /// Exact int32 lane sums of <c>Σ w[i]·q[i]</c> over one 32-element block (two 16-byte halves),
    /// via the sign trick + PMADDUBSW + PMADDWD. The two halves are added in integer before return,
    /// so the horizontal total of the result is the exact block sum.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> BlockDotSsse3(sbyte* w, byte* q, Vector128<short> ones)
    {
        Vector128<sbyte> w0 = Unsafe.ReadUnaligned<Vector128<sbyte>>(w);
        Vector128<sbyte> w1 = Unsafe.ReadUnaligned<Vector128<sbyte>>(w + 16);
        Vector128<sbyte> q0 = Unsafe.ReadUnaligned<Vector128<sbyte>>(q);
        Vector128<sbyte> q1 = Unsafe.ReadUnaligned<Vector128<sbyte>>(q + 16);

        Vector128<short> p0 = Ssse3.MultiplyAddAdjacent(Ssse3.Sign(w0, w0).AsByte(), Ssse3.Sign(q0, w0));
        Vector128<short> p1 = Ssse3.MultiplyAddAdjacent(Ssse3.Sign(w1, w1).AsByte(), Ssse3.Sign(q1, w1));

        return Sse2.Add(Sse2.MultiplyAddAdjacent(p0, ones), Sse2.MultiplyAddAdjacent(p1, ones));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSumSse(Vector128<float> v)
    {
        v = Sse3.HorizontalAdd(v, v);
        v = Sse3.HorizontalAdd(v, v);
        return v.ToScalar();
    }
}
