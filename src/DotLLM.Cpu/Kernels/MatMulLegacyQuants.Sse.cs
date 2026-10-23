using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// 128-bit (SSE2/SSSE3) tier of the packed Q4_0/Q4_1/Q5_1/IQ4_NL × Q8_1 dots (issue #489, for the
/// pre-AVX2 / Westmere target of #477). Structure mirrors <c>MatMul.Q5_0Sse.cs</c>: a per-block
/// <c>*BlockDotSsse3</c> returning exact int32 lane partials, a single-row kernel that folds four
/// blocks at a time, and a 4-row kernel that puts one <em>row</em> in each lane.
///
/// <para>Both are bit-exact with the scalar references in <c>MatMulLegacyQuants.cs</c>: the block
/// sum is exact in int32, and each output performs the scalar float recurrence in block order.
/// Half scales go through the exact <see cref="MatMul.HalfToSingleSse2"/>, so no F16C is needed.</para>
///
/// <para>Saturation and the IQ4_NL sign trick are argued in the <see cref="MatMul"/> header for
/// <c>MatMulLegacyQuants.cs</c>; the 128-bit bounds are the same numbers.</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>IQ4_NL codebook as one PSHUFB table.</summary>
    private static Vector128<sbyte> Iq4NlLutSse => Vector128.Create(
        (sbyte)-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

    /// <summary>Low nibbles of <paramref name="qs"/> (elements 0–15) and high nibbles (16–31).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackNibblesSse(byte* qs, out Vector128<byte> lo, out Vector128<byte> hi)
    {
        Vector128<byte> raw = Unsafe.ReadUnaligned<Vector128<byte>>(qs);
        Vector128<byte> nib = Vector128.Create((byte)0x0F);
        lo = Sse2.And(raw, nib);
        hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), nib);
    }

    /// <summary>
    /// Exact int32 lane partials of <c>Σ q·q8</c> over one 32-element block of unsigned quants.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> UnsignedPairsSse(Vector128<byte> lo, Vector128<byte> hi,
        Vector128<sbyte> q8Lo, Vector128<sbyte> q8Hi, Vector128<short> ones)
        => Sse2.Add(
            Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(lo, q8Lo), ones),
            Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(hi, q8Hi), ones));

    /// <summary>Q4_0 block dot — lane partials of <c>Σ nibble·q8</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Q4_0BlockDotSsse3(byte* wb, Vector128<sbyte> q8Lo,
        Vector128<sbyte> q8Hi, Vector128<short> ones)
    {
        UnpackNibblesSse(wb + 2, out var lo, out var hi);
        return UnsignedPairsSse(lo, hi, q8Lo, q8Hi, ones);
    }

    /// <summary>Q4_1 block dot — identical unpack to Q4_0, <c>qs</c> starts two bytes later.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Q4_1BlockDotSsse3(byte* wb, Vector128<sbyte> q8Lo,
        Vector128<sbyte> q8Hi, Vector128<short> ones)
    {
        UnpackNibblesSse(wb + 4, out var lo, out var hi);
        return UnsignedPairsSse(lo, hi, q8Lo, q8Hi, ones);
    }

    /// <summary>Q5_1 block dot — nibbles OR'd with bit 4 spread from <c>qh</c>, as in the Q5_0 tier.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Q5_1BlockDotSsse3(byte* wb, Vector128<sbyte> q8Lo,
        Vector128<sbyte> q8Hi, Vector128<short> ones)
    {
        UnpackNibblesSse(wb + 8, out var lo, out var hi);
        Vector128<byte> qhVec = Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(wb + 4)).AsByte();
        Vector128<byte> bitMask = Q5_0_BitMask128;
        Vector128<byte> bit4 = Vector128.Create((byte)0x10);
        Vector128<byte> hbLo = Sse2.And(Sse2.CompareEqual(Sse2.And(Ssse3.Shuffle(qhVec, Q5_0_QhShuffleLo), bitMask), bitMask), bit4);
        Vector128<byte> hbHi = Sse2.And(Sse2.CompareEqual(Sse2.And(Ssse3.Shuffle(qhVec, Q5_0_QhShuffleHi), bitMask), bitMask), bit4);
        return UnsignedPairsSse(Sse2.Or(lo, hbLo), Sse2.Or(hi, hbHi), q8Lo, q8Hi, ones);
    }

    /// <summary>
    /// IQ4_NL block dot — PSHUFB the signed codebook, then ggml's sign trick so PMADDUBSW's
    /// first operand stays unsigned. Requires the Q8_1 bytes to be in [−127, 127].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Iq4NlBlockDotSsse3(byte* wb, Vector128<sbyte> q8Lo,
        Vector128<sbyte> q8Hi, Vector128<short> ones)
    {
        UnpackNibblesSse(wb + 2, out var loIdx, out var hiIdx);
        Vector128<sbyte> lut = Iq4NlLutSse;
        Vector128<sbyte> lo = Ssse3.Shuffle(lut, loIdx.AsSByte());
        Vector128<sbyte> hi = Ssse3.Shuffle(lut, hiIdx.AsSByte());
        return Sse2.Add(
            Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(Ssse3.Abs(lo), Ssse3.Sign(q8Lo, lo)), ones),
            Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(Ssse3.Abs(hi), Ssse3.Sign(q8Hi, hi)), ones));
    }

    /// <summary>Exact int32 horizontal sum of four lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HorizontalSumInt(Vector128<int> v)
    {
        v = Ssse3.HorizontalAdd(v, v);
        v = Ssse3.HorizontalAdd(v, v);
        return v.ToScalar();
    }

    /// <summary>
    /// Per-block lane partials for whichever packed legacy quant <paramref name="qt"/> names.
    /// The <c>switch</c> is hoisted out of the block loop by the JIT for a constant
    /// <paramref name="qt"/>; the callers below pass a compile-time-constant literal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> LegacyBlockDotSsse3(QuantizationType qt, byte* wb,
        Vector128<sbyte> q8Lo, Vector128<sbyte> q8Hi, Vector128<short> ones) => qt switch
    {
        QuantizationType.Q4_0 => Q4_0BlockDotSsse3(wb, q8Lo, q8Hi, ones),
        QuantizationType.Q4_1 => Q4_1BlockDotSsse3(wb, q8Lo, q8Hi, ones),
        QuantizationType.Q5_1 => Q5_1BlockDotSsse3(wb, q8Lo, q8Hi, ones),
        _ => Iq4NlBlockDotSsse3(wb, q8Lo, q8Hi, ones)
    };

    /// <summary>
    /// Single-row packed legacy-quant × Q8_1 dot, one block per step. Bit-exact with the matching
    /// <c>VecDot*Scalar</c>: block sums are exact ints and the float recurrence runs in block order.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float VecDotLegacySse(QuantizationType qt, byte* w, byte* q8, int blockCount)
    {
        int blockBytes = LegacyBlockBytes(qt);
        bool hasMin = qt is QuantizationType.Q4_1 or QuantizationType.Q5_1;
        bool isIq4 = qt == QuantizationType.IQ4_NL;
        Vector128<short> ones = Vector128.Create((short)1);

        float sumf = 0;
        float offsetSum = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * blockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            int sumi = HorizontalSumInt(LegacyBlockDotSsse3(qt, wb,
                Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 4),
                Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 20), ones));

            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            sumf += d * (float)Unsafe.ReadUnaligned<Half>(xb) * sumi;
            if (isIq4) continue;
            float offd = hasMin ? (float)Unsafe.ReadUnaligned<Half>(wb + 2) : d;
            offsetSum += offd * (float)Unsafe.ReadUnaligned<Half>(xb + 2);
        }

        if (isIq4) return sumf;
        return hasMin ? sumf + offsetSum : sumf - 8.0f * offsetSum;
    }

    /// <inheritdoc cref="VecDotQ4_0Q8_1Scalar"/>
    internal static float VecDotQ4_0Q8_1Sse(byte* w, byte* q8, int blockCount)
        => VecDotLegacySse(QuantizationType.Q4_0, w, q8, blockCount);

    /// <inheritdoc cref="VecDotQ4_1Q8_1Scalar"/>
    internal static float VecDotQ4_1Q8_1Sse(byte* w, byte* q8, int blockCount)
        => VecDotLegacySse(QuantizationType.Q4_1, w, q8, blockCount);

    /// <inheritdoc cref="VecDotQ5_1Q8_1Scalar"/>
    internal static float VecDotQ5_1Q8_1Sse(byte* w, byte* q8, int blockCount)
        => VecDotLegacySse(QuantizationType.Q5_1, w, q8, blockCount);

    /// <inheritdoc cref="VecDotIQ4_NLQ8_1Scalar"/>
    internal static float VecDotIQ4_NLQ8_1Sse(byte* w, byte* q8, int blockCount)
        => VecDotLegacySse(QuantizationType.IQ4_NL, w, q8, blockCount);

    /// <summary>
    /// Four rows at once with one <em>row</em> per 32-bit lane: the Q8_1 block is loaded once and
    /// reused across the four weight rows. Bit-exact per row with the scalar reference.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotLegacyQuantSse_4Rows(QuantizationType qt, byte* w, nint rowStride,
        byte* q8, int blockCount, float* results)
    {
        int blockBytes = LegacyBlockBytes(qt);
        bool hasMin = qt is QuantizationType.Q4_1 or QuantizationType.Q5_1;
        bool isIq4 = qt == QuantizationType.IQ4_NL;
        Vector128<short> ones = Vector128.Create((short)1);

        Vector128<float> acc = Vector128<float>.Zero;
        Vector128<float> off = Vector128<float>.Zero;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * blockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 4);
            Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 20);

            Vector128<int> s = Reduce4(
                LegacyBlockDotSsse3(qt, wb, q8Lo, q8Hi, ones),
                LegacyBlockDotSsse3(qt, wb + rowStride, q8Lo, q8Hi, ones),
                LegacyBlockDotSsse3(qt, wb + 2 * rowStride, q8Lo, q8Hi, ones),
                LegacyBlockDotSsse3(qt, wb + 3 * rowStride, q8Lo, q8Hi, ones));

            Vector128<float> d = HalfToSingleSse2(LoadHalf4(wb, rowStride));
            Vector128<float> d8 = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(xb)));
            acc = Sse.Add(acc, Sse.Multiply(Sse.Multiply(d, d8), Sse2.ConvertToVector128Single(s)));

            if (isIq4) continue;
            Vector128<float> offd = hasMin ? HalfToSingleSse2(LoadHalf4(wb + 2, rowStride)) : d;
            Vector128<float> s8 = HalfToSingleSse2(Vector128.Create((int)Unsafe.ReadUnaligned<ushort>(xb + 2)));
            off = Sse.Add(off, Sse.Multiply(offd, s8));
        }

        Vector128<float> result = isIq4 ? acc
            : hasMin ? Sse.Add(acc, off)
            : Sse.Subtract(acc, Sse.Multiply(Vector128.Create(8.0f), off));
        Unsafe.WriteUnaligned(results, result);
    }

    /// <summary>
    /// The unsigned PMADDUBSW operands (elements 0–15 and 16–31) for one block, plus the sign
    /// sources IQ4_NL needs.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackLegacySse(QuantizationType qt, byte* wb,
        out Vector128<byte> lo, out Vector128<byte> hi,
        out Vector128<sbyte> signLo, out Vector128<sbyte> signHi)
    {
        switch (qt)
        {
            case QuantizationType.Q4_0:
                UnpackNibblesSse(wb + 2, out lo, out hi);
                signLo = default; signHi = default;
                return;
            case QuantizationType.Q4_1:
                UnpackNibblesSse(wb + 4, out lo, out hi);
                signLo = default; signHi = default;
                return;
            case QuantizationType.Q5_1:
            {
                UnpackNibblesSse(wb + 8, out var nLo, out var nHi);
                Vector128<byte> qhVec = Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(wb + 4)).AsByte();
                Vector128<byte> bitMask = Q5_0_BitMask128;
                Vector128<byte> bit4 = Vector128.Create((byte)0x10);
                lo = Sse2.Or(nLo, Sse2.And(Sse2.CompareEqual(Sse2.And(Ssse3.Shuffle(qhVec, Q5_0_QhShuffleLo), bitMask), bitMask), bit4));
                hi = Sse2.Or(nHi, Sse2.And(Sse2.CompareEqual(Sse2.And(Ssse3.Shuffle(qhVec, Q5_0_QhShuffleHi), bitMask), bitMask), bit4));
                signLo = default; signHi = default;
                return;
            }
            default:
            {
                UnpackNibblesSse(wb + 2, out var loIdx, out var hiIdx);
                Vector128<sbyte> lut = Iq4NlLutSse;
                signLo = Ssse3.Shuffle(lut, loIdx.AsSByte());
                signHi = Ssse3.Shuffle(lut, hiIdx.AsSByte());
                lo = Ssse3.Abs(signLo);
                hi = Ssse3.Abs(signHi);
                return;
            }
        }
    }

    /// <inheritdoc cref="ComputeRowsLegacyQuantMulti"/>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeRowsLegacyMultiSse(QuantizationType qt, byte* weights, byte* xQ8,
        int q8RowBytes, float* c, int mTotal, int m, int n, int blockCount)
    {
        int blockBytes = LegacyBlockBytes(qt);
        bool hasMin = qt is QuantizationType.Q4_1 or QuantizationType.Q5_1;
        bool isIq4 = qt == QuantizationType.IQ4_NL;
        nint rowBytes = (nint)blockCount * blockBytes;
        Vector128<short> ones = Vector128.Create((short)1);

        float* acc = stackalloc float[LegacyMultiTokenTile];
        float* off = stackalloc float[LegacyMultiTokenTile];

        for (int row = 0; row < m; row++)
        {
            byte* wRow = weights + row * rowBytes;
            for (int t0 = 0; t0 < n; t0 += LegacyMultiTokenTile)
            {
                int nt = Math.Min(LegacyMultiTokenTile, n - t0);
                for (int j = 0; j < nt; j++) { acc[j] = 0; off[j] = 0; }

                for (int block = 0; block < blockCount; block++)
                {
                    byte* wb = wRow + block * blockBytes;
                    UnpackLegacySse(qt, wb, out var lo, out var hi, out var signLo, out var signHi);
                    float d = (float)Unsafe.ReadUnaligned<Half>(wb);
                    float offd = hasMin ? (float)Unsafe.ReadUnaligned<Half>(wb + 2) : d;

                    for (int j = 0; j < nt; j++)
                    {
                        byte* xb = xQ8 + (long)(t0 + j) * q8RowBytes + block * Q8_1BlockBytes;
                        Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 4);
                        Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(xb + 20);
                        if (isIq4)
                        {
                            q8Lo = Ssse3.Sign(q8Lo, signLo);
                            q8Hi = Ssse3.Sign(q8Hi, signHi);
                        }
                        int sumi = HorizontalSumInt(UnsignedPairsSse(lo, hi, q8Lo, q8Hi, ones));
                        acc[j] += d * (float)Unsafe.ReadUnaligned<Half>(xb) * sumi;
                        if (!isIq4) off[j] += offd * (float)Unsafe.ReadUnaligned<Half>(xb + 2);
                    }
                }

                for (int j = 0; j < nt; j++)
                    c[(long)(t0 + j) * mTotal + row] = isIq4 ? acc[j]
                        : hasMin ? acc[j] + off[j]
                        : acc[j] - 8.0f * off[j];
            }
        }
    }

    /// <summary>Row-major <see cref="ComputeRowsLegacyQuant"/> body for the SSSE3 tier.</summary>
    [SkipLocalsInit]
    internal static void ComputeRowsLegacyQuantSse(QuantizationType qt, byte* weights, byte* xQ8,
        float* result, int m, int blockCount)
    {
        nint rowBytes = (nint)blockCount * LegacyBlockBytes(qt);
        int row = 0;
        for (; row + 3 < m; row += 4)
            VecDotLegacyQuantSse_4Rows(qt, weights + row * rowBytes, rowBytes, xQ8, blockCount, result + row);
        for (; row < m; row++)
            result[row] = VecDotLegacySse(qt, weights + row * rowBytes, xQ8, blockCount);
    }
}
