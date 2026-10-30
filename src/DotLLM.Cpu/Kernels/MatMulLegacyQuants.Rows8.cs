using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Eight-row AVX2 (+ AVX-VNNI) packed Q4_0 / Q4_1 x Q8_1 dot (#1159).
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Eight rows at once with one row per 32-bit lane, bit-exact per row with <see cref="VecDotLegacyQuantScalar"/> (the same
    /// per-block <c>(d·d8)·sumi</c> and <c>m·s8</c> recurrences as <see cref="VecDotLegacyQuantAvx2_4Rows"/>, eight wide).
    /// What makes it ~3x the four-row kernel on the 155H decode shapes (Flint's 512x2048 experts):
    /// <list type="bullet">
    ///   <item>one 32-bit gather fetches every row's <c>d</c> (and Q4_1's <c>m</c>) for the block, instead of eight to sixteen
    ///   scalar loads and inserts;</item>
    ///   <item>the nibble unpack is a 128-bit broadcast load, one shift, one blend and one mask (no cross-lane insert);</item>
    ///   <item>one <c>vpdpbusd</c> per row-block where AVX-VNNI exists;</item>
    ///   <item>the eight block sums fold with six <c>vphaddd</c> and two lane permutes (0.9 per row, against 0.75 hadd plus an
    ///   extract per row for four), and the activation's <c>d8</c>/<c>s8</c> convert once per eight rows.</item>
    /// </list>
    /// </summary>
    /// <param name="qt">Q4_0 or Q4_1.</param>
    /// <param name="w">First row's first block.</param>
    /// <param name="rowStride">Row stride in bytes.</param>
    /// <param name="q8">Q8_1 activation row.</param>
    /// <param name="blockCount">Blocks per row.</param>
    /// <param name="results">Eight outputs.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotQ4Avx2_8Rows(QuantizationType qt, byte* w, int rowStride, byte* q8, int blockCount, float* results)
    {
        bool hasMin = qt == QuantizationType.Q4_1;
        int blockBytes = hasMin ? Q4_1BlockBytes : Q4_0BlockBytes;
        int qsOffset = hasMin ? 4 : 2;
        Vector256<int> rowIdx = Vector256.Create(0, rowStride, 2 * rowStride, 3 * rowStride,
            4 * rowStride, 5 * rowStride, 6 * rowStride, 7 * rowStride);
        Vector256<byte> nib = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<float> off = Vector256<float>.Zero;
        nint rs = rowStride;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wb = w + block * blockBytes;
            byte* xb = q8 + block * Q8_1BlockBytes;
            Vector256<sbyte> q8v = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 4);
            byte* qs = wb + qsOffset;

            Vector256<int> p0 = DotU8S8Avx2(UnpackQ4Broadcast(qs, nib), q8v, ones);
            Vector256<int> p1 = DotU8S8Avx2(UnpackQ4Broadcast(qs + rs, nib), q8v, ones);
            Vector256<int> p2 = DotU8S8Avx2(UnpackQ4Broadcast(qs + 2 * rs, nib), q8v, ones);
            Vector256<int> p3 = DotU8S8Avx2(UnpackQ4Broadcast(qs + 3 * rs, nib), q8v, ones);
            Vector256<int> p4 = DotU8S8Avx2(UnpackQ4Broadcast(qs + 4 * rs, nib), q8v, ones);
            Vector256<int> p5 = DotU8S8Avx2(UnpackQ4Broadcast(qs + 5 * rs, nib), q8v, ones);
            Vector256<int> p6 = DotU8S8Avx2(UnpackQ4Broadcast(qs + 6 * rs, nib), q8v, ones);
            Vector256<int> p7 = DotU8S8Avx2(UnpackQ4Broadcast(qs + 7 * rs, nib), q8v, ones);

            // [p0..p3 low-lane partials | p0..p3 high-lane partials], likewise p4..p7; folding the lanes gives the eight sums.
            Vector256<int> s0123 = Avx2.HorizontalAdd(Avx2.HorizontalAdd(p0, p1), Avx2.HorizontalAdd(p2, p3));
            Vector256<int> s4567 = Avx2.HorizontalAdd(Avx2.HorizontalAdd(p4, p5), Avx2.HorizontalAdd(p6, p7));
            Vector256<int> sumi = Avx2.Add(
                Avx2.Permute2x128(s0123, s4567, 0x20),
                Avx2.Permute2x128(s0123, s4567, 0x31));

            // d at byte 0 and (Q4_1) m at byte 2 of every row's block: one dword gather.
            Vector256<int> dm = Avx2.GatherVector256((int*)wb, rowIdx, 1);
            Vector256<float> d = HalfToSingleAvx2(dm);
            Vector256<float> d8 = Vector256.Create(HalfToSingle(xb));
            Vector256<float> s8 = Vector256.Create(HalfToSingle(xb + 2));
            acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(d, d8), Avx.ConvertToVector256Single(sumi)));
            Vector256<float> offd = hasMin ? HalfToSingleAvx2(Avx2.ShiftRightLogical(dm, 16)) : d;
            off = Avx.Add(off, Avx.Multiply(offd, s8));
        }

        Vector256<float> result = hasMin ? Avx.Add(acc, off) : Avx.Subtract(acc, Avx.Multiply(Vector256.Create(8.0f), off));
        Unsafe.WriteUnaligned(results, result);
    }

    /// <summary>Tokens whose accumulators <see cref="ComputeRowsQ4MultiAvx2_8Rows"/> keeps on the stack per pass over the blocks.</summary>
    private const int Q4MultiTokenChunk = 32;

    /// <summary>
    /// Multi-column <see cref="VecDotQ4Avx2_8Rows"/>: <c>c[t·mTotal + row] = dot(weights[row], xQ8[t])</c> for eight rows starting at
    /// <paramref name="weights"/> and every <c>t &lt; n</c>, bit-exact per column with the scalar (and so the n = 1) dot: a token's
    /// result does not depend on how many other tokens share the GEMM, the batch invariance grouped MoE experts need (#530).
    /// <para>
    /// Each block's eight rows are unpacked once per <see cref="Q4MultiTokenChunk"/> tokens and transposed (4x4 dwords per lane,
    /// rows 4-7 unpacked lane-swapped so two blends finish it) so that lane <c>r</c> of vector <c>V_k</c> holds row <c>r</c>'s
    /// elements <c>4k..4k+3</c>. A token's eight block sums are then eight <c>vpdpbusd</c> against broadcast four-byte chunks of
    /// its Q8_1 block plus one lane swap, with no horizontal adds; the per-row unpack/transpose cost is shared by all tokens.
    /// </para>
    /// </summary>
    /// <param name="qt">Q4_0 or Q4_1.</param>
    /// <param name="weights">First of the eight rows.</param>
    /// <param name="rowStride">Row stride in bytes.</param>
    /// <param name="xQ8">Q8_1 activations, one row per token.</param>
    /// <param name="q8RowBytes">Activation row stride in bytes.</param>
    /// <param name="scales">Per token and block, the activation's <c>d8</c> and <c>s8</c> as floats (<c>[t][block][2]</c>).</param>
    /// <param name="c">Output for row 0 of token 0; token <c>t</c> starts at <c>c + t·mTotal</c>.</param>
    /// <param name="mTotal">Output row stride (the GEMM's total rows).</param>
    /// <param name="n">Token count.</param>
    /// <param name="blockCount">Blocks per row.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ4MultiAvx2_8Rows(QuantizationType qt, byte* weights, int rowStride, byte* xQ8, int q8RowBytes,
        float* scales, float* c, int mTotal, int n, int blockCount)
    {
        bool hasMin = qt == QuantizationType.Q4_1;
        int blockBytes = hasMin ? Q4_1BlockBytes : Q4_0BlockBytes;
        int qsOffset = hasMin ? 4 : 2;
        Vector256<int> rowIdx = Vector256.Create(0, rowStride, 2 * rowStride, 3 * rowStride,
            4 * rowStride, 5 * rowStride, 6 * rowStride, 7 * rowStride);
        Vector256<byte> nib = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);
        nint rs = rowStride;

        Vector256<float>* acc = stackalloc Vector256<float>[Q4MultiTokenChunk];
        Vector256<float>* off = stackalloc Vector256<float>[Q4MultiTokenChunk];

        for (int t0 = 0; t0 < n; t0 += Q4MultiTokenChunk)
        {
            int nt = Math.Min(Q4MultiTokenChunk, n - t0);
            for (int j = 0; j < nt; j++) { acc[j] = Vector256<float>.Zero; off[j] = Vector256<float>.Zero; }

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = weights + block * blockBytes;
                byte* qs = wb + qsOffset;
                Transpose4Rows(UnpackQ4Broadcast(qs, nib), UnpackQ4Broadcast(qs + rs, nib),
                    UnpackQ4Broadcast(qs + 2 * rs, nib), UnpackQ4Broadcast(qs + 3 * rs, nib),
                    out Vector256<int> a0, out Vector256<int> a1, out Vector256<int> a2, out Vector256<int> a3);
                Transpose4Rows(UnpackQ4BroadcastSwapped(qs + 4 * rs, nib), UnpackQ4BroadcastSwapped(qs + 5 * rs, nib),
                    UnpackQ4BroadcastSwapped(qs + 6 * rs, nib), UnpackQ4BroadcastSwapped(qs + 7 * rs, nib),
                    out Vector256<int> b0, out Vector256<int> b1, out Vector256<int> b2, out Vector256<int> b3);

                // V_k: rows 0-7 (lanes 0-7), elements 4k..4k+3. W_k: rows 4-7 then 0-3, elements 16+4k..16+4k+3.
                Vector256<byte> v0 = Avx2.Blend(a0, b0, 0xF0).AsByte(), w0 = Avx2.Blend(b0, a0, 0xF0).AsByte();
                Vector256<byte> v1 = Avx2.Blend(a1, b1, 0xF0).AsByte(), w1 = Avx2.Blend(b1, a1, 0xF0).AsByte();
                Vector256<byte> v2 = Avx2.Blend(a2, b2, 0xF0).AsByte(), w2 = Avx2.Blend(b2, a2, 0xF0).AsByte();
                Vector256<byte> v3 = Avx2.Blend(a3, b3, 0xF0).AsByte(), w3 = Avx2.Blend(b3, a3, 0xF0).AsByte();

                Vector256<int> dm = Avx2.GatherVector256((int*)wb, rowIdx, 1);
                Vector256<float> d = HalfToSingleAvx2(dm);
                Vector256<float> offd = hasMin ? HalfToSingleAvx2(Avx2.ShiftRightLogical(dm, 16)) : d;

                for (int j = 0; j < nt; j++)
                {
                    int* xq = (int*)(xQ8 + (long)(t0 + j) * q8RowBytes + block * Q8_1BlockBytes + 4);
                    Vector256<int> iv = DotU8S8Avx2(v0, Vector256.Create(xq[0]).AsSByte(), ones);
                    iv = DotAccU8S8Avx2(iv, v1, Vector256.Create(xq[1]).AsSByte(), ones);
                    iv = DotAccU8S8Avx2(iv, v2, Vector256.Create(xq[2]).AsSByte(), ones);
                    iv = DotAccU8S8Avx2(iv, v3, Vector256.Create(xq[3]).AsSByte(), ones);
                    Vector256<int> iw = DotU8S8Avx2(w0, Vector256.Create(xq[4]).AsSByte(), ones);
                    iw = DotAccU8S8Avx2(iw, w1, Vector256.Create(xq[5]).AsSByte(), ones);
                    iw = DotAccU8S8Avx2(iw, w2, Vector256.Create(xq[6]).AsSByte(), ones);
                    iw = DotAccU8S8Avx2(iw, w3, Vector256.Create(xq[7]).AsSByte(), ones);
                    Vector256<int> sumi = Avx2.Add(iv, Avx2.Permute2x128(iw, iw, 0x01));

                    float* sc = scales + ((long)(t0 + j) * blockCount + block) * 2;
                    acc[j] = Avx.Add(acc[j], Avx.Multiply(Avx.Multiply(d, Vector256.Create(sc[0])), Avx.ConvertToVector256Single(sumi)));
                    off[j] = Avx.Add(off[j], Avx.Multiply(offd, Vector256.Create(sc[1])));
                }
            }

            for (int j = 0; j < nt; j++)
            {
                Vector256<float> result = hasMin ? Avx.Add(acc[j], off[j])
                    : Avx.Subtract(acc[j], Avx.Multiply(Vector256.Create(8.0f), off[j]));
                Unsafe.WriteUnaligned(c + (long)(t0 + j) * mTotal, result);
            }
        }
    }

    /// <summary>Per 128-bit lane, transposes four rows' four dwords: output <c>k</c> lane-dword <c>r</c> = input <c>r</c> lane-dword <c>k</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4Rows(Vector256<byte> r0, Vector256<byte> r1, Vector256<byte> r2, Vector256<byte> r3,
        out Vector256<int> t0, out Vector256<int> t1, out Vector256<int> t2, out Vector256<int> t3)
    {
        Vector256<long> lo01 = Avx2.UnpackLow(r0.AsInt32(), r1.AsInt32()).AsInt64();
        Vector256<long> hi01 = Avx2.UnpackHigh(r0.AsInt32(), r1.AsInt32()).AsInt64();
        Vector256<long> lo23 = Avx2.UnpackLow(r2.AsInt32(), r3.AsInt32()).AsInt64();
        Vector256<long> hi23 = Avx2.UnpackHigh(r2.AsInt32(), r3.AsInt32()).AsInt64();
        t0 = Avx2.UnpackLow(lo01, lo23).AsInt32();
        t1 = Avx2.UnpackHigh(lo01, lo23).AsInt32();
        t2 = Avx2.UnpackLow(hi01, hi23).AsInt32();
        t3 = Avx2.UnpackHigh(hi01, hi23).AsInt32();
    }

    /// <summary>As <see cref="UnpackQ4Broadcast"/> with the lanes swapped: high nibbles low, low nibbles high.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> UnpackQ4BroadcastSwapped(byte* qs, Vector256<byte> nib)
    {
        Vector256<byte> raw = Vector256.Create(Unsafe.ReadUnaligned<Vector128<byte>>(qs));
        Vector256<byte> both = Avx2.Blend(raw.AsInt32(), Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsInt32(), 0x0F).AsByte();
        return Avx2.And(both, nib);
    }

    /// <summary><paramref name="acc"/> plus <see cref="DotU8S8Avx2"/> (fused into one <c>vpdpbusd</c> with AVX-VNNI).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> DotAccU8S8Avx2(Vector256<int> acc, Vector256<byte> q, Vector256<sbyte> q8v, Vector256<short> ones)
        => AvxVnni.IsSupported
            ? AvxVnni.MultiplyWideningAndAdd(acc, q, q8v)
            : Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q, q8v), ones));

    /// <summary>Low nibbles in the low lane and high nibbles in the high lane, from one 128-bit broadcast load.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> UnpackQ4Broadcast(byte* qs, Vector256<byte> nib)
    {
        Vector256<byte> raw = Vector256.Create(Unsafe.ReadUnaligned<Vector128<byte>>(qs));
        Vector256<byte> both = Avx2.Blend(raw.AsInt32(), Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsInt32(), 0xF0).AsByte();
        return Avx2.And(both, nib);
    }

    /// <summary>Eight-lane <see cref="HalfToSingleSse2"/>: exact for every Half bit pattern in the low 16 bits of each lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> HalfToSingleAvx2(Vector256<int> h)
    {
        Vector256<int> em = Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x7FFF)), 13);
        Vector256<float> f = Avx.Multiply(em.AsSingle(), Vector256.Create(HalfToSingleRebias));
        Vector256<int> infNan = Avx2.CompareGreaterThan(em, Vector256.Create(0x0F7FFFFF));
        Vector256<int> sign = Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x8000)), 16);
        return Avx.Or(f, Avx2.Or(Avx2.And(infNan, Vector256.Create(0x7F800000)), sign).AsSingle());
    }
}
