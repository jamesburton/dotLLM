using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Eight-row, multi-column AVX2 (+ AVX-VNNI) NVFP4 x Q8_0 dot (#1587), the NVFP4 counterpart of the Q4_0/Q4_1 kernel in
/// <c>MatMulLegacyQuants.Rows8.cs</c>.
/// </summary>
/// <remarks>
/// <para><b>Why.</b> The row-at-a-time NVFP4 GEMM re-unpacked every weight block (pshufb LUT, shuffles, four scale loads and
/// converts) once per token, so prefill cost ~25 vector ops per 64 weights per token against Q4_1's amortised ~8 (4.6x slower on
/// Flint's attn_q shape at n = 256). This kernel unpacks and transposes the eight rows' block once per
/// <see cref="Nvfp4MultiTokenChunk"/> tokens, then each token costs 16 <c>vpdpbusd</c> plus four float steps.</para>
/// <para><b>Bit-exact with the scalar dot.</b> Rows sit one per 32-bit lane, so each row's float recurrence is the scalar one
/// (<c>sumf += (dy·d)·sumi</c>, sub-blocks in order, no FMA). The integer sub-block sums are exact: the kvalue table is shifted to
/// an unsigned <c>u = kvalue + 12</c> (0..24) so <c>vpdpbusd</c> can take it as the unsigned operand, and the
/// <c>-12·Σq</c> correction is folded into the accumulator's starting value (it is shared by every row, so it is computed once per
/// token, not per row). A token's result therefore does not depend on how many other tokens share the GEMM (the batch invariance
/// grouped MoE needs), and equals the n = 1 / decode path.</para>
/// </remarks>
public static unsafe partial class MatMul
{
    /// <summary>Tokens whose accumulators the NVFP4 8-row kernel keeps on the stack per pass over the blocks.</summary>
    private const int Nvfp4MultiTokenChunk = 64;

    /// <summary>Offset added to each kvalue so it is unsigned: the E2M1 table spans [-12, 12] (doubled).</summary>
    private const int Nvfp4UnsignedBias = 12;

    /// <summary>UE4M3 half-scale table in native memory, for <c>vgatherdps</c> (indexed by the raw scale byte).</summary>
    private static readonly float* Nvfp4ScaleNative = BuildNvfp4ScaleNative();

    private static float* BuildNvfp4ScaleNative()
    {
        float* p = (float*)NativeMemory.AlignedAlloc(256 * sizeof(float), 64);
        for (int i = 0; i < 256; i++) p[i] = Dequantize.Ue4m3ToFloatHalfReference((byte)i);
        return p;
    }

    /// <summary>
    /// Per-token activation side data the 8-row kernel needs: for each Q8_0 block <c>dy</c> as a float, and for each of the four
    /// NVFP4 sub-blocks per 64 elements <c>-12·Σq</c> (the unsigned-bias correction, exact int32).
    /// </summary>
    /// <param name="xQ8">Q8_0 rows, <paramref name="n"/> rows of <c>2·blockCount</c> 34-byte blocks.</param>
    /// <param name="q8RowBytes">Row stride in bytes.</param>
    /// <param name="n">Token count.</param>
    /// <param name="blockCount">NVFP4 blocks per row.</param>
    /// <param name="dy">Out: <c>[t][2·blockCount]</c> floats.</param>
    /// <param name="corr">Out: <c>[t][4·blockCount]</c> ints.</param>
    [SkipLocalsInit]
    internal static void PrepNvfp4Activations(byte* xQ8, int q8RowBytes, int n, int blockCount, float* dy, int* corr)
    {
        for (int t = 0; t < n; t++)
        {
            byte* row = xQ8 + (long)t * q8RowBytes;
            float* dyT = dy + (long)t * 2 * blockCount;
            int* cT = corr + (long)t * 4 * blockCount;
            for (int qb = 0; qb < 2 * blockCount; qb++)
            {
                byte* y = row + qb * Q8_0BlockBytes;
                dyT[qb] = HalfToSingle(y);
                cT[2 * qb] = -Nvfp4UnsignedBias * SumSbytes16(y + 2);
                cT[2 * qb + 1] = -Nvfp4UnsignedBias * SumSbytes16(y + 18);
            }
        }
    }

    /// <summary>Exact sum of 16 signed bytes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SumSbytes16(byte* p)
    {
        if (Sse2.IsSupported && Ssse3.IsSupported)
        {
            Vector128<short> pairs = Ssse3.MultiplyAddAdjacent(Vector128.Create((byte)1), Sse2.LoadVector128((sbyte*)p));
            Vector128<int> quads = Sse2.MultiplyAddAdjacent(pairs, Vector128.Create((short)1));
            quads = Sse2.Add(quads, Sse2.Shuffle(quads, 0x4E));
            quads = Sse2.Add(quads, Sse2.Shuffle(quads, 0xB1));
            return quads.ToScalar();
        }
        int s = 0;
        for (int i = 0; i < 16; i++) s += (sbyte)p[i];
        return s;
    }

    /// <summary>Whether the 8-row NVFP4 kernel can run (AVX2 gather/permutes; VNNI optional).</summary>
    private static bool Nvfp4Rows8 => Avx2.IsSupported && Fma.IsSupported && Environment.GetEnvironmentVariable("DOTLLM_CPU_NVFP4_ROWS8") != "0";

    private static readonly bool Nvfp4Rows8Enabled = Nvfp4Rows8;

    /// <summary>
    /// <c>c[t·mTotal + row] = dot(weights[row], xQ8[t])</c> for eight rows starting at <paramref name="weights"/> and every
    /// <c>t &lt; n</c>, bit-exact per column with <see cref="VecDotNvfp4Q8_0Scalar"/>.
    /// </summary>
    /// <param name="weights">First of the eight rows.</param>
    /// <param name="rowStride">Row stride in bytes.</param>
    /// <param name="xQ8">Q8_0 activations (<c>q8RowBytes</c> per token).</param>
    /// <param name="q8RowBytes">Activation row stride in bytes.</param>
    /// <param name="dy">Per token, per Q8_0 block scale (<see cref="PrepNvfp4Activations"/>).</param>
    /// <param name="corr">Per token, per sub-block <c>-12·Σq</c> (<see cref="PrepNvfp4Activations"/>).</param>
    /// <param name="c">Output for row 0 of token 0.</param>
    /// <param name="mTotal">Output row stride.</param>
    /// <param name="n">Token count.</param>
    /// <param name="blockCount">NVFP4 blocks per row.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsNvfp4Avx2_8Rows(byte* weights, int rowStride, byte* xQ8, int q8RowBytes,
        float* dy, int* corr, float* c, int mTotal, int n, int blockCount)
    {
        Vector256<int> rowIdx = Vector256.Create(0, rowStride, 2 * rowStride, 3 * rowStride,
            4 * rowStride, 5 * rowStride, 6 * rowStride, 7 * rowStride);
        Vector256<byte> nib = Vector256.Create((byte)0x0F);
        Vector256<byte> table = Vector256.Create(Vector128.Create(
            (byte)12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
        Vector256<short> ones = Vector256.Create((short)1);
        Vector256<int> byteMask = Vector256.Create(0xFF);
        nint rs = rowStride;
        float* scaleTable = Nvfp4ScaleNative;

        Vector256<float>* acc = stackalloc Vector256<float>[Nvfp4MultiTokenChunk];
        Vector256<byte>* v = stackalloc Vector256<byte>[16];
        Vector256<float>* wd = stackalloc Vector256<float>[4];

        for (int t0 = 0; t0 < n; t0 += Nvfp4MultiTokenChunk)
        {
            int nt = Math.Min(Nvfp4MultiTokenChunk, n - t0);
            for (int j = 0; j < nt; j++) acc[j] = Vector256<float>.Zero;

            for (int ib = 0; ib < blockCount; ib++)
            {
                byte* wb = weights + (long)ib * Dequantize.Nvfp4BlockBytes;
                byte* qs = wb + Dequantize.Nvfp4QsOffset;

                // Unpack rows 0-3 and 4-7 to unsigned codes, arranged [sub-block s lane-wise][L0 L1 H0 H1] per row.
                Vector256<int> a0, a1, a2, a3, b0, b1, b2, b3, e0, e1, e2, e3, f0, f1, f2, f3;
                UnpackNvfp4Row(qs, table, nib, out Vector256<long> ar0, out Vector256<long> br0);
                UnpackNvfp4Row(qs + rs, table, nib, out Vector256<long> ar1, out Vector256<long> br1);
                UnpackNvfp4Row(qs + 2 * rs, table, nib, out Vector256<long> ar2, out Vector256<long> br2);
                UnpackNvfp4Row(qs + 3 * rs, table, nib, out Vector256<long> ar3, out Vector256<long> br3);
                Transpose4Rows(ar0.AsByte(), ar1.AsByte(), ar2.AsByte(), ar3.AsByte(), out a0, out a1, out a2, out a3);
                Transpose4Rows(br0.AsByte(), br1.AsByte(), br2.AsByte(), br3.AsByte(), out b0, out b1, out b2, out b3);
                UnpackNvfp4Row(qs + 4 * rs, table, nib, out Vector256<long> ar4, out Vector256<long> br4);
                UnpackNvfp4Row(qs + 5 * rs, table, nib, out Vector256<long> ar5, out Vector256<long> br5);
                UnpackNvfp4Row(qs + 6 * rs, table, nib, out Vector256<long> ar6, out Vector256<long> br6);
                UnpackNvfp4Row(qs + 7 * rs, table, nib, out Vector256<long> ar7, out Vector256<long> br7);
                Transpose4Rows(ar4.AsByte(), ar5.AsByte(), ar6.AsByte(), ar7.AsByte(), out e0, out e1, out e2, out e3);
                Transpose4Rows(br4.AsByte(), br5.AsByte(), br6.AsByte(), br7.AsByte(), out f0, out f1, out f2, out f3);

                // A = lanes {sub-block 0 | sub-block 2}, B = lanes {sub-block 1 | sub-block 3}; rows 0-3 in a*/b*, rows 4-7 in e*/f*.
                // v[4s + k] lane r = row r's dword k of sub-block s.
                v[0] = Avx2.Permute2x128(a0, e0, 0x20).AsByte(); v[1] = Avx2.Permute2x128(a1, e1, 0x20).AsByte();
                v[2] = Avx2.Permute2x128(a2, e2, 0x20).AsByte(); v[3] = Avx2.Permute2x128(a3, e3, 0x20).AsByte();
                v[8] = Avx2.Permute2x128(a0, e0, 0x31).AsByte(); v[9] = Avx2.Permute2x128(a1, e1, 0x31).AsByte();
                v[10] = Avx2.Permute2x128(a2, e2, 0x31).AsByte(); v[11] = Avx2.Permute2x128(a3, e3, 0x31).AsByte();
                v[4] = Avx2.Permute2x128(b0, f0, 0x20).AsByte(); v[5] = Avx2.Permute2x128(b1, f1, 0x20).AsByte();
                v[6] = Avx2.Permute2x128(b2, f2, 0x20).AsByte(); v[7] = Avx2.Permute2x128(b3, f3, 0x20).AsByte();
                v[12] = Avx2.Permute2x128(b0, f0, 0x31).AsByte(); v[13] = Avx2.Permute2x128(b1, f1, 0x31).AsByte();
                v[14] = Avx2.Permute2x128(b2, f2, 0x31).AsByte(); v[15] = Avx2.Permute2x128(b3, f3, 0x31).AsByte();

                // The four scale bytes of every row (one dword gather), then one table gather per sub-block.
                Vector256<int> sc = Avx2.GatherVector256((int*)wb, rowIdx, 1);
                wd[0] = Avx2.GatherVector256(scaleTable, Avx2.And(sc, byteMask), 4);
                wd[1] = Avx2.GatherVector256(scaleTable, Avx2.And(Avx2.ShiftRightLogical(sc, 8), byteMask), 4);
                wd[2] = Avx2.GatherVector256(scaleTable, Avx2.And(Avx2.ShiftRightLogical(sc, 16), byteMask), 4);
                wd[3] = Avx2.GatherVector256(scaleTable, Avx2.ShiftRightLogical(sc, 24), 4);

                for (int j = 0; j < nt; j++)
                {
                    long t = t0 + j;
                    byte* y0 = xQ8 + t * q8RowBytes + 2 * ib * Q8_0BlockBytes;
                    float* dyT = dy + t * 2 * blockCount + 2 * ib;
                    int* cT = corr + t * 4 * blockCount + 4 * ib;
                    Vector256<float> dyA = Vector256.Create(dyT[0]);
                    Vector256<float> dyB = Vector256.Create(dyT[1]);

                    Vector256<float> accJ = acc[j];
                    accJ = Avx.Add(accJ, Nvfp4SubBlock(v, 0, y0 + 2, cT[0], wd[0], dyA, ones));
                    accJ = Avx.Add(accJ, Nvfp4SubBlock(v, 4, y0 + 18, cT[1], wd[1], dyA, ones));
                    accJ = Avx.Add(accJ, Nvfp4SubBlock(v, 8, y0 + Q8_0BlockBytes + 2, cT[2], wd[2], dyB, ones));
                    accJ = Avx.Add(accJ, Nvfp4SubBlock(v, 12, y0 + Q8_0BlockBytes + 18, cT[3], wd[3], dyB, ones));
                    acc[j] = accJ;
                }
            }

            for (int j = 0; j < nt; j++)
                Unsafe.WriteUnaligned(c + (long)(t0 + j) * mTotal, acc[j]);
        }
    }

    /// <summary>One sub-block of one token for eight rows: <c>(dy·d) · (float)(Σ u·q − 12·Σ q)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Nvfp4SubBlock(Vector256<byte>* v, int vi, byte* y, int corr, Vector256<float> d,
        Vector256<float> dy, Vector256<short> ones)
    {
        Vector256<int> iv = DotAccU8S8Avx2(Vector256.Create(corr), v[vi],
            Vector256.Create(Unsafe.ReadUnaligned<int>(y)).AsSByte(), ones);
        iv = DotAccU8S8Avx2(iv, v[vi + 1], Vector256.Create(Unsafe.ReadUnaligned<int>(y + 4)).AsSByte(), ones);
        iv = DotAccU8S8Avx2(iv, v[vi + 2], Vector256.Create(Unsafe.ReadUnaligned<int>(y + 8)).AsSByte(), ones);
        iv = DotAccU8S8Avx2(iv, v[vi + 3], Vector256.Create(Unsafe.ReadUnaligned<int>(y + 12)).AsSByte(), ones);
        return Avx.Multiply(Avx.Multiply(dy, d), Avx.ConvertToVector256Single(iv));
    }

    /// <summary>
    /// Unpacks one row's 32 <c>qs</c> bytes to unsigned codes (<c>kvalue + 12</c>). Output <paramref name="a"/> holds, per 128-bit
    /// lane, the 16 bytes <c>[low nibbles | high nibbles]</c> of sub-block 0 (lane 0) and 2 (lane 1); <paramref name="b"/> those of
    /// sub-blocks 1 and 3 — i.e. four dwords <c>[L0 L1 H0 H1]</c> = activation elements 0-3, 4-7, 8-11, 12-15.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackNvfp4Row(byte* qs, Vector256<byte> table, Vector256<byte> nib,
        out Vector256<long> a, out Vector256<long> b)
    {
        Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs);
        Vector256<long> lo = Avx2.Shuffle(table, Avx2.And(raw, nib)).AsInt64();
        Vector256<long> hi = Avx2.Shuffle(table, Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), nib)).AsInt64();
        a = Avx2.UnpackLow(lo, hi);
        b = Avx2.UnpackHigh(lo, hi);
    }

    /// <summary>
    /// Rows <c>[0, m)</c> of an NVFP4 GEMM against <paramref name="n"/> prepared tokens: eight-row groups through
    /// <see cref="ComputeRowsNvfp4Avx2_8Rows"/>, the remainder through the (identically exact) single-row dot.
    /// </summary>
    [SkipLocalsInit]
    internal static void ComputeRowsNvfp4Multi(byte* weights, byte* xQ8, int q8RowBytes, float* dy, int* corr,
        float* c, int mTotal, int m, int n, int blockCount, bool legacyOrder = false)
    {
        long rowBytes = (long)blockCount * Dequantize.Nvfp4BlockBytes;
        int row = 0;
        if (!legacyOrder && Nvfp4Rows8Enabled && rowBytes <= int.MaxValue / 8)
        {
            for (; row + 7 < m; row += 8)
                ComputeRowsNvfp4Avx2_8Rows(weights + row * rowBytes, (int)rowBytes, xQ8, q8RowBytes, dy, corr,
                    c + row, mTotal, n, blockCount);
        }
        for (; row < m; row++)
            for (int t = 0; t < n; t++)
                c[(long)t * mTotal + row] = VecDotNvfp4Q8_0(weights + row * rowBytes, xQ8 + (long)t * q8RowBytes, blockCount, legacyOrder);
    }
}
