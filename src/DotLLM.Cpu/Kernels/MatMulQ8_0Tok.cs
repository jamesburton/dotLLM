using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Issue #1119: Q8_0 R4 group x four tokens, so each weight block is loaded once for four tokens instead of
/// once per token. Every (row, token) output runs the exact operation sequence of
/// <see cref="MatMul.VecDotQ8_0Avx2_4RowsR4"/> (same accumulator lanes, same order, same dx*dw), so results are
/// bit-identical to the per-token path (#530).
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// 4 rows x 4 tokens over an R4 group: per block, each row's weights are loaded once for four tokens and the
    /// sixteen dx*dw scales come from two vector conversions. Writes <c>results[t * ldc + r]</c>; token rows of the
    /// Q8_0 input are <paramref name="xRowBytes"/> apart. The accumulators live in a stack array: on AVX2's sixteen
    /// registers that measured 1.5-1.6x the per-token path at n = 64 (8192 x 3072, 155H, 6 threads), while a 4x2
    /// split that fits the registers measured no gain.
    /// </summary>
    /// <param name="groupBase">The R4 group (4 rows' blocks interleaved).</param>
    /// <param name="blockCount">Blocks per row.</param>
    /// <param name="xq">The first token's Q8_0 row.</param>
    /// <param name="xRowBytes">Bytes between token rows.</param>
    /// <param name="results">Output for token 0, row 0.</param>
    /// <param name="ldc">Output stride between tokens.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Q8_0Rows4Tokens4(byte* groupBase, int blockCount, byte* xq, int xRowBytes, float* results, int ldc)
    {
        nint rowStride = (nint)blockCount * Q8_0BlockBytes;
        Vector256<float>* acc = stackalloc Vector256<float>[16];
        for (int i = 0; i < 16; i++) acc[i] = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        const int wStride = Q8_0BlockBytes; // #1156: original row-major rows, rowStride apart
        float* sc = stackalloc float[16];

        for (int block = 0; block < blockCount; block++)
        {
            byte* bb = groupBase + block * wStride;
            byte* xb = xq + block * Q8_0BlockBytes;
            Vector128<float> dw4 = HalfToSingleSse2(LoadHalf4(bb, rowStride));
            Vector128<float> dx4 = HalfToSingleSse2(LoadHalf4(xb, xRowBytes));
            Sse.Store(sc, Sse.Multiply(Sse.Shuffle(dx4, dx4, 0x00), dw4));
            Sse.Store(sc + 4, Sse.Multiply(Sse.Shuffle(dx4, dx4, 0x55), dw4));
            Sse.Store(sc + 8, Sse.Multiply(Sse.Shuffle(dx4, dx4, 0xAA), dw4));
            Sse.Store(sc + 12, Sse.Multiply(Sse.Shuffle(dx4, dx4, 0xFF), dw4));

            for (int t = 0; t < 4; t++)
            {
                byte* xbt = xb + (long)t * xRowBytes;
                Vector256<sbyte> v = Unsafe.ReadUnaligned<Vector256<sbyte>>(xbt + 2);
                Vector256<sbyte> ab = Avx2.Sign(v, v);
                acc[t * 4] = Q8_0BlockFma(acc[t * 4], bb, v, ab, sc[t * 4], ones);
                acc[t * 4 + 1] = Q8_0BlockFma(acc[t * 4 + 1], bb + rowStride, v, ab, sc[t * 4 + 1], ones);
                acc[t * 4 + 2] = Q8_0BlockFma(acc[t * 4 + 2], bb + 2 * rowStride, v, ab, sc[t * 4 + 2], ones);
                acc[t * 4 + 3] = Q8_0BlockFma(acc[t * 4 + 3], bb + 3 * rowStride, v, ab, sc[t * 4 + 3], ones);
            }
        }

        for (int t = 0; t < 4; t++)
            for (int r = 0; r < 4; r++)
                results[(long)t * ldc + r] = HorizontalSumAvx2Float(acc[t * 4 + r]);
    }
}
