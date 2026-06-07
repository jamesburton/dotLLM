using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace LoraQ8Stage1Probe;

/// <summary>
/// Candidate kernels for the LoRA stage 1 Q8_0 micro-bench.
/// Stage 1 computes <c>tmp[t, r] = sum_i x[t, i] * B[r, i]</c> for
/// <c>t ∈ [0, seqLen)</c>, <c>r ∈ [0, rank)</c>, <c>i ∈ [0, K)</c>.
/// All kernels here consume B in Q8_0 layout (<c>(K/32) * 34</c> bytes per row).
/// Activation <c>x</c> is consumed in two flavours:
///   1. F32 (legacy paths quantise on the fly — not measured here, see GemmQ8_0 path).
///   2. Pre-quantised Q8_0 (production path: base proj already quantises x once).
///
/// All kernels write a <c>[seqLen, rank]</c> F32 output buffer in row-major order.
/// </summary>
public static unsafe class Kernels
{
    // ─────────────────────────── Q8_0 layout constants ──────────────────────────
    // 32 elements per block, 34 bytes per block (2-byte F16 scale + 32 sbytes).
    public const int Q8_0Group = 32;
    public const int Q8_0Block = 34;

    // ─────────────────────────────────────────────────────────────────────────────
    //  PATH C — explicit Vector512<float> accumulator locals, rank-specialised.
    //  Mirrors MatMul.VecDotQ8_0Avx512_4Rows but extends to 16 named accumulators.
    //  Inner K-loop processes one block at a time (NOT dual-block) so the
    //  per-row weight load stride matches one block; this keeps register
    //  pressure to: 16 acc + 1 vx + 1 absX + working YMM + 1 scale ZMM ≈ 20 ZMM.
    // ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Path C, single-block flavour: process one K-block at a time. Each token
    /// holds 16 explicit Vector256&lt;int&gt; integer accumulators across the
    /// K-block loop, FMA'd into a Vector512&lt;float&gt; with the per-block
    /// scale. ConvertToVector256Single + Vector256.Sum or Vector256 horizontal
    /// add is used at the end.
    /// Actually we keep one Vector512&lt;float&gt; accumulator per row and
    /// FMA-add the scaled fsum256 (broadcast to 512) per block. Simpler.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Stage1_PathC_R16_Avx512(
        byte* xQ8, byte* bQ8, float* tmp,
        int seqLen, int blockCount, int rowBytes)
    {
        const int rank = 16;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int t = 0; t < seqLen; t++)
        {
            byte* xRow = xQ8 + (long)t * (blockCount * Q8_0Block);

            // 16 explicit accumulators — RyuJIT keeps these in ZMM regs.
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            Vector256<float> acc2 = Vector256<float>.Zero;
            Vector256<float> acc3 = Vector256<float>.Zero;
            Vector256<float> acc4 = Vector256<float>.Zero;
            Vector256<float> acc5 = Vector256<float>.Zero;
            Vector256<float> acc6 = Vector256<float>.Zero;
            Vector256<float> acc7 = Vector256<float>.Zero;
            Vector256<float> acc8 = Vector256<float>.Zero;
            Vector256<float> acc9 = Vector256<float>.Zero;
            Vector256<float> acc10 = Vector256<float>.Zero;
            Vector256<float> acc11 = Vector256<float>.Zero;
            Vector256<float> acc12 = Vector256<float>.Zero;
            Vector256<float> acc13 = Vector256<float>.Zero;
            Vector256<float> acc14 = Vector256<float>.Zero;
            Vector256<float> acc15 = Vector256<float>.Zero;

            for (int b = 0; b < blockCount; b++)
            {
                byte* xBlock = xRow + b * Q8_0Block;
                float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
                Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);
                Vector256<sbyte> absX = Avx2.Sign(vx, vx);

                // Row 0..15. Each is a single block dot, scaled, FMA'd into its acc.
                // Manual unroll — 16 sites — explicit so no rank-loop spill.
                acc0 = ProcessRow(bQ8 + 0L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc0);
                acc1 = ProcessRow(bQ8 + 1L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc1);
                acc2 = ProcessRow(bQ8 + 2L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc2);
                acc3 = ProcessRow(bQ8 + 3L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc3);
                acc4 = ProcessRow(bQ8 + 4L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc4);
                acc5 = ProcessRow(bQ8 + 5L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc5);
                acc6 = ProcessRow(bQ8 + 6L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc6);
                acc7 = ProcessRow(bQ8 + 7L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc7);
                acc8 = ProcessRow(bQ8 + 8L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc8);
                acc9 = ProcessRow(bQ8 + 9L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc9);
                acc10 = ProcessRow(bQ8 + 10L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc10);
                acc11 = ProcessRow(bQ8 + 11L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc11);
                acc12 = ProcessRow(bQ8 + 12L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc12);
                acc13 = ProcessRow(bQ8 + 13L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc13);
                acc14 = ProcessRow(bQ8 + 14L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc14);
                acc15 = ProcessRow(bQ8 + 15L * rowBytes + b * Q8_0Block, vx, absX, dx, ones, acc15);
            }

            float* outRow = tmp + (long)t * rank;
            outRow[0] = HSum(acc0);
            outRow[1] = HSum(acc1);
            outRow[2] = HSum(acc2);
            outRow[3] = HSum(acc3);
            outRow[4] = HSum(acc4);
            outRow[5] = HSum(acc5);
            outRow[6] = HSum(acc6);
            outRow[7] = HSum(acc7);
            outRow[8] = HSum(acc8);
            outRow[9] = HSum(acc9);
            outRow[10] = HSum(acc10);
            outRow[11] = HSum(acc11);
            outRow[12] = HSum(acc12);
            outRow[13] = HSum(acc13);
            outRow[14] = HSum(acc14);
            outRow[15] = HSum(acc15);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ProcessRow(
        byte* wBlock,
        Vector256<sbyte> vx, Vector256<sbyte> absX, float dx,
        Vector256<short> ones, Vector256<float> acc)
    {
        float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
        Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
        Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
        Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
        Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
        Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
        if (Fma.IsSupported)
            return Fma.MultiplyAdd(Vector256.Create(dx * dw), fsum, acc);
        return acc + Vector256.Create(dx * dw) * fsum;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  PATH C2 — same as C but with dual-block inner loop (matches the existing
    //  _4Rows kernel's dual-block strategy). Adds 2 x-loads per outer iter, but
    //  halves the loop trip count and packs two int32 sums into one Vector512<int>.
    //  This is the most direct extension of MatMul.VecDotQ8_0Avx512_4Rows to 16
    //  rows: 16 explicit Vector512<float> accumulators, each FMA'd with a dual
    //  scale (block i, block i+1).
    // ─────────────────────────────────────────────────────────────────────────────

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Stage1_PathC2_R16_Avx512Dual(
        byte* xQ8, byte* bQ8, float* tmp,
        int seqLen, int blockCount, int rowBytes)
    {
        const int rank = 16;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int t = 0; t < seqLen; t++)
        {
            byte* xRow = xQ8 + (long)t * (blockCount * Q8_0Block);

            Vector512<float> acc0 = Vector512<float>.Zero;
            Vector512<float> acc1 = Vector512<float>.Zero;
            Vector512<float> acc2 = Vector512<float>.Zero;
            Vector512<float> acc3 = Vector512<float>.Zero;
            Vector512<float> acc4 = Vector512<float>.Zero;
            Vector512<float> acc5 = Vector512<float>.Zero;
            Vector512<float> acc6 = Vector512<float>.Zero;
            Vector512<float> acc7 = Vector512<float>.Zero;
            Vector512<float> acc8 = Vector512<float>.Zero;
            Vector512<float> acc9 = Vector512<float>.Zero;
            Vector512<float> acc10 = Vector512<float>.Zero;
            Vector512<float> acc11 = Vector512<float>.Zero;
            Vector512<float> acc12 = Vector512<float>.Zero;
            Vector512<float> acc13 = Vector512<float>.Zero;
            Vector512<float> acc14 = Vector512<float>.Zero;
            Vector512<float> acc15 = Vector512<float>.Zero;

            int b = 0;
            for (; b + 1 < blockCount; b += 2)
            {
                byte* xBlock0 = xRow + b * Q8_0Block;
                byte* xBlock1 = xRow + (b + 1) * Q8_0Block;
                float dx0 = (float)Unsafe.ReadUnaligned<Half>(xBlock0);
                float dx1 = (float)Unsafe.ReadUnaligned<Half>(xBlock1);
                Vector256<sbyte> vx0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock0 + 2);
                Vector256<sbyte> vx1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock1 + 2);
                Vector256<sbyte> absX0 = Avx2.Sign(vx0, vx0);
                Vector256<sbyte> absX1 = Avx2.Sign(vx1, vx1);

                acc0 = DualBlock(bQ8 + 0L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc0);
                acc1 = DualBlock(bQ8 + 1L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc1);
                acc2 = DualBlock(bQ8 + 2L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc2);
                acc3 = DualBlock(bQ8 + 3L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc3);
                acc4 = DualBlock(bQ8 + 4L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc4);
                acc5 = DualBlock(bQ8 + 5L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc5);
                acc6 = DualBlock(bQ8 + 6L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc6);
                acc7 = DualBlock(bQ8 + 7L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc7);
                acc8 = DualBlock(bQ8 + 8L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc8);
                acc9 = DualBlock(bQ8 + 9L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc9);
                acc10 = DualBlock(bQ8 + 10L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc10);
                acc11 = DualBlock(bQ8 + 11L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc11);
                acc12 = DualBlock(bQ8 + 12L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc12);
                acc13 = DualBlock(bQ8 + 13L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc13);
                acc14 = DualBlock(bQ8 + 14L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc14);
                acc15 = DualBlock(bQ8 + 15L * rowBytes + b * Q8_0Block, vx0, vx1, absX0, absX1, dx0, dx1, ones, acc15);
            }

            float* outRow = tmp + (long)t * rank;
            outRow[0] = HSum(acc0);
            outRow[1] = HSum(acc1);
            outRow[2] = HSum(acc2);
            outRow[3] = HSum(acc3);
            outRow[4] = HSum(acc4);
            outRow[5] = HSum(acc5);
            outRow[6] = HSum(acc6);
            outRow[7] = HSum(acc7);
            outRow[8] = HSum(acc8);
            outRow[9] = HSum(acc9);
            outRow[10] = HSum(acc10);
            outRow[11] = HSum(acc11);
            outRow[12] = HSum(acc12);
            outRow[13] = HSum(acc13);
            outRow[14] = HSum(acc14);
            outRow[15] = HSum(acc15);

            // Trailing odd block (none for K=2048 / blockCount=64, but keep for safety).
            for (; b < blockCount; b++)
            {
                byte* xBlock = xRow + b * Q8_0Block;
                float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
                Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);
                Vector256<sbyte> absX = Avx2.Sign(vx, vx);

                for (int r = 0; r < rank; r++)
                {
                    byte* wBlock = bQ8 + (long)r * rowBytes + b * Q8_0Block;
                    float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                    Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                    Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
                    Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
                    Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
                    Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
                    outRow[r] += dx * dw * HSum256(fsum);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> DualBlock(
        byte* wBlock0Base,
        Vector256<sbyte> vx0, Vector256<sbyte> vx1,
        Vector256<sbyte> absX0, Vector256<sbyte> absX1,
        float dx0, float dx1,
        Vector256<short> ones,
        Vector512<float> acc)
    {
        byte* wBlock1 = wBlock0Base + Q8_0Block;
        float dw0 = (float)Unsafe.ReadUnaligned<Half>(wBlock0Base);
        float dw1 = (float)Unsafe.ReadUnaligned<Half>(wBlock1);

        Vector256<sbyte> vw0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock0Base + 2);
        Vector256<sbyte> vw1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock1 + 2);

        Vector256<sbyte> adjW0 = Avx2.Sign(vw0, vx0);
        Vector256<sbyte> adjW1 = Avx2.Sign(vw1, vx1);

        Vector256<short> prod0 = Avx2.MultiplyAddAdjacent(absX0.AsByte(), adjW0);
        Vector256<int> isum0 = Avx2.MultiplyAddAdjacent(prod0, ones);
        Vector256<short> prod1 = Avx2.MultiplyAddAdjacent(absX1.AsByte(), adjW1);
        Vector256<int> isum1 = Avx2.MultiplyAddAdjacent(prod1, ones);

        Vector512<int> isum512 = Vector512.Create(isum0, isum1);
        Vector512<float> fsum512 = Avx512F.ConvertToVector512Single(isum512);

        Vector512<float> scale = Vector512.Create(
            Vector256.Create(dx0 * dw0),
            Vector256.Create(dx1 * dw1));

        return Avx512F.FusedMultiplyAdd(fsum512, scale, acc);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  PATH B — R16 interleaved B layout. Repacked at adapter-load time so all
    //  16 rows for a single K-block are stored contiguously:
    //    [r0_b0][r1_b0]...[r15_b0][r0_b1][r1_b1]...[r15_b1]...
    //  Each "K-step chunk" is 16 × 34 = 544 bytes (8.5 cache lines, exactly
    //  contiguous). This is the strict R-extension of WeightRepacking.RepackR4.
    //  Stage 1 reads sequentially instead of striding by rowBytes.
    // ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Repack <paramref name="bRowMajor"/> from <c>[rank, blockCount * 34]</c>
    /// row-major Q8_0 into the same R4 layout the base-model GEMM uses (see
    /// <c>WeightRepacking.RepackR4</c> in the production code), so the existing
    /// <c>MatMul.OuterProductGemmQ8_0</c> kernel can be invoked directly on a
    /// LoRA-B factor without rewriting any SIMD. Layout: per fullGroups of 4
    /// rows, per K-block, 4 consecutive Q8_0 blocks (one per row in the group).
    /// Tail rows (<c>rank % 4</c>) are dropped here — the probe focuses on
    /// <c>rank=16</c> which is a clean multiple. Productionisation
    /// (<c>WeightRepacking.RepackR4_Q8_0</c>) will handle tail rows.
    /// </summary>
    public static byte* RepackR4(byte* bRowMajor, int rank, int blockCount, int rowBytes)
    {
        int fullGroups = rank / 4;
        long totalBytes = (long)fullGroups * 4 * blockCount * Q8_0Block;
        byte* dst = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        int groupBytes = 4 * blockCount * Q8_0Block;
        for (int g = 0; g < fullGroups; g++)
        {
            byte* groupDst = dst + (long)g * groupBytes;
            for (int b = 0; b < blockCount; b++)
                for (int r = 0; r < 4; r++)
                {
                    byte* src = bRowMajor + (long)(g * 4 + r) * rowBytes + b * Q8_0Block;
                    byte* d = groupDst + b * 4 * Q8_0Block + r * Q8_0Block;
                    Buffer.MemoryCopy(src, d, Q8_0Block, Q8_0Block);
                }
        }
        return dst;
    }

    /// <summary>
    /// Repack <paramref name="bRowMajor"/> from <c>[rank=16, K_blocks * 34]</c>
    /// row-major Q8_0 into <c>[K_blocks, 16 * 34]</c> R16 interleaved.
    /// Allocates the destination via NativeMemory.AlignedAlloc(_, 64); caller
    /// frees with NativeMemory.AlignedFree.
    /// </summary>
    public static byte* RepackR16(byte* bRowMajor, int rank, int blockCount, int rowBytes)
    {
        // Layout: per K-block, rank Q8_0 blocks contiguously.
        //   chunk(b)[r] = source(r, b)  (1 Q8_0 block of 34 bytes)
        long totalBytes = (long)rank * blockCount * Q8_0Block;
        byte* dst = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)totalBytes, 64);

        for (int b = 0; b < blockCount; b++)
        {
            byte* chunkDst = dst + (long)b * rank * Q8_0Block;
            for (int r = 0; r < rank; r++)
            {
                byte* srcBlock = bRowMajor + (long)r * rowBytes + b * Q8_0Block;
                byte* dstBlock = chunkDst + r * Q8_0Block;
                Buffer.MemoryCopy(srcBlock, dstBlock, Q8_0Block, Q8_0Block);
            }
        }
        return dst;
    }

    /// <summary>
    /// Path B + C combined: explicit-locals 16-row kernel consuming R16-interleaved B.
    /// Inner loop: per K-block, the 16 weight blocks are 544 contiguous bytes —
    /// sequential reads with HW-prefetch friendly stride.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Stage1_PathBC_R16Interleaved_Avx512(
        byte* xQ8, byte* bR16, float* tmp,
        int seqLen, int blockCount)
    {
        const int rank = 16;
        Vector256<short> ones = Vector256.Create((short)1);
        int chunkBytes = rank * Q8_0Block; // 544

        for (int t = 0; t < seqLen; t++)
        {
            byte* xRow = xQ8 + (long)t * (blockCount * Q8_0Block);

            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            Vector256<float> acc2 = Vector256<float>.Zero;
            Vector256<float> acc3 = Vector256<float>.Zero;
            Vector256<float> acc4 = Vector256<float>.Zero;
            Vector256<float> acc5 = Vector256<float>.Zero;
            Vector256<float> acc6 = Vector256<float>.Zero;
            Vector256<float> acc7 = Vector256<float>.Zero;
            Vector256<float> acc8 = Vector256<float>.Zero;
            Vector256<float> acc9 = Vector256<float>.Zero;
            Vector256<float> acc10 = Vector256<float>.Zero;
            Vector256<float> acc11 = Vector256<float>.Zero;
            Vector256<float> acc12 = Vector256<float>.Zero;
            Vector256<float> acc13 = Vector256<float>.Zero;
            Vector256<float> acc14 = Vector256<float>.Zero;
            Vector256<float> acc15 = Vector256<float>.Zero;

            for (int b = 0; b < blockCount; b++)
            {
                byte* xBlock = xRow + b * Q8_0Block;
                float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
                Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);
                Vector256<sbyte> absX = Avx2.Sign(vx, vx);

                byte* chunk = bR16 + (long)b * chunkBytes;
                acc0 = ProcessRow(chunk + 0 * Q8_0Block, vx, absX, dx, ones, acc0);
                acc1 = ProcessRow(chunk + 1 * Q8_0Block, vx, absX, dx, ones, acc1);
                acc2 = ProcessRow(chunk + 2 * Q8_0Block, vx, absX, dx, ones, acc2);
                acc3 = ProcessRow(chunk + 3 * Q8_0Block, vx, absX, dx, ones, acc3);
                acc4 = ProcessRow(chunk + 4 * Q8_0Block, vx, absX, dx, ones, acc4);
                acc5 = ProcessRow(chunk + 5 * Q8_0Block, vx, absX, dx, ones, acc5);
                acc6 = ProcessRow(chunk + 6 * Q8_0Block, vx, absX, dx, ones, acc6);
                acc7 = ProcessRow(chunk + 7 * Q8_0Block, vx, absX, dx, ones, acc7);
                acc8 = ProcessRow(chunk + 8 * Q8_0Block, vx, absX, dx, ones, acc8);
                acc9 = ProcessRow(chunk + 9 * Q8_0Block, vx, absX, dx, ones, acc9);
                acc10 = ProcessRow(chunk + 10 * Q8_0Block, vx, absX, dx, ones, acc10);
                acc11 = ProcessRow(chunk + 11 * Q8_0Block, vx, absX, dx, ones, acc11);
                acc12 = ProcessRow(chunk + 12 * Q8_0Block, vx, absX, dx, ones, acc12);
                acc13 = ProcessRow(chunk + 13 * Q8_0Block, vx, absX, dx, ones, acc13);
                acc14 = ProcessRow(chunk + 14 * Q8_0Block, vx, absX, dx, ones, acc14);
                acc15 = ProcessRow(chunk + 15 * Q8_0Block, vx, absX, dx, ones, acc15);
            }

            float* outRow = tmp + (long)t * rank;
            outRow[0] = HSum(acc0);
            outRow[1] = HSum(acc1);
            outRow[2] = HSum(acc2);
            outRow[3] = HSum(acc3);
            outRow[4] = HSum(acc4);
            outRow[5] = HSum(acc5);
            outRow[6] = HSum(acc6);
            outRow[7] = HSum(acc7);
            outRow[8] = HSum(acc8);
            outRow[9] = HSum(acc9);
            outRow[10] = HSum(acc10);
            outRow[11] = HSum(acc11);
            outRow[12] = HSum(acc12);
            outRow[13] = HSum(acc13);
            outRow[14] = HSum(acc14);
            outRow[15] = HSum(acc15);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  Helpers
    // ─────────────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum(Vector256<float> v)
    {
        // 8-lane horizontal sum.
        Vector128<float> lo = v.GetLower();
        Vector128<float> hi = v.GetUpper();
        Vector128<float> s = Sse.Add(lo, hi);
        s = Sse.Add(s, Sse.Shuffle(s, s, 0b_10_11_00_01));
        s = Sse.Add(s, Sse.Shuffle(s, s, 0b_01_00_11_10));
        return s.GetElement(0);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum256(Vector256<float> v) => HSum(v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum(Vector512<float> v)
    {
        Vector256<float> lo = v.GetLower();
        Vector256<float> hi = v.GetUpper();
        return HSum(Avx.Add(lo, hi));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  STAGE 2 — outer-product fused stage 2.
    //
    //  Standard form: y[t, o] += scale * sum_r A[o, r] * tmp[t, r].
    //  Production loops `for each token, GemvF32(A, tmp_t, delta) ; y += scale*delta`
    //  which produces ~2048 × 512 = 1M short Dot calls — function-entry dominated
    //  even though each Dot is just 16 elements.
    //
    //  Outer-product fusion: for each token, tmp[t, 0..15] occupies exactly one
    //  Vector512<float>. For each output-tile of 16 floats:
    //      acc_tile = 0
    //      for r in 0..15:
    //          a_r_tile = A[o-tile..o-tile+15, r]   // requires column-stride of A[*, r]
    //          acc_tile = FMA(broadcast(tmp_t[r]), a_r_tile, acc_tile)
    //      y[t, o-tile..] += scale * acc_tile
    //
    //  Two layout flavours:
    //    (a) A row-major [outputDim, rank] — column-r access strides by rank=16.
    //        That's a strided load — bad. Need to load 16 separate 4-byte values
    //        and pack them into a Vector512. Slow.
    //    (b) A transposed [rank, outputDim] (call it AT) — column-r becomes row-r,
    //        sequential outputDim-long Vector512 loads. Fast.
    //
    //  Path E1: in-place transpose A on adapter load → standard outer product.
    //  Path E2: keep A row-major, broadcast each tmp_t[r] and scatter via gather.
    //  Path E3: keep A row-major; compute `delta = A · tmp_t` as M=outputDim,
    //           K=rank=16 GEMV but with the K=16 "broadcast x → FMA into 16
    //           row accumulators per output tile" pattern. This requires writing
    //           a custom GEMV that exploits K=16 fitting in a Vector512.
    // ─────────────────────────────────────────────────────────────────────────────

    // ─────────────────────────────────────────────────────────────────────────────
    //  PATH F — fully fused F32 stage-1 + outer-product stage-2.
    //  Per token: (1) compute tmp_t[0..15] (stage 1) using F32 dequanted B,
    //  hold the 16 results as 16 broadcast Vector512<float> registers; then
    //  (2) immediately consume them in the outer-product stage-2 loop
    //  against transposed A. No materialised [N, rank] tmp buffer.
    //
    //  Stage 1 is itself an outer-product / row-broadcast pattern: for each
    //  K-tile of 16 floats, we load x[t, k..k+15] as one ZMM, then for each
    //  of 16 B-rows accumulate FMA(B_row[k..], x[k..], acc_r). 16 row
    //  accumulators (Vector512) live across the inner K-loop. At end of
    //  K-loop: hsum each acc → 16 scalars → 16 Vector512 broadcasts for
    //  stage 2.
    // ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Path F — fused F32 stage-1 + outer-product stage-2. B is F32
    /// (dequant-once into the ArrayPool scratch happens upstream just like
    /// the production F32 path). AT is the transposed A from
    /// <see cref="TransposeA_OutputDimByRank"/>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Stage12_Fused_F32_R16_Avx512(
        float* x, float* bF32, float* aTransposed, float* y,
        int seqLen, int K, int outputDim, float scale)
    {
        const int rank = 16;
        Vector512<float> scaleVec = Vector512.Create(scale);

        for (int t = 0; t < seqLen; t++)
        {
            float* xRow = x + (long)t * K;

            // ── Stage 1 inner: 16 row accumulators (one per LoRA-B row), inner
            // loop over K in tiles of 16. ──────────────────────────────────────
            Vector512<float> a0 = Vector512<float>.Zero;
            Vector512<float> a1 = Vector512<float>.Zero;
            Vector512<float> a2 = Vector512<float>.Zero;
            Vector512<float> a3 = Vector512<float>.Zero;
            Vector512<float> a4 = Vector512<float>.Zero;
            Vector512<float> a5 = Vector512<float>.Zero;
            Vector512<float> a6 = Vector512<float>.Zero;
            Vector512<float> a7 = Vector512<float>.Zero;
            Vector512<float> a8 = Vector512<float>.Zero;
            Vector512<float> a9 = Vector512<float>.Zero;
            Vector512<float> a10 = Vector512<float>.Zero;
            Vector512<float> a11 = Vector512<float>.Zero;
            Vector512<float> a12 = Vector512<float>.Zero;
            Vector512<float> a13 = Vector512<float>.Zero;
            Vector512<float> a14 = Vector512<float>.Zero;
            Vector512<float> a15 = Vector512<float>.Zero;

            int k = 0;
            for (; k + 16 <= K; k += 16)
            {
                Vector512<float> xv = Avx512F.LoadVector512(xRow + k);
                a0 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 0 * K + k), xv, a0);
                a1 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 1 * K + k), xv, a1);
                a2 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 2 * K + k), xv, a2);
                a3 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 3 * K + k), xv, a3);
                a4 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 4 * K + k), xv, a4);
                a5 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 5 * K + k), xv, a5);
                a6 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 6 * K + k), xv, a6);
                a7 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 7 * K + k), xv, a7);
                a8 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 8 * K + k), xv, a8);
                a9 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 9 * K + k), xv, a9);
                a10 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 10 * K + k), xv, a10);
                a11 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 11 * K + k), xv, a11);
                a12 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 12 * K + k), xv, a12);
                a13 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 13 * K + k), xv, a13);
                a14 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 14 * K + k), xv, a14);
                a15 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(bF32 + 15 * K + k), xv, a15);
            }
            // Tail (K % 16 != 0) — scalar fallback. CA2014 silence: K is
            // typically a multiple of 16 (transformer hidden dims are powers
            // of 2 ≥ 32), so this branch is unreachable on production shapes.
            if (k < K)
            {
#pragma warning disable CA2014  // stackalloc-in-loop is bounded by rank=16 and only on the K%16!=0 tail
                Span<float> tmpScalars = stackalloc float[rank];
#pragma warning restore CA2014
                tmpScalars[0] = HSum(a0); tmpScalars[1] = HSum(a1);
                tmpScalars[2] = HSum(a2); tmpScalars[3] = HSum(a3);
                tmpScalars[4] = HSum(a4); tmpScalars[5] = HSum(a5);
                tmpScalars[6] = HSum(a6); tmpScalars[7] = HSum(a7);
                tmpScalars[8] = HSum(a8); tmpScalars[9] = HSum(a9);
                tmpScalars[10] = HSum(a10); tmpScalars[11] = HSum(a11);
                tmpScalars[12] = HSum(a12); tmpScalars[13] = HSum(a13);
                tmpScalars[14] = HSum(a14); tmpScalars[15] = HSum(a15);
                for (; k < K; k++)
                    for (int r = 0; r < rank; r++)
                        tmpScalars[r] += bF32[r * K + k] * xRow[k];
                Stage2OneToken(aTransposed, tmpScalars, y + (long)t * outputDim, outputDim, scaleVec);
                continue;
            }

            // ── Stage 2 inline: hsum 16 row-accs → 16 broadcasts → outer-product
            // over outputDim tiles. Reuse the same ZMM register pool — by the
            // time we reach the inner stage-2 loop, the row accs have been
            // hsummed and we only need 16 broadcasts live + 1 acc + 1 weight
            // load + working = ~19 ZMM live, OK on AVX-512 (32 ZMM). ─────────
            float t0 = HSum(a0), t1 = HSum(a1), t2 = HSum(a2), t3 = HSum(a3);
            float t4 = HSum(a4), t5 = HSum(a5), t6 = HSum(a6), t7 = HSum(a7);
            float t8 = HSum(a8), t9 = HSum(a9), t10 = HSum(a10), t11 = HSum(a11);
            float t12 = HSum(a12), t13 = HSum(a13), t14 = HSum(a14), t15 = HSum(a15);

            Vector512<float> b0 = Vector512.Create(t0);
            Vector512<float> b1 = Vector512.Create(t1);
            Vector512<float> b2 = Vector512.Create(t2);
            Vector512<float> b3 = Vector512.Create(t3);
            Vector512<float> b4 = Vector512.Create(t4);
            Vector512<float> b5 = Vector512.Create(t5);
            Vector512<float> b6 = Vector512.Create(t6);
            Vector512<float> b7 = Vector512.Create(t7);
            Vector512<float> b8 = Vector512.Create(t8);
            Vector512<float> b9 = Vector512.Create(t9);
            Vector512<float> b10 = Vector512.Create(t10);
            Vector512<float> b11 = Vector512.Create(t11);
            Vector512<float> b12 = Vector512.Create(t12);
            Vector512<float> b13 = Vector512.Create(t13);
            Vector512<float> b14 = Vector512.Create(t14);
            Vector512<float> b15 = Vector512.Create(t15);

            float* yRow = y + (long)t * outputDim;
            int o = 0;
            for (; o + 16 <= outputDim; o += 16)
            {
                Vector512<float> acc = Avx512F.LoadVector512(aTransposed + 0 * outputDim + o) * b0;
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 1 * outputDim + o), b1, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 2 * outputDim + o), b2, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 3 * outputDim + o), b3, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 4 * outputDim + o), b4, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 5 * outputDim + o), b5, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 6 * outputDim + o), b6, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 7 * outputDim + o), b7, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 8 * outputDim + o), b8, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 9 * outputDim + o), b9, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 10 * outputDim + o), b10, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 11 * outputDim + o), b11, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 12 * outputDim + o), b12, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 13 * outputDim + o), b13, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 14 * outputDim + o), b14, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 15 * outputDim + o), b15, acc);

                Vector512<float> yVec = Avx512F.LoadVector512(yRow + o);
                yVec = Avx512F.FusedMultiplyAdd(scaleVec, acc, yVec);
                Avx512F.Store(yRow + o, yVec);
            }
            // outputDim tail (rare — outputDim usually multiple of 16).
            for (; o < outputDim; o++)
            {
                float s = t0 * aTransposed[0L * outputDim + o]
                        + t1 * aTransposed[1L * outputDim + o]
                        + t2 * aTransposed[2L * outputDim + o]
                        + t3 * aTransposed[3L * outputDim + o]
                        + t4 * aTransposed[4L * outputDim + o]
                        + t5 * aTransposed[5L * outputDim + o]
                        + t6 * aTransposed[6L * outputDim + o]
                        + t7 * aTransposed[7L * outputDim + o]
                        + t8 * aTransposed[8L * outputDim + o]
                        + t9 * aTransposed[9L * outputDim + o]
                        + t10 * aTransposed[10L * outputDim + o]
                        + t11 * aTransposed[11L * outputDim + o]
                        + t12 * aTransposed[12L * outputDim + o]
                        + t13 * aTransposed[13L * outputDim + o]
                        + t14 * aTransposed[14L * outputDim + o]
                        + t15 * aTransposed[15L * outputDim + o];
                yRow[o] += scale * s;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Stage2OneToken(
        float* aTransposed, ReadOnlySpan<float> tmpScalars, float* yRow,
        int outputDim, Vector512<float> scaleVec)
    {
        const int rank = 16;
        Vector512<float> b0 = Vector512.Create(tmpScalars[0]);
        Vector512<float> b1 = Vector512.Create(tmpScalars[1]);
        Vector512<float> b2 = Vector512.Create(tmpScalars[2]);
        Vector512<float> b3 = Vector512.Create(tmpScalars[3]);
        Vector512<float> b4 = Vector512.Create(tmpScalars[4]);
        Vector512<float> b5 = Vector512.Create(tmpScalars[5]);
        Vector512<float> b6 = Vector512.Create(tmpScalars[6]);
        Vector512<float> b7 = Vector512.Create(tmpScalars[7]);
        Vector512<float> b8 = Vector512.Create(tmpScalars[8]);
        Vector512<float> b9 = Vector512.Create(tmpScalars[9]);
        Vector512<float> b10 = Vector512.Create(tmpScalars[10]);
        Vector512<float> b11 = Vector512.Create(tmpScalars[11]);
        Vector512<float> b12 = Vector512.Create(tmpScalars[12]);
        Vector512<float> b13 = Vector512.Create(tmpScalars[13]);
        Vector512<float> b14 = Vector512.Create(tmpScalars[14]);
        Vector512<float> b15 = Vector512.Create(tmpScalars[15]);

        int o = 0;
        for (; o + 16 <= outputDim; o += 16)
        {
            Vector512<float> acc = Avx512F.LoadVector512(aTransposed + 0 * outputDim + o) * b0;
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 1 * outputDim + o), b1, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 2 * outputDim + o), b2, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 3 * outputDim + o), b3, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 4 * outputDim + o), b4, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 5 * outputDim + o), b5, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 6 * outputDim + o), b6, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 7 * outputDim + o), b7, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 8 * outputDim + o), b8, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 9 * outputDim + o), b9, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 10 * outputDim + o), b10, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 11 * outputDim + o), b11, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 12 * outputDim + o), b12, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 13 * outputDim + o), b13, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 14 * outputDim + o), b14, acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 15 * outputDim + o), b15, acc);

            Vector512<float> yVec = Avx512F.LoadVector512(yRow + o);
            yVec = Avx512F.FusedMultiplyAdd(scaleVec, acc, yVec);
            Avx512F.Store(yRow + o, yVec);
        }
        for (; o < outputDim; o++)
        {
            float s = 0;
            for (int r = 0; r < rank; r++) s += aTransposed[r * outputDim + o] * tmpScalars[r];
            yRow[o] += scaleVec.GetElement(0) * s;
        }
    }

    /// <summary>
    /// Repack A from [outputDim, rank=16] row-major into [rank=16, outputDim]
    /// row-major (transpose). Allocates 64-byte aligned destination.
    /// </summary>
    public static float* TransposeA_OutputDimByRank(float* aRowMajor, int outputDim, int rank)
    {
        long total = (long)rank * outputDim;
        float* dst = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
            (nuint)(total * sizeof(float)), 64);
        for (int r = 0; r < rank; r++)
            for (int o = 0; o < outputDim; o++)
                dst[(long)r * outputDim + o] = aRowMajor[(long)o * rank + r];
        return dst;
    }

    /// <summary>
    /// Path E1 stage-2: outer-product fused stage 2 with transposed A.
    /// AT is <c>[rank=16, outputDim]</c>, tmp is <c>[N, rank=16]</c>, y is
    /// <c>[N, outputDim]</c>. Per token: load tmp_t (1 ZMM), then for each
    /// output-tile of 16 floats, FMA over 16 ranks broadcasting from tmp_t.
    /// Writes <c>y[t, :] += scale * (AT^T · tmp_t)</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Stage2_OuterProduct_R16_Avx512(
        float* aTransposed, float* tmp, float* y,
        int seqLen, int outputDim, float scale)
    {
        const int rank = 16;
        Vector512<float> scaleVec = Vector512.Create(scale);

        for (int t = 0; t < seqLen; t++)
        {
            // tmp[t, 0..15] — 16 floats, exactly one ZMM register.
            float* tmpRow = tmp + (long)t * rank;

            // We'll broadcast tmp_t[r] into a ZMM for each r. Pre-broadcast all 16
            // values once per token so the inner output-tile loop only does FMA
            // + load. With 16 broadcasts + 1 acc + 1 weight load + working = 18
            // ZMM live, fits comfortably.
            Vector512<float> b0 = Vector512.Create(tmpRow[0]);
            Vector512<float> b1 = Vector512.Create(tmpRow[1]);
            Vector512<float> b2 = Vector512.Create(tmpRow[2]);
            Vector512<float> b3 = Vector512.Create(tmpRow[3]);
            Vector512<float> b4 = Vector512.Create(tmpRow[4]);
            Vector512<float> b5 = Vector512.Create(tmpRow[5]);
            Vector512<float> b6 = Vector512.Create(tmpRow[6]);
            Vector512<float> b7 = Vector512.Create(tmpRow[7]);
            Vector512<float> b8 = Vector512.Create(tmpRow[8]);
            Vector512<float> b9 = Vector512.Create(tmpRow[9]);
            Vector512<float> b10 = Vector512.Create(tmpRow[10]);
            Vector512<float> b11 = Vector512.Create(tmpRow[11]);
            Vector512<float> b12 = Vector512.Create(tmpRow[12]);
            Vector512<float> b13 = Vector512.Create(tmpRow[13]);
            Vector512<float> b14 = Vector512.Create(tmpRow[14]);
            Vector512<float> b15 = Vector512.Create(tmpRow[15]);

            float* yRow = y + (long)t * outputDim;
            int o = 0;
            for (; o + 16 <= outputDim; o += 16)
            {
                // 4 independent accumulators to break the 16-deep FMA chain.
                Vector512<float> a0 = Avx512F.LoadVector512(aTransposed + 0 * outputDim + o) * b0;
                Vector512<float> a1 = Avx512F.LoadVector512(aTransposed + 1 * outputDim + o) * b1;
                Vector512<float> a2 = Avx512F.LoadVector512(aTransposed + 2 * outputDim + o) * b2;
                Vector512<float> a3 = Avx512F.LoadVector512(aTransposed + 3 * outputDim + o) * b3;
                a0 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 4 * outputDim + o), b4, a0);
                a1 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 5 * outputDim + o), b5, a1);
                a2 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 6 * outputDim + o), b6, a2);
                a3 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 7 * outputDim + o), b7, a3);
                a0 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 8 * outputDim + o), b8, a0);
                a1 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 9 * outputDim + o), b9, a1);
                a2 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 10 * outputDim + o), b10, a2);
                a3 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 11 * outputDim + o), b11, a3);
                a0 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 12 * outputDim + o), b12, a0);
                a1 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 13 * outputDim + o), b13, a1);
                a2 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 14 * outputDim + o), b14, a2);
                a3 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 15 * outputDim + o), b15, a3);
                Vector512<float> acc = (a0 + a1) + (a2 + a3);

                // y[t, o..o+15] += scale * acc.
                Vector512<float> yVec = Avx512F.LoadVector512(yRow + o);
                yVec = Avx512F.FusedMultiplyAdd(scaleVec, acc, yVec);
                Avx512F.Store(yRow + o, yVec);
            }
            // Tail (outputDim % 16 != 0) — scalar fallback.
            for (; o < outputDim; o++)
            {
                float s = 0;
                for (int r = 0; r < rank; r++)
                    s += aTransposed[(long)r * outputDim + o] * tmpRow[r];
                yRow[o] += scale * s;
            }
        }
    }
}
