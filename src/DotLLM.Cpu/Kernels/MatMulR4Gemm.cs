using System.Buffers;
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Multi-token GEMM over R4-interleaved (repacked) weights, for every quantization that has a
/// repacked layout — not just Q8_0.
/// </summary>
/// <remarks>
/// <para>
/// Issue #530: <c>GemmInterleaved</c> used to dispatch on batch size — <c>n == 1</c> went through
/// the repacked <c>ComputeRows*Interleaved</c> kernels while <c>n &gt; 1</c> fell through to the
/// original, row-major weights for every non-Q8_0 quant. Two implementations of the same
/// mathematics selected by row count, so a token's logits depended on how many tokens shared its
/// forward pass.
/// </para>
/// <para>
/// This file generalises <see cref="MatMul.GemmR4TiledQ8_0(byte*, byte*, float*, int, int, int, int, int, ComputeThreadPool?)"/>
/// to any repackable quant: tile over 4-row groups (the R4 interleave unit), iterate all tokens
/// inside the tile so a tile's weights stream once and are reused across the batch. Every output
/// element is produced by the same <c>ComputeRows*Interleaved</c> call the <c>n == 1</c> arm uses,
/// so results are bit-identical across batch sizes by construction.
/// </para>
/// </remarks>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Tiled multi-token GEMM over R4-interleaved weights. C is [N x M] row-major
    /// (<c>C[token * m + row]</c>).
    /// </summary>
    /// <param name="repackedWeights">R4-interleaved weight data.</param>
    /// <param name="b">F32 input [N x K]; used only when <paramref name="inputQ"/> is null.</param>
    /// <param name="inputQ">Pre-quantized input [N x inputRowBytes], or null to self-quantize.</param>
    /// <param name="c">Output [N x M], row-major.</param>
    /// <param name="fullGroups">Complete 4-row groups (M / 4).</param>
    /// <param name="tailRows">Leftover rows (M % 4), stored row-major after the interleaved data.</param>
    /// <param name="blockCount">Blocks (or super-blocks) per weight row.</param>
    /// <param name="m">Total output rows.</param>
    /// <param name="k">Input dimension.</param>
    /// <param name="n">Token count.</param>
    /// <param name="blockBytes">Bytes per weight block for this quantization.</param>
    /// <param name="inputRowBytes">Bytes per quantized input row.</param>
    /// <param name="computeRowsInterleaved">The R4 ComputeRows kernel for this quantization.</param>
    /// <param name="quantizeRow">Quantizer used when <paramref name="inputQ"/> is null.</param>
    /// <param name="pool">Thread pool, or null for single-threaded.</param>
    [SkipLocalsInit]
    internal static void GemmR4Tiled(
        byte* repackedWeights, float* b, byte* inputQ, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int k, int n,
        int blockBytes, int inputRowBytes,
        delegate*<byte*, byte*, float*, int, int, int, void> computeRowsInterleaved,
        delegate*<float*, byte*, int, void> quantizeRow,
        ComputeThreadPool? pool)
    {
        if (inputQ != null)
        {
            GemmR4TiledCore(repackedWeights, inputQ, c, fullGroups, tailRows, blockCount,
                m, n, blockBytes, inputRowBytes, computeRowsInterleaved, pool);
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(n * inputRowBytes);
        try
        {
            fixed (byte* scratch = rented)
            {
                for (int t = 0; t < n; t++)
                    quantizeRow(b + (long)t * k, scratch + (long)t * inputRowBytes, k);

                GemmR4TiledCore(repackedWeights, scratch, c, fullGroups, tailRows, blockCount,
                    m, n, blockBytes, inputRowBytes, computeRowsInterleaved, pool);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static void GemmR4TiledCore(
        byte* repackedWeights, byte* inputQ, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int n,
        int blockBytes, int inputRowBytes,
        delegate*<byte*, byte*, float*, int, int, int, void> computeRowsInterleaved,
        ComputeThreadPool? pool)
    {
        int tileGroups = Math.Max(1, ComputeTileM(blockCount * blockBytes) / 4);

        if (pool is null || m < ParallelMinRows)
        {
            GemmR4TiledRange(repackedWeights, inputQ, c, 0, fullGroups, fullGroups,
                tailRows, blockCount, m, n, blockBytes, inputRowBytes, tileGroups,
                doTail: true, computeRowsInterleaved);
            return;
        }

        var ctx = new GemmR4TiledCtx
        {
            RepackedWeights = repackedWeights, InputQ = inputQ, C = c,
            FullGroups = fullGroups, TailRows = tailRows, BlockCount = blockCount,
            M = m, N = n, BlockBytes = blockBytes, InputRowBytes = inputRowBytes,
            TileGroups = tileGroups, ComputeRowsInterleaved = computeRowsInterleaved,
        };
        pool.Dispatch((nint)(&ctx), &GemmR4TiledWorker);
    }

    /// <summary>
    /// Computes the row groups [<paramref name="startGroup"/>, <paramref name="endGroup"/>) for
    /// every token, tiling within that range; optionally also the row-major tail rows.
    /// </summary>
    [SkipLocalsInit]
    private static void GemmR4TiledRange(
        byte* repackedWeights, byte* inputQ, float* c,
        int startGroup, int endGroup, int fullGroups, int tailRows, int blockCount,
        int m, int n, int blockBytes, int inputRowBytes, int tileGroups, bool doTail,
        delegate*<byte*, byte*, float*, int, int, int, void> computeRowsInterleaved)
    {
        int groupBytes = 4 * blockCount * blockBytes;

        for (int gStart = startGroup; gStart < endGroup; gStart += tileGroups)
        {
            int gCount = Math.Min(tileGroups, endGroup - gStart);
            byte* tileWeights = repackedWeights + (long)gStart * groupBytes;

            for (int t = 0; t < n; t++)
            {
                computeRowsInterleaved(
                    tileWeights,
                    inputQ + (long)t * inputRowBytes,
                    c + (long)t * m + gStart * 4,
                    gCount, 0, blockCount);
            }
        }

        // Tail rows live row-major after all interleaved groups. Passing fullGroups: 0 with the
        // tail base makes the same kernel take its row-major branch for exactly those rows.
        if (doTail && tailRows > 0)
        {
            byte* tailBase = repackedWeights + (long)fullGroups * groupBytes;
            for (int t = 0; t < n; t++)
            {
                computeRowsInterleaved(
                    tailBase,
                    inputQ + (long)t * inputRowBytes,
                    c + (long)t * m + fullGroups * 4,
                    0, tailRows, blockCount);
            }
        }
    }

    private struct GemmR4TiledCtx
    {
        public byte* RepackedWeights;
        public byte* InputQ;
        public float* C;
        public int FullGroups;
        public int TailRows;
        public int BlockCount;
        public int M;
        public int N;
        public int BlockBytes;
        public int InputRowBytes;
        public int TileGroups;
        public delegate*<byte*, byte*, float*, int, int, int, void> ComputeRowsInterleaved;
    }

    private static void GemmR4TiledWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmR4TiledCtx>((void*)ctxPtr);

        // Partition whole 4-row groups across threads: each thread owns a disjoint row range of C,
        // and group boundaries never move, so the arithmetic is thread-count independent.
        int groupsPerThread = (ctx.FullGroups + threadCount - 1) / threadCount;
        int startGroup = threadIdx * groupsPerThread;
        int endGroup = Math.Min(startGroup + groupsPerThread, ctx.FullGroups);
        if (startGroup > endGroup) startGroup = endGroup;

        GemmR4TiledRange(ctx.RepackedWeights, ctx.InputQ, ctx.C, startGroup, endGroup,
            ctx.FullGroups, ctx.TailRows, ctx.BlockCount, ctx.M, ctx.N, ctx.BlockBytes,
            ctx.InputRowBytes, ctx.TileGroups, doTail: threadIdx == 0, ctx.ComputeRowsInterleaved);
    }

    // ──────────────────── Per-quant wrappers ────────────────────

    /// <inheritdoc cref="GemmR4Tiled"/>
    [SkipLocalsInit]
    internal static void GemmR4TiledQ8_0(byte* repackedWeights, float* b, byte* inputQ, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int k, int n, ComputeThreadPool? pool)
        => GemmR4Tiled(repackedWeights, b, inputQ, c, fullGroups, tailRows, blockCount, m, k, n,
            Q8_0BlockBytes, blockCount * Q8_0BlockBytes,
            &ComputeRowsQ8_0Interleaved, &QuantizeF32ToQ8_0, pool);

    /// <inheritdoc cref="GemmR4Tiled"/>
    [SkipLocalsInit]
    internal static void GemmR4TiledQ5_0(byte* repackedWeights, float* b, byte* inputQ, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int k, int n, ComputeThreadPool? pool)
        => GemmR4Tiled(repackedWeights, b, inputQ, c, fullGroups, tailRows, blockCount, m, k, n,
            Q5_0BlockBytes, blockCount * Q8_1BlockBytes,
            &ComputeRowsQ5_0Interleaved, &QuantizeF32ToQ8_1, pool);

    /// <inheritdoc cref="GemmR4Tiled"/>
    [SkipLocalsInit]
    internal static void GemmR4TiledQ4_K(byte* repackedWeights, float* b, byte* inputQ, float* c,
        int fullGroups, int tailRows, int superBlockCount, int m, int k, int n, ComputeThreadPool? pool)
        => GemmR4Tiled(repackedWeights, b, inputQ, c, fullGroups, tailRows, superBlockCount, m, k, n,
            Q4_K_BlockBytes, superBlockCount * Q8_K_BlockBytes,
            &ComputeRowsQ4_KInterleaved, &QuantizeF32ToQ8_K, pool);

    /// <inheritdoc cref="GemmR4Tiled"/>
    [SkipLocalsInit]
    internal static void GemmR4TiledQ5_K(byte* repackedWeights, float* b, byte* inputQ, float* c,
        int fullGroups, int tailRows, int superBlockCount, int m, int k, int n, ComputeThreadPool? pool)
        => GemmR4Tiled(repackedWeights, b, inputQ, c, fullGroups, tailRows, superBlockCount, m, k, n,
            Q5_K_BlockBytes, superBlockCount * Q8_K_BlockBytes,
            &ComputeRowsQ5_KInterleaved, &QuantizeF32ToQ8_K, pool);

    /// <inheritdoc cref="GemmR4Tiled"/>
    [SkipLocalsInit]
    internal static void GemmR4TiledQ6_K(byte* repackedWeights, float* b, byte* inputQ, float* c,
        int fullGroups, int tailRows, int superBlockCount, int m, int k, int n, ComputeThreadPool? pool)
        => GemmR4Tiled(repackedWeights, b, inputQ, c, fullGroups, tailRows, superBlockCount, m, k, n,
            Q6_K_BlockBytes, superBlockCount * Q8_K_BlockBytes,
            &ComputeRowsQ6_KInterleaved, &QuantizeF32ToQ8_K, pool);
}
