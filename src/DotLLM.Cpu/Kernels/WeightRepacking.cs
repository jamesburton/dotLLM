using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// The "R4" view the 4-row kernels consume: a quantized [M, K] weight seen as full groups of 4 consecutive rows
/// plus M % 4 tail rows, over the original row-major (memory-mapped) weights.
/// </summary>
/// <remarks>
/// This used to build an interleaved copy (each group's blocks column by column). That copy was a second committed
/// copy of the projections, about the size of the GGUF, and measured no faster: the same kernels over the original
/// rows ran within noise and bit-identical (#1156). Nothing is allocated now.
/// </remarks>
public static unsafe class WeightRepacking
{
    /// <summary>Number of rows interleaved per group. Matches existing 4-row kernel batching.</summary>
    public const int InterleaveFactor = 4;

    /// <summary>
    /// A 4-row-group view over row-major quantized weights: group <c>g</c> starts at <c>Ptr + g·4·RowBytes</c>, its rows
    /// are <see cref="RowBytes"/> apart, and the M % 4 tail rows follow the last group.
    /// </summary>
    internal readonly struct RepackedWeight : IDisposable
    {
        /// <summary>The original row-major weights (not owned).</summary>
        public readonly nint Ptr;

        /// <summary>Number of complete 4-row groups (M / 4).</summary>
        public readonly int FullGroupCount;

        /// <summary>Number of leftover rows (M % 4).</summary>
        public readonly int TailRows;

        /// <summary>Number of quant blocks per row (K / groupSize).</summary>
        public readonly int BlocksPerRow;

        /// <summary>Byte size of one quant block.</summary>
        public readonly int BlockBytes;

        /// <summary>Bytes allocated for the view: always 0 (#1156).</summary>
        public readonly long AllocatedBytes;

        public RepackedWeight(nint ptr, int fullGroupCount, int tailRows,
                              int blocksPerRow, int blockBytes, long allocatedBytes)
        {
            Ptr = ptr;
            FullGroupCount = fullGroupCount;
            TailRows = tailRows;
            BlocksPerRow = blocksPerRow;
            BlockBytes = blockBytes;
            AllocatedBytes = allocatedBytes;
        }

        /// <summary>Byte size of one original row: blocksPerRow * blockBytes.</summary>
        public int RowBytes
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => BlocksPerRow * BlockBytes;
        }

        /// <summary>Pointer to the start of the tail rows (after all full groups).</summary>
        public byte* TailPtr
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (byte*)Ptr + (long)FullGroupCount * InterleaveFactor * RowBytes;
        }

        /// <summary>Nothing to free: the view does not own <see cref="Ptr"/> (#1156).</summary>
        public void Dispose()
        {
        }
    }

    /// <summary>
    /// Returns block byte size and group size for a given quantization type.
    /// Returns (0, 0) for unsupported types (F32, F16).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (int blockBytes, int groupSize) GetBlockInfo(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => (34, 32),    // Q8_0BlockBytes, Q8_0GroupSize
        QuantizationType.Q5_0 => (22, 32),    // Q5_0BlockBytes, Q5_0GroupSize
        QuantizationType.Q4_K => (144, 256),  // Q4_K_BlockBytes, KQuantGroupSize
        QuantizationType.Q5_K => (176, 256),  // Q5_K_BlockBytes, KQuantGroupSize
        QuantizationType.Q6_K => (210, 256),  // Q6_K_BlockBytes, KQuantGroupSize
        _ => (0, 0),
    };

    /// <summary>
    /// Returns true if the given quant type supports R4 repacking.
    /// F32/F16 are not block-structured and are skipped.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsRepackable(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q5_0
            or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;

    /// <summary>
    /// Returns the 4-row-group view of a [M, K] row-major quantized weight matrix. Allocates and copies nothing (#1156).
    /// </summary>
    /// <param name="sourcePtr">Pointer to original row-major quantized weights.</param>
    /// <param name="qt">Quantization type of the weights.</param>
    /// <param name="m">Number of rows (output dimension).</param>
    /// <param name="k">Number of columns / elements per row (input dimension).</param>
    /// <returns>A <see cref="RepackedWeight"/> view over <paramref name="sourcePtr"/>, which must outlive it.</returns>
    internal static RepackedWeight RepackR4(nint sourcePtr, QuantizationType qt, int m, int k)
    {
        var (blockBytes, groupSize) = GetBlockInfo(qt);
        if (blockBytes == 0)
            throw new ArgumentException($"Quantization type {qt} does not support R4 repacking.", nameof(qt));

        if (k % groupSize != 0)
            throw new ArgumentException(
                $"k ({k}) must be a multiple of group size ({groupSize}) for {qt}.", nameof(k));

        int blocksPerRow = k / groupSize;
        int fullGroups = m / InterleaveFactor;
        int tailRows = m % InterleaveFactor;

        if (m == 0)
            return default;

        // A view, not a copy (#1156): the 4-row kernels read the original rows in place (rows rowBytes apart,
        // blocks blockBytes apart). Same group offsets and tail position the interleaved copy had, same arithmetic.
        return new RepackedWeight(sourcePtr, fullGroups, tailRows, blocksPerRow, blockBytes, allocatedBytes: 0);
    }
}
