using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Cuda;

/// <summary>
/// Issue #1099 — the bounded dequant + cuBLAS fallback for a multi-row projection too wide to
/// expand whole: the LM head of an all-logits forward (verify, perplexity, a full-logits prefill).
/// </summary>
/// <remarks>
/// <para>
/// The LM head is the one projection in these hosts whose F16 expansion dwarfs every layer's:
/// Bonsai 2 27B's <c>[248320, 5120]</c> head is 2.5 GiB in F16 against 178 MiB for its widest layer
/// projection. Expanding it whole into the grow-only dequant scratch is what pushed a 12 GB card to
/// 0 MiB free and into WDDM shared memory.
/// </para>
/// <para>
/// Hosts with a packed multi-row kernel for the head's quant type (PQ2_0 on the Qwen3.5 dense hybrid)
/// run the head through that kernel in row chunks and never get here. This is for the quant types
/// without one: the head is dequantized in output-row tiles of at most <c>boundElems</c> elements
/// into the existing scratch, and each tile is one cuBLAS GEMM writing its column block of the
/// <c>[seqLen, m]</c> output in place (<c>ldc = m</c>). Every output element is still one
/// k-length dot product of the same F16 operands, so the numerics are those of the whole-matrix path;
/// only the scratch is bounded.
/// </para>
/// </remarks>
public static class CudaWideProjection
{
    /// <summary>
    /// Smallest tile bound, in F16 elements (32 Mi = 64 MiB): used when the scratch is still smaller
    /// (a model whose layers never dequantize), so a tile is never so thin that launch overhead dominates.
    /// </summary>
    public const long TileFloorElems = 32L << 20;

    /// <summary>
    /// In-process switch for tests: <see langword="false"/> sends the LM head back to the pre-#1099
    /// whole-matrix dequant, the reference arm of the A/B parity tests. Not for production use.
    /// </summary>
    internal static bool? BoundedLmHeadOverride { get; set; }

    /// <summary>
    /// Whether a <paramref name="qt"/> weight can be dequantized in output-row tiles: its rows must be
    /// contiguous, independently decodable byte ranges. Excludes F32 (never dequantized), PQ2_0 (repacked
    /// to the split layout at load, not row-contiguous) and I2_S (one per-tensor scale at the tail).
    /// </summary>
    /// <param name="qt">The weight's quantization type.</param>
    /// <returns><see langword="true"/> when <see cref="RunTiled"/> accepts the type.</returns>
    public static bool SupportsRowTiles(QuantizationType qt)
        => qt is not (QuantizationType.F32 or QuantizationType.PQ2_0 or QuantizationType.I2_S);

    /// <summary>
    /// Output rows per tile: the most whole rows of length <paramref name="k"/> that fit
    /// <paramref name="boundElems"/> elements, at least one, at most <paramref name="m"/>.
    /// </summary>
    /// <param name="m">Output rows of the weight.</param>
    /// <param name="k">Input width (row length).</param>
    /// <param name="boundElems">F16 elements a tile may occupy.</param>
    /// <returns>Rows per tile.</returns>
    public static int TileRows(int m, int k, long boundElems)
        => (int)Math.Clamp(boundElems / k, 1, m);

    /// <summary>
    /// <c>y[seqLen, m] = x[seqLen, k] @ W[m, k]ᵀ</c> with <c>W</c> dequantized one output-row tile at a
    /// time into <paramref name="scratchF16"/>.
    /// </summary>
    /// <param name="kernels">Kernel set providing <see cref="CudaKernels.LaunchDequantToF16"/>.</param>
    /// <param name="cublas">cuBLAS handle.</param>
    /// <param name="weight">Device pointer to the row-major quantized weight.</param>
    /// <param name="qt">Its quantization type (<see cref="SupportsRowTiles"/> must hold).</param>
    /// <param name="xF16">F16 activations <c>[seqLen, k]</c>.</param>
    /// <param name="y">Output <c>[seqLen, m]</c>: F16, or F32 when <paramref name="f32Out"/>.</param>
    /// <param name="f32Out">Write F32 (FP32 accumulate, as <see cref="CudaGemm.LinearF16InF32Out"/>) instead of F16.</param>
    /// <param name="m">Output rows of the weight.</param>
    /// <param name="k">Input width.</param>
    /// <param name="seqLen">Token rows.</param>
    /// <param name="scratchF16">F16 weight scratch of at least <paramref name="tileRows"/> * <paramref name="k"/> elements.</param>
    /// <param name="tileRows">Output rows per tile, from <see cref="TileRows"/>.</param>
    /// <param name="stream">CUDA stream.</param>
    /// <returns>The number of tiles launched.</returns>
    /// <exception cref="ArgumentException"><paramref name="qt"/> is not row-tileable.</exception>
    public static int RunTiled(CudaKernels kernels, nint cublas, nint weight, QuantizationType qt,
        nint xF16, nint y, bool f32Out, int m, int k, int seqLen, nint scratchF16, int tileRows, nint stream)
    {
        if (!SupportsRowTiles(qt))
            throw new ArgumentException($"{qt} weights are not row-contiguous and cannot be dequantized in row tiles.", nameof(qt));

        long rowBytes = Dequantize.RowByteSize(k, qt);
        int outElemBytes = f32Out ? sizeof(float) : sizeof(ushort);
        int tiles = 0;
        for (int r0 = 0; r0 < m; r0 += tileRows, tiles++)
        {
            int rows = Math.Min(tileRows, m - r0);
            kernels.LaunchDequantToF16(weight + (nint)(r0 * rowBytes), qt, scratchF16, checked(rows * k), stream);
            nint yTile = y + (nint)((long)r0 * outElemBytes);
            if (f32Out)
                CudaGemm.LinearF16InF32Out(cublas, xF16, scratchF16, yTile, seqLen, k, rows, stream, ldy: m);
            else
                CudaGemm.LinearF16(cublas, xF16, scratchF16, yTile, seqLen, k, rows, stream, ldy: m);
        }
        return tiles;
    }
}
