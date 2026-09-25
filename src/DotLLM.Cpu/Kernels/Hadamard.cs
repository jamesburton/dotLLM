using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Blockwise normalized Walsh-Hadamard transform for PrismML Hadamard-folded checkpoints
/// (<c>prism.hadamard.*</c>, see <see cref="DotLLM.Core.Models.HadamardFoldConfig"/>).
/// </summary>
/// <remarks>
/// <para>
/// The stored weights live in a rotated basis, so each folded weight's input activation must be
/// rotated to match before the matmul. The operator is a Sylvester-ordered Walsh-Hadamard transform
/// scaled by <c>1/sqrt(blockSize)</c>, applied independently to each contiguous <c>blockSize</c>-wide
/// block of the row — equivalently a dense multiply by <c>H</c> where
/// <c>H[r,c] = ±1/sqrt(n)</c> with the sign given by the parity of <c>popcount(r &amp; c)</c>.
/// </para>
/// <para>
/// It is computed as an in-place butterfly rather than that dense matmul, so the rotation matrix is
/// never materialized and the cost is O(n log n) over the activation alone — no weight traffic.
/// </para>
/// </remarks>
public static class Hadamard
{
    /// <summary>
    /// Applies the forward transform to one activation row: <b>signs first, then the rotation</b>.
    /// </summary>
    /// <param name="src">Source activation row; length must be a positive multiple of <paramref name="blockSize"/>.</param>
    /// <param name="signs">Per-element ±1 vector of the same length, or empty for the identity sign step.</param>
    /// <param name="dst">Destination row, same length as <paramref name="src"/>. May alias <paramref name="src"/>.</param>
    /// <param name="blockSize">Hadamard block width; must be a power of two.</param>
    public static void ForwardRow(
        ReadOnlySpan<float> src, ReadOnlySpan<sbyte> signs, Span<float> dst, int blockSize)
    {
        ValidateRow(src, signs, dst, blockSize);

        float scale = 1f / MathF.Sqrt(blockSize);
        if (signs.IsEmpty)
        {
            for (int i = 0; i < src.Length; i++)
                dst[i] = src[i] * scale;
        }
        else
        {
            // Fused: the sign flip and the 1/sqrt(n) normalization are both elementwise and both
            // precede every butterfly pass, so they cost one pass together rather than two.
            for (int i = 0; i < src.Length; i++)
                dst[i] = signs[i] < 0 ? -src[i] * scale : src[i] * scale;
        }

        ButterflyBlocks(dst, blockSize);
    }

    /// <summary>
    /// Applies the inverse transform to one looked-up row: <b>the rotation first, then signs</b>.
    /// </summary>
    /// <remarks>
    /// The order is reversed relative to <see cref="ForwardRow"/> because the rotation is its own
    /// inverse and <c>(H∘S)⁻¹ = S⁻¹∘H⁻¹ = S∘H</c>. Applying the signs on the wrong side is silent
    /// corruption, not a crash.
    /// </remarks>
    /// <param name="src">Source row (a row fetched from a rotated lookup table).</param>
    /// <param name="signs">Per-element ±1 vector of the same length, or empty for the identity sign step.</param>
    /// <param name="dst">Destination row, same length as <paramref name="src"/>. May alias <paramref name="src"/>.</param>
    /// <param name="blockSize">Hadamard block width; must be a power of two.</param>
    public static void InverseRow(
        ReadOnlySpan<float> src, ReadOnlySpan<sbyte> signs, Span<float> dst, int blockSize)
    {
        ValidateRow(src, signs, dst, blockSize);

        float scale = 1f / MathF.Sqrt(blockSize);
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] * scale;

        ButterflyBlocks(dst, blockSize);

        if (!signs.IsEmpty)
        {
            for (int i = 0; i < dst.Length; i++)
            {
                if (signs[i] < 0)
                    dst[i] = -dst[i];
            }
        }
    }

    /// <summary>
    /// Reorders a GDN value-head activation from the recurrence's <i>tiled</i> head order
    /// <c>[dState, nKHead, rep]</c> into the <i>grouped</i> order <c>[dState, rep, nKHead]</c> that
    /// an <c>ssm_out</c> fold was computed in (<c>prism.hadamard.gdn_v_grouped</c>).
    /// </summary>
    /// <remarks>
    /// Our recurrence numbers value heads <c>vh = k + nKHead·r</c> (k-head minor), because
    /// <c>GatedDeltaNetScan</c> broadcasts with <c>kh = vh % NKHead</c> to match llama.cpp's tiled
    /// convention. The fold instead assumes <c>vh' = r + rep·k</c>. Both layouts keep
    /// <paramref name="dState"/> contiguous and minor.
    /// </remarks>
    /// <param name="src">Source row of <c>dState · nKHead · rep</c> elements.</param>
    /// <param name="dst">Destination row, same length. Must NOT alias <paramref name="src"/>.</param>
    /// <param name="dState">Per-head width (contiguous, fastest-varying).</param>
    /// <param name="nKHead">Number of key heads.</param>
    /// <param name="rep">Value heads per key head (<c>NVHead / NKHead</c>).</param>
    public static void PermuteTiledToGrouped(
        ReadOnlySpan<float> src, Span<float> dst, int dState, int nKHead, int rep)
    {
        int expected = dState * nKHead * rep;
        if (src.Length != expected || dst.Length != expected)
            throw new ArgumentException(
                $"GDN permute expects {expected} elements (dState {dState} x nKHead {nKHead} x rep {rep}), " +
                $"got src {src.Length} / dst {dst.Length}.");

        for (int k = 0; k < nKHead; k++)
        {
            for (int r = 0; r < rep; r++)
            {
                int srcHead = k + nKHead * r; // tiled   — what the recurrence emits
                int dstHead = r + rep * k;    // grouped — what the fold assumed
                src.Slice(srcHead * dState, dState).CopyTo(dst.Slice(dstHead * dState, dState));
            }
        }
    }

    private static void ValidateRow(
        ReadOnlySpan<float> src, ReadOnlySpan<sbyte> signs, Span<float> dst, int blockSize)
    {
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
            throw new ArgumentException($"Hadamard block size must be a power of two, got {blockSize}.", nameof(blockSize));
        if (src.Length == 0 || src.Length % blockSize != 0)
            throw new ArgumentException(
                $"Activation width {src.Length} is not a positive multiple of Hadamard block size {blockSize}.");
        if (dst.Length != src.Length)
            throw new ArgumentException($"Destination length {dst.Length} does not match source {src.Length}.");
        if (!signs.IsEmpty && signs.Length != src.Length)
            throw new ArgumentException($"Sign vector length {signs.Length} does not match activation width {src.Length}.");
    }

    /// <summary>
    /// Runs the in-place butterfly passes over each <paramref name="blockSize"/>-wide block.
    /// Assumes the <c>1/sqrt(n)</c> normalization has already been folded into the input.
    /// </summary>
    private static void ButterflyBlocks(Span<float> row, int blockSize)
    {
        for (int offset = 0; offset < row.Length; offset += blockSize)
            Butterflies(row.Slice(offset, blockSize));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Butterflies(Span<float> block)
    {
        int n = block.Length;

        fixed (float* p = block)
        {
            // Strides narrower than one vector cannot be expressed as whole-vector loads of the two
            // halves, so they run scalar; every wider stride is vectorized below.
            int vectorStride = Vector256.IsHardwareAccelerated ? Vector256<float>.Count
                             : Vector128.IsHardwareAccelerated ? Vector128<float>.Count
                             : n;

            for (int len = 1; len < vectorStride && len < n; len <<= 1)
            {
                for (int i = 0; i < n; i += 2 * len)
                {
                    for (int j = 0; j < len; j++)
                    {
                        float u = p[i + j];
                        float v = p[i + len + j];
                        p[i + j] = u + v;
                        p[i + len + j] = u - v;
                    }
                }
            }

            if (Vector256.IsHardwareAccelerated)
            {
                for (int len = vectorStride; len < n; len <<= 1)
                {
                    for (int i = 0; i < n; i += 2 * len)
                    {
                        for (int j = 0; j < len; j += Vector256<float>.Count)
                        {
                            var u = Vector256.Load(p + i + j);
                            var v = Vector256.Load(p + i + len + j);
                            Vector256.Store(u + v, p + i + j);
                            Vector256.Store(u - v, p + i + len + j);
                        }
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                for (int len = vectorStride; len < n; len <<= 1)
                {
                    for (int i = 0; i < n; i += 2 * len)
                    {
                        for (int j = 0; j < len; j += Vector128<float>.Count)
                        {
                            var u = Vector128.Load(p + i + j);
                            var v = Vector128.Load(p + i + len + j);
                            Vector128.Store(u + v, p + i + j);
                            Vector128.Store(u - v, p + i + len + j);
                        }
                    }
                }
            }
        }
    }
}
