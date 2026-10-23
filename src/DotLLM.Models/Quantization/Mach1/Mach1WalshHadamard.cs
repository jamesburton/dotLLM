// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Buffers;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Orthonormal Walsh-Hadamard transform (Sylvester order), along the last
/// axis of a row-major matrix. Mirrors decode.py's <c>_np_hadamard</c>
/// exactly, including its exact op order: butterflies pair at stride 1, 2,
/// 4, ... in <see cref="float"/>, with a single <see cref="float"/>
/// <b>division</b> by <c>sqrt(dim)</c> after the final pass.
/// </summary>
/// <remarks>
/// Bit-exactness note: the reference divides by <c>sqrt(dim)</c> rather than
/// multiplying by a precomputed reciprocal, and never fuses the butterfly
/// add/subtract into a single fused-multiply-add. This port does the same —
/// using a precomputed <c>1/sqrt(dim)</c> multiply, or an FMA-fused
/// butterfly, would round differently in the last bit and break the golden
/// bit-exactness gate.
/// </remarks>
public static class Mach1WalshHadamard
{
    /// <summary>
    /// Transforms every row of a <c>[rows, dim]</c> row-major matrix in
    /// place. <paramref name="dim"/> must be a power of two.
    /// </summary>
    public static void TransformRowsInPlace(Span<float> data, int rows, int dim)
    {
        if (dim <= 0 || (dim & (dim - 1)) != 0)
            throw new ArgumentException($"WHT requires a power-of-2 dim, got {dim}", nameof(dim));
        if (data.Length != rows * dim)
            throw new ArgumentException($"data length {data.Length} must equal rows*dim={rows * dim}");

        float[] scratch = ArrayPool<float>.Shared.Rent(dim);
        try
        {
            Span<float> scratchSpan = scratch.AsSpan(0, dim);
            for (int r = 0; r < rows; r++)
                TransformRow(data.Slice(r * dim, dim), scratchSpan, dim);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    private static void TransformRow(Span<float> row, Span<float> scratch, int dim)
    {
        int span = 1;
        while (span < dim)
        {
            int blockSize = span * 2;
            for (int start = 0; start < dim; start += blockSize)
            {
                for (int i = 0; i < span; i++)
                {
                    float a = row[start + i];
                    float b = row[start + span + i];
                    scratch[i] = a + b;
                    scratch[span + i] = a - b;
                }
                scratch.Slice(0, blockSize).CopyTo(row.Slice(start, blockSize));
            }
            span *= 2;
        }

        // Single fp32 division by sqrt(dim) after the final butterfly pass —
        // not a precomputed-reciprocal multiply (see remarks above).
        float sqrtDim = MathF.Sqrt((float)dim);
        for (int i = 0; i < dim; i++)
            row[i] /= sqrtDim;
    }
}
