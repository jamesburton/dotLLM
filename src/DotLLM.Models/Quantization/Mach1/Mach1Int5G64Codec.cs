// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Buffers;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Symmetric int5-g64 codec used by the LM head tier (<c>codec:
/// "int5g64_packed"</c>): 8 codes packed into 5 little-endian bytes (code
/// <c>i</c> occupies bits <c>[5i, 5i+5)</c> of the 40-bit block; stored value
/// is <c>q + 16</c> for <c>q in [-16, 15]</c>), one fp16 scale per 64
/// reduction-dim weights, raw domain (no rotation). Mirrors decode.py's
/// <c>pack_int5</c> / <c>unpack_int5</c> / <c>decode_int5g64</c>.
/// </summary>
public static class Mach1Int5G64Codec
{
    /// <summary>
    /// Packs <c>q[m, n]</c> (values in <c>[-16, 15]</c>, <c>n</c> a multiple
    /// of 8) into <c>packed[m, n/8*5]</c>.
    /// </summary>
    public static void Pack(ReadOnlySpan<sbyte> q, int m, int n, Span<byte> packed)
    {
        if (n % 8 != 0)
            throw new ArgumentException($"n={n} must be a multiple of 8.", nameof(n));
        int blocksPerRow = n / 8;
        if (q.Length != m * n)
            throw new ArgumentException($"q length {q.Length} must equal m*n={m * n}");
        if (packed.Length != m * blocksPerRow * 5)
            throw new ArgumentException($"packed length {packed.Length} must equal m*(n/8*5)={m * blocksPerRow * 5}");

        for (int r = 0; r < m; r++)
        {
            int rowBase = r * n;
            int outRowBase = r * blocksPerRow * 5;
            for (int b = 0; b < blocksPerRow; b++)
            {
                ulong word = 0;
                for (int i = 0; i < 8; i++)
                {
                    int code = q[rowBase + b * 8 + i] + 16;
                    if (code < 0 || code > 31)
                        throw new ArgumentOutOfRangeException(nameof(q), code - 16, "int5 code must be in [-16, 15].");
                    word |= (ulong)(uint)code << (5 * i);
                }
                int outBase = outRowBase + b * 5;
                for (int byteI = 0; byteI < 5; byteI++)
                    packed[outBase + byteI] = (byte)((word >> (8 * byteI)) & 0xFF);
            }
        }
    }

    /// <summary>Inverse of <see cref="Pack"/>: <c>packed[m, n/8*5]</c> -&gt; <c>q[m, n]</c>.</summary>
    public static void Unpack(ReadOnlySpan<byte> packed, int m, int n, Span<sbyte> q)
    {
        if (n % 8 != 0)
            throw new ArgumentException($"n={n} must be a multiple of 8.", nameof(n));
        int blocksPerRow = n / 8;
        if (packed.Length != m * blocksPerRow * 5)
            throw new ArgumentException($"packed length {packed.Length} must equal m*(n/8*5)={m * blocksPerRow * 5}");
        if (q.Length != m * n)
            throw new ArgumentException($"q length {q.Length} must equal m*n={m * n}");

        for (int r = 0; r < m; r++)
        {
            int inRowBase = r * blocksPerRow * 5;
            int rowBase = r * n;
            for (int b = 0; b < blocksPerRow; b++)
            {
                int inBase = inRowBase + b * 5;
                ulong word = 0;
                for (int byteI = 0; byteI < 5; byteI++)
                    word |= (ulong)packed[inBase + byteI] << (8 * byteI);
                for (int i = 0; i < 8; i++)
                {
                    int code = (int)((word >> (5 * i)) & 0x1F);
                    q[rowBase + b * 8 + i] = (sbyte)(code - 16);
                }
            }
        }
    }

    /// <summary>
    /// Decodes one int5-g64 chunk to dense <c>[m0, n0]</c> fp32:
    /// <c>W[r,j] = q[r,j] * gscale[r, j/group]</c>, then overwrites any
    /// protected rows with their exact dense values.
    /// </summary>
    /// <param name="qp">Packed codes, <c>[m0, n0/8*5]</c>.</param>
    /// <param name="gscale">Per-group scale, <c>[m0, ceil(n0/group)]</c>.</param>
    /// <param name="m0">Row count.</param>
    /// <param name="n0">Column count (must be a multiple of 8).</param>
    /// <param name="group">Reduction-dim group size for the scale (64 in the shipped head tier).</param>
    /// <param name="protRows">Optional row indices to overwrite exactly (empty if none).</param>
    /// <param name="protDense">Optional dense values for those rows, <c>[protRows.Length, n0]</c>.</param>
    /// <param name="dest">Destination for the dense <c>[m0, n0]</c> result, row-major.</param>
    public static void Decode(
        ReadOnlySpan<byte> qp, ReadOnlySpan<float> gscale, int m0, int n0, int group,
        ReadOnlySpan<int> protRows, ReadOnlySpan<float> protDense, Span<float> dest)
    {
        int ng = (n0 + group - 1) / group;
        sbyte[] qArr = ArrayPool<sbyte>.Shared.Rent(m0 * n0);
        try
        {
            Span<sbyte> q = qArr.AsSpan(0, m0 * n0);
            Unpack(qp, m0, n0, q);

            for (int r = 0; r < m0; r++)
            {
                int rowBase = r * n0;
                int scaleRowBase = r * ng;
                for (int j = 0; j < n0; j++)
                {
                    float scale = gscale[scaleRowBase + j / group];
                    dest[rowBase + j] = q[rowBase + j] * scale;
                }
            }

            for (int k = 0; k < protRows.Length; k++)
            {
                int row = protRows[k];
                int destRowBase = row * n0;
                int srcRowBase = k * n0;
                for (int j = 0; j < n0; j++)
                    dest[destRowBase + j] = protDense[srcRowBase + j];
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(qArr);
        }
    }
}
