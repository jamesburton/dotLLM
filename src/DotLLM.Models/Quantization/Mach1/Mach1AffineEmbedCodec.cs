// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Buffers;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Affine asymmetric group codec used by the embedding tier
/// (<c>packed/ne/embed_int{bits}.safetensors</c>): per-row groups of
/// <c>group</c> hidden elements share an <c>(mn, mx)</c> fp16 range; each
/// element is a <c>bits</c>-wide unsigned code, MSB-first bit-packed
/// per row (matching numpy's <c>packbits</c>/<c>unpackbits</c> big-bit-order
/// convention — for <c>bits=4</c> this is exactly one nibble per element,
/// high nibble first). Optional exact-overwrite exceptions
/// (<c>exc_idx</c>/<c>exc_bits</c>) restore encoder-GPU FMA-boundary
/// elements to their exact bf16 bit pattern. Mirrors decode.py's
/// <c>pack_embed_q</c> / <c>unpack_embed_q</c> / <c>decode_embed</c>.
/// </summary>
public static class Mach1AffineEmbedCodec
{
    /// <summary>
    /// Packs <c>q[rows, hid]</c> (values <c>&lt; 2^bits</c>) into
    /// <c>packed[rows, ceil(hid*bits/8)]</c>, MSB-first per row.
    /// </summary>
    public static void PackCodes(ReadOnlySpan<byte> q, int rows, int hid, int bits, Span<byte> packed)
    {
        int bytesPerRow = (hid * bits + 7) / 8;
        if (q.Length != rows * hid)
            throw new ArgumentException($"q length {q.Length} must equal rows*hid={rows * hid}");
        if (packed.Length != rows * bytesPerRow)
            throw new ArgumentException($"packed length {packed.Length} must equal rows*bytesPerRow={rows * bytesPerRow}");

        packed.Clear();
        for (int r = 0; r < rows; r++)
        {
            int byteBase = r * bytesPerRow;
            int bitPos = 0;
            for (int e = 0; e < hid; e++)
            {
                int code = q[r * hid + e];
                for (int b = bits - 1; b >= 0; b--)
                {
                    int bitVal = (code >> b) & 1;
                    int byteIdx = byteBase + bitPos / 8;
                    int bitInByte = 7 - (bitPos % 8);
                    if (bitVal != 0)
                        packed[byteIdx] |= (byte)(1 << bitInByte);
                    bitPos++;
                }
            }
        }
    }

    /// <summary>
    /// Inverse of <see cref="PackCodes"/>: <c>packed[rows,
    /// ceil(hid*bits/8)]</c> -&gt; <c>codes[rows, hid]</c> (values in
    /// <c>[0, 2^bits)</c>).
    /// </summary>
    public static void UnpackCodes(ReadOnlySpan<byte> packed, int rows, int hid, int bits, Span<byte> codes)
    {
        int bytesPerRow = (hid * bits + 7) / 8;
        if (packed.Length != rows * bytesPerRow)
            throw new ArgumentException($"packed length {packed.Length} must equal rows*bytesPerRow={rows * bytesPerRow}");
        if (codes.Length != rows * hid)
            throw new ArgumentException($"codes length {codes.Length} must equal rows*hid={rows * hid}");

        for (int r = 0; r < rows; r++)
        {
            int byteBase = r * bytesPerRow;
            int bitPos = 0;
            for (int e = 0; e < hid; e++)
            {
                int code = 0;
                for (int b = 0; b < bits; b++)
                {
                    int byteIdx = byteBase + bitPos / 8;
                    int bitInByte = 7 - (bitPos % 8);
                    int bitVal = (packed[byteIdx] >> bitInByte) & 1;
                    code = (code << 1) | bitVal;
                    bitPos++;
                }
                codes[r * hid + e] = (byte)code;
            }
        }
    }

    /// <summary>
    /// Decodes affine int-<paramref name="bits"/> codes to dense
    /// <c>[rows, hid]</c> fp32: <c>W[r, g*group+k] = mn[r,g] + q[r,g,k] *
    /// max(mx[r,g]-mn[r,g], 1e-8) / (2^bits - 1)</c>, then overwrites any
    /// exception elements with their exact bf16 bit pattern (stored as the
    /// top 16 bits of an fp32 word).
    /// </summary>
    /// <param name="qPacked">Packed codes, <c>[rows, ceil(hid*bits/8)]</c>.</param>
    /// <param name="mn">Per-group minimum, <c>[rows, hid/group]</c>.</param>
    /// <param name="mx">Per-group maximum, <c>[rows, hid/group]</c>.</param>
    /// <param name="rows">Row count.</param>
    /// <param name="hid">Hidden size.</param>
    /// <param name="bits">Bits per code (4 for the shipped tier).</param>
    /// <param name="group">Group size (64 for the shipped tier).</param>
    /// <param name="excIdx">Optional flat element indices to overwrite exactly (empty if none).</param>
    /// <param name="excBits">Optional bf16 bit patterns for those elements (top 16 bits of the fp32 word).</param>
    /// <param name="dest">Destination for the dense <c>[rows, hid]</c> result, row-major.</param>
    public static void Decode(
        ReadOnlySpan<byte> qPacked, ReadOnlySpan<Half> mn, ReadOnlySpan<Half> mx,
        int rows, int hid, int bits, int group,
        ReadOnlySpan<int> excIdx, ReadOnlySpan<ushort> excBits,
        Span<float> dest)
    {
        int ng = hid / group;
        if (hid % group != 0)
            throw new ArgumentException($"hid={hid} must be a multiple of group={group}");
        if (mn.Length != rows * ng || mx.Length != rows * ng)
            throw new ArgumentException($"mn/mx length must equal rows*(hid/group)={rows * ng}");
        if (dest.Length != rows * hid)
            throw new ArgumentException($"dest length {dest.Length} must equal rows*hid={rows * hid}");

        byte[] codesArr = ArrayPool<byte>.Shared.Rent(rows * hid);
        try
        {
            Span<byte> codes = codesArr.AsSpan(0, rows * hid);
            UnpackCodes(qPacked, rows, hid, bits, codes);

            float lv = (1 << bits) - 1;
            for (int r = 0; r < rows; r++)
            {
                int rowBase = r * hid;
                int scaleRowBase = r * ng;
                for (int g = 0; g < ng; g++)
                {
                    float mnF = (float)mn[scaleRowBase + g];
                    float mxF = (float)mx[scaleRowBase + g];
                    float range = MathF.Max(mxF - mnF, 1e-8f);
                    float step = range / lv;
                    int groupBase = rowBase + g * group;
                    for (int k = 0; k < group; k++)
                        dest[groupBase + k] = mnF + codes[groupBase + k] * step;
                }
            }

            for (int i = 0; i < excIdx.Length; i++)
            {
                uint bits32 = (uint)excBits[i] << 16;
                dest[excIdx[i]] = BitConverter.UInt32BitsToSingle(bits32);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(codesArr);
        }
    }
}
