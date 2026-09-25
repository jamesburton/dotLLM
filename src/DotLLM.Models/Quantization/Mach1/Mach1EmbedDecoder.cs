// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Runtime.InteropServices;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Decodes the embedding tier from an already-open
/// <c>packed/ne/embed_int{bits}.safetensors</c> file: tensors
/// <c>q_packed|mn|mx|exc_idx|exc_bits</c>, format "affine int4 asym g64 +
/// exact-overwrite exceptions". Mirrors decode.py's <c>decode_embed</c>.
/// </summary>
public static class Mach1EmbedDecoder
{
    /// <summary>
    /// Decodes the whole embedding table to dense <c>[rows, hid]</c> fp32.
    /// <c>group</c> is inferred from the stored shapes when not given, so
    /// differently-grouped packs decode identically.
    /// </summary>
    /// <param name="file">The open <c>embed_int{bits}.safetensors</c> file.</param>
    /// <param name="bits">Bits per code (4 for the shipped tier).</param>
    /// <param name="dest">Destination for the dense <c>[rows, hid]</c> result, row-major.</param>
    /// <param name="group">Group size; if <c>null</c>, inferred from the stored tensor shapes.</param>
    public static void Decode(ISafetensorsTensorSource file, int bits, Span<float> dest, int? group = null)
    {
        var qPackedDesc = file.TensorsByName["q_packed"]; // [rows, bytesPerRow]
        int rows = qPackedDesc.Shape[0];
        int bytesPerRow = qPackedDesc.Shape[1];

        var mnDesc = file.TensorsByName["mn"]; // [rows, ng]
        int ng = mnDesc.Shape[1];

        int resolvedGroup = group ?? (bytesPerRow * 8 / bits) / ng;
        int hid = ng * resolvedGroup;

        var qPacked = file.GetTensorSpan("q_packed");
        var mn = MemoryMarshal.Cast<byte, Half>(file.GetTensorSpan("mn"));
        var mx = MemoryMarshal.Cast<byte, Half>(file.GetTensorSpan("mx"));

        ReadOnlySpan<int> excIdx = default;
        ReadOnlySpan<ushort> excBits = default;
        if (file.TensorsByName.ContainsKey("exc_idx"))
        {
            excIdx = MemoryMarshal.Cast<byte, int>(file.GetTensorSpan("exc_idx"));
            excBits = MemoryMarshal.Cast<byte, ushort>(file.GetTensorSpan("exc_bits"));
        }

        Mach1AffineEmbedCodec.Decode(qPacked, mn, mx, rows, hid, bits, resolvedGroup, excIdx, excBits, dest);
    }
}
