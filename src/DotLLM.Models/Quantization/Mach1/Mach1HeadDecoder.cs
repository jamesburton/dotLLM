// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Assembles the LM head from its row-chunked int5-g64 files
/// (<c>packed/head/head_c{0..7}of8.safetensors</c>). Chunk keys are
/// <c>LMHEADCHUNK:{r0}:{r1}|{qp,gscale}</c> (+ optional
/// <c>|prot_rows</c>/<c>|prot_dense</c>); each file's <c>dims</c> metadata
/// gives the chunk's <c>[rows, n]</c>. Mirrors decode.py's <c>decode_head</c>.
/// </summary>
public static class Mach1HeadDecoder
{
    /// <summary>
    /// Decodes one chunk file's tensor(s) into the corresponding row range
    /// of <paramref name="dest"/> (a dense <c>[vocab, hid]</c> buffer the
    /// caller owns and has sized for the whole head).
    /// </summary>
    /// <param name="chunkFile">One already-open <c>head_c{c}of8.safetensors</c> file.</param>
    /// <param name="dest">The full <c>[vocab, hid]</c> destination buffer.</param>
    /// <param name="hid">Hidden size (<c>n</c> — must match every chunk's declared width).</param>
    public static void DecodeChunkInto(ISafetensorsTensorSource chunkFile, Span<float> dest, int hid)
    {
        if (chunkFile is not SafetensorsFile sf)
            throw new NotSupportedException("Mach1HeadDecoder requires metadata access (SafetensorsFile).");

        int group = sf.Metadata.TryGetValue("group", out string? groupStr) ? int.Parse(groupStr) : 64;
        string dimsJson = sf.Metadata["dims"];

        using JsonDocument dimsDoc = JsonDocument.Parse(dimsJson);
        foreach (JsonProperty p in dimsDoc.RootElement.EnumerateObject())
        {
            string name = p.Name; // "LMHEADCHUNK:{r0}:{r1}"
            int[] dims = p.Value.EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int rows = dims[0], n0 = dims[1];
            if (n0 != hid)
                throw new InvalidDataException($"Head chunk '{name}' declares n0={n0}, expected hid={hid}.");

            string[] parts = name.Split(':');
            int r0 = int.Parse(parts[1]);

            var qp = sf.GetTensorSpan($"{name}|qp");
            var gscaleHalf = MemoryMarshal.Cast<byte, Half>(sf.GetTensorSpan($"{name}|gscale"));
            float[] gscaleF = new float[gscaleHalf.Length];
            for (int i = 0; i < gscaleHalf.Length; i++)
                gscaleF[i] = (float)gscaleHalf[i];

            ReadOnlySpan<int> protRows = default;
            ReadOnlySpan<float> protDense = default;
            int[]? protRowsArr = null;
            float[]? protDenseArr = null;
            if (sf.TensorsByName.ContainsKey($"{name}|prot_rows"))
            {
                var protRowsI32 = MemoryMarshal.Cast<byte, int>(sf.GetTensorSpan($"{name}|prot_rows"));
                protRowsArr = protRowsI32.ToArray();
                var protDenseDesc = sf.TensorsByName[$"{name}|prot_dense"];
                var protDenseBytes = sf.GetTensorSpan($"{name}|prot_dense");
                protDenseArr = protDenseDesc.DType switch
                {
                    SafetensorsDType.F32 => MemoryMarshal.Cast<byte, float>(protDenseBytes).ToArray(),
                    SafetensorsDType.F16 => ToFloatArray(MemoryMarshal.Cast<byte, Half>(protDenseBytes)),
                    _ => throw new NotSupportedException($"Unsupported prot_dense dtype {protDenseDesc.DType}."),
                };
                protRows = protRowsArr;
                protDense = protDenseArr;
            }

            Span<float> chunkDest = dest.Slice(r0 * hid, rows * hid);
            Mach1Int5G64Codec.Decode(qp, gscaleF, rows, n0, group, protRows, protDense, chunkDest);
        }
    }

    private static float[] ToFloatArray(ReadOnlySpan<Half> half)
    {
        var arr = new float[half.Length];
        for (int i = 0; i < half.Length; i++)
            arr[i] = (float)half[i];
        return arr;
    }
}
