// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Decodes the NE ("non-expert" spine — attention, linear-attn, shared
/// expert) tier: codec <c>canon_rht_bitshift_trellis_intlattice</c>
/// (<c>packed/ne/L00..L39.safetensors</c>, zero-padded, no manifest key).
/// Per-file metadata carries <c>cb_params</c> and <c>dims</c> (<c>name -&gt;
/// [m0, n0, m, n]</c>); keys are <c>&lt;tensor&gt;|{trellis,SU,SV,Wscale}</c>
/// with int8 sign SU/SV and a scalar Wscale. The tier-shared codebook lives
/// in <c>packed/ne/tlut.safetensors</c>. Mirrors decode.py's
/// <c>decode_ne_shard_canon</c>.
/// </summary>
public sealed class Mach1NeSpineDecoder
{
    private readonly ISafetensorsTensorSource _shardFile;
    private readonly Mach1CbParams _cb;
    private readonly float[] _fullLut;
    private readonly IReadOnlyDictionary<string, int[]> _dims;

    /// <summary>
    /// Creates a decoder for one already-open NE shard file
    /// (<c>packed/ne/L{LL}.safetensors</c>) plus the tier-shared tlut file
    /// (<c>packed/ne/tlut.safetensors</c>).
    /// </summary>
    public Mach1NeSpineDecoder(ISafetensorsTensorSource shardFile, ISafetensorsTensorSource tlutFile)
    {
        _shardFile = shardFile;

        string cbParamsJson = GetMetadata(shardFile, "cb_params");
        using (JsonDocument cbDoc = JsonDocument.Parse(cbParamsJson))
        {
            JsonElement root = cbDoc.RootElement;
            _cb = new Mach1CbParams(
                K: root.GetProperty("K").GetDouble(),
                L: root.GetProperty("L").GetInt32(),
                V: root.GetProperty("V").GetInt32(),
                TlutBits: root.GetProperty("tlut_bits").GetInt32(),
                TdX: root.GetProperty("td_x").GetInt32(),
                TdY: root.GetProperty("td_y").GetInt32());
        }

        string dimsJson = GetMetadata(shardFile, "dims");
        var dims = new Dictionary<string, int[]>(StringComparer.Ordinal);
        using (JsonDocument dimsDoc = JsonDocument.Parse(dimsJson))
        {
            foreach (JsonProperty p in dimsDoc.RootElement.EnumerateObject())
                dims[p.Name] = p.Value.EnumerateArray().Select(e => e.GetInt32()).ToArray();
        }
        _dims = dims;

        var tlutBytes = tlutFile.GetTensorSpan("tlut");
        var tlutFloats = MemoryMarshal.Cast<byte, float>(tlutBytes);
        _fullLut = Mach1LutCache.GetOrExpand(tlutFloats, _cb.V, _cb.L, _cb.TlutBits);
    }

    /// <summary>
    /// Returns <c>true</c> if the given metadata describes this shard's
    /// codec (no <c>manifest</c> key, and <c>codec ==
    /// "canon_rht_bitshift_trellis_intlattice"</c>).
    /// </summary>
    public static bool IsCanonRhtTrellisContainer(IReadOnlyDictionary<string, string> metadata) =>
        !metadata.ContainsKey("manifest") &&
        metadata.TryGetValue("codec", out string? codec) &&
        codec == "canon_rht_bitshift_trellis_intlattice";

    /// <summary>All tensor names declared in this shard's <c>dims</c> metadata.</summary>
    public IEnumerable<string> TensorNames => _dims.Keys;

    /// <summary>
    /// Decodes one named tensor in this shard to dense <c>[m0, n0]</c> fp32.
    /// </summary>
    public void DecodeTensor(string name, Span<float> dest)
    {
        int[] dim = _dims[name]; // [m0, n0, m, n]
        int m0 = dim[0], n0 = dim[1];

        var trellisDesc = _shardFile.TensorsByName[$"{name}|trellis"]; // [ntiles, wordsPerTile]
        int wordsPerTile = trellisDesc.Shape[1];
        var trellisWords = MemoryMarshal.Cast<byte, ushort>(_shardFile.GetTensorSpan($"{name}|trellis"));

        int m = Mach1Padding.PadToPowerOfTwo(m0);
        int n = Mach1Padding.PadToPowerOfTwo(n0);

        var suI8 = MemoryMarshal.Cast<byte, sbyte>(_shardFile.GetTensorSpan($"{name}|SU"));
        var svI8 = MemoryMarshal.Cast<byte, sbyte>(_shardFile.GetTensorSpan($"{name}|SV"));

        Span<float> suF = n <= 4096 ? stackalloc float[n] : new float[n];
        Span<float> svF = m <= 4096 ? stackalloc float[m] : new float[m];
        for (int i = 0; i < n; i++) suF[i] = suI8[i];
        for (int i = 0; i < m; i++) svF[i] = svI8[i];

        float wscale = ReadScalarFloat($"{name}|Wscale");

        Mach1TrellisWeightDecoder.Decode(
            trellisWords, wordsPerTile,
            suF, svF,
            _fullLut, m0, n0, _cb,
            wscale: wscale,
            waveGamma: default,
            dest: dest);
    }

    private float ReadScalarFloat(string tensorName)
    {
        var desc = _shardFile.TensorsByName[tensorName];
        var bytes = _shardFile.GetTensorSpan(tensorName);
        return desc.DType switch
        {
            SafetensorsDType.F32 => MemoryMarshal.Cast<byte, float>(bytes)[0],
            SafetensorsDType.F16 => (float)MemoryMarshal.Cast<byte, Half>(bytes)[0],
            _ => throw new NotSupportedException($"Unsupported scalar dtype {desc.DType} for '{tensorName}'."),
        };
    }

    private static string GetMetadata(ISafetensorsTensorSource file, string key)
    {
        if (file is SafetensorsFile sf && sf.Metadata.TryGetValue(key, out string? value))
            return value;
        throw new InvalidOperationException($"Safetensors source has no '{key}' metadata entry.");
    }
}
