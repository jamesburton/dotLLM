// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Buffers;
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Decodes the older, per-expert (non-chunked) expert container: each
/// projection has its own <c>e{expert}.{proj}.trellis</c> tensor plus either
/// continuous fp16 <c>su</c>/<c>sv</c> ("K2" experts, Wscale absorbed into
/// <c>sv</c>) or int8-sign <c>SU</c>/<c>SV</c> + scalar <c>Wscale</c>
/// ("K1"/"demoted" experts), optionally with a shared cold-expert low-rank
/// basis residual (<c>basis.{proj}.A</c>/<c>B</c> shared across experts,
/// <c>e{e}.{proj}.c</c> per demoted expert). Discriminator: the file
/// declares a <c>manifest</c> metadata key (no <c>wave_gamma</c> field) —
/// see <see cref="Mach1ExpertLayerDecoderV3T.IsV3TContainer"/> for the v3t
/// discriminator this complements. Mirrors decode.py's
/// <c>decode_expert_layer</c> (its non-v3t branch).
/// </summary>
/// <remarks>
/// <b>Validation status:</b> this container is not exercised by any sample
/// currently in the <c>SyzygyResearch/Mach-1-Additive-35B</c> repo (every
/// <c>packed/experts/L*.safetensors</c> file there is the v3t chunked
/// container) — it is structurally ported from decode.py's manifest-driven
/// branch and dispatched correctly, but has no golden tensor to check
/// bit-exactness against. Treat it as unverified until a v2-container
/// sample surfaces.
/// </remarks>
public sealed class Mach1ExpertLayerDecoderV2
{
    private readonly ISafetensorsTensorSource _layerFile;
    private readonly ISafetensorsTensorSource _codebookFile;
    private readonly ExpertManifest _manifest;
    private readonly float[] _fullLut;

    /// <summary>
    /// Creates a decoder for one already-open layer file
    /// (<c>packed/experts/L{LL}.safetensors</c>) plus the shared codebook file.
    /// </summary>
    public Mach1ExpertLayerDecoderV2(ISafetensorsTensorSource layerFile, ISafetensorsTensorSource codebookFile)
    {
        _layerFile = layerFile;
        _codebookFile = codebookFile;

        string manifestJson = GetMetadata(layerFile, "manifest");
        _manifest = ExpertManifest.Parse(manifestJson);

        var tlutBytes = codebookFile.GetTensorSpan("tlut");
        var tlutFloats = MemoryMarshal.Cast<byte, float>(tlutBytes);
        _fullLut = Mach1LutCache.GetOrExpand(tlutFloats, _manifest.Cb2.V, _manifest.Cb2.L, _manifest.Cb2.TlutBits);
    }

    /// <summary>
    /// Returns <c>true</c> if the given metadata dictionary describes the v2
    /// manifest-driven container (a <c>manifest</c> key is present).
    /// </summary>
    public static bool IsV2Container(IReadOnlyDictionary<string, string> metadata) =>
        metadata.ContainsKey("manifest");

    /// <summary>
    /// Decodes one expert's one projection to dense <c>[m0, n0]</c> fp32.
    /// </summary>
    public void DecodeExpertProjection(int expertIndex, string proj, Span<float> dest)
    {
        (int m0, int n0) = _manifest.Geom[proj];
        bool demoted = _manifest.Demoted.Contains(expertIndex);
        Mach1CbParams cb = demoted ? _manifest.Cb1 : _manifest.Cb2;

        string trellisKey = $"e{expertIndex}.{proj}.trellis";
        var trellisDesc = _layerFile.TensorsByName[trellisKey]; // [ntiles, wordsPerTile]
        int ntiles = trellisDesc.Shape[0];
        int wordsPerTile = trellisDesc.Shape[1];
        var trellisWords = MemoryMarshal.Cast<byte, ushort>(_layerFile.GetTensorSpan(trellisKey));

        int m = Mach1Padding.PadToPowerOfTwo(m0);
        int n = Mach1Padding.PadToPowerOfTwo(n0);

        float[] suF = ArrayPool<float>.Shared.Rent(n);
        float[] svF = ArrayPool<float>.Shared.Rent(m);
        float? wscale = null;
        try
        {
            if (demoted)
            {
                var suI8 = MemoryMarshal.Cast<byte, sbyte>(_layerFile.GetTensorSpan($"e{expertIndex}.{proj}.SU"));
                var svI8 = MemoryMarshal.Cast<byte, sbyte>(_layerFile.GetTensorSpan($"e{expertIndex}.{proj}.SV"));
                for (int i = 0; i < n; i++) suF[i] = suI8[i];
                for (int i = 0; i < m; i++) svF[i] = svI8[i];
                wscale = ReadScalarFloat(_layerFile, $"e{expertIndex}.{proj}.Wscale");
            }
            else
            {
                var suHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan($"e{expertIndex}.{proj}.su"));
                var svHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan($"e{expertIndex}.{proj}.sv"));
                for (int i = 0; i < n; i++) suF[i] = (float)suHalf[i];
                for (int i = 0; i < m; i++) svF[i] = (float)svHalf[i];
            }

            Mach1TrellisWeightDecoder.Decode(
                trellisWords, wordsPerTile,
                suF.AsSpan(0, n), svF.AsSpan(0, m),
                _fullLut, m0, n0, cb,
                wscale: wscale,
                waveGamma: default,
                dest: dest);

            if (demoted && _manifest.Basis != null && _manifest.Basis.TryGetValue(proj, out var basisDims))
            {
                AddLowRankResidual(expertIndex, proj, m0, n0, basisDims.R, dest);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(suF);
            ArrayPool<float>.Shared.Return(svF);
        }
    }

    /// <summary>
    /// Adds the shared cold-expert low-rank basis residual <c>rs[m,n] =
    /// sum_r B[m,r]*c[r]*A[r,n]</c>, fp32, computed and added BEFORE any
    /// further precision reduction — the op order is part of the format.
    /// </summary>
    private void AddLowRankResidual(int expertIndex, string proj, int m0, int n0, int r, Span<float> dest)
    {
        var aHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan($"basis.{proj}.A")); // [r, n0]
        var bHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan($"basis.{proj}.B")); // [m0, r]
        var cHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan($"e{expertIndex}.{proj}.c")); // [r]

        for (int i = 0; i < m0; i++)
        {
            int destRowBase = i * n0;
            for (int j = 0; j < n0; j++)
            {
                float acc = 0f;
                for (int k = 0; k < r; k++)
                {
                    float bik = (float)bHalf[i * r + k];
                    float ck = (float)cHalf[k];
                    float akj = (float)aHalf[k * n0 + j];
                    acc += bik * ck * akj;
                }
                dest[destRowBase + j] += acc;
            }
        }
    }

    private static string GetMetadata(ISafetensorsTensorSource file, string key)
    {
        if (file is SafetensorsFile sf && sf.Metadata.TryGetValue(key, out string? value))
            return value;
        throw new InvalidOperationException($"Safetensors source has no '{key}' metadata entry.");
    }

    private static float ReadScalarFloat(ISafetensorsTensorSource file, string name)
    {
        var desc = file.TensorsByName[name];
        var bytes = file.GetTensorSpan(name);
        return desc.DType switch
        {
            SafetensorsDType.F32 => MemoryMarshal.Cast<byte, float>(bytes)[0],
            SafetensorsDType.F16 => (float)MemoryMarshal.Cast<byte, Half>(bytes)[0],
            _ => throw new NotSupportedException($"Unsupported scalar dtype {desc.DType} for '{name}'."),
        };
    }

    /// <summary>Parsed <c>manifest</c> metadata JSON for the v2 container.</summary>
    private sealed class ExpertManifest
    {
        public required Mach1CbParams Cb2 { get; init; }
        public required Mach1CbParams Cb1 { get; init; }
        public required HashSet<int> Demoted { get; init; }
        public required Dictionary<string, (int M, int N)> Geom { get; init; }
        public Dictionary<string, (int R, string Dtype)>? Basis { get; init; }

        public static ExpertManifest Parse(string json)
        {
            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;

            Mach1CbParams ParseCb(string key, Mach1CbParams fallback)
            {
                if (!root.TryGetProperty(key, out JsonElement el))
                    return fallback;
                return new Mach1CbParams(
                    K: el.GetProperty("K").GetDouble(),
                    L: el.GetProperty("L").GetInt32(),
                    V: el.GetProperty("V").GetInt32(),
                    TlutBits: el.GetProperty("tlut_bits").GetInt32(),
                    TdX: el.GetProperty("td_x").GetInt32(),
                    TdY: el.GetProperty("td_y").GetInt32());
            }

            // decode.py's CB2/CB1 defaults for the NE-style shared codebook.
            var cb2Default = new Mach1CbParams(2, 16, 2, 9, 16, 16);
            var cb1Default = new Mach1CbParams(1, 16, 2, 9, 16, 16);

            var demoted = new HashSet<int>();
            if (root.TryGetProperty("demoted", out JsonElement demotedEl))
                foreach (JsonElement e in demotedEl.EnumerateArray())
                    demoted.Add(e.GetInt32());

            var geom = new Dictionary<string, (int, int)>(StringComparer.Ordinal);
            foreach (JsonProperty p in root.GetProperty("geom").EnumerateObject())
            {
                var arr = p.Value.EnumerateArray().Select(e => e.GetInt32()).ToArray();
                geom[p.Name] = (arr[0], arr[1]);
            }

            Dictionary<string, (int, string)>? basis = null;
            if (root.TryGetProperty("basis", out JsonElement basisEl) && basisEl.ValueKind == JsonValueKind.Object)
            {
                int r = basisEl.GetProperty("r").GetInt32();
                string dtype = basisEl.TryGetProperty("dtype", out var dt) ? dt.GetString() ?? "fp16" : "fp16";
                basis = new Dictionary<string, (int, string)>(StringComparer.Ordinal);
                foreach (string proj in geom.Keys)
                    basis[proj] = (r, dtype);
            }

            return new ExpertManifest
            {
                Cb2 = ParseCb("cb2", cb2Default),
                Cb1 = ParseCb("cb1", cb1Default),
                Demoted = demoted,
                Geom = geom,
                Basis = basis,
            };
        }
    }
}
