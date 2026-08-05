// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Buffers;
using System.Runtime.InteropServices;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Decodes the chunked-container expert tier (<c>container:
/// "trained_susv_wave_gamma_chunked_v1"</c>, <c>codec:
/// "int_l1ball_trellis_onesided_wavenorm"</c>): 32-expert CHUNK-STACKED keys
/// <c>e{c0}.{proj}.{trellis|su|sv|wave_gamma}</c> with <c>c0 in
/// {0,32,...,224}</c>, continuous fp16 su/sv (Wscale absorbed into sv), and
/// a per-(expert, wavefront) gamma. Discriminator: the file's <c>fields</c>
/// metadata contains <c>"wave_gamma"</c>. Mirrors decode.py's
/// <c>decode_expert_v3t</c> / <c>decode_expert_layer_v3t</c>.
/// </summary>
public sealed class Mach1ExpertLayerDecoderV3T
{
    /// <summary>Experts stacked per chunk file entry (<c>e{c0}...</c> spans this many experts).</summary>
    public const int ChunkSize = 32;

    private readonly ISafetensorsTensorSource _layerFile;
    private readonly float[] _fullLut;
    private readonly Mach1CbParams _cb;

    /// <summary>
    /// Creates a decoder for one already-open layer file
    /// (<c>packed/experts/L{LL}.safetensors</c>).
    /// </summary>
    /// <param name="layerFile">The open per-layer chunked-container file.</param>
    /// <param name="smallTlut">
    /// The persisted codebook table (from <c>packed/experts/codebook.safetensors</c>),
    /// row-major <c>[2^tlutBits, V]</c>.
    /// </param>
    /// <param name="cb">Codebook/trellis parameters (from <c>codec.json</c>'s <c>cb_params</c>).</param>
    public Mach1ExpertLayerDecoderV3T(ISafetensorsTensorSource layerFile, ReadOnlySpan<float> smallTlut, Mach1CbParams cb)
    {
        _layerFile = layerFile;
        _cb = cb;
        _fullLut = Mach1LutCache.GetOrExpand(smallTlut, cb.V, cb.L, cb.TlutBits);
    }

    /// <summary>
    /// Returns <c>true</c> if the given safetensors metadata dictionary
    /// describes a chunked v3t container (per decode.py's dispatch: the
    /// <c>fields</c> value contains <c>"wave_gamma"</c>).
    /// </summary>
    public static bool IsV3TContainer(IReadOnlyDictionary<string, string> metadata) =>
        metadata.TryGetValue("fields", out string? fields) && fields.Contains("wave_gamma", StringComparison.Ordinal);

    /// <summary>
    /// Decodes one expert's one projection (<c>"gate"</c>, <c>"up"</c>, or
    /// <c>"down"</c>) to dense <c>[m0, n0]</c> fp32.
    /// </summary>
    /// <param name="expertIndex">Expert index in <c>[0, 256)</c>.</param>
    /// <param name="proj">Projection name: <c>"gate"</c>, <c>"up"</c>, or <c>"down"</c>.</param>
    /// <param name="m0">Unpadded row count for this projection.</param>
    /// <param name="n0">Unpadded column count for this projection.</param>
    /// <param name="dest">Destination for the dense <c>[m0, n0]</c> result, row-major.</param>
    public void DecodeExpertProjection(int expertIndex, string proj, int m0, int n0, Span<float> dest)
    {
        if (expertIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(expertIndex));

        int c0 = (expertIndex / ChunkSize) * ChunkSize;
        int off = expertIndex % ChunkSize;

        string trellisKey = $"e{c0}.{proj}.trellis";
        string suKey = $"e{c0}.{proj}.su";
        string svKey = $"e{c0}.{proj}.sv";
        string gammaKey = $"e{c0}.{proj}.wave_gamma";

        var trellisDesc = _layerFile.TensorsByName[trellisKey]; // shape [ChunkSize, ntiles, wordsPerTile]
        int ntiles = trellisDesc.Shape[1];
        int wordsPerTile = trellisDesc.Shape[2];
        var trellisAll = MemoryMarshal.Cast<byte, ushort>(_layerFile.GetTensorSpan(trellisKey));
        var trellisForExpert = trellisAll.Slice(off * ntiles * wordsPerTile, ntiles * wordsPerTile);

        var suDesc = _layerFile.TensorsByName[suKey]; // shape [ChunkSize, n]
        int nDim = suDesc.Shape[1];
        var suAll = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(suKey));
        var suHalf = suAll.Slice(off * nDim, nDim);

        var svDesc = _layerFile.TensorsByName[svKey]; // shape [ChunkSize, m]
        int mDim = svDesc.Shape[1];
        var svAll = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(svKey));
        var svHalf = svAll.Slice(off * mDim, mDim);

        float[] suF = ArrayPool<float>.Shared.Rent(nDim);
        float[] svF = ArrayPool<float>.Shared.Rent(mDim);
        float[]? gammaF = null;
        try
        {
            for (int i = 0; i < nDim; i++)
                suF[i] = (float)suHalf[i];
            for (int i = 0; i < mDim; i++)
                svF[i] = (float)svHalf[i];

            ReadOnlySpan<float> gammaSpan = default;
            if (_layerFile.TensorsByName.TryGetValue(gammaKey, out var gammaDesc))
            {
                int gLen = gammaDesc.Shape[1];
                var gammaAll = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(gammaKey));
                var gammaHalf = gammaAll.Slice(off * gLen, gLen);
                gammaF = ArrayPool<float>.Shared.Rent(gLen);
                for (int i = 0; i < gLen; i++)
                    gammaF[i] = (float)gammaHalf[i];
                gammaSpan = gammaF.AsSpan(0, gLen);
            }

            Mach1TrellisWeightDecoder.Decode(
                trellisForExpert, wordsPerTile,
                suF.AsSpan(0, nDim), svF.AsSpan(0, mDim),
                _fullLut, m0, n0, _cb,
                wscale: null,
                waveGamma: gammaSpan,
                dest: dest);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(suF);
            ArrayPool<float>.Shared.Return(svF);
            if (gammaF != null)
                ArrayPool<float>.Shared.Return(gammaF);
        }
    }
}
