// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder; Phase C: fused GEMV adds
// GemvExpertProjection alongside the original DecodeExpertProjection).
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
        var keys = ExpertProjectionKeys.Resolve(expertIndex, proj);
        var trellisDesc = _layerFile.TensorsByName[keys.TrellisKey]; // shape [ChunkSize, ntiles, wordsPerTile]
        int ntiles = trellisDesc.Shape[1];
        int wordsPerTile = trellisDesc.Shape[2];
        var trellisAll = MemoryMarshal.Cast<byte, ushort>(_layerFile.GetTensorSpan(keys.TrellisKey));
        var trellisForExpert = trellisAll.Slice(keys.Offset * ntiles * wordsPerTile, ntiles * wordsPerTile);

        var suDesc = _layerFile.TensorsByName[keys.SuKey]; // shape [ChunkSize, n]
        int nDim = suDesc.Shape[1];
        var suHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(keys.SuKey)).Slice(keys.Offset * nDim, nDim);

        var svDesc = _layerFile.TensorsByName[keys.SvKey]; // shape [ChunkSize, m]
        int mDim = svDesc.Shape[1];
        var svHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(keys.SvKey)).Slice(keys.Offset * mDim, mDim);

        float[] suF = ArrayPool<float>.Shared.Rent(nDim);
        float[] svF = ArrayPool<float>.Shared.Rent(mDim);
        float[]? gammaF = null;
        try
        {
            for (int i = 0; i < nDim; i++)
                suF[i] = (float)suHalf[i];
            for (int i = 0; i < mDim; i++)
                svF[i] = (float)svHalf[i];

            ReadOnlySpan<float> gammaSpan = ExtractGamma(keys, ref gammaF);

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

    /// <summary>
    /// Computes <c>y[m0] = W[m0, n0] * x[n0]</c> for one expert's one
    /// projection directly from the packed trellis stream, via
    /// <see cref="Mach1FusedExpertGemv"/> — the issue #266 Phase C fused path.
    /// Never materializes the dense <c>[m0, n0]</c> weight matrix that
    /// <see cref="DecodeExpertProjection"/> does. Key lookup mirrors
    /// <see cref="DecodeExpertProjection"/> exactly (same tensors, same
    /// su/sv/gamma extraction) so the two are directly A/B-comparable.
    /// </summary>
    /// <param name="expertIndex">Expert index in <c>[0, 256)</c>.</param>
    /// <param name="proj">Projection name: <c>"gate"</c>, <c>"up"</c>, or <c>"down"</c>.</param>
    /// <param name="m0">Unpadded row count for this projection.</param>
    /// <param name="n0">Unpadded column count for this projection.</param>
    /// <param name="x">Input activation, length <c>n0</c>.</param>
    /// <param name="y">Output, length <c>m0</c>.</param>
    public void GemvExpertProjection(int expertIndex, string proj, int m0, int n0, ReadOnlySpan<float> x, Span<float> y)
    {
        var keys = ExpertProjectionKeys.Resolve(expertIndex, proj);
        var trellisDesc = _layerFile.TensorsByName[keys.TrellisKey];
        int ntiles = trellisDesc.Shape[1];
        int wordsPerTile = trellisDesc.Shape[2];
        var trellisAll = MemoryMarshal.Cast<byte, ushort>(_layerFile.GetTensorSpan(keys.TrellisKey));
        var trellisForExpert = trellisAll.Slice(keys.Offset * ntiles * wordsPerTile, ntiles * wordsPerTile);

        var suDesc = _layerFile.TensorsByName[keys.SuKey];
        int nDim = suDesc.Shape[1];
        var suHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(keys.SuKey)).Slice(keys.Offset * nDim, nDim);

        var svDesc = _layerFile.TensorsByName[keys.SvKey];
        int mDim = svDesc.Shape[1];
        var svHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(keys.SvKey)).Slice(keys.Offset * mDim, mDim);

        float[] suF = ArrayPool<float>.Shared.Rent(nDim);
        float[] svF = ArrayPool<float>.Shared.Rent(mDim);
        float[]? gammaF = null;
        try
        {
            for (int i = 0; i < nDim; i++)
                suF[i] = (float)suHalf[i];
            for (int i = 0; i < mDim; i++)
                svF[i] = (float)svHalf[i];

            ReadOnlySpan<float> gammaSpan = ExtractGamma(keys, ref gammaF);

            Mach1FusedExpertGemv.Compute(
                trellisForExpert, wordsPerTile,
                suF.AsSpan(0, nDim), svF.AsSpan(0, mDim),
                _fullLut, m0, n0, _cb,
                wscale: null,
                waveGamma: gammaSpan,
                x: x, y: y);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(suF);
            ArrayPool<float>.Shared.Return(svF);
            if (gammaF != null)
                ArrayPool<float>.Shared.Return(gammaF);
        }
    }

    private ReadOnlySpan<float> ExtractGamma(in ExpertProjectionKeys keys, ref float[]? gammaF)
    {
        if (!_layerFile.TensorsByName.TryGetValue(keys.GammaKey, out var gammaDesc))
            return default;

        int gLen = gammaDesc.Shape[1];
        var gammaHalf = MemoryMarshal.Cast<byte, Half>(_layerFile.GetTensorSpan(keys.GammaKey)).Slice(keys.Offset * gLen, gLen);
        gammaF = ArrayPool<float>.Shared.Rent(gLen);
        for (int i = 0; i < gLen; i++)
            gammaF[i] = (float)gammaHalf[i];
        return gammaF.AsSpan(0, gLen);
    }

    private readonly record struct ExpertProjectionKeys(string TrellisKey, string SuKey, string SvKey, string GammaKey, int Offset)
    {
        public static ExpertProjectionKeys Resolve(int expertIndex, string proj)
        {
            if (expertIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(expertIndex));

            int c0 = (expertIndex / ChunkSize) * ChunkSize;
            int off = expertIndex % ChunkSize;
            return new ExpertProjectionKeys(
                $"e{c0}.{proj}.trellis", $"e{c0}.{proj}.su", $"e{c0}.{proj}.sv", $"e{c0}.{proj}.wave_gamma", off);
        }
    }
}
