// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Buffers;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Shared trellis-to-dense reconstruction used by every tier that carries a
/// two-sided randomized-Hadamard-transform (RHT) weight: the v3t chunked
/// expert container (<see cref="Mach1ExpertLayerDecoderV3T"/>), the v2/manifest
/// expert container, and the NE spine tier. Mirrors decode.py's
/// <c>decode_trellis</c> / <c>decode_expert_v3t</c>.
/// </summary>
/// <remarks>
/// Op order (part of the on-disk format — fp32 add/sub/mul/div are
/// IEEE-exact, so following the reference's exact order reproduces its
/// decode bit-for-bit):
/// <c>unpack -&gt; recons -&gt; fp16 round-trip -&gt; [wave_gamma] -&gt; [x wscale]
/// -&gt; H_n (WHT over n) -&gt; x su -&gt; transpose -&gt; H_m (WHT over m) -&gt; x sv
/// -&gt; transpose -&gt; crop</c>.
/// <c>wave_gamma</c> (v3t only) and <c>wscale</c> (v2/spine only) never both
/// apply to the same tensor in the real format, so their relative order is
/// unobserved by any golden fixture; this port applies wave_gamma first to
/// match <c>codec.json</c>'s explicit v3t <c>op_order</c> string.
/// </remarks>
public static class Mach1TrellisWeightDecoder
{
    /// <summary>
    /// Decodes one trellis-coded, RHT-rotated weight matrix to dense
    /// <c>[m0, n0]</c> fp32.
    /// </summary>
    /// <param name="trellisWords">
    /// This tensor's packed trellis bitstream, tile-major,
    /// <c>ntiles * wordsPerTile</c> long.
    /// </param>
    /// <param name="wordsPerTile">Packed <see cref="ushort"/> words per tile.</param>
    /// <param name="su">RHT sign/scale vector over the padded <c>n</c> dim.</param>
    /// <param name="sv">RHT sign/scale vector over the padded <c>m</c> dim.</param>
    /// <param name="fullLut">The expanded <c>[2^L, V]</c> LUT (see <see cref="Mach1QuantLutSym"/>).</param>
    /// <param name="m0">Unpadded row count.</param>
    /// <param name="n0">Unpadded column count.</param>
    /// <param name="cb">Codebook/trellis parameters.</param>
    /// <param name="wscale">
    /// Optional scalar weight scale, multiplied in right after the fp16
    /// round-trip (v2/spine's raw int8-sign packs; <c>null</c> for v3t's
    /// continuous su/sv, which already absorb it).
    /// </param>
    /// <param name="waveGamma">
    /// Optional per-wavefront gamma vector, length <c>Mb+Nb</c> (v3t only;
    /// pass an empty span for tiers without it).
    /// </param>
    /// <param name="dest">Destination for the dense <c>[m0, n0]</c> result, row-major.</param>
    public static void Decode(
        ReadOnlySpan<ushort> trellisWords,
        int wordsPerTile,
        ReadOnlySpan<float> su,
        ReadOnlySpan<float> sv,
        ReadOnlySpan<float> fullLut,
        int m0, int n0,
        Mach1CbParams cb,
        float? wscale,
        ReadOnlySpan<float> waveGamma,
        Span<float> dest)
    {
        int m = Mach1Padding.PadToPowerOfTwo(m0);
        int n = Mach1Padding.PadToPowerOfTwo(n0);
        if (su.Length != n)
            throw new ArgumentException($"su length {su.Length} must equal padded n={n}");
        if (sv.Length != m)
            throw new ArgumentException($"sv length {sv.Length} must equal padded m={m}");
        if (dest.Length != m0 * n0)
            throw new ArgumentException($"dest length {dest.Length} must equal m0*n0={m0 * n0}");

        int tileElemCount = cb.TileElementCount;
        int ntiles = (m / cb.TdX) * (n / cb.TdY);
        int stepsPerTile = tileElemCount / cb.V;

        int[] statesArr = ArrayPool<int>.Shared.Rent(ntiles * stepsPerTile);
        float[] unitArr = ArrayPool<float>.Shared.Rent(m * n);
        float[] transposedArr = ArrayPool<float>.Shared.Rent(n * m);
        try
        {
            Span<int> states = statesArr.AsSpan(0, ntiles * stepsPerTile);
            Mach1TrellisCodec.UnpackAllTiles(trellisWords, ntiles, wordsPerTile, tileElemCount, cb, states);

            Span<float> unit = unitArr.AsSpan(0, m * n);
            Mach1TrellisCodec.ReconstructWeights(states, ntiles, m, n, cb.TdX, cb.TdY, cb.V, fullLut, unit);

            // The reference casts the reconstructed lattice to fp16 and back
            // BEFORE anything else touches it — this materially changes values.
            Mach1NumericUtil.RoundTripThroughHalf(unit);

            if (!waveGamma.IsEmpty)
                Mach1WaveGamma.Apply(unit, m, n, waveGamma, cb.TdX);

            if (wscale is float ws)
            {
                for (int i = 0; i < unit.Length; i++)
                    unit[i] *= ws;
            }

            // Row side: WHT over n, then * su (broadcast down each column).
            Mach1WalshHadamard.TransformRowsInPlace(unit, m, n);
            for (int i = 0; i < m; i++)
            {
                int rowBase = i * n;
                for (int j = 0; j < n; j++)
                    unit[rowBase + j] *= su[j];
            }

            // Transpose to [n, m].
            Span<float> transposed = transposedArr.AsSpan(0, n * m);
            for (int i = 0; i < m; i++)
            {
                int rowBase = i * n;
                for (int j = 0; j < n; j++)
                    transposed[j * m + i] = unit[rowBase + j];
            }

            // Col side: WHT over m, then * sv (broadcast across each
            // transposed "row", i.e. down each original column).
            Mach1WalshHadamard.TransformRowsInPlace(transposed, n, m);
            for (int j = 0; j < n; j++)
            {
                int rowBase = j * m;
                for (int i = 0; i < m; i++)
                    transposed[rowBase + i] *= sv[i];
            }

            // Transpose back and crop to [m0, n0] in one pass.
            for (int i = 0; i < m0; i++)
            {
                int destRowBase = i * n0;
                for (int j = 0; j < n0; j++)
                    dest[destRowBase + j] = transposed[j * m + i];
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(statesArr);
            ArrayPool<float>.Shared.Return(unitArr);
            ArrayPool<float>.Shared.Return(transposedArr);
        }
    }
}
