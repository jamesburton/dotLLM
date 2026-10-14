// Fused additive expert GEMV for issue #266 Phase C. Builds on the Phase A codec
// decoder ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache
// License 2.0): https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md).
using System.Buffers;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Computes <c>y = W * x</c> for one Mach-1 additive-codec expert projection
/// directly from the packed trellis stream, WITHOUT ever materializing the
/// dense <c>[m0, n0]</c> weight matrix that <see cref="Mach1TrellisWeightDecoder.Decode"/>
/// produces. This is the issue #266 Phase C "fused additive expert GEMV":
/// the memory- and compute-relevant deliverable of the whole codec, since the
/// routed experts are 6.21 of the model's 7.53 GB and can never all be kept
/// resident as dense fp32/bf16 (256 experts x 40 layers x 3 projections dense
/// is ~64 GB, see issue #266's "Why the full implementation is the bar").
/// </summary>
/// <remarks>
/// <para><b>Derivation.</b> <see cref="Mach1TrellisWeightDecoder.Decode"/> builds
/// the dense weight as (op order per <c>codec.json</c>):
/// <c>W = diag(sv) . H_m . [wave_gamma ⊙ Wunit] . H_n . diag(su)</c>
/// (crop to <c>[m0,n0]</c> omitted for clarity), where <c>Wunit</c> is the
/// trellis-decoded, fp16-round-tripped lattice value matrix and <c>H_n</c>/
/// <c>H_m</c> are orthonormal (hence symmetric) Walsh-Hadamard transforms.
/// For a GEMV <c>y = W . x</c>, associativity moves both Hadamard transforms
/// and <c>diag(su)</c> onto the activation side instead of materializing them
/// into the weight:
/// <code>
/// y = diag(sv) . H_m . [wave_gamma ⊙ Wunit] . (H_n . diag(su) . x)
///   = diag(sv) . H_m . [wave_gamma ⊙ Wunit] . x'      where x' = H_n(su ⊙ x)
/// </code>
/// So: (1) pad+scale the activation by <c>su</c> and Hadamard-transform it —
/// <c>O(n log n)</c>, once per GEMV, size <c>n</c> not <c>m*n</c>; (2) decode
/// the trellis tile-by-tile and immediately multiply-accumulate against
/// <c>x'</c> instead of writing to a dense buffer (this is exactly
/// <c>codec.json</c>'s <c>kernel_contract.adds</c> — the vendor's own
/// documented intent for how a GEMV kernel should consume this format); (3)
/// Hadamard-transform the accumulated <c>[m]</c>-length result and scale by
/// <c>sv</c>, then crop to <c>m0</c>.
/// </para>
/// <para><b>Not yet done (perf, left for a follow-up pass):</b> the tile loop
/// here applies <c>wave_gamma</c> per element (matching
/// <see cref="Mach1TrellisWeightDecoder.Decode"/>'s elementwise multiply
/// order exactly, for the closest possible numerical agreement with the
/// dense-decode reference). <c>codec.json</c>'s <c>kernel_contract</c>
/// documents a cheaper reordering — accumulate each tile's raw
/// lattice-value.dot(x') partial sum first (add/subtract-dominated, since
/// <c>Wunit</c> is an exact integer lattice per <c>tlut.structure</c>), then
/// apply the tile's single <c>wave_gamma</c> multiply to the partial sum
/// ("~1 multiply per 16 weights") — algebraically identical by linearity but
/// not bit-for-bit identical in float, and not implemented in this pass. The
/// per-element accumulation loop below is also scalar (no SIMD); see
/// <c>docs/QUANTIZATION.md</c>'s Phase C section for what's measured and what
/// remains.
/// </para>
/// </remarks>
public static class Mach1FusedExpertGemv
{
    /// <summary>
    /// Computes <c>y[m0] = W[m0,n0] * x[n0]</c> for one trellis-coded,
    /// RHT-rotated weight matrix, fusing decode and matrix-vector multiply.
    /// </summary>
    /// <param name="trellisWords">
    /// This tensor's packed trellis bitstream, tile-major, <c>ntiles *
    /// wordsPerTile</c> long — identical input to
    /// <see cref="Mach1TrellisWeightDecoder.Decode"/>.
    /// </param>
    /// <param name="wordsPerTile">Packed <see cref="ushort"/> words per tile.</param>
    /// <param name="su">RHT sign/scale vector over the padded <c>n</c> dim.</param>
    /// <param name="sv">RHT sign/scale vector over the padded <c>m</c> dim.</param>
    /// <param name="fullLut">The expanded <c>[2^L, V]</c> LUT (see <see cref="Mach1QuantLutSym"/>).</param>
    /// <param name="m0">Unpadded row count.</param>
    /// <param name="n0">Unpadded column count.</param>
    /// <param name="cb">Codebook/trellis parameters. Requires square tiles (<c>TdX == TdY</c>).</param>
    /// <param name="wscale">
    /// Optional scalar weight scale (v2/spine's raw int8-sign packs;
    /// <c>null</c> for v3t's continuous su/sv, which already absorb it) —
    /// applied after the fp16 round-trip and before <c>wave_gamma</c>, same
    /// order as <see cref="Mach1TrellisWeightDecoder.Decode"/>.
    /// </param>
    /// <param name="waveGamma">
    /// Optional per-wavefront gamma vector, length <c>Mb+Nb</c> (v3t only;
    /// pass an empty span for tiers without it).
    /// </param>
    /// <param name="x">Input activation, length <c>n0</c>.</param>
    /// <param name="y">Output, length <c>m0</c>.</param>
    public static void Compute(
        ReadOnlySpan<ushort> trellisWords,
        int wordsPerTile,
        ReadOnlySpan<float> su,
        ReadOnlySpan<float> sv,
        ReadOnlySpan<float> fullLut,
        int m0, int n0,
        Mach1CbParams cb,
        float? wscale,
        ReadOnlySpan<float> waveGamma,
        ReadOnlySpan<float> x,
        Span<float> y)
    {
        if (cb.TdX != cb.TdY)
            throw new NotSupportedException(
                $"Mach1FusedExpertGemv requires square tiles (TdX==TdY), got TdX={cb.TdX} TdY={cb.TdY}.");

        int m = Mach1Padding.PadToPowerOfTwo(m0);
        int n = Mach1Padding.PadToPowerOfTwo(n0);
        if (su.Length != n)
            throw new ArgumentException($"su length {su.Length} must equal padded n={n}");
        if (sv.Length != m)
            throw new ArgumentException($"sv length {sv.Length} must equal padded m={m}");
        if (x.Length != n0)
            throw new ArgumentException($"x length {x.Length} must equal n0={n0}");
        if (y.Length != m0)
            throw new ArgumentException($"y length {y.Length} must equal m0={m0}");

        int td = cb.TdX;
        int mb = m / td;
        int nb = n / td;
        int ntiles = mb * nb;
        int tileElemCount = cb.TileElementCount;
        int stepsPerTile = tileElemCount / cb.V;

        if (trellisWords.Length != ntiles * wordsPerTile)
            throw new ArgumentException(
                $"trellisWords length {trellisWords.Length} must equal ntiles*wordsPerTile={ntiles * wordsPerTile}");

        int[]? waveIndex = null;
        if (!waveGamma.IsEmpty)
        {
            if (waveGamma.Length < mb + nb)
                throw new ArgumentException(
                    $"waveGamma length {waveGamma.Length} is shorter than the wavefront count Mb+Nb={mb + nb}");
            waveIndex = Mach1WaveGamma.BuildWaveIndexMap(mb, nb);
        }

        float[] xPrimeArr = ArrayPool<float>.Shared.Rent(n);
        float[] tempArr = ArrayPool<float>.Shared.Rent(m);
        int[] statesArr = ArrayPool<int>.Shared.Rent(stepsPerTile);
        try
        {
            // x' = H_n(su ⊙ x_padded) — activation-side transform, O(n log n),
            // computed once regardless of m.
            Span<float> xPrime = xPrimeArr.AsSpan(0, n);
            x.CopyTo(xPrime);
            xPrime.Slice(n0).Clear();
            for (int j = 0; j < n; j++)
                xPrime[j] *= su[j];
            Mach1WalshHadamard.TransformRowsInPlace(xPrime, rows: 1, dim: n);

            Span<float> temp = tempArr.AsSpan(0, m);
            temp.Clear();

            Span<int> states = statesArr.AsSpan(0, stepsPerTile);

            for (int tile = 0; tile < ntiles; tile++)
            {
                int tileRow = tile / nb;
                int tileCol = tile % nb;
                int rowBase = tileRow * td;
                int colBase = tileCol * td;

                ReadOnlySpan<ushort> tileWords = trellisWords.Slice(tile * wordsPerTile, wordsPerTile);
                Mach1TrellisCodec.UnpackTileStates(tileWords, tileElemCount, cb, states);

                float gammaTile = waveIndex is null ? 1f : waveGamma[waveIndex[tileRow * nb + tileCol]];

                // Decode this tile's V-wide lattice rows and immediately
                // multiply-accumulate against x' — the trellis value is never
                // written to a dense [m,n] buffer (issue #266 Phase C's core
                // memory/perf claim).
                for (int si = 0; si < stepsPerTile; si++)
                {
                    int lutOffset = states[si] * cb.V;
                    int eBase = si * cb.V;
                    for (int vc = 0; vc < cb.V; vc++)
                    {
                        int e = eBase + vc;
                        int localRow = e / td;
                        int localCol = e % td;

                        // Same fp16 round-trip point as the dense decoder
                        // (op_order: recons -> fp16 cast -> wave_gamma -> ...).
                        float w = (float)(Half)fullLut[lutOffset + vc];
                        if (wscale is float ws)
                            w *= ws;
                        w *= gammaTile;

                        temp[rowBase + localRow] += w * xPrime[colBase + localCol];
                    }
                }
            }

            // y = sv ⊙ H_m(temp), cropped to m0.
            Mach1WalshHadamard.TransformRowsInPlace(temp, rows: 1, dim: m);
            for (int i = 0; i < m0; i++)
                y[i] = temp[i] * sv[i];
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xPrimeArr);
            ArrayPool<float>.Shared.Return(tempArr);
            ArrayPool<int>.Shared.Return(statesArr);
        }
    }
}
