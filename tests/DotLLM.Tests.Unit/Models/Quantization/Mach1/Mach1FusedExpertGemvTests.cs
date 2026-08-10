using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Correctness tests for <see cref="Mach1FusedExpertGemv"/> (issue #266 Phase C)
/// against <see cref="Mach1TrellisWeightDecoder.Decode"/> (Phase A/B, already
/// bit-exact-validated against the vendor golden — see
/// <c>Mach1GoldenBitExactTests</c>) followed by a plain matrix-vector product.
/// These two paths compute the same mathematical quantity
/// (<c>y = W . x</c>) via different, non-bit-identical floating-point
/// summation orders (direct row.dot(x) vs. the Hadamard-transform-on-the-
/// activation-side reformulation), so agreement is checked to a numerical
/// tolerance rather than bit-exactly.
/// </summary>
/// <remarks>
/// Uses the same production trellis parameters as the real expert tier
/// (K=1.5, L=16, V=8, 16x16 tiles) but synthetic (random, seeded) trellis
/// bits/LUT/su/sv/gamma — no fixture required. Dimensions are deliberately
/// non-power-of-two and non-square with <c>Mb != Nb</c>
/// (<c>m0=20,n0=48</c> -&gt; padded <c>m=32 (Mb=2), n=64 (Nb=4)</c>), per
/// CLAUDE.md's cross-backend rule that degenerate/square shapes can hide
/// wave_index_map broadcast bugs.
/// </remarks>
public sealed class Mach1FusedExpertGemvTests
{
    private static readonly Mach1CbParams ProductionExpertCb = new(K: 1.5, L: 16, V: 8, TlutBits: 15, TdX: 16, TdY: 16);
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Mach1FusedExpertGemvTests(Xunit.Abstractions.ITestOutputHelper output) => _output = output;

    [Fact]
    public void Compute_WithWaveGamma_MatchesDenseDecodeThenMatVec()
    {
        AssertFusedMatchesDense(m0: 20, n0: 48, useGamma: true, seed: 1234);
    }

    [Fact]
    public void Compute_WithoutWaveGamma_MatchesDenseDecodeThenMatVec()
    {
        AssertFusedMatchesDense(m0: 20, n0: 48, useGamma: false, seed: 5678);
    }

    /// <summary>Single-tile edge case (m0=n0=16, already power-of-two, no padding).</summary>
    [Fact]
    public void Compute_SingleTile_MatchesDenseDecodeThenMatVec()
    {
        AssertFusedMatchesDense(m0: 16, n0: 16, useGamma: true, seed: 999);
    }

    /// <summary>
    /// A second non-square, non-power-of-two case with the row/column padding
    /// relationship flipped (m0 &gt; n0 here vs n0 &gt; m0 above) so a
    /// transpose-direction bug in either the fused GEMV or the reference
    /// can't cancel out by symmetry.
    /// </summary>
    [Fact]
    public void Compute_WideToTallProjection_MatchesDenseDecodeThenMatVec()
    {
        AssertFusedMatchesDense(m0: 48, n0: 20, useGamma: true, seed: 4242);
    }

    private void AssertFusedMatchesDense(int m0, int n0, bool useGamma, int seed)
    {
        var cb = ProductionExpertCb;
        var rng = new Random(seed);

        int m = Mach1Padding.PadToPowerOfTwo(m0);
        int n = Mach1Padding.PadToPowerOfTwo(n0);
        int td = cb.TdX;
        int mb = m / td, nb = n / td;
        int ntiles = mb * nb;
        int tileElemCount = cb.TileElementCount;
        (int stepBits, int tileBits) = Mach1TrellisCodec.ComputeRateBits(tileElemCount, cb.K, cb.V);
        Assert.True(stepBits <= cb.L);
        int wordsPerTile = tileBits / 16;
        Assert.Equal(tileBits, wordsPerTile * 16); // exact word count, as production cb_params guarantee

        ushort[] trellisWords = new ushort[ntiles * wordsPerTile];
        for (int i = 0; i < trellisWords.Length; i++)
            trellisWords[i] = (ushort)rng.Next(0, 65536);

        // Full [2^L, V] LUT: random but finite/bounded, mirrors the real
        // codec's "exact scaled integer lattice" (small magnitude) values.
        float[] fullLut = new float[(1 << cb.L) * cb.V];
        for (int i = 0; i < fullLut.Length; i++)
            fullLut[i] = (float)(rng.NextDouble() * 10.0 - 5.0);

        float[] su = new float[n];
        for (int i = 0; i < n; i++)
            su[i] = (float)(rng.NextDouble() * 1.0 + 0.5);
        float[] sv = new float[m];
        for (int i = 0; i < m; i++)
            sv[i] = (float)(rng.NextDouble() * 1.0 + 0.5);

        float[] gamma = Array.Empty<float>();
        if (useGamma)
        {
            gamma = new float[mb + nb];
            for (int i = 0; i < gamma.Length; i++)
                gamma[i] = (float)(rng.NextDouble() * 1.0 + 0.5);
        }

        float[] x = new float[n0];
        for (int i = 0; i < n0; i++)
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // Reference: dense decode (already golden-validated primitive) + plain matvec.
        float[] w = new float[m0 * n0];
        Mach1TrellisWeightDecoder.Decode(
            trellisWords, wordsPerTile, su, sv, fullLut, m0, n0, cb,
            wscale: null, waveGamma: gamma, dest: w);

        float[] yRef = new float[m0];
        for (int i = 0; i < m0; i++)
        {
            float acc = 0f;
            int rowBase = i * n0;
            for (int j = 0; j < n0; j++)
                acc += w[rowBase + j] * x[j];
            yRef[i] = acc;
        }

        // Fused: decode-in-the-loop GEMV, never materializes w.
        float[] yFused = new float[m0];
        Mach1FusedExpertGemv.Compute(
            trellisWords, wordsPerTile, su, sv, fullLut, m0, n0, cb,
            wscale: null, waveGamma: gamma, x: x, y: yFused);

        double maxAbsErr = 0, maxRelErr = 0;
        for (int i = 0; i < m0; i++)
        {
            double diff = Math.Abs(yFused[i] - yRef[i]);
            double rel = diff / (Math.Abs((double)yRef[i]) + 1e-6);
            maxAbsErr = Math.Max(maxAbsErr, diff);
            maxRelErr = Math.Max(maxRelErr, rel);
        }

        _output.WriteLine($"m0={m0} n0={n0} useGamma={useGamma}: maxAbsErr={maxAbsErr:E4} maxRelErr={maxRelErr:E4}");

        // Loose-but-discriminating tolerance: a real transpose/index/gamma
        // bug produces errors many orders of magnitude larger than fp32
        // reassociation noise over a few hundred terms.
        Assert.True(maxRelErr < 1e-2 || maxAbsErr < 1e-2,
            $"Fused GEMV diverged from dense-decode reference: maxAbsErr={maxAbsErr:E4}, maxRelErr={maxRelErr:E4}");
    }
}
