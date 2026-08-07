using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Tests for <see cref="Mach1WaveGamma"/> at a deliberately non-square grid
/// (Mb=3, Nb=5 — Mb != Nb) so a row/column swap or an off-by-one in the
/// wavefront schedule cannot cancel itself out, per the issue's own warning
/// that this grid's top-right-corner duplication is an "off-by-one trap
/// that corrupts a whole tile row".
/// </summary>
public sealed class Mach1WaveGammaTests
{
    /// <summary>
    /// Hand-traced for Mb=3, Nb=5. <c>starts</c> =
    /// [(2,4),(1,4),(0,4), (0,4),(0,3),(0,2),(0,1),(0,0)] (8 entries; the
    /// first group's last start and the second group's first start are both
    /// (0,4) — the deliberate corner duplication). Tracing each anti-diagonal
    /// (jm++, jn--) from each start and keeping the LAST writer per tile
    /// gives, row-major (Mb=3 rows, Nb=5 cols):
    /// <code>
    /// row0: [7,6,5,4,3]
    /// row1: [6,5,4,3,1]
    /// row2: [5,4,3,1,0]
    /// </code>
    /// </summary>
    [Fact]
    public void BuildWaveIndexMap_HandTracedNonSquareGrid()
    {
        int[] idx = Mach1WaveGamma.BuildWaveIndexMap(mb: 3, nb: 5);

        int[] expected =
        {
            7, 6, 5, 4, 3,
            6, 5, 4, 3, 1,
            5, 4, 3, 1, 0,
        };
        Assert.Equal(expected, idx);
    }

    [Fact]
    public void BuildWaveIndexMap_TopRightCornerIsOverwrittenBySecondGroup()
    {
        // (0, Nb-1) is written once by the first group's last start (w=Mb-1)
        // and again by the second group's first start (w=Mb) - the LATER
        // write must win, i.e. idx[0, Nb-1] == Mb, not Mb-1.
        const int mb = 4, nb = 6;
        int[] idx = Mach1WaveGamma.BuildWaveIndexMap(mb, nb);
        Assert.Equal(mb, idx[0 * nb + (nb - 1)]);
    }

    [Fact]
    public void Apply_ScalesEachTileByItsOwnWavefrontGamma()
    {
        // 2x3 tile grid (Mb=2, Nb=3, Mb != Nb), td=2 -> m=4, n=6.
        const int mb = 2, nb = 3, td = 2;
        const int m = mb * td, n = nb * td;
        var weights = new float[m * n];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = 1f;

        int[] waveIndex = Mach1WaveGamma.BuildWaveIndexMap(mb, nb);
        int waveCount = mb + nb;
        var gamma = new float[waveCount];
        for (int w = 0; w < waveCount; w++)
            gamma[w] = 10f + w; // distinguishable per-wavefront value

        Mach1WaveGamma.Apply(weights, m, n, gamma, td);

        for (int bm = 0; bm < mb; bm++)
        {
            for (int bn = 0; bn < nb; bn++)
            {
                float expected = gamma[waveIndex[bm * nb + bn]];
                for (int ti = 0; ti < td; ti++)
                    for (int tj = 0; tj < td; tj++)
                        Assert.Equal(expected, weights[(bm * td + ti) * n + (bn * td + tj)]);
            }
        }
    }

    [Fact]
    public void Apply_NonDivisibleDim_Throws()
    {
        var weights = new float[5 * 16];
        Assert.Throws<ArgumentException>(() =>
            Mach1WaveGamma.Apply(weights, m: 5, n: 16, gamma: new float[10], td: 16));
    }
}
