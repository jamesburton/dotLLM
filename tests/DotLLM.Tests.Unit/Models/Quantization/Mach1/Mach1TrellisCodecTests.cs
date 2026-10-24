using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Tests for the tail-biting trellis bit-unpack (<see cref="Mach1TrellisCodec"/>),
/// the primitive the issue flags as an "easy place to get subtly wrong in a
/// way that produces plausible-looking garbage rather than an error".
/// </summary>
public sealed class Mach1TrellisCodecTests
{
    /// <summary>
    /// Hand-computed example: L=4, K=1, V=1, T=4 (tileElementCount), so
    /// step=K*V=1, tileBits=K*T=4, wrapBits=L-step=3 (a non-trivial
    /// tail-biting wrap — most of the register's final state comes from bits
    /// wrapped from the start of the stream, not fresh bits).
    /// <para>
    /// Packed word = 0xB000 = 0b1011_0000_0000_0000. MSB-first, the first
    /// tileBits=4 bits are [1,0,1,1]; tail-biting appends the first
    /// wrapBits=3 of those again: bits=[1,0,1,1,1,0,1].
    /// </para>
    /// <para>
    /// seed reg = bits[0..4) = 1011b = 11. states[0]=11.
    /// step 1: fresh=bits[4]=1 -&gt; reg=((11&lt;&lt;1)&amp;15)|1=7. states[1]=7.
    /// step 2: fresh=bits[5]=0 -&gt; reg=((7&lt;&lt;1)&amp;15)|0=14. states[2]=14.
    /// step 3: fresh=bits[6]=1 -&gt; reg=((14&lt;&lt;1)&amp;15)|1=13. states[3]=13.
    /// </para>
    /// </summary>
    [Fact]
    public void UnpackTileStates_HandComputedTailBitingExample()
    {
        ushort[] words = { 0xB000 };
        var cb = new Mach1CbParams(K: 1, L: 4, V: 1, TlutBits: 0, TdX: 1, TdY: 1);
        Span<int> states = stackalloc int[4];

        Mach1TrellisCodec.UnpackTileStates(words, tileElementCount: 4, cb, states);

        Assert.Equal(new[] { 11, 7, 14, 13 }, states.ToArray());
    }

    /// <summary>
    /// A second hand-computed wrap example with a different word to make
    /// sure the first isn't an accidental match: word = 0x4000 =
    /// 0b0100_0000_0000_0000. First 4 bits = [0,1,0,0]; wrap appends first 3:
    /// [0,1,0,0,0,1,0].
    /// seed = 0100b = 4. states[0]=4.
    /// step1: fresh=bits[4]=0 -&gt; reg=((4&lt;&lt;1)&amp;15)|0=8. states[1]=8.
    /// step2: fresh=bits[5]=1 -&gt; reg=((8&lt;&lt;1)&amp;15)|1=1. states[2]=1.
    /// step3: fresh=bits[6]=0 -&gt; reg=((1&lt;&lt;1)&amp;15)|0=2. states[3]=2.
    /// </summary>
    [Fact]
    public void UnpackTileStates_SecondHandComputedTailBitingExample()
    {
        ushort[] words = { 0x4000 };
        var cb = new Mach1CbParams(K: 1, L: 4, V: 1, TlutBits: 0, TdX: 1, TdY: 1);
        Span<int> states = stackalloc int[4];

        Mach1TrellisCodec.UnpackTileStates(words, tileElementCount: 4, cb, states);

        Assert.Equal(new[] { 4, 8, 1, 2 }, states.ToArray());
    }

    [Fact]
    public void ComputeRateBits_FractionalK_ComputesWholeBitCounts()
    {
        // The v3t expert tier: K=1.5, V=8, T=256 -> step=12, tileBits=384.
        (int step, int tileBits) = Mach1TrellisCodec.ComputeRateBits(tileElementCount: 256, k: 1.5, v: 8);
        Assert.Equal(12, step);
        Assert.Equal(384, tileBits);
    }

    [Fact]
    public void ComputeRateBits_NonWholeBitRate_Throws()
    {
        Assert.Throws<ArgumentException>(() => Mach1TrellisCodec.ComputeRateBits(tileElementCount: 5, k: 1.5, v: 1));
    }

    [Fact]
    public void UnpackTileStates_StepExceedsRegisterWidth_Throws()
    {
        ushort[] words = { 0 };
        var cb = new Mach1CbParams(K: 8, L: 4, V: 1, TlutBits: 0, TdX: 1, TdY: 1); // step=8 > L=4
        Assert.Throws<ArgumentException>(() =>
            Mach1TrellisCodec.UnpackTileStates(words, tileElementCount: 4, cb, new int[4]));
    }

    /// <summary>
    /// Full-tile reconstruction with a NON-SQUARE, non-16x16 tile shape
    /// (tdX=2, tdY=4 -&gt; T=8) and a two-tile row-major grid (m=2 tiles tall,
    /// n=2 tiles wide -&gt; Mb=2, Nb=2 is square here on purpose, to isolate
    /// ReconstructWeights' own row/col addressing from the wave-map
    /// asymmetry already covered by <see cref="Mach1WaveGammaTests"/>).
    /// Verifies the within-tile row-major layout and the tile-grid row-major
    /// layout independently by placing distinguishable LUT rows at
    /// non-adjacent state indices.
    /// </summary>
    [Fact]
    public void ReconstructWeights_TileAndGridLayoutAreRowMajor()
    {
        const int tdX = 2, tdY = 4, v = 2;
        const int m = 4, n = 8; // 2x2 tile grid
        int stepsPerTile = (tdX * tdY) / v; // 4
        int ntiles = (m / tdX) * (n / tdY); // 4

        // Build a full LUT where state s maps to [s, s+1000] so each state's
        // output is trivially identifiable in the destination matrix.
        var fullLut = new float[64 * v];
        for (int s = 0; s < 64; s++)
        {
            fullLut[s * v + 0] = s;
            fullLut[s * v + 1] = s + 1000;
        }

        // States chosen so tile 0 uses states [0,1,2,3], tile 1 uses [10,11,12,13],
        // tile 2 uses [20,21,22,23], tile 3 uses [30,31,32,33].
        var states = new int[ntiles * stepsPerTile];
        for (int tile = 0; tile < ntiles; tile++)
            for (int si = 0; si < stepsPerTile; si++)
                states[tile * stepsPerTile + si] = tile * 10 + si;

        var dest = new float[m * n];
        Mach1TrellisCodec.ReconstructWeights(states, ntiles, m, n, tdX, tdY, v, fullLut, dest);

        // Tile 0 occupies rows [0,2) cols [0,4); tile 1 rows [0,2) cols [4,8);
        // tile 2 rows [2,4) cols [0,4); tile 3 rows [2,4) cols [4,8).
        // Within a tile, element e -> localRow=e/tdY, localCol=e%tdY; step si,
        // component vc -> e = si*v+vc.
        // Tile 0, si=0 (e=0..1 -> row0,col0..1): state 0 -> [0,1000] at (0,0),(0,1)
        Assert.Equal(0f, dest[0 * n + 0]);
        Assert.Equal(1000f, dest[0 * n + 1]);
        // si=1 (e=2..3 -> row0,col2..3): state 1
        Assert.Equal(1f, dest[0 * n + 2]);
        Assert.Equal(1001f, dest[0 * n + 3]);
        // si=2 (e=4..5 -> row1,col0..1): state 2
        Assert.Equal(2f, dest[1 * n + 0]);
        // Tile 1 (grid col 1) occupies columns [4,8): si=0 -> row0,col4..5 = state 10
        Assert.Equal(10f, dest[0 * n + 4]);
        // Tile 2 (grid row 1) occupies rows [2,4): si=0 -> row2,col0..1 = state 20
        Assert.Equal(20f, dest[2 * n + 0]);
        // Tile 3 occupies rows[2,4) cols[4,8): si=0 -> row2,col4..5 = state 30
        Assert.Equal(30f, dest[2 * n + 4]);
    }

    /// <summary>
    /// Full pipeline round-trip against decode.py's own <c>decode_expert_v3t</c>,
    /// at a deliberately non-power-of-two, Mb != Nb shape (m0=20, n0=50 pads
    /// to m=32, n=64 -&gt; Mb=2, Nb=4). This is the strongest available
    /// primitive-composition check for Phase A: the trellis words, su/sv,
    /// wave_gamma and the small tlut are random, but decode.py itself (not a
    /// re-derivation) computed <see cref="Mach1OracleTestVectors.PipelineExpectedResult"/>.
    /// </summary>
    [Fact]
    public void Decode_MatchesDecodePyOracle_AtNonPowerOfTwoMbNeNbShape()
    {
        var cb = new Mach1CbParams(
            K: Mach1OracleTestVectors.PipelineK,
            L: Mach1OracleTestVectors.QuantLutL,
            V: Mach1OracleTestVectors.QuantLutV,
            TlutBits: Mach1OracleTestVectors.QuantLutTlutBits,
            TdX: 16, TdY: 16);

        float[] fullLut = Mach1QuantLutSym.ExpandFullLut(
            Mach1OracleTestVectors.SmallTlut, cb.V, cb.L, cb.TlutBits);

        var dest = new float[Mach1OracleTestVectors.PipelineM0 * Mach1OracleTestVectors.PipelineN0];

        Mach1TrellisWeightDecoder.Decode(
            Mach1OracleTestVectors.PipelineTrellisWords,
            Mach1OracleTestVectors.PipelineWordsPerTile,
            Mach1OracleTestVectors.PipelineSu,
            Mach1OracleTestVectors.PipelineSv,
            fullLut,
            Mach1OracleTestVectors.PipelineM0,
            Mach1OracleTestVectors.PipelineN0,
            cb,
            wscale: null,
            waveGamma: Mach1OracleTestVectors.PipelineWaveGamma,
            dest: dest);

        Assert.Equal(Mach1OracleTestVectors.PipelineExpectedResult.Length, dest.Length);
        for (int i = 0; i < dest.Length; i++)
        {
            float expected = Mach1OracleTestVectors.PipelineExpectedResult[i];
            float actual = dest[i];
            // Reference values were captured via numpy's %.9g formatting, so
            // compare with a tight relative tolerance rather than requiring
            // exact text-roundtrip equality.
            float tol = Math.Max(1e-4f, Math.Abs(expected) * 1e-5f);
            Assert.True(
                Math.Abs(expected - actual) <= tol,
                $"index {i}: expected {expected}, got {actual} (tol {tol})");
        }
    }
}
