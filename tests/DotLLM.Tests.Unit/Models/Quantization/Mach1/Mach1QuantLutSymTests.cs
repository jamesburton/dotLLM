using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Cross-checks <see cref="Mach1QuantLutSym.ExpandFullLut"/> against a full
/// <c>[2^L, V]</c> table captured directly from decode.py's own
/// <c>_np_full_lut</c> (L=8, tlutBits=5, V=4 — small enough to embed, large
/// enough that many states hit both the sign-flip bit and a variety of
/// hashed rows).
/// </summary>
public sealed class Mach1QuantLutSymTests
{
    [Fact]
    public void ExpandFullLut_MatchesDecodePyOracleExactly()
    {
        float[] expanded = Mach1QuantLutSym.ExpandFullLut(
            Mach1OracleTestVectors.SmallTlut,
            Mach1OracleTestVectors.QuantLutV,
            Mach1OracleTestVectors.QuantLutL,
            Mach1OracleTestVectors.QuantLutTlutBits);

        Assert.Equal(Mach1OracleTestVectors.FullLut.Length, expanded.Length);
        for (int i = 0; i < expanded.Length; i++)
            Assert.Equal(Mach1OracleTestVectors.FullLut[i], expanded[i]);
    }

    [Fact]
    public void ExpandFullLut_RejectsMismatchedSmallTableLength()
    {
        Assert.Throws<ArgumentException>(() =>
            Mach1QuantLutSym.ExpandFullLut(new float[10], v: 4, l: 8, tlutBits: 5));
    }

    [Fact]
    public void Mach1LutCache_ReturnsSameExpansionForSameInputs()
    {
        Mach1LutCache.Clear();
        float[] first = Mach1LutCache.GetOrExpand(
            Mach1OracleTestVectors.SmallTlut,
            Mach1OracleTestVectors.QuantLutV,
            Mach1OracleTestVectors.QuantLutL,
            Mach1OracleTestVectors.QuantLutTlutBits);
        float[] second = Mach1LutCache.GetOrExpand(
            Mach1OracleTestVectors.SmallTlut,
            Mach1OracleTestVectors.QuantLutV,
            Mach1OracleTestVectors.QuantLutL,
            Mach1OracleTestVectors.QuantLutTlutBits);

        Assert.Same(first, second);
    }
}
