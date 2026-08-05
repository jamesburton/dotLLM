using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Tests for <see cref="Mach1Int5G64Codec"/> at a deliberately non-64-related
/// shape (m=5, n=24 — 3 blocks of 8, nothing tied to the real head's
/// group=64/hid=2048), cross-checked against decode.py's own
/// <c>pack_int5</c>/<c>unpack_int5</c>.
/// </summary>
public sealed class Mach1Int5G64CodecTests
{
    // m=5, n=24; generated via numpy's default_rng(0xBEEF).integers(-16, 16, ...)
    private static readonly sbyte[] Q =
    {
        -7,-5,11,7,4,2,5,8,-14,-5,12,1,-6,9,15,10,6,-11,-3,-5,11,-12,-12,-5,
        12,15,12,6,-7,0,-8,-5,-3,3,-12,-9,-6,9,12,-2,-14,9,0,5,8,7,-5,-10,
        -9,-6,-14,-11,-1,-1,-10,12,-14,-3,2,14,1,15,9,7,8,7,-2,-16,-13,12,-7,
        10,7,3,-5,-10,-13,11,-15,14,-7,-16,11,-2,-15,5,8,-11,2,12,-15,15,-8,
        -7,-13,-1,10,3,-7,-1,-16,-9,-16,-11,-5,6,2,-12,15,-3,-7,-4,-10,15,14,8,14,-13,3,0,
    };

    private static readonly byte[] Packed =
    {
        105,237,75,101,197,98,241,168,242,215,182,180,181,9,89,252,115,155,32,90,109,146,163,50,
        119,34,195,138,239,50,71,137,242,158,225,162,73,31,127,190,248,58,48,120,210,119,46,51,
        118,240,9,108,23,42,46,146,135,143,210,120,122,166,7,14,40,203,74,242,91,98,230,123,236,199,132,
    };

    private const int M = 5, N = 24;

    [Fact]
    public void Pack_MatchesDecodePyOracle()
    {
        var packed = new byte[M * (N / 8) * 5];
        Mach1Int5G64Codec.Pack(Q, M, N, packed);
        Assert.Equal(Packed, packed);
    }

    [Fact]
    public void Unpack_MatchesDecodePyOracle()
    {
        var q = new sbyte[M * N];
        Mach1Int5G64Codec.Unpack(Packed, M, N, q);
        Assert.Equal(Q, q);
    }

    [Fact]
    public void PackThenUnpack_RoundTrips()
    {
        var packed = new byte[M * (N / 8) * 5];
        Mach1Int5G64Codec.Pack(Q, M, N, packed);
        var back = new sbyte[M * N];
        Mach1Int5G64Codec.Unpack(packed, M, N, back);
        Assert.Equal(Q, back);
    }

    [Fact]
    public void Decode_AppliesPerGroupScaleAndProtectedRowOverride()
    {
        // group=8 (smaller than the shipped 64, to keep this test small);
        // n=24 -> 3 groups per row.
        var packed = new byte[M * (N / 8) * 5];
        Mach1Int5G64Codec.Pack(Q, M, N, packed);

        var gscale = new float[M * (N / 8)]; // ng = ceil(24/8) = 3
        for (int i = 0; i < gscale.Length; i++)
            gscale[i] = 0.5f + 0.1f * i;

        var dest = new float[M * N];
        Mach1Int5G64Codec.Decode(packed, gscale, M, N, group: 8, protRows: default, protDense: default, dest);

        for (int r = 0; r < M; r++)
        {
            for (int j = 0; j < N; j++)
            {
                float expected = Q[r * N + j] * gscale[r * 3 + j / 8];
                Assert.Equal(expected, dest[r * N + j]);
            }
        }

        // Now overwrite row 2 with exact protected values.
        var protDense = new float[N];
        for (int j = 0; j < N; j++)
            protDense[j] = 999f + j;
        Mach1Int5G64Codec.Decode(packed, gscale, M, N, group: 8, protRows: new[] { 2 }, protDense: protDense, dest);
        for (int j = 0; j < N; j++)
            Assert.Equal(999f + j, dest[2 * N + j]);
    }
}
