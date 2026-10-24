using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Tests for <see cref="Mach1AffineEmbedCodec"/>, cross-checked against
/// decode.py's own <c>pack_embed_q</c>/<c>unpack_embed_q</c> at shapes that
/// deliberately do NOT byte-align every row (bits=4, hid=13 -&gt; 52 bits ->
/// 7 bytes, 4 bits of padding per row) and a genuinely sub-byte-per-element
/// case (bits=3, which never nibble-aligns).
/// </summary>
public sealed class Mach1AffineEmbedCodecTests
{
    // bits=4, rows=3, hid=13 (52 bits/row -> 7 bytes/row, non-byte-aligned)
    private static readonly byte[] Codes4 =
    {
        15,7,13,9,2,0,3,13,12,9,15,13,3,
        1,1,5,12,8,7,10,14,7,7,14,15,0,
        15,1,0,3,11,3,14,0,7,8,6,10,6,
    };

    private static readonly byte[] Packed4 =
    {
        247,217,32,61,201,253,48,
        17,92,135,174,119,239,0,
        241,3,179,224,120,106,96,
    };

    private const int Rows4 = 3, Hid4 = 13, Bits4 = 4;

    [Fact]
    public void PackCodes_Bits4_MatchesDecodePyOracle()
    {
        var packed = new byte[Rows4 * ((Hid4 * Bits4 + 7) / 8)];
        Mach1AffineEmbedCodec.PackCodes(Codes4, Rows4, Hid4, Bits4, packed);
        Assert.Equal(Packed4, packed);
    }

    [Fact]
    public void UnpackCodes_Bits4_MatchesDecodePyOracle()
    {
        var codes = new byte[Rows4 * Hid4];
        Mach1AffineEmbedCodec.UnpackCodes(Packed4, Rows4, Hid4, Bits4, codes);
        Assert.Equal(Codes4, codes);
    }

    // bits=3, rows=2, hid=10 (30 bits/row -> 4 bytes/row; 3 never byte-aligns
    // per element, so every row after the first starts mid-byte relative to
    // element boundaries within the SAME row -- but per decode.py's own
    // per-row packbits call, padding only occurs at the row's end)
    private static readonly byte[] Codes3 =
    {
        4,4,2,5,7,1,5,0,2,3,
        1,2,7,3,0,2,2,6,5,2,
    };

    private static readonly byte[] Packed3 =
    {
        145,94,104,76,
        43,176,150,168,
    };

    private const int Rows3 = 2, Hid3 = 10, Bits3 = 3;

    [Fact]
    public void PackCodes_Bits3_MatchesDecodePyOracle()
    {
        var packed = new byte[Rows3 * ((Hid3 * Bits3 + 7) / 8)];
        Mach1AffineEmbedCodec.PackCodes(Codes3, Rows3, Hid3, Bits3, packed);
        Assert.Equal(Packed3, packed);
    }

    [Fact]
    public void UnpackCodes_Bits3_MatchesDecodePyOracle()
    {
        var codes = new byte[Rows3 * Hid3];
        Mach1AffineEmbedCodec.UnpackCodes(Packed3, Rows3, Hid3, Bits3, codes);
        Assert.Equal(Codes3, codes);
    }

    [Fact]
    public void PackThenUnpack_RoundTrips_Bits4()
    {
        var packed = new byte[Rows4 * ((Hid4 * Bits4 + 7) / 8)];
        Mach1AffineEmbedCodec.PackCodes(Codes4, Rows4, Hid4, Bits4, packed);
        var back = new byte[Rows4 * Hid4];
        Mach1AffineEmbedCodec.UnpackCodes(packed, Rows4, Hid4, Bits4, back);
        Assert.Equal(Codes4, back);
    }

    [Fact]
    public void Decode_AppliesAffineScaleAndExceptionOverrides()
    {
        // 2 rows, hid=8, group=4 (2 groups per row), bits=4.
        const int rows = 2, hid = 8, bits = 4, group = 4;
        byte[] codes = { 0, 5, 10, 15, 3, 7, 12, 1, 2, 4, 6, 8, 9, 11, 13, 14 };
        var packed = new byte[rows * ((hid * bits + 7) / 8)];
        Mach1AffineEmbedCodec.PackCodes(codes, rows, hid, bits, packed);

        Half[] mn = { (Half)(-1.0f), (Half)0.0f, (Half)(-2.0f), (Half)1.0f };
        Half[] mx = { (Half)1.0f, (Half)2.0f, (Half)2.0f, (Half)3.0f };

        var dest = new float[rows * hid];
        Mach1AffineEmbedCodec.Decode(packed, mn, mx, rows, hid, bits, group, default, default, dest);

        float lv = (1 << bits) - 1;
        for (int r = 0; r < rows; r++)
        {
            for (int g = 0; g < hid / group; g++)
            {
                float mnF = (float)mn[r * (hid / group) + g];
                float mxF = (float)mx[r * (hid / group) + g];
                float step = (mxF - mnF) / lv;
                for (int k = 0; k < group; k++)
                {
                    int j = g * group + k;
                    float expected = mnF + codes[r * hid + j] * step;
                    Assert.Equal(expected, dest[r * hid + j], precision: 4);
                }
            }
        }

        // Exception override: overwrite element 5 (row0, col5) with an exact
        // bf16 bit pattern -- truncate 3.14f to bf16-equivalent precision by
        // keeping only the top 16 bits of its fp32 representation.
        uint fp32Bits = BitConverter.SingleToUInt32Bits(3.14f);
        ushort bf16Bits = (ushort)(fp32Bits >> 16);
        var excIdx = new[] { 5 };
        var excBits = new[] { bf16Bits };
        Mach1AffineEmbedCodec.Decode(packed, mn, mx, rows, hid, bits, group, excIdx, excBits, dest);

        float expectedBf16AsF32 = BitConverter.UInt32BitsToSingle((uint)bf16Bits << 16);
        Assert.Equal(expectedBf16AsF32, dest[5]);
    }
}
