using System;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Guards the TurboQuant per-head-vector norm codecs and the shared storage layout (#1892 task 1).
/// </summary>
public sealed class TurboQuantNormCodecTests
{
    private static readonly TqNormFormat[] Formats =
        [TqNormFormat.F16, TqNormFormat.Bf16, TqNormFormat.Fp8E4M3, TqNormFormat.Fp8E5M2, TqNormFormat.Log8];

    public static TheoryData<TqNormFormat> QuantisedFormats => new(Formats);

    private static double RelBound(TqNormFormat f) => f switch
    {
        TqNormFormat.F16 => Math.Pow(2, -11),
        TqNormFormat.Bf16 => Math.Pow(2, -8),
        TqNormFormat.Fp8E4M3 => Math.Pow(2, -4),
        TqNormFormat.Fp8E5M2 => Math.Pow(2, -3),
        // The brief's bound is one relative step of the log grid (~3.7%). That is too loose to catch a
        // floor-instead-of-round bug (floor's worst error is 1 - 1/step = ~3.57% < 3.7%), so the test asserts
        // the true round-to-nearest bound: HALF a step in the log domain (~1.84%).
        TqNormFormat.Log8 => Math.Pow(2, (Math.Log2(TurboQuantNormCodec.Log8Max) - Math.Log2(TurboQuantNormCodec.Log8Min)) / 255.0 / 2) - 1.0,
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    // The raw FP8 formats cannot span the whole [1e-3, 1e3] test range: E4M3 normals are [2^-6, 448] and
    // saturate above; below the normal range the relative bound does not apply (subnormals). The bound is
    // therefore asserted over the intersection of the brief's range with each format's normal range.
    private static (double lo, double hi) Range(TqNormFormat f) => f switch
    {
        TqNormFormat.Fp8E4M3 => (Math.Pow(2, -6), 448.0),
        _ => (1e-3, 1e3),
    };

    [Theory]
    [MemberData(nameof(QuantisedFormats))]
    public void Roundtrip_RelativeError_WithinFormatBound(TqNormFormat format)
    {
        var (lo, hi) = Range(format);
        var rng = new Random(42);
        const int n = 20000;
        var src = new float[n + 1];
        for (int i = 0; i < n; i++)
            src[i] = (float)Math.Exp(Math.Log(lo) + rng.NextDouble() * (Math.Log(hi) - Math.Log(lo)));
        src[n] = 0f; // exact zero must survive

        int nb = TurboQuantNormCodec.BytesPerNorm(format);
        var dst = new byte[src.Length * nb];
        TurboQuantNormCodec.Encode(format, src, dst);

        double bound = RelBound(format);
        double worst = 0;
        for (int i = 0; i < src.Length; i++)
        {
            float got = TurboQuantNormCodec.Decode(format, dst, i);
            if (src[i] == 0f) { Assert.Equal(0f, got); continue; }
            double rel = Math.Abs((double)got - src[i]) / src[i];
            worst = Math.Max(worst, rel);
            Assert.True(rel <= bound * 1.001, $"{format}: x={src[i]:R} got={got:R} rel={rel:E3} > bound {bound:E3}");
        }
        Assert.True(worst > 0, "a roundtrip that is exact for every input cannot be a quantiser");
    }

    [Theory]
    [MemberData(nameof(QuantisedFormats))]
    public void Saturation_IsCountedNotInfinite(TqNormFormat format)
    {
        // float.MaxValue is the only value every format (bf16 shares F32's range) must saturate.
        float[] src = [float.MaxValue, float.MaxValue, float.MaxValue, 1.0f];
        var dst = new byte[src.Length * TurboQuantNormCodec.BytesPerNorm(format)];
        TurboQuantNormCodec.Encode(format, src, dst, out int saturated);

        Assert.Equal(3, saturated);
        float max = TurboQuantNormCodec.Decode(format, dst, 0);
        Assert.True(float.IsFinite(max) && max > 0);
        for (int i = 1; i < 3; i++)
            Assert.Equal(max, TurboQuantNormCodec.Decode(format, dst, i));
        Assert.True(Math.Abs(TurboQuantNormCodec.Decode(format, dst, 3) - 1.0f) < 0.05f);
    }

    [Fact]
    public void Fp8_SaturatesAtFormatMax()
    {
        var dst = new byte[1];
        TurboQuantNormCodec.Encode(TqNormFormat.Fp8E4M3, [1e6f], dst);
        Assert.Equal(448f, TurboQuantNormCodec.Decode(TqNormFormat.Fp8E4M3, dst, 0));
        TurboQuantNormCodec.Encode(TqNormFormat.Fp8E5M2, [1e9f], dst);
        Assert.Equal(57344f, TurboQuantNormCodec.Decode(TqNormFormat.Fp8E5M2, dst, 0));
    }

    [Fact]
    public void F32Reference_IsLossless()
    {
        float[] src = [0f, 1.2345678f, 3.0e-9f, 7.7e12f];
        var dst = new byte[16];
        TurboQuantNormCodec.Encode(TqNormFormat.F32Reference, src, dst);
        for (int i = 0; i < src.Length; i++)
            Assert.Equal(src[i], TurboQuantNormCodec.Decode(TqNormFormat.F32Reference, dst, i));
    }

    [Fact]
    public void DefaultNorm_IsF16_TwoBytes()
    {
        Assert.Equal(TqNormFormat.F16, TurboQuantLayout.DefaultNorm);
        Assert.Equal(2, new TurboQuantLayout(128, 8, 4, TurboQuantLayout.DefaultNorm).NormBytes);
    }

    [Fact]
    public void Layout_ByteCounts()
    {
        var l = new TurboQuantLayout(HeadDim: 128, NumKvHeads: 8, MseBits: 4, Norm: TqNormFormat.F16);
        Assert.Equal(16, l.CodeUintsPerVec);
        Assert.Equal(2, l.NormBytes);
        Assert.Equal(100L * 8 * 16 * 4 + 4, l.CodeBytesPerLayer(100));
        Assert.Equal(100L * 8 * 2, l.NormBytesPerLayer(100));

        // hd 64 @ 3 bits = 24 bytes = 6 uints; hd 128 @ 2 bits = 32 bytes = 8 uints; hd 96 @ 3 = 36 bytes = 9 uints.
        Assert.Equal(6, new TurboQuantLayout(64, 1, 3, TqNormFormat.Log8).CodeUintsPerVec);
        Assert.Equal(8, new TurboQuantLayout(128, 1, 2, TqNormFormat.Log8).CodeUintsPerVec);
        Assert.Equal(9, new TurboQuantLayout(96, 1, 3, TqNormFormat.Log8).CodeUintsPerVec);
        // 5 bytes per vec rounds up to 2 uints.
        Assert.Equal(2, new TurboQuantLayout(40, 1, 1, TqNormFormat.Bf16).CodeUintsPerVec);
        Assert.Equal(4, new TurboQuantLayout(1, 1, 1, TqNormFormat.F32Reference).NormBytes);
        Assert.Equal(1, new TurboQuantLayout(1, 1, 1, TqNormFormat.Fp8E4M3).NormBytes);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void ArgminTie_PicksLowerIndex(int bits)
    {
        var codec = new TurboQuantCodec(64, bits, seed: 7);
        var boundaries = codec.Boundaries;
        Assert.Equal(codec.LevelCount - 1, boundaries.Length);
        var ties = TurboQuantByteExactCorpus.TieValues(codec);
        Assert.NotEmpty(ties);
        foreach (var (t, k) in ties)
        {
            // Exactly equidistant in the F32 arithmetic the encoders use (|t-c_k| == |c_k+1-t|): the LOWER index must win.
            Assert.Equal(k, codec.NearestIndex(t));
            Assert.Equal(k, codec.NearestIndexFullScan(t));
        }
        for (int k = 0; k < boundaries.Length; k++)
        {
            // The float midpoint is not always an exact F32 distance tie, so around it the answer is whatever the
            // argmin gives: NearestIndex must agree with the full-scan argmin (the GPU arithmetic) one ulp either side.
            float b = boundaries[k];
            Assert.Equal(codec.NearestIndexFullScan(b), codec.NearestIndex(b));
            Assert.Equal(codec.NearestIndexFullScan(MathF.BitIncrement(b)), codec.NearestIndex(MathF.BitIncrement(b)));
            Assert.Equal(codec.NearestIndexFullScan(MathF.BitDecrement(b)), codec.NearestIndex(MathF.BitDecrement(b)));
            // Clear of the boundary the answer is unambiguous.
            Assert.Equal(k, codec.NearestIndex(b - 1e-3f * Math.Abs(b) - 1e-6f));
            Assert.Equal(k + 1, codec.NearestIndex(b + 1e-3f * Math.Abs(b) + 1e-6f));
        }
    }
}
