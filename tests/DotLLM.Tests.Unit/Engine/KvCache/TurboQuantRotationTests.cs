using System;
using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// #1892 Task 3: the CPU Q/O rotation kernels that let attention run in TurboQuant code space, checked against the
/// shared <see cref="TurboQuantCodec"/> rotation (<c>Rotate</c> / <c>InverseRotate</c>) and the per-side format pair.
/// </summary>
public sealed unsafe class TurboQuantRotationTests
{
    internal const ulong Seed = 0x7A2B_C0DEUL;
    internal const ulong VSeedXor = TurboQuantCodec.VSeedMask;

    public static TheoryData<int, int, int> Geometries => new()
    {
        { 64, 3, 2 }, { 128, 8, 4 }, { 256, 2, 3 },
    };

    internal static float[] Gaussian(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            v[i] = scale * (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }

    [Theory]
    [MemberData(nameof(Geometries))]
    public void RotateQuery_MatchesCodecRotate(int headDim, int numHeads, int rows)
    {
        var codecK = new TurboQuantCodec(headDim, 4, Seed);
        var codecV = new TurboQuantCodec(headDim, 4, Seed ^ VSeedXor);
        sbyte[] signsK = TurboQuantRotation.SignsToSbyte(codecK.RotationSigns);
        var rng = new Random(headDim * 31 + numHeads);
        float[] q = Gaussian(rng, rows * numHeads * headDim, 3f);
        float[] expected = (float[])q.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecK.Rotate(expected.AsSpan(v * headDim, headDim));

        float[] got = (float[])q.Clone();
        fixed (float* p = got) TurboQuantRotation.RotateQueryInPlace(p, rows, numHeads, headDim, signsK);
        AssertClose(expected, got);

        // Substitution: the V signs are a different diagonal, so rotating with them must be visibly wrong. Without this the
        // assertion above could not tell a K-signs query rotation from a V-signs one.
        float[] wrong = (float[])q.Clone();
        sbyte[] signsV = TurboQuantRotation.SignsToSbyte(codecV.RotationSigns);
        fixed (float* p = wrong) TurboQuantRotation.RotateQueryInPlace(p, rows, numHeads, headDim, signsV);
        Assert.True(MaxAbsDiff(expected, wrong) > 0.5f, "V signs are indistinguishable from K signs on this fixture");
    }

    [Theory]
    [MemberData(nameof(Geometries))]
    public void RotateOutput_InvertsCodec(int headDim, int numHeads, int rows)
    {
        var codecV = new TurboQuantCodec(headDim, 4, Seed ^ VSeedXor);
        var codecK = new TurboQuantCodec(headDim, 4, Seed);
        sbyte[] signsV = TurboQuantRotation.SignsToSbyte(codecV.RotationSigns);
        var rng = new Random(headDim * 17 + rows);
        float[] a = Gaussian(rng, rows * numHeads * headDim, 2f);
        float[] expected = (float[])a.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecV.InverseRotate(expected.AsSpan(v * headDim, headDim));

        float[] got = (float[])a.Clone();
        fixed (float* p = got) TurboQuantRotation.RotateOutputInPlace(p, rows, numHeads, headDim, signsV);
        AssertClose(expected, got);

        // and it really inverts the V-side forward rotation
        float[] rt = (float[])got.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecV.Rotate(rt.AsSpan(v * headDim, headDim));
        AssertClose(a, rt);

        float[] wrong = (float[])a.Clone();
        sbyte[] signsK = TurboQuantRotation.SignsToSbyte(codecK.RotationSigns);
        fixed (float* p = wrong) TurboQuantRotation.RotateOutputInPlace(p, rows, numHeads, headDim, signsK);
        Assert.True(MaxAbsDiff(expected, wrong) > 0.5f, "K signs are indistinguishable from V signs on this fixture");
    }

    [Theory]
    [InlineData(64, 2)]
    [InlineData(128, 4)]
    [InlineData(256, 3)]
    public void QueryDotCodes_EqualsQueryDotDecodedK(int headDim, int bits)
    {
        var codec = new TurboQuantCodec(headDim, bits, Seed);
        sbyte[] signs = TurboQuantRotation.SignsToSbyte(codec.RotationSigns);
        var rng = new Random(headDim + bits);
        var codes = new byte[codec.CodeBytesPerVector];
        var kHat = new float[headDim];
        for (int trial = 0; trial < 20; trial++)
        {
            float[] k = Gaussian(rng, headDim, 1f + trial);
            float[] q = Gaussian(rng, headDim, 2f);
            float norm = codec.Encode(k, codes);
            codec.Decode(codes, norm, kHat);
            double viaDecode = 0;
            for (int i = 0; i < headDim; i++) viaDecode += (double)q[i] * kHat[i];

            float[] qt = (float[])q.Clone();
            fixed (float* p = qt) TurboQuantRotation.RotateQueryInPlace(p, 1, 1, headDim, signs);
            double viaCodes = 0;
            for (int i = 0; i < headDim; i++) viaCodes += (double)qt[i] * norm * codec.Centroids[codec.ReadCode(codes, i)];

            double scale = Math.Max(1.0, Math.Abs(viaDecode));
            Assert.True(Math.Abs(viaDecode - viaCodes) <= 1e-5 * scale, $"trial {trial}: decode={viaDecode} codes={viaCodes}");
        }
    }

    [Fact]
    public void KvFormatPair_ExpressesPerSideFormats()
    {
        var tq = KvFormatPair.Both(KvSideFormat.Tq);
        Assert.True(tq.RotatesQuery);
        Assert.True(tq.RotatesOutput);
        Assert.False(tq.IsMixed);
        var mixed = new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Tq);
        Assert.False(mixed.RotatesQuery);
        Assert.True(mixed.RotatesOutput);
        Assert.True(mixed.IsMixed);
        Assert.False(KvFormatPair.Both(KvSideFormat.F16).RotatesQuery);
    }

    internal static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static void AssertClose(float[] expected, float[] got)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - got[i]) <= 1e-5f * MathF.Max(1f, MathF.Abs(expected[i])),
                $"i={i}: codec={expected[i]} kernel={got[i]}");
    }

    // Head dims the rotation kernels cannot serve are rejected loudly, never rotated with a wrong butterfly (#1897 keeps them refused above).
    [Theory]
    [InlineData(96)]
    [InlineData(100)]
    public void NonPowerOfTwoHeadDim_IsRejected(int headDim)
    {
        var signs = new sbyte[headDim];
        Array.Fill(signs, (sbyte)1);
        var data = new float[2 * 3 * headDim];
        Assert.Throws<ArgumentException>(() => { fixed (float* p = data) TurboQuantRotation.RotateQueryInPlace(p, 2, 3, headDim, signs); });
        Assert.Throws<ArgumentException>(() => { fixed (float* p = data) TurboQuantRotation.RotateOutputInPlace(p, 2, 3, headDim, signs); });
    }

    [Fact]
    public void SignsLengthMismatch_IsRejected()
    {
        var data = new float[2 * 3 * 64];
        Assert.Throws<ArgumentException>(() => { fixed (float* p = data) TurboQuantRotation.RotateQueryInPlace(p, 2, 3, 64, new sbyte[32]); });
    }
}
