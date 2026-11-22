using System;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// #1892 Task 2, CPU side: the oracle stores norms in the decided format through the shared codec, and the encode
/// arithmetic (norm reduction order, normalise, argmin tie policy) is the fixed one every GPU backend reproduces.
/// </summary>
public sealed unsafe class TurboQuantNormFormatTests
{
    private const int NumLayers = 2, NumKvHeads = 3, HeadDim = 128, MaxSeqLen = 8, Bits = 3;
    private const ulong Seed = 0xFEED_F00D;

    private readonly ITestOutputHelper _output;
    public TurboQuantNormFormatTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void DefaultNormFormat_IsF16()
    {
        using var cache = new TurboQuantKvCache(1, 1, 64, 4, 4, Seed);
        Assert.Equal(TqNormFormat.F16, cache.NormFormat);
        Assert.Equal(TurboQuantLayout.DefaultNorm, cache.NormFormat);
        // 2 sides * 4 positions * (32 code bytes + 2 norm bytes)
        Assert.Equal(2L * 4 * (32 + 2), cache.AllocatedBytes);
    }

    [Theory]
    [InlineData(TqNormFormat.F32Reference)]
    [InlineData(TqNormFormat.F16)]
    [InlineData(TqNormFormat.Bf16)]
    [InlineData(TqNormFormat.Fp8E4M3)]
    [InlineData(TqNormFormat.Fp8E5M2)]
    [InlineData(TqNormFormat.Log8)]
    public void Oracle_NormsRoundTripThroughTheSharedCodec(TqNormFormat format)
    {
        using var cache = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed, normFormat: format);
        var codecK = new TurboQuantCodec(HeadDim, Bits, Seed);
        var codecV = new TurboQuantCodec(HeadDim, Bits, Seed ^ TurboQuantByteExactCorpus.VSeedXor);

        const int seqLen = 5;
        int stride = NumKvHeads * HeadDim;
        var rng = new Random(11);
        var k = new float[seqLen * stride];
        var v = new float[seqLen * stride];
        for (int i = 0; i < k.Length; i++)
        {
            k[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i / stride * 3);
            v[i] = (float)(rng.NextDouble() * 2 - 1) * (2 + i / stride);
        }
        int[] positions = { 0, 1, 2, 3, 4 };
        fixed (float* pk = k)
        fixed (float* pv = v)
            cache.Update(new TensorRef(seqLen, stride, DType.Float32, -1, (nint)pk),
                         new TensorRef(seqLen, stride, DType.Float32, -1, (nint)pv), positions, 1);

        foreach (bool values in new[] { false, true })
        {
            var codec = values ? codecV : codecK;
            var src = values ? v : k;
            var expectedNorms = new float[seqLen * NumKvHeads];
            var tmp = new byte[codec.CodeBytesPerVector];
            for (int pos = 0; pos < seqLen; pos++)
                for (int h = 0; h < NumKvHeads; h++)
                    expectedNorms[pos * NumKvHeads + h] = codec.Encode(src.AsSpan(pos * stride + h * HeadDim, HeadDim), tmp);

            var expected = new byte[expectedNorms.Length * TurboQuantNormCodec.BytesPerNorm(format)];
            TurboQuantNormCodec.Encode(format, expectedNorms, expected);

            Assert.Equal(expected, cache.StoredNormBytes(1, values).ToArray());
        }
    }

    [Fact]
    public void Dequant_ReadsTheStoredFormat_NotTheEncodeTimeFloat()
    {
        // F16 stores a rounded norm; the dequantised vector must carry exactly the decoded F16 norm, not the F32 one.
        using var cache = new TurboQuantKvCache(1, 1, 64, 4, 4, Seed);
        var x = new float[64];
        for (int i = 0; i < 64; i++) x[i] = 0.1234567f * (i % 7 + 1);
        int[] pos = { 0 };
        float norm;
        fixed (float* p = x)
        {
            cache.Update(new TensorRef(1, 64, DType.Float32, -1, (nint)p), new TensorRef(1, 64, DType.Float32, -1, (nint)p), pos, 0);
            norm = new TurboQuantCodec(64, 4, Seed).Encode(x, new byte[32]);
        }
        float stored = TurboQuantNormCodec.Decode(TqNormFormat.F16, cache.StoredNormBytes(0, false), 0);
        Assert.NotEqual(norm, stored); // the fixture would be vacuous if F16 happened to be exact
        var outK = new ReadOnlySpan<float>((void*)cache.GetKeysRef(0).DataPointer, 64);
        double ss = 0; foreach (float f in outK) ss += (double)f * f;
        // MSE reconstruction is a contraction by the codec; the norm ratio to the decoded F16 norm is what the codec gives.
        var codec = new TurboQuantCodec(64, 4, Seed);
        var codes = new byte[codec.CodeBytesPerVector]; codec.Encode(x, codes);
        var exp = new float[64]; codec.Decode(codes, stored, exp);
        Assert.Equal(exp, outK.ToArray());
    }

    // The norm's sum of squares is a fixed pairwise tree (see TurboQuantCodec.NormalizeAndRotate).
    private static float TreeNorm(ReadOnlySpan<float> x)
    {
        int lanes = Math.Min(x.Length, 256);
        var l = new float[lanes];
        for (int t = 0; t < lanes; t++) { float a = x[t] * x[t]; for (int j = t + lanes; j < x.Length; j += lanes) { float q = x[j] * x[j]; a = a + q; } l[t] = a; }
        for (int r = lanes / 2; r > 0; r /= 2) for (int t = 0; t < r; t++) l[t] = l[t] + l[t + r];
        return MathF.Sqrt(l[0]);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void Encode_NormIsTheFixedPairwiseTree_NotSequentialOrDouble(int headDim)
    {
        var codec = new TurboQuantCodec(headDim, 4, Seed);
        var rng = new Random(headDim);
        var buf = new byte[codec.CodeBytesPerVector];
        int differsFromSequential = 0, differsFromDouble = 0;
        for (int trial = 0; trial < 200; trial++)
        {
            var x = new float[headDim];
            for (int i = 0; i < headDim; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (trial + 1);
            float norm = codec.Encode(x, buf);
            Assert.Equal(BitConverter.SingleToUInt32Bits(TreeNorm(x)), BitConverter.SingleToUInt32Bits(norm));

            float seq = 0; foreach (float f in x) { float q = f * f; seq += q; }
            if (MathF.Sqrt(seq) != norm) differsFromSequential++;
            double dbl = 0; foreach (float f in x) dbl += (double)f * f;
            if ((float)Math.Sqrt(dbl) != norm) differsFromDouble++;
        }
        _output.WriteLine($"hd={headDim}: tree norm differs from sequential in {differsFromSequential}/200, from fp64 in {differsFromDouble}/200");
        // Substitution check: the alternatives must be observable on this data, or the assertion above could not fail.
        Assert.True(differsFromSequential > 0, "sequential sum is indistinguishable from the tree on this fixture");
        Assert.True(differsFromDouble > 0, "fp64 sum is indistinguishable from the tree on this fixture");
    }

    [Theory]
    [MemberData(nameof(Configs))]
    public void Corpus_ContainsExactTies_AndCpuPicksTheLowerCentroid(int bits, int headDim, bool half)
    {
        var corpus = TurboQuantByteExactCorpus.Build(bits, headDim, half);
        foreach (var (name, side) in new[] { ("K", corpus.K), ("V", corpus.V) })
        {
            var codec = side.Codec;
            int possible = TurboQuantByteExactCorpus.TieValues(codec).Count;
            _output.WriteLine($"b={bits} hd={headDim} half={half} {name}: {side.TieCoords} tie coords in {side.TieVectors} vectors ({possible}/{codec.LevelCount - 1} boundaries admit an exact F32 tie)");
            // Every config in Configs() admits at least one exact F32 tie (measured: no (0/N) line over all configs), so this is
            // unconditional: a guard on `possible > 0` would let a corpus that lost its ties pass vacuously.
            Assert.True(possible > 0, $"{name}: no centroid boundary admits an exact F32 tie at b={bits} hd={headDim}");
            Assert.True(side.TieCoords >= 4, $"{name}: corpus has only {side.TieCoords} exact ties");

            var rot = new float[headDim];
            int n = TurboQuantByteExactCorpus.SeqLen * TurboQuantByteExactCorpus.NumKvHeads;
            var c = codec.Centroids;
            int tiesChecked = 0;
            for (int v = 0; v < n; v++)
            {
                codec.NormalizeAndRotate(side.Data.AsSpan(v * headDim, headDim), rot);
                for (int i = 0; i < headDim; i++)
                {
                    float best = float.MaxValue; int lowest = -1, hits = 0;
                    for (int j = 0; j < c.Length; j++)
                    {
                        float dist = MathF.Abs(rot[i] - c[j]);
                        if (dist < best) { best = dist; lowest = j; hits = 1; }
                        else if (dist == best) hits++;
                    }
                    Assert.Equal(lowest, codec.NearestIndex(rot[i]));        // lower index on an exact tie
                    Assert.Equal(lowest, codec.NearestIndexFullScan(rot[i]));
                    if (hits >= 2) tiesChecked++;
                }
            }
            // zero / constant vectors add further exact ties on top of the injected ones
            Assert.True(tiesChecked >= side.TieCoords, $"{tiesChecked} < {side.TieCoords}");
        }
    }

    [Fact]
    public void NearestIndex_MatchesFullScanArgmin_OnRandomValues()
    {
        var rng = new Random(5);
        foreach (int bits in TurboQuantByteExactCorpus.BitWidths)
        {
            var codec = new TurboQuantCodec(128, bits, Seed);
            var c = codec.Centroids;
            for (int i = 0; i < 20000; i++)
            {
                float v = (float)(rng.NextDouble() * 2 - 1) * 0.3f;
                Assert.Equal(codec.NearestIndexFullScan(v), codec.NearestIndex(v));
            }
            foreach (float edge in new[] { c[0] - 1f, c[^1] + 1f, c[0], c[^1], 0f })
                Assert.Equal(codec.NearestIndexFullScan(edge), codec.NearestIndex(edge));
        }
    }

    public static System.Collections.Generic.IEnumerable<object[]> Configs()
    {
        foreach (int b in TurboQuantByteExactCorpus.BitWidths)
            foreach (int d in TurboQuantByteExactCorpus.HeadDims)
                foreach (bool half in new[] { false, true })
                    yield return new object[] { b, d, half };
    }
}
