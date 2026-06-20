using System;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Cache-level validation of <see cref="TurboQuantKvCache"/>: per-(position, head) storage
/// addressing, dequant quality vs the F32 store, agreement with the standalone codec,
/// determinism, and rollback. Lossy end-to-end logit quality is a model-level benchmark
/// (measured perplexity delta), not a unit test; here we prove the cache stores, addresses,
/// and reconstructs each head vector correctly through the codec.
/// </summary>
public sealed unsafe class TurboQuantKvCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 3;
    private const int HeadDim = 64;       // power of two (RHT)
    private const int Stride = NumKvHeads * HeadDim;
    private const int MaxSeqLen = 16;
    private const int Bits = 4;
    private const ulong Seed = 0xC0FFEE123;

    private readonly ITestOutputHelper _output;
    public TurboQuantKvCacheTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Dequant_RecoversPerHeadPerPositionNorms_DistinctAddressing()
    {
        using var cache = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed);

        // Each (pos, head) gets a DISTINCT target norm, widely separated, so a stride/head
        // mis-index would recover a different norm and fail. Norms are stored fp32 exactly,
        // and the codec preserves norm, so recovery is tight.
        int seqLen = 5;
        var (k, v) = BuildKv(seqLen, layerSalt: 0);
        int[] positions = { 0, 1, 2, 3, 4 };

        try
        {
            UpdateCache(cache, k, v, positions, 1);

            var kRef = cache.GetKeysRef(1);
            var vRef = cache.GetValuesRef(1);
            Assert.Equal(seqLen, kRef.Dim0);
            Assert.Equal(Stride, kRef.Dim1);

            var kOut = new ReadOnlySpan<float>((void*)kRef.DataPointer, seqLen * Stride);
            var vOut = new ReadOnlySpan<float>((void*)vRef.DataPointer, seqLen * Stride);

            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int h = 0; h < NumKvHeads; h++)
                {
                    float expK = TargetNorm(pos, h, isValue: false);
                    float expV = TargetNorm(pos, h, isValue: true);
                    float gotK = HeadNorm(kOut, pos, h);
                    float gotV = HeadNorm(vOut, pos, h);
                    // Codec preserves norm to within a few percent; bound generously but far
                    // tighter than the spacing between distinct (pos,head) norms (≥ ~100).
                    Assert.InRange(gotK, expK * 0.94f, expK * 1.06f);
                    Assert.InRange(gotV, expV * 0.94f, expV * 1.06f);
                }
            }
            _output.WriteLine("per-(pos,head) norm recovery within ±6% across all heads/positions.");
        }
        finally { FreeKv(k, v); }
    }

    [Fact]
    public void Dequant_MatchesStandaloneCodec_ForAHead()
    {
        using var cache = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed);
        var codecK = new TurboQuantCodec(HeadDim, Bits, Seed); // same K seed the cache derives

        int seqLen = 3;
        var (k, v) = BuildKv(seqLen, layerSalt: 7);
        int[] positions = { 0, 1, 2 };
        try
        {
            UpdateCache(cache, k, v, positions, 0);
            var kRef = cache.GetKeysRef(0);
            var kOut = new ReadOnlySpan<float>((void*)kRef.DataPointer, seqLen * Stride);

            // Reproduce the cache's per-head K encode/decode with the standalone codec; the
            // cache must reconstruct identically (proves it uses the codec correctly per head).
            Span<byte> codes = stackalloc byte[codecK.CodeBytesPerVector];
            var expected = new float[HeadDim];
            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int h = 0; h < NumKvHeads; h++)
                {
                    var src = new ReadOnlySpan<float>((float*)k[pos] + h * HeadDim, HeadDim);
                    float norm = codecK.Encode(src, codes);
                    codecK.Decode(codes, norm, expected);
                    for (int d = 0; d < HeadDim; d++)
                        Assert.Equal(expected[d], kOut[pos * Stride + h * HeadDim + d], 5);
                }
            }
        }
        finally { FreeKv(k, v); }
    }

    [Fact]
    public void SameSeed_ProducesIdenticalDequant_Deterministic()
    {
        using var a = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed);
        using var b = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed);

        int seqLen = 4;
        var (k, v) = BuildKv(seqLen, layerSalt: 3);
        int[] positions = { 0, 1, 2, 3 };
        try
        {
            UpdateCache(a, k, v, positions, 0);
            UpdateCache(b, k, v, positions, 0);

            var ra = new ReadOnlySpan<float>((void*)a.GetKeysRef(0).DataPointer, seqLen * Stride);
            var rb = new ReadOnlySpan<float>((void*)b.GetKeysRef(0).DataPointer, seqLen * Stride);
            for (int i = 0; i < seqLen * Stride; i++) Assert.Equal(ra[i], rb[i]);
        }
        finally { FreeKv(k, v); }
    }

    [Fact]
    public void Rollback_TruncatesVisibleLength()
    {
        using var cache = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed);
        int seqLen = 6;
        var (k, v) = BuildKv(seqLen, layerSalt: 1);
        int[] positions = { 0, 1, 2, 3, 4, 5 };
        try
        {
            UpdateCache(cache, k, v, positions, 0);
            Assert.Equal(6, cache.CurrentLength);
            cache.Rollback(3);
            Assert.Equal(3, cache.CurrentLength);
            Assert.Equal(3, cache.GetKeysRef(0).Dim0);
        }
        finally { FreeKv(k, v); }
    }

    [Fact]
    public void AllocatedBytes_IsCompressedVsF32()
    {
        using var tq = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed);
        using var f32 = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        Assert.True(tq.AllocatedBytes < f32.AllocatedBytes,
            $"TurboQuant store ({tq.AllocatedBytes}) should be smaller than F32 ({f32.AllocatedBytes}).");
    }

    [Fact]
    public void Qjl_Cache_MatchesStandaloneQjlCodec_ForAHead()
    {
        // The QJL cache must reconstruct each head identically to a standalone QJL codec built with
        // the cache's derived K seed — proving it wires useQjl + the same seed/bit-width per head.
        using var cache = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed, useQjl: true);
        Assert.True(cache.UseQjl);
        var codecK = new TurboQuantCodec(HeadDim, Bits, Seed, useQjl: true);

        int seqLen = 3;
        var (k, v) = BuildKv(seqLen, layerSalt: 7);
        int[] positions = { 0, 1, 2 };
        try
        {
            UpdateCache(cache, k, v, positions, 0);
            var kRef = cache.GetKeysRef(0);
            var kOut = new ReadOnlySpan<float>((void*)kRef.DataPointer, seqLen * Stride);

            Span<byte> codes = stackalloc byte[codecK.CodeBytesPerVector];
            var expected = new float[HeadDim];
            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int h = 0; h < NumKvHeads; h++)
                {
                    var src = new ReadOnlySpan<float>((float*)k[pos] + h * HeadDim, HeadDim);
                    float norm = codecK.Encode(src, codes);
                    codecK.Decode(codes, norm, expected);
                    for (int d = 0; d < HeadDim; d++)
                        Assert.Equal(expected[d], kOut[pos * Stride + h * HeadDim + d], 5);
                }
            }
        }
        finally { FreeKv(k, v); }
    }

    [Fact]
    public void Qjl_Cache_StillCompressedVsF32()
    {
        using var tq = new TurboQuantKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen, Bits, Seed, useQjl: true);
        using var f32 = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        Assert.True(tq.UseQjl);
        Assert.True(tq.AllocatedBytes < f32.AllocatedBytes,
            $"QJL store ({tq.AllocatedBytes}) should still be smaller than F32 ({f32.AllocatedBytes}).");
    }

    [Fact]
    public void NonPowerOfTwoHeadDim_Throws()
        => Assert.Throws<ArgumentException>(() => new TurboQuantKvCache(1, 1, 48, 8, Bits, Seed));

    [Theory]
    [InlineData("turboquant", KvCacheDType.TurboQuant, 0, false)]
    [InlineData("tq", KvCacheDType.TurboQuant, 0, false)]
    [InlineData("tq4", KvCacheDType.TurboQuant, 4, false)]
    [InlineData("tq2", KvCacheDType.TurboQuant, 2, false)]
    [InlineData("tqq", KvCacheDType.TurboQuant, 0, true)]
    [InlineData("tq4q", KvCacheDType.TurboQuant, 4, true)]
    [InlineData("tq2q", KvCacheDType.TurboQuant, 2, true)]
    [InlineData("TQ4Q", KvCacheDType.TurboQuant, 4, true)]   // case-insensitive
    [InlineData("q8_0", KvCacheDType.Q8_0, 0, false)]
    [InlineData("f32", KvCacheDType.F32, 0, false)]
    public void ParseDType_HandlesTurboQuantBitsAndQjl(string input, KvCacheDType expected, int expectedBits, bool expectedQjl)
    {
        var dt = KvCacheConfig.ParseDType(input, out int bits, out bool useQjl);
        Assert.Equal(expected, dt);
        Assert.Equal(expectedBits, bits);
        Assert.Equal(expectedQjl, useQjl);
    }

    [Theory]
    [InlineData("tq9")]     // bits out of range
    [InlineData("tqx")]     // malformed
    [InlineData("tq4x")]    // trailing non-q
    public void ParseDType_RejectsMalformedTurboQuant(string input)
        => Assert.Throws<ArgumentException>(() => KvCacheConfig.ParseDType(input));

    [Fact]
    public void Config_IsTurboQuant_AndIsQuantizedAreDistinct()
    {
        var tq = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant);
        Assert.True(tq.IsTurboQuant);
        Assert.False(tq.IsQuantized);   // block-quant path must NOT claim TurboQuant
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        Assert.True(q8.IsQuantized);
        Assert.False(q8.IsTurboQuant);
    }

    // ── helpers ──────────────────────────────────────────────────────────

    // Distinct, widely-spaced target norm per (pos, head, K/V).
    private static float TargetNorm(int pos, int head, bool isValue)
        => (isValue ? 5000f : 0f) + (pos + 1) * 100f + head * 17f + 13f;

    private static float HeadNorm(ReadOnlySpan<float> rows, int pos, int head)
    {
        double s = 0;
        int off = pos * Stride + head * HeadDim;
        for (int d = 0; d < HeadDim; d++) s += (double)rows[off + d] * rows[off + d];
        return (float)Math.Sqrt(s);
    }

    // Builds seqLen native rows of [Stride] for K and V; each head is a deterministic Gaussian
    // direction (matches the codec's assumption) scaled to its distinct target norm.
    private static (nint[] k, nint[] v) BuildKv(int seqLen, int layerSalt)
    {
        var k = new nint[seqLen];
        var v = new nint[seqLen];
        for (int pos = 0; pos < seqLen; pos++)
        {
            k[pos] = (nint)NativeMemory.AlignedAlloc((nuint)(Stride * sizeof(float)), 64);
            v[pos] = (nint)NativeMemory.AlignedAlloc((nuint)(Stride * sizeof(float)), 64);
            for (int h = 0; h < NumKvHeads; h++)
            {
                FillHead((float*)k[pos] + h * HeadDim, TargetNorm(pos, h, false), (pos * 31 + h * 7 + layerSalt) * 2 + 1);
                FillHead((float*)v[pos] + h * HeadDim, TargetNorm(pos, h, true), (pos * 31 + h * 7 + layerSalt) * 2 + 2);
            }
        }
        return (k, v);
    }

    private static void FillHead(float* dst, float targetNorm, int seed)
    {
        var rng = new Random(seed);
        double sq = 0;
        for (int d = 0; d < HeadDim; d++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            dst[d] = g;
            sq += (double)g * g;
        }
        float scale = targetNorm / (float)Math.Sqrt(sq);
        for (int d = 0; d < HeadDim; d++) dst[d] *= scale;
    }

    // Packs the per-row K/V allocations into contiguous buffers, runs Update, and frees the
    // temporaries (the cache reads each row at i*stride, so it needs contiguous input).
    private static void UpdateCache(TurboQuantKvCache cache, nint[] k, nint[] v, int[] positions, int layer)
    {
        int seqLen = positions.Length;
        nint pk = Pack(k, seqLen), pv = Pack(v, seqLen);
        try
        {
            cache.Update(
                new TensorRef(seqLen, Stride, DType.Float32, -1, pk),
                new TensorRef(seqLen, Stride, DType.Float32, -1, pv),
                positions, layer);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)pk);
            NativeMemory.AlignedFree((void*)pv);
        }
    }

    private static nint Pack(nint[] rows, int seqLen)
    {
        nint packed = (nint)NativeMemory.AlignedAlloc((nuint)((long)seqLen * Stride * sizeof(float)), 64);
        for (int i = 0; i < seqLen; i++)
            Buffer.MemoryCopy((void*)rows[i], (void*)((float*)packed + i * Stride), Stride * sizeof(float), Stride * sizeof(float));
        return packed;
    }

    private static void FreeKv(nint[] k, nint[] v)
    {
        foreach (var p in k) if (p != 0) NativeMemory.AlignedFree((void*)p);
        foreach (var p in v) if (p != 0) NativeMemory.AlignedFree((void*)p);
    }
}
