using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu;

/// <summary>
/// CPU in-place TurboQuant attention (#1892): the codes are read where they sit (decoded one row at a time), the query is
/// rotated into K code space and the output rotated back out of V code space. The oracle is the scratch-path
/// <see cref="TurboQuantKvCache"/> (full-history decode into F32, then ordinary F32 attention) built with the same seed and
/// norm format. K and V sign seeds differ by construction (V = seed ^ constant), so a swapped rotation cannot pass.
/// </summary>
public sealed unsafe class TurboQuantInPlaceAttentionTests
{
    private const ulong Seed = 0x5EEDC0DEUL;
    private const int History = 40;

    private sealed record Case(float[] KHist, float[] VHist, float[] KChunk, float[] VChunk, float[] Q);

    private static float[] Gauss(Random rng, int n, float scale)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            a[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)) * scale;
        }
        return a;
    }

    private static Case MakeCase(int seed, int hist, int seqQ, int kvStride, int qStride)
    {
        var rng = new Random(seed);
        return new Case(Gauss(rng, hist * kvStride, 1.3f), Gauss(rng, hist * kvStride, 0.9f),
                        Gauss(rng, seqQ * kvStride, 1.1f), Gauss(rng, seqQ * kvStride, 1.7f),
                        Gauss(rng, seqQ * qStride, 1.0f));
    }

    private static void Fill(IKvCache cache, float[] k, float[] v, int firstPos, int kvStride)
    {
        int n = k.Length / kvStride;
        var pos = new int[n];
        for (int i = 0; i < n; i++) pos[i] = firstPos + i;
        fixed (float* kp = k)
        fixed (float* vp = v)
            cache.Update(new TensorRef(n, kvStride, DType.Float32, -1, (nint)kp),
                         new TensorRef(n, kvStride, DType.Float32, -1, (nint)vp), pos, 0);
    }

    /// <summary>Runs the scratch oracle and the in-place path on one case; returns (inPlace, oracle) outputs.</summary>
    private static (float[] InPlace, float[] Oracle) Run(Case c, int bits, int hd, int groups, int seqQ,
                                                          float[]? sinks, int? slidingWindow, TqNormFormat norm = TqNormFormat.F16)
    {
        const int numKvHeads = 2;
        int numHeads = numKvHeads * groups, kvStride = numKvHeads * hd, qStride = numHeads * hd;
        int seqKv = History + seqQ;

        using var oracle = new TurboQuantKvCache(1, numKvHeads, hd, seqKv + 8, bits, Seed, normFormat: norm);
        using var inplace = new TurboQuantInPlaceKvCache(1, numKvHeads, hd, seqKv + 8, bits, Seed, norm);
        foreach (IKvCache cache in new IKvCache[] { oracle, inplace })
        {
            Fill(cache, c.KHist, c.VHist, 0, kvStride);
            Fill(cache, c.KChunk, c.VChunk, History, kvStride);
        }

        var want = new float[seqQ * qStride];
        var kRef = oracle.GetKeysRef(0);
        var vRef = oracle.GetValuesRef(0);
        Attention.Execute(new ReadOnlySpan<float>(c.Q),
                          new ReadOnlySpan<float>((void*)kRef.DataPointer, seqKv * kvStride),
                          new ReadOnlySpan<float>((void*)vRef.DataPointer, seqKv * kvStride),
                          want, seqQ, seqKv, numHeads, numKvHeads, hd, History, slidingWindow, 0f,
                          sinks is null ? default : sinks);

        var got = new float[seqQ * qStride];
        var q = (float[])c.Q.Clone();
        fixed (float* qp = q)
        fixed (float* op = got)
            Attention.ExecuteTurboQuant(qp, inplace, 0, op, seqQ, seqKv, numHeads, numKvHeads, hd, History, null,
                                        slidingWindow, 0f, sinks);
        return (got, want);
    }

    private static double L2(float[] a) { double s = 0; foreach (float x in a) s += (double)x * x; return Math.Sqrt(s); }

    private static double RelErr(float[] got, float[] want)
    {
        double d = 0;
        for (int i = 0; i < got.Length; i++) { double e = got[i] - want[i]; d += e * e; }
        return Math.Sqrt(d) / L2(want);
    }

    public static TheoryData<int, int, int, int, bool> Matrix()
    {
        var d = new TheoryData<int, int, int, int, bool>();
        foreach (int bits in new[] { 2, 3, 4, 8 })
            foreach (int hd in new[] { 64, 128, 256 })
                foreach (int groups in new[] { 1, 4 })
                    foreach (int seqQ in new[] { 1, 7, 64 })
                        foreach (bool sinks in new[] { false, true })
                            d.Add(bits, hd, groups, seqQ, sinks);
        return d;
    }

    // The cache has no original-space window (--cache-window > 0 with tq is refused), so the Theory covers window = null.
    [Theory]
    [MemberData(nameof(Matrix))]
    public void InPlace_MatchesScratchOracle(int bits, int hd, int groups, int seqQ, bool sinksOn)
    {
        int numHeads = 2 * groups;
        var c = MakeCase(11, History, seqQ, 2 * hd, numHeads * hd);
        float[]? sinks = sinksOn ? Enumerable.Range(0, numHeads).Select(h => 0.3f * h - 0.2f).ToArray() : null;
        var (got, want) = Run(c, bits, hd, groups, seqQ, sinks, null);

        double tol = 1e-4;  // brief allows 1e-4*sqrt(seqKv*hd) (2-10%); measured error is ~1e-6, so a tight bound is used
        Assert.True(RelErr(got, want) <= tol, $"rel err {RelErr(got, want):G4} > {tol:G4}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void InPlace_MatchesScratchOracle_SlidingWindowMask(int bits)
    {
        var c = MakeCase(12, History, 7, 2 * 64, 4 * 64);
        var (got, want) = Run(c, bits, 64, 2, 7, null, 16);
        Assert.True(RelErr(got, want) <= 1e-3, $"rel err {RelErr(got, want):G4}");
    }

    // hd=16 / 3 bits = 6 code bytes -> 2 uints per vector: the only shape where per-vector padding differs from tight packing.
    [Theory]
    [InlineData(3, 16)]
    [InlineData(5, 16)]
    [InlineData(3, 32)]
    public void InPlace_MatchesScratchOracle_PaddedCodeWords(int bits, int hd)
    {
        var c = MakeCase(13, History, 5, 2 * hd, 4 * hd);
        var (got, want) = Run(c, bits, hd, 2, 5, null, null);
        Assert.True(RelErr(got, want) <= 1e-3, $"rel err {RelErr(got, want):G4}");
    }

    [Fact]
    public void CurrentChunkRows_ComeFromTheCache()
    {
        const int hd = 64, groups = 2, seqQ = 7, bits = 4;
        int numHeads = 2 * groups;
        var a = MakeCase(21, History, seqQ, 2 * hd, numHeads * hd);
        var b = a with { KChunk = Gauss(new Random(99), seqQ * 2 * hd, 1.1f), VChunk = Gauss(new Random(98), seqQ * 2 * hd, 1.7f) };

        var (gotA, wantA) = Run(a, bits, hd, groups, seqQ, null, null);
        var (gotB, wantB) = Run(b, bits, hd, groups, seqQ, null, null);

        Assert.True(RelErr(gotA, gotB) > 1e-2, "changing the current chunk's K/V must change the in-place output");
        Assert.True(RelErr(wantA, wantB) > 1e-2);
        Assert.True(RelErr(gotA, wantA) <= 1e-3);
        Assert.True(RelErr(gotB, wantB) <= 1e-3);
    }

    [Fact]
    public void HistoryIsNeverDecodedWhole()
    {
        const int hd = 128, numKvHeads = 2, numHeads = 4, rows = 4096, bits = 4;
        int kvStride = numKvHeads * hd;
        using var cache = new TurboQuantInPlaceKvCache(1, numKvHeads, hd, rows, bits, Seed);
        var rng = new Random(5);
        var k = Gauss(rng, rows * kvStride, 1f);
        var v = Gauss(rng, rows * kvStride, 1f);
        Fill(cache, k, v, 0, kvStride);
        var q = Gauss(rng, numHeads * hd, 1f);
        var o = new float[numHeads * hd];

        long before = GC.GetAllocatedBytesForCurrentThread();
        fixed (float* qp = q)
        fixed (float* op = o)
            Attention.ExecuteTurboQuant(qp, cache, 0, op, 1, rows, numHeads, numKvHeads, hd, rows - 1, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated < (long)rows * hd * 4, $"allocated {allocated} bytes during attention");
        // No decoded view exists to hand out, and the cache owns exactly codes + norms.
        Assert.Throws<NotSupportedException>(() => { cache.GetKeysRef(0); });
        Assert.Equal(cache.Layout.CodeBytesPerLayer(rows) * 2 + cache.Layout.NormBytesPerLayer(rows) * 2, cache.AllocatedBytes);
    }

    [Fact]
    public void Window_GreaterThanZero_IsRefused()
        => Assert.Throws<ArgumentException>(() => new TurboQuantInPlaceKvCache(1, 2, 64, 32, 4, Seed, window: 8));

    [Fact]
    public void Surface_DeclaresTurboQuantWithNoWindow()
    {
        using var cache = new TurboQuantInPlaceKvCache(2, 2, 64, 32, 3, Seed);
        Assert.Equal(DotLLM.Core.Configuration.KvCacheDType.TurboQuant, cache.KeyDType);
        Assert.Equal(DotLLM.Core.Configuration.KvCacheDType.TurboQuant, cache.ValueDType);
        Assert.Equal(0, cache.WindowCapacity);
        Assert.Equal(0, cache.WindowLength);
        Assert.Equal(8, cache.Centroids.Length);
        Assert.NotEqual(cache.SignsK.ToArray(), cache.SignsV.ToArray());
        Assert.Equal(new TurboQuantLayout(64, 2, 3, TurboQuantLayout.DefaultNorm), cache.Layout);
    }
}
