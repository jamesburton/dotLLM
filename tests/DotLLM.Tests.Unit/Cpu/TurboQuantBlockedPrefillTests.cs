using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu;

/// <summary>
/// The multi-row (prefill) TurboQuant path (#1892): key tiles outer, each K/V tile decoded once for a block of query rows. It must
/// reproduce the per-row path (<c>Attention.TurboQuantPerRowOnly</c>, the pre-change kernel) BIT FOR BIT, because every row walks the same
/// tile grid and issues the same dots / online-softmax updates in the same order (#530). Runs alone: the hooks are process-wide statics.
/// </summary>
[Collection(TurboQuantHistoryProbeCollection.Name)]
public sealed unsafe class TurboQuantBlockedPrefillTests
{
    private const ulong Seed = TurboQuantInPlaceAttentionTests.Seed;
    private const int NumKvHeads = 2;

    private static float[] Attend(int bits, int hd, int groups, int hist, int seqQ, int? window, float softCap, float[]? sinks,
                                  bool perRow, int seed, ComputeThreadPool? pool = null)
    {
        int numHeads = NumKvHeads * groups, kvStride = NumKvHeads * hd, qStride = numHeads * hd;
        int seqKv = hist + seqQ;
        var rng = new Random(seed);
        var k = TurboQuantInPlaceAttentionTests.Gauss(rng, seqKv * kvStride, 1.3f);
        var v = TurboQuantInPlaceAttentionTests.Gauss(rng, seqKv * kvStride, 0.9f);
        var q = TurboQuantInPlaceAttentionTests.Gauss(rng, seqQ * qStride, 1.0f);
        using var cache = new TurboQuantInPlaceKvCache(1, NumKvHeads, hd, seqKv + 8, bits, Seed);
        TurboQuantInPlaceAttentionTests.Fill(cache, k, v, 0, kvStride);
        var got = new float[seqQ * qStride];
        Attention.TurboQuantPerRowOnly = perRow;
        try
        {
            fixed (float* qp = q)
            fixed (float* op = got)
                Attention.ExecuteTurboQuant(qp, cache, 0, op, seqQ, seqKv, numHeads, NumKvHeads, hd, hist, pool, window, softCap, sinks);
        }
        finally { Attention.TurboQuantPerRowOnly = false; }
        return got;
    }

    private static void AssertBitIdentical(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                Assert.Fail($"{what}: element {i} differs ({a[i]:R} vs {b[i]:R})");
    }

    // hd 64 -> tile 256, hd 128 -> tile 256, hd 256 -> tile 128; seqQ 300 spans three 128-row blocks; hist 301 / 517 make every tile ragged.
    public static TheoryData<int, int, int, int, int, int> Matrix()
    {
        var d = new TheoryData<int, int, int, int, int, int>();
        foreach (int bits in new[] { 2, 3, 4, 5, 8 })
            foreach (int hd in new[] { 64, 128 })
                foreach (int groups in new[] { 1, 4 })
                    foreach ((int hist, int seqQ) in new[] { (301, 2), (301, 7), (517, 300), (0, 130) })
                        foreach (int window in new[] { 0, 100, 300 })
                            d.Add(bits, hd, groups, hist, seqQ, window);
        return d;
    }

    [Theory]
    [MemberData(nameof(Matrix))]
    public void BlockedPrefill_IsBitIdenticalToThePerRowPath(int bits, int hd, int groups, int hist, int seqQ, int window)
    {
        int numHeads = NumKvHeads * groups;
        float[] sinks = Enumerable.Range(0, numHeads).Select(h => 0.3f * h - 0.2f).ToArray();
        bool sinkArm = (bits + hist + window) % 2 == 0;
        float softCap = (bits + groups) % 3 == 0 ? 30f : 0f;
        int? w = window == 0 ? null : window;
        var want = Attend(bits, hd, groups, hist, seqQ, w, softCap, sinkArm ? sinks : null, perRow: true, seed: 5);
        var got = Attend(bits, hd, groups, hist, seqQ, w, softCap, sinkArm ? sinks : null, perRow: false, seed: 5);
        AssertBitIdentical(got, want, $"bits {bits} hd {hd} groups {groups} hist {hist} seqQ {seqQ} window {window}");
    }

    [Fact]
    public void BlockedPrefill_WithAWorkerPool_IsBitIdenticalToThePerRowPath()
    {
        using var pool = new ComputeThreadPool(4);
        var want = Attend(4, 128, 4, 517, 200, 150, 0f, null, perRow: true, seed: 9, pool);
        var got = Attend(4, 128, 4, 517, 200, 150, 0f, null, perRow: false, seed: 9, pool);
        AssertBitIdentical(got, want, "pooled");
    }

    // The bounded per-tile decode buffer never scales with the cached length: 2 * ringRows * headDim floats (K and V).
    [Fact]
    public void TileBuffer_IsBoundedAndIndependentOfHistory()
    {
        var box = new System.Runtime.CompilerServices.StrongBox<long>(-1);
        Attention.TqTileBufferObserver = box;
        try
        {
            var seen = new List<long>();
            foreach (int hist in new[] { 100, 3000 })
            {
                box.Value = -1;
                Attend(4, 128, 1, hist, 4, null, 0f, null, perRow: false, seed: 3);
                seen.Add(box.Value);
            }
            // hd 128 -> tile 256 rows, one tile ring, K + V, F32: 2 * 256 * 128 * 4 bytes.
            Assert.Equal(new long[] { 262144, 262144 }, seen);

            box.Value = -1;
            Attend(4, 128, 1, 3000, 4, 100, 0f, null, perRow: false, seed: 3);
            Assert.Equal(524288, box.Value);   // sliding window: two tiles
        }
        finally { Attention.TqTileBufferObserver = null; }
    }

    [Fact]
    public void BlockedPrefill_MatchesTheScratchOracle()
    {
        // Same data as the in-place path's own oracle test, at a prefill-sized chunk: a wrong rotation/offset would miss by O(1).
        const int bits = 4, hd = 64, groups = 2, hist = 300, seqQ = 140;
        int numHeads = NumKvHeads * groups, kvStride = NumKvHeads * hd, qStride = numHeads * hd, seqKv = hist + seqQ;
        var rng = new Random(77);
        var k = TurboQuantInPlaceAttentionTests.Gauss(rng, seqKv * kvStride, 1.3f);
        var v = TurboQuantInPlaceAttentionTests.Gauss(rng, seqKv * kvStride, 0.9f);
        var q = TurboQuantInPlaceAttentionTests.Gauss(rng, seqQ * qStride, 1.0f);
        using var oracle = new TurboQuantKvCache(1, NumKvHeads, hd, seqKv + 8, bits, Seed);
        using var inplace = new TurboQuantInPlaceKvCache(1, NumKvHeads, hd, seqKv + 8, bits, Seed);
        TurboQuantInPlaceAttentionTests.Fill(oracle, k, v, 0, kvStride);
        TurboQuantInPlaceAttentionTests.Fill(inplace, k, v, 0, kvStride);
        var kRef = oracle.GetKeysRef(0);
        var vRef = oracle.GetValuesRef(0);
        var want = new float[seqQ * qStride];
        Attention.Execute(new ReadOnlySpan<float>(q), new ReadOnlySpan<float>((void*)kRef.DataPointer, seqKv * kvStride),
                          new ReadOnlySpan<float>((void*)vRef.DataPointer, seqKv * kvStride), want, seqQ, seqKv, numHeads,
                          NumKvHeads, hd, hist, 120, 0f, default);
        var got = new float[seqQ * qStride];
        var qc = (float[])q.Clone();
        fixed (float* qp = qc)
        fixed (float* op = got)
            Attention.ExecuteTurboQuant(qp, inplace, 0, op, seqQ, seqKv, numHeads, NumKvHeads, hd, hist, null, 120);
        double d = 0, n = 0;
        for (int i = 0; i < got.Length; i++) { double e = got[i] - want[i]; d += e * e; n += (double)want[i] * want[i]; }
        Assert.True(Math.Sqrt(d / n) <= 1e-3, $"rel err {Math.Sqrt(d / n):G4}");
    }
}
