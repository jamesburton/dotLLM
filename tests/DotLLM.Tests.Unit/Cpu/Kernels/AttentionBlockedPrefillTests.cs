using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1570: the blocked causal prefill kernel (KV head x query-row block x key tile, dynamic work items) must be bit-identical to the
/// single-threaded per-row kernel. Shapes are the real LFM3 ones (GQA 16/4, hd128) plus MHA and hd256 (tile 128), with the query
/// block straddling the one-shot / tiled threshold (2048 visible keys), a row count that is not a multiple of the 32-row block, a
/// causal end that is not a multiple of the tile, and a cache longer than the last query position.
/// </summary>
public sealed unsafe class AttentionBlockedPrefillTests : IDisposable
{
    private readonly ComputeThreadPool _pool = new(6);   // not a divisor of the item count: uneven dynamic scheduling

    public void Dispose() => _pool.Dispose();

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    [Theory]
    [InlineData(16, 4, 128, 300, 1900, 2200)]   // LFM3 shape; rows 1901..2200 visible: straddles the 2048 one-shot bound and a 256-tile edge
    [InlineData(16, 4, 128, 77, 2500, 2577)]    // 77 rows: 3 blocks, the last partial
    [InlineData(16, 4, 128, 2300, 0, 2300)]     // prefill from position 0 crossing the bound inside the forward
    [InlineData(16, 4, 128, 130, 4000, 4500)]   // cache longer than the last query position
    [InlineData(8, 8, 64, 200, 2048, 2248)]     // MHA
    [InlineData(6, 2, 256, 150, 3000, 3150)]    // hd256: tile size 128
    // #1927 packed-tile shapes: Qwen3-4B (32/8/128), group 7 (14/2/64), seqKv just past the bound / a tile edge / odd lengths.
    [InlineData(32, 8, 128, 64, 1985, 2049)]    // one tiled row block, last tile holds a single key
    [InlineData(32, 8, 128, 2304, 0, 2304)]     // from position 0, 9 tiles
    [InlineData(32, 8, 128, 100, 3997, 4097)]   // 17 tiles, last tile 1 key
    [InlineData(14, 2, 64, 90, 8101, 8191)]     // GQA group 7, hd64, 32 tiles, last partial
    public void Blocked_IsBitIdentical_ToPerRowKernel(int numHeads, int numKvHeads, int headDim, int seqQ, int positionOffset, int seqKv)
    {
        bool savedPack = Attention.BlockedPackEnabled;
        try
        {
            // Both tile sources of the blocked kernel (dense packed copy, default; in-place strided) against the per-row oracle.
            foreach (bool pack in new[] { true, false })
            {
                Attention.BlockedPackEnabled = pack;
                AssertBlockedMatchesPerRow(numHeads, numKvHeads, headDim, seqQ, positionOffset, seqKv, pack);
            }
        }
        finally { Attention.BlockedPackEnabled = savedPack; }
    }

    private void AssertBlockedMatchesPerRow(int numHeads, int numKvHeads, int headDim, int seqQ, int positionOffset, int seqKv, bool pack)
    {
        float[] q = Random(seqQ * numHeads * headDim, 1), k = Random(seqKv * numKvHeads * headDim, 2), v = Random(seqKv * numKvHeads * headDim, 3);
        // Scale the queries so the softmax is peaked (large score range): a wrong rescale or boundary moves the output far.
        for (int i = 0; i < q.Length; i++) q[i] *= 4f;
        float[] expected = new float[q.Length];
        Attention.Execute(q, k, v, expected, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

        float* outP = (float*)NativeMemory.AlignedAlloc((nuint)(q.Length * sizeof(float)), 64);
        try
        {
            new Span<float>(outP, q.Length).Fill(float.NaN);
            long before = Attention.BlockedPrefillCount;
            fixed (float* qp = q, kp = k, vp = v)
                Attention.Execute(qp, kp, vp, outP, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, _pool);
            Assert.True(Attention.BlockedPrefillCount > before, "the blocked kernel did not run");

            int mismatches = 0; float maxAbs = 0;
            for (int i = 0; i < q.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(outP[i]))
                {
                    mismatches++;
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[i] - outP[i]));
                }
            }
            Assert.True(mismatches == 0, $"pack={pack}: {mismatches} of {q.Length} outputs differ (max |d| {maxAbs:E3})");
        }
        finally
        {
            NativeMemory.AlignedFree(outP);
        }
    }

    /// <summary>
    /// In-process timing of the two kernels on the LFM3 shape at depth (run with <c>DOTLLM_ATTN_BENCH=1</c>): a 1024-token chunk whose
    /// KV ends at 16K and at 32K, all cores, interleaved old/new x 3. Prints ms per call; indicative only on a contended box.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Blocked_Timing_Lfm3Shape()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_ATTN_BENCH") != "1", "set DOTLLM_ATTN_BENCH=1");
        const int numHeads = 16, numKvHeads = 4, headDim = 128, seqQ = 1024;
        using var pool = new ComputeThreadPool(Environment.ProcessorCount);
        foreach (int seqKv in new[] { 16384, 32768 })
        {
            int positionOffset = seqKv - seqQ;
            float[] q = Random(seqQ * numHeads * headDim, 1), k = Random(seqKv * numKvHeads * headDim, 2), v = Random(seqKv * numKvHeads * headDim, 3);
            float* outP = (float*)NativeMemory.AlignedAlloc((nuint)(q.Length * sizeof(float)), 64);
            try
            {
                fixed (float* qp = q, kp = k, vp = v)
                {
                    for (int rep = 0; rep < 3; rep++)
                    {
                        foreach (bool blocked in new[] { false, true })
                        {
                            Attention.BlockedPrefillEnabled = blocked;
                            var sw = System.Diagnostics.Stopwatch.StartNew();
                            Attention.Execute(qp, kp, vp, outP, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool);
                            sw.Stop();
                            Console.WriteLine($"[attn-bench] seqKv={seqKv} seqQ={seqQ} {(blocked ? "blocked" : "per-row ")} {sw.Elapsed.TotalMilliseconds:F0} ms");
                        }
                    }
                }
            }
            finally
            {
                Attention.BlockedPrefillEnabled = true;
                NativeMemory.AlignedFree(outP);
            }
        }
    }

    /// <summary>
    /// #1927 E1 kill test (run with <c>DOTLLM_ATTN_BENCH=1</c>; threads from <c>DOTLLM_ATTN_BENCH_THREADS</c>, default 16): Qwen3-4B
    /// shape (32/8/hd128), a 512-token chunk whose KV ends at 8,192 (and at 7,600 from position 0 over 1,024 rows), packed vs in-place
    /// tiles, interleaved and position-balanced, 6 rounds. Prints ms per call per arm and the min.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Blocked_PackTiming_Qwen3Shape()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_ATTN_BENCH") != "1", "set DOTLLM_ATTN_BENCH=1");
        const int numHeads = 32, numKvHeads = 8, headDim = 128;
        int threads = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_ATTN_BENCH_THREADS"), out int t) ? t : 16;
        using var pool = new ComputeThreadPool(threads);
        bool saved = Attention.BlockedPackEnabled;
        foreach ((int seqQ, int seqKv) in new[] { (512, 8192), (1024, 7600) })
        {
            int positionOffset = seqKv - seqQ;
            float[] q = Random(seqQ * numHeads * headDim, 1), k = Random(seqKv * numKvHeads * headDim, 2), v = Random(seqKv * numKvHeads * headDim, 3);
            float* outP = (float*)NativeMemory.AlignedAlloc((nuint)(q.Length * sizeof(float)), 64);
            var ms = new Dictionary<bool, List<double>> { [true] = new(), [false] = new() };
            try
            {
                fixed (float* qp = q, kp = k, vp = v)
                {
                    Attention.Execute(qp, kp, vp, outP, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool);   // warm-up
                    for (int rep = 0; rep < 6; rep++)
                    {
                        foreach (bool pack in rep % 2 == 0 ? new[] { false, true } : new[] { true, false })
                        {
                            Attention.BlockedPackEnabled = pack;
                            var sw = System.Diagnostics.Stopwatch.StartNew();
                            Attention.Execute(qp, kp, vp, outP, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool);
                            sw.Stop();
                            ms[pack].Add(sw.Elapsed.TotalMilliseconds);
                        }
                    }
                }
            }
            finally
            {
                Attention.BlockedPackEnabled = saved;
                NativeMemory.AlignedFree(outP);
            }
            foreach (bool pack in new[] { false, true })
            {
                var l = new List<double>(ms[pack]); l.Sort();
                Console.WriteLine($"[attn-pack-bench] threads={threads} seqQ={seqQ} seqKv={seqKv} {(pack ? "packed  " : "in-place")} " +
                                  $"min {l[0]:F1} median {(l[2] + l[3]) / 2:F1} max {l[^1]:F1} ms  all [{string.Join(", ", ms[pack].Select(x => x.ToString("F1")))}]");
            }
        }
    }

    [Fact]
    public void Blocked_DoesNotRun_ForSlidingWindowOrShortRows()
    {
        const int numHeads = 16, numKvHeads = 4, headDim = 128, seqQ = 64;
        float[] q = Random(seqQ * numHeads * headDim, 1), k = Random(3000 * numKvHeads * headDim, 2), v = Random(3000 * numKvHeads * headDim, 3);
        float* outP = (float*)NativeMemory.AlignedAlloc((nuint)(q.Length * sizeof(float)), 64);
        try
        {
            long before = Attention.BlockedPrefillCount;
            fixed (float* qp = q, kp = k, vp = v)
            {
                Attention.Execute(qp, kp, vp, outP, seqQ, 3000, numHeads, numKvHeads, headDim, 2936, _pool, slidingWindowSize: 256);
                Attention.Execute(qp, kp, vp, outP, seqQ, 2000, numHeads, numKvHeads, headDim, 1936, _pool);
            }
            Assert.Equal(before, Attention.BlockedPrefillCount);
        }
        finally
        {
            NativeMemory.AlignedFree(outP);
        }
    }
}
