using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1927 E2a: the software prefetch in the tiled per-row kernel (decode at depth) must not change a bit, and its in-process timing
/// on the Qwen3-4B shape decides whether the page-stride latency diagnosis holds.
/// </summary>
public sealed unsafe class AttentionDecodePrefetchTests
{
    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    [Theory]
    [InlineData(32, 8, 128, 7600)]   // Qwen3-4B, 4 KiB key stride: prefetch engaged
    [InlineData(14, 2, 64, 4097)]    // 512 B stride: below the prefetch threshold
    [InlineData(32, 8, 128, 2049)]   // just past the one-shot bound, last tile 1 key
    public void Decode_WithPrefetch_IsBitIdentical_ToSingleThreadedReference(int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        float[] q = Random(numHeads * headDim, 1), k = Random(seqKv * numKvHeads * headDim, 2), v = Random(seqKv * numKvHeads * headDim, 3);
        for (int i = 0; i < q.Length; i++) q[i] *= 4f;
        float[] expected = new float[q.Length];
        bool saved = Attention.RowPrefetchEnabled;
        try
        {
            Attention.RowPrefetchEnabled = false;
            Attention.Execute(q, k, v, expected, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);

            // Thread-count independence: the head partition changes, the per-head arithmetic must not.
            foreach (int threads in new[] { 2, 7, 14, 16 })   // 1 thread is the reference above
            foreach (bool prefetch in new[] { true, false })
            {
                using var pool = new ComputeThreadPool(threads);
                Attention.RowPrefetchEnabled = prefetch;
                float[] got = new float[q.Length];
                fixed (float* qp = q, kp = k, vp = v, op = got)
                    Attention.Execute(qp, kp, vp, op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool);
                for (int i = 0; i < q.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(got[i]),
                        $"threads={threads} prefetch={prefetch}: output {i} differs ({expected[i]} vs {got[i]})");
            }
        }
        finally { Attention.RowPrefetchEnabled = saved; }
    }

    /// <summary>
    /// Decode attention timing (run with <c>DOTLLM_ATTN_BENCH=1</c>; threads <c>DOTLLM_ATTN_BENCH_THREADS</c>, default 14 = the decode
    /// default at 16): Qwen3-4B shape, one query row over 7,600 and 16,384 keys, prefetch on vs off interleaved, 7 rounds of 36 calls
    /// (one token's layers, each layer a distinct K/V buffer so the history is not L3-resident across calls).
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Decode_PrefetchTiming_Qwen3Shape()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_ATTN_BENCH") != "1", "set DOTLLM_ATTN_BENCH=1");
        const int numHeads = 32, numKvHeads = 8, headDim = 128, layers = 36;
        int threads = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_ATTN_BENCH_THREADS"), out int t) ? t : 14;
        using var pool = new ComputeThreadPool(threads);
        bool saved = Attention.RowPrefetchEnabled;
        foreach (int seqKv in new[] { 7600, 16384 })
        {
            long kvFloats = (long)seqKv * numKvHeads * headDim;
            var kBufs = new float*[layers];
            var vBufs = new float*[layers];
            float* qp = (float*)NativeMemory.AlignedAlloc((nuint)(numHeads * headDim * sizeof(float)), 64);
            float* op = (float*)NativeMemory.AlignedAlloc((nuint)(numHeads * headDim * sizeof(float)), 64);
            var ms = new Dictionary<bool, List<double>> { [true] = new(), [false] = new() };
            try
            {
                var rng = new Random(5);
                for (int i = 0; i < numHeads * headDim; i++) qp[i] = (float)(rng.NextDouble() * 2 - 1);
                for (int l = 0; l < layers; l++)
                {
                    kBufs[l] = (float*)NativeMemory.AlignedAlloc((nuint)(kvFloats * sizeof(float)), 64);
                    vBufs[l] = (float*)NativeMemory.AlignedAlloc((nuint)(kvFloats * sizeof(float)), 64);
                    for (long i = 0; i < kvFloats; i++) { kBufs[l][i] = (float)(rng.NextDouble() * 2 - 1); vBufs[l][i] = (float)(rng.NextDouble() * 2 - 1); }
                }
                void Token()
                {
                    for (int l = 0; l < layers; l++)
                        Attention.Execute(qp, kBufs[l], vBufs[l], op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool);
                }
                Token();   // warm-up
                for (int rep = 0; rep < 7; rep++)
                {
                    foreach (bool prefetch in rep % 2 == 0 ? new[] { false, true } : new[] { true, false })
                    {
                        Attention.RowPrefetchEnabled = prefetch;
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        Token();
                        sw.Stop();
                        ms[prefetch].Add(sw.Elapsed.TotalMilliseconds);
                    }
                }
            }
            finally
            {
                Attention.RowPrefetchEnabled = saved;
                for (int l = 0; l < layers; l++) { if (kBufs[l] != null) NativeMemory.AlignedFree(kBufs[l]); if (vBufs[l] != null) NativeMemory.AlignedFree(vBufs[l]); }
                NativeMemory.AlignedFree(qp);
                NativeMemory.AlignedFree(op);
            }
            foreach (bool prefetch in new[] { false, true })
            {
                var l = new List<double>(ms[prefetch]); l.Sort();
                Console.WriteLine($"[attn-decode-bench] threads={threads} seqKv={seqKv} {(prefetch ? "prefetch" : "plain   ")} ms/token(36 layers) " +
                                  $"median {l[3]:F2} [{l[0]:F2}-{l[^1]:F2}]  all [{string.Join(", ", ms[prefetch].Select(x => x.ToString("F2")))}]");
            }
        }
    }
}
