using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1927 E3d: the Q4_K row-lane GEMM and the row-lane arithmetic (RLA) every other Q4_K kernel follows when it is on (the default).
/// The oracle is <see cref="MatMul.Q4KRlaDot1Scalar"/> (plain integer loops, the RLA float recurrence): every path, at every batch
/// size and position in the batch, must give a row exactly its bits. A float64 dot bounds the accuracy.
/// </summary>
public sealed unsafe class Q4KRowLaneTests
{
    private const int BlockBytes = 144, Q8Bytes = 292;

    private sealed class Fixture : IDisposable
    {
        public readonly byte* W;
        public readonly byte* X;     // Q8_K rows [n]
        public readonly float* XF;   // the F32 rows they were quantized from
        public readonly int M, K, N, Sb;

        public Fixture(int m, int k, int n, int seed)
        {
            M = m; K = k; N = n; Sb = k / 256;
            var rng = new Random(seed);
            W = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * Sb * BlockBytes), 64);
            X = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * Sb * Q8Bytes), 64);
            XF = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * 4), 64);
            for (long i = 0; i < (long)m * Sb * BlockBytes; i++) W[i] = (byte)rng.Next(256);
            for (long b = 0; b < (long)m * Sb; b++)
            {
                *(Half*)(W + b * BlockBytes) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                *(Half*)(W + b * BlockBytes + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
            }
            for (long i = 0; i < (long)n * k; i++) XF[i] = (float)((rng.NextDouble() - 0.5) * 4);
            for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_K(XF + (long)t * k, X + (long)t * Sb * Q8Bytes, k);
        }

        public byte* Row(int r) => W + (long)r * Sb * BlockBytes;
        public byte* Tok(int t) => X + (long)t * Sb * Q8Bytes;

        public void Dispose() { NativeMemory.AlignedFree(W); NativeMemory.AlignedFree(X); NativeMemory.AlignedFree(XF); }
    }

    private static double RefDot(byte* wRow, byte* xRow, int k)
    {
        var wf = new float[k];
        Dequantize.DequantizeQ4_KScalar((nint)wRow, k, wf);
        double s = 0;
        for (int b = 0; b < k / 256; b++)
        {
            byte* q = xRow + b * Q8Bytes;
            float d = *(float*)q;
            for (int i = 0; i < 256; i++) s += (double)wf[b * 256 + i] * d * (sbyte)q[4 + i];
        }
        return s;
    }

    private static void SkipUnlessRla()
    {
        Skip.IfNot(MatMul.Q4KRowLaneIsa, "row-lane kernel needs AVX2 + SSE4.1 + FMA");
        Skip.IfNot(MatMul.Q4KRlaActive, "DOTLLM_CPU_Q4K_ROWLANE=0: the R4 arithmetic is in force");
    }

    /// <summary>The row-lane GEMM (one- and two-token passes) is bit-identical to the scalar RLA oracle.</summary>
    [SkippableTheory]
    [InlineData(64, 512, 5)]
    [InlineData(40, 2560, 17)]
    public void RowLane_IsBitIdenticalToTheScalarRlaOracle(int m, int k, int n)
    {
        Skip.IfNot(MatMul.Q4KRowLaneIsa);
        using var f = new Fixture(m, k, n, 1927 + m);
        float* lane = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * 4), 64);
        try
        {
            MatMul.GemmQ4KRowLane8(f.W, f.X, lane, m, f.Sb, n, null);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                {
                    float o = MatMul.Q4KRlaDot1Scalar(f.Row(r), f.Tok(t), f.Sb, BlockBytes);
                    Assert.True(BitConverter.SingleToInt32Bits(o) == BitConverter.SingleToInt32Bits(lane[t * m + r]),
                        $"row {r} token {t}: row-lane {lane[t * m + r]:R} vs oracle {o:R}");
                }
        }
        finally { NativeMemory.AlignedFree(lane); }
    }

    /// <summary>RLA is at least as close to float64 as the R4 arithmetic it replaces (measured: 1.5e-6 / 2.3e-5 vs 2.0e-5 / 2.8e-5).</summary>
    [SkippableTheory]
    [InlineData(64, 512, 5)]
    [InlineData(40, 2560, 17)]
    public void Rla_IsCloserToFloat64_ThanTheR4Arithmetic(int m, int k, int n)
    {
        Skip.IfNot(MatMul.FastKQuantIsa);
        using var f = new Fixture(m, k, n, 1927 + m);
        var r4 = new float[4];
        double worstRla = 0, worstR4 = 0;
        for (int t = 0; t < n; t++)
            for (int r = 0; r + 3 < m; r += 4)
            {
                fixed (float* p = r4) MatMul.Q4KRows4Tokens1(f.Row(r), f.Row(r + 1), f.Row(r + 2), f.Row(r + 3), f.Tok(t), f.Sb, BlockBytes, p);
                for (int i = 0; i < 4; i++)
                {
                    double refv = RefDot(f.Row(r + i), f.Tok(t), k), scale = Math.Max(1e-3, Math.Abs(refv));
                    worstRla = Math.Max(worstRla, Math.Abs(MatMul.Q4KRlaDot1Scalar(f.Row(r + i), f.Tok(t), f.Sb, BlockBytes) - refv) / scale);
                    worstR4 = Math.Max(worstR4, Math.Abs(r4[i] - refv) / scale);
                }
            }
        Console.WriteLine($"[q4k-rla] m={m} k={k} n={n}: max rel err vs f64 RLA {worstRla:E2}, R4 {worstR4:E2}");
        Assert.True(worstRla <= worstR4, $"RLA {worstRla:E2} vs R4 {worstR4:E2}");
    }

    /// <summary>
    /// The batch-invariance gate (#530, #1927): a token's output rows are bit-identical whatever path and batch serve it: the row-major
    /// ComputeRows (decode / fused decode), the F32-input GEMV, the R4 GEMM at n = 1 and 7 (4-token and single-token kernels), and
    /// the row-lane GEMM at n = 16, 17 and 64 (paired and tail positions), serial and pooled. m = 44 (m % 8 = 4) cannot take the
    /// row-lane kernel and must still give the same bits through the R4 kernels.
    /// </summary>
    [SkippableTheory]
    [InlineData(48, 512)]
    [InlineData(44, 512)]
    [InlineData(64, 2560)]
    public void Row_IsBitIdentical_AcrossPathsBatchSizesAndPositions(int m, int k)
    {
        SkipUnlessRla();
        const int nMax = 64;
        using var f = new Fixture(m, k, nMax, 530 + m);
        int sb = f.Sb;
        // Oracle per (token, row).
        var oracle = new float[nMax * m];
        for (int t = 0; t < nMax; t++)
            for (int r = 0; r < m; r++) oracle[t * m + r] = MatMul.Q4KRlaDot1Scalar(f.Row(r), f.Tok(t), sb, BlockBytes);

        float* c = (float*)NativeMemory.AlignedAlloc((nuint)(nMax * m * 4), 64);
        using var pool = new ComputeThreadPool(4);
        try
        {
            void Check(string what, int t0, int n)
            {
                for (int t = 0; t < n; t++)
                    for (int r = 0; r < m; r++)
                    {
                        float want = oracle[(t0 + t) * m + r], got = c[t * m + r];
                        Assert.True(BitConverter.SingleToInt32Bits(want) == BitConverter.SingleToInt32Bits(got),
                            $"{what}: token {t0 + t} row {r}: {got:R} vs oracle {want:R}");
                    }
            }

            // n = 1 paths for every token.
            for (int t = 0; t < nMax; t += 9)
            {
                MatMul.ComputeRowsQ4_K(f.W, f.Tok(t), c, m, sb);
                Check("ComputeRowsQ4_K n=1", t, 1);
                MatMul.GemvQ4_K(f.W, f.XF + (long)t * k, c, m, k, pool);
                Check("GemvQ4_K (pooled)", t, 1);
                MatMul.GemmR4TiledQ4_K(f.W, null, f.Tok(t), c, m / 4, m % 4, sb, m, k, 1, null);
                Check("R4 GEMM n=1", t, 1);
            }

            // Batches: different starting tokens, so a row's batch-mates and position differ.
            foreach ((int t0, int n) in new[] { (3, 7), (40, 7), (0, 16), (47, 16), (5, 17), (0, 64) })
            {
                foreach (ComputeThreadPool? p in new[] { null, pool })
                {
                    long before = MatMul.t_q4kRowLaneCalls;
                    MatMul.GemmR4TiledQ4_K(f.W, null, f.Tok(t0), c, m / 4, m % 4, sb, m, k, n, p);
                    bool rowLane = MatMul.t_q4kRowLaneCalls > before;
                    Assert.Equal(n >= 16 && m % 8 == 0, rowLane);
                    Check($"GEMM n={n} from token {t0} pool={(p is null ? 0 : 4)} rowLane={rowLane}", t0, n);
                }
            }
        }
        finally { NativeMemory.AlignedFree(c); }
    }

    /// <summary>#1927 review I1: the row-lane tile has an owner and is freed by <see cref="CpuKernelScratch.Trim"/>, inline and on workers.</summary>
    [SkippableFact]
    public void RowLaneTile_IsFreedByTrim_OnTheCallerAndOnPoolWorkers()
    {
        Skip.IfNot(MatMul.Q4KRowLaneIsa);
        using var f = new Fixture(64, 512, 16, 7);
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)(16 * 64 * 4), 64);
        try
        {
            MatMul.GemmQ4KRowLane8(f.W, f.X, c, 64, f.Sb, 16, null);
            Assert.True(MatMul.RowLaneTileBytesOnThisThread > 0);
            CpuKernelScratch.Trim(null);
            Assert.Equal(0, MatMul.RowLaneTileBytesOnThisThread);

            using var pool = new ComputeThreadPool(4);
            MatMul.GemmQ4KRowLane8(f.W, f.X, c, 64, f.Sb, 16, pool);
            s_liveWorkers = 0;
            pool.Dispatch(0, &CountLiveTiles);
            Assert.True(s_liveWorkers > 0, "no worker holds a tile after a pooled GEMM: the probe cannot see them");
            CpuKernelScratch.Trim(pool);
            s_liveWorkers = 0;
            pool.Dispatch(0, &CountLiveTiles);
            Assert.Equal(0, s_liveWorkers);
        }
        finally { NativeMemory.AlignedFree(c); }
    }

    private static int s_liveWorkers;

    private static void CountLiveTiles(nint _, int __, int ___)
    {
        if (MatMul.RowLaneTileBytesOnThisThread > 0) Interlocked.Increment(ref s_liveWorkers);
    }

    /// <summary>
    /// E3d kill test (DOTLLM_Q4K_BENCH=1): m = 12800, k = 4096, n = 512 / 2048, 1 and 16 threads, 6 interleaved rounds, row-lane vs
    /// the R4 GEMM (in whichever arithmetic is in force: run with DOTLLM_CPU_Q4K_ROWLANE=0 for the original R4 kernels).
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void RowLane_VsR4_Timing()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_Q4K_BENCH") != "1", "set DOTLLM_Q4K_BENCH=1");
        const int m = 12800, k = 4096, nMax = 2048;
        using var f = new Fixture(m, k, nMax, 19270);
        int sb = f.Sb;
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)((long)nMax * m * 4), 64);
        try
        {
            foreach (int threads in new[] { 1, 16 })
            foreach (int n in new[] { 512, 2048 })
            {
                if (threads == 1 && n > 512) continue;
                using ComputeThreadPool? pool = threads > 1 ? new ComputeThreadPool(threads) : null;
                bool saved = MatMul.Q4KRowLaneEnabled;
                var ms = new[] { new List<double>(), new List<double>() };
                try
                {
                    for (int r = -1; r < 6; r++)
                        foreach (int arm in r % 2 == 0 ? new[] { 0, 1 } : new[] { 1, 0 })
                        {
                            MatMul.Q4KRowLaneEnabled = arm == 1;   // arm 0: the original R4 kernels and arithmetic
                            var sw = Stopwatch.StartNew();
                            MatMul.GemmR4TiledQ4_K(f.W, null, f.X, c, m / 4, 0, sb, m, k, n, pool);
                            if (r >= 0) ms[arm].Add(sw.Elapsed.TotalMilliseconds);
                        }
                }
                finally { MatMul.Q4KRowLaneEnabled = saved; }
                var p0 = ms[0].OrderBy(v => v).ToArray(); var l = ms[1].OrderBy(v => v).ToArray();
                double mp = (p0[2] + p0[3]) / 2, ml = (l[2] + l[3]) / 2;
                Console.WriteLine($"[q4k-rowlane-bench] threads={threads,2} n={n,4} R4 median {mp:F1} [{p0[0]:F1}-{p0[^1]:F1}] ms  " +
                                  $"row-lane median {ml:F1} [{l[0]:F1}-{l[^1]:F1}] ms  ratio {mp / ml:F3}x");
            }
        }
        finally { NativeMemory.AlignedFree(c); }
    }

    /// <summary>
    /// The n = 1 decode cost of the batch-invariant design (DOTLLM_Q4K_BENCH=1): Q4_K GEMV through the production decode entry
    /// (<see cref="MatMul.GemvQ4_K(byte*, float*, float*, int, int, ComputeThreadPool?)"/>), RLA vs the R4 arithmetic, at two shapes
    /// (an L3-resident 4096 x 2560 projection and a 12800 x 4096 one past the L3), 14 threads, 9 interleaved rounds of 64 calls.
    /// Toggles the process-wide switch, so run it alone.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Gemv_N1_Rla_VsR4_Timing()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_Q4K_BENCH") != "1", "set DOTLLM_Q4K_BENCH=1");
        using var pool = new ComputeThreadPool(14);
        bool saved = MatMul.Q4KRowLaneEnabled;
        try
        {
            foreach ((int m, int k) in new[] { (4096, 2560), (12800, 4096), (9728, 2560) })
            {
                using var f = new Fixture(m, k, 1, m ^ k);
                float* y = (float*)NativeMemory.AlignedAlloc((nuint)(m * 4), 64);
                var us = new[] { new List<double>(), new List<double>() };   // 0 = R4 arithmetic, 1 = RLA
                for (int r = -1; r < 9; r++)
                    foreach (int arm in r % 2 == 0 ? new[] { 0, 1 } : new[] { 1, 0 })
                    {
                        MatMul.Q4KRowLaneEnabled = arm == 1;
                        var sw = Stopwatch.StartNew();
                        for (int i = 0; i < 64; i++) MatMul.GemvQ4_K(f.W, f.XF, y, m, k, pool);
                        if (r >= 0) us[arm].Add(sw.Elapsed.TotalMilliseconds * 1000 / 64);
                    }
                NativeMemory.AlignedFree(y);
                var a = us[0].OrderBy(v => v).ToArray(); var b = us[1].OrderBy(v => v).ToArray();
                Console.WriteLine($"[q4k-gemv-n1] m={m} k={k}: R4 median {a[4]:F1} [{a[0]:F1}-{a[^1]:F1}] us  RLA median {b[4]:F1} [{b[0]:F1}-{b[^1]:F1}] us  " +
                                  $"RLA/R4 time {b[4] / a[4]:F3}");
            }
        }
        finally { MatMul.Q4KRowLaneEnabled = saved; }
    }
}
