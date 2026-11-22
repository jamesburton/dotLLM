using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1927 E3d prototype: the 8-row row-lane Q4_K GEMM (native layout read in place, transposed tile per worker). Not bit-identical to
/// the production kernel, so the oracle is a float64 dot of the dequantised weights with the Q8_K activations (both kernels are held
/// to it, and to each other), plus a kill-criterion timing test.
/// </summary>
public sealed unsafe class Q4KRowLanePrototypeTests
{
    private const int BlockBytes = 144, Q8Bytes = 292;

    private static (nint w, nint x) Fixture(int m, int k, int n, int seed)
    {
        int sb = k / 256;
        var rng = new Random(seed);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * sb * BlockBytes), 64);
        byte* x = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * sb * Q8Bytes), 64);
        for (long i = 0; i < (long)m * sb * BlockBytes; i++) w[i] = (byte)rng.Next(256);
        for (long b = 0; b < (long)m * sb; b++)
        {
            *(Half*)(w + b * BlockBytes) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
            *(Half*)(w + b * BlockBytes + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
        }
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        for (int t = 0; t < n; t++)
        {
            for (int i = 0; i < k; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4);
            MatMul.QuantizeF32ToQ8_K(src, x + (long)t * sb * Q8Bytes, k);
        }
        NativeMemory.AlignedFree(src);
        return ((nint)w, (nint)x);
    }

    /// <summary>float64 reference: dequantise row (Q4_K) and the Q8_K row, dot in double.</summary>
    private static double RefDot(byte* wRow, byte* xRow, int k)
    {
        var wf = new float[k];
        Dequantize.DequantizeQ4_KScalar((nint)wRow, k, wf);
        double s = 0;
        int sb = k / 256;
        for (int b = 0; b < sb; b++)
        {
            byte* q = xRow + b * Q8Bytes;
            float d = *(float*)q;
            for (int i = 0; i < 256; i++) s += (double)wf[b * 256 + i] * d * (sbyte)q[4 + i];
        }
        return s;
    }

    [SkippableTheory]
    [InlineData(64, 512, 5)]
    [InlineData(40, 2560, 17)]
    public void RowLane_MatchesFloat64Reference_AsCloseAsTheProductionKernel(int m, int k, int n)
    {
        Skip.IfNot(MatMul.Q4KRowLaneIsa);
        var (wp, xp) = Fixture(m, k, n, 1927 + m);
        byte* w = (byte*)wp, x = (byte*)xp;
        int sb = k / 256;
        float* prod = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * 4), 64);
        float* lane = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * 4), 64);
        try
        {
            MatMul.GemmR4TiledQ4_K(w, null, x, prod, m / 4, m % 4, sb, m, k, n, null);
            MatMul.GemmQ4KRowLane8(w, x, lane, m, sb, n, null);
            double worstLane = 0, worstProd = 0, worstPair = 0;
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                {
                    double refv = RefDot(w + (long)r * sb * BlockBytes, x + (long)t * sb * Q8Bytes, k);
                    double scale = Math.Max(1e-3, Math.Abs(refv));
                    worstLane = Math.Max(worstLane, Math.Abs(lane[t * m + r] - refv) / scale);
                    worstProd = Math.Max(worstProd, Math.Abs(prod[t * m + r] - refv) / scale);
                    worstPair = Math.Max(worstPair, Math.Abs(prod[t * m + r] - lane[t * m + r]) / scale);
                }
            Console.WriteLine($"[q4k-rowlane] m={m} k={k} n={n}: max rel err vs f64 lane {worstLane:E2} prod {worstProd:E2}; lane vs prod {worstPair:E2}");
            Assert.True(worstLane <= Math.Max(2 * worstProd, 1e-5), $"row-lane {worstLane:E2} vs production {worstProd:E2}");
            Assert.True(worstPair <= 1e-4, $"row-lane vs production {worstPair:E2}");
        }
        finally
        {
            NativeMemory.AlignedFree(prod); NativeMemory.AlignedFree(lane);
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x);
        }
    }

    /// <summary>
    /// The production entry: with the switch on, a Q4_K GEMM of n >= 16 tokens (m % 8 == 0, no tail rows) takes the row-lane kernel
    /// and matches it bit for bit; n = 15 keeps the R4 kernel bit for bit (so decode / verify / short chunks are untouched, #530).
    /// </summary>
    [SkippableTheory]
    [InlineData(16, true)]
    [InlineData(15, false)]
    public void Entry_EngagesFrom16Tokens_AndOtherwiseKeepsTheR4Kernel(int n, bool expectRowLane)
    {
        Skip.IfNot(MatMul.Q4KRowLaneIsa);
        const int m = 48, k = 512;
        int sb = k / 256;
        var (wp, xp) = Fixture(m, k, n, 77 + n);
        byte* w = (byte*)wp, x = (byte*)xp;
        float* viaEntry = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * 4), 64);
        float* oracle = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * 4), 64);
        bool? saved = MatMul.t_q4kRowLaneOverride;
        try
        {
            MatMul.t_q4kRowLaneOverride = false;
            if (expectRowLane) MatMul.GemmQ4KRowLane8(w, x, oracle, m, sb, n, null);
            else MatMul.GemmR4TiledQ4_K(w, null, x, oracle, m / 4, 0, sb, m, k, n, null);
            MatMul.t_q4kRowLaneOverride = true;
            long before = MatMul.Q4KRowLaneCalls;
            using var pool = new ComputeThreadPool(4);
            MatMul.GemmR4TiledQ4_K(w, null, x, viaEntry, m / 4, 0, sb, m, k, n, pool);
            Assert.Equal(expectRowLane, MatMul.Q4KRowLaneCalls > before);
            for (int i = 0; i < n * m; i++)
                Assert.True(BitConverter.SingleToInt32Bits(oracle[i]) == BitConverter.SingleToInt32Bits(viaEntry[i]), $"output {i} differs");
        }
        finally
        {
            MatMul.t_q4kRowLaneOverride = saved;
            NativeMemory.AlignedFree(viaEntry); NativeMemory.AlignedFree(oracle);
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x);
        }
    }

    /// <summary>
    /// E3d kill test (DOTLLM_Q4K_BENCH=1): m = 12800, k = 4096, n = 512 (and 2048), 1 and 16 threads, 6 interleaved position-balanced
    /// rounds, row-lane prototype vs the production R4 GEMM. Kill below 1.3x.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void RowLane_VsProduction_Timing()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_Q4K_BENCH") != "1", "set DOTLLM_Q4K_BENCH=1");
        const int m = 12800, k = 4096, nMax = 2048;
        int sb = k / 256;
        var (wp, xp) = Fixture(m, k, nMax, 19270);
        byte* w = (byte*)wp, x = (byte*)xp;
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)((long)nMax * m * 4), 64);
        try
        {
            foreach (int threads in new[] { 1, 16 })
            foreach (int n in new[] { 512, 2048 })
            {
                if (threads == 1 && n > 512) continue;
                using ComputeThreadPool? pool = threads > 1 ? new ComputeThreadPool(threads) : null;
                MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, 0, sb, m, k, n, pool);
                MatMul.GemmQ4KRowLane8(w, x, c, m, sb, n, pool);
                var ms = new[] { new List<double>(), new List<double>() };   // 0 = production, 1 = row-lane
                for (int r = 0; r < 6; r++)
                    foreach (int arm in r % 2 == 0 ? new[] { 0, 1 } : new[] { 1, 0 })
                    {
                        var sw = Stopwatch.StartNew();
                        if (arm == 0) MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, 0, sb, m, k, n, pool);
                        else MatMul.GemmQ4KRowLane8(w, x, c, m, sb, n, pool);
                        ms[arm].Add(sw.Elapsed.TotalMilliseconds);
                    }
                var p = ms[0].OrderBy(v => v).ToArray(); var l = ms[1].OrderBy(v => v).ToArray();
                double mp = (p[2] + p[3]) / 2, ml = (l[2] + l[3]) / 2;
                Console.WriteLine($"[q4k-rowlane-bench] threads={threads,2} n={n,4} production median {mp:F1} [{p[0]:F1}-{p[^1]:F1}] ms  " +
                                  $"row-lane median {ml:F1} [{l[0]:F1}-{l[^1]:F1}] ms  ratio {mp / ml:F3}x");
            }
        }
        finally { NativeMemory.AlignedFree(c); NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); }
    }
}
