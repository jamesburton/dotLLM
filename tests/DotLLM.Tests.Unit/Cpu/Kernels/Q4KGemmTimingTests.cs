using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// In-process timing of the R4 tiled Q4_K prefill GEMM on the TwIL-8B gate/up shape (#1675). Run with <c>DOTLLM_Q4K_BENCH=1</c>.
/// Diagnostic only: prints ms per call and per-thread GMAC/s over a thread sweep; indicative on a contended box.
/// </summary>
public sealed unsafe class Q4KGemmTimingTests
{
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Q4K_Gemm_ThreadSweep_TwilGateUpShape()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_Q4K_BENCH") != "1", "set DOTLLM_Q4K_BENCH=1");
        int m = int.Parse(Environment.GetEnvironmentVariable("DOTLLM_Q4K_M") ?? "12800");
        const int k = 4096, n = 512, blockBytes = 144, q8Bytes = 292;
        int sb = k / 256;
        var rng = new Random(1675);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * sb * blockBytes), 64);
        byte* x = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * sb * q8Bytes), 64);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        try
        {
            for (long i = 0; i < (long)m * sb * blockBytes; i++) w[i] = (byte)rng.Next(256);
            for (long b = 0; b < (long)m * sb; b++)
            {
                *(Half*)(w + b * blockBytes) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                *(Half*)(w + b * blockBytes + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
            }
            for (int t = 0; t < n; t++)
            {
                for (int i = 0; i < k; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4);
                MatMul.QuantizeF32ToQ8_K(src, x + (long)t * sb * q8Bytes, k);
            }
            foreach (int threads in new[] { 1, 4, 8, 12, 16, 24, 32 })
            {
                ComputeThreadPool? pool = threads > 1 ? new ComputeThreadPool(threads) : null;
                try
                {
                    MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, m % 4, sb, m, k, n, pool); // warm
                    double[] best = { double.MaxValue, double.MaxValue };
                    for (int r = 0; r < 6; r++)
                    {
                        for (int arm = 0; arm < 2; arm++) // interleaved: 0 = static partition, 1 = dynamic chunks
                        {
                            MatMul.ForceStaticGemmPartition = arm == 0;
                            var sw = Stopwatch.StartNew();
                            MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, m % 4, sb, m, k, n, pool);
                            best[arm] = Math.Min(best[arm], sw.Elapsed.TotalMilliseconds);
                        }
                    }
                    MatMul.ForceStaticGemmPartition = false;
                    double gmac = (double)m * k * n / 1e9;
                    Console.WriteLine($"[q4k-bench] m={m} threads={threads,2} static={best[0],7:F1} ms ({gmac / best[0] * 1000:F0} GMAC/s)  dynamic={best[1],7:F1} ms ({gmac / best[1] * 1000:F0} GMAC/s)");
                }
                finally { pool?.Dispose(); }
            }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(src); NativeMemory.AlignedFree(c); }
    }

    /// <summary>
    /// #1927 E3c kill test (run with <c>DOTLLM_Q4K_BENCH=1</c>): 8-token vs 4-token Q4_K kernel, m = 12800, k = 4096, n = 512 and
    /// 2048, at 1 and 16 threads, 6 interleaved position-balanced rounds; prints min and median ms per arm and the ratio.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Q4K_Gemm_Tok8VsTok4()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_Q4K_BENCH") != "1", "set DOTLLM_Q4K_BENCH=1");
        const int m = 12800, k = 4096, blockBytes = 144, q8Bytes = 292, nMax = 2048;
        int sb = k / 256;
        var rng = new Random(1927);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * sb * blockBytes), 64);
        byte* x = (byte*)NativeMemory.AlignedAlloc((nuint)((long)nMax * sb * q8Bytes), 64);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)((long)nMax * m * 4), 64);
        bool saved = MatMul.Q4KTok8Enabled;
        try
        {
            for (long i = 0; i < (long)m * sb * blockBytes; i++) w[i] = (byte)rng.Next(256);
            for (long b = 0; b < (long)m * sb; b++)
            {
                *(Half*)(w + b * blockBytes) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                *(Half*)(w + b * blockBytes + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
            }
            for (int t = 0; t < nMax; t++)
            {
                for (int i = 0; i < k; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4);
                MatMul.QuantizeF32ToQ8_K(src, x + (long)t * sb * q8Bytes, k);
            }
            foreach (int threads in new[] { 1, 16 })
            foreach (int n in new[] { 512, 2048 })
            {
                if (threads == 1 && n > 512) continue;
                ComputeThreadPool? pool = threads > 1 ? new ComputeThreadPool(threads) : null;
                try
                {
                    MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, m % 4, sb, m, k, n, pool); // warm
                    var ms = new List<double>[] { new(), new() };   // 0 = tok4, 1 = tok8
                    for (int r = 0; r < 6; r++)
                    {
                        foreach (int arm in r % 2 == 0 ? new[] { 0, 1 } : new[] { 1, 0 })
                        {
                            MatMul.Q4KTok8Enabled = arm == 1;
                            var sw = Stopwatch.StartNew();
                            MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, m % 4, sb, m, k, n, pool);
                            ms[arm].Add(sw.Elapsed.TotalMilliseconds);
                        }
                    }
                    var s4 = new List<double>(ms[0]); s4.Sort();
                    var s8 = new List<double>(ms[1]); s8.Sort();
                    double med4 = (s4[2] + s4[3]) / 2, med8 = (s8[2] + s8[3]) / 2;
                    Console.WriteLine($"[q4k-tok8] threads={threads,2} n={n,4} tok4 median {med4,7:F1} [{s4[0]:F1}-{s4[^1]:F1}] ms  " +
                                      $"tok8 median {med8,7:F1} [{s8[0]:F1}-{s8[^1]:F1}] ms  ratio {med4 / med8:F3}x  active={MatMul.Q4KTok8Active}");
                }
                finally { pool?.Dispose(); MatMul.Q4KTok8Enabled = saved; }
            }
        }
        finally
        {
            MatMul.Q4KTok8Enabled = saved;
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(src); NativeMemory.AlignedFree(c);
        }
    }
}
