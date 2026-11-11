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
}
