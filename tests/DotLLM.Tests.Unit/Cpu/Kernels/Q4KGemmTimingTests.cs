using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// In-process timing of the R4 tiled K-quant prefill GEMM on TwIL-8B shapes (#1675). Run with <c>DOTLLM_Q4K_BENCH=1</c>.
/// Diagnostic only: prints ms per call and GMAC/s; indicative on a contended box.
/// </summary>
/// <remarks>
/// Knobs: <c>DOTLLM_Q4K_M</c> / <c>DOTLLM_Q4K_K</c> (shape, default the gate/up shape 12800 x 4096), <c>DOTLLM_Q4K_N</c> (tokens, 512),
/// <c>DOTLLM_Q4K_QUANT</c> (<c>Q4_K</c> or <c>Q6_K</c>), <c>DOTLLM_Q4K_THREADS</c> (comma list, default 1,16).
/// Arms, interleaved per rep: <c>base</c> = the current 4-row x 4-token kernel; <c>tile</c> = the #1675 8-row tile (when the quant has one).
/// </remarks>
public sealed unsafe class Q4KGemmTimingTests
{
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void KQuant_Gemm_TwilShapes()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_Q4K_BENCH") != "1", "set DOTLLM_Q4K_BENCH=1");
        int m = int.Parse(Environment.GetEnvironmentVariable("DOTLLM_Q4K_M") ?? "12800");
        int k = int.Parse(Environment.GetEnvironmentVariable("DOTLLM_Q4K_K") ?? "4096");
        int n = int.Parse(Environment.GetEnvironmentVariable("DOTLLM_Q4K_N") ?? "512");
        string quant = Environment.GetEnvironmentVariable("DOTLLM_Q4K_QUANT") ?? "Q4_K";
        int[] threadList = (Environment.GetEnvironmentVariable("DOTLLM_Q4K_THREADS") ?? "1,16").Split(',').Select(int.Parse).ToArray();
        bool q6 = quant == "Q6_K";
        int blockBytes = q6 ? 210 : 144;
        const int q8Bytes = 292;
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
                if (q6)
                {
                    *(Half*)(w + b * blockBytes + 208) = (Half)(float)(rng.NextDouble() * 0.002 + 0.0001);
                }
                else
                {
                    *(Half*)(w + b * blockBytes) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                    *(Half*)(w + b * blockBytes + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                }
            }
            for (int t = 0; t < n; t++)
            {
                for (int i = 0; i < k; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4);
                MatMul.QuantizeF32ToQ8_K(src, x + (long)t * sb * q8Bytes, k);
            }

            bool hasTile = MatMul.KQuantTileAvailable(q6);
            string[] arms = hasTile ? new[] { "base", "tile" } : new[] { "base" };
            bool savedTile = MatMul.KQuantTile;
            foreach (int threads in threadList)
            {
                ComputeThreadPool? pool = threads > 1 ? new ComputeThreadPool(threads) : null;
                try
                {
                    var best = new double[arms.Length];
                    Array.Fill(best, double.MaxValue);
                    int reps = threads == 1 ? 3 : 6;
                    for (int r = -1; r < reps; r++)
                    {
                        for (int arm = 0; arm < arms.Length; arm++)
                        {
                            MatMul.KQuantTile = arms[arm] == "tile";
                            var sw = Stopwatch.StartNew();
                            if (q6) MatMul.GemmR4TiledQ6_K(w, null, x, c, m / 4, m % 4, sb, m, k, n, pool);
                            else MatMul.GemmR4TiledQ4_K(w, null, x, c, m / 4, m % 4, sb, m, k, n, pool);
                            if (r >= 0) best[arm] = Math.Min(best[arm], sw.Elapsed.TotalMilliseconds);
                        }
                    }
                    double gmac = (double)m * k * n / 1e9;
                    string line = $"[kq-bench] {quant} m={m} k={k} n={n} threads={threads,2}";
                    for (int arm = 0; arm < arms.Length; arm++)
                        line += $"  {arms[arm]}={best[arm],8:F1} ms ({gmac / best[arm] * 1000:F0} GMAC/s)";
                    Console.WriteLine(line);
                }
                finally { pool?.Dispose(); MatMul.KQuantTile = savedTile; }
            }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(src); NativeMemory.AlignedFree(c); }
    }
}
