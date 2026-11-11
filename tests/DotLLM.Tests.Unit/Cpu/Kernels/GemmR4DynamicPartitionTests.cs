using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #1675: the R4 tiled GEMM claims row chunks dynamically instead of splitting statically. The result must be bit-identical
/// to the single-threaded call and to the static partition for every thread count, with and without tail rows, and every output
/// element must be written (the output is pre-filled with NaN so a skipped or double-claimed chunk is visible).
/// </summary>
public sealed unsafe class GemmR4DynamicPartitionTests
{
    private const int Q8KBytes = 292, BlockBytes = 144;

    [SkippableTheory]
    [InlineData(4 * 301, 3, 2)]   // many chunks, 3 tail rows
    [InlineData(4 * 301, 0, 2)]
    [InlineData(4 * 8, 1, 2)]     // above ParallelMinRows, 2 chunks for up to 16 threads
    [InlineData(4 * 200, 2, 120)] // K = 30720: TileGroups < 4 (regression: Math.Clamp min > max)
    public void DynamicPartition_BitIdenticalToSingleThreadAndStatic(int fullRows, int tail, int sb)
    {
        Skip.IfNot(MatMul.FastKQuantIsa, "needs AVX2 + SSE4.1");
        int m = fullRows + tail, k = sb * 256, n = 9;
        var rng = new Random(1675);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * sb * BlockBytes), 64);
        byte* x = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * sb * Q8KBytes), 64);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        long outLen = (long)n * m;
        float* c0 = (float*)NativeMemory.AlignedAlloc((nuint)(outLen * 4), 64);
        float* c1 = (float*)NativeMemory.AlignedAlloc((nuint)(outLen * 4), 64);
        try
        {
            for (long i = 0; i < (long)m * sb * BlockBytes; i++) w[i] = (byte)rng.Next(256);
            for (long b = 0; b < (long)m * sb; b++)
            {
                *(Half*)(w + b * BlockBytes) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
                *(Half*)(w + b * BlockBytes + 2) = (Half)(float)(rng.NextDouble() * 0.02 + 0.001);
            }
            for (int t = 0; t < n; t++)
            {
                for (int i = 0; i < k; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4);
                MatMul.QuantizeF32ToQ8_K(src, x + (long)t * sb * Q8KBytes, k);
            }

            for (long i = 0; i < outLen; i++) c0[i] = float.NaN;
            MatMul.GemmR4TiledQ4_K(w, null, x, c0, m / 4, m % 4, sb, m, k, n, null);

            foreach (int threads in new[] { 2, 3, 8, 16 })
            {
                using var pool = new ComputeThreadPool(threads);
                foreach (bool forceStatic in new[] { true, false })
                {
                    MatMul.ForceStaticGemmPartition = forceStatic;
                    try
                    {
                        for (long i = 0; i < outLen; i++) c1[i] = float.NaN;
                        MatMul.GemmR4TiledQ4_K(w, null, x, c1, m / 4, m % 4, sb, m, k, n, pool);
                    }
                    finally { MatMul.ForceStaticGemmPartition = false; }

                    for (long i = 0; i < outLen; i++)
                        Assert.True(BitConverter.SingleToInt32Bits(c0[i]) == BitConverter.SingleToInt32Bits(c1[i]),
                            $"threads={threads} static={forceStatic} element {i}: {c0[i]} vs {c1[i]}");
                }
            }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(src); NativeMemory.AlignedFree(c0); NativeMemory.AlignedFree(c1); }
    }
}
