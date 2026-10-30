using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>#1404: F16 GEMM (converted-tile FMA kernel) vs a double-precision scalar reference.</summary>
public sealed unsafe class GemmF16FmaTileTests
{
    [Theory]
    [InlineData(4, 8, 2)]
    [InlineData(7, 13, 3)]       // m%4, k%8, odd n
    [InlineData(387, 391, 7)]
    [InlineData(768, 768, 64)]
    [InlineData(5, 1024, 1)]     // n==1 -> GEMV path
    public void GemmF16_MatchesDoubleReference(int m, int k, int n)
    {
        var rng = new Random(7);
        Half* w = (Half*)NativeMemory.AlignedAlloc((nuint)((long)m * k * 2), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * 4), 64);
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        try
        {
            for (long i = 0; i < (long)m * k; i++) w[i] = (Half)(rng.NextSingle() * 2f - 1f);
            for (long i = 0; i < (long)n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;
            MatMul.GemmF16((nint)w, b, c, m, k, n);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                {
                    double acc = 0;
                    for (int i = 0; i < k; i++) acc += (double)(float)w[(long)r * k + i] * b[(long)t * k + i];
                    Assert.True(Math.Abs(acc - c[(long)t * m + r]) <= 1e-4 * Math.Sqrt(k),
                        $"t={t} r={r} ref={acc} got={c[(long)t * m + r]}");
                }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(b); NativeMemory.AlignedFree(c); }
    }
}
