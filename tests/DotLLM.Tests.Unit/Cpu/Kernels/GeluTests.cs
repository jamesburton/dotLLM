using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>#1403: vectorised/parallel tanh-GELU against the exact double-precision tanh GELU and the old scalar loop.</summary>
public sealed class GeluTests(ITestOutputHelper output)
{
    private static double Exact(double x) => 0.5 * x * (1.0 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

    private static float[] Inputs(int n)
    {
        var rng = new Random(1403);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * (i % 7 == 0 ? 40 : 6));
        if (n > 8) { x[0] = 0; x[1] = -0f; x[2] = 100f; x[3] = -100f; x[4] = 1e-8f; x[5] = -1e-8f; x[6] = 9f; x[7] = -9f; }
        return x;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(37)]
    [InlineData(512)]
    [InlineData(100_003)]   // > MinParallel and not a multiple of the tile
    public unsafe void Matches_exact_tanh_gelu(int n)
    {
        var x = Inputs(n);
        double worstAbs = 0;
        foreach (int threads in new[] { 1, 6 })
        {
            using var pool = threads > 1 ? new ComputeThreadPool(threads) : null;
            var y = (float[])x.Clone();
            fixed (float* p = y) Gelu.ExecuteTanh(p, n, pool);
            for (int i = 0; i < n; i++)
            {
                double err = Math.Abs(y[i] - Exact(x[i]));
                worstAbs = Math.Max(worstAbs, err);
                Assert.True(err <= 2e-6 * Math.Max(1.0, Math.Abs(x[i])), $"x={x[i]} got {y[i]} want {Exact(x[i])}");
            }
        }
        output.WriteLine($"n={n} max abs error vs exact = {worstAbs:E3}");
    }

    [Fact]
    public unsafe void Parallel_equals_serial_bitwise()
    {
        int n = 200_011;
        var x = Inputs(n);
        var a = (float[])x.Clone(); var b = (float[])x.Clone();
        using var pool = new ComputeThreadPool(8);
        fixed (float* pa = a, pb = b) { Gelu.ExecuteTanh(pa, n, null); Gelu.ExecuteTanh(pb, n, pool); }
        Assert.True(a.AsSpan().SequenceEqual(b));
    }

    [Fact]
    public void Scalar_reference_tracks_exact()
    {
        var x = Inputs(5000);
        var y = new float[x.Length];
        Gelu.ExecuteTanhScalar(x, y);
        for (int i = 0; i < x.Length; i++)
            Assert.True(Math.Abs(y[i] - Exact(x[i])) <= 2e-6 * Math.Max(1.0, Math.Abs(x[i])));
    }
}
