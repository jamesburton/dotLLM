using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477 perf probe: 128-bit (SSSE3) K-quant dots vs the scalar dots they replace on
/// pre-AVX2 hardware, single-threaded GEMV over contiguous rows. Kernels are called directly, so
/// the numbers are meaningful on any SSSE3 box.
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class KQuantSseTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public KQuantSseTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 15;

    [SkippableTheory]
    [InlineData("Q4_K", 2048, 2048)]
    [InlineData("Q4_K", 8192, 2048)]
    [InlineData("Q5_K", 2048, 2048)]
    [InlineData("Q6_K", 2048, 2048)]
    [InlineData("Q6_K", 2048, 8192)]
    public void KQuant_Gemv_Sse128VsScalar(string fmt, int m, int k)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        int bb = fmt switch { "Q4_K" => 144, "Q5_K" => 176, _ => 210 };
        int n = k / 256;
        long rowBytes = (long)n * bb, wBytes = m * rowBytes;
        var rng = new Random(477);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        byte* x = (byte*)NativeMemory.Alloc((nuint)(n * 292));
        float* xf = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        float* y = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (long i = 0; i < wBytes; i++) w[i] = (byte)rng.Next(256);
            for (long r = 0; r < (long)m * n; r++)
            {
                byte* b = w + r * bb;
                if (fmt == "Q6_K") *(Half*)(b + 208) = (Half)0.01f;
                else { *(Half*)b = (Half)0.01f; *(Half*)(b + 2) = (Half)0.005f; }
            }
            for (int i = 0; i < k; i++) xf[i] = rng.NextSingle() * 2f - 1f;
            MatMul.QuantizeF32ToQ8_KScalar(xf, x, k);

            delegate*<byte*, byte*, int, float> scalar = fmt switch
            {
                "Q4_K" => &MatMul.VecDotQ4_K_Q8_KScalar,
                "Q5_K" => &MatMul.VecDotQ5_K_Q8_KScalar,
                _ => &MatMul.VecDotQ6_K_Q8_KScalar,
            };
            delegate*<byte*, byte*, int, float> sse = fmt switch
            {
                "Q4_K" => &MatMul.VecDotQ4_K_Q8_KSse,
                "Q5_K" => &MatMul.VecDotQ5_K_Q8_KSse,
                _ => &MatMul.VecDotQ6_K_Q8_KSse,
            };

            float sink = 0;
            double tScalar = Median(() => { for (int r = 0; r < m; r++) sink += scalar(w + r * rowBytes, x, n); });
            double tSse = Median(() => { for (int r = 0; r < m; r++) sink += sse(w + r * rowBytes, x, n); });
            double nsSb(double ms) => ms * 1e6 / ((double)m * n);
            _output.WriteLine($"[{fmt}] m={m} k={k} weights={wBytes / 1e6:F1} MB (single-threaded, " +
                              $"AVX2={Avx2.IsSupported})");
            _output.WriteLine($"  scalar : {tScalar:F3} ms  {nsSb(tScalar):F1} ns/superblock");
            _output.WriteLine($"  128-bit: {tSse:F3} ms  {nsSb(tSse):F1} ns/superblock  ({tScalar / tSse:F2}x, " +
                              $"{wBytes / (tSse * 1e6):F2} GB/s)  (sink {sink})");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(x); NativeMemory.Free(xf); NativeMemory.Free(y);
        }
    }

    private static double Median(Action body)
    {
        for (int i = 0; i < 5; i++) body();
        double[] ms = new double[Trials];
        for (int t = 0; t < Trials; t++)
        {
            var sw = Stopwatch.StartNew();
            body();
            ms[t] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(ms);
        return ms[Trials / 2];
    }
}
