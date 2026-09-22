using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #497 perf probe: the packed Q2_K / Q3_K × Q8_K dots against each other and against the
/// dequantize-and-dot path they replace, single-threaded GEMV over contiguous rows. Kernels are
/// called directly, so the tier numbers are meaningful on any SSSE3 box; the AVX2 columns are
/// skipped where the ISA is absent (including under <c>DOTNET_EnableAVX=0</c>).
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class LowBitKQuantTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public LowBitKQuantTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 15;

    [Theory]
    [InlineData("Q2_K", 2048, 2048)]
    [InlineData("Q2_K", 8192, 2048)]
    [InlineData("Q3_K", 2048, 2048)]
    [InlineData("Q3_K", 8192, 2048)]
    public void LowBitKQuant_Gemv_TiersVsScalar(string fmt, int m, int k)
    {
        bool isQ2 = fmt == "Q2_K";
        int bb = isQ2 ? 84 : 110;
        int n = k / 256;
        long rowBytes = (long)n * bb, wBytes = m * rowBytes;
        var rng = new Random(497);
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
                if (isQ2) { *(Half*)(b + 80) = (Half)0.01f; *(Half*)(b + 82) = (Half)0.005f; }
                else *(Half*)(b + 108) = (Half)0.01f;
            }
            for (int i = 0; i < k; i++) xf[i] = rng.NextSingle() * 2f - 1f;
            MatMul.QuantizeF32ToQ8_KScalar(xf, x, k);

            delegate*<byte*, byte*, int, float> scalar = isQ2
                ? &MatMul.VecDotQ2_K_Q8_KScalar : &MatMul.VecDotQ3_K_Q8_KScalar;
            delegate*<byte*, byte*, int, float> sse = isQ2
                ? &MatMul.VecDotQ2_K_Q8_KSse : &MatMul.VecDotQ3_K_Q8_KSse;
            delegate*<byte*, byte*, int, float> avx = isQ2
                ? &MatMul.VecDotQ2_K_Q8_KAvx2 : &MatMul.VecDotQ3_K_Q8_KAvx2;
            QuantizationType qt = isQ2 ? QuantizationType.Q2_K : QuantizationType.Q3_K;

            float sink = 0;
            double tScalar = Median(() => { for (int r = 0; r < m; r++) sink += scalar(w + r * rowBytes, x, n); });
            double tSse = Ssse3.IsSupported
                ? Median(() => { for (int r = 0; r < m; r++) sink += sse(w + r * rowBytes, x, n); })
                : double.NaN;
            double tAvx = Avx2.IsSupported
                ? Median(() => { for (int r = 0; r < m; r++) sink += avx(w + r * rowBytes, x, n); })
                : double.NaN;
            // The pre-#497 path: dequantize each weight row to F32, dot in F32, serial.
            double tDeq = Median(() => MatMul.GemvDequantRows(w, qt, xf, y, m, k, null));

            double nsSb(double ms) => ms * 1e6 / ((double)m * n);
            double gbs(double ms) => wBytes / (ms * 1e6);
            _output.WriteLine($"[{fmt}] m={m} k={k} weights={wBytes / 1e6:F1} MB (single-threaded, " +
                              $"SSSE3={Ssse3.IsSupported} AVX2={Avx2.IsSupported})");
            _output.WriteLine($"  dequant+dot : {tDeq,8:F3} ms  {nsSb(tDeq),7:F1} ns/superblock  {gbs(tDeq),6:F2} GB/s");
            _output.WriteLine($"  scalar dot  : {tScalar,8:F3} ms  {nsSb(tScalar),7:F1} ns/superblock  {gbs(tScalar),6:F2} GB/s  ({tDeq / tScalar:F2}x vs dequant)");
            if (Ssse3.IsSupported)
                _output.WriteLine($"  128-bit     : {tSse,8:F3} ms  {nsSb(tSse),7:F1} ns/superblock  {gbs(tSse),6:F2} GB/s  ({tDeq / tSse:F2}x vs dequant, {tScalar / tSse:F2}x vs scalar)");
            if (Avx2.IsSupported)
                _output.WriteLine($"  AVX2        : {tAvx,8:F3} ms  {nsSb(tAvx),7:F1} ns/superblock  {gbs(tAvx),6:F2} GB/s  ({tDeq / tAvx:F2}x vs dequant, {tScalar / tAvx:F2}x vs scalar)");
            _output.WriteLine($"  (sink {sink})");
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
