using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477 perf probe: the 128-bit (SSSE3) Q5_0 x Q8_1 tier vs the scalar tier it replaces on
/// pre-AVX2 hardware, single-threaded, at real decode GEMV shapes, for both the row-major
/// (<c>ComputeRowsQ5_0</c>, what prefill uses) and R4-interleaved (<c>ComputeRowsQ5_0Interleaved</c>, what repacked
/// transformer projections use) layouts. Kernels are called directly, so the numbers are
/// meaningful on any SSSE3 box; the "dispatch" rows go through the public drivers and show the
/// 128-bit tier only under <c>DOTNET_EnableAVX=0</c> (or on real Westmere).
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class Q5_0SseTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public Q5_0SseTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 15;

    [SkippableTheory]
    [InlineData(1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(576, 1536, "SmolLM-135M ffn_down")]
    [InlineData(2048, 2048, "Llama-3.2-1B-sized attn_q")]
    public void Q5_0_Gemv_Sse128VsScalar(int m, int k, string label)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"AVX512BW={Avx512BW.IsSupported}");
        var rng = new Random(477);
        int blocks = k / 32, rowBytes = blocks * 22, xBytes = blocks * 36;
        long wBytes = (long)m * rowBytes;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        float* xf = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        byte* x = (byte*)NativeMemory.Alloc((nuint)xBytes);
        float* y = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        float* yRef = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (long i = 0; i < wBytes; i++) w[i] = (byte)rng.Next(256);
            for (long r = 0; r < (long)m * blocks; r++) *(Half*)(w + r * 22) = (Half)(0.01f + rng.NextSingle() * 0.02f);
            for (int i = 0; i < k; i++) xf[i] = rng.NextSingle() * 2f - 1f;
            MatMul.QuantizeF32ToQ8_1(xf, x, k);
            using var rw = WeightRepacking.RepackR4((nint)w, QuantizationType.Q5_0, m, k);
            byte* r4 = (byte*)rw.Ptr;
            int groups = rw.FullGroupCount;

            double scalarRm = Median(() => { for (int r = 0; r < m; r++) yRef[r] = MatMul.VecDotQ5_0Q8_1Scalar(w + (long)r * rowBytes, x, blocks); });
            double sseRm = Median(() => MatMul.ComputeRowsQ5_0Sse(w, x, y, m, blocks));
            Check(y, yRef, m, "row-major");
            double scalarR4 = Median(() =>
            {
                for (int g = 0; g < groups; g++)
                for (int r = 0; r < 4; r++)
                    yRef[g * 4 + r] = MatMul.VecDotQ5_0Q8_1ScalarR4(r4 + (long)g * 4 * rowBytes, r, x, blocks);
            });
            double sseR4 = Median(() =>
            {
                for (int g = 0; g < groups; g++)
                    MatMul.VecDotQ5_0Q8_1Sse_4RowsR4(r4 + (long)g * 4 * rowBytes, x, blocks, y + g * 4);
            });
            Check(y, yRef, groups * 4, "R4");
            double dispRm = Median(() => MatMul.ComputeRowsQ5_0(w, x, y, m, blocks));
            double dispR4 = Median(() => MatMul.ComputeRowsQ5_0Interleaved(r4, x, y, groups, rw.TailRows, blocks));

            double nsBlk(double ms) => ms * 1e6 / ((double)m * blocks);
            _output.WriteLine($"[{label}] m={m} k={k} weights={wBytes / 1e6:F1} MB (single-threaded)");
            _output.WriteLine($"  row-major scalar : {scalarRm:F3} ms  {nsBlk(scalarRm):F2} ns/block");
            _output.WriteLine($"  row-major 128-bit: {sseRm:F3} ms  {nsBlk(sseRm):F2} ns/block  ({scalarRm / sseRm:F2}x, {wBytes / (sseRm * 1e6):F2} GB/s)");
            _output.WriteLine($"  R4 scalar        : {scalarR4:F3} ms  {nsBlk(scalarR4):F2} ns/block");
            _output.WriteLine($"  R4 128-bit       : {sseR4:F3} ms  {nsBlk(sseR4):F2} ns/block  ({scalarR4 / sseR4:F2}x, {wBytes / (sseR4 * 1e6):F2} GB/s)");
            _output.WriteLine($"  dispatch row-major {dispRm:F3} ms, R4 {dispR4:F3} ms");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xf); NativeMemory.Free(x);
            NativeMemory.Free(y); NativeMemory.Free(yRef);
        }
    }

    private static void Check(float* a, float* b, int n, string what)
    {
        for (int i = 0; i < n; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]),
                $"{what}[{i}]: {a[i]} vs {b[i]}");
    }

    private static double Median(Action body)
    {
        for (int i = 0; i < 5; i++) body(); // warm-up: let tiered JIT reach tier 1 (incl. the lambdas)
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
