using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477 perf probe: the 128-bit (SSSE3) Q8_0 tier vs the scalar tier it replaces on
/// pre-AVX2 hardware, single-threaded, at real decode GEMV shapes, for both the row-major
/// (<c>ComputeRows</c>) and R4-interleaved (<c>ComputeRowsQ8_0Interleaved</c>, what repacked
/// transformer projections use) layouts. Kernels are called directly, so the numbers are
/// meaningful on any SSSE3 box; the "dispatch" rows go through the public drivers and show the
/// 128-bit tier only under <c>DOTNET_EnableAVX=0</c> (or on real Westmere).
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class Q8_0SseTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public Q8_0SseTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 15;

    [SkippableTheory]
    [InlineData(1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(2048, 2048, "Llama-3.2-1B attn_q")]
    [InlineData(8192, 2048, "Llama-3.2-1B ffn_gate")]
    [InlineData(2048, 8192, "Llama-3.2-1B ffn_down")]
    public void Q8_0_Gemv_Sse128VsScalar(int m, int k, string label)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"AVX512BW={Avx512BW.IsSupported}");
        var rng = new Random(477);
        int blocks = k / 32, rowBytes = blocks * 34;
        long wBytes = (long)m * rowBytes;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        byte* x = (byte*)NativeMemory.Alloc((nuint)rowBytes);
        float* y = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        float* yRef = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (long i = 0; i < wBytes; i++) w[i] = (byte)rng.Next(1, 255);
            // Activation quants in [-127, 127], as QuantizeF32ToQ8_0* guarantees (the kernel's precondition).
            for (int i = 0; i < rowBytes; i++) x[i] = unchecked((byte)(sbyte)rng.Next(-127, 128));
            for (long r = 0; r < (long)m * blocks; r++) *(Half*)(w + r * 34) = (Half)(0.01f + rng.NextSingle() * 0.02f);
            for (int b = 0; b < blocks; b++) *(Half*)(x + b * 34) = (Half)0.02f;
            using var rw = WeightRepacking.RepackR4((nint)w, QuantizationType.Q8_0, m, k);
            byte* r4 = (byte*)rw.Ptr;
            int groups = rw.FullGroupCount;

            double scalarRm = Median(() => { for (int r = 0; r < m; r++) yRef[r] = MatMul.VecDotQ8_0Scalar(w + (long)r * rowBytes, x, blocks); });
            double sseRm = Median(() => MatMul.ComputeRowsQ8_0Sse(w, x, y, m, blocks));
            Check(y, yRef, m, "row-major");
            double scalarR4 = Median(() =>
            {
                for (int g = 0; g < groups; g++)
                for (int r = 0; r < 4; r++)
                    yRef[g * 4 + r] = MatMul.VecDotQ8_0ScalarR4(r4 + (long)g * 4 * rowBytes, r, x, blocks);
            });
            double sseR4 = Median(() =>
            {
                for (int g = 0; g < groups; g++)
                    MatMul.VecDotQ8_0Sse_4RowsR4(r4 + (long)g * 4 * rowBytes, x, blocks, y + g * 4);
            });
            Check(y, yRef, groups * 4, "R4");
            double dispRm = Median(() => MatMul.ComputeRows(w, x, y, m, blocks));
            double dispR4 = Median(() => MatMul.ComputeRowsQ8_0Interleaved(r4, x, y, groups, rw.TailRows, blocks));

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
            NativeMemory.AlignedFree(w); NativeMemory.Free(x); NativeMemory.Free(y); NativeMemory.Free(yRef);
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
