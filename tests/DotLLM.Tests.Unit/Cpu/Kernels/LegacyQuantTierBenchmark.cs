using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #489 perf probe: the packed Q4_0/Q4_1/Q5_1/IQ4_NL x Q8_1 dots against the
/// dequantize-then-GEMM path they replace, single-threaded, at real projection shapes.
///
/// <para>Each tier is called directly, so the per-block numbers are meaningful on any box; the
/// <c>dispatch</c> row goes through <c>ComputeRowsLegacyQuant</c> and shows the 128-bit tier only
/// under <c>DOTNET_EnableAVX=0</c> (or on real pre-AVX2 hardware). The <c>dequant</c> row is
/// <c>GemvDequantRows</c> — the exact code path these kernels displace — so the ratio is the
/// decode win, not a synthetic comparison.</para>
///
/// <para>GB/s is computed over the packed weight bytes actually streamed, which is the quantity a
/// decode GEMV is bound by.</para>
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class LegacyQuantTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public LegacyQuantTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 15;

    [Theory]
    [InlineData(QuantizationType.Q4_0, 1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(QuantizationType.Q4_1, 1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(QuantizationType.Q5_1, 1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(QuantizationType.IQ4_NL, 1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(QuantizationType.Q4_0, 2048, 2048, "Llama-3.2-1B attn_q")]
    [InlineData(QuantizationType.IQ4_NL, 2048, 2048, "Llama-3.2-1B attn_q (Nemotron's dominant type)")]
    public void Gemv_PackedVsDequantize(QuantizationType qt, int m, int k, string label)
    {
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"AVX512BW={Avx512BW.IsSupported}");
        var rng = new Random(489);
        int blocks = k / 32, blockBytes = BlockBytes(qt);
        int rowBytes = blocks * blockBytes, xBytes = blocks * 36;
        long wBytes = (long)m * rowBytes;

        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        float* xf = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        byte* x = (byte*)NativeMemory.Alloc((nuint)xBytes);
        float* y = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        float* yRef = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (long i = 0; i < wBytes; i++) w[i] = (byte)rng.Next(256);
            for (long b = 0; b < (long)m * blocks; b++)
            {
                *(Half*)(w + b * blockBytes) = (Half)(0.01f + rng.NextSingle() * 0.02f);
                if (qt is QuantizationType.Q4_1 or QuantizationType.Q5_1)
                    *(Half*)(w + b * blockBytes + 2) = (Half)(rng.NextSingle() * 0.01f);
            }
            for (int i = 0; i < k; i++) xf[i] = rng.NextSingle() * 2f - 1f;
            MatMul.QuantizeF32ToQ8_1(xf, x, k);

            double scalar = Median(() =>
            {
                for (int r = 0; r < m; r++)
                    yRef[r] = MatMul.VecDotLegacyQuantScalar(qt, w + (long)r * rowBytes, x, blocks);
            });

            double sse = double.NaN;
            if (Ssse3.IsSupported)
            {
                sse = Median(() => MatMul.ComputeRowsLegacyQuantSse(qt, w, x, y, m, blocks));
                CheckBitExact(y, yRef, m, $"{qt} 128-bit");
            }

            double avx2 = double.NaN;
            if (Avx2.IsSupported)
            {
                avx2 = Median(() =>
                {
                    for (int r = 0; r < m; r++)
                        y[r] = MatMul.VecDotLegacyQuantRow(qt, w + (long)r * rowBytes, x, blocks);
                });
            }

            double dispatch = Median(() => MatMul.ComputeRowsLegacyQuant(qt, w, x, y, m, blocks));
            double gemvPacked = Median(() => MatMul.GemvLegacyQuant(w, qt, xf, y, m, k));
            double gemvDequant = Median(() => MatMul.GemvDequantRows(w, qt, xf, y, m, k));

            double nsBlk(double ms) => ms * 1e6 / ((double)m * blocks);
            string gbps(double ms) => double.IsNaN(ms) ? "n/a" : $"{wBytes / (ms * 1e6):F2} GB/s";

            _output.WriteLine($"[{qt} {label}] m={m} k={k} packed weights={wBytes / 1e6:F1} MB (single-threaded)");
            _output.WriteLine($"  vec_dot scalar  : {scalar:F3} ms  {nsBlk(scalar):F2} ns/block  {gbps(scalar)}");
            if (Ssse3.IsSupported)
                _output.WriteLine($"  vec_dot 128-bit : {sse:F3} ms  {nsBlk(sse):F2} ns/block  {gbps(sse)}  ({scalar / sse:F2}x scalar)");
            if (Avx2.IsSupported)
                _output.WriteLine($"  vec_dot AVX2    : {avx2:F3} ms  {nsBlk(avx2):F2} ns/block  {gbps(avx2)}  ({scalar / avx2:F2}x scalar)");
            _output.WriteLine($"  dispatch        : {dispatch:F3} ms  {nsBlk(dispatch):F2} ns/block  {gbps(dispatch)}");
            _output.WriteLine($"  GEMV packed     : {gemvPacked:F3} ms  {gbps(gemvPacked)}");
            _output.WriteLine($"  GEMV dequantize : {gemvDequant:F3} ms  ({gemvDequant / gemvPacked:F2}x slower than packed)");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xf); NativeMemory.Free(x);
            NativeMemory.Free(y); NativeMemory.Free(yRef);
        }
    }

    /// <summary>
    /// The prefill side: the multi-column packed GEMM against <c>GemmDequantRows</c>. Records why
    /// <c>MatMul.PackedLegacyMaxColumns</c> is 1 — dequantizing a row once and running
    /// <c>TensorPrimitives.Dot</c> per column wins as soon as there is more than one column.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.Q4_0, 1536, 576, 1)]
    [InlineData(QuantizationType.Q4_0, 1536, 576, 2)]
    [InlineData(QuantizationType.Q4_0, 1536, 576, 4)]
    [InlineData(QuantizationType.Q4_0, 1536, 576, 32)]
    [InlineData(QuantizationType.IQ4_NL, 1536, 576, 1)]
    [InlineData(QuantizationType.IQ4_NL, 1536, 576, 2)]
    [InlineData(QuantizationType.IQ4_NL, 1536, 576, 4)]
    [InlineData(QuantizationType.IQ4_NL, 1536, 576, 32)]
    public void Gemm_PackedVsDequantize_ColumnCrossover(QuantizationType qt, int m, int k, int n)
    {
        var rng = new Random(4891);
        int blocks = k / 32, blockBytes = BlockBytes(qt), rowBytes = blocks * blockBytes;
        long wBytes = (long)m * rowBytes;

        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        float* xf = (float*)NativeMemory.Alloc((nuint)((long)n * k * sizeof(float)));
        float* c = (float*)NativeMemory.Alloc((nuint)((long)n * m * sizeof(float)));
        try
        {
            for (long i = 0; i < wBytes; i++) w[i] = (byte)rng.Next(256);
            for (long b = 0; b < (long)m * blocks; b++)
                *(Half*)(w + b * blockBytes) = (Half)(0.01f + rng.NextSingle() * 0.02f);
            for (long i = 0; i < (long)n * k; i++) xf[i] = rng.NextSingle() * 2f - 1f;

            double packed = Median(() => MatMul.GemmLegacyQuant(w, qt, xf, c, m, k, n));
            double dequant = Median(() => MatMul.GemmDequantRows(w, qt, xf, c, m, k, n, pool: null));
            _output.WriteLine($"[{qt}] m={m} k={k} n={n}: packed {packed:F3} ms, dequantize {dequant:F3} ms " +
                              $"-> packed is {dequant / packed:F2}x the dequantize path");
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.Free(xf); NativeMemory.Free(c); }
    }

    private static int BlockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_0 => 18,
        QuantizationType.Q4_1 => 20,
        QuantizationType.Q5_1 => 24,
        _ => 18
    };

    private static void CheckBitExact(float* a, float* b, int n, string what)
    {
        for (int i = 0; i < n; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]),
                $"{what}[{i}]: {a[i]} vs {b[i]}");
    }

    private static double Median(Action body)
    {
        for (int i = 0; i < 5; i++) body();   // warm-up: let tiered JIT reach tier 1
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
