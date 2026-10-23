using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477 perf probe: F16 x F32 GEMV on pre-F16C hardware. Compares a plain scalar
/// <c>(float)Half</c> loop, the existing <c>MatMul.GemvF16</c> (per-row
/// <c>TensorPrimitives.ConvertToSingle</c> into scratch + <c>TensorPrimitives.Dot</c>), and the
/// fused 128-bit <c>GemvF16Sse</c> (integer Half→Single, no scratch round-trip), single-threaded;
/// plus GEMM (n = 32) per-token re-conversion vs <c>GemmF16RowsSse</c>'s convert-once-per-row.
/// Meaningful under <c>DOTNET_EnableAVX=0</c> (or on real Westmere).
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class F16SseTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public F16SseTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 11;

    [SkippableTheory]
    [InlineData(1536, 576, "SmolLM-135M ffn_gate")]
    [InlineData(576, 1536, "SmolLM-135M ffn_down")]
    [InlineData(2048, 2048, "Llama-3.2-1B attn_q")]
    [InlineData(49152, 576, "SmolLM-135M lm_head")]
    public void F16_Gemv_Sse128VsExisting(int m, int k, string label)
    {
        Skip.IfNot(Sse3.IsSupported, "needs SSE3");
        _output.WriteLine($"ISA: SSE3={Sse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} F16C-class AVX2={Avx2.IsSupported}");
        var rng = new Random(477);
        long n = (long)m * k;
        ushort* w = (ushort*)NativeMemory.AlignedAlloc((nuint)(n * 2), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
        float* y = (float*)NativeMemory.Alloc((nuint)(m * 4));
        float* yRef = (float*)NativeMemory.Alloc((nuint)(m * 4));
        try
        {
            for (long i = 0; i < n; i++) w[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextSingle() * 2f - 1f) * 0.1f));
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            double scalar = Median(() =>
            {
                for (int r = 0; r < m; r++)
                {
                    float s = 0;
                    ushort* row = w + (long)r * k;
                    for (int i = 0; i < k; i++) s += (float)BitConverter.UInt16BitsToHalf(row[i]) * x[i];
                    yRef[r] = s;
                }
            });
            float* buf = (float*)NativeMemory.Alloc((nuint)(k * 4));
            double existing = Median(() => TpGemv(w, x, y, m, k, buf));
            Check(y, yRef, m, "TP convert+dot");
            double sse = Median(() => MatMul.GemvF16Sse(w, x, y, m, k));
            Check(y, yRef, m, "GemvF16Sse");
            double disp = Median(() => MatMul.GemvF16((nint)w, x, y, m, k));
            Check(y, yRef, m, "GemvF16 dispatch");

            double gbs(double ms) => n * 2 / (ms * 1e6);
            _output.WriteLine($"[{label}] m={m} k={k} weights={n * 2 / 1e6:F1} MB (single-threaded)");
            _output.WriteLine($"  scalar (float)Half loop : {scalar:F3} ms  {gbs(scalar):F2} GB/s");
            _output.WriteLine($"  TP convert+dot (pre-#477 GemvF16 body): {existing:F3} ms  {gbs(existing):F2} GB/s  ({scalar / existing:F2}x vs scalar)");
            _output.WriteLine($"  GemvF16Sse (fused)      : {sse:F3} ms  {gbs(sse):F2} GB/s  ({scalar / sse:F2}x vs scalar, {existing / sse:F2}x vs TP)");
            _output.WriteLine($"  GemvF16 dispatch        : {disp:F3} ms");

            // GEMM, n = 32 prefill tokens: per-token re-convert (pre-#477 body) vs convert-once-per-row.
            if (m * (long)k <= 4L << 20)
            {
                const int nTok = 32;
                float* b = (float*)NativeMemory.Alloc((nuint)((long)nTok * k * 4));
                float* c1 = (float*)NativeMemory.Alloc((nuint)((long)nTok * m * 4));
                float* c2 = (float*)NativeMemory.Alloc((nuint)((long)nTok * m * 4));
                try
                {
                    for (long i = 0; i < (long)nTok * k; i++) b[i] = rng.NextSingle() * 2f - 1f;
                    double perToken = Median(() => { for (int t = 0; t < nTok; t++) TpGemv(w, b + (long)t * k, c1 + (long)t * m, m, k, buf); });
                    double once = Median(() => MatMul.GemmF16RowsSse((Half*)w, m, b, c2, m, k, nTok, buf));
                    Check(c2, c1, nTok * m, "GemmF16RowsSse");
                    _output.WriteLine($"  GEMM nTok={nTok}: per-token convert {perToken:F3} ms, convert-once-per-row {once:F3} ms ({perToken / once:F2}x)");
                }
                finally { NativeMemory.Free(b); NativeMemory.Free(c1); NativeMemory.Free(c2); }
            }
            NativeMemory.Free(buf);
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.Free(y); NativeMemory.Free(yRef);
        }
    }

    /// <summary>The pre-#477 <c>GemvF16</c> body: per-row TP convert into scratch, then TP dot.</summary>
    private static void TpGemv(ushort* w, float* x, float* y, int m, int k, float* buf)
    {
        var dest = new Span<float>(buf, k);
        var xs = new ReadOnlySpan<float>(x, k);
        for (int r = 0; r < m; r++)
        {
            System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((Half*)w + (long)r * k, k), dest);
            y[r] = System.Numerics.Tensors.TensorPrimitives.Dot((ReadOnlySpan<float>)dest, xs);
        }
    }

    private static void Check(float* a, float* b, int n, string what)
    {
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= 1e-3f + 1e-4f * Math.Abs(b[i]), $"{what}[{i}]: {a[i]} vs {b[i]}");
    }

    private static double Median(Action body)
    {
        for (int i = 0; i < 3; i++) body();
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
