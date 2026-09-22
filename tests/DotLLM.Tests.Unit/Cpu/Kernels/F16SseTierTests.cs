using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477: the 128-bit F16 x F32 tier (<c>MatMul.F16Sse.cs</c>) used on pre-AVX2 hardware
/// (no F16C): fused <c>DotF16Sse</c> for GEMV, convert-once-per-row for GEMM.
///
/// <para><b>Exact oracle.</b> With weights in eighths and small-integer activations every product and
/// every partial sum is exactly representable, so any summation order gives the same float —
/// the kernels and the public drivers must equal the exact dot, on every tier. That turns
/// a conversion, lane, tail or row-offset bug into a hard failure without a tolerance, and lets the
/// dispatch tests prove wiring under <c>DOTNET_EnableAVX=0</c>. Random-data tests add a tolerance
/// check against a double reference, including Half subnormals.</para>
/// </summary>
public sealed unsafe class F16SseTierTests
{
    private readonly ITestOutputHelper _output;
    public F16SseTierTests(ITestOutputHelper output) => _output = output;

    private static bool SseTierDispatched => Sse3.IsSupported && !Avx2.IsSupported;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(576)]
    [InlineData(1003)]
    public void DotF16Sse_ExactData_EqualsExactDot(int k)
    {
        Skip.IfNot(Sse3.IsSupported, "needs SSE3");
        var rng = new Random(4770 + k);
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)(k * 2));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * 4));
        try
        {
            long exact8 = FillEighths(rng, w, x, k);
            Assert.Equal(exact8 / 8f, MatMul.DotF16Sse(w, x, k));
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [InlineData(9)]
    [InlineData(576)]
    [InlineData(1536)]
    [InlineData(2051)]
    public void DotF16Sse_RandomData_MatchesDoubleReference(int k)
    {
        Skip.IfNot(Sse3.IsSupported, "needs SSE3");
        var rng = new Random(4790 + k);
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)(k * 2));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * 4));
        try
        {
            for (int i = 0; i < k; i++)
            {
                float v = rng.Next(10) == 0 ? (rng.NextSingle() - 0.5f) * 1e-5f : (rng.NextSingle() * 2f - 1f) * 4f;
                w[i] = BitConverter.HalfToUInt16Bits((Half)v);
                x[i] = rng.NextSingle() * 2f - 1f;
            }
            double reference = 0, mag = 0;
            for (int i = 0; i < k; i++)
            {
                double p = (double)(float)BitConverter.UInt16BitsToHalf(w[i]) * x[i];
                reference += p; mag += Math.Abs(p);
            }
            float actual = MatMul.DotF16Sse(w, x, k);
            Assert.True(Math.Abs(actual - reference) <= 1e-5 * mag + 1e-30,
                $"k={k}: {actual} vs {reference} (mag {mag})");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>
    /// Public drivers — <c>GemvF16</c> single-threaded and pooled, <c>GemmF16</c> single-threaded
    /// and pooled (multi-tile) — on exactly-summable data (eighths x integers), so every tier must equal the exact dot.
    /// Under <c>DOTNET_EnableAVX=0</c> these run the 128-bit tier.
    /// </summary>
    [SkippableTheory]
    [InlineData(7, 17)]
    [InlineData(37, 576)]
    [InlineData(300, 1000)]   // >= ParallelMinRows; k=1000 -> tileM 128 -> 3 tiles (pooled tiled GEMM)
    public void F16_Dispatch_ExactData_EqualsExactDot(int m, int k)
    {
        _output.WriteLine($"ISA: SSE3={Sse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"-> F16 SSE tier dispatched: {SseTierDispatched}");
        var rng = new Random(4800 + m * 131 + k);
        const int n = 3;
        ushort* w = (ushort*)NativeMemory.AlignedAlloc((nuint)((long)m * k * 2), 64);
        float* x = (float*)NativeMemory.Alloc((nuint)(n * k * 4));
        float* expected = (float*)NativeMemory.Alloc((nuint)(n * m * 4));
        float* actual = (float*)NativeMemory.Alloc((nuint)(n * m * 4));
        try
        {
            for (long i = 0; i < (long)m * k; i++) w[i] = BitConverter.HalfToUInt16Bits((Half)(rng.Next(-64, 65) / 8f));
            for (int i = 0; i < n * k; i++) x[i] = rng.Next(-8, 9);
            for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                long e8 = 0;
                for (int i = 0; i < k; i++)
                    e8 += (long)((float)BitConverter.UInt16BitsToHalf(w[(long)r * k + i]) * 8f) * (long)x[t * k + i];
                expected[t * m + r] = e8 / 8f;
            }

            MatMul.GemvF16((nint)w, x, actual, m, k);
            AssertExact(expected, actual, m, "GemvF16");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmF16((nint)w, x, actual, m, k, n);
            AssertExact(expected, actual, n * m, "GemmF16");

            using var pool = new ComputeThreadPool(4);
            new Span<float>(actual, m).Clear();
            MatMul.GemvF16((nint)w, x, actual, m, k, pool);
            AssertExact(expected, actual, m, "GemvF16 pooled");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmF16((nint)w, x, actual, m, k, n, pool);
            AssertExact(expected, actual, n * m, "GemmF16 pooled");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(x);
            NativeMemory.Free(expected); NativeMemory.Free(actual);
        }
    }

    private static void AssertExact(float* expected, float* actual, int count, string what)
    {
        for (int i = 0; i < count; i++)
            Assert.True(expected[i] == actual[i], $"{what} [{i}]: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>
    /// Weights in eighths over [-8, 8] (exact in Half, several exponents) and integer activations in
    /// [-8, 8]: every partial sum is a multiple of 1/8 below 2^24/8, so exact in float. Returns 8·Σ.
    /// </summary>
    private static long FillEighths(Random rng, ushort* w, float* x, int k)
    {
        long exact8 = 0;
        for (int i = 0; i < k; i++)
        {
            int w8 = rng.Next(-64, 65), xv = rng.Next(-8, 9);
            w[i] = BitConverter.HalfToUInt16Bits((Half)(w8 / 8f));
            x[i] = xv;
            exact8 += w8 * xv;
        }
        return exact8;
    }
}
