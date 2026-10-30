using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477 perf probe: the 128-bit (SSSE3) ternary W2A8 tier vs the float scalar tier it
/// replaces on pre-AVX2 hardware, single-threaded, at real decode shapes. The 128-bit tier is
/// forced via <c>MatMul.*Sse128ForBench</c> so it can be timed on any SSSE3 box; on a real
/// Westmere the public entries (<see cref="MatMul.GemvPQ2_0"/>, <see cref="MatMul.GemvI2_S(byte*, float*, float*, int, int, DotLLM.Cpu.Threading.ComputeThreadPool?)"/>)
/// dispatch to the same kernels, and the "dispatch" row shows that.
///
/// <para>On an AVX2 box the "scalar" baseline's <c>TensorPrimitives.Dot</c> runs 256/512-bit, so
/// it flatters the baseline; run under <c>DOTNET_EnableAVX=0</c> for the Westmere-shaped ratio.</para>
///
/// <para>Also measures the software <c>Half</c>→<c>float</c> conversion cost (issue #477 item 2):
/// on .NET 10 <c>(float)Half</c> is a managed bit-manipulation routine on every x64 CPU (F16C is
/// not used by the JIT for it), so the number measured here is the code path Westmere runs.</para>
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class W2A8SseTierBenchmark
{
    private readonly ITestOutputHelper _output;
    public W2A8SseTierBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 7;

    private void Header() =>
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"AVX512BW={Avx512BW.IsSupported}");

    [SkippableTheory]
    [InlineData(5120, 5120, "attn (hidden x hidden)")]
    [InlineData(17408, 5120, "ffn_gate (ffn x hidden)")]
    [InlineData(5120, 17408, "ffn_down (hidden x ffn)")]
    public void PQ2_0_Gemv_Sse128VsScalar(int m, int k, string label)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        Header();
        var rng = new Random(477);
        int groups = k / 128, rowBytes = groups * 34;
        byte* w = (byte*)NativeMemory.Alloc((nuint)((long)m * rowBytes));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        float* ySse = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        float* yRef = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        byte* xQ8 = (byte*)NativeMemory.Alloc((nuint)(k / 32 * 34));
        float* xs = (float*)NativeMemory.Alloc((nuint)(k / 32 * sizeof(float)));
        sbyte* row = (sbyte*)NativeMemory.Alloc((nuint)k);
        float* gs = (float*)NativeMemory.Alloc((nuint)(groups * sizeof(float)));
        try
        {
            for (long r = 0; r < m; r++)
            for (int g = 0; g < groups; g++)
            {
                byte* gb = w + r * rowBytes + g * 34;
                *(Half*)gb = (Half)(0.01f + rng.NextSingle() * 0.05f);
                for (int i = 0; i < 32; i++)
                    gb[2 + i] = (byte)(rng.Next(3) | (rng.Next(3) << 2) | (rng.Next(3) << 4) | (rng.Next(3) << 6));
            }
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            MatMul.GemvPQ2_0Sse128ForBench(w, x, ySse, m, k, xQ8, xs, row, gs);
            MatMul.GemvPQ2_0Scalar(w, x, yRef, m, k);
            AssertMeanRel(ySse, yRef, m, label);

            double sse = Median(() => MatMul.GemvPQ2_0Sse128ForBench(w, x, ySse, m, k, xQ8, xs, row, gs));
            double scalar = Median(() => MatMul.GemvPQ2_0Scalar(w, x, yRef, m, k));
            double dispatch = Median(() => MatMul.GemvPQ2_0(w, x, ySse, m, k, null));
            Report(label, m, k, (long)m * rowBytes, scalar, sse, dispatch);
        }
        finally
        {
            NativeMemory.Free(w); NativeMemory.Free(x); NativeMemory.Free(ySse); NativeMemory.Free(yRef);
            NativeMemory.Free(xQ8); NativeMemory.Free(xs); NativeMemory.Free(row); NativeMemory.Free(gs);
        }
    }

    [SkippableTheory]
    [InlineData(2560, 2560, "attn (hidden x hidden)")]
    [InlineData(6912, 2560, "ffn_gate (ffn x hidden)")]
    [InlineData(2560, 6912, "ffn_down (hidden x ffn)")]
    public void I2S_Gemv_Sse128VsScalar(int m, int k, string label)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        Header();
        var rng = new Random(478);
        long packed = (long)m * k / 4;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(packed + 4));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        float* ySse = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        float* yRef = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        byte* xQ8 = (byte*)NativeMemory.Alloc((nuint)(k / 32 * 34));
        float* xs = (float*)NativeMemory.Alloc((nuint)(k / 32 * sizeof(float)));
        sbyte* row = (sbyte*)NativeMemory.Alloc((nuint)k);
        try
        {
            for (long i = 0; i < packed; i++)
                w[i] = (byte)(rng.Next(3) | (rng.Next(3) << 2) | (rng.Next(3) << 4) | (rng.Next(3) << 6));
            *(float*)(w + packed) = 0.03f;
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            MatMul.GemvI2_SSse128ForBench(w, x, ySse, m, k, xQ8, xs, row);
            MatMul.GemvI2_SScalar(w, x, yRef, m, k);
            AssertMeanRel(ySse, yRef, m, label);

            double sse = Median(() => MatMul.GemvI2_SSse128ForBench(w, x, ySse, m, k, xQ8, xs, row));
            double scalar = Median(() => MatMul.GemvI2_SScalar(w, x, yRef, m, k));
            double dispatch = Median(() => MatMul.GemvI2_S(w, x, ySse, m, k, null));
            Report(label, m, k, packed, scalar, sse, dispatch);
        }
        finally
        {
            NativeMemory.Free(w); NativeMemory.Free(x); NativeMemory.Free(ySse); NativeMemory.Free(yRef);
            NativeMemory.Free(xQ8); NativeMemory.Free(xs); NativeMemory.Free(row);
        }
    }

    /// <summary>
    /// Dot-only A/B (compute-bound: one pre-unpacked row reused, L1-hot): the 128-bit dot vs the
    /// AVX2/VNNI dot, alternating order across trials to cancel clock-ramp drift.
    /// </summary>
    [SkippableTheory]
    [InlineData(5120)]
    [InlineData(17408)]
    public void PQ2_0_DotOnly_Sse128VsAvx2(int k)
    {
        Skip.IfNot(Avx2.IsSupported, "A/B needs AVX2 as well");
        Header();
        var rng = new Random(480);
        int groups = k / 128, blocks = k / 32;
        const int reps = 20000;
        sbyte* w = (sbyte*)NativeMemory.Alloc((nuint)k);
        float* gs = (float*)NativeMemory.Alloc((nuint)(groups * sizeof(float)));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        byte* xQ8 = (byte*)NativeMemory.Alloc((nuint)(blocks * 34));
        try
        {
            for (int i = 0; i < k; i++) { w[i] = (sbyte)(rng.Next(3) - 1); x[i] = rng.NextSingle() * 2f - 1f; }
            for (int g = 0; g < groups; g++) gs[g] = 0.02f;
            MatMul.QuantizeF32ToQ8_0Scalar(x, xQ8, k);
            float* xs = stackalloc float[blocks];
            MatMul.ConvertQ8_0Scales(xQ8, xs, blocks);

            float sink = 0;
            var sse = new List<double>();
            var avx = new List<double>();
            for (int trial = 0; trial < 2 * Trials; trial++)
            {
                bool sseFirst = trial % 2 == 0;
                for (int pass = 0; pass < 2; pass++)
                {
                    bool doSse = (pass == 0) == sseFirst;
                    var sw = Stopwatch.StartNew();
                    if (doSse) for (int r = 0; r < reps; r++) sink += MatMul.VecDotPQ2_0Q8Sse(w, gs, xQ8, xs, blocks);
                    else for (int r = 0; r < reps; r++) sink += MatMul.VecDotPQ2_0Q8Avx2(w, gs, xQ8, xs, blocks);
                    (doSse ? sse : avx).Add(sw.Elapsed.TotalMilliseconds * 1e6 / ((double)reps * blocks));
                }
            }
            sse.Sort(); avx.Sort();
            _output.WriteLine($"k={k}: 128-bit dot {sse[sse.Count / 2]:F3} ns/block, AVX2/VNNI dot " +
                              $"{avx[avx.Count / 2]:F3} ns/block (sink {sink})");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(gs); NativeMemory.Free(x); NativeMemory.Free(xQ8); }
    }

    /// <summary>
    /// Cost of <c>(float)Half</c> vs a 64K-entry float lookup table, per conversion, in context of
    /// the scalar Q8_0 dot (2 conversions per 32-element block).
    /// </summary>
    [Fact]
    public void HalfToFloat_SoftwareVsLut_Cost()
    {
        Header();
        const int count = 1 << 20;
        var rng = new Random(479);
        ushort[] bits = new ushort[count];
        // Realistic scale values (small positive/negative normals), as Q8_0/K-quant d fields are.
        for (int i = 0; i < count; i++)
            bits[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextSingle() * 2f - 1f) * 0.05f));
        float[] lut = new float[65536];
        for (int i = 0; i < 65536; i++) lut[i] = (float)BitConverter.UInt16BitsToHalf((ushort)i);

        float sink = 0;
        double soft = Median(() =>
        {
            float s = 0;
            fixed (ushort* p = bits)
                for (int i = 0; i < count; i++) s += (float)*(Half*)(p + i);
            sink += s;
        });
        double table = Median(() =>
        {
            float s = 0;
            fixed (ushort* p = bits)
            fixed (float* t = lut)
                for (int i = 0; i < count; i++) s += t[p[i]];
            sink += s;
        });
        double baseline = Median(() =>
        {
            float s = 0;
            fixed (ushort* p = bits)
                for (int i = 0; i < count; i++) s += p[i];
            sink += s;
        });

        // Scalar Q8_0 dot for context: ns per 32-element block (includes its 2 Half conversions).
        const int blocks = 4096;
        byte* a = (byte*)NativeMemory.Alloc(blocks * 34);
        byte* b = (byte*)NativeMemory.Alloc(blocks * 34);
        try
        {
            for (int i = 0; i < blocks * 34; i++) { a[i] = (byte)rng.Next(256); b[i] = (byte)rng.Next(256); }
            for (int blk = 0; blk < blocks; blk++)
            {
                *(Half*)(a + blk * 34) = (Half)0.01f;
                *(Half*)(b + blk * 34) = (Half)0.02f;
            }
            double q8 = Median(() => { for (int rep = 0; rep < 64; rep++) sink += MatMul.VecDotQ8_0Scalar(a, b, blocks); });
            double nsPerBlock = q8 * 1e6 / (64.0 * blocks);

            _output.WriteLine($"(float)Half  : {soft * 1e6 / count:F3} ns/convert");
            _output.WriteLine($"LUT[65536]   : {table * 1e6 / count:F3} ns/convert");
            _output.WriteLine($"load+add only: {baseline * 1e6 / count:F3} ns/element");
            _output.WriteLine($"VecDotQ8_0Scalar: {nsPerBlock:F2} ns/block (2 Half converts ≈ " +
                              $"{2 * (soft - baseline) * 1e6 / count / nsPerBlock:P1} of block time)");
            _output.WriteLine($"(sink {sink})");
        }
        finally { NativeMemory.Free(a); NativeMemory.Free(b); }
    }

    private void Report(string label, int m, int k, long weightBytes, double scalar, double sse, double dispatch)
    {
        _output.WriteLine($"[{label}] m={m} k={k} weightBytes={weightBytes}");
        _output.WriteLine($"  scalar float tier : {scalar:F3} ms");
        _output.WriteLine($"  128-bit W2A8 tier : {sse:F3} ms  ({scalar / sse:F2}x vs scalar, " +
                          $"{weightBytes / (sse * 1e6):F2} GB/s weights)");
        _output.WriteLine($"  dispatch (public) : {dispatch:F3} ms  ({scalar / dispatch:F2}x vs scalar)");
    }

    private static void AssertMeanRel(float* a, float* b, int n, string label)
    {
        double d = 0, s = 0;
        for (int i = 0; i < n; i++) { d += Math.Abs(a[i] - b[i]); s += Math.Abs(b[i]); }
        Assert.True(d / s <= 0.05, $"[{label}] mean relative error {d / s:P2} > 5%");
    }

    private static double Median(Action body)
    {
        body(); // warm-up / JIT
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
