using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #497: the packed <b>Q2_K</b> and <b>Q3_K</b> × Q8_K dots — the low-bit half of the
/// K-quant family that #489 left open. Before these kernels both formats went through
/// <c>GemmDequantRows</c>, expanding the weight matrix to F32 on every call.
///
/// <para><b>Two independent oracles.</b> The scalar dots are transcriptions of llama.cpp's
/// <c>ggml_vec_dot_q{2,3}_K_q8_K_generic</c>; the comparison oracle here is the scalar
/// <b>dequantizer</b> (<c>Dequantize.DequantizeQ2_K</c> / <c>DequantizeQ3_KScalar</c>, which is
/// itself pinned against <c>dequantize_row_q{2,3}_K</c> by
/// <c>DequantizeKQuantTests.Q{2,3}_K_DenseRandomBlocks_MatchLlamaCppReference</c>) times the
/// Q8_K activations, summed in double. Agreement is therefore evidence, not a shared-mistake
/// tautology.</para>
///
/// <para>The exact-integer variant (unit scales, small activations) pins the arithmetic with
/// <b>zero</b> tolerance — every intermediate is an integer below 2^24.</para>
///
/// <para>The <c>*_Dispatch_*</c> test goes through <c>ComputeRows</c>; under
/// <c>DOTNET_EnableAVX=0</c> (Westmere's ISA) that takes the 128-bit tier, and the test then
/// additionally asserts bit-exactness against the 128-bit kernel called directly.</para>
/// </summary>
public sealed unsafe class LowBitKQuantDotTests
{
    private const int Q8KBytes = 292;
    private readonly ITestOutputHelper _output;
    public LowBitKQuantDotTests(ITestOutputHelper output) => _output = output;

    public enum Fmt { Q2_K, Q3_K }

    private static bool SseTierDispatched => Ssse3.IsSupported && !Avx2.IsSupported;

    private static int BlockBytes(Fmt f) => f == Fmt.Q2_K ? 84 : 110;

    private static float Sse(Fmt f, byte* w, byte* x, int n) => f == Fmt.Q2_K
        ? MatMul.VecDotQ2_K_Q8_KSse(w, x, n)
        : MatMul.VecDotQ3_K_Q8_KSse(w, x, n);

    private static float Scalar(Fmt f, byte* w, byte* x, int n) => f == Fmt.Q2_K
        ? MatMul.VecDotQ2_K_Q8_KScalar(w, x, n)
        : MatMul.VecDotQ3_K_Q8_KScalar(w, x, n);

    private static float Avx2Dot(Fmt f, byte* w, byte* x, int n) => f == Fmt.Q2_K
        ? MatMul.VecDotQ2_K_Q8_KAvx2(w, x, n)
        : MatMul.VecDotQ3_K_Q8_KAvx2(w, x, n);

    private static void Dequant(Fmt f, byte* w, int n, float[] dest)
    {
        if (f == Fmt.Q2_K) Dequantize.DequantizeQ2_K((nint)w, n * 256L, dest);
        else Dequantize.DequantizeQ3_KScalar((nint)w, n * 256L, dest);
    }

    // ──────────────────── Kernels ────────────────────

    public static TheoryData<Fmt, int> Shapes()
    {
        var d = new TheoryData<Fmt, int>();
        foreach (Fmt f in Enum.GetValues<Fmt>())
            foreach (int n in new[] { 1, 3, 8, 11 })  // k = 256 .. 2816
                d.Add(f, n);
        return d;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Scalar_MatchesDequantOracle(Fmt f, int n)
    {
        var rng = new Random(4970 + (int)f * 100 + n);
        byte* w = RandomKRow(rng, f, n, unitScales: false);
        byte* x = RandomQ8K(rng, n, qMax: 127, allowMinus128: true, unitScale: false);
        try
        {
            (double expected, double mag) = Oracle(f, w, x, n);
            Near(expected, mag, Scalar(f, w, x, n), $"{f} n={n} scalar");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Avx2_MatchesDequantOracle_AndScalar(Fmt f, int n)
    {
        Skip.IfNot(Avx2.IsSupported, "needs AVX2");
        var rng = new Random(4971 + (int)f * 100 + n);
        byte* w = RandomKRow(rng, f, n, unitScales: false);
        byte* x = RandomQ8K(rng, n, qMax: 127, allowMinus128: true, unitScale: false);
        try
        {
            (double expected, double mag) = Oracle(f, w, x, n);
            float tol = (float)(1e-5 * mag);
            float avx = Avx2Dot(f, w, x, n);
            Assert.True(Math.Abs(avx - expected) <= tol, $"{f} n={n}: AVX2 {avx} vs oracle {expected} (tol {tol})");
            float scalar = Scalar(f, w, x, n);
            Assert.True(Math.Abs(avx - scalar) <= tol, $"{f} n={n}: AVX2 {avx} vs scalar {scalar}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Sse_MatchesDequantOracle_AndScalar_AndAvx2(Fmt f, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4972 + (int)f * 100 + n);
        byte* w = RandomKRow(rng, f, n, unitScales: false);
        byte* x = RandomQ8K(rng, n, qMax: 127, allowMinus128: true, unitScale: false);
        try
        {
            (double expected, double mag) = Oracle(f, w, x, n);
            float tol = (float)(1e-5 * mag);
            float sse = Sse(f, w, x, n);
            Assert.True(Math.Abs(sse - expected) <= tol, $"{f} n={n}: SSE {sse} vs oracle {expected} (tol {tol})");
            float scalar = Scalar(f, w, x, n);
            Assert.True(Math.Abs(sse - scalar) <= tol, $"{f} n={n}: SSE {sse} vs scalar {scalar}");
            if (Avx2.IsSupported)
            {
                float avx = Avx2Dot(f, w, x, n);
                Assert.True(Math.Abs(sse - avx) <= tol, $"{f} n={n}: SSE {sse} vs AVX2 {avx}");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>Unit scales + small activations: every intermediate is an integer &lt; 2^24, so exact.</summary>
    [Theory]
    [MemberData(nameof(Shapes))]
    public void UnitScales_EqualsExactIntegerDot(Fmt f, int n)
    {
        var rng = new Random(4973 + (int)f * 100 + n);
        byte* w = RandomKRow(rng, f, n, unitScales: true);
        byte* x = RandomQ8K(rng, n, qMax: f == Fmt.Q2_K ? 16 : 4, allowMinus128: false, unitScale: true);
        try
        {
            float[] deq = new float[n * 256];
            Dequant(f, w, n, deq);
            long exact = 0;
            for (int sb = 0; sb < n; sb++)
            for (int e = 0; e < 256; e++)
            {
                float v = deq[sb * 256 + e];
                Assert.Equal(Math.Round(v), v);  // unit scales ⇒ integer weights
                exact += (long)v * ((sbyte*)(x + sb * Q8KBytes + 4))[e];
            }
            Assert.True(Math.Abs(exact) < (1 << 24));

            Assert.Equal((float)exact, Scalar(f, w, x, n));
            if (Ssse3.IsSupported) Assert.Equal((float)exact, Sse(f, w, x, n));
            if (Avx2.IsSupported) Assert.Equal((float)exact, Avx2Dot(f, w, x, n));
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    // ──────────────────── Dispatch ────────────────────

    public static TheoryData<Fmt, int, int> DispatchShapes()
    {
        var d = new TheoryData<Fmt, int, int>();
        foreach (Fmt f in Enum.GetValues<Fmt>())
        {
            d.Add(f, 7, 3);    // ragged m, below ParallelMinRows
            d.Add(f, 39, 8);   // ragged m, pooled
        }
        return d;
    }

    [Theory]
    [MemberData(nameof(DispatchShapes))]
    public void ComputeRows_MatchesReference(Fmt f, int m, int n)
    {
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"-> SSE tier dispatched: {SseTierDispatched}");
        var rng = new Random(4980 + (int)f * 1000 + m * 10 + n);
        int bb = BlockBytes(f), rowBytes = n * bb;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* x = RandomQ8K(rng, n, qMax: 127, allowMinus128: false, unitScale: false);
        float* actual = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = RandomKRow(rng, f, n, unitScales: false);
                Buffer.MemoryCopy(row, w + r * rowBytes, rowBytes, rowBytes);
                NativeMemory.Free(row);
            }
            var expected = new double[m];
            var mag = new double[m];
            for (int r = 0; r < m; r++) (expected[r], mag[r]) = Oracle(f, w + r * rowBytes, x, n);

            if (f == Fmt.Q2_K) MatMul.ComputeRowsQ2_K(w, x, actual, m, n);
            else MatMul.ComputeRowsQ3_K(w, x, actual, m, n);

            for (int r = 0; r < m; r++)
            {
                Near(expected[r], mag[r], actual[r], $"{f} row-major [{r}]");
                if (SseTierDispatched) BitEq(Sse(f, w + r * rowBytes, x, n), actual[r], $"{f} row-major [{r}]");
            }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.Free(x); NativeMemory.Free(actual); }
    }

    /// <summary>
    /// The public drivers on ragged shapes: <c>GemvQ*_K</c> / <c>GemmQ*_K</c>, serial and pooled,
    /// against the same dequantize-and-dot the formats used before #497. This is the test that
    /// would catch a mis-wired dispatch table rather than a mis-written kernel.
    /// </summary>
    [Theory]
    [MemberData(nameof(DispatchShapes))]
    public void GemvGemm_MatchDequantizeAndDot(Fmt f, int m, int n)
    {
        var rng = new Random(4990 + (int)f * 1000 + m * 10 + n);
        int bb = BlockBytes(f), rowBytes = n * bb, k = n * 256;
        const int cols = 3;
        QuantizationType qt = f == Fmt.Q2_K ? QuantizationType.Q2_K : QuantizationType.Q3_K;

        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)(cols * k * sizeof(float)), 64);
        float* got = (float*)NativeMemory.Alloc((nuint)(cols * m * sizeof(float)));
        float* want = (float*)NativeMemory.Alloc((nuint)(cols * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = RandomKRow(rng, f, n, unitScales: false);
                Buffer.MemoryCopy(row, w + r * rowBytes, rowBytes, rowBytes);
                NativeMemory.Free(row);
            }
            for (int i = 0; i < cols * k; i++) b[i] = (rng.NextSingle() * 2f - 1f);

            // Reference: the pre-#497 path — dequantize each weight row, dot in F32.
            MatMul.GemmDequantRows(w, qt, b, want, m, k, cols, null);

            // Per-output Σ|w·x|. The two paths differ by activation quantization, whose error is
            // proportional to that magnitude, NOT to the (heavily cancelling) result — random
            // weights against random activations produce near-zero dots for which a
            // result-relative bound is meaningless.
            var mag = new double[cols * m];
            float[] deq = new float[k];
            for (int r = 0; r < m; r++)
            {
                Dequant(f, w + r * rowBytes, n, deq);
                for (int t = 0; t < cols; t++)
                {
                    double a = 0;
                    for (int i = 0; i < k; i++) a += Math.Abs((double)deq[i] * b[t * k + i]);
                    mag[t * m + r] = a;
                }
            }

            using var pool = new ComputeThreadPool(4);
            foreach (ComputeThreadPool? p in new[] { null, pool })
            {
                string what = p is null ? "serial" : "pooled";

                new Span<float>(got, cols * m).Clear();
                if (f == Fmt.Q2_K) MatMul.GemvQ2_K(w, b, got, m, k, p);
                else MatMul.GemvQ3_K(w, b, got, m, k, p);
                for (int r = 0; r < m; r++) NearMag(want[r], mag[r], got[r], $"{f} GEMV {what} [{r}]");

                new Span<float>(got, cols * m).Clear();
                if (f == Fmt.Q2_K) MatMul.GemmQ2_K(w, b, got, m, k, cols, p);
                else MatMul.GemmQ3_K(w, b, got, m, k, cols, p);
                for (int i = 0; i < cols * m; i++) NearMag(want[i], mag[i], got[i], $"{f} GEMM {what} [{i}]");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(b);
            NativeMemory.Free(got); NativeMemory.Free(want);
        }
    }

    // ──────────────────── helpers ────────────────────

    private static (double expected, double mag) Oracle(Fmt f, byte* w, byte* x, int n)
    {
        float[] deq = new float[n * 256];
        Dequant(f, w, n, deq);
        double s = 0, a = 0;
        for (int sb = 0; sb < n; sb++)
        {
            float d8 = *(float*)(x + sb * Q8KBytes);
            sbyte* q = (sbyte*)(x + sb * Q8KBytes + 4);
            for (int e = 0; e < 256; e++)
            {
                double t = (double)deq[sb * 256 + e] * d8 * q[e];
                s += t; a += Math.Abs(t);
            }
        }
        return (s, a);
    }

    private static void Near(double expected, double mag, float actual, string what) =>
        Assert.True(Math.Abs(actual - expected) <= 1e-5 * mag + 1e-30, $"{what}: expected {expected}, got {actual}");

    /// <summary>
    /// Magnitude-relative comparison for the driver-vs-dequantize check. The two paths differ by
    /// activation quantization (Q8_K, ~7 bits per 256-element super-block) plus a different
    /// summation order, so this is a quality bound, not a bit-exactness one. The bound is
    /// 1e-3·Σ|w·x|: the analytic Q8_K error over k=2048 random elements is ~6e-5·Σ|w·x|, while a
    /// single mis-ordered element, wrong scale, wrong min or missing bias moves the result by an
    /// O(1) fraction of the same quantity — so this discriminates by more than an order of
    /// magnitude in both directions.
    /// </summary>
    private static void NearMag(float expected, double mag, float actual, string what)
    {
        double tol = 1e-3 * mag + 1e-30;
        Assert.True(Math.Abs(actual - expected) <= tol,
            $"{what}: expected {expected}, got {actual} (tol {tol}, mag {mag})");
    }

    private static void BitEq(float expected, float actual, string what) =>
        Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
            $"{what}: expected {expected:R}, got {actual:R} (bit-exact)");

    /// <summary>Random low-bit K-quant row. Quant bytes are fully random (all codes, all scales).</summary>
    private static byte* RandomKRow(Random rng, Fmt f, int n, bool unitScales)
    {
        int bb = BlockBytes(f);
        byte* p = (byte*)NativeMemory.Alloc((nuint)(n * bb));
        for (int i = 0; i < n * bb; i++) p[i] = (byte)rng.Next(256);
        for (int sb = 0; sb < n; sb++)
        {
            byte* b = p + sb * bb;
            if (f == Fmt.Q2_K)
            {
                *(Half*)(b + 80) = unitScales ? (Half)1f : (Half)(rng.NextSingle() * 0.01f);
                *(Half*)(b + 82) = unitScales ? (Half)1f : (Half)(rng.NextSingle() * 0.01f);
            }
            else
            {
                *(Half*)(b + 108) = unitScales ? (Half)1f : (Half)((rng.NextSingle() * 2f - 1f) * 0.01f);
            }
        }
        return p;
    }

    /// <summary>Q8_K block(s) with bsums consistent with qs.</summary>
    private static byte* RandomQ8K(Random rng, int n, int qMax, bool allowMinus128, bool unitScale)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)(n * Q8KBytes));
        for (int sb = 0; sb < n; sb++)
        {
            byte* b = p + sb * Q8KBytes;
            *(float*)b = unitScale ? 1f : 0.001f + rng.NextSingle() * 0.02f;
            sbyte* qs = (sbyte*)(b + 4);
            short* bsums = (short*)(b + 260);
            for (int g = 0; g < 16; g++)
            {
                int s = 0;
                for (int i = 0; i < 16; i++)
                {
                    int lo = allowMinus128 && qMax == 127 ? -128 : -qMax;
                    sbyte v = (sbyte)rng.Next(lo, qMax + 1);
                    qs[g * 16 + i] = v; s += v;
                }
                bsums[g] = (short)s;
            }
        }
        return p;
    }
}
