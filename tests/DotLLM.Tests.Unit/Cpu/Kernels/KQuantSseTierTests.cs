using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477: the 128-bit (SSSE3) K-quant x Q8_K tier (<c>MatMulKQuants.Sse.cs</c>) for Q4_K,
/// Q5_K and Q6_K — the path those formats take on Westmere-class hardware.
///
/// <para>Oracle: the scalar <b>dequantizer</b> (<c>Dequantize.DequantizeQ*_KScalar</c>, an
/// independent code path from every dot kernel) times the Q8_K activations, summed in double.
/// The tolerance is relative to <c>Σ|w·x|</c> (1e-5), far below the effect of any element-order,
/// nibble, 5th/6th-bit, scale, min or bias bug. An exact-integer variant (unit scales, small
/// activations) pins the arithmetic with zero tolerance.</para>
///
/// <para>The <c>*_Dispatch_*</c> test goes through the public drivers; under
/// <c>DOTNET_EnableAVX=0</c> (Westmere's ISA) it takes the 128-bit tier and additionally asserts
/// bit-exactness against the 128-bit kernel called directly in the same summation order.</para>
/// </summary>
public sealed unsafe class KQuantSseTierTests
{
    private const int Q8KBytes = 292;
    private readonly ITestOutputHelper _output;
    public KQuantSseTierTests(ITestOutputHelper output) => _output = output;

    public enum Fmt { Q4_K, Q5_K, Q6_K }

    private static bool SseTierDispatched => Ssse3.IsSupported && !Avx2.IsSupported;

    private static int BlockBytes(Fmt f) => f switch { Fmt.Q4_K => 144, Fmt.Q5_K => 176, _ => 210 };

    private static float Sse(Fmt f, byte* w, byte* x, int n) => f switch
    {
        Fmt.Q4_K => MatMul.VecDotQ4_K_Q8_KSse(w, x, n),
        Fmt.Q5_K => MatMul.VecDotQ5_K_Q8_KSse(w, x, n),
        _ => MatMul.VecDotQ6_K_Q8_KSse(w, x, n),
    };

    private static float Scalar(Fmt f, byte* w, byte* x, int n) => f switch
    {
        Fmt.Q4_K => MatMul.VecDotQ4_K_Q8_KScalar(w, x, n),
        Fmt.Q5_K => MatMul.VecDotQ5_K_Q8_KScalar(w, x, n),
        _ => MatMul.VecDotQ6_K_Q8_KScalar(w, x, n),
    };

    private static float Avx2Dot(Fmt f, byte* w, byte* x, int n) => f switch
    {
        Fmt.Q4_K => MatMul.VecDotQ4_K_Q8_KAvx2(w, x, n),
        Fmt.Q5_K => MatMul.VecDotQ5_K_Q8_KAvx2(w, x, n),
        _ => MatMul.VecDotQ6_K_Q8_KAvx2(w, x, n),
    };

    private static void Dequant(Fmt f, byte* w, int n, float[] dest)
    {
        switch (f)
        {
            case Fmt.Q4_K: Dequantize.DequantizeQ4_KScalar((nint)w, n * 256L, dest); break;
            case Fmt.Q5_K: Dequantize.DequantizeQ5_KScalar((nint)w, n * 256L, dest); break;
            default: Dequantize.DequantizeQ6_KScalar((nint)w, n * 256L, dest); break;
        }
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

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Sse_MatchesDequantOracle_AndScalar_AndAvx2(Fmt f, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4810 + (int)f * 100 + n);
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
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Sse_UnitScales_EqualsExactIntegerDot(Fmt f, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4820 + (int)f * 100 + n);
        byte* w = RandomKRow(rng, f, n, unitScales: true);
        byte* x = RandomQ8K(rng, n, qMax: f == Fmt.Q6_K ? 4 : 16, allowMinus128: false, unitScale: true);
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
            Assert.Equal((float)exact, Sse(f, w, x, n));
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    // ──────────────────── Dispatch ────────────────────

    public static TheoryData<Fmt, int, int> DispatchShapes()
    {
        var d = new TheoryData<Fmt, int, int>();
        foreach (Fmt f in Enum.GetValues<Fmt>())
        {
            d.Add(f, 7, 3);
            d.Add(f, 39, 8);   // >= ParallelMinRows: pooled R4 path, 9 groups + 3 tail rows
        }
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(DispatchShapes))]
    public void KQuant_Dispatch_MatchesReference(Fmt f, int m, int n)
    {
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"-> SSE tier dispatched: {SseTierDispatched}");
        var rng = new Random(4830 + (int)f * 1000 + m * 10 + n);
        int bb = BlockBytes(f), rowBytes = n * bb, k = n * 256;
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

            // Row-major: pre-AVX2 fallback is the single-row 128-bit dot per row.
            switch (f)
            {
                case Fmt.Q4_K: MatMul.ComputeRowsQ4_K(w, x, actual, m, n); break;
                case Fmt.Q5_K: MatMul.ComputeRowsQ5_K(w, x, actual, m, n); break;
                default: MatMul.ComputeRowsQ6_K(w, x, actual, m, n); break;
            }
            for (int r = 0; r < m; r++)
            {
                Near(expected[r], mag[r], actual[r], $"{f} row-major [{r}]");
                if (SseTierDispatched) BitEq(Sse(f, w + r * rowBytes, x, n), actual[r], $"{f} row-major [{r}]");
            }

            // R4: full groups sum one super-block at a time; tail rows are row-major.
            QuantizationType qt = f switch { Fmt.Q4_K => QuantizationType.Q4_K, Fmt.Q5_K => QuantizationType.Q5_K, _ => QuantizationType.Q6_K };
            using var rw = WeightRepacking.RepackR4((nint)w, qt, m, k);
            using var pool = new ComputeThreadPool(4);
            foreach (ComputeThreadPool? p in new[] { null, pool })
            {
                new Span<float>(actual, m).Clear();
                switch (f)
                {
                    case Fmt.Q4_K: MatMul.ComputeRowsQ4_KInterleaved((byte*)rw.Ptr, x, actual, rw.FullGroupCount, rw.TailRows, n, p); break;
                    case Fmt.Q5_K: MatMul.ComputeRowsQ5_KInterleaved((byte*)rw.Ptr, x, actual, rw.FullGroupCount, rw.TailRows, n, p); break;
                    default: MatMul.ComputeRowsQ6_KInterleaved((byte*)rw.Ptr, x, actual, rw.FullGroupCount, rw.TailRows, n, p); break;
                }
                string what = p is null ? "R4" : "R4 pooled";
                for (int r = 0; r < m; r++)
                {
                    Near(expected[r], mag[r], actual[r], $"{f} {what} [{r}]");
                    if (!SseTierDispatched) continue;
                    float want;
                    if (r < rw.FullGroupCount * 4)
                    {
                        want = 0;
                        byte* row = w + r * rowBytes;
                        for (int sb = 0; sb < n; sb++) want += Sse(f, row + sb * bb, x + sb * Q8KBytes, 1);
                    }
                    else want = Sse(f, w + r * rowBytes, x, n);
                    BitEq(want, actual[r], $"{f} {what} [{r}]");
                }
            }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.Free(x); NativeMemory.Free(actual); }
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

    private static void BitEq(float expected, float actual, string what) =>
        Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
            $"{what}: expected {expected:R}, got {actual:R} (bit-exact)");

    /// <summary>Random K-quant row. Quant bytes are fully random (all codes, all 6-bit scales/mins).</summary>
    private static byte* RandomKRow(Random rng, Fmt f, int n, bool unitScales)
    {
        int bb = BlockBytes(f);
        byte* p = (byte*)NativeMemory.Alloc((nuint)(n * bb));
        for (int i = 0; i < n * bb; i++) p[i] = (byte)rng.Next(256);
        for (int sb = 0; sb < n; sb++)
        {
            byte* b = p + sb * bb;
            if (f == Fmt.Q6_K)
                *(Half*)(b + 208) = unitScales ? (Half)1f : (Half)((rng.NextSingle() * 2f - 1f) * 0.01f);
            else
            {
                *(Half*)b = unitScales ? (Half)1f : (Half)(rng.NextSingle() * 0.01f);
                *(Half*)(b + 2) = unitScales ? (Half)1f : (Half)(rng.NextSingle() * 0.01f);
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
