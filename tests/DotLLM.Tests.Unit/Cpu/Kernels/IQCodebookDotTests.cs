using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #605: packed × Q8_K dots for IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_S.
/// Oracle = the format's dequantiser (<c>Dequantize.ToFloat32</c>) × Q8_K activations summed in
/// double. Weight blocks are fully random bytes (every grid index, sign index, scale and — for IQ1_S
/// — every delta sign); activations span the production range ±127. Every tier (scalar, SSSE3,
/// AVX2, 4-column AVX2) and the public GEMV/GEMM drivers are checked against that oracle.
/// </summary>
public sealed unsafe class IQCodebookDotTests
{
    private const int Q8KBytes = 292;

    public enum Fmt { IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S }

    private static QuantizationType Qt(Fmt f) => f switch
    {
        Fmt.IQ2_XXS => QuantizationType.IQ2_XXS,
        Fmt.IQ2_XS => QuantizationType.IQ2_XS,
        Fmt.IQ2_S => QuantizationType.IQ2_S,
        Fmt.IQ3_XXS => QuantizationType.IQ3_XXS,
        Fmt.IQ3_S => QuantizationType.IQ3_S,
        _ => QuantizationType.IQ1_S,
    };

    private static int BlockBytes(Fmt f) => (int)QuantFormat.TryGetInfo(Qt(f))!.Value.BlockBytes;

    private static float Scalar(Fmt f, byte* w, byte* x, int n) => f switch
    {
        Fmt.IQ2_XXS => MatMul.VecDotIq_Q8_KScalar<MatMul.Iq2XxsFmt>(w, x, n),
        Fmt.IQ2_XS => MatMul.VecDotIq_Q8_KScalar<MatMul.Iq2XsFmt>(w, x, n),
        Fmt.IQ2_S => MatMul.VecDotIq_Q8_KScalar<MatMul.Iq2SFmt>(w, x, n),
        Fmt.IQ3_XXS => MatMul.VecDotIq_Q8_KScalar<MatMul.Iq3XxsFmt>(w, x, n),
        Fmt.IQ3_S => MatMul.VecDotIq_Q8_KScalar<MatMul.Iq3SFmt>(w, x, n),
        _ => MatMul.VecDotIq_Q8_KScalar<MatMul.Iq1SFmt>(w, x, n),
    };

    private static float Sse(Fmt f, byte* w, byte* x, int n) => f switch
    {
        Fmt.IQ2_XXS => MatMul.VecDotIq_Q8_KSse<MatMul.Iq2XxsFmt>(w, x, n),
        Fmt.IQ2_XS => MatMul.VecDotIq_Q8_KSse<MatMul.Iq2XsFmt>(w, x, n),
        Fmt.IQ2_S => MatMul.VecDotIq_Q8_KSse<MatMul.Iq2SFmt>(w, x, n),
        Fmt.IQ3_XXS => MatMul.VecDotIq_Q8_KSse<MatMul.Iq3XxsFmt>(w, x, n),
        Fmt.IQ3_S => MatMul.VecDotIq_Q8_KSse<MatMul.Iq3SFmt>(w, x, n),
        _ => MatMul.VecDotIq_Q8_KSse<MatMul.Iq1SFmt>(w, x, n),
    };

    private static float Avx2Dot(Fmt f, byte* w, byte* x, int n) => f switch
    {
        Fmt.IQ2_XXS => MatMul.VecDotIq_Q8_KAvx2<MatMul.Iq2XxsFmt>(w, x, n),
        Fmt.IQ2_XS => MatMul.VecDotIq_Q8_KAvx2<MatMul.Iq2XsFmt>(w, x, n),
        Fmt.IQ2_S => MatMul.VecDotIq_Q8_KAvx2<MatMul.Iq2SFmt>(w, x, n),
        Fmt.IQ3_XXS => MatMul.VecDotIq_Q8_KAvx2<MatMul.Iq3XxsFmt>(w, x, n),
        Fmt.IQ3_S => MatMul.VecDotIq_Q8_KAvx2<MatMul.Iq3SFmt>(w, x, n),
        _ => MatMul.VecDotIq_Q8_KAvx2<MatMul.Iq1SFmt>(w, x, n),
    };

    private static void Avx2x4(Fmt f, byte* w, byte* x, int rowBytes, int n, float* o)
    {
        switch (f)
        {
            case Fmt.IQ2_XXS: MatMul.VecDotIq_Q8_KAvx2x4<MatMul.Iq2XxsFmt>(w, x, rowBytes, n, o); break;
            case Fmt.IQ2_XS: MatMul.VecDotIq_Q8_KAvx2x4<MatMul.Iq2XsFmt>(w, x, rowBytes, n, o); break;
            case Fmt.IQ2_S: MatMul.VecDotIq_Q8_KAvx2x4<MatMul.Iq2SFmt>(w, x, rowBytes, n, o); break;
            case Fmt.IQ3_XXS: MatMul.VecDotIq_Q8_KAvx2x4<MatMul.Iq3XxsFmt>(w, x, rowBytes, n, o); break;
            case Fmt.IQ3_S: MatMul.VecDotIq_Q8_KAvx2x4<MatMul.Iq3SFmt>(w, x, rowBytes, n, o); break;
            default: MatMul.VecDotIq_Q8_KAvx2x4<MatMul.Iq1SFmt>(w, x, rowBytes, n, o); break;
        }
    }

    public static TheoryData<Fmt, int> Shapes()
    {
        var d = new TheoryData<Fmt, int>();
        foreach (Fmt f in Enum.GetValues<Fmt>())
            foreach (int n in new[] { 1, 3, 8, 11 })
                d.Add(f, n);
        return d;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Scalar_MatchesDequantOracle(Fmt f, int n)
    {
        byte* w = RandomRow(new Random(6100 + (int)f * 100 + n), f, n);
        byte* x = RandomQ8K(new Random(6101 + (int)f * 100 + n), n);
        try
        {
            (double e, double mag) = Oracle(f, w, x, n);
            Near(e, mag, Scalar(f, w, x, n), $"{f} n={n} scalar");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Sse_MatchesOracle(Fmt f, int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        byte* w = RandomRow(new Random(6102 + (int)f * 100 + n), f, n);
        byte* x = RandomQ8K(new Random(6103 + (int)f * 100 + n), n);
        try
        {
            (double e, double mag) = Oracle(f, w, x, n);
            Near(e, mag, Sse(f, w, x, n), $"{f} n={n} sse");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Avx2_MatchesOracle(Fmt f, int n)
    {
        Skip.IfNot(Avx2.IsSupported, "needs AVX2");
        byte* w = RandomRow(new Random(6104 + (int)f * 100 + n), f, n);
        byte* x = RandomQ8K(new Random(6105 + (int)f * 100 + n), n);
        try
        {
            (double e, double mag) = Oracle(f, w, x, n);
            Near(e, mag, Avx2Dot(f, w, x, n), $"{f} n={n} avx2");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>The 4-column kernel must match the oracle for EACH column (distinct columns catch stride bugs).</summary>
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void FourColumnKernel_MatchesOraclePerColumn(Fmt f, int n)
    {
        Skip.IfNot(Avx2.IsSupported, "needs AVX2");
        var rng = new Random(6106 + (int)f * 100 + n);
        byte* w = RandomRow(rng, f, n);
        byte* x = (byte*)NativeMemory.Alloc((nuint)(4 * n * Q8KBytes));
        try
        {
            for (int t = 0; t < 4; t++)
            {
                byte* col = RandomQ8K(rng, n);
                Buffer.MemoryCopy(col, x + t * n * Q8KBytes, n * Q8KBytes, n * Q8KBytes);
                NativeMemory.Free(col);
            }
            float* got = stackalloc float[4];
            Avx2x4(f, w, x, n * Q8KBytes, n, got);
            for (int t = 0; t < 4; t++)
            {
                (double e, double mag) = Oracle(f, w, x + t * n * Q8KBytes, n);
                Near(e, mag, got[t], $"{f} n={n} col {t}");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>Public drivers, serial + pooled, incl. 4-column blocks and ragged tails, vs dequantize-and-dot.</summary>
    [Theory]
    [InlineData(Fmt.IQ2_XXS, 1)] [InlineData(Fmt.IQ2_XXS, 7)]
    [InlineData(Fmt.IQ2_XS, 1)] [InlineData(Fmt.IQ2_XS, 7)]
    [InlineData(Fmt.IQ2_S, 1)] [InlineData(Fmt.IQ2_S, 7)]
    [InlineData(Fmt.IQ3_XXS, 1)] [InlineData(Fmt.IQ3_XXS, 7)]
    [InlineData(Fmt.IQ3_S, 1)] [InlineData(Fmt.IQ3_S, 7)]
    [InlineData(Fmt.IQ1_S, 1)] [InlineData(Fmt.IQ1_S, 7)]
    public void GemvGemm_MatchDequantizeAndDot(Fmt f, int cols)
    {
        const int m = 39, n = 4;
        var rng = new Random(6110 + (int)f * 100 + cols);
        int bb = BlockBytes(f), rowBytes = n * bb, k = n * 256;
        QuantizationType qt = Qt(f);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)(cols * k * sizeof(float)), 64);
        float* got = (float*)NativeMemory.Alloc((nuint)(cols * m * sizeof(float)));
        float* want = (float*)NativeMemory.Alloc((nuint)(cols * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = RandomRow(rng, f, n);
                Buffer.MemoryCopy(row, w + r * rowBytes, rowBytes, rowBytes);
                NativeMemory.Free(row);
            }
            for (int i = 0; i < cols * k; i++) b[i] = rng.NextSingle() * 2f - 1f;

            MatMul.GemmDequantRows(w, qt, b, want, m, k, cols, null);

            var mag = new double[cols * m];
            float[] deq = new float[k];
            for (int r = 0; r < m; r++)
            {
                Dequantize.ToFloat32((nint)(w + r * rowBytes), k, qt, deq);
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
                MatMul.GemmIQCodebook(qt, w, b, got, m, k, cols, p);
                for (int i = 0; i < cols * m; i++) NearMag(want[i], mag[i], got[i], $"{f} GEMM {what} [{i}]");

                if (cols == 1)
                {
                    new Span<float>(got, m).Clear();
                    MatMul.GemvIQCodebook(qt, w, b, got, m, k, p);
                    for (int r = 0; r < m; r++) NearMag(want[r], mag[r], got[r], $"{f} GEMV {what} [{r}]");
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(b);
            NativeMemory.Free(got); NativeMemory.Free(want);
        }
    }

    [Theory]
    [InlineData(Fmt.IQ2_XXS)] [InlineData(Fmt.IQ2_XS)] [InlineData(Fmt.IQ2_S)]
    [InlineData(Fmt.IQ3_XXS)] [InlineData(Fmt.IQ3_S)] [InlineData(Fmt.IQ1_S)]
    public void Dispatch_TablesRouteToThePackedDot(Fmt f)
    {
        Assert.True(MatMul.UsesQ8KDot(Qt(f)));
        Assert.True(MatMul.SupportsFusedDecode(Qt(f)));
    }

    // ──────────────────── helpers ────────────────────

    private static (double expected, double mag) Oracle(Fmt f, byte* w, byte* x, int n)
    {
        float[] deq = new float[n * 256];
        Dequantize.ToFloat32((nint)w, n * 256L, Qt(f), deq);
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

    private static void NearMag(float expected, double mag, float actual, string what)
    {
        double tol = 1e-3 * mag + 1e-30;
        Assert.True(Math.Abs(actual - expected) <= tol,
            $"{what}: expected {expected}, got {actual} (tol {tol}, mag {mag})");
    }

    /// <summary>Fully random bytes (every index / sign / scale / delta bit), finite small <c>d</c>.</summary>
    private static byte* RandomRow(Random rng, Fmt f, int n)
    {
        int bb = BlockBytes(f);
        byte* p = (byte*)NativeMemory.Alloc((nuint)(n * bb));
        for (int i = 0; i < n * bb; i++) p[i] = (byte)rng.Next(256);
        for (int sb = 0; sb < n; sb++)
            *(Half*)(p + sb * bb) = (Half)((rng.NextSingle() * 2f - 1f) * 0.01f);
        return p;
    }

    private static byte* RandomQ8K(Random rng, int n)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)(n * Q8KBytes));
        for (int sb = 0; sb < n; sb++)
        {
            byte* b = p + sb * Q8KBytes;
            *(float*)b = 0.001f + rng.NextSingle() * 0.02f;
            sbyte* qs = (sbyte*)(b + 4);
            short* bsums = (short*)(b + 260);
            for (int g = 0; g < 16; g++)
            {
                int s = 0;
                for (int i = 0; i < 16; i++)
                {
                    sbyte v = (sbyte)rng.Next(-127, 128);
                    qs[g * 16 + i] = v; s += v;
                }
                bsums[g] = (short)s;
            }
        }
        return p;
    }
}
