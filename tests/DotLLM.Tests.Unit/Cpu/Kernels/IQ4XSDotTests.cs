using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #605: the packed <b>IQ4_XS</b> × Q8_K dot. Before it every IQ4_XS matmul expanded the
/// weights to F32 per call (decode 1.1–3.7 tok/s).
///
/// <para>Oracle = <c>Dequantize.DequantizeIQ4_XS</c> (pinned independently of the dot) times the
/// Q8_K activations summed in double; the scalar dot is a transcription of llama.cpp's
/// <c>ggml_vec_dot_iq4_xs_q8_K_generic</c>. Weight blocks are fully random (every nibble, every
/// 6-bit scale). Activations span the full production range ±127 — the sign trick relies on the
/// quantiser's clamp (scale = maxAbs/127), so −128 is deliberately not generated.</para>
/// </summary>
public sealed unsafe class IQ4XSDotTests
{
    private const int Q8KBytes = 292;
    private const int BlockBytes = 136;

    public static TheoryData<int> Sizes() => new() { 1, 3, 8, 11 };

    [Theory]
    [MemberData(nameof(Sizes))]
    public void Scalar_MatchesDequantOracle(int n)
    {
        byte* w = RandomRow(new Random(6050 + n), n);
        byte* x = RandomQ8K(new Random(6051 + n), n);
        try
        {
            (double expected, double mag) = Oracle(w, x, n);
            Near(expected, mag, MatMul.VecDotIQ4_XS_Q8_KScalar(w, x, n), $"scalar n={n}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [MemberData(nameof(Sizes))]
    public void Avx2_MatchesOracle_AndScalar(int n)
    {
        Skip.IfNot(Avx2.IsSupported, "needs AVX2");
        byte* w = RandomRow(new Random(6052 + n), n);
        byte* x = RandomQ8K(new Random(6053 + n), n);
        try
        {
            (double expected, double mag) = Oracle(w, x, n);
            float avx = MatMul.VecDotIQ4_XS_Q8_KAvx2(w, x, n);
            Near(expected, mag, avx, $"avx2 n={n}");
            Near(MatMul.VecDotIQ4_XS_Q8_KScalar(w, x, n), mag, avx, $"avx2 vs scalar n={n}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [MemberData(nameof(Sizes))]
    public void Sse_MatchesOracle_AndScalar(int n)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        byte* w = RandomRow(new Random(6054 + n), n);
        byte* x = RandomQ8K(new Random(6055 + n), n);
        try
        {
            (double expected, double mag) = Oracle(w, x, n);
            float sse = MatMul.VecDotIQ4_XS_Q8_KSse(w, x, n);
            Near(expected, mag, sse, $"sse n={n}");
            Near(MatMul.VecDotIQ4_XS_Q8_KScalar(w, x, n), mag, sse, $"sse vs scalar n={n}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>Ragged row counts through <c>ComputeRows</c> (whichever tier this process dispatches).</summary>
    [Theory]
    [InlineData(7, 3)]
    [InlineData(39, 8)]
    public void ComputeRows_MatchesOracle(int m, int n)
    {
        var rng = new Random(6060 + m * 10 + n);
        int rowBytes = n * BlockBytes;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* x = RandomQ8K(rng, n);
        float* actual = (float*)NativeMemory.Alloc((nuint)(m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = RandomRow(rng, n);
                Buffer.MemoryCopy(row, w + r * rowBytes, rowBytes, rowBytes);
                NativeMemory.Free(row);
            }
            MatMul.ComputeRowsIQ4_XS(w, x, actual, m, n);
            for (int r = 0; r < m; r++)
            {
                (double expected, double mag) = Oracle(w + r * rowBytes, x, n);
                Near(expected, mag, actual[r], $"row {r}");
            }
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.Free(x); NativeMemory.Free(actual); }
    }

    /// <summary>
    /// Public GEMV/GEMM drivers (serial + pooled) vs dequantize-and-dot — catches a mis-wired driver
    /// rather than a mis-written kernel. Bound is relative to Σ|w·x| because the two paths differ by
    /// Q8_K activation quantisation.
    /// </summary>
    [Theory]
    [InlineData(7, 3, 3)]
    [InlineData(39, 8, 3)]
    [InlineData(39, 8, 4)]   // exactly one 4-column block
    [InlineData(39, 8, 7)]   // 4-column block + ragged 3-column tail
    [InlineData(23, 5, 16)]  // several blocks
    public void GemvGemm_MatchDequantizeAndDot(int m, int n, int cols)
    {
        var rng = new Random(6070 + m * 10 + n);
        int rowBytes = n * BlockBytes, k = n * 256;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)(cols * k * sizeof(float)), 64);
        float* got = (float*)NativeMemory.Alloc((nuint)(cols * m * sizeof(float)));
        float* want = (float*)NativeMemory.Alloc((nuint)(cols * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = RandomRow(rng, n);
                Buffer.MemoryCopy(row, w + r * rowBytes, rowBytes, rowBytes);
                NativeMemory.Free(row);
            }
            for (int i = 0; i < cols * k; i++) b[i] = rng.NextSingle() * 2f - 1f;

            MatMul.GemmDequantRows(w, QuantizationType.IQ4_XS, b, want, m, k, cols, null);

            var mag = new double[cols * m];
            float[] deq = new float[k];
            for (int r = 0; r < m; r++)
            {
                Dequantize.DequantizeIQ4_XS((nint)(w + r * rowBytes), k, deq);
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
                MatMul.GemvIQ4_XS(w, b, got, m, k, p);
                for (int r = 0; r < m; r++) NearMag(want[r], mag[r], got[r], $"GEMV {what} [{r}]");

                new Span<float>(got, cols * m).Clear();
                MatMul.GemmIQ4_XS(w, b, got, m, k, cols, p);
                for (int i = 0; i < cols * m; i++) NearMag(want[i], mag[i], got[i], $"GEMM {what} [{i}]");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(b);
            NativeMemory.Free(got); NativeMemory.Free(want);
        }
    }

    /// <summary>The 4-column kernel must equal four independent single-column dots (same tier, tight).</summary>
    [SkippableFact]
    public void FourColumnKernel_EqualsFourSingleColumnDots()
    {
        Skip.IfNot(Avx2.IsSupported, "needs AVX2");
        const int n = 5;
        var rng = new Random(6080);
        byte* w = RandomRow(rng, n);
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
            MatMul.VecDotIQ4_XS_Q8_KAvx2x4(w, x, n * Q8KBytes, n, got);
            for (int t = 0; t < 4; t++)
            {
                float want = MatMul.VecDotIQ4_XS_Q8_KAvx2(w, x + t * n * Q8KBytes, n);
                Assert.True(Math.Abs(got[t] - want) <= 1e-5 * Math.Max(1, Math.Abs(want)), $"col {t}: {got[t]} vs {want}");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [Fact]
    public void Dispatch_TablesRouteIQ4_XS_ToThePackedDot()
    {
        Assert.True(MatMul.UsesQ8KDot(QuantizationType.IQ4_XS));
        Assert.True(MatMul.SupportsFusedDecode(QuantizationType.IQ4_XS));
    }

    // ──────────────────── helpers ────────────────────

    private static (double expected, double mag) Oracle(byte* w, byte* x, int n)
    {
        float[] deq = new float[n * 256];
        Dequantize.DequantizeIQ4_XS((nint)w, n * 256L, deq);
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

    /// <summary>Random IQ4_XS row: every nibble, scale_l and scale_h byte random; finite small d.</summary>
    private static byte* RandomRow(Random rng, int n)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)(n * BlockBytes));
        for (int i = 0; i < n * BlockBytes; i++) p[i] = (byte)rng.Next(256);
        for (int sb = 0; sb < n; sb++)
            *(Half*)(p + sb * BlockBytes) = (Half)((rng.NextSingle() * 2f - 1f) * 0.01f);
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
