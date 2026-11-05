using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// PQ2_0 decode GEMV on the packed bytes IN PLACE with planar int8 activations (#1616; default ON with AVX2 since #1620,
/// off switch <c>DOTLLM_PQ2_PLANAR=0</c>; see docs/TERNARY_ADAPTER.md for the measurements). No row unpack/interleave: activations are re-laid out
/// per 128-group into 4 planes (plane i = x[4b+i], b=0..31) quantized to int8 with a per-plane scale, so
/// <c>t_i = (byte &gt;&gt; 2i) &amp; 3</c> lines up with plane i directly and
/// <c>sum((t-1)*q) = dpbusd(t, q) - sum(q)</c>; <c>sum_i s_i*sum(q_i)</c> is precomputed per group from the activations.
/// Quantization grid differs from the production Q8_0 path (32 strided elements instead of 32 consecutive), so results are
/// not bit-identical to <see cref="GemvPQ2_0"/>; accuracy vs an F64 oracle is equivalent (see tests / doc).
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// True when the planar PQ2_0 kernels are enabled: on by default with AVX2 (#1620), off with <c>DOTLLM_PQ2_PLANAR=0</c>, and off when
    /// <c>DOTLLM_PQ2_W2A8=0</c> asks for the unquantized-activation float tier (planar also quantizes activations, so it must not shadow that switch).
    /// </summary>
    private static bool s_pq2Planar =
        ResolvePlanarDefault(Environment.GetEnvironmentVariable("DOTLLM_PQ2_PLANAR"), Environment.GetEnvironmentVariable("DOTLLM_PQ2_W2A8")) && Avx2.IsSupported;

    /// <summary>Planar default from the two env switches (pure, for tests): on unless <paramref name="planar"/> or <paramref name="w2a8"/> is 0/false/off.</summary>
    internal static bool ResolvePlanarDefault(string? planar, string? w2a8) =>
        planar is not ("0" or "false" or "off") && w2a8 is not ("0" or "false" or "off");
    private static long s_planarCalls;

    internal static bool PQ2_0Planar => s_pq2Planar;

    /// <summary>
    /// Runtime toggle for the planar PQ2_0 kernels (initial value: on unless <c>DOTLLM_PQ2_PLANAR=0</c> or <c>DOTLLM_PQ2_W2A8=0</c>). Exists so a single process can
    /// alternate production/planar reps (same-session paired A/B, #1620); setting it true is ignored without AVX2.
    /// </summary>
    public static bool PQ2_0PlanarEnabled { get => s_pq2Planar; set => s_pq2Planar = value && Avx2.IsSupported; }

    /// <summary>Number of planar GEMV calls made so far (proof that the path ran).</summary>
    public static long PQ2_0PlanarCalls => Interlocked.Read(ref s_planarCalls);

    private struct GemvPQ2PlanarCtx
    {
        public byte* Weights; public sbyte* Xq; public float* Xs; public float* Xc; public float* Result; public int M, K;
    }

    /// <summary>Planar-activation PQ2_0 GEMV (n = 1). <paramref name="k"/> must be a multiple of 128.</summary>
    [SkipLocalsInit]
    internal static void GemvPQ2_0Planar(byte* weights, float* x, float* result, int m, int k, ComputeThreadPool? threadPool)
    {
        Interlocked.Increment(ref s_planarCalls);
        int groups = k / PQ2_0GroupSize;
        sbyte[] qBuf = ArrayPool<sbyte>.Shared.Rent(k);
        float[] sBuf = ArrayPool<float>.Shared.Rent(groups * 4);
        float[] cBuf = ArrayPool<float>.Shared.Rent(groups);
        try
        {
            fixed (sbyte* q = qBuf) fixed (float* s = sBuf) fixed (float* c = cBuf)
            {
                PreparePlanarActivations(x, k, q, s, c);
                if (threadPool is null || m < ParallelMinRows)
                {
                    PQ2PlanarRows(weights, q, s, c, result, 0, m, k);
                    return;
                }
                var ctx = new GemvPQ2PlanarCtx { Weights = weights, Xq = q, Xs = s, Xc = c, Result = result, M = m, K = k };
                threadPool.Dispatch((nint)(&ctx), &GemvPQ2PlanarWorker);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(qBuf);
            ArrayPool<float>.Shared.Return(sBuf);
            ArrayPool<float>.Shared.Return(cBuf);
        }
    }

    private static void GemvPQ2PlanarWorker(nint p, int idx, int cnt)
    {
        ref var c = ref Unsafe.AsRef<GemvPQ2PlanarCtx>((void*)p);
        PartitionRows(c.M, idx, cnt, out int start, out int count);
        if (count == 0) return;
        PQ2PlanarRows(c.Weights, c.Xq, c.Xs, c.Xc, c.Result, start, count, c.K);
    }

    /// <summary>Planar int8 quantization of one activation row; outputs <c>q[k]</c>, <c>s[groups*4]</c>, <c>c[groups]</c> (offset correction).</summary>
    [SkipLocalsInit]
    internal static void PreparePlanarActivations(float* x, int k, sbyte* q, float* s, float* c)
    {
        int groups = k / PQ2_0GroupSize;
        for (int g = 0; g < groups; g++)
        {
            float* xg = x + g * PQ2_0GroupSize;
            sbyte* qg = q + g * PQ2_0GroupSize;
            float corr = 0f;
            for (int i = 0; i < 4; i++)
            {
                float amax = 0f;
                for (int b = 0; b < 32; b++) amax = MathF.Max(amax, MathF.Abs(xg[4 * b + i]));
                float sc = amax / 127f;
                float inv = sc != 0f ? 1f / sc : 0f;
                int qsum = 0;
                for (int b = 0; b < 32; b++)
                {
                    int v = (int)MathF.Round(xg[4 * b + i] * inv);
                    qg[i * 32 + b] = (sbyte)v;
                    qsum += v;
                }
                s[g * 4 + i] = sc;
                corr += sc * qsum;
            }
            c[g] = corr;
        }
    }

    /// <summary>Rows [start, start+count) of the planar dot. See the class summary.</summary>
    internal static void PQ2PlanarRows(byte* w, sbyte* xq, float* xs, float* xc, float* result, int start, int count, int k)
    {
        if (Avx2.IsSupported) PQ2PlanarRowsAvx2(w, xq, xs, xc, result, start, count, k);
        else PQ2PlanarRowsScalar(w, xq, xs, xc, result, start, count, k);
    }

    /// <summary>
    /// Portable fallback (used when AVX2 is unavailable or disabled): exact integer per-plane sums, float combine in the
    /// same group order as the AVX2 path but with scalar accumulation, so results agree to float rounding, not bitwise.
    /// </summary>
    [SkipLocalsInit]
    internal static void PQ2PlanarRowsScalar(byte* w, sbyte* xq, float* xs, float* xc, float* result, int start, int count, int k)
    {
        int groups = k / PQ2_0GroupSize;
        for (int r = start; r < start + count; r++)
        {
            byte* row = w + (long)r * groups * PQ2_0GroupBytes;
            float acc = 0f, corr = 0f;
            for (int g = 0; g < groups; g++)
            {
                byte* gp = row + g * PQ2_0GroupBytes;
                float gs = HalfToSingle(gp);
                byte* codes = gp + 2;
                sbyte* xg = xq + g * PQ2_0GroupSize;
                float gacc = 0f;
                for (int i = 0; i < 4; i++)
                {
                    int isum = 0;
                    for (int b = 0; b < 32; b++)
                        isum += ((codes[b] >> (2 * i)) & 3) * xg[i * 32 + b];
                    gacc += isum * xs[g * 4 + i];
                }
                acc += gacc * gs;
                corr += gs * xc[g];
            }
            result[r] = acc - corr;
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void PQ2PlanarRowsAvx2(byte* w, sbyte* xq, float* xs, float* xc, float* result, int start, int count, int k)
    {
        int groups = k / PQ2_0GroupSize;
        bool vnni = AvxVnni.IsSupported;
        Vector256<byte> mask3 = Vector256.Create((byte)3);
        Vector256<short> ones16 = Vector256.Create((short)1);
        for (int r = start; r < start + count; r++)
        {
            byte* row = w + (long)r * groups * PQ2_0GroupBytes;
            Vector256<float> acc = Vector256<float>.Zero;
            float corr = 0f;
            for (int g = 0; g < groups; g++)
            {
                byte* gp = row + g * PQ2_0GroupBytes;
                float gs = HalfToSingle(gp);
                Vector256<byte> v = Unsafe.ReadUnaligned<Vector256<byte>>(gp + 2);
                Vector256<byte> t0 = Avx2.And(v, mask3);
                Vector256<byte> t1 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 2).AsByte(), mask3);
                Vector256<byte> t2 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte(), mask3);
                Vector256<byte> t3 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 6).AsByte(), mask3);
                sbyte* xg = xq + g * PQ2_0GroupSize;
                Vector256<int> i0, i1, i2, i3;
                if (vnni)
                {
                    i0 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t0, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg));
                    i1 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t1, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 32));
                    i2 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t2, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 64));
                    i3 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t3, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 96));
                }
                else
                {
                    i0 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t0, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg)), ones16);
                    i1 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t1, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 32)), ones16);
                    i2 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t2, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 64)), ones16);
                    i3 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t3, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 96)), ones16);
                }
                Vector256<float> gacc = Avx.Multiply(Avx.ConvertToVector256Single(i0), Vector256.Create(xs[g * 4]));
                gacc = Fma.IsSupported ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(xs[g * 4 + 1]), gacc)
                                       : gacc + Avx.ConvertToVector256Single(i1) * Vector256.Create(xs[g * 4 + 1]);
                gacc = Fma.IsSupported ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), Vector256.Create(xs[g * 4 + 2]), gacc)
                                       : gacc + Avx.ConvertToVector256Single(i2) * Vector256.Create(xs[g * 4 + 2]);
                gacc = Fma.IsSupported ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), Vector256.Create(xs[g * 4 + 3]), gacc)
                                       : gacc + Avx.ConvertToVector256Single(i3) * Vector256.Create(xs[g * 4 + 3]);
                acc = Fma.IsSupported ? Fma.MultiplyAdd(gacc, Vector256.Create(gs), acc) : acc + gacc * Vector256.Create(gs);
                corr += gs * xc[g];
            }
            result[r] = HorizontalSumAvx2Float(acc) - corr;
        }
    }
}
