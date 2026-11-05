using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Token-blocked planar PQ2_0 GEMM for small n (#1620): the same in-place, planar-activation dot as
/// <see cref="GemvPQ2_0Planar"/> but each weight group is decoded (load + 3 shifts + 4 ands) ONCE per block of up to
/// <see cref="PlanarTokenBlock"/> tokens and dotted against each token's planar activations, so the decode cost (which is
/// what made the unblocked planar kernel lose at n &gt;= 16: it re-decoded the codes per token) amortizes over the block.
/// Reachable when the planar toggle is on (default since #1620) and <c>2 &lt;= n &lt;= PQ2_0PlanarMaxN</c> (<c>DOTLLM_PQ2_PLANAR_NMAX</c>, default 16).
/// Same numerics per token as <see cref="GemvPQ2_0Planar"/> (identical per-token op order), so n=1 and blocked agree bitwise.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>Tokens per register block (4 accumulators).</summary>
    internal const int PlanarTokenBlock = 4;

    /// <summary>Default <see cref="PQ2_0PlanarMaxN"/> (#1620, measured in docs/TERNARY_ADAPTER.md).</summary>
    internal const int PQ2_0PlanarDefaultMaxN = 16;

    /// <summary>
    /// Largest n routed to the blocked planar GEMM when the planar toggle is on (<c>DOTLLM_PQ2_PLANAR_NMAX</c>; 1 = decode GEMV only; default
    /// <see cref="PQ2_0PlanarDefaultMaxN"/>). 16 is the largest cut that never lost to production on the three PQ2_0 models measured: Bonsai-2 27B gains at
    /// every n, but on Bonsai-2 2B (smaller matrices, so the scalar O(n*k) planar activation prep weighs more) the blocked kernel drops to 0.96x at n = 32
    /// and 0.93x at n = 128. It also covers the default prompt-lookup verify (8 drafts = 9 rows), so those verify rows stay bit-identical per token to
    /// the planar decode GEMV (the #1367 invariant). Batches above the cut run the W2A8 tier (Q8_0 activation grid).
    /// </summary>
    internal static int PQ2_0PlanarMaxN { get; set; } = ResolvePlanarMaxN(Environment.GetEnvironmentVariable("DOTLLM_PQ2_PLANAR_NMAX"));

    /// <summary><c>DOTLLM_PQ2_PLANAR_NMAX</c> parse (pure, for tests): an integer, else <see cref="PQ2_0PlanarDefaultMaxN"/>.</summary>
    internal static int ResolvePlanarMaxN(string? env) => int.TryParse(env, out int v) ? v : PQ2_0PlanarDefaultMaxN;

    private struct GemmPQ2PlanarCtx
    {
        public byte* Weights; public sbyte* Xq; public float* Xs; public float* Xc; public float* C; public int M, K, N;
    }

    /// <summary>Planar-activation PQ2_0 GEMM, <c>c[t*m + r]</c>, <paramref name="b"/> is <c>[n, k]</c>. <paramref name="k"/> must be a multiple of 128.</summary>
    [SkipLocalsInit]
    internal static void GemmPQ2_0PlanarBlocked(byte* weights, float* b, float* c, int m, int k, int n, ComputeThreadPool? threadPool)
    {
        Interlocked.Increment(ref s_planarCalls);
        int groups = k / PQ2_0GroupSize;
        sbyte[] qBuf = ArrayPool<sbyte>.Shared.Rent(n * k);
        float[] sBuf = ArrayPool<float>.Shared.Rent(n * groups * 4);
        float[] cBuf = ArrayPool<float>.Shared.Rent(n * groups);
        try
        {
            fixed (sbyte* q = qBuf) fixed (float* s = sBuf) fixed (float* cc = cBuf)
            {
                for (int t = 0; t < n; t++)
                    PreparePlanarActivations(b + (long)t * k, k, q + (long)t * k, s + (long)t * groups * 4, cc + (long)t * groups);
                if (threadPool is null || m < ParallelMinRows)
                {
                    PQ2PlanarBlockRows(weights, q, s, cc, c, m, 0, m, k, n);
                    return;
                }
                var ctx = new GemmPQ2PlanarCtx { Weights = weights, Xq = q, Xs = s, Xc = cc, C = c, M = m, K = k, N = n };
                threadPool.Dispatch((nint)(&ctx), &GemmPQ2PlanarWorker);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(qBuf);
            ArrayPool<float>.Shared.Return(sBuf);
            ArrayPool<float>.Shared.Return(cBuf);
        }
    }

    private static void GemmPQ2PlanarWorker(nint p, int idx, int cnt)
    {
        ref var c = ref Unsafe.AsRef<GemmPQ2PlanarCtx>((void*)p);
        PartitionRows(c.M, idx, cnt, out int start, out int count);
        if (count == 0) return;
        PQ2PlanarBlockRows(c.Weights, c.Xq, c.Xs, c.Xc, c.C, c.M, start, count, c.K, c.N);
    }

    internal static void PQ2PlanarBlockRows(byte* w, sbyte* xq, float* xs, float* xc, float* c, int m, int start, int count, int k, int n)
    {
        if (CpuFeatures.Avx2) { PQ2PlanarBlockRowsAvx2(w, xq, xs, xc, c, m, start, count, k, n); return; }
        int groups = k / PQ2_0GroupSize;
        for (int t = 0; t < n; t++)
            PQ2PlanarRowsScalar(w, xq + (long)t * k, xs + (long)t * groups * 4, xc + (long)t * groups, c + (long)t * m, start, count, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> PlanarGroupDot(Vector256<byte> t0, Vector256<byte> t1, Vector256<byte> t2, Vector256<byte> t3,
        sbyte* xg, float* xsg, float gs, Vector256<float> acc, bool vnni)
    {
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
            Vector256<short> ones16 = Vector256.Create((short)1);
            i0 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t0, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg)), ones16);
            i1 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t1, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 32)), ones16);
            i2 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t2, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 64)), ones16);
            i3 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t3, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 96)), ones16);
        }
        Vector256<float> gacc = Avx.Multiply(Avx.ConvertToVector256Single(i0), Vector256.Create(xsg[0]));
        gacc = CpuFeatures.Fma ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(xsg[1]), gacc)
                               : gacc + Avx.ConvertToVector256Single(i1) * Vector256.Create(xsg[1]);
        gacc = CpuFeatures.Fma ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), Vector256.Create(xsg[2]), gacc)
                               : gacc + Avx.ConvertToVector256Single(i2) * Vector256.Create(xsg[2]);
        gacc = CpuFeatures.Fma ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), Vector256.Create(xsg[3]), gacc)
                               : gacc + Avx.ConvertToVector256Single(i3) * Vector256.Create(xsg[3]);
        return CpuFeatures.Fma ? Fma.MultiplyAdd(gacc, Vector256.Create(gs), acc) : acc + gacc * Vector256.Create(gs);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void PQ2PlanarBlockRowsAvx2(byte* w, sbyte* xq, float* xs, float* xc, float* c, int m, int start, int count, int k, int n)
    {
        int groups = k / PQ2_0GroupSize;
        bool vnni = CpuFeatures.AvxVnni;
        Vector256<byte> mask3 = Vector256.Create((byte)3);
        long qStride = k, sStride = (long)groups * 4, cStride = groups;
        for (int r = start; r < start + count; r++)
        {
            byte* row = w + (long)r * groups * PQ2_0GroupBytes;
            for (int tb = 0; tb < n; tb += PlanarTokenBlock)
            {
                int nb = Math.Min(PlanarTokenBlock, n - tb);
                sbyte* q0 = xq + tb * qStride; float* s0 = xs + tb * sStride; float* c0 = xc + tb * cStride;
                Vector256<float> a0 = Vector256<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                float k0 = 0f, k1 = 0f, k2 = 0f, k3 = 0f;
                for (int g = 0; g < groups; g++)
                {
                    byte* gp = row + g * PQ2_0GroupBytes;
                    float gs = HalfToSingle(gp);
                    Vector256<byte> v = Unsafe.ReadUnaligned<Vector256<byte>>(gp + 2);
                    Vector256<byte> t0 = Avx2.And(v, mask3);
                    Vector256<byte> t1 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 2).AsByte(), mask3);
                    Vector256<byte> t2 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte(), mask3);
                    Vector256<byte> t3 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 6).AsByte(), mask3);
                    long qo = (long)g * PQ2_0GroupSize, so = (long)g * 4;
                    a0 = PlanarGroupDot(t0, t1, t2, t3, q0 + qo, s0 + so, gs, a0, vnni); k0 += gs * c0[g];
                    if (nb > 1) { a1 = PlanarGroupDot(t0, t1, t2, t3, q0 + qStride + qo, s0 + sStride + so, gs, a1, vnni); k1 += gs * c0[cStride + g]; }
                    if (nb > 2) { a2 = PlanarGroupDot(t0, t1, t2, t3, q0 + 2 * qStride + qo, s0 + 2 * sStride + so, gs, a2, vnni); k2 += gs * c0[2 * cStride + g]; }
                    if (nb > 3) { a3 = PlanarGroupDot(t0, t1, t2, t3, q0 + 3 * qStride + qo, s0 + 3 * sStride + so, gs, a3, vnni); k3 += gs * c0[3 * cStride + g]; }
                }
                c[(long)tb * m + r] = HorizontalSumAvx2Float(a0) - k0;
                if (nb > 1) c[(long)(tb + 1) * m + r] = HorizontalSumAvx2Float(a1) - k1;
                if (nb > 2) c[(long)(tb + 2) * m + r] = HorizontalSumAvx2Float(a2) - k2;
                if (nb > 3) c[(long)(tb + 3) * m + r] = HorizontalSumAvx2Float(a3) - k3;
            }
        }
    }
}
