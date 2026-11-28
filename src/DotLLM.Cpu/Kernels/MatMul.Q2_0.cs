using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Upstream ggml <c>Q2_0</c> x Q8_0 kernels (issue #2074), read in place: the weights are never expanded.
/// Block: fp16 <c>d</c> + 16 code bytes = 18 B per 64 elements; element j sits in bits <c>2*(j%4)</c> of byte <c>j/4</c>;
/// code 0..3 means -1, 0, +1, +2 (times <c>d</c>). Port of llama.cpp b12780 <c>ggml_vec_dot_q2_0_q8_0_generic</c>
/// (<c>vec_dot_type = Q8_0</c>, one Q2_0 block = two Q8_0 blocks): the integer sum per Q8_0 block is exact and the float
/// combine order is the reference's, so the SIMD path is bit-identical to the scalar one.
/// </summary>
public static unsafe partial class MatMul
{
    private const int Q2_0Group = QuantFormat.Q2_0GroupSize;     // 64
    private const int Q2_0Bytes = QuantFormat.Q2_0BlockBytes;    // 18

    /// <summary>Scalar reference: line-for-line port of <c>ggml_vec_dot_q2_0_q8_0_generic</c>. <paramref name="blocks"/> counts Q2_0 blocks.</summary>
    internal static float VecDotQ2_0Q8_0Scalar(byte* w, byte* q8, int blocks)
    {
        float sumf = 0f;
        for (int i = 0; i < blocks; i++)
        {
            byte* wb = w + (long)i * Q2_0Bytes;
            float d0 = HalfToSingle(wb);
            float sumi = 0f;
            for (int k = 0; k < 2; k++)
            {
                byte* yb = q8 + (long)(i * 2 + k) * Q8_0BlockBytes;
                float d1 = HalfToSingle(yb);
                sbyte* qy = (sbyte*)(yb + 2);
                byte* qs = wb + 2 + k * 8;
                int s = 0;
                for (int b = 0; b < 8; b++)
                {
                    int by = qs[b];
                    s += ((by & 3) - 1) * qy[b * 4 + 0];
                    s += (((by >> 2) & 3) - 1) * qy[b * 4 + 1];
                    s += (((by >> 4) & 3) - 1) * qy[b * 4 + 2];
                    s += (((by >> 6) & 3) - 1) * qy[b * 4 + 3];
                }
                sumi += d1 * s;
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

    /// <summary>
    /// AVX2 Q2_0 x Q8_0 dot. <c>sum((q-1)*y) = sum(q*y) - sum(y)</c>: unsigned codes through <c>vpmaddubsw</c> (codes are 0..3, so the int16
    /// pair sums cannot saturate: 2*3*127), the activation sum through <c>vpmaddubsw(1, y)</c>. Same float combine order as the scalar.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ2_0Q8_0Avx2(byte* w, byte* q8, int blocks)
    {
        var mask3 = Vector128.Create((byte)3);
        var ones8 = Vector256.Create((byte)1);
        var ones16 = Vector256.Create((short)1);
        float sumf = 0f;
        for (int i = 0; i < blocks; i++)
        {
            byte* wb = w + (long)i * Q2_0Bytes;
            float d0 = HalfToSingle(wb);
            var v = Sse2.LoadVector128(wb + 2);
            var t0 = Sse2.And(v, mask3);
            var t1 = Sse2.And(Sse2.ShiftRightLogical(v.AsUInt16(), 2).AsByte(), mask3);
            var t2 = Sse2.And(Sse2.ShiftRightLogical(v.AsUInt16(), 4).AsByte(), mask3);
            var t3 = Sse2.And(Sse2.ShiftRightLogical(v.AsUInt16(), 6).AsByte(), mask3);
            // byte b holds elements 4b+s in t_s[b]: interleave back to element order.
            var loA = Sse2.UnpackLow(t0, t1);  var loB = Sse2.UnpackLow(t2, t3);
            var hiA = Sse2.UnpackHigh(t0, t1); var hiB = Sse2.UnpackHigh(t2, t3);
            var e0 = Sse2.UnpackLow(loA.AsUInt16(), loB.AsUInt16()).AsByte();    // elements 0..15
            var e1 = Sse2.UnpackHigh(loA.AsUInt16(), loB.AsUInt16()).AsByte();   // 16..31
            var e2 = Sse2.UnpackLow(hiA.AsUInt16(), hiB.AsUInt16()).AsByte();    // 32..47
            var e3 = Sse2.UnpackHigh(hiA.AsUInt16(), hiB.AsUInt16()).AsByte();   // 48..63
            var w0 = Vector256.Create(e0, e1);
            var w1 = Vector256.Create(e2, e3);

            float sumi = 0f;
            for (int k = 0; k < 2; k++)
            {
                byte* yb = q8 + (long)(i * 2 + k) * Q8_0BlockBytes;
                float d1 = HalfToSingle(yb);
                var y = Avx.LoadVector256((sbyte*)(yb + 2));
                var wv = k == 0 ? w0 : w1;
                var dot = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(wv, y), ones16);
                var ysum = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(ones8, y), ones16);
                var d = Avx2.Subtract(dot, ysum);
                var s128 = Sse2.Add(d.GetLower(), d.GetUpper());
                s128 = Sse2.Add(s128, Sse2.Shuffle(s128, 0x4E));
                s128 = Sse2.Add(s128, Sse2.Shuffle(s128, 0xB1));
                sumi += d1 * s128.ToScalar();
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float VecDotQ2_0Q8_0(byte* w, byte* q8, int blocks)
        => CpuFeatures.Avx2 ? VecDotQ2_0Q8_0Avx2(w, q8, blocks) : VecDotQ2_0Q8_0Scalar(w, q8, blocks);

    private struct Q2_0Ctx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* C;
        public int M, K, N;
    }

    private static void Q2_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<Q2_0Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count > 0) Q2_0Rows(ref ctx, start, count);
    }

    private static void Q2_0Rows(ref Q2_0Ctx ctx, int start, int count)
    {
        int blocks = ctx.K / Q2_0Group;
        long wRow = (long)blocks * Q2_0Bytes;
        long xRow = (long)blocks * 2 * Q8_0BlockBytes;
        for (int r = start; r < start + count; r++)
        {
            byte* wr = ctx.Weights + r * wRow;
            for (int t = 0; t < ctx.N; t++)
                ctx.C[(long)t * ctx.M + r] = VecDotQ2_0Q8_0(wr, ctx.XQ8 + t * xRow, blocks);
        }
    }

    /// <summary>Q2_0 GEMV: <c>result[m] = W[m,k] . x[k]</c> with the activation quantized to Q8_0 once (ggml's <c>vec_dot_type</c>).</summary>
    public static void GemvQ2_0(byte* weights, float* x, float* result, int m, int k, ComputeThreadPool? pool)
        => GemmQ2_0(weights, x, result, m, k, 1, pool);

    /// <summary>Q2_0 GEMM: <c>c[t*m + r] = W[r,:] . b[t,:]</c> for n activation rows; every row is quantized to Q8_0 once, weights are read packed.</summary>
    public static void GemmQ2_0(byte* weights, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        if (k % Q2_0Group != 0)
            throw new ArgumentException($"k must be a multiple of {Q2_0Group}, got {k}", nameof(k));
        long xRow = (long)(k / Q8_0GroupSize) * Q8_0BlockBytes;
        byte[] rented = ArrayPool<byte>.Shared.Rent(checked((int)(n * xRow)));
        try
        {
            fixed (byte* xq = rented)
            {
                for (int t = 0; t < n; t++)
                    QuantizeF32ToQ8_0(b + (long)t * k, xq + t * xRow, k);
                var ctx = new Q2_0Ctx { Weights = weights, XQ8 = xq, C = c, M = m, K = k, N = n };
                if (pool is null || m < ParallelMinRows) Q2_0Rows(ref ctx, 0, m);
                else pool.Dispatch((nint)(&ctx), &Q2_0Worker);
            }
        }
        finally { ArrayPool<byte>.Shared.Return(rented); }
    }
}
