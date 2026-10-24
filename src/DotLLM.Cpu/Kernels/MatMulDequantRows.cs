using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Generic dequantize-and-dot GEMM/GEMV for weight formats that have no dedicated
/// <c>vec_dot</c> kernel (BF16, Q4_0/Q4_1/Q5_1, the K-quants below Q4_K, and the whole IQ family).
/// </summary>
/// <remarks>
/// <para>
/// This is the last-resort path, but "last resort" does not have to mean "single threaded".
/// The previous shape of this fallback dequantized one weight row at a time on the calling
/// thread and, for multi-token GEMM, repeated the whole dequantization once per token — so a
/// 512-token prefill dequantized every weight 512 times on one core. Both properties are fixed
/// here:
/// </para>
/// <list type="bullet">
///   <item><description><b>Row-parallel.</b> Output rows are independent, so the row range is
///   partitioned across <see cref="ComputeThreadPool"/> workers.</description></item>
///   <item><description><b>Dequantize once per row.</b> The decoded F32 row is reused for all
///   <c>n</c> input columns before moving on, turning an O(n·m) dequantize into O(m).</description></item>
/// </list>
/// <para>
/// Neither change alters the arithmetic: every output element is still
/// <c>TensorPrimitives.Dot(dequantize(row i), b + t*k)</c> over exactly the same inputs, so results
/// are bit-identical to the serial fallback this replaces. That is what makes the speedup safe to
/// take on formats whose perplexity is already pinned by the reference matrix.
/// </para>
/// </remarks>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Minimum output rows before the dequantize-and-dot path is worth dispatching to the pool.
    /// Below this the barrier cost dominates the (already small) row work.
    /// </summary>
    private const int DequantRowsParallelMinRows = 8;

    /// <summary>Work descriptor for <see cref="DequantRowsWorker"/>. Blittable — lives on the caller's stack.</summary>
    private struct DequantRowsCtx
    {
        public byte* Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public long RowBytes;
        public QuantizationType Qt;
    }

    private static void DequantRowsWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<DequantRowsCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        DequantRowsRange(ref ctx, start, count);
    }

    /// <summary>
    /// Computes output rows <c>[start, start+count)</c> for every input column.
    /// </summary>
    /// <remarks>
    /// The rented row buffer comes from <see cref="ArrayPool{T}.Shared"/>, whose per-thread cache
    /// makes the rent contention-free once each worker has run at least once. It is deliberately
    /// not <see cref="ComputeThreadPool.GetWorkerScratch"/>: several callers hand a worker-scratch
    /// pointer in as the pre-quantized input, and reusing the same arena here would alias it.
    /// </remarks>
    private static void DequantRowsRange(ref DequantRowsCtx ctx, int start, int count)
    {
        int k = ctx.K;
        int m = ctx.M;
        int n = ctx.N;
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            var rowRead = (ReadOnlySpan<float>)rowSpan;
            int end = start + count;
            for (int i = start; i < end; i++)
            {
                Dequantize.ToFloat32((nint)ctx.Weights + (nint)(i * ctx.RowBytes), k, ctx.Qt, rowSpan);

                // Reuse the decoded row across all n columns before evicting it.
                float* c = ctx.C + i;
                float* b = ctx.B;
                for (int t = 0; t < n; t++)
                {
                    c[(long)t * m] = TensorPrimitives.Dot(rowRead, new ReadOnlySpan<float>(b, k));
                    b += k;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Dequantize-and-dot GEMM: <c>c[t*m + i] = dot(dequantize(weights row i), b + t*k)</c>.
    /// </summary>
    /// <param name="weights">Quantized weight matrix, row-major, <paramref name="m"/> rows of <paramref name="k"/> elements.</param>
    /// <param name="qt">Weight quantization type. Any type <see cref="Dequantize.ToFloat32"/> accepts is valid.</param>
    /// <param name="b">Input matrix [<paramref name="n"/>, <paramref name="k"/>], row-major per column-vector.</param>
    /// <param name="c">Output matrix [<paramref name="n"/>, <paramref name="m"/>], row-major per column-vector.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Shared inner dimension.</param>
    /// <param name="n">Number of input column-vectors (1 for decode, seqLen for prefill).</param>
    /// <param name="pool">Thread pool for row-parallel execution. <see langword="null"/> runs serially.</param>
    public static void GemmDequantRows(byte* weights, QuantizationType qt, float* b, float* c,
                                       int m, int k, int n, ComputeThreadPool? pool)
    {
        if (m <= 0 || n <= 0) return;

        var ctx = new DequantRowsCtx
        {
            Weights = weights,
            B = b,
            C = c,
            M = m,
            K = k,
            N = n,
            RowBytes = Dequantize.RowByteSize(k, qt),
            Qt = qt,
        };

        if (pool != null && pool.ThreadCount > 1 && m >= DequantRowsParallelMinRows)
            pool.Dispatch((nint)(&ctx), &DequantRowsWorker);
        else
            DequantRowsRange(ref ctx, 0, m);
    }

    /// <summary>
    /// Dequantize-and-dot GEMV — the <c>n == 1</c> case of <see cref="GemmDequantRows"/>.
    /// </summary>
    /// <param name="weights">Quantized weight matrix, row-major, <paramref name="m"/> rows of <paramref name="k"/> elements.</param>
    /// <param name="qt">Weight quantization type.</param>
    /// <param name="x">Input vector of length <paramref name="k"/>.</param>
    /// <param name="y">Output vector of length <paramref name="m"/>.</param>
    /// <param name="m">Number of output rows.</param>
    /// <param name="k">Number of input elements per row.</param>
    /// <param name="pool">Thread pool for row-parallel execution. <see langword="null"/> runs serially.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void GemvDequantRows(byte* weights, QuantizationType qt, float* x, float* y,
                                       int m, int k, ComputeThreadPool? pool)
        => GemmDequantRows(weights, qt, x, y, m, k, 1, pool);

    /// <summary>
    /// Serial dequantize-and-dot GEMV, kept for callers with no thread pool in scope.
    /// </summary>
    /// <param name="weights">Quantized weight matrix, row-major, <paramref name="m"/> rows of <paramref name="k"/> elements.</param>
    /// <param name="qt">Weight quantization type.</param>
    /// <param name="x">Input vector of length <paramref name="k"/>.</param>
    /// <param name="y">Output vector of length <paramref name="m"/>.</param>
    /// <param name="m">Number of output rows.</param>
    /// <param name="k">Number of input elements per row.</param>
    internal static void GemvDequantRows(byte* weights, QuantizationType qt, float* x, float* y,
                                         int m, int k)
        => GemmDequantRows(weights, qt, x, y, m, k, 1, null);
}
