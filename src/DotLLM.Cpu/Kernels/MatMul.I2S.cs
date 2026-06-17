using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// BitNet b1.58 ternary (I2_S) matrix multiplication kernels.
/// Weights are ternary {-1, 0, +1} packed 4 codes per byte (block size 128), with a single
/// per-tensor float32 scale stored at the tensor tail (byte offset <c>m·k/4</c>).
///
/// Correctness-first implementation: each weight row is unpacked to float {-1,0,+1} and dotted
/// against the (full-precision) activation via <see cref="TensorPrimitives.Dot"/>, then scaled by
/// the per-tensor weight scale. Work is partitioned across output rows over the compute thread
/// pool. Phase 2 will replace the float unpack with an int8-activation (W2A8) VNNI/maddubs kernel
/// that consumes the packed 2-bit weights directly.
/// </summary>
public static unsafe partial class MatMul
{
    private const int I2SBlockSize = 128;

    // ── Context structs ──

    private struct GemvI2SCtx
    {
        public byte* Weights;
        public float* X;
        public float* Result;
        public int M;
        public int K;
        public float Scale;
    }

    private struct GemmI2SCtx
    {
        public byte* Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public float Scale;
    }

    /// <summary>
    /// I2_S ternary GEMV: <c>result[r] = scale · dot(ternary(A[r,:]), x)</c>.
    /// A is [M,K] packed I2_S (row-major, K a multiple of 128); x is f32 [K]; result is f32 [M].
    /// The per-tensor scale is read from the tail of <paramref name="weights"/>. Output rows are
    /// partitioned across <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvI2_S(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? threadPool)
    {
        if (k % I2SBlockSize != 0)
            throw new ArgumentException($"k must be a multiple of {I2SBlockSize}, got {k}", nameof(k));

        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);

        if (threadPool is null || m < ParallelMinRows)
        {
            GemvI2_SRows(weights, x, result, 0, m, k, scale);
            return;
        }

        var ctx = new GemvI2SCtx { Weights = weights, X = x, Result = result, M = m, K = k, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemvI2_SWorker);
    }

    [SkipLocalsInit]
    private static void GemvI2_SWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvI2_SRows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvI2_SRows(byte* weights, float* x, float* result,
                                     int startRow, int rowCount, int k, float scale)
    {
        int rowBytes = k / 4;
        var xSpan = new ReadOnlySpan<float>(x, k);

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackRow(weights + (long)r * rowBytes, rowSpan, k);
                result[r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan) * scale;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// I2_S ternary GEMM: <c>C[N,M] = (B[N,K] × ternary(A[M,K])^T) · scale</c>.
    /// Each weight row is unpacked once and dotted against all N input rows. Output rows are
    /// partitioned across <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmI2_S(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? threadPool)
    {
        if (n == 1)
        {
            GemvI2_S(weights, b, c, m, k, threadPool);
            return;
        }

        if (k % I2SBlockSize != 0)
            throw new ArgumentException($"k must be a multiple of {I2SBlockSize}, got {k}", nameof(k));

        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);

        if (threadPool is null || m < ParallelMinRows)
        {
            GemmI2_SRows(weights, b, c, m, 0, m, k, n, scale);
            return;
        }

        var ctx = new GemmI2SCtx { Weights = weights, B = b, C = c, M = m, K = k, N = n, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemmI2_SWorker);
    }

    [SkipLocalsInit]
    private static void GemmI2_SWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmI2_SRows(ctx.Weights, ctx.B, ctx.C, ctx.M, start, count, ctx.K, ctx.N, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> weight rows (over all N tokens) starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmI2_SRows(byte* weights, float* b, float* c, int m,
                                     int startRow, int rowCount, int k, int n, float scale)
    {
        int rowBytes = k / 4;

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackRow(weights + (long)r * rowBytes, rowSpan, k);
                for (int t = 0; t < n; t++)
                {
                    var xSpan = new ReadOnlySpan<float>(b + (long)t * k, k);
                    c[(long)t * m + r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Unpacks one I2_S-packed weight row (K codes, K a multiple of 128) into float {-1,0,+1}.
    /// Within each 128-element block, byte at <c>gp</c> holds elements {gp, +32, +64, +96}
    /// at bit offsets {6,4,2,0}; code value maps via <c>(code - 1)</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackRow(byte* rowPtr, Span<float> dest, int k)
    {
        int blocks = k / I2SBlockSize;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* bp = rowPtr + blk * 32;
            int outBase = blk * I2SBlockSize;
            for (int gp = 0; gp < 32; gp++)
            {
                byte packed = bp[gp];
                dest[outBase + gp] = ((packed >> 6) & 0x3) - 1;
                dest[outBase + gp + 32] = ((packed >> 4) & 0x3) - 1;
                dest[outBase + gp + 64] = ((packed >> 2) & 0x3) - 1;
                dest[outBase + gp + 96] = (packed & 0x3) - 1;
            }
        }
    }
}
