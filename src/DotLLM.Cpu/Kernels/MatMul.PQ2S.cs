using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// PrismML Bonsai PQ2_0 ternary matrix multiplication kernels.
/// Weights are ternary {-1, 0, +1} packed 4 codes per byte, grouped in 128-element groups;
/// unlike I2_S (a single per-tensor tail scale), each group carries its own Half scale
/// immediately before its 32 packed code bytes — <c>scale(Half) + codes[32]</c>, 34 bytes/group
/// (see <see cref="Dequantize.DequantizePQ2_0"/> for the empirically-verified byte layout).
///
/// <para>Correctness-first scalar implementation only (mirrors <c>MatMul.I2S.cs</c>'s float
/// fallback tier): each weight row is unpacked once into a per-group-scaled float buffer, then
/// dotted via <see cref="TensorPrimitives.Dot"/>. An AVX2/AVX-VNNI W2A8 tier (like I2_S's) is a
/// follow-on optimization, not required for initial correctness.</para>
/// </summary>
public static unsafe partial class MatMul
{
    private const int PQ2_0GroupSize = 128;
    private const int PQ2_0GroupBytes = 34;

    private struct GemvPQ2SCtx
    {
        public byte* Weights;
        public float* X;
        public float* Result;
        public int M;
        public int K;
    }

    private struct GemmPQ2SCtx
    {
        public byte* Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
    }

    /// <summary>
    /// PQ2_0 ternary GEMV: <c>result[r] = dot(perGroupScaled(A[r,:]), x)</c>.
    /// A is [M,K] packed PQ2_0 (row-major, K a multiple of 128, row stride
    /// <c>(K/128)·34</c> bytes); x is f32 [K]; result is f32 [M]. Output rows are partitioned
    /// across <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvPQ2_0(byte* weights, float* x, float* result, int m, int k,
                                 ComputeThreadPool? threadPool)
    {
        if (k % PQ2_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));

        if (threadPool is null || m < ParallelMinRows)
        {
            GemvPQ2_0Rows(weights, x, result, 0, m, k);
            return;
        }

        var ctx = new GemvPQ2SCtx { Weights = weights, X = x, Result = result, M = m, K = k };
        threadPool.Dispatch((nint)(&ctx), &GemvPQ2_0Worker);
    }

    [SkipLocalsInit]
    private static void GemvPQ2_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvPQ2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvPQ2_0Rows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvPQ2_0Rows(byte* weights, float* x, float* result,
                                      int startRow, int rowCount, int k)
    {
        int rowBytes = (k / PQ2_0GroupSize) * PQ2_0GroupBytes;
        var xSpan = new ReadOnlySpan<float>(x, k);

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackPQ2_0Row(weights + (long)r * rowBytes, rowSpan, k);
                result[r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// PQ2_0 ternary GEMM: <c>C[N,M] = B[N,K] × perGroupScaled(A[M,K])^T</c>.
    /// Each weight row is unpacked once and dotted against all N input rows. Output rows are
    /// partitioned across <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmPQ2_0(byte* weights, float* b, float* c, int m, int k, int n,
                                 ComputeThreadPool? threadPool)
    {
        if (n == 1)
        {
            GemvPQ2_0(weights, b, c, m, k, threadPool);
            return;
        }

        if (k % PQ2_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));

        if (threadPool is null || m < ParallelMinRows)
        {
            GemmPQ2_0Rows(weights, b, c, m, 0, m, k, n);
            return;
        }

        var ctx = new GemmPQ2SCtx { Weights = weights, B = b, C = c, M = m, K = k, N = n };
        threadPool.Dispatch((nint)(&ctx), &GemmPQ2_0Worker);
    }

    [SkipLocalsInit]
    private static void GemmPQ2_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmPQ2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmPQ2_0Rows(ctx.Weights, ctx.B, ctx.C, ctx.M, start, count, ctx.K, ctx.N);
    }

    /// <summary>Computes <paramref name="rowCount"/> weight rows (over all N tokens) starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmPQ2_0Rows(byte* weights, float* b, float* c, int m,
                                      int startRow, int rowCount, int k, int n)
    {
        int rowBytes = (k / PQ2_0GroupSize) * PQ2_0GroupBytes;

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackPQ2_0Row(weights + (long)r * rowBytes, rowSpan, k);
                for (int t = 0; t < n; t++)
                {
                    var xSpan = new ReadOnlySpan<float>(b + (long)t * k, k);
                    c[(long)t * m + r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Unpacks one PQ2_0-packed weight row (K codes, K a multiple of 128) into float
    /// {-scale,0,+scale}, per-group. Same bit convention as I2_S's <c>UnpackRow</c> (byte at
    /// <c>gp</c> holds elements {gp, +32, +64, +96} at bit offsets {6,4,2,0}), but each 34-byte
    /// group carries its own leading Half scale rather than sharing one per-tensor tail scale.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackPQ2_0Row(byte* rowPtr, Span<float> dest, int k)
    {
        int groups = k / PQ2_0GroupSize;
        for (int g = 0; g < groups; g++)
        {
            byte* groupBase = rowPtr + g * PQ2_0GroupBytes;
            float scale = (float)Unsafe.ReadUnaligned<Half>(groupBase);
            byte* codes = groupBase + 2;
            int outBase = g * PQ2_0GroupSize;

            for (int gp = 0; gp < 32; gp++)
            {
                byte packed = codes[gp];
                dest[outBase + gp] = (((packed >> 6) & 0x3) - 1) * scale;
                dest[outBase + gp + 32] = (((packed >> 4) & 0x3) - 1) * scale;
                dest[outBase + gp + 64] = (((packed >> 2) & 0x3) - 1) * scale;
                dest[outBase + gp + 96] = ((packed & 0x3) - 1) * scale;
            }
        }
    }
}
