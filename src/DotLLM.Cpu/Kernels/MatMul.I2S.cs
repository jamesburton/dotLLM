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
/// the per-tensor weight scale. Using full-precision activations is at least as accurate as the
/// W2A8 reference. Phase 2 will add a parallel, int8-activation SIMD ternary kernel.
/// </summary>
public static unsafe partial class MatMul
{
    private const int I2SBlockSize = 128;

    /// <summary>
    /// I2_S ternary GEMV: <c>result[r] = scale · dot(ternary(A[r,:]), x)</c>.
    /// A is [M,K] packed I2_S (row-major, K a multiple of 128); x is f32 [K]; result is f32 [M].
    /// The per-tensor scale is read from the tail of <paramref name="weights"/>.
    /// </summary>
    /// <param name="weights">Pointer to the I2_S tensor base (packed data + trailing float32 scale).</param>
    /// <param name="x">Pointer to f32 input vector [K].</param>
    /// <param name="result">Pointer to f32 output vector [M].</param>
    /// <param name="m">Number of weight rows (output dimension).</param>
    /// <param name="k">Number of columns (input dimension). Must be a multiple of 128.</param>
    /// <param name="threadPool">Compute thread pool (currently unused; reserved for the Phase 2 parallel kernel).</param>
    [SkipLocalsInit]
    public static void GemvI2_S(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? threadPool)
    {
        if (k % I2SBlockSize != 0)
            throw new ArgumentException($"k must be a multiple of {I2SBlockSize}, got {k}", nameof(k));

        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);
        int rowBytes = k / 4;
        var xSpan = new ReadOnlySpan<float>(x, k);

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = 0; r < m; r++)
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
    /// Each weight row is unpacked once and dotted against all N input rows.
    /// </summary>
    /// <param name="weights">I2_S weight matrix [M,K] (packed + trailing float32 scale).</param>
    /// <param name="b">f32 input matrix [N,K], row-major.</param>
    /// <param name="c">f32 output matrix [N,M], row-major.</param>
    /// <param name="m">Number of weight rows (output dimension).</param>
    /// <param name="k">Number of columns (input dimension). Must be a multiple of 128.</param>
    /// <param name="n">Number of input tokens (batch size).</param>
    /// <param name="threadPool">Compute thread pool (currently unused; reserved for the Phase 2 parallel kernel).</param>
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
        int rowBytes = k / 4;

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = 0; r < m; r++)
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
