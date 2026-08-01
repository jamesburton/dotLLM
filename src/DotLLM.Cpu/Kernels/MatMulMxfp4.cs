using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// MXFP4 matrix-vector / matrix-matrix kernels. MXFP4 blocks pack 32 elements
/// as one E8M0 power-of-two scale byte plus 16 nibble bytes indexing the
/// doubled-E2M1 value table (see <see cref="Dequantize.Mxfp4Values"/>).
/// Activations are quantized to Q8_0 on the fly and the dot product runs on
/// integers per block — mirrors llama.cpp's <c>ggml_vec_dot_mxfp4_q8_0</c>.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>MXFP4 block size in bytes: 1 (E8M0 scale) + 16 (nibbles).</summary>
    private const int Mxfp4BlockBytes = 17;

    /// <summary>Elements per MXFP4 block.</summary>
    private const int Mxfp4GroupSize = 32;

    /// <summary>
    /// MXFP4 GEMV: A is MXFP4 [M,K], x is f32 [K]. Quantizes x to Q8_0 once,
    /// then computes the fused integer dot per row.
    /// </summary>
    /// <param name="weights">Pointer to MXFP4 weight data. Each row is K/32 blocks of 17 bytes.</param>
    /// <param name="x">Pointer to f32 input vector [K].</param>
    /// <param name="result">Pointer to f32 output vector [M].</param>
    /// <param name="m">Number of rows (output dimension).</param>
    /// <param name="k">Number of columns (input dimension). Must be a multiple of 32.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvMxfp4(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % Mxfp4GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Mxfp4GroupSize}, got {k}", nameof(k));

        int blockCount = k / Mxfp4GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        byte[]? rented = null;
        byte* xQ8;

        if (xQ8Bytes <= StackAllocThreshold)
        {
            byte* stackBuf = stackalloc byte[xQ8Bytes];
            xQ8 = stackBuf;
        }
        else
        {
            rented = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
            fixed (byte* rentedPtr = rented)
            {
                xQ8 = rentedPtr;
                QuantizeF32ToQ8_0(x, xQ8, k);
                ComputeRowsMxfp4(weights, xQ8, result, m, blockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_0(x, xQ8, k);
        ComputeRowsMxfp4(weights, xQ8, result, m, blockCount);
    }

    /// <summary>
    /// MXFP4 GEMV with optional row-parallel execution via <paramref name="pool"/>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvMxfp4(byte* weights, float* x, float* result, int m, int k,
                                 ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvMxfp4(weights, x, result, m, k);
            return;
        }

        if (k % Mxfp4GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Mxfp4GroupSize}, got {k}", nameof(k));

        int blockCount = k / Mxfp4GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, xQ8Bytes);
        QuantizeF32ToQ8_0(x, xQ8, k);

        var ctx = new ComputeRowsMxfp4Ctx
        {
            Weights = weights, XQ8 = xQ8, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsMxfp4Worker);
    }

    /// <summary>
    /// MXFP4 GEMM: C[N,M] = B[N,K] × A[M,K]^T where A is MXFP4 weights and B
    /// is f32 activations. Each input row is quantized to Q8_0 once and reused
    /// across all M weight rows.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmMxfp4(byte* weights, float* b, float* c, int m, int k, int n,
                                 ComputeThreadPool? pool = null, byte* preQuantizedInput = null)
    {
        if (k % Mxfp4GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Mxfp4GroupSize}, got {k}", nameof(k));

        int blockCount = k / Mxfp4GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;

        if (preQuantizedInput != null)
        {
            for (int t = 0; t < n; t++)
            {
                byte* xQ8 = preQuantizedInput + (long)t * q8RowBytes;
                ComputeRowsMxfp4Parallel(weights, xQ8, c + (long)t * m, m, blockCount, pool);
            }
            return;
        }

        byte[] rentedQ8 = ArrayPool<byte>.Shared.Rent(q8RowBytes);
        try
        {
            fixed (byte* xQ8 = rentedQ8)
            {
                for (int t = 0; t < n; t++)
                {
                    QuantizeF32ToQ8_0(b + (long)t * k, xQ8, k);
                    ComputeRowsMxfp4Parallel(weights, xQ8, c + (long)t * m, m, blockCount, pool);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rentedQ8);
        }
    }

    private struct ComputeRowsMxfp4Ctx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* Result;
        public int M;
        public int BlockCount;
    }

    private static void ComputeRowsMxfp4Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<ComputeRowsMxfp4Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        int rowBytes = ctx.BlockCount * Mxfp4BlockBytes;
        ComputeRowsMxfp4(ctx.Weights + (long)start * rowBytes, ctx.XQ8,
            ctx.Result + start, count, ctx.BlockCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeRowsMxfp4Parallel(byte* weights, byte* xQ8, float* result,
        int m, int blockCount, ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            ComputeRowsMxfp4(weights, xQ8, result, m, blockCount);
            return;
        }

        var ctx = new ComputeRowsMxfp4Ctx
        {
            Weights = weights, XQ8 = xQ8, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsMxfp4Worker);
    }

    /// <summary>
    /// Computes <paramref name="m"/> MXFP4×Q8_0 row dot products.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsMxfp4(byte* weights, byte* xQ8, float* result, int m, int blockCount)
    {
        long rowBytes = (long)blockCount * Mxfp4BlockBytes;

        if (Avx2.IsSupported && Ssse3.IsSupported)
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotMxfp4Q8_0Avx2(weights + row * rowBytes, xQ8, blockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotMxfp4Q8_0Scalar(weights + row * rowBytes, xQ8, blockCount);
        }
    }

    /// <summary>
    /// Scalar MXFP4 × Q8_0 dot product — reference implementation, port of
    /// llama.cpp's <c>ggml_vec_dot_mxfp4_q8_0</c> generic path:
    /// <c>sum over blocks of e8m0_half(e) * d_q8 * Σ kvalues[nib] * q8[i]</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotMxfp4Q8_0Scalar(byte* w, byte* xQ8, int blockCount)
    {
        float sumf = 0;
        ReadOnlySpan<sbyte> kvalues = Dequantize.Mxfp4Values;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wBlock = w + block * Mxfp4BlockBytes;
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;

            float d = Dequantize.E8M0ToFloatHalf(wBlock[0]);
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            byte* qs = wBlock + 1;
            sbyte* qx = (sbyte*)(xBlock + 2);

            int sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                sumi += kvalues[qs[j] & 0x0F] * qx[j];
                sumi += kvalues[qs[j] >> 4] * qx[j + 16];
            }

            sumf += d * dx * sumi;
        }

        return sumf;
    }

    /// <summary>
    /// AVX2 MXFP4 × Q8_0 dot product. Expands nibbles to sbyte weights via a
    /// <c>vpshufb</c> lookup of the 16-entry kvalue table, then reuses the
    /// signed×signed integer MAC pattern of the Q8_0 kernels (sign-flip trick +
    /// <c>vpmaddubsw</c> / <c>vpmaddwd</c>) with per-block float scaling.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotMxfp4Q8_0Avx2(byte* w, byte* xQ8, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        Vector128<byte> nibbleMask = Vector128.Create((byte)0x0F);
        Vector128<sbyte> kvalueTable = Vector128.Create(
            (sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

        for (int block = 0; block < blockCount; block++)
        {
            byte* wBlock = w + block * Mxfp4BlockBytes;
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;

            float d = Dequantize.E8M0ToFloatHalf(wBlock[0]);
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            // Unpack 16 nibble bytes → 32 sbyte weight values via table lookup.
            Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(wBlock + 1);
            Vector128<byte> lo = Sse2.And(qsRaw, nibbleMask);
            Vector128<byte> hi = Sse2.And(
                Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(), nibbleMask);
            Vector128<sbyte> wLo = Ssse3.Shuffle(kvalueTable, lo.AsSByte());
            Vector128<sbyte> wHi = Ssse3.Shuffle(kvalueTable, hi.AsSByte());
            Vector256<sbyte> vw = Vector256.Create(wLo, wHi);

            // Q8_0 activations: elements 0..15 pair with low nibbles, 16..31 with high.
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

            // Sign trick: abs(vw) unsigned × sign-adjusted vx.
            Vector256<sbyte> absW = Avx2.Sign(vw, vw);
            Vector256<sbyte> adjX = Avx2.Sign(vx, vw);

            Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), adjX);
            Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);

            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
            acc += fsum * Vector256.Create(d * dx);
        }

        return HorizontalSumAvx2Float(acc);
    }
}
