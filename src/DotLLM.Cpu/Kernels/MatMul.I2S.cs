using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// BitNet b1.58 ternary (I2_S) matrix multiplication kernels.
/// Weights are ternary {-1, 0, +1} packed 4 codes per byte (block size 128), with a single
/// per-tensor float32 scale stored at the tensor tail (byte offset <c>m·k/4</c>).
///
/// Two compute paths exist, selected at runtime by available ISA:
/// <list type="bullet">
/// <item><b>W2A8 (int8 activations)</b> on AVX2/AVX-VNNI hardware. Activations are quantized
/// once per token to Q8_0 (per-32-block absmax int8); each weight row is unpacked once to
/// int8 ternary {-1,0,+1} and dotted against the int8 activation blocks. For GEMM the row is
/// unpacked a single time and reused across all N tokens.</item>
/// <item><b>Float fallback</b> on hardware without AVX2 (e.g. Westmere). Each weight row is
/// unpacked to float {-1,0,+1} and dotted via <see cref="TensorPrimitives.Dot"/>.</item>
/// </list>
///
/// <para><b>ISA tiers (highest available first):</b></para>
/// <list type="number">
/// <item><see cref="AvxVnni"/> → <c>AvxVnni.MultiplyWideningAndAdd</c> (VPDPBUSD, 256-bit):
/// the ideal int8 dot. One instruction does unsigned×signed byte multiply with int32 accumulate
/// over 4-element groups.</item>
/// <item><see cref="Avx2"/> → <c>Avx2.MultiplyAddAdjacent</c>
/// (VPMADDUBSW, maddubs) followed by a widening MAD to int32. Uses the same sign trick as
/// <c>VecDotQ8_0Avx2</c>.</item>
/// <item>Float fallback via <see cref="TensorPrimitives.Dot"/> (kept for non-AVX2 boxes).</item>
/// </list>
///
/// <para><b>Sign trick (shared by both SIMD tiers).</b> VPDPBUSD and VPMADDUBSW both require an
/// unsigned left operand and a signed right operand. The weight ternary <c>w∈{-1,0,+1}</c> and the
/// activation int8 <c>q</c> are both signed, so we transform: <c>absW = |w| = Sign(w,w) ∈ {0,1}</c>
/// (unsigned) and <c>adjQ = sign(w)·q = Sign(q,w)</c> (signed). Then
/// <c>Σ absW·adjQ = Σ |w|·sign(w)·q = Σ w·q</c>. This mirrors the proven <c>VecDotQ8_0Avx2</c>
/// kernel exactly. (The algebraic alternative — the offset trick <c>Σw·q = Σcode·q − Σq</c> with
/// the unsigned code <c>w+1∈{0,1,2}</c> — is equivalent; the sign trick is used here because it
/// reuses one unpacked representation and matches the existing Q8_0 dot.)</para>
///
/// <para>Per-block dot scaling: for a Q8_0 activation block <c>b</c> with float scale <c>d_b</c>,
/// the int32 accumulator <c>Σ w·q</c> is multiplied by <c>d_b</c> and float-accumulated across
/// blocks; the final sum is multiplied by the per-tensor weight scale.</para>
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

    private struct GemvI2SQ8Ctx
    {
        public byte* Weights;
        public byte* XQ8;   // quantized activations (Q8_0), one token
        public float* Result;
        public int M;
        public int K;
        public float Scale;
    }

    private struct GemmI2SQ8Ctx
    {
        public byte* Weights;
        public byte* BQ8;   // quantized activations (Q8_0), N tokens contiguous
        public float* C;
        public int M;
        public int K;
        public int N;
        public float Scale;
    }

    /// <summary>True when a SIMD W2A8 (int8-activation) path is available.</summary>
    private static bool I2SUseW2A8 => Avx2.IsSupported;

    /// <summary>
    /// I2_S ternary GEMV: <c>result[r] = scale · dot(ternary(A[r,:]), x)</c>.
    /// A is [M,K] packed I2_S (row-major, K a multiple of 128); x is f32 [K]; result is f32 [M].
    /// The per-tensor scale is read from the tail of <paramref name="weights"/>. On AVX2/AVX-VNNI
    /// hardware the activation is quantized to int8 (Q8_0) once and the W2A8 SIMD path runs; older
    /// hardware falls back to the float path. Output rows are partitioned across
    /// <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvI2_S(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? threadPool)
    {
        if (k % I2SBlockSize != 0)
            throw new ArgumentException($"k must be a multiple of {I2SBlockSize}, got {k}", nameof(k));

        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);

        if (I2SUseW2A8)
        {
            GemvI2_SW2A8(weights, x, result, m, k, scale, threadPool);
            return;
        }

        if (threadPool is null || m < ParallelMinRows)
        {
            GemvI2_SRows(weights, x, result, 0, m, k, scale);
            return;
        }

        var ctx = new GemvI2SCtx { Weights = weights, X = x, Result = result, M = m, K = k, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemvI2_SWorker);
    }

    // ─────────────────────────── Float fallback (no AVX2) ───────────────────────────

    [SkipLocalsInit]
    private static void GemvI2_SWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvI2_SRows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows starting at <paramref name="startRow"/> (float path).</summary>
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
    /// Each weight row is unpacked once and dotted against all N input rows. On AVX2/AVX-VNNI
    /// hardware all N tokens are quantized to int8 (Q8_0) once and the W2A8 SIMD path runs (each
    /// weight row unpacked to int8 exactly once, then dotted against every token); older hardware
    /// falls back to the float path. Output rows are partitioned across <paramref name="threadPool"/>
    /// workers when present.
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

        if (I2SUseW2A8)
        {
            GemmI2_SW2A8(weights, b, c, m, k, n, scale, threadPool);
            return;
        }

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

    /// <summary>Computes <paramref name="rowCount"/> weight rows (over all N tokens) starting at <paramref name="startRow"/> (float path).</summary>
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

    // ─────────────────────────── W2A8 (int8-activation) SIMD path ───────────────────────────

    /// <summary>
    /// W2A8 GEMV: quantizes the activation to Q8_0 once, then partitions weight rows over the pool.
    /// </summary>
    [SkipLocalsInit]
    private static void GemvI2_SW2A8(byte* weights, float* x, float* result, int m, int k,
                                     float scale, ComputeThreadPool? threadPool)
    {
        int blockCount = k / Q8_0GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        byte[] xQ8Buf = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
        try
        {
            fixed (byte* xQ8 = xQ8Buf)
            {
                QuantizeF32ToQ8_0(x, xQ8, k);

                if (threadPool is null || m < ParallelMinRows)
                {
                    GemvI2_SW2A8Rows(weights, xQ8, result, 0, m, k, scale);
                    return;
                }

                var ctx = new GemvI2SQ8Ctx { Weights = weights, XQ8 = xQ8, Result = result, M = m, K = k, Scale = scale };
                threadPool.Dispatch((nint)(&ctx), &GemvI2_SW2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(xQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemvI2_SW2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvI2SQ8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvI2_SW2A8Rows(ctx.Weights, ctx.XQ8, ctx.Result, start, count, ctx.K, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows (W2A8) starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvI2_SW2A8Rows(byte* weights, byte* xQ8, float* result,
                                         int startRow, int rowCount, int k, float scale)
    {
        int rowBytes = k / 4;
        int blockCount = k / Q8_0GroupSize;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackRowI8(weights + (long)r * rowBytes, wI8, k);
                    result[r] = VecDotI2SQ8(wI8, xQ8, blockCount) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// W2A8 GEMM: quantizes all N tokens to Q8_0 once, then partitions weight rows over the pool.
    /// Each weight row is unpacked to int8 exactly once and dotted against every token (amortized).
    /// </summary>
    [SkipLocalsInit]
    private static void GemmI2_SW2A8(byte* weights, float* b, float* c, int m, int k, int n,
                                     float scale, ComputeThreadPool? threadPool)
    {
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        long bQ8Bytes = (long)n * q8RowBytes;

        byte[] bQ8Buf = ArrayPool<byte>.Shared.Rent(checked((int)bQ8Bytes));
        try
        {
            fixed (byte* bQ8 = bQ8Buf)
            {
                for (int t = 0; t < n; t++)
                    QuantizeF32ToQ8_0(b + (long)t * k, bQ8 + (long)t * q8RowBytes, k);

                if (threadPool is null || m < ParallelMinRows)
                {
                    GemmI2_SW2A8Rows(weights, bQ8, c, m, 0, m, k, n, scale);
                    return;
                }

                var ctx = new GemmI2SQ8Ctx { Weights = weights, BQ8 = bQ8, C = c, M = m, K = k, N = n, Scale = scale };
                threadPool.Dispatch((nint)(&ctx), &GemmI2_SW2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemmI2_SW2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmI2SQ8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmI2_SW2A8Rows(ctx.Weights, ctx.BQ8, ctx.C, ctx.M, start, count, ctx.K, ctx.N, ctx.Scale);
    }

    /// <summary>
    /// Computes <paramref name="rowCount"/> weight rows (over all N tokens, W2A8) starting at
    /// <paramref name="startRow"/>. The weight row is unpacked to int8 once then reused for all N.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmI2_SW2A8Rows(byte* weights, byte* bQ8, float* c, int m,
                                         int startRow, int rowCount, int k, int n, float scale)
    {
        int rowBytes = k / 4;
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackRowI8(weights + (long)r * rowBytes, wI8, k);   // unpack ONCE per row
                    for (int t = 0; t < n; t++)
                    {
                        byte* xQ8 = bQ8 + (long)t * q8RowBytes;
                        c[(long)t * m + r] = VecDotI2SQ8(wI8, xQ8, blockCount) * scale;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// W2A8 dot: <c>Σ_blocks d_b · Σ_{i∈block} w[i]·q[i]</c> for one int8 ternary weight row
    /// (<paramref name="wI8"/>, contiguous int8 {-1,0,+1}, length <c>k</c>) and one Q8_0-quantized
    /// activation row (<paramref name="xQ8"/>). Dispatches to the VNNI tier when available, else the
    /// AVX2 (maddubs) tier. Both tiers use the sign trick (see class summary).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float VecDotI2SQ8(sbyte* wI8, byte* xQ8, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        bool useVnni = AvxVnni.IsSupported;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            // Activation Q8_0 block: 2-byte Half scale + 32 sbyte values.
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            // 32 contiguous int8 weights aligned with this Q8_0 block.
            Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wI8 + block * Q8_0GroupSize);
            Vector256<sbyte> vq = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

            // Sign trick: absW = |w| ∈ {0,1} (unsigned operand), adjQ = sign(w)·q (signed operand).
            Vector256<sbyte> absW = Avx2.Sign(vw, vw);
            Vector256<sbyte> adjQ = Avx2.Sign(vq, vw);

            Vector256<int> isum;
            if (useVnni)
            {
                // VPDPBUSD: int32 += Σ (unsigned byte · signed byte) over 4-element groups.
                isum = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, absW.AsByte(), adjQ);
            }
            else
            {
                // VPMADDUBSW (maddubs): unsigned×signed → int16 pairs; then widen to int32.
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), adjQ);
                isum = Avx2.MultiplyAddAdjacent(prod, ones);
            }

            // int32 block sum → float, scale by activation block scale, accumulate.
            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
            Vector256<float> bscale = Vector256.Create(dx);
            if (Fma.IsSupported)
                acc = Fma.MultiplyAdd(bscale, fsum, acc);
            else
                acc += fsum * bscale;
        }

        return HorizontalSumAvx2Float(acc);
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

    /// <summary>
    /// Unpacks one I2_S-packed weight row (K codes, K a multiple of 128) into int8 ternary
    /// {-1,0,+1}, laid out contiguously so that each 32-element slice aligns with a Q8_0 block.
    /// Same bit layout as <see cref="UnpackRow"/>: within a 128-element block, byte at <c>gp</c>
    /// holds elements {gp, +32, +64, +96} at bit offsets {6,4,2,0}; ternary = <c>code - 1</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackRowI8(byte* rowPtr, sbyte* dest, int k)
    {
        int blocks = k / I2SBlockSize;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* bp = rowPtr + blk * 32;
            int outBase = blk * I2SBlockSize;
            for (int gp = 0; gp < 32; gp++)
            {
                byte packed = bp[gp];
                dest[outBase + gp] = (sbyte)(((packed >> 6) & 0x3) - 1);
                dest[outBase + gp + 32] = (sbyte)(((packed >> 4) & 0x3) - 1);
                dest[outBase + gp + 64] = (sbyte)(((packed >> 2) & 0x3) - 1);
                dest[outBase + gp + 96] = (sbyte)((packed & 0x3) - 1);
            }
        }
    }
}
