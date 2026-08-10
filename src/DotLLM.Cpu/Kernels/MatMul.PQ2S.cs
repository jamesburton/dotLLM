using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// PrismML Bonsai PQ2_0 ternary matrix multiplication kernels.
/// Weights are ternary {-1, 0, +1} packed 4 codes per byte, grouped in 128-element groups;
/// unlike I2_S (a single per-tensor tail scale), each group carries its own Half scale
/// immediately before its 32 packed code bytes — <c>scale(Half) + codes[32]</c>, 34 bytes/group
/// (see <see cref="Dequantize.DequantizePQ2_0"/> for the empirically-verified byte layout).
///
/// <para>Two compute paths exist, mirroring <c>MatMul.I2S.cs</c>'s structure:</para>
/// <list type="bullet">
/// <item><b>W2A8 (int8 activations)</b> on AVX2/AVX-VNNI hardware. Activations are quantized
/// once per token to Q8_0 (per-32-element absmax int8); each weight row is unpacked once to
/// int8 ternary {-1,0,+1} (plus its per-128-group float scales) and dotted against the int8
/// activation blocks via the same VPDPBUSD/VPMADDUBSW sign-trick dot as I2_S's
/// <c>VecDotI2SQ8</c>. <b>Key difference from I2_S</b>: I2_S has a single per-tensor scale
/// applied once to the finished dot; PQ2_0 has a per-128-element group scale that must be
/// folded into the per-Q8_0-block accumulation (each 128-element PQ2_0 group spans exactly 4
/// Q8_0 blocks of 32 elements, since <see cref="PQ2_0GroupSize"/> == 4 · <c>Q8_0GroupSize</c>),
/// multiplied in alongside the activation's own Q8_0 block scale — see
/// <see cref="VecDotPQ2_0Q8"/>.</item>
/// <item><b>Float fallback</b> (unpack row once into a per-group-scaled float buffer, then dot
/// via <see cref="TensorPrimitives.Dot"/>): used on hardware without AVX2 (e.g. Westmere), and
/// — <b>unlike I2_S</b> — always used by <see cref="GemmPQ2_0"/> (GEMM/prefill) regardless of
/// ISA. Issue #204 added the W2A8 tier and benchmarked both entry points: GEMV/decode got a
/// solid, repeatable 2.4x-3.1x win and dispatches to W2A8 on AVX2/AVX-VNNI hardware as expected,
/// but GEMM/prefill showed no reliable win and a real regression on one shape (ffn_gate,
/// 0.58x-0.86x vs scalar across separate runs) — so <see cref="GemmPQ2_0"/> was kept on the
/// scalar tier unconditionally after review. The W2A8 GEMM code
/// (<see cref="GemmPQ2_0W2A8"/>/<see cref="GemmPQ2_0W2A8Rows"/>) is retained, tested via
/// <see cref="GemmPQ2_0W2A8ForTest"/>, and available for a future investigation.</item>
/// </list>
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

    private struct GemvPQ2SQ8Ctx
    {
        public byte* Weights;
        public byte* XQ8;   // quantized activations (Q8_0), one token
        public float* Result;
        public int M;
        public int K;
    }

    private struct GemmPQ2SQ8Ctx
    {
        public byte* Weights;
        public byte* BQ8;   // quantized activations (Q8_0), N tokens contiguous
        public float* C;
        public int M;
        public int K;
        public int N;
    }

    /// <summary>True when a SIMD W2A8 (int8-activation) path is available.</summary>
    private static bool PQ2_0UseW2A8 => Avx2.IsSupported;

    /// <summary>
    /// PQ2_0 ternary GEMV: <c>result[r] = dot(perGroupScaled(A[r,:]), x)</c>.
    /// A is [M,K] packed PQ2_0 (row-major, K a multiple of 128, row stride
    /// <c>(K/128)·34</c> bytes); x is f32 [K]; result is f32 [M]. On AVX2/AVX-VNNI hardware the
    /// activation is quantized to int8 (Q8_0) once and the W2A8 SIMD path runs; older hardware
    /// falls back to the float path. Output rows are partitioned across
    /// <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvPQ2_0(byte* weights, float* x, float* result, int m, int k,
                                 ComputeThreadPool? threadPool)
    {
        if (k % PQ2_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));

        if (PQ2_0UseW2A8)
        {
            GemvPQ2_0W2A8(weights, x, result, m, k, threadPool);
            return;
        }

        if (threadPool is null || m < ParallelMinRows)
        {
            GemvPQ2_0Rows(weights, x, result, 0, m, k);
            return;
        }

        var ctx = new GemvPQ2SCtx { Weights = weights, X = x, Result = result, M = m, K = k };
        threadPool.Dispatch((nint)(&ctx), &GemvPQ2_0Worker);
    }

    /// <summary>
    /// Benchmark/test-only entry point that always takes the scalar reference tier (unpack row to
    /// per-group-scaled float + <see cref="TensorPrimitives.Dot"/>), bypassing the
    /// <see cref="PQ2_0UseW2A8"/> SIMD dispatch that <see cref="GemvPQ2_0"/> otherwise always
    /// takes on AVX2 hardware. Exists so <c>PQ2_0DecodeBenchmark</c> can compare the two tiers
    /// head-to-head on the same box (mirrors the same "always-available baseline" need the
    /// scalar tier serves for I2_S, minus a dedicated bench file — PQ2_0 only needed this one
    /// forwarding shim, not a whole streaming/unpack-only probe suite).
    /// </summary>
    [SkipLocalsInit]
    internal static void GemvPQ2_0Scalar(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % PQ2_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));
        GemvPQ2_0Rows(weights, x, result, 0, m, k);
    }

    /// <summary>GEMM analog of <see cref="GemvPQ2_0Scalar"/> — always takes the scalar reference tier.</summary>
    [SkipLocalsInit]
    internal static void GemmPQ2_0Scalar(byte* weights, float* b, float* c, int m, int k, int n)
    {
        if (n == 1)
        {
            GemvPQ2_0Scalar(weights, b, c, m, k);
            return;
        }

        if (k % PQ2_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));
        GemmPQ2_0Rows(weights, b, c, m, 0, m, k, n);
    }

    // ─────────────────────────── Float fallback (no AVX2) ───────────────────────────

    [SkipLocalsInit]
    private static void GemvPQ2_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvPQ2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvPQ2_0Rows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows starting at <paramref name="startRow"/> (float path).</summary>
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
    /// Each weight row is unpacked once and dotted against all N input rows via the scalar
    /// float reference tier. Output rows are partitioned across <paramref name="threadPool"/>
    /// workers when present.
    ///
    /// <para><b>Not wired to the W2A8 SIMD tier (issue #204 review follow-up).</b> Unlike
    /// <see cref="GemvPQ2_0"/>, this entry point deliberately does <i>not</i> dispatch to
    /// <see cref="GemmPQ2_0W2A8"/> even on AVX2/AVX-VNNI hardware: benchmarking
    /// (<c>PQ2_0DecodeBenchmark.GemmPQ2_0_W2A8VsScalar_MedianOf5</c>) showed the W2A8 tier gives
    /// no reliable win for GEMM/prefill and a real, repeatable regression on the ffn_gate shape
    /// (0.58x-0.86x vs scalar across separate runs), unlike GEMV/decode's solid 2.4x-3.1x. The
    /// W2A8 GEMM code path (<see cref="GemmPQ2_0W2A8"/>/<see cref="GemmPQ2_0W2A8Rows"/>) is kept
    /// — it is exercised directly by <see cref="GemmPQ2_0Scalar"/>'s sibling benchmark comparison
    /// and by <c>GemmPQ2_0_W2A8_MatchesFloatReference_WithinQuantTolerance</c> — so it remains
    /// available for a future investigation, but is not reachable from the public GEMM entry
    /// point until that investigation lands.</para>
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

    /// <summary>Computes <paramref name="rowCount"/> weight rows (over all N tokens) starting at <paramref name="startRow"/> (float path).</summary>
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
    /// {-scale,0,+scale,+2*scale}, per-group. NOT the same bit convention as I2_S's
    /// <c>UnpackRow</c> — verified against PrismML's own reference <c>dequantize_row_q2_0</c>
    /// (<c>PrismML-Eng/llama.cpp</c>, <c>ggml-quants.c</c>): byte at index <c>b</c> holds the 4
    /// CONSECUTIVE elements {4b, 4b+1, 4b+2, 4b+3} at ASCENDING bit offsets {0,2,4,6}. An earlier
    /// version of this function wrongly copied I2_S's strided {gp,+32,+64,+96}/descending-bits
    /// scheme (issue #269 follow-up, 2026-08-05) — see <see cref="Dequantize.DequantizePQ2_0"/>
    /// for the full root-cause writeup. Each 34-byte group carries its own leading Half scale
    /// rather than sharing one per-tensor tail scale.
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

            for (int b = 0; b < 32; b++)
            {
                byte packed = codes[b];
                int outIdx = outBase + 4 * b;
                dest[outIdx] = ((packed & 0x3) - 1) * scale;
                dest[outIdx + 1] = (((packed >> 2) & 0x3) - 1) * scale;
                dest[outIdx + 2] = (((packed >> 4) & 0x3) - 1) * scale;
                dest[outIdx + 3] = (((packed >> 6) & 0x3) - 1) * scale;
            }
        }
    }

    // ─────────────────────────── W2A8 (int8-activation) SIMD path ───────────────────────────

    /// <summary>
    /// W2A8 GEMV: quantizes the activation to Q8_0 once, then partitions weight rows over the pool.
    /// </summary>
    [SkipLocalsInit]
    private static void GemvPQ2_0W2A8(byte* weights, float* x, float* result, int m, int k,
                                      ComputeThreadPool? threadPool)
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
                    GemvPQ2_0W2A8Rows(weights, xQ8, result, 0, m, k);
                    return;
                }

                var ctx = new GemvPQ2SQ8Ctx { Weights = weights, XQ8 = xQ8, Result = result, M = m, K = k };
                threadPool.Dispatch((nint)(&ctx), &GemvPQ2_0W2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(xQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemvPQ2_0W2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvPQ2SQ8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvPQ2_0W2A8Rows(ctx.Weights, ctx.XQ8, ctx.Result, start, count, ctx.K);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows (W2A8) starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvPQ2_0W2A8Rows(byte* weights, byte* xQ8, float* result,
                                          int startRow, int rowCount, int k)
    {
        int rowBytes = (k / PQ2_0GroupSize) * PQ2_0GroupBytes;
        int blockCount = k / Q8_0GroupSize;
        int groupCount = k / PQ2_0GroupSize;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        float[] groupScaleBuf = ArrayPool<float>.Shared.Rent(groupCount);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            fixed (float* groupScales = groupScaleBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackPQ2_0RowI8(weights + (long)r * rowBytes, wI8, groupScales, k);
                    result[r] = VecDotPQ2_0Q8(wI8, groupScales, xQ8, blockCount);
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
            ArrayPool<float>.Shared.Return(groupScaleBuf);
        }
    }

    /// <summary>
    /// Test-only entry point for the W2A8 SIMD GEMM tier, which <see cref="GemmPQ2_0"/> does
    /// <b>not</b> dispatch to (see its doc comment — issue #204 review found no reliable GEMM
    /// win and a real regression on the ffn_gate shape). Lets
    /// <c>GemmPQ2_0_W2A8_MatchesFloatReference_WithinQuantTolerance</c> keep validating this
    /// path's correctness (in particular the per-group-scale folding shared with the GEMV tier)
    /// even though it is currently unreachable from production code, so it stays ready for a
    /// future investigation without silently bit-rotting.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemmPQ2_0W2A8ForTest(byte* weights, float* b, float* c, int m, int k, int n)
    {
        if (k % PQ2_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));
        GemmPQ2_0W2A8(weights, b, c, m, k, n, null);
    }

    /// <summary>
    /// W2A8 GEMM: quantizes all N tokens to Q8_0 once, then partitions weight rows over the pool.
    /// Each weight row is unpacked to int8 (+ per-group scales) exactly once and dotted against
    /// every token (amortized). <b>Not currently called by <see cref="GemmPQ2_0"/></b> — see that
    /// method's doc comment.
    /// </summary>
    [SkipLocalsInit]
    private static void GemmPQ2_0W2A8(byte* weights, float* b, float* c, int m, int k, int n,
                                      ComputeThreadPool? threadPool)
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
                    GemmPQ2_0W2A8Rows(weights, bQ8, c, m, 0, m, k, n);
                    return;
                }

                var ctx = new GemmPQ2SQ8Ctx { Weights = weights, BQ8 = bQ8, C = c, M = m, K = k, N = n };
                threadPool.Dispatch((nint)(&ctx), &GemmPQ2_0W2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemmPQ2_0W2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmPQ2SQ8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmPQ2_0W2A8Rows(ctx.Weights, ctx.BQ8, ctx.C, ctx.M, start, count, ctx.K, ctx.N);
    }

    /// <summary>
    /// Computes <paramref name="rowCount"/> weight rows (over all N tokens, W2A8) starting at
    /// <paramref name="startRow"/>. The weight row is unpacked to int8 + per-group scales once
    /// then reused for all N.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmPQ2_0W2A8Rows(byte* weights, byte* bQ8, float* c, int m,
                                          int startRow, int rowCount, int k, int n)
    {
        int rowBytes = (k / PQ2_0GroupSize) * PQ2_0GroupBytes;
        int blockCount = k / Q8_0GroupSize;
        int groupCount = k / PQ2_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        float[] groupScaleBuf = ArrayPool<float>.Shared.Rent(groupCount);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            fixed (float* groupScales = groupScaleBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackPQ2_0RowI8(weights + (long)r * rowBytes, wI8, groupScales, k);   // unpack ONCE per row
                    for (int t = 0; t < n; t++)
                    {
                        byte* xQ8 = bQ8 + (long)t * q8RowBytes;
                        c[(long)t * m + r] = VecDotPQ2_0Q8(wI8, groupScales, xQ8, blockCount);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
            ArrayPool<float>.Shared.Return(groupScaleBuf);
        }
    }

    /// <summary>
    /// W2A8 dot: <c>Σ_blocks (d_b · g_{block/4}) · Σ_{i∈block} w[i]·q[i]</c> for one int8 ternary
    /// weight row (<paramref name="wI8"/>, contiguous int8 {-1,0,+1}, length <c>k</c>), its
    /// per-128-element PQ2_0 group scales (<paramref name="groupScales"/>, length
    /// <c>k/128</c>), and one Q8_0-quantized activation row (<paramref name="xQ8"/>).
    ///
    /// <para><b>Group-scale folding (the key difference from I2_S's <c>VecDotI2SQ8</c>).</b>
    /// I2_S has a single per-tensor scale applied once to the whole finished dot. PQ2_0 instead
    /// carries one Half scale per 128-element group, and since
    /// <see cref="PQ2_0GroupSize"/> (128) is exactly 4 × <c>Q8_0GroupSize</c> (32), each Q8_0
    /// activation block falls entirely within one PQ2_0 group — so the weight-side group scale
    /// for block <c>b</c> is <c>groupScales[b / 4]</c>. This is multiplied together with the
    /// activation's own Q8_0 block scale <c>d_b</c> into a single combined per-block float
    /// scale before the int32→float block-sum conversion, exactly where <c>VecDotI2SQ8</c>
    /// applies <c>d_b</c> alone — the per-tensor <c>scale</c> multiply I2_S does at the very end
    /// has no equivalent here (there is no single tensor-wide scale to apply).</para>
    ///
    /// <para>Dispatches to the VNNI tier when available, else the AVX2 (maddubs) tier. Both
    /// tiers use the sign trick (see <c>MatMul.I2S.cs</c>'s class summary for the derivation).</para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float VecDotPQ2_0Q8(sbyte* wI8, float* groupScales, byte* xQ8, int blockCount)
    {
        const int blocksPerGroup = PQ2_0GroupSize / Q8_0GroupSize; // 128 / 32 = 4

        Vector256<float> acc = Vector256<float>.Zero;
        bool useVnni = AvxVnni.IsSupported;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            // Activation Q8_0 block: 2-byte Half scale + 32 sbyte values.
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            // Weight-side per-128-group scale: 4 consecutive Q8_0 blocks share one PQ2_0 group.
            float gScale = groupScales[block / blocksPerGroup];
            float dCombined = dx * gScale;

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

            // int32 block sum → float, scale by combined (activation-block · weight-group) scale, accumulate.
            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
            Vector256<float> bscale = Vector256.Create(dCombined);
            if (Fma.IsSupported)
                acc = Fma.MultiplyAdd(bscale, fsum, acc);
            else
                acc += fsum * bscale;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// Unpacks one PQ2_0-packed weight row (K codes, K a multiple of 128) into int8 ternary
    /// {-1,0,+1,+2} in true logical element order, plus the row's per-128-element group scales
    /// into <paramref name="groupScales"/> (length <c>k/128</c>). NOT the same bit layout as
    /// I2_S's <c>UnpackRowI8</c> — verified against PrismML's own reference
    /// <c>dequantize_row_q2_0</c> (<c>PrismML-Eng/llama.cpp</c>, <c>ggml-quants.c</c>): byte at
    /// index <c>b</c> holds the 4 CONSECUTIVE elements {4b,4b+1,4b+2,4b+3} at ASCENDING bit
    /// offsets {0,2,4,6}. An earlier version of this function wrongly assumed I2_S's strided
    /// {gp,+32,+64,+96}/descending-bits layout (issue #269 follow-up, 2026-08-05) — see
    /// <see cref="Dequantize.DequantizePQ2_0"/> for the full root-cause writeup. Still reuses
    /// I2_S's AVX2 per-bit-field extraction helpers (each gives 32 contiguous byte-order-aligned
    /// values for one bit position across the group) but interleaves their four outputs back into
    /// true consecutive element order afterward, since the fields no longer land on contiguous
    /// output slices the way I2_S's own strided convention does.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackPQ2_0RowI8(byte* rowPtr, sbyte* dest, float* groupScales, int k)
    {
        int groups = k / PQ2_0GroupSize;

        if (Avx2.IsSupported)
        {
            sbyte* field0 = stackalloc sbyte[32]; // element 4b   (bit offset 0)
            sbyte* field2 = stackalloc sbyte[32]; // element 4b+1 (bit offset 2)
            sbyte* field4 = stackalloc sbyte[32]; // element 4b+2 (bit offset 4)
            sbyte* field6 = stackalloc sbyte[32]; // element 4b+3 (bit offset 6)

            for (int g = 0; g < groups; g++)
            {
                byte* groupBase = rowPtr + g * PQ2_0GroupBytes;
                groupScales[g] = (float)Unsafe.ReadUnaligned<Half>(groupBase);
                byte* bp = groupBase + 2;
                sbyte* outp = dest + g * PQ2_0GroupSize;

                Vector256<byte> packed = Unsafe.ReadUnaligned<Vector256<byte>>(bp);

                // Zero-extend all 32 packed bytes to int16 lanes (two 128-bit halves → two
                // full 256-bit int16 vectors of 16 lanes each).
                Vector256<short> w0 = Avx2.ConvertToVector256Int16(packed.GetLower());
                Vector256<short> w1 = Avx2.ConvertToVector256Int16(packed.GetUpper());

                UnpackI2SField0(w0, w1, field0);
                UnpackI2SField2(w0, w1, field2);
                UnpackI2SField4(w0, w1, field4);
                UnpackI2SField6(w0, w1, field6);

                for (int b = 0; b < 32; b++)
                {
                    int outIdx = 4 * b;
                    outp[outIdx] = field0[b];
                    outp[outIdx + 1] = field2[b];
                    outp[outIdx + 2] = field4[b];
                    outp[outIdx + 3] = field6[b];
                }
            }
            return;
        }

        for (int g = 0; g < groups; g++)
        {
            byte* groupBase = rowPtr + g * PQ2_0GroupBytes;
            groupScales[g] = (float)Unsafe.ReadUnaligned<Half>(groupBase);
            byte* bp = groupBase + 2;
            int outBase = g * PQ2_0GroupSize;
            for (int b = 0; b < 32; b++)
            {
                byte packed = bp[b];
                int outIdx = outBase + 4 * b;
                dest[outIdx] = (sbyte)((packed & 0x3) - 1);
                dest[outIdx + 1] = (sbyte)(((packed >> 2) & 0x3) - 1);
                dest[outIdx + 2] = (sbyte)(((packed >> 4) & 0x3) - 1);
                dest[outIdx + 3] = (sbyte)(((packed >> 6) & 0x3) - 1);
            }
        }
    }
}
