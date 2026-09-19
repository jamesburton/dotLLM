using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// PrismML Bonsai PTQ1_0 ternary matrix multiplication kernels (<c>GGML_TYPE_PTQ1_0</c> = 143,
/// issue #436). Weights are ternary {-1, 0, +1} packed densely in base 3 — five trits to a byte —
/// grouped in 128-element groups of 28 bytes: <c>qs[24] + qh[2] + scale(Half)</c>, 1.75 bpw.
///
/// <para><b>Relationship to <c>MatMul.PQ2S.cs</c>.</b> The two formats differ only in how a group
/// of 128 ternary weights is packed and where its scale lives; everything downstream of the unpack
/// is identical. So this file supplies only the two unpack routines and the dispatch/partition
/// scaffolding, and the W2A8 (int8-activation) inner product is the <i>same</i>
/// <see cref="VecDotPQ2_0Q8"/> used by PQ2_0 — it takes an already-unpacked int8 ternary row plus
/// the row's per-128 group scales, so it is packing-agnostic. Writing a second VNNI dot would have
/// duplicated the sign-trick and scale-folding logic for no gain.</para>
///
/// <para><b>Two traps the packing sets, both silent when sprung.</b> First, the trit extraction
/// multiplies into an 8-bit register and <i>relies on the wrap</i>:
/// <c>uint8_t q = byte * pow3[n]</c> then <c>(q * 3) &gt;&gt; 8</c>. Widening that multiply yields a
/// correct trit 0 and garbage for the other four. Second, the scale is the <b>last</b> field of the
/// block, not the first as in PQ2_0 — templating the offsets from PQ2_0 reads every group's scale
/// out of the next group's payload.</para>
///
/// <para>Tier choice mirrors PQ2_0's, which was settled empirically in issue #204: GEMV/decode
/// dispatches to W2A8 on AVX2 hardware, GEMM/prefill stays on the scalar float tier
/// unconditionally (W2A8 GEMM showed no reliable win there and a real regression on the ffn_gate
/// shape). PTQ1_0 has not been independently benchmarked — perf validation is a separate
/// coordinated pass — so it inherits the shape of the decision rather than a measurement of its
/// own.</para>
/// </summary>
public static unsafe partial class MatMul
{
    private const int PTQ1_0GroupSize = QuantFormat.TernaryGroupSize;
    private const int PTQ1_0GroupBytes = QuantFormat.PTQ1_0BlockBytes;
    private const int PTQ1_0QsBytes = 24;
    private const int PTQ1_0QhBytes = 2;

    /// <summary>
    /// <c>qs</c> chunk widths, in the fork's own order (<c>ptq1_0_stages</c>). The 32-wide stage
    /// never fires because <c>sizeof(qs)</c> is 24, so the effective layout is a 16-byte chunk
    /// then an 8-byte chunk; the loop is kept general to match the reference exactly.
    /// </summary>
    private static ReadOnlySpan<int> PTQ1_0Stages => [32, 16, 8];

    /// <summary>Powers of three that shift a packed trit up into the top of a byte.</summary>
    private static ReadOnlySpan<byte> PTQ1_0Pow3 => [1, 3, 9, 27, 81, 243];

    /// <summary>
    /// Extracts base-3 trit <paramref name="n"/> from a packed byte, mapped to {-1, 0, +1}.
    /// The <c>(byte)</c> cast is the wrap the format depends on — see the class summary.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int PTQ1_0Trit(byte packed, int n)
    {
        byte q = (byte)(packed * PTQ1_0Pow3[n]);
        return ((q * 3) >> 8) - 1;
    }

    private struct GemvPTQ1Ctx
    {
        public byte* Weights;
        public float* X;
        public float* Result;
        public int M;
        public int K;
    }

    private struct GemmPTQ1Ctx
    {
        public byte* Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
    }

    private struct GemvPTQ1Q8Ctx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* Result;
        public int M;
        public int K;
    }

    /// <summary>True when the W2A8 (int8-activation) SIMD path is available.</summary>
    private static bool PTQ1_0UseW2A8 => Avx2.IsSupported;

    /// <summary>
    /// PTQ1_0 ternary GEMV: <c>result[r] = dot(perGroupScaled(A[r,:]), x)</c>.
    /// A is [M,K] packed PTQ1_0 (row-major, K a multiple of 128, row stride
    /// <c>(K/128)·28</c> bytes); x is f32 [K]; result is f32 [M]. Dispatches to the W2A8 tier on
    /// AVX2 hardware, else the float tier. Output rows are partitioned across
    /// <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvPTQ1_0(byte* weights, float* x, float* result, int m, int k,
                                  ComputeThreadPool? threadPool)
    {
        if (k % PTQ1_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PTQ1_0GroupSize}, got {k}", nameof(k));

        if (PTQ1_0UseW2A8)
        {
            GemvPTQ1_0W2A8(weights, x, result, m, k, threadPool);
            return;
        }

        if (threadPool is null || m < ParallelMinRows)
        {
            GemvPTQ1_0Rows(weights, x, result, 0, m, k);
            return;
        }

        var ctx = new GemvPTQ1Ctx { Weights = weights, X = x, Result = result, M = m, K = k };
        threadPool.Dispatch((nint)(&ctx), &GemvPTQ1_0Worker);
    }

    /// <summary>
    /// Test/benchmark entry point that always takes the scalar float reference tier, bypassing the
    /// W2A8 dispatch. This is the tier validated directly against
    /// <see cref="Dequantize.DequantizePTQ1_0"/>, which is itself gated bit-exact against the
    /// fork's reference — so it is the oracle the SIMD tier is compared to.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemvPTQ1_0Scalar(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % PTQ1_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PTQ1_0GroupSize}, got {k}", nameof(k));
        GemvPTQ1_0Rows(weights, x, result, 0, m, k);
    }

    /// <summary>GEMM analog of <see cref="GemvPTQ1_0Scalar"/> — always the scalar reference tier.</summary>
    [SkipLocalsInit]
    internal static void GemmPTQ1_0Scalar(byte* weights, float* b, float* c, int m, int k, int n)
    {
        if (n == 1)
        {
            GemvPTQ1_0Scalar(weights, b, c, m, k);
            return;
        }

        if (k % PTQ1_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PTQ1_0GroupSize}, got {k}", nameof(k));
        GemmPTQ1_0Rows(weights, b, c, m, 0, m, k, n);
    }

    /// <summary>
    /// PTQ1_0 ternary GEMM: <c>C[N,M] = B[N,K] × perGroupScaled(A[M,K])^T</c>. Each weight row is
    /// unpacked once and dotted against all N input rows. As with
    /// <see cref="GemmPQ2_0"/> this stays on the scalar float tier — see the class summary.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmPTQ1_0(byte* weights, float* b, float* c, int m, int k, int n,
                                  ComputeThreadPool? threadPool)
    {
        if (n == 1)
        {
            GemvPTQ1_0(weights, b, c, m, k, threadPool);
            return;
        }

        if (k % PTQ1_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {PTQ1_0GroupSize}, got {k}", nameof(k));

        if (threadPool is null || m < ParallelMinRows)
        {
            GemmPTQ1_0Rows(weights, b, c, m, 0, m, k, n);
            return;
        }

        var ctx = new GemmPTQ1Ctx { Weights = weights, B = b, C = c, M = m, K = k, N = n };
        threadPool.Dispatch((nint)(&ctx), &GemmPTQ1_0Worker);
    }

    // ─────────────────────────── Float tier ───────────────────────────

    [SkipLocalsInit]
    private static void GemvPTQ1_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvPTQ1Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvPTQ1_0Rows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvPTQ1_0Rows(byte* weights, float* x, float* result,
                                       int startRow, int rowCount, int k)
    {
        int rowBytes = (k / PTQ1_0GroupSize) * PTQ1_0GroupBytes;
        var xSpan = new ReadOnlySpan<float>(x, k);

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackPTQ1_0Row(weights + (long)r * rowBytes, rowSpan, k);
                result[r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    [SkipLocalsInit]
    private static void GemmPTQ1_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmPTQ1Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmPTQ1_0Rows(ctx.Weights, ctx.B, ctx.C, ctx.M, start, count, ctx.K, ctx.N);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmPTQ1_0Rows(byte* weights, float* b, float* c, int m,
                                       int startRow, int rowCount, int k, int n)
    {
        int rowBytes = (k / PTQ1_0GroupSize) * PTQ1_0GroupBytes;

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackPTQ1_0Row(weights + (long)r * rowBytes, rowSpan, k);
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
    /// Unpacks one PTQ1_0-packed weight row (K weights, K a multiple of 128) into scaled floats
    /// {-scale, 0, +scale}. Element order within a group is trit-major over a byte chunk: for
    /// chunk width <c>c</c> starting at <c>qs[j]</c>, values come out as
    /// <c>(n, m) → qs[j+m]</c> trit <c>n</c>, and the <c>qh</c> tail supplies the last 8 as
    /// <c>(n, h) → qh[h]</c> trit <c>n</c> for <c>n</c> in 0..3. Kept structurally identical to
    /// <see cref="Dequantize.DequantizePTQ1_0"/>, which is the bit-exact-gated transcription.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackPTQ1_0Row(byte* rowPtr, Span<float> dest, int k)
    {
        int groups = k / PTQ1_0GroupSize;
        for (int g = 0; g < groups; g++)
        {
            byte* qs = rowPtr + g * PTQ1_0GroupBytes;
            byte* qh = qs + PTQ1_0QsBytes;
            float scale = (float)Unsafe.ReadUnaligned<Half>(qh + PTQ1_0QhBytes);
            int outIdx = g * PTQ1_0GroupSize;

            int j = 0;
            foreach (int c in PTQ1_0Stages)
            {
                while (j + c <= PTQ1_0QsBytes)
                {
                    for (int nn = 0; nn < 5; nn++)
                        for (int mm = 0; mm < c; mm++)
                            dest[outIdx++] = PTQ1_0Trit(qs[j + mm], nn) * scale;

                    j += c;
                }
            }

            for (int nn = 0; nn < 4; nn++)
                for (int h = 0; h < PTQ1_0QhBytes; h++)
                    dest[outIdx++] = PTQ1_0Trit(qh[h], nn) * scale;
        }
    }

    // ─────────────────────────── W2A8 (int8-activation) tier ───────────────────────────

    /// <summary>
    /// W2A8 GEMV: quantizes the activation to Q8_0 once, then partitions weight rows over the pool.
    /// </summary>
    [SkipLocalsInit]
    private static void GemvPTQ1_0W2A8(byte* weights, float* x, float* result, int m, int k,
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
                    GemvPTQ1_0W2A8Rows(weights, xQ8, result, 0, m, k);
                    return;
                }

                var ctx = new GemvPTQ1Q8Ctx { Weights = weights, XQ8 = xQ8, Result = result, M = m, K = k };
                threadPool.Dispatch((nint)(&ctx), &GemvPTQ1_0W2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(xQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemvPTQ1_0W2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvPTQ1Q8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvPTQ1_0W2A8Rows(ctx.Weights, ctx.XQ8, ctx.Result, start, count, ctx.K);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvPTQ1_0W2A8Rows(byte* weights, byte* xQ8, float* result,
                                           int startRow, int rowCount, int k)
    {
        int rowBytes = (k / PTQ1_0GroupSize) * PTQ1_0GroupBytes;
        int blockCount = k / Q8_0GroupSize;
        int groupCount = k / PTQ1_0GroupSize;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        float[] groupScaleBuf = ArrayPool<float>.Shared.Rent(groupCount);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            fixed (float* groupScales = groupScaleBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackPTQ1_0RowI8(weights + (long)r * rowBytes, wI8, groupScales, k);
                    // Shared with PQ2_0 on purpose: the dot consumes unpacked int8 ternary plus
                    // per-128 group scales, which is packing-independent. PTQ1_0's group size is
                    // also 128, so the 4-Q8_0-blocks-per-group folding is identical.
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
    /// Unpacks one PTQ1_0-packed weight row into int8 ternary {-1, 0, +1} in true logical element
    /// order, plus the row's per-128-element group scales into <paramref name="groupScales"/>
    /// (length <c>k/128</c>). Same traversal as <see cref="UnpackPTQ1_0Row"/>, without the scale
    /// multiply — the scale is applied once per group inside the dot instead.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackPTQ1_0RowI8(byte* rowPtr, sbyte* dest, float* groupScales, int k)
    {
        int groups = k / PTQ1_0GroupSize;
        for (int g = 0; g < groups; g++)
        {
            byte* qs = rowPtr + g * PTQ1_0GroupBytes;
            byte* qh = qs + PTQ1_0QsBytes;
            groupScales[g] = (float)Unsafe.ReadUnaligned<Half>(qh + PTQ1_0QhBytes);
            sbyte* outp = dest + g * PTQ1_0GroupSize;
            int outIdx = 0;

            int j = 0;
            foreach (int c in PTQ1_0Stages)
            {
                while (j + c <= PTQ1_0QsBytes)
                {
                    for (int nn = 0; nn < 5; nn++)
                        for (int mm = 0; mm < c; mm++)
                            outp[outIdx++] = (sbyte)PTQ1_0Trit(qs[j + mm], nn);

                    j += c;
                }
            }

            for (int nn = 0; nn < 4; nn++)
                for (int h = 0; h < PTQ1_0QhBytes; h++)
                    outp[outIdx++] = (sbyte)PTQ1_0Trit(qh[h], nn);
        }
    }
}
