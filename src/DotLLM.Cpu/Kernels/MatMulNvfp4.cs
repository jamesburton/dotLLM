using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// NVFP4 matmul kernels (issue #1038): NVFP4 weights dotted against <b>Q8_0</b>-quantized activations,
/// as llama.cpp's CPU backend does (<c>type_traits_cpu[NVFP4].vec_dot_type = Q8_0</c>,
/// <c>ggml_vec_dot_nvfp4_q8_0</c>).
/// </summary>
/// <remarks>
/// <para><b>Why Q8_0 activations, unlike MXFP4's f32 dot (#275).</b> MXFP4 moved to f32 activations so
/// the CPU would match the GPU backends' dequantize-then-GEMM precision. NVFP4 has no GPU kernel yet
/// (Vulkan and CUDA refuse it), so the reference to match is llama.cpp's CPU path, which this follows
/// exactly. One 64-element NVFP4 block lines up with two Q8_0 blocks: sub-block <c>s</c> pairs with Q8_0
/// block <c>2·ib + s/2</c> at offset <c>(s % 2)·16</c>.</para>
/// <para><b>Integer exactness.</b> Each sub-block reduces to an exact int32 sum of
/// <c>kvalue · q8</c> products (|kvalue| ≤ 12, |q8| ≤ 127, so a PMADDUBSW pair is at most 3048 and cannot
/// saturate int16); only the per-sub-block float scaling and the accumulation order differ from the
/// scalar transcription.</para>
/// <para>The per-tensor second-level scale (<c>X.scale</c>) is applied by the caller after the GEMM, as
/// llama.cpp's <c>build_lora_mm</c> does.</para>
/// </remarks>
public static unsafe partial class MatMul
{
    /// <summary>Q8_0 blocks per NVFP4 block (64 / 32).</summary>
    private const int Nvfp4Q8Blocks = QuantFormat.Nvfp4GroupSize / QuantFormat.LegacyGroupSize;

    /// <summary>
    /// Scalar NVFP4 × Q8_0 dot product — a literal transcription of llama.cpp's
    /// <c>ggml_vec_dot_nvfp4_q8_0_generic</c> (<c>ggml-cpu/quants.c</c>), written from that function
    /// rather than from <see cref="Dequantize.DequantizeNvfp4"/> so the two stay independent oracles.
    /// </summary>
    /// <param name="w">One NVFP4 row: <paramref name="blockCount"/> blocks of 36 bytes.</param>
    /// <param name="q8">The Q8_0-quantized activation: <c>2 · blockCount</c> blocks of 34 bytes.</param>
    /// <param name="blockCount">NVFP4 blocks in the row (<c>K / 64</c>).</param>
    /// <returns>The dot product (before the per-tensor scale).</returns>
    [SkipLocalsInit]
    internal static float VecDotNvfp4Q8_0Scalar(byte* w, byte* q8, int blockCount)
    {
        ReadOnlySpan<sbyte> kvalues = Dequantize.Mxfp4Values;
        float sumf = 0;

        for (int ib = 0; ib < blockCount; ib++)
        {
            byte* x = w + ib * Dequantize.Nvfp4BlockBytes;
            byte* xqs = x + Dequantize.Nvfp4QsOffset;
            for (int sIdx = 0; sIdx < 4; sIdx++)
            {
                float d = Dequantize.Ue4m3ToFloatHalf(x[sIdx]);
                int q8Block = sIdx / 2;
                int q8Off = (sIdx % 2) * Dequantize.Nvfp4SubBlockSize;
                byte* y = q8 + (2 * ib + q8Block) * Q8_0BlockBytes;
                float dy = HalfToSingle((byte*)y);
                sbyte* yqs = (sbyte*)(y + 2);

                int sumiLo = 0, sumiHi = 0;
                for (int j = 0; j < Dequantize.Nvfp4SubBlockSize / 2; j++)
                {
                    byte qv = xqs[sIdx * (Dequantize.Nvfp4SubBlockSize / 2) + j];
                    sumiLo += yqs[q8Off + j] * kvalues[qv & 0xF];
                    sumiHi += yqs[q8Off + j + Dequantize.Nvfp4SubBlockSize / 2] * kvalues[qv >> 4];
                }

                sumf += dy * d * (sumiLo + sumiHi);
            }
        }

        return sumf;
    }

    /// <summary>
    /// AVX2 NVFP4 × Q8_0 dot product. Mirrors llama.cpp's x86 <c>ggml_vec_dot_nvfp4_q8_0</c>: each
    /// 16-byte half of <c>qs</c> is unpacked to signed kvalues with <c>pshufb</c>, re-ordered with
    /// <c>unpack{lo,hi}_epi64</c> so the lanes follow element order, multiplied against the two Q8_0
    /// blocks with the sign trick (<c>maddubs(|w|, sign(y, w))</c>), and each 4-lane group is scaled by
    /// its sub-block's <c>ue4m3 · d_q8</c>.
    /// </summary>
    /// <param name="w">One NVFP4 row.</param>
    /// <param name="q8">The Q8_0-quantized activation.</param>
    /// <param name="blockCount">NVFP4 blocks in the row.</param>
    /// <returns>The dot product (before the per-tensor scale).</returns>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotNvfp4Q8_0Avx2(byte* w, byte* q8, int blockCount)
    {
        Vector128<sbyte> table = Vector128.Create((sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        Vector128<byte> m4 = Vector128.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);
        Vector256<float> acc = Vector256<float>.Zero;

        for (int ib = 0; ib < blockCount; ib++)
        {
            byte* x = w + ib * Dequantize.Nvfp4BlockBytes;
            byte* y0 = q8 + 2 * ib * Q8_0BlockBytes;
            byte* y1 = y0 + Q8_0BlockBytes;

            Vector256<sbyte> w01 = UnpackNvfp4Half(Sse2.LoadVector128(x + Dequantize.Nvfp4QsOffset), table, m4);
            Vector256<sbyte> w23 = UnpackNvfp4Half(Sse2.LoadVector128(x + Dequantize.Nvfp4QsOffset + 16), table, m4);
            Vector256<sbyte> q01 = Avx.LoadVector256((sbyte*)(y0 + 2));
            Vector256<sbyte> q23 = Avx.LoadVector256((sbyte*)(y1 + 2));

            Vector256<int> p01 = Avx2.MultiplyAddAdjacent(
                Avx2.MultiplyAddAdjacent(Avx2.Abs(w01), Avx2.Sign(q01, w01)), ones);
            Vector256<int> p23 = Avx2.MultiplyAddAdjacent(
                Avx2.MultiplyAddAdjacent(Avx2.Abs(w23), Avx2.Sign(q23, w23)), ones);

            float dy0 = HalfToSingle((byte*)y0);
            float dy1 = HalfToSingle((byte*)y1);
            Vector256<float> s01 = Vector256.Create(
                Vector128.Create(Dequantize.Ue4m3ToFloatHalf(x[0]) * dy0),
                Vector128.Create(Dequantize.Ue4m3ToFloatHalf(x[1]) * dy0));
            Vector256<float> s23 = Vector256.Create(
                Vector128.Create(Dequantize.Ue4m3ToFloatHalf(x[2]) * dy1),
                Vector128.Create(Dequantize.Ue4m3ToFloatHalf(x[3]) * dy1));

            acc = Fma.MultiplyAdd(s01, Avx.ConvertToVector256Single(p01), acc);
            acc = Fma.MultiplyAdd(s23, Avx.ConvertToVector256Single(p23), acc);
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// Unpacks 16 <c>qs</c> bytes (two sub-blocks) to 32 signed kvalues in element order: the low half
    /// is sub-block <c>2h</c> (low nibbles of bytes 0..7, then their high nibbles), the high half is
    /// sub-block <c>2h + 1</c> (bytes 8..15).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<sbyte> UnpackNvfp4Half(Vector128<byte> bits, Vector128<sbyte> table, Vector128<byte> m4)
    {
        Vector128<long> lo = Ssse3.Shuffle(table, (bits & m4).AsSByte()).AsInt64();
        Vector128<long> hi = Ssse3.Shuffle(table, (Sse2.ShiftRightLogical(bits.AsUInt16(), 4).AsByte() & m4).AsSByte()).AsInt64();
        return Vector256.Create(Sse2.UnpackLow(lo, hi), Sse2.UnpackHigh(lo, hi)).AsSByte();
    }

    /// <summary>Whether the AVX2 NVFP4 dot can run on this CPU.</summary>
    private static bool Nvfp4Avx2 => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>NVFP4 × Q8_0 row dot, dispatched to AVX2 or the scalar transcription.</summary>
    /// <param name="w">One NVFP4 row.</param>
    /// <param name="q8">The Q8_0-quantized activation.</param>
    /// <param name="blockCount">NVFP4 blocks in the row.</param>
    /// <returns>The dot product (before the per-tensor scale).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotNvfp4Q8_0(byte* w, byte* q8, int blockCount)
        => Nvfp4Avx2 ? VecDotNvfp4Q8_0Avx2(w, q8, blockCount) : VecDotNvfp4Q8_0Scalar(w, q8, blockCount);

    /// <summary>
    /// NVFP4 GEMM: <c>C[n, m] = B[n, k] × W[m, k]ᵀ</c>, with <c>W</c> NVFP4 and <c>B</c> f32. Every input
    /// row is quantized to Q8_0 once, then rows of <c>W</c> are partitioned across
    /// <paramref name="pool"/> (bit-identical to the serial path) and each thread walks its rows in
    /// L2-sized tiles with the columns inner, so a weight tile is read once for all <paramref name="n"/>
    /// tokens. The per-tensor <c>X.scale</c> is NOT applied.
    /// </summary>
    /// <param name="weights">NVFP4 weights, <paramref name="m"/> rows of <c>k / 64</c> blocks.</param>
    /// <param name="b">f32 input, <paramref name="n"/> rows of <paramref name="k"/>.</param>
    /// <param name="c">f32 output, <paramref name="n"/> rows of <paramref name="m"/>.</param>
    /// <param name="m">Output rows (weight rows).</param>
    /// <param name="k">Contraction length; a multiple of 64.</param>
    /// <param name="n">Input rows (tokens).</param>
    /// <param name="pool">Thread pool; <see langword="null"/> runs serially.</param>
    /// <exception cref="ArgumentException"><paramref name="k"/> is not a multiple of 64.</exception>
    [SkipLocalsInit]
    public static void GemmNvfp4(byte* weights, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        if (k % QuantFormat.Nvfp4GroupSize != 0)
            throw new ArgumentException($"NVFP4 k must be a multiple of {QuantFormat.Nvfp4GroupSize}, got {k}", nameof(k));
        if (m <= 0 || n <= 0) return;

        int blockCount = k / QuantFormat.Nvfp4GroupSize;
        int q8RowBytes = blockCount * Nvfp4Q8Blocks * Q8_0BlockBytes;
        int wRowBytes = blockCount * Dequantize.Nvfp4BlockBytes;

        byte[] rented = ArrayPool<byte>.Shared.Rent(checked(n * q8RowBytes));
        try
        {
            fixed (byte* q8 = rented)
            {
                for (int t = 0; t < n; t++)
                    QuantizeF32ToQ8_0(b + (long)t * k, q8 + (long)t * q8RowBytes, k);

                var ctx = new Nvfp4GemmCtx
                {
                    Weights = weights, InputQ8 = q8, C = c, M = m, N = n, BlockCount = blockCount,
                    TileM = ComputeTileM(wRowBytes), WRowBytes = wRowBytes, Q8RowBytes = q8RowBytes,
                };
                if (pool is null || pool.ThreadCount <= 1 || m < ParallelMinRows)
                    Nvfp4GemmRows(ref ctx, 0, m);
                else
                    pool.Dispatch((nint)(&ctx), &Nvfp4GemmWorker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private struct Nvfp4GemmCtx
    {
        public byte* Weights;
        public byte* InputQ8;
        public float* C;
        public int M, N, BlockCount, TileM, WRowBytes, Q8RowBytes;
    }

    private static void Nvfp4GemmWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<Nvfp4GemmCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        Nvfp4GemmRows(ref ctx, start, count);
    }

    /// <summary>Rows <c>[rowStart, rowStart + rowCount)</c> of <see cref="GemmNvfp4"/>, tiled.</summary>
    private static void Nvfp4GemmRows(ref Nvfp4GemmCtx ctx, int rowStart, int rowCount)
    {
        bool avx2 = Nvfp4Avx2;
        int end = rowStart + rowCount;
        for (int tile = rowStart; tile < end; tile += ctx.TileM)
        {
            int tileEnd = Math.Min(tile + ctx.TileM, end);
            for (int t = 0; t < ctx.N; t++)
            {
                byte* xq = ctx.InputQ8 + (long)t * ctx.Q8RowBytes;
                float* cRow = ctx.C + (long)t * ctx.M;
                for (int row = tile; row < tileEnd; row++)
                {
                    byte* wRow = ctx.Weights + (long)row * ctx.WRowBytes;
                    cRow[row] = avx2
                        ? VecDotNvfp4Q8_0Avx2(wRow, xq, ctx.BlockCount)
                        : VecDotNvfp4Q8_0Scalar(wRow, xq, ctx.BlockCount);
                }
            }
        }
    }
}
