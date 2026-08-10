using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// MXFP4 matrix-vector / matrix-matrix kernels. MXFP4 blocks pack 32 elements
/// as one E8M0 power-of-two scale byte plus 16 nibble bytes indexing the
/// doubled-E2M1 value table (see <see cref="Dequantize.Mxfp4Values"/>).
/// </summary>
/// <remarks>
/// <para><b>Why this dots against f32 activations instead of Q8_0-quantizing them first
/// (issue #275).</b> llama.cpp's own generic CPU kernel
/// (<c>ggml_vec_dot_mxfp4_q8_0_generic</c>) DOES quantize the activation to Q8_0 first, and
/// this file did too until #275 — the #256 cross-backend gate found CPU diverging from BOTH
/// CUDA and Vulkan (independent implementations that agree with EACH OTHER to under 5e-4) by
/// up to 1.2e-2 in logit cosine on MXFP4 specifically, a clean 4x outlier against every other
/// quantized type's 1e-5–3e-3 continuum. Root-caused via a CPU-only, GPU-independent
/// diagnostic (<c>Mxfp4CausalConsistencyDiagnosticTests</c>): the MXFP4 row-dot formula,
/// nibble order, and E8M0 scale convention are bit-exact against llama.cpp's reference (and
/// self-consistent across batch sizes) — no transcription bug like #254's. The actual cause is
/// a backend PRECISION asymmetry: CUDA/Vulkan's MXFP4 support (added in #258) has no dedicated
/// int8-quantized vec-dot kernel — both dequantize the MXFP4 block to F16 once
/// (<c>dequant_mxfp4_f16</c>/<c>_f32</c> in <c>native/kernels/dequant_bf16_mxfp4.cu</c>, see
/// also <c>CudaKernels.cs</c>) and run the ordinary full-precision GEMM/GEMV, with NO
/// activation-quantization step. Every other type in the gate has matching lossiness on both
/// sides (either both sides use a Q8-quantized vec-dot, or both fall back to
/// dequant+dense-GEMM), so the shared noise cancels in a cross-backend diff; MXFP4 was the one
/// type where CPU carried an EXTRA ~0.4%-per-block noise source the GPU side didn't, compounding
/// over ~16 decoder layers into the observed 1.2e-2 logit divergence. This kernel now dots
/// directly against the f32 activation — still nibble-unpacking the weight via the same fast
/// table lookup as before, just skipping the Q8_0 round-trip — matching GPU's precision profile
/// instead of adding noise GPU doesn't have. (The GPU side ideally gets a proper
/// int8-quantized MXFP4 vec-dot kernel of its own eventually, matching llama.cpp CUDA
/// convention and recovering the throughput this trade gives up — tracked separately; that is
/// native CUDA/Vulkan kernel work, out of scope here.)</para>
/// </remarks>
public static unsafe partial class MatMul
{
    /// <summary>MXFP4 block size in bytes: 1 (E8M0 scale) + 16 (nibbles).</summary>
    private const int Mxfp4BlockBytes = 17;

    /// <summary>Elements per MXFP4 block.</summary>
    private const int Mxfp4GroupSize = 32;

    /// <summary>
    /// MXFP4 GEMV: A is MXFP4 [M,K], x is f32 [K]. Dots each row directly against the f32
    /// input vector (see file remarks for why no Q8_0 activation quantization).
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
        ComputeRowsMxfp4(weights, x, result, m, blockCount);
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
        var ctx = new ComputeRowsMxfp4Ctx
        {
            Weights = weights, X = x, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsMxfp4Worker);
    }

    /// <summary>
    /// MXFP4 GEMM: C[N,M] = B[N,K] × A[M,K]^T where A is MXFP4 weights and B
    /// is f32 activations. Each of the N columns dots directly against its own f32 row.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmMxfp4(byte* weights, float* b, float* c, int m, int k, int n,
                                 ComputeThreadPool? pool = null)
    {
        if (k % Mxfp4GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Mxfp4GroupSize}, got {k}", nameof(k));

        int blockCount = k / Mxfp4GroupSize;

        for (int t = 0; t < n; t++)
            ComputeRowsMxfp4Parallel(weights, b + (long)t * k, c + (long)t * m, m, blockCount, pool);
    }

    private struct ComputeRowsMxfp4Ctx
    {
        public byte* Weights;
        public float* X;
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
        ComputeRowsMxfp4(ctx.Weights + (long)start * rowBytes, ctx.X,
            ctx.Result + start, count, ctx.BlockCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeRowsMxfp4Parallel(byte* weights, float* x, float* result,
        int m, int blockCount, ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            ComputeRowsMxfp4(weights, x, result, m, blockCount);
            return;
        }

        var ctx = new ComputeRowsMxfp4Ctx
        {
            Weights = weights, X = x, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsMxfp4Worker);
    }

    /// <summary>
    /// Computes <paramref name="m"/> MXFP4×f32 row dot products.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsMxfp4(byte* weights, float* x, float* result, int m, int blockCount)
    {
        // rowBytes is deliberately `long`: that alone makes every `row * rowBytes` below a
        // 64-bit multiply (int `row` is promoted), so the issue-#429 wrap class cannot occur
        // here. Do not "fix" the call sites by adding a (long) cast to `row` — and do not
        // narrow this local to int.
        long rowBytes = (long)blockCount * Mxfp4BlockBytes;

        if (Avx2.IsSupported && Ssse3.IsSupported)
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotMxfp4F32Avx2(weights + row * rowBytes, x, blockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotMxfp4F32Scalar(weights + row * rowBytes, x, blockCount);
        }
    }

    /// <summary>
    /// Scalar MXFP4 × f32 dot product — reference implementation. Numerically equal to
    /// dequantizing the row to f32 (<see cref="Dequantize.DequantizeMxfp4Scalar"/>) and
    /// computing a plain dot product against <paramref name="x"/>, just without materialising
    /// the dequantized row: <c>sum over blocks of e8m0_half(e) * Σ kvalues[nib] * x[i]</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotMxfp4F32Scalar(byte* w, float* x, int blockCount)
    {
        float sumf = 0;
        ReadOnlySpan<sbyte> kvalues = Dequantize.Mxfp4Values;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wBlock = w + block * Mxfp4BlockBytes;
            float* xBlock = x + block * Mxfp4GroupSize;

            float d = Dequantize.E8M0ToFloatHalf(wBlock[0]);
            byte* qs = wBlock + 1;

            float blockSum = 0;
            for (int j = 0; j < 16; j++)
            {
                blockSum += kvalues[qs[j] & 0x0F] * xBlock[j];
                blockSum += kvalues[qs[j] >> 4] * xBlock[j + 16];
            }

            sumf += d * blockSum;
        }

        return sumf;
    }

    /// <summary>
    /// AVX2 MXFP4 × f32 dot product. Expands nibbles to sbyte weights via a <c>vpshufb</c>
    /// lookup of the 16-entry kvalue table (identical unpack to the previous Q8_0-based
    /// kernel), widens each 16-lane group to int32 then float, scales by the per-block E8M0
    /// half-scale, and accumulates against the f32 activation directly — no int8 activation
    /// quantization step.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotMxfp4F32Avx2(byte* w, float* x, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector128<byte> nibbleMask = Vector128.Create((byte)0x0F);
        Vector128<sbyte> kvalueTable = Vector128.Create(
            (sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

        for (int block = 0; block < blockCount; block++)
        {
            byte* wBlock = w + block * Mxfp4BlockBytes;
            float* xBlock = x + block * Mxfp4GroupSize;

            float d = Dequantize.E8M0ToFloatHalf(wBlock[0]);
            Vector256<float> vd = Vector256.Create(d);

            // Unpack 16 nibble bytes → 32 sbyte weight values via table lookup.
            Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(wBlock + 1);
            Vector128<byte> lo = Sse2.And(qsRaw, nibbleMask);
            Vector128<byte> hi = Sse2.And(
                Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(), nibbleMask);
            Vector128<sbyte> wLo = Ssse3.Shuffle(kvalueTable, lo.AsSByte());  // elements 0..15
            Vector128<sbyte> wHi = Ssse3.Shuffle(kvalueTable, hi.AsSByte());  // elements 16..31

            acc = Mxfp4FmaGroup(wLo, xBlock, vd, acc);
            acc = Mxfp4FmaGroup(wHi, xBlock + 16, vd, acc);
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// Widens 16 sbyte kvalue-table lookups (2×8-lane groups) to float, scales by the
    /// per-block E8M0 half-scale, and multiply-accumulates against 16 consecutive f32
    /// activation elements starting at <paramref name="xPtr"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Mxfp4FmaGroup(
        Vector128<sbyte> w16, float* xPtr, Vector256<float> vd, Vector256<float> acc)
    {
        (Vector128<short> shortLo, Vector128<short> shortHi) = Vector128.Widen(w16);
        (Vector128<int> intA, Vector128<int> intB) = Vector128.Widen(shortLo); // elems 0-3, 4-7
        (Vector128<int> intC, Vector128<int> intD) = Vector128.Widen(shortHi); // elems 8-11, 12-15

        Vector256<float> wfLo = Avx.ConvertToVector256Single(Vector256.Create(intA, intB)) * vd; // 0..7
        Vector256<float> wfHi = Avx.ConvertToVector256Single(Vector256.Create(intC, intD)) * vd; // 8..15

        Vector256<float> xLo = Avx.LoadVector256(xPtr);
        Vector256<float> xHi = Avx.LoadVector256(xPtr + 8);

        acc += wfLo * xLo;
        acc += wfHi * xHi;
        return acc;
    }
}
