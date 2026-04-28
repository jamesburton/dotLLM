using System.Buffers;
using System.Buffers.Binary;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Lora;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// LoRA delta accumulation: <c>y += alpha × (x · B) · A</c>.
/// </summary>
/// <remarks>
/// <para>
/// Tensor layouts (row-major F32, matching <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>):
/// </para>
/// <list type="bullet">
///   <item><c>x</c>: <c>[seqLen, inputDim]</c> — the same input that fed the base projection.</item>
///   <item><c>B</c>: <c>[rank, inputDim]</c> — LoRA down-projection weight, stored row-major
///     so <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c> matches the standard
///     "weight matrix as <c>[outputDim, inputDim]</c>" convention dotLLM uses everywhere.</item>
///   <item><c>A</c>: <c>[outputDim, rank]</c> — LoRA up-projection weight (same convention).</item>
///   <item><c>y</c>: <c>[seqLen, outputDim]</c> — base-projection output; LoRA delta added in-place.</item>
/// </list>
/// <para>
/// Implementation: two stacked GEMMs through a small <c>[seqLen, rank]</c>
/// scratch — for typical r∈[8, 64] each is a thin matmul that <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>
/// already handles efficiently. We rent the scratch from <see cref="ArrayPool{T}"/>
/// so there is no per-call native allocation.
/// </para>
/// </remarks>
public static unsafe class LoraDelta
{
    /// <summary>
    /// Accumulates <c>y += scale × (x · B) · A</c> in-place. Mathematically:
    /// <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c>; then
    /// <c>y[t, o] += scale × sum_r A[o, r] · tmp[t, r]</c>.
    /// </summary>
    /// <param name="x">Input pointer, row-major <c>[seqLen, inputDim]</c>.</param>
    /// <param name="bWeight">B (down-proj) pointer, row-major <c>[rank, inputDim]</c>.</param>
    /// <param name="aWeight">A (up-proj) pointer, row-major <c>[outputDim, rank]</c>.</param>
    /// <param name="y">Output pointer (read-modify-write), row-major <c>[seqLen, outputDim]</c>.</param>
    /// <param name="seqLen">Number of input tokens in this call.</param>
    /// <param name="inputDim">Projection input dimension.</param>
    /// <param name="outputDim">Projection output dimension.</param>
    /// <param name="rank">LoRA rank (typical 8..64).</param>
    /// <param name="scale">Scaling factor — typically <c>alpha / rank</c>.</param>
    [SkipLocalsInit]
    public static void Apply(float* x, float* bWeight, float* aWeight, float* y,
                             int seqLen, int inputDim, int outputDim, int rank, float scale)
    {
        if (seqLen <= 0 || rank <= 0) return;

        // Stage 1: tmp[t, r] = sum_i x[t, i] · B[r, i].
        // GemmF32 contracts as C[N, M] = B[N, K] × A[M, K]^T, so we pass
        //   a = bWeight   (M=rank, K=inputDim)
        //   b = x         (N=seqLen, K=inputDim)
        //   c = tmp       (N=seqLen, M=rank)
        int tmpElems = seqLen * rank;
        float[] tmpBuf = ArrayPool<float>.Shared.Rent(tmpElems);
        try
        {
            fixed (float* tmp = tmpBuf)
            {
                MatMul.GemmF32(bWeight, x, tmp, rank, inputDim, seqLen);

                // Stage 2: y[t, o] += scale × sum_r A[o, r] · tmp[t, r].
                // We reuse a small per-token scratch for the A·tmp product
                // and add it into y. For typical small rank+outputDim this
                // stays in L1; a per-token GemvF32 keeps the inner loop
                // straight and reuses dotLLM's existing F32 SIMD path.
                int deltaScratchElems = outputDim;
                float[] deltaBuf = ArrayPool<float>.Shared.Rent(deltaScratchElems);
                try
                {
                    fixed (float* delta = deltaBuf)
                    {
                        for (int t = 0; t < seqLen; t++)
                        {
                            // delta[o] = sum_r A[o, r] · tmp[t, r]
                            MatMul.GemvF32(aWeight, tmp + t * rank, delta, outputDim, rank);

                            // y[t, o] += scale * delta[o] via TensorPrimitives.
                            var deltaSpan = new ReadOnlySpan<float>(delta, outputDim);
                            var ySpan = new Span<float>(y + t * outputDim, outputDim);
                            TensorPrimitives.MultiplyAdd(deltaSpan, scale, ySpan, ySpan);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(deltaBuf);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tmpBuf);
        }
    }

    /// <summary>
    /// Quantised-weight overload (Phase 4d.1). Dequantises B and A from
    /// <paramref name="bDType"/> / <paramref name="aDType"/> into a small F32
    /// scratch and reuses the standard F32 path. Both buffers are typically
    /// the same dtype (PEFT writes lora_A and lora_B together) but we permit
    /// independent dtypes for completeness.
    /// </summary>
    /// <remarks>
    /// We dequantise the entire (small) B and A factors once per call rather
    /// than inlining dequant into the inner GEMM loop. For r=16 typical
    /// shapes the factors are kilobytes — staying in L1 — and this avoids
    /// duplicating the GEMM kernel for each dtype combination.
    /// </remarks>
    [SkipLocalsInit]
    public static void Apply(float* x, void* bWeight, void* aWeight, float* y,
                             int seqLen, int inputDim, int outputDim, int rank, float scale,
                             LoraWeightDType bDType, LoraWeightDType aDType)
    {
        if (seqLen <= 0 || rank <= 0) return;

        // Fast path: both F32 — go straight to the existing kernel without
        // copies. Common when callers haven't migrated to the dtype-aware path.
        if (bDType == LoraWeightDType.F32 && aDType == LoraWeightDType.F32)
        {
            Apply(x, (float*)bWeight, (float*)aWeight, y, seqLen, inputDim, outputDim, rank, scale);
            return;
        }

        long bElems = (long)rank * inputDim;
        long aElems = (long)outputDim * rank;
        float[] bBuf = ArrayPool<float>.Shared.Rent((int)bElems);
        float[] aBuf = ArrayPool<float>.Shared.Rent((int)aElems);
        try
        {
            DequantToF32(bWeight, bDType, bBuf, (int)bElems);
            DequantToF32(aWeight, aDType, aBuf, (int)aElems);

            fixed (float* bF32 = bBuf)
            fixed (float* aF32 = aBuf)
            {
                Apply(x, bF32, aF32, y, seqLen, inputDim, outputDim, rank, scale);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(aBuf);
            ArrayPool<float>.Shared.Return(bBuf);
        }
    }

    private static void DequantToF32(void* src, LoraWeightDType dtype, float[] dst, int count)
    {
        switch (dtype)
        {
            case LoraWeightDType.F32:
                {
                    var srcSpan = new ReadOnlySpan<float>(src, count);
                    srcSpan.CopyTo(dst.AsSpan(0, count));
                    break;
                }
            case LoraWeightDType.F16:
                {
                    var srcSpan = new ReadOnlySpan<Half>(src, count);
                    TensorPrimitives.ConvertToSingle(srcSpan, dst.AsSpan(0, count));
                    break;
                }
            case LoraWeightDType.BF16:
                {
                    byte* p = (byte*)src;
                    for (int i = 0; i < count; i++)
                    {
                        ushort raw = BinaryPrimitives.ReadUInt16LittleEndian(
                            new ReadOnlySpan<byte>(p + i * 2, 2));
                        uint asF32 = (uint)raw << 16;
                        dst[i] = BitConverter.UInt32BitsToSingle(asF32);
                    }
                    break;
                }
            default:
                throw new NotSupportedException(
                    $"LoRA weight dtype {dtype} is not supported by LoraDelta.Apply.");
        }
    }

    /// <summary>
    /// Convenience overload using <see cref="ReadOnlySpan{T}"/> / <see cref="Span{T}"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Apply(ReadOnlySpan<float> x, ReadOnlySpan<float> bWeight, ReadOnlySpan<float> aWeight,
                             Span<float> y, int seqLen, int inputDim, int outputDim, int rank, float scale)
    {
        if (x.Length < seqLen * inputDim)
            throw new ArgumentException($"x span too small: {x.Length} < {seqLen * inputDim}", nameof(x));
        if (bWeight.Length < rank * inputDim)
            throw new ArgumentException($"bWeight span too small: {bWeight.Length} < {rank * inputDim}", nameof(bWeight));
        if (aWeight.Length < outputDim * rank)
            throw new ArgumentException($"aWeight span too small: {aWeight.Length} < {outputDim * rank}", nameof(aWeight));
        if (y.Length < seqLen * outputDim)
            throw new ArgumentException($"y span too small: {y.Length} < {seqLen * outputDim}", nameof(y));

        fixed (float* xPtr = x)
        fixed (float* bPtr = bWeight)
        fixed (float* aPtr = aWeight)
        fixed (float* yPtr = y)
        {
            Apply(xPtr, bPtr, aPtr, yPtr, seqLen, inputDim, outputDim, rank, scale);
        }
    }
}
