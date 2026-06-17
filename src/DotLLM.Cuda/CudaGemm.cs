using System.Runtime.InteropServices;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// cuBLAS GEMM/GEMV wrappers for linear projections.
/// Weight matrices are FP16, stored as [outputDim, inputDim] (row-major).
/// Input/output are FP16. FP32 accumulation via cublasGemmEx.
/// Caller converts FP32→FP16 before and FP16→FP32 after as needed.
/// </summary>
public static class CudaGemm
{
    private static readonly float FloatOne = 1.0f;
    private static readonly float FloatZero = 0.0f;

    // G1 experiment toggle: DOTLLM_CUDA_GEMM_16F=1 switches the prefill GEMM compute type from
    // CUBLAS_COMPUTE_32F (FP16 inputs, FP32 accumulate — throttled to ~half rate on GeForce Ampere)
    // to CUBLAS_COMPUTE_16F (FP16 accumulate — full tensor-core rate, lower precision). Compute-16F
    // requires half-typed alpha/beta, so the scalar constants differ per path.
    // Mutable so a benchmark can interleave 32F/16F reps within one warmed process (consumer GPUs
    // drift clocks across consecutive runs, so a fresh-process A/B is confounded by thermal state).
    internal static bool Use16FCompute =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_GEMM_16F") == "1";

    /// <summary>
    /// Linear projection: Y_f16[m, n] = X_f16[m, k] × W_f16^T.
    /// FP32 accumulation by default; FP16 accumulation when DOTLLM_CUDA_GEMM_16F=1 (G1 experiment).
    /// </summary>
    public static unsafe void LinearF16(nint handle, nint xF16, nint wF16, nint yF16,
                                          int m, int k, int n, nint stream)
    {
        CublasApi.cublasSetStream_v2(handle, stream).ThrowOnCublasError();

        if (Use16FCompute)
        {
            ushort halfOne = 0x3C00, halfZero = 0x0000; // FP16 1.0, 0.0
            CublasApi.cublasGemmEx(
                handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                n, m, k,
                (nint)(&halfOne),
                wF16, CublasApi.CUDA_R_16F, k,
                xF16, CublasApi.CUDA_R_16F, k,
                (nint)(&halfZero),
                yF16, CublasApi.CUDA_R_16F, n,
                CublasApi.CUBLAS_COMPUTE_16F,
                CublasApi.CUBLAS_GEMM_DEFAULT
            ).ThrowOnCublasError();
            return;
        }

        float alpha = FloatOne;
        float beta = FloatZero;

        CublasApi.cublasGemmEx(
            handle,
            CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            n, m, k,
            (nint)(&alpha),
            wF16, CublasApi.CUDA_R_16F, k,
            xF16, CublasApi.CUDA_R_16F, k,
            (nint)(&beta),
            yF16, CublasApi.CUDA_R_16F, n,
            CublasApi.CUBLAS_COMPUTE_32F,
            CublasApi.CUBLAS_GEMM_DEFAULT
        ).ThrowOnCublasError();
    }

    /// <summary>
    /// GEMV for single token: y_f16[n] = W_f16[n,k] × x_f16[k].
    /// </summary>
    public static void GemvF16(nint handle, nint wF16, nint xF16, nint yF16,
                                 int n, int k, nint stream)
    {
        LinearF16(handle, xF16, wF16, yF16, 1, k, n, stream);
    }

    /// <summary>
    /// FP32 linear projection: <c>Y_f32[m, n] = X_f32[m, k] × W_f32^T</c> where
    /// <c>W</c> is row-major <c>[n, k]</c>. Uses <c>cublasGemmEx</c> with
    /// <c>CUDA_R_32F</c> + <c>CUBLAS_COMPUTE_32F</c>. Used by the MLA Phase 1
    /// path which keeps the entire attention block in F32 for byte-near-equivalence
    /// with the CPU oracle.
    /// </summary>
    /// <remarks>
    /// Mirrors the layout convention of <see cref="LinearF16"/>: caller-side
    /// math is row-major <c>[m, k] × [n, k]^T = [m, n]</c>; cuBLAS sees the
    /// transposed column-major view (<c>op(W)=T, op(X)=N, lda=k, ldb=k, ldc=n</c>),
    /// so the leading dims and ldc match the underlying row contiguity.
    /// </remarks>
    public static unsafe void LinearF32(nint handle, nint xF32, nint wF32, nint yF32,
                                          int m, int k, int n, nint stream)
    {
        CublasApi.cublasSetStream_v2(handle, stream).ThrowOnCublasError();

        float alpha = FloatOne;
        float beta = FloatZero;

        CublasApi.cublasGemmEx(
            handle,
            CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            n, m, k,
            (nint)(&alpha),
            wF32, CublasApi.CUDA_R_32F, k,
            xF32, CublasApi.CUDA_R_32F, k,
            (nint)(&beta),
            yF32, CublasApi.CUDA_R_32F, n,
            CublasApi.CUBLAS_COMPUTE_32F_PEDANTIC,
            CublasApi.CUBLAS_GEMM_DEFAULT
        ).ThrowOnCublasError();
    }

    /// <summary>
    /// FP32 GEMV for single token: <c>y_f32[n] = W_f32[n,k] × x_f32[k]</c>.
    /// Convenience wrapper over <see cref="LinearF32"/>.
    /// </summary>
    public static void GemvF32(nint handle, nint wF32, nint xF32, nint yF32,
                                 int n, int k, nint stream)
    {
        LinearF32(handle, xF32, wF32, yF32, 1, k, n, stream);
    }

    /// <summary>
    /// Accumulating scaled GEMV for single token: y_f16[n] += alpha × W_f16[n,k] × x_f16[k].
    /// Uses FP32 accumulation via cublasGemmEx with beta=1 so it adds into the existing y.
    /// </summary>
    public static void GemvF16Accum(nint handle, nint wF16, nint xF16, nint yF16,
                                     int n, int k, float alpha, nint stream)
    {
        LinearF16Accum(handle, xF16, wF16, yF16, 1, k, n, alpha, stream);
    }

    /// <summary>
    /// Accumulating scaled batched projection: y_f16[m, n] += alpha × X_f16[m, k] × W_f16^T.
    /// Uses FP32 accumulation via cublasGemmEx with beta=1 so it adds into the existing y.
    /// </summary>
    public static unsafe void LinearF16Accum(nint handle, nint xF16, nint wF16, nint yF16,
                                              int m, int k, int n, float alpha, nint stream)
    {
        CublasApi.cublasSetStream_v2(handle, stream).ThrowOnCublasError();

        float a = alpha;
        float b = 1.0f; // beta=1: accumulate into y

        CublasApi.cublasGemmEx(
            handle,
            CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            n, m, k,
            (nint)(&a),
            wF16, CublasApi.CUDA_R_16F, k,
            xF16, CublasApi.CUDA_R_16F, k,
            (nint)(&b),
            yF16, CublasApi.CUDA_R_16F, n,
            CublasApi.CUBLAS_COMPUTE_32F,
            CublasApi.CUBLAS_GEMM_DEFAULT
        ).ThrowOnCublasError();
    }
}
