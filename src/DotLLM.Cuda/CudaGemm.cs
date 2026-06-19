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

    /// <summary>
    /// Linear projection: Y_f16[m, n] = X_f16[m, k] × W_f16^T.
    /// FP32 accumulation, FP16 output.
    /// </summary>
    public static unsafe void LinearF16(nint handle, nint xF16, nint wF16, nint yF16,
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
