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
    /// Linear projection (F32 in, F32 out): Y_f32[m, n] = X_f32[m, k] × W_f32^T.
    /// FP32 accumulation. Weight layout matches LinearF16: W[outputDim, inputDim]
    /// row-major. Used by MoE forward where the per-expert kernel chain stays
    /// in F32 for parity with the CPU oracle.
    /// </summary>
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
            CublasApi.CUBLAS_COMPUTE_32F,
            CublasApi.CUBLAS_GEMM_DEFAULT
        ).ThrowOnCublasError();
    }
}
