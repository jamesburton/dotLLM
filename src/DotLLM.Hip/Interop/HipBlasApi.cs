using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Minimal hipBLAS P/Invoke. libhipblas.so / hipblas.dll — from the ROCm stack.
/// Mirrors <c>DotLLM.Cuda.Interop.CublasApi</c> 1:1 — hipBLAS is the source-level
/// compatible AMD counterpart of cuBLAS.
/// </summary>
internal static partial class HipBlasApi
{
    // .NET resolves "hipblas" via HipLibraryResolver.
    private const string LibName = "hipblas";

    [LibraryImport(LibName)]
    internal static partial int hipblasCreate(out nint handle);

    [LibraryImport(LibName)]
    internal static partial int hipblasDestroy(nint handle);

    [LibraryImport(LibName)]
    internal static partial int hipblasSetStream(nint handle, nint stream);

    /// <summary>
    /// FP16 GEMM — C = alpha * op(A) * op(B) + beta * C, all FP16.
    /// Matrix cores used automatically on supported AMD architectures (CDNA, RDNA3+).
    /// Row-major trick: compute C^T = B^T @ A^T via swapped args (same pattern as cuBLAS).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int hipblasHgemm(
        nint handle,
        int transa, int transb,     // hipblasOperation_t: 111=N, 112=T, 113=C
        int m, int n, int k,
        in ushort alpha,            // __half passed as ushort
        nint A, int lda,
        nint B, int ldb,
        in ushort beta,
        nint C, int ldc);

    /// <summary>
    /// Mixed-precision GEMM — FP16 input, FP32 accumulate.
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int hipblasGemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        nint alpha,
        nint A, int Atype, int lda,
        nint B, int Btype, int ldb,
        nint beta,
        nint C, int Ctype, int ldc,
        int computeType, int algo);

    // ── hipBLAS constants ─────────────────────────────────────────────
    // NOTE: hipBLAS enum values differ from cuBLAS. hipBLAS uses 111/112/113
    // for operation flags (matching rocBLAS and the legacy BLAS tradition),
    // while cuBLAS uses 0/1/2.

    /// <summary>HIPBLAS_OP_N — no transpose.</summary>
    internal const int HIPBLAS_OP_N = 111;

    /// <summary>HIPBLAS_OP_T — transpose.</summary>
    internal const int HIPBLAS_OP_T = 112;

    /// <summary>HIPBLAS_R_16F — FP16 data type.</summary>
    internal const int HIPBLAS_R_16F = 150;

    /// <summary>HIPBLAS_R_32F — FP32 data type.</summary>
    internal const int HIPBLAS_R_32F = 151;

    /// <summary>HIPBLAS_COMPUTE_16F — FP16 compute.</summary>
    internal const int HIPBLAS_COMPUTE_16F = 0;

    /// <summary>HIPBLAS_COMPUTE_32F — FP32 compute.</summary>
    internal const int HIPBLAS_COMPUTE_32F = 1;

    /// <summary>HIPBLAS_GEMM_DEFAULT — default algorithm selection.</summary>
    internal const int HIPBLAS_GEMM_DEFAULT = 160;
}
