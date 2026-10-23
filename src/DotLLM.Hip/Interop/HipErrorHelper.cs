using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Extension methods for checking HIP and hipBLAS return codes.
/// </summary>
internal static class HipErrorHelper
{
    /// <summary>
    /// Throws <see cref="HipException"/> if <paramref name="result"/> is non-zero (hipSuccess = 0).
    /// </summary>
    internal static void ThrowOnError(this int result)
    {
        if (result == 0) return;

        string message = "Unknown HIP error";
        nint strPtr = HipDriverApi.hipGetErrorString(result);
        if (strPtr != 0)
            message = Marshal.PtrToStringAnsi(strPtr) ?? message;

        throw new HipException(result, message);
    }

    /// <summary>
    /// Throws <see cref="HipException"/> for hipBLAS errors with a "hipBLAS" prefix.
    /// </summary>
    internal static void ThrowOnHipBlasError(this int result)
    {
        if (result == 0) return;

        // hipBLAS status codes largely mirror cuBLAS for portability.
        string message = result switch
        {
            1 => "HIPBLAS_STATUS_NOT_INITIALIZED",
            2 => "HIPBLAS_STATUS_ALLOC_FAILED",
            3 => "HIPBLAS_STATUS_INVALID_VALUE",
            4 => "HIPBLAS_STATUS_MAPPING_ERROR",
            5 => "HIPBLAS_STATUS_EXECUTION_FAILED",
            6 => "HIPBLAS_STATUS_INTERNAL_ERROR",
            7 => "HIPBLAS_STATUS_NOT_SUPPORTED",
            8 => "HIPBLAS_STATUS_ARCH_MISMATCH",
            9 => "HIPBLAS_STATUS_HANDLE_IS_NULLPTR",
            10 => "HIPBLAS_STATUS_INVALID_ENUM",
            11 => "HIPBLAS_STATUS_UNKNOWN",
            _ => "Unknown hipBLAS error"
        };

        throw new HipException(result, $"hipBLAS: {message}");
    }
}
