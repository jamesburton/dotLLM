namespace DotLLM.Hip.Interop;

/// <summary>
/// Exception thrown when a HIP runtime or hipBLAS call fails.
/// </summary>
public sealed class HipException : Exception
{
    /// <summary>HIP error code (hipError_t or hipblasStatus_t).</summary>
    public int ErrorCode { get; }

    /// <summary>Creates a HIP exception with the given error code and message.</summary>
    public HipException(int errorCode, string message)
        : base($"HIP error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }
}
