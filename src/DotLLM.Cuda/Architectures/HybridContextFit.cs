namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Result of <see cref="CudaQwen3HybridDenseTransformerModel.MeasureServingContextFit"/> (issue #1068): the longest context
/// that fits in free VRAM, and the numbers it came from, for the startup log.
/// </summary>
/// <param name="MaxContext">Fitted context in tokens, rounded down to 1024 and capped at the model's window; 0 when nothing fits.</param>
/// <param name="FreeBytes">Free device memory after the sizing forward released its context-sized buffers.</param>
/// <param name="ReserveBytes">Device memory kept free.</param>
/// <param name="FixedBytes">Context-independent bytes still to come (a second recurrent checkpoint).</param>
/// <param name="BytesPerToken">Peak device bytes per context token.</param>
/// <param name="SizingError">Set when the sizing forward itself was refused; the fit then rests on the current free figure.</param>
public readonly record struct HybridContextFit(
    int MaxContext, long FreeBytes, long ReserveBytes, long FixedBytes, long BytesPerToken, string? SizingError)
{
    /// <summary>One-line summary for the startup log.</summary>
    public string Describe()
        => $"{FreeBytes >> 20} MiB free after load, {ReserveBytes >> 20} MiB reserve, {FixedBytes >> 20} MiB fixed, " +
           $"{BytesPerToken / 1024.0:F1} KiB/token";
}
