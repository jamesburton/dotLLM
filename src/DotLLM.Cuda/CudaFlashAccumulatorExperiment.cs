namespace DotLLM.Cuda;

/// <summary>
/// Accumulator precision of the head_dim-256 flash prefill kernel, for the issue #1109 precision study.
/// </summary>
internal enum FlashAccumulatorMode
{
    /// <summary>Shipped kernel: F32 accumulation for both QK^T and PV.</summary>
    F32,

    /// <summary>F16 accumulation of the QK^T scores (Q pre-scaled by the exact 1/16), F32 PV.</summary>
    Qk16,

    /// <summary>F32 scores; F16 PV accumulator carried over the whole key range.</summary>
    Pv16,

    /// <summary>F32 scores; F16 PV within one 16-key tile, promoted into the F32 accumulator every tile.</summary>
    Pv16Tile,

    /// <summary>F16 accumulation for both QK^T and PV.</summary>
    Qk16Pv16,
}

/// <summary>
/// Issue #1109 experiment switch: selects an F16-accumulating variant of the flash prefill kernel
/// (<c>attention_flash_mma_prefill_hd256_xacc.ptx</c>). <c>DOTLLM_CUDA_FLASH_ACC</c> =
/// <c>qk16</c> | <c>pv16</c> | <c>pv16t</c> | <c>qk16pv16</c>; unset (the default) keeps the shipped F32-accumulate
/// kernels and never loads the experiment module. Measurement only: not a supported configuration.
/// </summary>
internal static class CudaFlashAccumulatorExperiment
{
    /// <summary>Environment variable that selects the variant at process start.</summary>
    internal const string EnvVar = "DOTLLM_CUDA_FLASH_ACC";

    /// <summary>
    /// Effective mode. Mutable so a harness can interleave variants within one warmed process; read on every
    /// flash prefill launch.
    /// </summary>
    internal static FlashAccumulatorMode Mode { get; set; } = Parse(Environment.GetEnvironmentVariable(EnvVar));

    /// <summary>Parses an environment value; anything unrecognised (including null) is <see cref="FlashAccumulatorMode.F32"/>.</summary>
    /// <param name="value">The raw value.</param>
    /// <returns>The mode.</returns>
    internal static FlashAccumulatorMode Parse(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        "qk16" => FlashAccumulatorMode.Qk16,
        "pv16" => FlashAccumulatorMode.Pv16,
        "pv16t" => FlashAccumulatorMode.Pv16Tile,
        "qk16pv16" => FlashAccumulatorMode.Qk16Pv16,
        _ => FlashAccumulatorMode.F32,
    };

    /// <summary>Kernel-name suffix of a non-default mode (the <c>_xacc.cu</c> entry naming).</summary>
    /// <param name="mode">A mode other than <see cref="FlashAccumulatorMode.F32"/>.</param>
    /// <returns>The suffix, e.g. <c>qk16</c>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is <see cref="FlashAccumulatorMode.F32"/>.</exception>
    internal static string Suffix(FlashAccumulatorMode mode) => mode switch
    {
        FlashAccumulatorMode.Qk16 => "qk16",
        FlashAccumulatorMode.Pv16 => "pv16",
        FlashAccumulatorMode.Pv16Tile => "pv16t",
        FlashAccumulatorMode.Qk16Pv16 => "qk16pv16",
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "F32 is the shipped kernel and has no experiment entry."),
    };
}
