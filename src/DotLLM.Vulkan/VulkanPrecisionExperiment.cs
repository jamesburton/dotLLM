namespace DotLLM.Vulkan;

/// <summary>Accumulator variant of the GQA-grouped flash prefill (<see cref="Kernels.VulkanFlashAttentionGqaKernel"/>), issue #1109.</summary>
internal enum FlashAccVariant
{
    /// <summary>The shipped kernel: F32 S and O, Q hi/lo split on the coopmat path.</summary>
    Default,

    /// <summary>Coopmat only: the shipped kernel with the Q lo-half term dropped (Q rounded to F16).</summary>
    NoQlo,

    /// <summary>F16 S = Q K^T accumulator, Q pre-scaled by the softmax scale (coopmat: Q hi only).</summary>
    Qk16,

    /// <summary>One KV tile's P.V accumulates in F16 and is added to the F32 O per tile.</summary>
    Pv16Tile,

    /// <summary>O is an F16 accumulator over the whole KV range.</summary>
    Pv16,

    /// <summary>Coopmat only: the default arithmetic plus score statistics written to <see cref="VulkanPrecisionExperiment.StatsBuffer"/>.</summary>
    Stats,
}

/// <summary>Accumulator variant of the GQA-grouped split-KV decode pass (<see cref="Kernels.VulkanSplitKvAttentionKernel"/>), issue #1109.</summary>
internal enum DecodeAccVariant
{
    /// <summary>The shipped pass: F32 running max / sum and P.V.</summary>
    Default,

    /// <summary>Running max, correction and sum rounded to F16 at every update.</summary>
    Stats16,

    /// <summary>F16 P.V accumulators.</summary>
    Pv16,

    /// <summary>Both <see cref="Stats16"/> and <see cref="Pv16"/>.</summary>
    Stats16Pv16,
}

/// <summary>Accumulator variant of the blocked cooperative-matrix projection GEMMs (PQ2_0 ladder), issue #1109.</summary>
internal enum GemmAccVariant
{
    /// <summary>The shipped kernel: F32 accumulator over the whole K range.</summary>
    Default,

    /// <summary>F16 accumulator over the whole K range.</summary>
    F16,

    /// <summary>F16 accumulator promoted into an F32 accumulator every 256 K elements.</summary>
    F16Promote256,
}

/// <summary>
/// Experiment-only switches for the #1109 precision study on Vulkan. Every switch defaults to the shipped behaviour and is
/// read from the environment once (so a server or CLI run can select a variant), and the in-process harness
/// (<c>precision-vulkan</c> in DotLLM.Benchmarks) flips them between interleaved A/B arms. The kernels resolve the switch at
/// record time and build the variant pipeline on first use; nothing about the shipped pipelines changes.
/// </summary>
internal static class VulkanPrecisionExperiment
{
    /// <summary>Environment variable selecting <see cref="Flash"/>: <c>noqlo</c>, <c>qk16</c>, <c>pv16t</c>, <c>pv16</c> or <c>stats</c>.</summary>
    public const string FlashEnvVar = "DOTLLM_VK_EXP_FA_ACC";

    /// <summary>Environment variable selecting <see cref="Decode"/>: <c>st16</c>, <c>pv16</c> or <c>st16pv16</c>.</summary>
    public const string DecodeEnvVar = "DOTLLM_VK_EXP_DECODE_ACC";

    /// <summary>Environment variable selecting <see cref="Gemm"/>: <c>f16</c> or <c>f16p256</c>.</summary>
    public const string GemmEnvVar = "DOTLLM_VK_EXP_GEMM_ACC";

    /// <summary>Environment variable: <c>1</c> rounds every row written to an F32 KV store to F16 values (<see cref="KvRoundF16"/>).</summary>
    public const string KvRoundEnvVar = "DOTLLM_VK_EXP_KV_F16_ROUND";

    /// <summary>Gets or sets the GQA flash prefill accumulator variant.</summary>
    public static FlashAccVariant Flash { get; set; } = ParseFlash(Environment.GetEnvironmentVariable(FlashEnvVar));

    /// <summary>Gets or sets the GQA split-KV decode accumulator variant.</summary>
    public static DecodeAccVariant Decode { get; set; } = ParseDecode(Environment.GetEnvironmentVariable(DecodeEnvVar));

    /// <summary>Gets or sets the blocked coopmat GEMM accumulator variant.</summary>
    public static GemmAccVariant Gemm { get; set; } = ParseGemm(Environment.GetEnvironmentVariable(GemmEnvVar));

    /// <summary>
    /// Gets or sets a value indicating whether rows written to an F32 KV store (the stores #1107 still tracks) are rounded to
    /// F16-representable values in place. Every attention kernel then computes on exactly the values an F16 store would hold,
    /// so the KL of this arm is the accuracy cost of narrowing that store, with no F16 kernel needed.
    /// </summary>
    public static bool KvRoundF16 { get; set; } = Environment.GetEnvironmentVariable(KvRoundEnvVar) == "1";

    /// <summary>
    /// Gets or sets the statistics buffer (<see cref="StatsWords"/> uints, zeroed by the caller) the <see cref="FlashAccVariant.Stats"/>
    /// flash variant accumulates into.
    /// </summary>
    public static VulkanDevice.Buffer? StatsBuffer { get; set; }

    /// <summary>Size of <see cref="StatsBuffer"/> in 32-bit words (layout in <c>attention_flash_gqa_cm.glsl</c>).</summary>
    public const int StatsWords = 323;

    /// <summary>Parses a <see cref="FlashEnvVar"/> value.</summary>
    /// <param name="s">The value (null or empty = default).</param>
    /// <returns>The variant.</returns>
    public static FlashAccVariant ParseFlash(string? s) => s?.Trim().ToLowerInvariant() switch
    {
        null or "" or "f32" or "default" => FlashAccVariant.Default,
        "noqlo" => FlashAccVariant.NoQlo,
        "qk16" => FlashAccVariant.Qk16,
        "pv16t" => FlashAccVariant.Pv16Tile,
        "pv16" => FlashAccVariant.Pv16,
        "stats" => FlashAccVariant.Stats,
        _ => throw new ArgumentException($"Unknown {FlashEnvVar} value '{s}'."),
    };

    /// <summary>Parses a <see cref="DecodeEnvVar"/> value.</summary>
    /// <param name="s">The value (null or empty = default).</param>
    /// <returns>The variant.</returns>
    public static DecodeAccVariant ParseDecode(string? s) => s?.Trim().ToLowerInvariant() switch
    {
        null or "" or "f32" or "default" => DecodeAccVariant.Default,
        "st16" => DecodeAccVariant.Stats16,
        "pv16" => DecodeAccVariant.Pv16,
        "st16pv16" => DecodeAccVariant.Stats16Pv16,
        _ => throw new ArgumentException($"Unknown {DecodeEnvVar} value '{s}'."),
    };

    /// <summary>Parses a <see cref="GemmEnvVar"/> value.</summary>
    /// <param name="s">The value (null or empty = default).</param>
    /// <returns>The variant.</returns>
    public static GemmAccVariant ParseGemm(string? s) => s?.Trim().ToLowerInvariant() switch
    {
        null or "" or "f32" or "default" => GemmAccVariant.Default,
        "f16" => GemmAccVariant.F16,
        "f16p256" => GemmAccVariant.F16Promote256,
        _ => throw new ArgumentException($"Unknown {GemmEnvVar} value '{s}'."),
    };

    /// <summary>The SPIR-V name suffix of a flash variant (<c>_x…</c>), or null for the variants that reuse the shipped shader.</summary>
    /// <param name="v">The variant.</param>
    /// <returns>The suffix.</returns>
    public static string? FlashSuffix(FlashAccVariant v) => v switch
    {
        FlashAccVariant.Qk16 => "_xqk16",
        FlashAccVariant.Pv16Tile => "_xpv16t",
        FlashAccVariant.Pv16 => "_xpv16",
        FlashAccVariant.Stats => "_xstats",
        _ => null,
    };

    /// <summary>The SPIR-V name suffix of a decode variant.</summary>
    /// <param name="v">The variant.</param>
    /// <returns>The suffix, or null for <see cref="DecodeAccVariant.Default"/>.</returns>
    public static string? DecodeSuffix(DecodeAccVariant v) => v switch
    {
        DecodeAccVariant.Stats16 => "_xst16",
        DecodeAccVariant.Pv16 => "_xpv16",
        DecodeAccVariant.Stats16Pv16 => "_xst16pv16",
        _ => null,
    };
}
