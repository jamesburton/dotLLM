namespace DotLLM.Core.Configuration;

/// <summary>
/// User-supplied overrides for a model's GGUF-derived RoPE configuration (llama.cpp
/// <c>--rope-scaling</c>/<c>--rope-freq-base</c>/<c>--rope-freq-scale</c>/<c>--yarn-*</c> flag
/// family). Every field is optional — a null field leaves the GGUF-derived value unchanged.
/// This is override plumbing only: no new scaling math, just a way to correct/replace metadata
/// that is wrong or absent (common on community context-extended quants).
/// </summary>
public sealed record RoPEOverrideOptions
{
    /// <summary>Overrides <see cref="PositionEncoding.RoPEConfig.ScalingType"/> (llama.cpp <c>--rope-scaling</c>).</summary>
    public RoPEScalingType? ScalingType { get; init; }

    /// <summary>Overrides <see cref="PositionEncoding.RoPEConfig.Theta"/> (llama.cpp <c>--rope-freq-base</c>).</summary>
    public float? FreqBase { get; init; }

    /// <summary>
    /// Overrides <see cref="PositionEncoding.RoPEConfig.ScalingFactor"/> (llama.cpp
    /// <c>--rope-scale</c> for the linear case, or the general scaling factor for YaRN/NTK).
    /// </summary>
    public float? ScalingFactor { get; init; }

    /// <summary>Overrides <see cref="PositionEncoding.RoPEConfig.OrigMaxSeqLen"/> (llama.cpp <c>--yarn-orig-ctx</c>).</summary>
    public int? OrigMaxSeqLen { get; init; }

    /// <summary>Overrides <see cref="PositionEncoding.RoPEConfig.AttnFactor"/> (llama.cpp <c>--yarn-attn-factor</c>).</summary>
    public float? AttnFactor { get; init; }

    /// <summary>Overrides <see cref="PositionEncoding.RoPEConfig.BetaFast"/> (llama.cpp <c>--yarn-beta-fast</c>).</summary>
    public float? BetaFast { get; init; }

    /// <summary>Overrides <see cref="PositionEncoding.RoPEConfig.BetaSlow"/> (llama.cpp <c>--yarn-beta-slow</c>).</summary>
    public float? BetaSlow { get; init; }

    /// <summary>True when at least one override field is set.</summary>
    public bool HasAnyOverride =>
        ScalingType is not null || FreqBase is not null || ScalingFactor is not null
        || OrigMaxSeqLen is not null || AttnFactor is not null || BetaFast is not null || BetaSlow is not null;
}
