using DotLLM.Core.Configuration;

namespace DotLLM.Core.PositionEncoding;

/// <summary>
/// Configuration for Rotary Position Embeddings (RoPE).
/// </summary>
/// <param name="Theta">Base frequency. Default 10000.0, Llama 3 uses 500000.0.</param>
/// <param name="DimensionCount">Number of dimensions for the rotation.</param>
/// <param name="Type">Element-pairing convention. Must match GGUF Q/K weight layout.</param>
/// <param name="ScalingType">Context-length scaling strategy.</param>
/// <param name="ScalingFactor">Scaling factor for Linear/NTK methods.</param>
/// <param name="OrigMaxSeqLen">Original max sequence length before scaling.</param>
/// <param name="AttnFactor">YaRN attention factor.</param>
/// <param name="BetaFast">YaRN beta-fast parameter.</param>
/// <param name="BetaSlow">YaRN beta-slow parameter.</param>
public readonly record struct RoPEConfig(
    float Theta = 10000.0f,
    int DimensionCount = 0,
    RoPEType Type = RoPEType.Norm,
    RoPEScalingType ScalingType = RoPEScalingType.None,
    float ScalingFactor = 1.0f,
    int OrigMaxSeqLen = 0,
    float AttnFactor = 1.0f,
    float BetaFast = 32.0f,
    float BetaSlow = 1.0f)
{
    /// <summary>
    /// <c>true</c> when the dense (non-MLA) YaRN ramp applies: a YaRN scaling type
    /// with a real (&gt; 1) factor and a known original context length. This is the
    /// single predicate every backend must agree on — the CPU reference gates its
    /// <c>RoPE.PrecomputeFrequencyTableYarn</c> rebuild on exactly these three
    /// conditions, and CUDA gates its inverse-frequency table upload on the same.
    /// </summary>
    public bool IsDenseYarnActive
        => ScalingType == RoPEScalingType.YaRN && ScalingFactor > 1.0f && OrigMaxSeqLen > 0;

    /// <summary>
    /// The context window YaRN extends the model to: <c>factor × original</c>
    /// (<see cref="ScalingFactor"/> × <see cref="OrigMaxSeqLen"/>), or 0 when
    /// <see cref="IsDenseYarnActive"/> is false. Qwen3.8's opt-in YaRN (factor 4 over
    /// 262,144) gives 1,048,576 (issue #1101).
    /// </summary>
    public int YarnExtendedContextLength
        => IsDenseYarnActive
            ? (int)Math.Min(int.MaxValue, Math.Round((double)ScalingFactor * OrigMaxSeqLen))
            : 0;

    /// <summary>
    /// The scalar YaRN "attention magnitude concentration" multiplier applied to
    /// <em>both</em> the cosine and sine tables — i.e. to every rotated element at
    /// <em>every</em> position, position 0 included. gpt-oss follows llama.cpp's
    /// <c>ggml_rope_ext</c> YaRN convention (<c>ext_factor = 1</c>), which folds
    /// <c>attn_factor * (1 + 0.1 * ln(factor))</c> into cos/sin. The Qwen3.5/3.8 hybrids
    /// (<see cref="Architecture.Qwen3HybridDense"/>, <see cref="Architecture.Qwen3MoeHybrid"/>) and
    /// Qwen3.8-Flash-Next (<see cref="Architecture.Qwen4Exp"/>) use the same value: HF
    /// <c>_compute_yarn_parameters</c> defaults <c>attention_factor</c> to
    /// <c>0.1·ln(factor) + 1</c> and multiplies cos and sin by it (issue #1101). The other
    /// dense-YaRN architectures (SmolLM3, Llama 3.1+) keep the plain <see cref="AttnFactor"/>
    /// convention established when they were wired.
    /// </summary>
    /// <param name="architecture">Model architecture selecting the mscale convention.</param>
    /// <remarks>
    /// Because this multiplies cos/sin rather than only the frequencies, it changes
    /// SHORT-context numerics too: at position 0 the rotation degenerates to
    /// <c>(cos, sin) = (mscale, 0)</c> instead of <c>(1, 0)</c>, scaling Q and K by
    /// <c>mscale</c>. Any backend that ignores it is wrong from the very first token,
    /// not merely beyond <see cref="OrigMaxSeqLen"/>.
    /// </remarks>
    public float ComputeYarnMscaleMultiplier(Architecture architecture)
        => UsesLogFactorYarnMscale(architecture)
            ? AttnFactor * (1.0f + 0.1f * MathF.Log(ScalingFactor))
            : AttnFactor;

    /// <summary>
    /// Whether <paramref name="architecture"/> folds <c>1 + 0.1·ln(factor)</c> into cos/sin
    /// (llama.cpp <c>ggml_rope_ext</c> / HF default <c>attention_factor</c>).
    /// </summary>
    /// <param name="architecture">Model architecture.</param>
    /// <returns><see langword="true"/> for gpt-oss and the Qwen3.5/3.8 family.</returns>
    private static bool UsesLogFactorYarnMscale(Architecture architecture)
        => architecture is Architecture.GptOss
            or Architecture.Qwen3HybridDense or Architecture.Qwen3MoeHybrid or Architecture.Qwen4Exp;
}
