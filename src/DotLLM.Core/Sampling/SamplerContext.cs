namespace DotLLM.Core.Sampling;

/// <summary>
/// Context passed to <see cref="ISamplerStep.Apply"/> providing request-level information.
/// </summary>
/// <param name="Temperature">Current temperature setting.</param>
/// <param name="TopK">Top-K value. 0 = disabled.</param>
/// <param name="TopP">Top-P (nucleus) threshold.</param>
/// <param name="MinP">Min-P threshold. 0 = disabled.</param>
/// <param name="Seed">Random seed for deterministic sampling. Null = non-deterministic.</param>
/// <param name="TopNSigma">
/// Top-nσ threshold: keeps tokens with logit ≥ max(logits) - n × stddev(logits), computed over the
/// raw (pre-temperature) logit distribution. Negative = disabled (matches llama.cpp's convention).
/// </param>
public readonly record struct SamplerContext(
    float Temperature,
    int TopK,
    float TopP,
    float MinP,
    int? Seed,
    float TopNSigma = -1f);
