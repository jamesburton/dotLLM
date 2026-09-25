namespace DotLLM.Core.Sampling;

/// <summary>
/// Context passed to <see cref="ILogitProcessor.Process"/> providing token history and request metadata.
/// </summary>
/// <param name="RepetitionPenalty">Repetition penalty factor. 1.0 = disabled.</param>
/// <param name="RepetitionPenaltyWindow">Number of recent tokens to consider for repetition penalty. 0 = full history.</param>
/// <param name="SequenceId">Identifier for the current sequence/request.</param>
/// <param name="FrequencyPenalty">OpenAI-style frequency penalty. Subtracted per-occurrence-count. 0 = disabled.</param>
/// <param name="PresencePenalty">OpenAI-style presence penalty. Subtracted once per token seen. 0 = disabled.</param>
/// <param name="LogitBias">Per-token additive bias map (token id → bias). Null/empty = disabled.</param>
/// <param name="DryMultiplier">DRY (Don't Repeat Yourself) penalty multiplier. 0 = disabled.</param>
/// <param name="DryBase">DRY exponential base for the match-length penalty curve.</param>
/// <param name="DryAllowedLength">Minimum matched n-gram length (inclusive) before DRY starts penalizing.</param>
/// <param name="DryPenaltyLastN">Number of recent tokens to consider for DRY matching. 0 or negative = full history.</param>
/// <param name="DrySequenceBreakers">Raw sequence-breaker strings (resolved to token ids by the caller, e.g. <c>SamplerPipeline</c>).</param>
public readonly record struct ProcessorContext(
    float RepetitionPenalty,
    int RepetitionPenaltyWindow,
    int SequenceId,
    float FrequencyPenalty = 0f,
    float PresencePenalty = 0f,
    IReadOnlyDictionary<int, float>? LogitBias = null,
    float DryMultiplier = 0f,
    float DryBase = 1.75f,
    int DryAllowedLength = 2,
    int DryPenaltyLastN = 0,
    IReadOnlyList<string>? DrySequenceBreakers = null);
