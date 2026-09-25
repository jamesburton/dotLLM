using DotLLM.Core.Sampling;

namespace DotLLM.Core.Configuration;

/// <summary>
/// Options controlling inference behavior: sampling parameters, stop conditions, and limits.
/// </summary>
public record InferenceOptions
{
    /// <summary>Temperature for sampling. 0 = greedy.</summary>
    public float Temperature { get; init; } = 0.7f;

    /// <summary>Top-K sampling. 0 = disabled.</summary>
    public int TopK { get; init; }

    /// <summary>Top-P (nucleus) sampling threshold.</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Min-P sampling threshold. 0 = disabled.</summary>
    public float MinP { get; init; }

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Number of recent tokens for repetition penalty lookback. 0 = full history.</summary>
    public int RepetitionPenaltyWindow { get; init; }

    /// <summary>OpenAI-style frequency penalty: subtracted proportionally to occurrence count. 0 = disabled.</summary>
    public float FrequencyPenalty { get; init; }

    /// <summary>OpenAI-style presence penalty: subtracted once for any token already seen. 0 = disabled.</summary>
    public float PresencePenalty { get; init; }

    /// <summary>Per-token additive logit bias (token id → bias), OpenAI API compatible. Null/empty = disabled.</summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }

    /// <summary>
    /// Top-nσ sampling threshold (llama.cpp <c>--top-nsigma</c>). Keeps tokens with
    /// logit ≥ max(logits) - n × stddev(logits), computed over the raw logit distribution
    /// before temperature/top-k/top-p/min-p are applied. Negative = disabled (default).
    /// </summary>
    public float TopNSigma { get; init; } = -1f;

    /// <summary>
    /// DRY (Don't Repeat Yourself) repetition penalty multiplier (llama.cpp <c>--dry-multiplier</c>).
    /// Penalizes tokens that would continue a previously-seen repeated n-gram, with an exponentially
    /// growing penalty as the matched pattern gets longer. 0 = disabled (default).
    /// </summary>
    public float DryMultiplier { get; init; }

    /// <summary>DRY exponential base for the match-length penalty curve (llama.cpp <c>--dry-base</c>).</summary>
    public float DryBase { get; init; } = 1.75f;

    /// <summary>Minimum matched n-gram length before DRY starts penalizing (llama.cpp <c>--dry-allowed-length</c>).</summary>
    public int DryAllowedLength { get; init; } = 2;

    /// <summary>
    /// Number of recent tokens considered for DRY n-gram matching (llama.cpp <c>--dry-penalty-last-n</c>).
    /// 0 = full history (default).
    /// </summary>
    public int DryPenaltyLastN { get; init; }

    /// <summary>
    /// Token strings that reset DRY n-gram matching (llama.cpp <c>--dry-sequence-breaker</c>), e.g.
    /// newline or punctuation. Resolved to token ids by the tokenizer when the pipeline is built.
    /// </summary>
    public IReadOnlyList<string> DrySequenceBreakers { get; init; } = ["\n", ":", "\"", "*"];

    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Random seed for reproducible sampling. Null = non-deterministic.</summary>
    public int? Seed { get; init; }

    /// <summary>Stop sequences that terminate generation.</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>
    /// Explicit sampler steps composing the sampling pipeline.
    /// When set, these steps are used instead of building from the flat properties
    /// (Temperature, TopK, TopP, MinP). Steps are applied in order.
    /// </summary>
    public IReadOnlyList<ISamplerStep>? SamplerSteps { get; init; }

    /// <summary>
    /// Explicit logit processors (e.g., repetition penalty).
    /// When set, used instead of building from RepetitionPenalty.
    /// </summary>
    public IReadOnlyList<ILogitProcessor>? LogitProcessors { get; init; }

    /// <summary>
    /// Explicit stop conditions. When set, used instead of the default
    /// (EOS + MaxTokens + StopSequences). The caller controls the full set.
    /// </summary>
    public IReadOnlyList<IStopCondition>? StopConditions { get; init; }

    /// <summary>
    /// Response format constraint. When set to <see cref="ResponseFormat.JsonObject"/>,
    /// output is guaranteed to be syntactically valid JSON via FSM-based constrained decoding.
    /// Null or <see cref="ResponseFormat.Text"/> means no constraint (default).
    /// </summary>
    public ResponseFormat? ResponseFormat { get; init; }

    /// <summary>Whether to return log-probabilities for each generated token.</summary>
    public bool Logprobs { get; init; }

    /// <summary>Number of top alternative tokens to include per position (0-20). Only used when <see cref="Logprobs"/> is true.</summary>
    public int TopLogprobs { get; init; }

    /// <summary>CPU threading configuration for parallel inference. Default: auto (all cores).</summary>
    public ThreadingConfig Threading { get; init; } = ThreadingConfig.Auto;
}
