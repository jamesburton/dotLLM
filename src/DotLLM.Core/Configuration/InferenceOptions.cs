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

    /// <summary>
    /// Repeat guard (#1291): end the request as <c>finish_reason=length</c> after this many identical tokens in a row.
    /// Null uses the engine default (<c>RepeatGuardStopCondition.DefaultLimit</c>, 256); 0 disables. Only applies when
    /// <see cref="StopConditions"/> is not supplied.
    /// </summary>
    public int? RepeatGuardLimit { get; init; }

    /// <summary>
    /// Thinking budget (#1292): after this many thinking tokens the engine forces the budget message and the close tag.
    /// Null (default) = unlimited. Needs the sampler pipeline to have a tokenizer.
    /// </summary>
    public ThinkingBudget? ThinkingBudget { get; init; }

    /// <summary>
    /// Never sample an end-of-generation token (llama.cpp <c>ignore_eos</c>, #1345): their logits are masked to -infinity, so
    /// generation runs to <c>MaxTokens</c> or an explicit stop string. Needs the sampler pipeline to have a tokenizer.
    /// </summary>
    public bool IgnoreEos { get; init; }

    /// <summary>Mask end-of-generation tokens until this many tokens are generated (<c>min_tokens</c>, #1345). 0 = off.</summary>
    public int MinTokens { get; init; }

    /// <summary>Random seed for reproducible sampling. Null = non-deterministic.</summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Draw stochastic tokens as a position-seeded Gumbel-max, <c>argmax(prepared logits + Gumbel(seed, generation index))</c> (#1358),
    /// instead of from a sequential RNG stream. Exact, and speculative decoding then yields the same text as plain decoding for the same
    /// seed. Default off (it changes which tokens a given <see cref="Seed"/> produces); <c>DOTLLM_GUMBEL_SAMPLING=1</c> turns it on globally.
    /// No property initializer: this type may be deserialized through a source-generated context.
    /// </summary>
    public bool GumbelSampling { get; init; }

    /// <summary>Stop sequences that terminate generation.</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>
    /// When non-null, <see cref="StopSequences"/> are suspended while the model is inside a reasoning block
    /// (#767) and only match answer text after it. Null = stop strings are active from the first token.
    /// EOS and max-tokens are never gated.
    /// </summary>
    public StopGate? ReasoningStopGate { get; init; }

    /// <summary>
    /// Entries of <see cref="StopSequences"/> exempt from <see cref="ReasoningStopGate"/> (template control
    /// tokens such as <c>&lt;|im_end|&gt;</c> that must end the turn wherever they appear).
    /// </summary>
    public IReadOnlyList<string>? StopSequencesUngated { get; init; }

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

    /// <summary>
    /// The text the chat template appended to the prompt as its generation prompt (e.g. <c>&lt;|im_start|&gt;assistant\n</c>),
    /// or <see langword="null"/> when unknown (raw completions, or a template that cannot render without it). A recurrent
    /// model's prefix snapshot then lands just before it, so the next turn of the conversation reuses this whole turn even
    /// when it re-renders the generation prompt differently (issue #1093). Null keeps the previous snapshot placement.
    /// </summary>
    public string? GenerationPromptSuffix { get; init; }
}
