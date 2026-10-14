using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible raw completion request (no chat template).
/// </summary>
public sealed record CompletionRequest
{
    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("stream")]
    public bool Stream { get; init; }

    /// <summary>Streaming options (#450). See <see cref="StreamOptionsDto"/>.</summary>
    [JsonPropertyName("stream_options")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StreamOptionsDto? StreamOptions { get; init; }

    /// <summary>True when this request asked for the final <c>choices: []</c> usage chunk.</summary>
    [JsonIgnore]
    public bool WantsUsageChunk => StreamOptions?.IncludeUsage == true;

    /// <summary>End-user identifier. Accepted and ignored (#450).</summary>
    [JsonPropertyName("user")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? User { get; init; }

    /// <summary>Completion persistence flag. Accepted and ignored (#450).</summary>
    [JsonPropertyName("store")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? Store { get; init; }

    /// <summary>Latency tier hint. Accepted and ignored (#450).</summary>
    [JsonPropertyName("service_tier")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ServiceTier { get; init; }

    /// <summary>Reasoning-budget hint. Accepted and ignored (#450).</summary>
    [JsonPropertyName("reasoning_effort")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ReasoningEffort { get; init; }

    /// <summary>Opaque client key/value tags. Accepted and ignored (#450).</summary>
    [JsonPropertyName("metadata")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Metadata { get; init; }

    [JsonPropertyName("stop")]
    public JsonElement? Stop { get; init; }

    [JsonPropertyName("seed")]
    public int? Seed { get; init; }

    [JsonPropertyName("repetition_penalty")]
    public float? RepetitionPenalty { get; init; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }

    [JsonPropertyName("min_p")]
    public float? MinP { get; init; }

    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; init; }

    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; init; }

    /// <summary>
    /// Per-token additive logit bias (OpenAI API compatible): a map from token id (as a string key)
    /// to a bias value applied before sampling. Typical range is -100..100.
    /// </summary>
    [JsonPropertyName("logit_bias")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, float>? LogitBias { get; init; }

    /// <summary>Top-nσ sampling threshold (llama.cpp <c>--top-nsigma</c>). Negative = disabled.</summary>
    [JsonPropertyName("top_n_sigma")]
    public float? TopNSigma { get; init; }

    /// <summary>DRY repetition penalty multiplier. 0/absent = disabled.</summary>
    [JsonPropertyName("dry_multiplier")]
    public float? DryMultiplier { get; init; }

    /// <summary>DRY exponential base for the match-length penalty curve.</summary>
    [JsonPropertyName("dry_base")]
    public float? DryBase { get; init; }

    /// <summary>Minimum matched n-gram length before DRY starts penalizing.</summary>
    [JsonPropertyName("dry_allowed_length")]
    public int? DryAllowedLength { get; init; }

    /// <summary>Number of recent tokens considered for DRY matching. 0 = full history.</summary>
    [JsonPropertyName("dry_penalty_last_n")]
    public int? DryPenaltyLastN { get; init; }

    /// <summary>Token strings that reset DRY n-gram matching.</summary>
    [JsonPropertyName("dry_sequence_breakers")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string[]? DrySequenceBreakers { get; init; }

    [JsonPropertyName("response_format")]
    public JsonElement? ResponseFormat { get; init; }

    [JsonPropertyName("logprobs")]
    public bool? Logprobs { get; init; }

    [JsonPropertyName("top_logprobs")]
    public int? TopLogprobs { get; init; }

    /// <summary>
    /// Optional LoRA adapter name (must already be registered with the server's
    /// <c>LoraAdapterRegistry</c>). When null/empty, the request runs against
    /// the base model with no adapter delta. Phase 4c additive field.
    /// </summary>
    [JsonPropertyName("lora_adapter")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? LoraAdapter { get; init; }

    /// <summary>
    /// Optional named prefix id registered via <c>POST /v1/prompt-cache/{id}</c>.
    /// Best-effort hint — the trie still does longest-prefix matching.
    /// </summary>
    [JsonPropertyName("prefix_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? PrefixId { get; init; }

    /// <summary>
    /// Idle-unload duration in seconds for the target model (#369, ollama parity). Null = use the
    /// server-wide default. 0 = unload immediately after this request. Negative = never
    /// auto-unload.
    /// </summary>
    [JsonPropertyName("keep_alive")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KeepAlive { get; init; }
}

/// <summary>
/// OpenAI-compatible raw completion response.
/// </summary>
public sealed record CompletionResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "text_completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required CompletionChoiceDto[] Choices { get; init; }

    [JsonPropertyName("usage")]
    public required UsageDto Usage { get; init; }
}

/// <summary>
/// A single choice in a raw completion response.
/// </summary>
public sealed record CompletionChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("text")]
    public required string Text { get; init; }

    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsDto? Logprobs { get; init; }

    [JsonPropertyName("finish_reason")]
    public required string FinishReason { get; init; }
}

/// <summary>
/// Streaming raw completion chunk.
/// </summary>
public sealed record CompletionChunk
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "text_completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required CompletionChunkChoiceDto[] Choices { get; init; }

    /// <summary>
    /// Populated only on the final usage chunk emitted for
    /// <c>stream_options.include_usage</c> (#450), which carries an empty <c>choices</c> array.
    /// </summary>
    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public UsageDto? Usage { get; init; }
}

/// <summary>
/// A single choice in a streaming completion chunk.
/// </summary>
public sealed record CompletionChunkChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("text")]
    public required string Text { get; init; }

    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsDto? Logprobs { get; init; }

    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; init; }
}
