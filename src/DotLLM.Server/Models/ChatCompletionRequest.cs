using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible chat completion request.
/// </summary>
public sealed record ChatCompletionRequest
{
    [JsonPropertyName("messages")]
    public required ChatMessageDto[] Messages { get; init; }

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

    /// <summary>
    /// Streaming options (#450). Today only <c>include_usage</c> is meaningful: it asks for a
    /// final usage-only chunk, which SDKs rely on for token accounting over a stream.
    /// </summary>
    [JsonPropertyName("stream_options")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StreamOptionsDto? StreamOptions { get; init; }

    /// <summary>
    /// True when this request asked for the final <c>choices: []</c> usage chunk.
    /// </summary>
    [JsonIgnore]
    public bool WantsUsageChunk => StreamOptions?.IncludeUsage == true;

    /// <summary>
    /// When <c>false</c>, the assistant may emit at most one tool call per turn (#450). Null =
    /// OpenAI's default, i.e. parallel calls are allowed. The model is not constrained during
    /// decode, so this is enforced on the response.
    /// </summary>
    [JsonPropertyName("parallel_tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? ParallelToolCalls { get; init; }

    /// <summary>
    /// End-user identifier for abuse tracking. Accepted and ignored — this server has no
    /// per-end-user concept, and a client that always sends it must not get a 400 (#450).
    /// </summary>
    [JsonPropertyName("user")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? User { get; init; }

    /// <summary>
    /// Whether to persist the completion for OpenAI's dashboard. Accepted and ignored — nothing
    /// is stored server-side here (#450).
    /// </summary>
    [JsonPropertyName("store")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? Store { get; init; }

    /// <summary>
    /// Latency tier hint. Accepted and ignored — there is one tier (#450).
    /// </summary>
    [JsonPropertyName("service_tier")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ServiceTier { get; init; }

    /// <summary>
    /// Reasoning-budget hint for o-series models. Accepted and ignored (#450).
    /// </summary>
    [JsonPropertyName("reasoning_effort")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ReasoningEffort { get; init; }

    /// <summary>
    /// Opaque client key/value tags. Accepted and ignored (#450).
    /// </summary>
    [JsonPropertyName("metadata")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Metadata { get; init; }

    [JsonPropertyName("stop")]
    public JsonElement? Stop { get; init; }

    [JsonPropertyName("tools")]
    public ToolDefinitionDto[]? Tools { get; init; }

    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; init; }

    [JsonPropertyName("response_format")]
    public JsonElement? ResponseFormat { get; init; }

    [JsonPropertyName("seed")]
    public int? Seed { get; init; }

    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; init; }

    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; init; }

    [JsonPropertyName("repetition_penalty")]
    public float? RepetitionPenalty { get; init; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }

    [JsonPropertyName("min_p")]
    public float? MinP { get; init; }

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

    [JsonPropertyName("logprobs")]
    public bool? Logprobs { get; init; }

    [JsonPropertyName("top_logprobs")]
    public int? TopLogprobs { get; init; }

    [JsonPropertyName("n")]
    public int N { get; init; } = 1;

    /// <summary>
    /// Optional LoRA adapter name (must already be registered with the server's
    /// <c>LoraAdapterRegistry</c>). When null/empty, the request runs against
    /// the base model with no adapter delta. Phase 4c additive field — does not
    /// alter behaviour for existing requests.
    /// </summary>
    [JsonPropertyName("lora_adapter")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? LoraAdapter { get; init; }

    /// <summary>
    /// Optional named prefix id registered via <c>POST /v1/prompt-cache/{id}</c>.
    /// When supplied the engine ensures the named prefix is honoured for this
    /// request (best-effort hint — the trie still does longest-prefix matching).
    /// </summary>
    [JsonPropertyName("prefix_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? PrefixId { get; init; }

    /// <summary>
    /// Optional diffusion-decode overrides. Honoured only when the loaded model
    /// is a diffusion model (its <c>ModelConfig.DiffusionConfig</c> is non-null);
    /// ignored entirely on the autoregressive path. When null, the model's
    /// verified <c>DiffusionConfig</c> defaults are used unchanged.
    /// </summary>
    [JsonPropertyName("diffusion")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public DiffusionOptionsDto? Diffusion { get; init; }

    /// <summary>
    /// Idle-unload duration in seconds for the target model (#369, ollama parity). Null = use the
    /// server-wide default. 0 = unload immediately after this request. Negative = never
    /// auto-unload. Combine with <see cref="Model"/> to route to (and keep resident) a specific
    /// model when the server has more than one loaded.
    /// </summary>
    [JsonPropertyName("keep_alive")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KeepAlive { get; init; }
}

/// <summary>
/// OpenAI <c>stream_options</c> (#450). Shared by the chat and raw-completion requests.
/// </summary>
public sealed record StreamOptionsDto
{
    /// <summary>
    /// When true, the stream emits one extra chunk before <c>[DONE]</c> carrying <c>usage</c>
    /// and an empty <c>choices</c> array. SDKs use it to report token counts for a stream.
    /// </summary>
    [JsonPropertyName("include_usage")]
    public bool IncludeUsage { get; init; }
}

/// <summary>
/// Per-request diffusion-decode overrides (additive — only consulted on the
/// diffusion path). Every field is nullable; a null field falls back to the
/// model's <c>DiffusionConfig</c> default. <c>max_tokens</c> still maps to the
/// overall target length; these tune the canvas/schedule shape.
/// </summary>
public sealed record DiffusionOptionsDto
{
    /// <summary>Override the per-canvas length (<c>DiffusionConfig.CanvasLength</c>).</summary>
    [JsonPropertyName("canvas_length")]
    public int? CanvasLength { get; init; }

    /// <summary>Override the max denoise steps per canvas (<c>DiffusionConfig.MaxDenoisingSteps</c>).</summary>
    [JsonPropertyName("max_denoising_steps")]
    public int? MaxDenoisingSteps { get; init; }

    /// <summary>Override the upper bound of the linear temperature schedule (<c>t_max</c>).</summary>
    [JsonPropertyName("temperature_max")]
    public float? TemperatureMax { get; init; }

    /// <summary>Override the lower bound of the linear temperature schedule (<c>t_min</c>).</summary>
    [JsonPropertyName("temperature_min")]
    public float? TemperatureMin { get; init; }
}

/// <summary>
/// A chat message in the OpenAI format.
/// </summary>
public sealed record ChatMessageDto
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("tool_calls")]
    public ToolCallDto[]? ToolCalls { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }
}

/// <summary>
/// Tool definition in the OpenAI format.
/// </summary>
public sealed record ToolDefinitionDto
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "function";

    [JsonPropertyName("function")]
    public required ToolFunctionDto Function { get; init; }
}

/// <summary>
/// Function definition within a tool.
/// </summary>
public sealed record ToolFunctionDto
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; init; }
}

/// <summary>
/// A tool call made by the assistant.
/// </summary>
public sealed record ToolCallDto
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("type")]
    public string Type { get; init; } = "function";

    [JsonPropertyName("function")]
    public required ToolCallFunctionDto Function { get; init; }
}

/// <summary>
/// Function invocation within a tool call.
/// </summary>
public sealed record ToolCallFunctionDto
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("arguments")]
    public required string Arguments { get; init; }
}
