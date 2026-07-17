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
