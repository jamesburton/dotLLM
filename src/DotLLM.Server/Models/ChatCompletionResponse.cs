using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible chat completion response (non-streaming).
/// </summary>
public sealed record ChatCompletionResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required ChatChoiceDto[] Choices { get; init; }

    [JsonPropertyName("usage")]
    public required UsageDto Usage { get; init; }

    /// <summary>Timing extension incl. the llama.cpp-shaped fields (#1345). Absent for multi-choice (<c>n &gt; 1</c>) responses.</summary>
    [JsonPropertyName("timings")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public TimingsDto? Timings { get; init; }
}

/// <summary>
/// A single choice in a chat completion response.
/// </summary>
public sealed record ChatChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("message")]
    public required ChatMessageDto Message { get; init; }

    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsDto? Logprobs { get; init; }

    [JsonPropertyName("finish_reason")]
    public required string FinishReason { get; init; }
}

/// <summary>
/// Token usage information.
/// </summary>
public sealed record UsageDto
{
    /// <summary>Breakdown of prompt tokens served from a reusable prefix.</summary>
    [JsonPropertyName("prompt_tokens_details")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public PromptTokensDetailsDto? PromptTokensDetails { get; init; }

    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }

    /// <summary>
    /// Breakdown of <see cref="CompletionTokens"/> (#767). Present only when reasoning was split from
    /// the answer; <c>reasoning_tokens</c> is part of, not in addition to, <c>completion_tokens</c>.
    /// </summary>
    [JsonPropertyName("completion_tokens_details")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public CompletionTokensDetailsDto? CompletionTokensDetails { get; init; }
}

/// <summary>OpenAI <c>usage.completion_tokens_details</c>.</summary>
public sealed record CompletionTokensDetailsDto
{
    /// <summary>Tokens spent on reasoning (everything up to and including <c>&lt;/think&gt;</c>).</summary>
    [JsonPropertyName("reasoning_tokens")]
    public int ReasoningTokens { get; init; }
}

/// <summary>OpenAI <c>usage.prompt_tokens_details</c>.</summary>
public sealed record PromptTokensDetailsDto
{
    /// <summary>Prompt tokens restored from a cache rather than prefilled again.</summary>
    [JsonPropertyName("cached_tokens")]
    public int CachedTokens { get; init; }
}
