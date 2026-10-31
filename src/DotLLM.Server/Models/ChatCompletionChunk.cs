using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible streaming chat completion chunk (SSE).
/// </summary>
public sealed record ChatCompletionChunk
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required ChatChunkChoiceDto[] Choices { get; init; }

    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public UsageDto? Usage { get; init; }

    [JsonPropertyName("timings")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public TimingsDto? Timings { get; init; }

}

/// <summary>
/// Inference timing details (dotLLM extension, not in OpenAI spec).
/// </summary>
public sealed record TimingsDto
{
    [JsonPropertyName("prefill_time_ms")]
    public double PrefillTimeMs { get; init; }

    [JsonPropertyName("decode_time_ms")]
    public double DecodeTimeMs { get; init; }

    [JsonPropertyName("sampling_time_ms")]
    public double SamplingTimeMs { get; init; }

    [JsonPropertyName("prefill_tokens_per_sec")]
    public double PrefillTokensPerSec { get; init; }

    [JsonPropertyName("decode_tokens_per_sec")]
    public double DecodeTokensPerSec { get; init; }

    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("generated_tokens")]
    public int GeneratedTokens { get; init; }

    [JsonPropertyName("cached_tokens")]
    public int CachedTokens { get; init; }

    /// <summary>Storage tier used to restore the cached prefix.</summary>
    [JsonPropertyName("cache_tier")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CacheTier { get; init; }

    /// <summary>Time spent restoring the cached prefix, in milliseconds.</summary>
    [JsonPropertyName("cache_restore_ms")]
    public double CacheRestoreMilliseconds { get; init; }

    [JsonPropertyName("speculative_draft_tokens")]
    public int SpeculativeDraftTokens { get; init; }

    [JsonPropertyName("speculative_accepted_tokens")]
    public int SpeculativeAcceptedTokens { get; init; }

    [JsonPropertyName("speculative_acceptance_rate")]
    public float SpeculativeAcceptanceRate { get; init; }

    // llama.cpp `timings` field names (#1345), so llama-swap and other proxies that read them record real speeds.
    // Added beside the dotLLM names above, not replacing them.

    /// <summary>Prompt tokens reused from the prefix cache (<c>cache_n</c>).</summary>
    [JsonPropertyName("cache_n")]
    public int CacheN { get; init; }

    /// <summary>Prompt tokens actually prefilled, excluding cached ones (<c>prompt_n</c>).</summary>
    [JsonPropertyName("prompt_n")]
    public int PromptN { get; init; }

    /// <summary>Prefill time in ms (<c>prompt_ms</c>).</summary>
    [JsonPropertyName("prompt_ms")]
    public double PromptMs { get; init; }

    /// <summary>Prefill ms per prefilled token; 0 when none were prefilled.</summary>
    [JsonPropertyName("prompt_per_token_ms")]
    public double PromptPerTokenMs { get; init; }

    /// <summary>Prefill tokens per second.</summary>
    [JsonPropertyName("prompt_per_second")]
    public double PromptPerSecond { get; init; }

    /// <summary>Tokens produced by decode passes (<c>predicted_n</c>); the first generated token comes from prefill and is not counted.</summary>
    [JsonPropertyName("predicted_n")]
    public int PredictedN { get; init; }

    /// <summary>Decode time in ms (<c>predicted_ms</c>).</summary>
    [JsonPropertyName("predicted_ms")]
    public double PredictedMs { get; init; }

    /// <summary>Decode ms per token; 0 when no decode pass ran.</summary>
    [JsonPropertyName("predicted_per_token_ms")]
    public double PredictedPerTokenMs { get; init; }

    /// <summary>Decode tokens per second.</summary>
    [JsonPropertyName("predicted_per_second")]
    public double PredictedPerSecond { get; init; }

    /// <summary>Maps engine measurements to the timing extension (dotLLM names plus the llama.cpp-shaped ones).</summary>
    public static TimingsDto From(DotLLM.Engine.InferenceTimings timings) => new()
    {
        CacheN = timings.CachedTokenCount,
        PromptN = Math.Max(0, timings.PrefillTokenCount - timings.CachedTokenCount),
        PromptMs = timings.PrefillTimeMs,
        PromptPerTokenMs = timings.PrefillTokenCount - timings.CachedTokenCount > 0
            ? timings.PrefillTimeMs / (timings.PrefillTokenCount - timings.CachedTokenCount) : 0,
        PromptPerSecond = timings.PrefillTokensPerSec,
        PredictedN = timings.DecodeTokenCount,
        PredictedMs = timings.DecodeTimeMs,
        PredictedPerTokenMs = timings.DecodeTokenCount > 0 ? timings.DecodeTimeMs / timings.DecodeTokenCount : 0,
        PredictedPerSecond = timings.DecodeTokensPerSec,
        PrefillTimeMs = timings.PrefillTimeMs,
        DecodeTimeMs = timings.DecodeTimeMs,
        SamplingTimeMs = timings.SamplingTimeMs,
        PrefillTokensPerSec = timings.PrefillTokensPerSec,
        DecodeTokensPerSec = timings.DecodeTokensPerSec,
        PromptTokens = timings.PrefillTokenCount,
        GeneratedTokens = timings.DecodeTokenCount,
        CachedTokens = timings.CachedTokenCount,
        CacheTier = timings.CacheTier,
        CacheRestoreMilliseconds = timings.CacheRestoreMilliseconds,
        SpeculativeDraftTokens = timings.SpeculativeDraftTokens,
        SpeculativeAcceptedTokens = timings.SpeculativeAcceptedTokens,
        SpeculativeAcceptanceRate = timings.SpeculativeAcceptanceRate,
    };
}

/// <summary>
/// A single choice in a streaming chunk.
/// </summary>
public sealed record ChatChunkChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("delta")]
    public required ChatDeltaDto Delta { get; init; }

    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsDto? Logprobs { get; init; }

    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; init; }
}

/// <summary>
/// Incremental message delta in a streaming chunk.
/// </summary>
public sealed record ChatDeltaDto
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; init; }

    /// <summary>Incremental reasoning text (#767), streamed separately from <see cref="Content"/>.</summary>
    [JsonPropertyName("reasoning_content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ReasoningContent { get; init; }

    [JsonPropertyName("tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StreamingToolCallDto[]? ToolCalls { get; init; }
}

/// <summary>
/// A streamed tool call, identified by its position in the assistant's tool-call list.
/// Unlike a non-streaming tool call, OpenAI clients require this index to assemble deltas.
/// </summary>
public sealed record StreamingToolCallDto
{
    [JsonPropertyName("index")]
    public required int Index { get; init; }

    /// <summary>Call id. Present on the first delta of a call only (#1342); absent on argument continuations.</summary>
    [JsonPropertyName("id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Id { get; init; }

    /// <summary><c>function</c> on the first delta of a call; absent on continuations, as OpenAI sends them.</summary>
    [JsonPropertyName("type")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Type => Id is null ? null : "function";

    [JsonPropertyName("function")]
    public required StreamingToolCallFunctionDto Function { get; init; }
}

/// <summary>The <c>function</c> part of a streamed tool-call delta: <c>name</c> on the first delta, <c>arguments</c> fragments after.</summary>
public sealed record StreamingToolCallFunctionDto
{
    [JsonPropertyName("name")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Name { get; init; }

    [JsonPropertyName("arguments")]
    public required string Arguments { get; init; }
}
