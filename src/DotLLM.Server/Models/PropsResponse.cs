using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Response for <c>GET /props</c> — server configuration and model info.
/// </summary>
public sealed record PropsResponse
{
    [JsonPropertyName("model_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ModelId { get; init; }

    [JsonPropertyName("model_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ModelPath { get; init; }

    [JsonPropertyName("architecture")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Architecture { get; init; }

    [JsonPropertyName("num_layers")]
    public int NumLayers { get; init; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; init; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; init; }

    [JsonPropertyName("max_sequence_length")]
    public int MaxSequenceLength { get; init; }

    [JsonPropertyName("device")]
    public string Device { get; init; } = "cpu";

    /// <summary>Device actually used when <c>device</c> is <c>auto</c> (null otherwise).</summary>
    [JsonPropertyName("resolved_device")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ResolvedDevice { get; init; }

    [JsonPropertyName("gpu_layers")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? GpuLayers { get; init; }

    [JsonPropertyName("threads")]
    public int Threads { get; init; }

    [JsonPropertyName("sampling_defaults")]
    public required SamplingDefaultsDto SamplingDefaults { get; init; }

    [JsonPropertyName("draft_model_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? DraftModelPath { get; init; }

    /// <summary>
    /// True when MTP (Multi-Token Prediction) self-speculative decoding (issue #253) is actually engaging requests for the loaded
    /// model. Since #757 it is on by default for any checkpoint with an embedded MTP head; it is false for a model without a head,
    /// with <c>--no-mtp</c>, with an external draft model, or when <c>--expected-concurrency</c> routes to the batch scheduler.
    /// See <see cref="MtpStatus"/> for the reason.
    /// </summary>
    [JsonPropertyName("mtp_active")]
    public bool MtpActive { get; init; }

    /// <summary>Why MTP is or is not active: <c>active</c>, <c>off (--no-mtp)</c>, <c>unavailable (no MTP head)</c>, <c>skipped (...)</c>.</summary>
    [JsonPropertyName("mtp_status")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? MtpStatus { get; init; }

    /// <summary>Non-null when <c>--device auto</c> fell back to the CPU after a GPU load failed (#733).</summary>
    [JsonPropertyName("device_fallback_warning")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? DeviceFallbackWarning { get; init; }

    /// <summary>What the loaded model's chat template can do with the thinking controls (#1106); null when no template is loaded.</summary>
    [JsonPropertyName("reasoning")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ReasoningCapabilitiesDto? Reasoning { get; init; }

    [JsonPropertyName("is_ready")]
    public bool IsReady { get; init; }

    /// <summary>What the server is doing right now - <c>loading</c> / <c>busy</c> / <c>idle</c> / <c>no_model</c> (#1023); same object as <c>GET /v1/status</c>.</summary>
    [JsonPropertyName("activity")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ServerStatusResponse? Activity { get; init; }

    /// <summary>The server build's informational version (<c>0.3.0-dev.N+sha</c>) (#774). Nullable: no initializer on an init-only DTO property (STJ source-gen drops it).</summary>
    [JsonPropertyName("version")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Version { get; init; }
}

/// <summary>
/// Sampling parameter defaults (used in both <c>/props</c> and <c>/v1/config</c>).
/// </summary>
public sealed record SamplingDefaultsDto
{
    [JsonPropertyName("temperature")]
    public float Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float TopP { get; init; }

    [JsonPropertyName("top_k")]
    public int TopK { get; init; }

    [JsonPropertyName("min_p")]
    public float MinP { get; init; }

    [JsonPropertyName("repetition_penalty")]
    public float RepetitionPenalty { get; init; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; init; }

    [JsonPropertyName("seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Seed { get; init; }
}

/// <summary>
/// Which thinking controls the loaded model's chat template reads (<c>/props</c> <c>reasoning</c>, #1106). Detected from the
/// template source, so it says what the template can honour, not what the model will do.
/// </summary>
public sealed record ReasoningCapabilitiesDto
{
    /// <summary>The template reads <c>enable_thinking</c> (so the on/off switch works).</summary>
    [JsonPropertyName("enable_thinking")]
    public bool EnableThinking { get; init; }

    /// <summary>The template reads <c>reasoning_effort</c>.</summary>
    [JsonPropertyName("reasoning_effort")]
    public bool ReasoningEffort { get; init; }

    /// <summary>The model emits a think block the server can split and budget (<c>thinking_budget_tokens</c> applies).</summary>
    [JsonPropertyName("thinking_block")]
    public bool ThinkingBlock { get; init; }
}
