using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// <c>GET /v1/status</c> (and <c>/props</c> <c>activity</c>): what the server is doing right now, so a client can tell
/// "loading X since T" from "busy with a request" from "idle" (#1023).
/// </summary>
public sealed record ServerStatusResponse
{
    /// <summary>
    /// Where each tuned setting came from (<c>flag</c>, <c>auto</c> or <c>default</c>) when the server was started with <c>--auto</c> (#1902);
    /// omitted otherwise. Nullable with no initializer, per the source-generation rule in CLAUDE.md.
    /// </summary>
    [JsonPropertyName("settings_source")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyDictionary<string, string>? SettingsSource { get; init; }

    /// <summary>
    /// <c>loading</c> (a model load is in progress - wins over busy, since a load holds the lock), <c>busy</c> (a request holds the
    /// execution lock), <c>idle</c> (a model is ready and nothing holds the lock), or <c>no_model</c>.
    /// </summary>
    [JsonPropertyName("state")]
    public required string State { get; init; }

    /// <summary>True when a model is loaded and ready (false for the whole of any load).</summary>
    [JsonPropertyName("is_ready")]
    public bool IsReady { get; init; }

    /// <summary>The active model's id when ready; null otherwise.</summary>
    [JsonPropertyName("model_id")]
    public string? ModelId { get; init; }

    /// <summary>The model being loaded, when <see cref="State"/> is <c>loading</c>.</summary>
    [JsonPropertyName("loading_model_id")]
    public string? LoadingModelId { get; init; }

    /// <summary>When the current load started (UTC, ISO 8601).</summary>
    [JsonPropertyName("loading_since")]
    public DateTimeOffset? LoadingSince { get; init; }

    /// <summary>Seconds the current load has been running.</summary>
    [JsonPropertyName("loading_elapsed_seconds")]
    public double? LoadingElapsedSeconds { get; init; }

    /// <summary>The request that triggered the load (<c>req#12</c>), when it came from one.</summary>
    [JsonPropertyName("loading_request")]
    public string? LoadingRequest { get; init; }

    /// <summary>Who holds the execution lock (<c>req#3 generate</c>, <c>req#4 model activation</c>, <c>idle sweep</c>); null when free.</summary>
    [JsonPropertyName("busy_with")]
    public string? BusyWith { get; init; }

    /// <summary>When the lock holder acquired it (UTC, ISO 8601).</summary>
    [JsonPropertyName("busy_since")]
    public DateTimeOffset? BusySince { get; init; }

    /// <summary>Seconds the lock has been held.</summary>
    [JsonPropertyName("busy_elapsed_seconds")]
    public double? BusyElapsedSeconds { get; init; }

    /// <summary>Requests waiting for the execution lock.</summary>
    [JsonPropertyName("waiting_requests")]
    public int WaitingRequests { get; init; }

    /// <summary>The device the active model runs on (<c>cpu</c>, <c>vulkan</c>, <c>gpu:0</c>; the resolved one for <c>auto</c>) (#1822). Null when no model is ready.</summary>
    [JsonPropertyName("device")]
    public string? Device { get; init; }

    /// <summary>
    /// The served context window in tokens (#1836): after <c>--ctx-size</c> and any fit to free device memory or the GPU paged pool,
    /// the same figure as <c>/props</c> and <c>/v1/models</c>. Null when no model is loaded.
    /// </summary>
    [JsonPropertyName("n_ctx")]
    public int? NCtx { get; init; }

    /// <summary>True when <c>--device auto</c> fell back to the CPU after a faster device failed to load the model (#1822). Null when no model is ready.</summary>
    [JsonPropertyName("device_fallback")]
    public bool? DeviceFallback { get; init; }

    /// <summary>Why it fell back (#1822); null when it did not.</summary>
    [JsonPropertyName("device_fallback_reason")]
    public string? DeviceFallbackReason { get; init; }

    /// <summary>The oldest in-flight request's computed prompt tokens finished so far (#1822); null when nothing is in flight.</summary>
    [JsonPropertyName("prefill_tokens_done")]
    public int? PrefillTokensDone { get; init; }

    /// <summary>The oldest in-flight request's prompt tokens to compute (prefix-cache hits excluded) (#1822).</summary>
    [JsonPropertyName("prefill_tokens_total")]
    public int? PrefillTokensTotal { get; init; }

    /// <summary>The oldest in-flight request's decoded tokens so far (#1822).</summary>
    [JsonPropertyName("decode_tokens")]
    public int? DecodeTokens { get; init; }

    /// <summary>Every in-flight request's progress, oldest first (#1822); null when nothing is in flight.</summary>
    [JsonPropertyName("requests")]
    public InFlightRequestStatus[]? Requests { get; init; }
}

/// <summary>One in-flight request's live progress in <c>GET /v1/status</c> (#1822).</summary>
public sealed record InFlightRequestStatus
{
    /// <summary>The request's log tag (<c>req#12</c>), when it has one.</summary>
    [JsonPropertyName("request")]
    public string? Request { get; init; }

    /// <summary><c>waiting</c> (admitted, no prefill yet), <c>prefill</c>, or <c>decode</c>.</summary>
    [JsonPropertyName("phase")]
    public string? Phase { get; init; }

    /// <summary>Seconds since the request started running.</summary>
    [JsonPropertyName("elapsed_seconds")]
    public double? ElapsedSeconds { get; init; }

    /// <summary>Tokens in the prompt, prefix-cache hits included.</summary>
    [JsonPropertyName("prompt_tokens")]
    public int? PromptTokens { get; init; }

    /// <summary>Prompt tokens reused from the prefix cache.</summary>
    [JsonPropertyName("cached_tokens")]
    public int? CachedTokens { get; init; }

    /// <summary>Computed prompt tokens finished so far (per layer inside a CPU forward, per chunk otherwise).</summary>
    [JsonPropertyName("prefill_tokens_done")]
    public int? PrefillTokensDone { get; init; }

    /// <summary>Prompt tokens this prefill computes (prefix-cache hits excluded).</summary>
    [JsonPropertyName("prefill_tokens_total")]
    public int? PrefillTokensTotal { get; init; }

    /// <summary>Tokens decoded so far.</summary>
    [JsonPropertyName("decode_tokens")]
    public int? DecodeTokens { get; init; }
}
