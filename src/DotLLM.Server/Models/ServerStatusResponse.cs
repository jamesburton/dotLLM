using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// <c>GET /v1/status</c> (and <c>/props</c> <c>activity</c>): what the server is doing right now, so a client can tell
/// "loading X since T" from "busy with a request" from "idle" (#1023).
/// </summary>
public sealed record ServerStatusResponse
{
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
}
