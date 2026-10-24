using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible model list response for GET /v1/models.
/// </summary>
public sealed record ModelListResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    [JsonPropertyName("data")]
    public required ModelInfoDto[] Data { get; init; }
}

/// <summary>
/// Individual model information.
/// </summary>
public sealed record ModelInfoDto
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; init; } = "dotllm";

    /// <summary>
    /// Whether this is the model directly wired to <see cref="ServerState"/>'s live inference
    /// path right now, versus resident-but-inactive (#369). Always <c>true</c> for the single
    /// model reported by pre-#369 callers (default <c>MaxResidentModels = 1</c>).
    /// </summary>
    [JsonPropertyName("is_active")]
    public bool IsActive { get; init; } = true;

    /// <summary>Seconds since this model last served (or was activated for) a request (#369).</summary>
    [JsonPropertyName("idle_seconds")]
    public double IdleSeconds { get; init; }

    /// <summary>Effective keep-alive for this model, in seconds. Negative = never auto-unload (#369).</summary>
    [JsonPropertyName("keep_alive_seconds")]
    public double KeepAliveSeconds { get; init; }

    /// <summary>Seconds until auto-unload, or null when the keep-alive never expires (#369).</summary>
    [JsonPropertyName("expires_in_seconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ExpiresInSeconds { get; init; }

    /// <summary>Approximate resident footprint in bytes (#369).</summary>
    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }
}
