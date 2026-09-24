using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// The body of an <see cref="ErrorResponse"/> — the OpenAI error object
/// (<c>{"message", "type", "param", "code"}</c>).
/// </summary>
/// <remarks>
/// #452: the server previously emitted a flat <c>{"error": "&lt;string&gt;"}</c>. Neither the
/// official OpenAI SDK nor the Anthropic SDK can read <c>.code</c>/<c>.type</c>/<c>.param</c> off
/// that shape, and third-party clients that parse the envelope to classify failures got nothing
/// usable. This is the shape both SDKs expect.
/// </remarks>
public sealed record ErrorDetail
{
    /// <summary>Human-readable description of the failure.</summary>
    [JsonPropertyName("message")]
    public required string Message { get; init; }

    /// <summary>
    /// Error class. OpenAI uses <c>invalid_request_error</c>, <c>rate_limit_error</c>,
    /// <c>authentication_error</c>, <c>not_found_error</c>, <c>api_error</c>. These names are
    /// shared with Anthropic's envelope, so #448 can reuse this type rather than fork one.
    /// </summary>
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    /// <summary>
    /// The request field that caused the failure, when one can be named (e.g. <c>"model"</c>).
    /// Serialized as an explicit <c>null</c> when unknown, matching OpenAI.
    /// </summary>
    [JsonPropertyName("param")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? Param { get; init; }

    /// <summary>
    /// Machine-readable code (e.g. <c>model_not_found</c>). Serialized as an explicit
    /// <c>null</c> when unknown, matching OpenAI.
    /// </summary>
    [JsonPropertyName("code")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? Code { get; init; }
}

/// <summary>
/// Standard error response DTO — an SDK-shaped <c>{"error": {...}}</c> envelope.
/// </summary>
/// <remarks>
/// The top-level <c>"type": "error"</c> discriminator is what Anthropic's envelope requires and is
/// ignored by OpenAI clients, so a single envelope serves both surfaces.
/// Construct through the factories (<see cref="InvalidRequest"/>, <see cref="NotFound"/>,
/// <see cref="RateLimit"/>, <see cref="Internal"/>) so every call site picks a deliberate
/// <see cref="ErrorDetail.Type"/>.
/// </remarks>
public sealed record ErrorResponse
{
    /// <summary>Envelope discriminator. Always <c>"error"</c>.</summary>
    [JsonPropertyName("type")]
    public string Type { get; init; } = "error";

    [JsonPropertyName("error")]
    public required ErrorDetail Error { get; init; }

    /// <summary>Error types used by this server.</summary>
    public const string InvalidRequestType = "invalid_request_error";

    /// <inheritdoc cref="InvalidRequestType"/>
    public const string RateLimitType = "rate_limit_error";

    /// <inheritdoc cref="InvalidRequestType"/>
    public const string NotFoundType = "not_found_error";

    /// <inheritdoc cref="InvalidRequestType"/>
    public const string ApiErrorType = "api_error";

    /// <summary>A malformed or unacceptable request (HTTP 400/409/422).</summary>
    public static ErrorResponse InvalidRequest(string message, string? param = null, string? code = null) =>
        new() { Error = new ErrorDetail { Message = message, Type = InvalidRequestType, Param = param, Code = code } };

    /// <summary>A referenced resource does not exist (HTTP 404).</summary>
    public static ErrorResponse NotFound(string message, string? param = null, string? code = null) =>
        new() { Error = new ErrorDetail { Message = message, Type = NotFoundType, Param = param, Code = code } };

    /// <summary>The caller exceeded a configured rate limit (HTTP 429).</summary>
    public static ErrorResponse RateLimit(string message, string? code = null) =>
        new() { Error = new ErrorDetail { Message = message, Type = RateLimitType, Param = null, Code = code } };

    /// <summary>A server-side failure or unavailability (HTTP 5xx).</summary>
    public static ErrorResponse Internal(string message, string? code = null) =>
        new() { Error = new ErrorDetail { Message = message, Type = ApiErrorType, Param = null, Code = code } };
}

/// <summary>
/// Standard status response DTO. Replaces anonymous <c>new { status = "..." }</c> types
/// for AOT-compatible source-generated serialization.
/// </summary>
public sealed record StatusResponse
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }
}
