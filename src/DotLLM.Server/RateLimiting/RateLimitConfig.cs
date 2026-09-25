using System.Collections.Generic;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// Per-API-key rate-limiting configuration. Loaded from <c>appsettings.json</c> /
/// the server options at startup. When <see cref="Enabled"/> is <c>false</c> (or
/// the whole config is omitted) the middleware is a pass-through and incurs no
/// allocation or limiter cost.
/// </summary>
public sealed record RateLimitConfig
{
    /// <summary>Master switch. When <c>false</c> the middleware short-circuits.</summary>
    public bool Enabled { get; init; }

    /// <summary>
    /// Policy applied to any request whose resolved API key is not present in
    /// <see cref="ApiKeys"/>. When null, unknown keys are unlimited
    /// (still subject to other middleware).
    /// </summary>
    public RateLimitPolicy? DefaultPolicy { get; init; }

    /// <summary>
    /// Per-API-key overrides. Key = the literal API key string the resolver
    /// returns (e.g. the value of <c>X-API-Key</c>). Lookup is ordinal,
    /// case-sensitive — API keys are opaque tokens.
    /// </summary>
    public IReadOnlyDictionary<string, RateLimitPolicy> ApiKeys { get; init; }
        = new Dictionary<string, RateLimitPolicy>(StringComparer.Ordinal);

    /// <summary>
    /// Approximate completion-token cost charged when a request's
    /// <c>max_tokens</c> is unspecified. Used so callers cannot bypass the
    /// tokens/min limit by omitting <c>max_tokens</c>. Refunded after
    /// generation via true-up.
    /// </summary>
    public int EstimatedCompletionTokensFallback { get; init; } = 256;

    /// <summary>
    /// Returns the policy for the resolved API key, falling back to the
    /// default policy. Returns <c>null</c> when no policy applies — caller
    /// should treat that as "unlimited" / pass-through.
    /// </summary>
    public RateLimitPolicy? PolicyFor(string apiKey) =>
        ApiKeys.TryGetValue(apiKey, out var p) ? p : DefaultPolicy;
}

/// <summary>
/// Concrete rate-limit policy. All three caps are independent — the most
/// restrictive triggers a 429.
/// </summary>
public sealed record RateLimitPolicy
{
    /// <summary>
    /// Maximum HTTP requests admitted per 60-second window for this key.
    /// 0 or negative = disabled (no per-request cap).
    /// </summary>
    public int RequestsPerMinute { get; init; }

    /// <summary>
    /// Maximum tokens (prompt + completion) consumed per 60-second window.
    /// Prompt tokens are counted at admission; completion tokens use the
    /// request's <c>max_tokens</c> upfront and are trued-up against the
    /// actual generated count after the response. 0 or negative = disabled.
    /// </summary>
    public int TokensPerMinute { get; init; }

    /// <summary>
    /// Maximum simultaneous in-flight requests for this key. Excess
    /// requests are queued in priority order (see <see cref="Priority"/>).
    /// 0 or negative = disabled.
    /// </summary>
    public int MaxConcurrent { get; init; }

    /// <summary>
    /// Priority level applied to admission queueing. When concurrency is
    /// saturated, higher-priority requests jump ahead of lower-priority
    /// waiters. Does not affect requests/min or tokens/min token-bucket
    /// scheduling — those are per-key isolation.
    /// </summary>
    public RequestPriority Priority { get; init; } = RequestPriority.Normal;

    /// <summary>
    /// Max time a request may wait in the priority concurrency queue
    /// before being rejected with 429. Default 5s.
    /// </summary>
    public TimeSpan QueueTimeout { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Per-API-key <b>fairness weight</b> for the continuous-batch scheduler's start-time
    /// fair-queuing (SFQ) admission. A key with weight <c>w</c> is charged <c>cost / w</c> into
    /// its SFQ finish tag, so a higher-weight key receives a proportionally larger share of
    /// admissions within its priority tier under contention. Only consulted when scheduler
    /// fairness is enabled. Default <c>1.0</c> (equal share); values ≤ 0 are treated as 1.0.
    /// </summary>
    public double Weight { get; init; } = 1.0;
}

/// <summary>
/// Request priority. Higher value = higher priority. The numeric values
/// are stable and may be used as queue keys.
/// </summary>
public enum RequestPriority
{
    /// <summary>Background / batch traffic. Yields to all other tiers.</summary>
    Low = 0,
    /// <summary>Default tier.</summary>
    Normal = 1,
    /// <summary>Latency-sensitive traffic.</summary>
    High = 2,
    /// <summary>Reserved for operator / health-check traffic.</summary>
    Critical = 3,
}
