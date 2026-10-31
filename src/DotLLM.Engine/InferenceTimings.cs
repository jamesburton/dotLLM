namespace DotLLM.Engine;

/// <summary>
/// Timing measurements captured during inference (prefill, decode, sampling).
/// </summary>
public readonly record struct InferenceTimings
{
    /// <summary>Time spent on the prefill forward pass, in milliseconds.</summary>
    public double PrefillTimeMs { get; init; }

    /// <summary>Time spent on decode forward passes, in milliseconds.</summary>
    public double DecodeTimeMs { get; init; }

    /// <summary>Time spent on sampling, in milliseconds.</summary>
    public double SamplingTimeMs { get; init; }

    /// <summary>Number of prompt tokens processed during prefill.</summary>
    public int PrefillTokenCount { get; init; }

    /// <summary>Number of decode forward passes (generated tokens minus 1, since the first token comes from prefill).</summary>
    public int DecodeTokenCount { get; init; }

    /// <summary>Actual bytes allocated for the KV-cache (reflects quantization compression).</summary>
    public long KvCacheBytes { get; init; }

    /// <summary>Number of prompt tokens served from the prefix cache (skipped during prefill).</summary>
    public int CachedTokenCount { get; init; }

    /// <summary>Snapshot storage tier used by the prefix hit, or null for a miss.</summary>
    public string? CacheTier { get; init; }

    /// <summary>Wall time spent restoring a cached prefix, in milliseconds.</summary>
    public double CacheRestoreMilliseconds { get; init; }

    /// <summary>Total draft tokens proposed during speculative decoding. 0 when not using speculative decoding.</summary>
    public int SpeculativeDraftTokens { get; init; }

    /// <summary>
    /// Draft tokens that the target model accepted and that made it into the final output (#1243). Always
    /// <c>&lt;= <see cref="SpeculativeDraftTokens"/></c>. The one target-model token every verify round also emits
    /// (the bonus token when all drafts match, otherwise the correction) is NOT counted here; it is
    /// <see cref="SpeculativeSteps"/> tokens in total. Tokens discarded by stop conditions are excluded.
    /// </summary>
    public int SpeculativeAcceptedTokens { get; init; }

    /// <summary>Draft-and-verify rounds run during speculative decoding (#1288). 0 when not speculating.</summary>
    public int SpeculativeSteps { get; init; }

    /// <summary>
    /// Mean tokens emitted per speculative verify round: accepted drafts plus the round's one target-model token
    /// (bonus or correction), i.e. <c>(accepted + steps) / steps</c>; 0 when no speculation ran.
    /// </summary>
    public double SpeculativeMeanAcceptedLength => SpeculativeSteps > 0 ? (double)(SpeculativeAcceptedTokens + SpeculativeSteps) / SpeculativeSteps : 0;

    /// <summary>
    /// Draft acceptance rate: proportion of draft tokens that matched the target model.
    /// Clamped to [0, 1] — bonus tokens (produced when all K drafts are accepted) are not counted
    /// as "accepted drafts" since they come from the target model directly.
    /// </summary>
    public float SpeculativeAcceptanceRate => SpeculativeDraftTokens > 0
        ? Math.Min(1f, (float)SpeculativeAcceptedTokens / SpeculativeDraftTokens) : 0f;

    /// <summary>Prefill throughput in tokens per second (excludes cached tokens that were not re-processed).</summary>
    public double PrefillTokensPerSec => (PrefillTokenCount - CachedTokenCount) > 0 && PrefillTimeMs > 0
        ? (PrefillTokenCount - CachedTokenCount) / (PrefillTimeMs / 1000.0) : 0;

    /// <summary>Decode throughput in tokens per second.</summary>
    public double DecodeTokensPerSec => DecodeTokenCount > 0 && DecodeTimeMs > 0
        ? DecodeTokenCount / (DecodeTimeMs / 1000.0) : 0;
}
