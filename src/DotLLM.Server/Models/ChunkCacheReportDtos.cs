using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// The response-level <c>chunk_cache</c> object (spec §5.9; contract <c>response-*.json</c>): what the server did with a
/// request's chunk annotations. Present only with <c>--chunk-cache</c> on and an annotated request; written in blocking
/// responses, the final streaming chunk and the Anthropic <c>message_delta</c>. Counts only: never chunk text, hashes or
/// the namespace.
/// </summary>
public sealed record ChunkCacheReportDto
{
    /// <summary>Convention version (1).</summary>
    [JsonPropertyName("v")]
    public int V { get; init; }

    /// <summary>The requested policy.</summary>
    [JsonPropertyName("policy_requested")]
    public required string PolicyRequested { get; init; }

    /// <summary>The applied policy (always <c>exact</c> in v1).</summary>
    [JsonPropertyName("policy_applied")]
    public required string PolicyApplied { get; init; }

    /// <summary>Why the applied policy differs from the requested one; null (written) when they agree.</summary>
    [JsonPropertyName("reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? Reason { get; init; }

    /// <summary>Chunk counts: <c>annotated = prefix + hit + miss + ignored</c>.</summary>
    [JsonPropertyName("chunks")]
    public required ChunkCacheChunksDto Chunks { get; init; }

    /// <summary>Actual token partition: <c>prompt = prefix_reused + prefix_reused_approx + chunk_reused + prefilled</c>.</summary>
    [JsonPropertyName("tokens")]
    public required ChunkCacheTokensDto Tokens { get; init; }

    /// <summary>Under <c>shadow</c>: the reuse plan, computed and not applied.</summary>
    [JsonPropertyName("plan")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChunkCachePlanDto? Plan { get; init; }

    /// <summary>Time per phase in milliseconds.</summary>
    [JsonPropertyName("timings_ms")]
    public required ChunkCacheTimingsDto TimingsMs { get; init; }

    /// <summary>Warning codes with counts, in order of first occurrence.</summary>
    [JsonPropertyName("warnings")]
    public required ChunkCacheWarningDto[] Warnings { get; init; }
}

/// <summary>Chunk counts of a <see cref="ChunkCacheReportDto"/>.</summary>
public sealed record ChunkCacheChunksDto
{
    /// <summary>Annotations in the request (accepted or not).</summary>
    [JsonPropertyName("annotated")]
    public int Annotated { get; init; }

    /// <summary>Chunks wholly inside the exact prefix the prompt cache restored.</summary>
    [JsonPropertyName("prefix")]
    public int Prefix { get; init; }

    /// <summary>Chunks found in the store (whether or not reuse was applied).</summary>
    [JsonPropertyName("hit")]
    public int Hit { get; init; }

    /// <summary>Chunks not found in the store.</summary>
    [JsonPropertyName("miss")]
    public int Miss { get; init; }

    /// <summary>Annotations treated as plain text (dropped, altered, too short, unverified, unsupported).</summary>
    [JsonPropertyName("ignored")]
    public int Ignored { get; init; }
}

/// <summary>Token partition of a <see cref="ChunkCacheReportDto"/> (actual) or of its <see cref="ChunkCachePlanDto"/> (predicted).</summary>
public sealed record ChunkCacheTokensDto
{
    /// <summary>Prompt tokens.</summary>
    [JsonPropertyName("prompt")]
    public int Prompt { get; init; }

    /// <summary>Exact rows restored from the prompt cache (= <c>cached_tokens</c>).</summary>
    [JsonPropertyName("prefix_reused")]
    public int PrefixReused { get; init; }

    /// <summary>Approximate rows restored from a prompt-cache snapshot (0 under <c>exact</c> / <c>shadow</c>).</summary>
    [JsonPropertyName("prefix_reused_approx")]
    public int PrefixReusedApprox { get; init; }

    /// <summary>Rows taken from stored chunks.</summary>
    [JsonPropertyName("chunk_reused")]
    public int ChunkReused { get; init; }

    /// <summary>Reused rows recomputed (a subset of <see cref="ChunkReused"/>).</summary>
    [JsonPropertyName("chunk_recomputed")]
    public int ChunkRecomputed { get; init; }

    /// <summary>Rows prefilled as ordinary text.</summary>
    [JsonPropertyName("prefilled")]
    public int Prefilled { get; init; }

    /// <summary>Chunk tokens queued for background encode.</summary>
    [JsonPropertyName("encode_queued")]
    public int EncodeQueued { get; init; }
}

/// <summary>The <c>shadow</c> plan (contract ruling 7): what applied reuse would have done, with its own partition.</summary>
public sealed record ChunkCachePlanDto
{
    /// <summary>Recompute selector the plan assumes.</summary>
    [JsonPropertyName("selector")]
    public required string Selector { get; init; }

    /// <summary>Recompute budget the plan assumes (fraction of each reused chunk's rows).</summary>
    [JsonPropertyName("budget")]
    public double Budget { get; init; }

    /// <summary>Predicted partition.</summary>
    [JsonPropertyName("tokens")]
    public required ChunkCacheTokensDto Tokens { get; init; }

    /// <summary>Predicted time to first token with the plan applied (cost model), ms.</summary>
    [JsonPropertyName("predicted_ttft_ms")]
    public long PredictedTtftMs { get; init; }

    /// <summary>Predicted time to prefill the same rows without chunk reuse, ms.</summary>
    [JsonPropertyName("full_prefill_estimate_ms")]
    public long FullPrefillEstimateMs { get; init; }
}

/// <summary>Phase timings of a <see cref="ChunkCacheReportDto"/>, ms (assembly, probe and recompute are 0 under <c>exact</c>).</summary>
public sealed record ChunkCacheTimingsDto
{
    /// <summary>Chunk assembly.</summary>
    [JsonPropertyName("assembly")]
    public long Assembly { get; init; }

    /// <summary>Query-relevance probe.</summary>
    [JsonPropertyName("probe")]
    public long Probe { get; init; }

    /// <summary>Selective recompute.</summary>
    [JsonPropertyName("recompute")]
    public long Recompute { get; init; }

    /// <summary>Ordinary prefill (measured).</summary>
    [JsonPropertyName("prefill")]
    public long Prefill { get; init; }

    /// <summary>Predicted prefill of the non-prefix rows without chunk reuse.</summary>
    [JsonPropertyName("full_prefill_estimate")]
    public long FullPrefillEstimate { get; init; }
}

/// <summary>A warning code and how many annotations (or requests, for request-level codes) it covers.</summary>
public sealed record ChunkCacheWarningDto
{
    /// <summary>The §5.8 code.</summary>
    [JsonPropertyName("code")]
    public required string Code { get; init; }

    /// <summary>How many.</summary>
    [JsonPropertyName("count")]
    public int Count { get; init; }
}
