using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// POST /v1/prompt-cache/{id} body — register a named prefix.
/// </summary>
public sealed record PromptCacheRegisterRequest
{
    /// <summary>Raw text prompt to register. Either this or <see cref="TokenIds"/> must be set.</summary>
    [JsonPropertyName("prompt")]
    public string? Prompt { get; init; }

    /// <summary>Pre-tokenised prompt (alternative to <see cref="Prompt"/>).</summary>
    [JsonPropertyName("token_ids")]
    public int[]? TokenIds { get; init; }
}

/// <summary>
/// Response describing a named prefix entry.
/// </summary>
public sealed record PromptCacheResponse
{
    [JsonPropertyName("prefix_id")]
    public required string PrefixId { get; init; }

    [JsonPropertyName("tokens")]
    public required int Tokens { get; init; }

    [JsonPropertyName("blocks")]
    public required int Blocks { get; init; }

    [JsonPropertyName("status")]
    public required string Status { get; init; }
}

/// <summary>
/// Summary of the cross-request prefix cache returned by GET /v1/prompt-cache.
/// </summary>
public sealed record PromptCacheStatsResponse
{
    /// <summary>Independent recurrent-prefix snapshots, including CUDA models without a paged trie.</summary>
    [JsonPropertyName("recurrent")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public DotLLM.Engine.PromptCache.RecurrentPrefixCacheStatistics? Recurrent { get; init; }

    /// <summary>The live-KV prefix cache (attention models without a paged trie: Vulkan, CUDA, <c>--no-paged</c>), when present (#1654).</summary>
    [JsonPropertyName("live")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LivePrefixCacheStats? Live { get; init; }

    /// <summary>Retention policy in force for every prefix cache of this server (#1654): <c>lru</c> or <c>agent</c>.</summary>
    [JsonPropertyName("policy")]
    public string Policy { get; init; } = "lru";

    [JsonPropertyName("enabled")]
    public required bool Enabled { get; init; }

    [JsonPropertyName("block_size")]
    public required int BlockSize { get; init; }

    [JsonPropertyName("nodes")]
    public required int Nodes { get; init; }

    [JsonPropertyName("hit_tokens")]
    public required long HitTokens { get; init; }

    [JsonPropertyName("miss_tokens")]
    public required long MissTokens { get; init; }

    [JsonPropertyName("lookups")]
    public required long Lookups { get; init; }

    [JsonPropertyName("hits")]
    public required long Hits { get; init; }

    [JsonPropertyName("misses")]
    public required long Misses { get; init; }

    [JsonPropertyName("eviction_refusals")]
    public required long EvictionRefusals { get; init; }

    [JsonPropertyName("free_blocks")]
    public required int FreeBlocks { get; init; }

    [JsonPropertyName("total_blocks")]
    public required int TotalBlocks { get; init; }
}

/// <summary>Live-KV prefix cache counters (<see cref="DotLLM.Engine.PromptCache.PrefixCache"/>), for GET /v1/prompt-cache (#1654).</summary>
public sealed record LivePrefixCacheStats
{
    [JsonPropertyName("entries")]
    public required int Entries { get; init; }

    [JsonPropertyName("hits")]
    public required long Hits { get; init; }

    [JsonPropertyName("misses")]
    public required long Misses { get; init; }

    /// <summary>Lookups the agent-aware policy made miss to keep another conversation's unexpired entry.</summary>
    [JsonPropertyName("protected_misses")]
    public required long ProtectedMisses { get; init; }

    /// <summary>Prompt tokens looked up so far: the clock the own-length rule is measured on.</summary>
    [JsonPropertyName("token_clock")]
    public required long TokenClock { get; init; }
}
