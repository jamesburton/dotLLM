using System.Diagnostics.Metrics;
using DotLLM.Server.Models;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// <c>/metrics</c> counters of the <c>chunk_cache</c> convention: <c>dotllm_chunk_cache_chunks_total{status}</c> (status =
/// <c>prefix</c>, <c>hit</c>, <c>miss</c>, <c>ignored</c>) and <c>dotllm_chunk_cache_tokens_total{kind}</c> (kind = each key of
/// the report's <c>tokens</c> partition). Labels carry no namespace, text or hash (spec §10).
/// </summary>
internal static class ChunkCacheMetrics
{
    /// <summary>Meter name (collected by the <c>DotLLM.*</c> Prometheus collector).</summary>
    public const string MeterName = "DotLLM.Server.ChunkCache";

    private static readonly Meter Meter = new(MeterName);

    private static readonly Counter<long> Chunks = Meter.CreateCounter<long>(
        "dotllm.chunk_cache.chunks", unit: "{chunk}", description: "Annotated chunks by outcome (prefix, hit, miss, ignored).");

    private static readonly Counter<long> Tokens = Meter.CreateCounter<long>(
        "dotllm.chunk_cache.tokens", unit: "{token}", description: "Prompt tokens of chunk-annotated requests by kind (the chunk_cache tokens partition).");

    /// <summary>Adds one annotated request's report to the counters.</summary>
    /// <param name="chunks">The report's chunk counts.</param>
    /// <param name="tokens">The report's actual token partition.</param>
    public static void Record(ChunkCacheChunksDto chunks, ChunkCacheTokensDto tokens)
    {
        Chunks.Add(chunks.Prefix, new KeyValuePair<string, object?>("status", "prefix"));
        Chunks.Add(chunks.Hit, new KeyValuePair<string, object?>("status", "hit"));
        Chunks.Add(chunks.Miss, new KeyValuePair<string, object?>("status", "miss"));
        Chunks.Add(chunks.Ignored, new KeyValuePair<string, object?>("status", "ignored"));

        Tokens.Add(tokens.Prompt, new KeyValuePair<string, object?>("kind", "prompt"));
        Tokens.Add(tokens.PrefixReused, new KeyValuePair<string, object?>("kind", "prefix_reused"));
        Tokens.Add(tokens.PrefixReusedApprox, new KeyValuePair<string, object?>("kind", "prefix_reused_approx"));
        Tokens.Add(tokens.ChunkReused, new KeyValuePair<string, object?>("kind", "chunk_reused"));
        Tokens.Add(tokens.ChunkRecomputed, new KeyValuePair<string, object?>("kind", "chunk_recomputed"));
        Tokens.Add(tokens.Prefilled, new KeyValuePair<string, object?>("kind", "prefilled"));
        Tokens.Add(tokens.EncodeQueued, new KeyValuePair<string, object?>("kind", "encode_queued"));
    }
}
