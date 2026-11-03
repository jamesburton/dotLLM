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

    private static readonly Counter<long> StoreWrites = Meter.CreateCounter<long>(
        "dotllm.chunk_store.writes", unit: "{entry}", description: "Sole-writer entry writes by result (stored, failed, too_large).");

    private static readonly Counter<long> StoreEvictions = Meter.CreateCounter<long>(
        "dotllm.chunk_store.evictions", unit: "{entry}", description: "Entry files deleted by the sole writer's budget eviction.");

    private static readonly Counter<long> StoreSkips = Meter.CreateCounter<long>(
        "dotllm.chunk_store.skips", unit: "{chunk}", description: "Misses refused for a background encode, by admission reason.");

    private static Func<int>? s_queueDepth;

    private static readonly ObservableGauge<int> QueueDepth = Meter.CreateObservableGauge(
        "dotllm.chunk_store.queue_depth", () => s_queueDepth?.Invoke() ?? 0, unit: "{item}",
        description: "Background encodes queued or running.");

    /// <summary>The fixed label set of <c>dotllm_chunk_store_skips_total{reason}</c> (an unknown reason is counted as <c>other</c>).</summary>
    private static readonly HashSet<string> SkipReasons = new(StringComparer.Ordinal)
    {
        AdmissionReasons.NoStore, AdmissionReasons.Volatile, AdmissionReasons.DeniedPath, AdmissionReasons.SourceUnreadable,
        AdmissionReasons.Backend, AdmissionReasons.TooLong, AdmissionReasons.NotSeenTwice, AdmissionReasons.IndexOnly,
        AdmissionReasons.WritesDisabled, AdmissionReasons.BudgetExhausted, AdmissionReasons.QueueFull,
        AdmissionReasons.NamespaceQueueShare, AdmissionReasons.RequestCap, AdmissionReasons.SecretContent, AdmissionReasons.OverBudget,
    };

    /// <summary>Sets the queue-depth source (the running writer).</summary>
    public static void SetQueueDepthProvider(Func<int>? provider)
    {
        s_queueDepth = provider;
        _ = QueueDepth;
    }

    /// <summary>Counts one write outcome (<c>stored</c>, <c>failed</c>, <c>too_large</c>).</summary>
    public static void RecordWrite(string result) => StoreWrites.Add(1, new KeyValuePair<string, object?>("result", result));

    /// <summary>Counts one evicted entry file.</summary>
    public static void RecordEviction() => StoreEvictions.Add(1);

    /// <summary>Counts one refused miss by reason (fixed label set).</summary>
    public static void RecordSkip(string reason) =>
        StoreSkips.Add(1, new KeyValuePair<string, object?>("reason", SkipReasons.Contains(reason) ? reason : "other"));

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
