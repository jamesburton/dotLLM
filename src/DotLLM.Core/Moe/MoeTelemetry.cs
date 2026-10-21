using System.Diagnostics.Metrics;
using System.Globalization;

namespace DotLLM.Core.Moe;

/// <summary>A point-in-time copy of the <see cref="MoeTelemetry"/> totals.</summary>
/// <param name="Hits">Expert lookups served from the device cache.</param>
/// <param name="Misses">Expert lookups that had to copy weights in.</param>
/// <param name="Evictions">Misses that displaced a cached expert.</param>
/// <param name="BytesH2D">Expert bytes copied host to device.</param>
/// <param name="CopyWaitMicros">Microseconds the compute stream waited for expert copies.</param>
/// <param name="Tokens">Tokens forwarded with offloaded experts.</param>
/// <param name="CacheSlots">Cache slots allocated (all pools).</param>
/// <param name="CacheBytes">Device bytes of the cache slots.</param>
/// <param name="Policy">Replacement policy of the active cache, or null when none is active.</param>
public readonly record struct MoeTelemetrySnapshot(long Hits, long Misses, long Evictions, long BytesH2D, long CopyWaitMicros,
    long Tokens, long CacheSlots, long CacheBytes, ExpertCachePolicy? Policy)
{
    /// <summary>Hit rate over all lookups (0 when none).</summary>
    public double HitRate => Hits + Misses == 0 ? 0 : (double)Hits / (Hits + Misses);

    /// <summary>Totals accumulated since <paramref name="start"/> (the cache size and policy are taken from this snapshot).</summary>
    /// <param name="start">An earlier snapshot.</param>
    /// <returns>The difference.</returns>
    public MoeTelemetrySnapshot Since(MoeTelemetrySnapshot start) => this with
    {
        Hits = Hits - start.Hits,
        Misses = Misses - start.Misses,
        Evictions = Evictions - start.Evictions,
        BytesH2D = BytesH2D - start.BytesH2D,
        CopyWaitMicros = CopyWaitMicros - start.CopyWaitMicros,
        Tokens = Tokens - start.Tokens,
    };

    /// <summary>
    /// The one-line summary printed by <c>bench</c> and <c>run --verbose</c>, e.g.
    /// <c>MoE offload: cache 6.1 GiB (812 slots, SLRU), hit 81.2%, H2D 71.3 MB/token, copy wait 9.8 ms/token</c>; null when no
    /// offloaded model is active.
    /// </summary>
    /// <returns>The line, or null.</returns>
    public string? FormatSummary()
    {
        if (Policy is not { } policy) return null;
        double perToken = Math.Max(Tokens, 1);
        var c = CultureInfo.InvariantCulture;
        return string.Format(c,
            "MoE offload: cache {0:F1} GiB ({1} slots, {2}), hit {3:F1}%, H2D {4:F1} MB/token, copy wait {5:F1} ms/token, evictions {6}",
            CacheBytes / (1024.0 * 1024 * 1024), CacheSlots, policy.ToString().ToUpperInvariant(), HitRate * 100,
            BytesH2D / 1e6 / perToken, CopyWaitMicros / 1000.0 / perToken, Evictions);
    }
}

/// <summary>
/// Process-wide expert-offload telemetry (issue #1070): a <see cref="Meter"/> named <c>DotLLM.Moe</c> whose observable
/// instruments read plain <see cref="Interlocked"/> totals (zero cost on the hot path when nothing listens), modelled on
/// <c>DotLLM.Ple</c>.
/// </summary>
public static class MoeTelemetry
{
    /// <summary>Meter name.</summary>
    public const string MeterName = "DotLLM.Moe";

    private static long s_hits, s_misses, s_evictions, s_bytesH2D, s_copyWaitMicros, s_tokens, s_cacheSlots, s_cacheBytes;
    private static int s_policy = -1;

    private static readonly Meter s_meter = new(MeterName);

    // Instruments are created once and observed through the meter; the fields keep them rooted.
    private static readonly ObservableCounter<long> s_hitsC = s_meter.CreateObservableCounter("dotllm.moe.expert_hits", () => Volatile.Read(ref s_hits), "experts");
    private static readonly ObservableCounter<long> s_missC = s_meter.CreateObservableCounter("dotllm.moe.expert_misses", () => Volatile.Read(ref s_misses), "experts");
    private static readonly ObservableCounter<long> s_evictC = s_meter.CreateObservableCounter("dotllm.moe.evictions", () => Volatile.Read(ref s_evictions), "experts");
    private static readonly ObservableCounter<long> s_bytesC = s_meter.CreateObservableCounter("dotllm.moe.bytes_h2d", () => Volatile.Read(ref s_bytesH2D), "By");
    private static readonly ObservableCounter<long> s_waitC = s_meter.CreateObservableCounter("dotllm.moe.copy_wait_us", () => Volatile.Read(ref s_copyWaitMicros), "us");
    private static readonly ObservableGauge<long> s_slotsG = s_meter.CreateObservableGauge("dotllm.moe.cache_slots", () => Volatile.Read(ref s_cacheSlots), "slots");
    private static readonly ObservableGauge<long> s_bytesG = s_meter.CreateObservableGauge("dotllm.moe.cache_bytes", () => Volatile.Read(ref s_cacheBytes), "By");

    /// <summary>Records one layer call's lookups.</summary>
    /// <param name="hits">Experts found resident.</param>
    /// <param name="misses">Experts copied in.</param>
    /// <param name="evictions">Cached experts displaced.</param>
    /// <param name="bytesH2D">Bytes copied host to device.</param>
    public static void RecordLookups(long hits, long misses, long evictions, long bytesH2D)
    {
        if (hits != 0) Interlocked.Add(ref s_hits, hits);
        if (misses != 0) Interlocked.Add(ref s_misses, misses);
        if (evictions != 0) Interlocked.Add(ref s_evictions, evictions);
        if (bytesH2D != 0) Interlocked.Add(ref s_bytesH2D, bytesH2D);
    }

    /// <summary>Records time the compute stream spent waiting for expert copies.</summary>
    /// <param name="micros">Microseconds.</param>
    public static void RecordCopyWait(long micros)
    {
        if (micros > 0) Interlocked.Add(ref s_copyWaitMicros, micros);
    }

    /// <summary>Records tokens forwarded with offloaded experts (the per-token denominator of the summary).</summary>
    /// <param name="tokens">Token count.</param>
    public static void RecordTokens(long tokens) => Interlocked.Add(ref s_tokens, tokens);

    /// <summary>Publishes the size of the active cache (an offloaded model was loaded).</summary>
    /// <param name="slots">Slots of all pools.</param>
    /// <param name="bytes">Device bytes of the slots.</param>
    /// <param name="policy">Replacement policy.</param>
    public static void SetCache(long slots, long bytes, ExpertCachePolicy policy)
    {
        Volatile.Write(ref s_cacheSlots, slots);
        Volatile.Write(ref s_cacheBytes, bytes);
        Volatile.Write(ref s_policy, (int)policy);
    }

    /// <summary>Clears the active-cache gauges (the offloaded model was disposed).</summary>
    public static void ClearCache()
    {
        Volatile.Write(ref s_cacheSlots, 0);
        Volatile.Write(ref s_cacheBytes, 0);
        Volatile.Write(ref s_policy, -1);
    }

    /// <summary>Current totals.</summary>
    /// <returns>The snapshot.</returns>
    public static MoeTelemetrySnapshot Snapshot()
    {
        int p = Volatile.Read(ref s_policy);
        return new MoeTelemetrySnapshot(Volatile.Read(ref s_hits), Volatile.Read(ref s_misses), Volatile.Read(ref s_evictions),
            Volatile.Read(ref s_bytesH2D), Volatile.Read(ref s_copyWaitMicros), Volatile.Read(ref s_tokens),
            Volatile.Read(ref s_cacheSlots), Volatile.Read(ref s_cacheBytes), p < 0 ? null : (ExpertCachePolicy)p);
    }

    /// <summary>Keeps the instrument fields referenced (they are observed through the meter, never read directly).</summary>
    internal static int InstrumentCount => new object[] { s_hitsC, s_missC, s_evictC, s_bytesC, s_waitC, s_slotsG, s_bytesG }.Length;
}
