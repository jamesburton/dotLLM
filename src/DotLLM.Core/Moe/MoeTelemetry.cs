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
/// <param name="Routed">Routed experts per layer call over every offloaded layer (cached, copied or computed on the CPU).</param>
/// <param name="ColdExpertsCpu">Routed experts computed on the CPU (#1071).</param>
/// <param name="CpuComputeMicros">Wall microseconds of the CPU cold legs (first worker start to last worker end).</param>
/// <param name="CpuWaitMicros">
/// Microseconds the compute stream waited for the cold legs' results (the CPU leg's, and the iGPU leg's when one ran in the call).
/// </param>
/// <param name="Promotions">Cold experts promote-only admission copied into the cache.</param>
/// <param name="ColdExpertsIgpu">Routed experts computed on the integrated GPU (#1083).</param>
/// <param name="IgpuComputeMicros">Wall microseconds of the iGPU legs, from the start of each submission until its fence was observed.</param>
/// <param name="IgpuWaitMicros">Microseconds the forward thread blocked on an iGPU fence after its own (CPU-leg) work was done.</param>
/// <param name="IgpuBanksImported">Expert banks of the active iGPU leg imported zero-copy (a gauge, like the cache size).</param>
/// <param name="IgpuBanksStaged">Expert banks of the active iGPU leg copied into device memory because the import was refused.</param>
/// <param name="IgpuStagedBytes">
/// Bytes of those staged banks: on a UMA iGPU, a second copy of them in the same RAM the memory-mapped GGUF already occupies.
/// </param>
public readonly record struct MoeTelemetrySnapshot(long Hits, long Misses, long Evictions, long BytesH2D, long CopyWaitMicros,
    long Tokens, long CacheSlots, long CacheBytes, ExpertCachePolicy? Policy, long Routed = 0, long ColdExpertsCpu = 0,
    long CpuComputeMicros = 0, long CpuWaitMicros = 0, long Promotions = 0, long ColdExpertsIgpu = 0, long IgpuComputeMicros = 0,
    long IgpuWaitMicros = 0, long IgpuBanksImported = 0, long IgpuBanksStaged = 0, long IgpuStagedBytes = 0)
{
    /// <summary>Share of routed experts computed on the CPU (0 when none were routed).</summary>
    public double ColdCpuFraction => Routed == 0 ? 0 : (double)ColdExpertsCpu / Routed;

    /// <summary>Share of routed experts computed on the integrated GPU (0 when none were routed; #1083).</summary>
    public double ColdIgpuFraction => Routed == 0 ? 0 : (double)ColdExpertsIgpu / Routed;

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
        Routed = Routed - start.Routed,
        ColdExpertsCpu = ColdExpertsCpu - start.ColdExpertsCpu,
        CpuComputeMicros = CpuComputeMicros - start.CpuComputeMicros,
        CpuWaitMicros = CpuWaitMicros - start.CpuWaitMicros,
        Promotions = Promotions - start.Promotions,
        ColdExpertsIgpu = ColdExpertsIgpu - start.ColdExpertsIgpu,
        IgpuComputeMicros = IgpuComputeMicros - start.IgpuComputeMicros,
        IgpuWaitMicros = IgpuWaitMicros - start.IgpuWaitMicros,
    };

    /// <summary>
    /// The one-line summary printed by <c>bench</c> and <c>run --verbose</c>, e.g.
    /// <c>MoE offload: cache 6.1 GiB (812 slots, SLRU), hit 81.2%, H2D 71.3 MB/token, copy wait 9.8 ms/token</c>, followed by
    /// <c>; cold: CPU 41% of routed, CPU wait 3.2 ms/token, promotions 12</c> when a CPU leg ran (#1071), and
    /// <c>; iGPU 30% of routed, iGPU compute 4.1 ms/token, iGPU wait 0.6 ms/token</c> when an iGPU leg ran (#1083), and
    /// <c>; iGPU banks staged 80 of 80 (9.2 GiB, a second copy in RAM), 0 zero-copy</c> when the iGPU leg holds staged banks; null
    /// when no offloaded model is active.
    /// </summary>
    /// <returns>The line, or null.</returns>
    public string? FormatSummary()
    {
        if (Policy is not { } policy) return null;
        double perToken = Math.Max(Tokens, 1);
        var c = CultureInfo.InvariantCulture;
        string line = string.Format(c,
            "MoE offload: cache {0:F1} GiB ({1} slots, {2}), hit {3:F1}%, H2D {4:F1} MB/token, copy wait {5:F1} ms/token, evictions {6}",
            CacheBytes / (1024.0 * 1024 * 1024), CacheSlots, policy.ToString().ToUpperInvariant(), HitRate * 100,
            BytesH2D / 1e6 / perToken, CopyWaitMicros / 1000.0 / perToken, Evictions);
        if (ColdExpertsCpu > 0 || ColdExpertsIgpu > 0 || Promotions > 0)
            line += string.Format(c, "; cold: CPU {0:F0}% of routed, CPU compute {1:F1} ms/token, CPU wait {2:F1} ms/token, promotions {3}",
                ColdCpuFraction * 100, CpuComputeMicros / 1000.0 / perToken, CpuWaitMicros / 1000.0 / perToken, Promotions);
        if (ColdExpertsIgpu > 0)
            line += string.Format(c, "; iGPU {0:F0}% of routed, iGPU compute {1:F1} ms/token, iGPU wait {2:F1} ms/token",
                ColdIgpuFraction * 100, IgpuComputeMicros / 1000.0 / perToken, IgpuWaitMicros / 1000.0 / perToken);
        if (IgpuBanksStaged > 0)
            line += string.Format(c, "; iGPU banks staged {0} of {1} ({2:F1} GiB, a second copy in RAM), {3} zero-copy",
                IgpuBanksStaged, IgpuBanksStaged + IgpuBanksImported, IgpuStagedBytes / (1024.0 * 1024 * 1024), IgpuBanksImported);
        return line;
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
    private static long s_routed, s_coldCpu, s_cpuComputeMicros, s_cpuWaitMicros, s_promotions;
    private static long s_coldIgpu, s_igpuComputeMicros, s_igpuWaitMicros;
    private static long s_igpuBanksImported, s_igpuBanksStaged, s_igpuStagedBytes;
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
    private static readonly ObservableCounter<long> s_coldCpuC = s_meter.CreateObservableCounter("dotllm.moe.cold_experts_cpu", () => Volatile.Read(ref s_coldCpu), "experts");
    private static readonly ObservableCounter<long> s_cpuComputeC = s_meter.CreateObservableCounter("dotllm.moe.cpu_compute_us", () => Volatile.Read(ref s_cpuComputeMicros), "us");
    private static readonly ObservableCounter<long> s_cpuWaitC = s_meter.CreateObservableCounter("dotllm.moe.cpu_wait_us", () => Volatile.Read(ref s_cpuWaitMicros), "us");
    private static readonly ObservableCounter<long> s_promoC = s_meter.CreateObservableCounter("dotllm.moe.promotions", () => Volatile.Read(ref s_promotions), "experts");
    private static readonly ObservableCounter<long> s_coldIgpuC = s_meter.CreateObservableCounter("dotllm.moe.cold_experts_igpu", () => Volatile.Read(ref s_coldIgpu), "experts");
    private static readonly ObservableCounter<long> s_igpuComputeC = s_meter.CreateObservableCounter("dotllm.moe.igpu_compute_us", () => Volatile.Read(ref s_igpuComputeMicros), "us");
    private static readonly ObservableCounter<long> s_igpuWaitC = s_meter.CreateObservableCounter("dotllm.moe.igpu_wait_us", () => Volatile.Read(ref s_igpuWaitMicros), "us");
    private static readonly ObservableGauge<long> s_igpuImportedG = s_meter.CreateObservableGauge("dotllm.moe.igpu_banks_imported", () => Volatile.Read(ref s_igpuBanksImported), "banks");
    private static readonly ObservableGauge<long> s_igpuStagedG = s_meter.CreateObservableGauge("dotllm.moe.igpu_banks_staged", () => Volatile.Read(ref s_igpuBanksStaged), "banks");
    private static readonly ObservableGauge<long> s_igpuStagedBytesG = s_meter.CreateObservableGauge("dotllm.moe.igpu_staged_bytes", () => Volatile.Read(ref s_igpuStagedBytes), "By");

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

    /// <summary>
    /// Records one offloaded layer call's routed experts and how many of them ran on the CPU (#1071) and on the integrated GPU (#1083).
    /// </summary>
    /// <param name="routed">Distinct experts the call routed to.</param>
    /// <param name="coldCpu">Of those, experts computed on the CPU.</param>
    /// <param name="coldIgpu">Of those, experts computed on the integrated GPU.</param>
    public static void RecordRouted(long routed, long coldCpu, long coldIgpu = 0)
    {
        if (routed != 0) Interlocked.Add(ref s_routed, routed);
        if (coldCpu != 0) Interlocked.Add(ref s_coldCpu, coldCpu);
        if (coldIgpu != 0) Interlocked.Add(ref s_coldIgpu, coldIgpu);
    }

    /// <summary>Records one iGPU cold leg: submission start until its fence was observed (#1083).</summary>
    /// <param name="micros">Microseconds.</param>
    public static void RecordIgpuCompute(long micros)
    {
        if (micros > 0) Interlocked.Add(ref s_igpuComputeMicros, micros);
    }

    /// <summary>Records time the forward thread blocked on an iGPU fence after its own work (#1083).</summary>
    /// <param name="micros">Microseconds.</param>
    public static void RecordIgpuWait(long micros)
    {
        if (micros > 0) Interlocked.Add(ref s_igpuWaitMicros, micros);
    }

    /// <summary>Records one CPU cold leg: its wall time on the CPU (#1071).</summary>
    /// <param name="micros">Microseconds.</param>
    public static void RecordCpuCompute(long micros)
    {
        if (micros > 0) Interlocked.Add(ref s_cpuComputeMicros, micros);
    }

    /// <summary>Records time the compute stream spent waiting for CPU leg results (#1071).</summary>
    /// <param name="micros">Microseconds.</param>
    public static void RecordCpuWait(long micros)
    {
        if (micros > 0) Interlocked.Add(ref s_cpuWaitMicros, micros);
    }

    /// <summary>Records experts promote-only admission copied into the cache (#1071).</summary>
    /// <param name="count">Promotions.</param>
    public static void RecordPromotions(long count)
    {
        if (count != 0) Interlocked.Add(ref s_promotions, count);
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

    /// <summary>
    /// Publishes the expert banks of the active iGPU leg (#1083): how many were imported zero-copy and how many staged, with the
    /// staged bytes. On a UMA iGPU every staged byte is RAM spent twice (the mapped GGUF and the copy); zeros when no leg is open.
    /// </summary>
    /// <param name="imported">Banks imported zero-copy.</param>
    /// <param name="staged">Banks staged because the import was refused.</param>
    /// <param name="stagedBytes">Bytes of the staged banks.</param>
    public static void SetIgpuBanks(long imported, long staged, long stagedBytes)
    {
        Volatile.Write(ref s_igpuBanksImported, imported);
        Volatile.Write(ref s_igpuBanksStaged, staged);
        Volatile.Write(ref s_igpuStagedBytes, stagedBytes);
    }

    /// <summary>Clears the active-cache gauges and the iGPU bank gauges (the offloaded model was disposed).</summary>
    public static void ClearCache()
    {
        Volatile.Write(ref s_cacheSlots, 0);
        Volatile.Write(ref s_cacheBytes, 0);
        Volatile.Write(ref s_policy, -1);
        SetIgpuBanks(0, 0, 0);
    }

    /// <summary>Current totals.</summary>
    /// <returns>The snapshot.</returns>
    public static MoeTelemetrySnapshot Snapshot()
    {
        int p = Volatile.Read(ref s_policy);
        return new MoeTelemetrySnapshot(Volatile.Read(ref s_hits), Volatile.Read(ref s_misses), Volatile.Read(ref s_evictions),
            Volatile.Read(ref s_bytesH2D), Volatile.Read(ref s_copyWaitMicros), Volatile.Read(ref s_tokens),
            Volatile.Read(ref s_cacheSlots), Volatile.Read(ref s_cacheBytes), p < 0 ? null : (ExpertCachePolicy)p,
            Volatile.Read(ref s_routed), Volatile.Read(ref s_coldCpu), Volatile.Read(ref s_cpuComputeMicros),
            Volatile.Read(ref s_cpuWaitMicros), Volatile.Read(ref s_promotions), Volatile.Read(ref s_coldIgpu),
            Volatile.Read(ref s_igpuComputeMicros), Volatile.Read(ref s_igpuWaitMicros), Volatile.Read(ref s_igpuBanksImported),
            Volatile.Read(ref s_igpuBanksStaged), Volatile.Read(ref s_igpuStagedBytes));
    }

    /// <summary>Keeps the instrument fields referenced (they are observed through the meter, never read directly).</summary>
    internal static int InstrumentCount => new object[]
    {
        s_hitsC, s_missC, s_evictC, s_bytesC, s_waitC, s_slotsG, s_bytesG, s_coldCpuC, s_cpuComputeC, s_cpuWaitC, s_promoC,
        s_coldIgpuC, s_igpuComputeC, s_igpuWaitC, s_igpuImportedG, s_igpuStagedG, s_igpuStagedBytesG,
    }.Length;
}
