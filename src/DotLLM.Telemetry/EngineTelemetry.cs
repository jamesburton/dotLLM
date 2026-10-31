using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Reflection;

namespace DotLLM.Telemetry;

/// <summary>
/// Central <see cref="Meter"/> and <see cref="ActivitySource"/> for dotLLM engine instrumentation.
/// <para>
/// Naming follows the OpenTelemetry semantic conventions — both surfaces are named
/// <c>DotLLM.Engine</c>. Instrument names use the <c>dotllm.engine.*</c> namespace.
/// </para>
/// <para>
/// All instruments are static so the hot path is a single <c>Instrument.Enabled</c> /
/// <see cref="ActivitySource.HasListeners"/> branch when nothing is subscribed.
/// </para>
/// </summary>
public static class EngineTelemetry
{
    /// <summary>OpenTelemetry-friendly name for both the <see cref="Meter"/> and <see cref="ActivitySource"/>.</summary>
    public const string Name = "DotLLM.Engine";

    private static readonly string s_version =
        typeof(EngineTelemetry).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(EngineTelemetry).Assembly.GetName().Version?.ToString()
        ?? "0.0.0";

    /// <summary>Shared <see cref="Meter"/> for all engine instruments.</summary>
    public static readonly Meter Meter = new(Name, s_version);

    /// <summary>Shared <see cref="ActivitySource"/> for engine spans.</summary>
    public static readonly ActivitySource ActivitySource = new(Name, s_version);

    /// <summary>Counter — prompt tokens processed during prefill, tagged by <c>model</c>.</summary>
    public static readonly Counter<long> PrefillTokens = Meter.CreateCounter<long>(
        "dotllm.engine.tokens.prefill", unit: "tokens",
        description: "Prompt tokens processed during prefill (excludes prefix-cache hits).");

    /// <summary>Counter — tokens generated during decode, tagged by <c>model</c>.</summary>
    public static readonly Counter<long> DecodeTokens = Meter.CreateCounter<long>(
        "dotllm.engine.tokens.decode", unit: "tokens",
        description: "Tokens generated during the decode phase.");

    /// <summary>
    /// Counter — generated tokens attributed to a fairness identity, tagged by <c>key</c> (the resolved
    /// API key). Emitted once per completed request by the continuous-batch scheduler when the request
    /// carries an <c>ApiKey</c>. Zero-overhead when no listener is subscribed. Cardinality is bounded by
    /// the number of distinct API keys — the host should ensure keys are low-cardinality before scraping.
    /// </summary>
    public static readonly Counter<long> TokensByKey = Meter.CreateCounter<long>(
        "dotllm.engine.tokens.by_key", unit: "tokens",
        description: "Generated tokens per API key (fairness accounting), tagged by key.");

    /// <summary>Histogram — prefill throughput in tokens/second.</summary>
    public static readonly Histogram<double> PrefillTokensPerSecond = Meter.CreateHistogram<double>(
        "dotllm.engine.tokens_per_second.prefill", unit: "tokens/s",
        description: "Prefill throughput in tokens per second.");

    /// <summary>Histogram — decode throughput in tokens/second.</summary>
    public static readonly Histogram<double> DecodeTokensPerSecond = Meter.CreateHistogram<double>(
        "dotllm.engine.tokens_per_second.decode", unit: "tokens/s",
        description: "Decode throughput in tokens per second.");

    /// <summary>Histogram — time to first token in milliseconds.</summary>
    public static readonly Histogram<double> TimeToFirstTokenMs = Meter.CreateHistogram<double>(
        "dotllm.engine.time_to_first_token_ms", unit: "ms",
        description: "Time from request start to first generated token.");

    /// <summary>Counter — completed requests, tagged by <c>model</c> and <c>finish_reason</c> (#1344).</summary>
    public static readonly Counter<long> Requests = Meter.CreateCounter<long>(
        "dotllm.engine.requests", unit: "{request}",
        description: "Completed generation requests by finish reason.");

    /// <summary>Histogram — end-to-end request duration in milliseconds (#1344).</summary>
    public static readonly Histogram<double> RequestDurationMs = Meter.CreateHistogram<double>(
        "dotllm.engine.request.duration_ms", unit: "ms",
        description: "Wall time of a completed generation request (prefill + decode).");

    /// <summary>Counter — prompt tokens served from the prefix cache instead of being prefilled (#1344).</summary>
    public static readonly Counter<long> CachedTokens = Meter.CreateCounter<long>(
        "dotllm.engine.tokens.cached", unit: "tokens",
        description: "Prompt tokens reused from the prefix cache.");

    /// <summary>
    /// Records every per-request counter and histogram for one finished request. Each instrument is guarded by its own
    /// <c>Enabled</c> check, so with no listener this is a handful of branches and no allocation.
    /// </summary>
    /// <param name="model">Model tag value (architecture name).</param>
    /// <param name="promptTokens">Prompt length including cached tokens.</param>
    /// <param name="cachedTokens">Prompt tokens reused from the prefix cache.</param>
    /// <param name="generatedTokens">Tokens generated (the first comes from prefill).</param>
    /// <param name="prefillMs">Prefill wall time.</param>
    /// <param name="decodeMs">Decode wall time.</param>
    /// <param name="totalMs">Request wall time; values at or below zero skip the duration histogram.</param>
    /// <param name="finishReason">Finish reason name (<c>Stop</c>, <c>Length</c>, ...).</param>
    public static void RecordCompletion(string model, int promptTokens, int cachedTokens, int generatedTokens,
        double prefillMs, double decodeMs, double totalMs, string finishReason)
    {
        var modelTag = new KeyValuePair<string, object?>(TelemetryTags.Model, model);

        int prefillEffective = promptTokens - cachedTokens;
        if (prefillEffective > 0)
        {
            if (PrefillTokens.Enabled) PrefillTokens.Add(prefillEffective, modelTag);
            if (prefillMs > 0 && PrefillTokensPerSecond.Enabled)
                PrefillTokensPerSecond.Record(prefillEffective / (prefillMs / 1000.0), modelTag);
        }
        int decodeEffective = generatedTokens > 1 ? generatedTokens - 1 : 0;
        if (decodeEffective > 0)
        {
            if (DecodeTokens.Enabled) DecodeTokens.Add(decodeEffective, modelTag);
            if (decodeMs > 0 && DecodeTokensPerSecond.Enabled)
                DecodeTokensPerSecond.Record(decodeEffective / (decodeMs / 1000.0), modelTag);
        }
        if (cachedTokens > 0 && CachedTokens.Enabled) CachedTokens.Add(cachedTokens, modelTag);
        if (Requests.Enabled)
            Requests.Add(1, modelTag, new KeyValuePair<string, object?>("finish_reason", finishReason));
        if (totalMs > 0 && RequestDurationMs.Enabled)
            RequestDurationMs.Record(totalMs, modelTag);
    }

    private static Func<long>? s_queueDepthProvider;
    private static Func<double>? s_kvCacheUtilizationProvider;

    static EngineTelemetry()
    {
        _ = Meter.CreateObservableGauge<long>(
            "dotllm.engine.request.queue_depth", ObserveQueueDepth,
            unit: "{request}",
            description: "Requests waiting for admission in the scheduler queue. -1 until the scheduler (Step 35) is wired.");

        _ = Meter.CreateObservableGauge<double>(
            "dotllm.engine.kvcache.utilization", ObserveKvCacheUtilization,
            unit: "1",
            description: "Fraction of paged KV-cache blocks in use (0..1). -1 until paged scheduler (Step 35) is wired.");
    }

    /// <summary>
    /// Registers a callback that returns the current scheduler queue depth.
    /// Without a callback the observable gauge emits <c>-1</c> as a sentinel.
    /// </summary>
    public static void SetQueueDepthProvider(Func<long>? provider) => s_queueDepthProvider = provider;

    /// <summary>
    /// Registers a callback returning paged KV-cache utilization in the range [0, 1].
    /// Without a callback the observable gauge emits <c>-1</c> as a sentinel.
    /// </summary>
    public static void SetKvCacheUtilizationProvider(Func<double>? provider) => s_kvCacheUtilizationProvider = provider;

    private static long ObserveQueueDepth()
    {
        var provider = s_queueDepthProvider;
        // TODO(step-35): wire to IScheduler.QueueDepth once continuous batching lands.
        return provider is null ? -1L : provider();
    }

    private static double ObserveKvCacheUtilization()
    {
        var provider = s_kvCacheUtilizationProvider;
        // TODO(step-35): wire to paged KV allocator usage once continuous batching lands.
        return provider is null ? -1.0 : provider();
    }
}
