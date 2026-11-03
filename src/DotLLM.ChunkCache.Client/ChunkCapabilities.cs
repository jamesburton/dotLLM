// SPDX-License-Identifier: MIT
namespace DotLLM.ChunkCache.Client;

/// <summary>
/// Where the pipeline learns whether the server speaks the convention. The body handler sends <c>chunk_cache</c> only
/// when this returns capabilities (contract ruling 16: anything else means strip annotations and send a plain request).
/// </summary>
public interface IChunkCapabilitySource
{
    /// <summary>
    /// The capabilities, or <see langword="null"/> when annotations must be stripped. Never throws for a server or
    /// transport failure; only the caller's own cancellation propagates.
    /// </summary>
    /// <param name="modelId">The model the request names, when known (answers are kept per model, spec 8.3).</param>
    /// <param name="ct">The caller's cancellation.</param>
    ValueTask<ChunkCacheCapabilities?> GetAsync(string? modelId, CancellationToken ct);

    /// <summary>Forgets any cached answer, so the next call probes again.</summary>
    void Invalidate();

    /// <summary>
    /// The server refused a non-strict manifest with a <c>chunk_*</c> error although it claimed the convention: answer
    /// "not capable" for a while (the failure period) instead of probing again, so a server that keeps refusing costs one
    /// extra request per period, not per request. The default forgets the cached answer.
    /// </summary>
    void MarkNotCapable() => Invalidate();
}

/// <summary>A fixed answer, for a provider probed once when it was built (<see langword="null"/> = not capable).</summary>
/// <remarks><see cref="MarkNotCapable"/> turns annotations off for <see cref="RejectionPeriod"/>.</remarks>
public sealed class FixedChunkCapabilities(ChunkCacheCapabilities? capabilities) : IChunkCapabilitySource
{
    /// <summary>How long a refusal turns annotations off.</summary>
    public static readonly TimeSpan RejectionPeriod = TimeSpan.FromSeconds(30);

    private long _rejectedUntil; // Environment.TickCount64

    /// <inheritdoc />
    public ValueTask<ChunkCacheCapabilities?> GetAsync(string? modelId, CancellationToken ct) =>
        ValueTask.FromResult(Environment.TickCount64 < Volatile.Read(ref _rejectedUntil) ? null : capabilities);

    /// <summary>Ends a refusal period early; the fixed answer itself has nothing to forget.</summary>
    public void Invalidate() => Volatile.Write(ref _rejectedUntil, 0);

    /// <inheritdoc />
    public void MarkNotCapable() =>
        Volatile.Write(ref _rejectedUntil, Environment.TickCount64 + (long)RejectionPeriod.TotalMilliseconds);
}

/// <summary>
/// A cached, fail-closed capability probe over <see cref="IChunkCacheClient.GetCapabilitiesAsync"/> (spec 8.3 "probe
/// when the provider is built, and again when the model id changes"; contract ruling 16).
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>A success is cached for <c>successTtl</c> (default 5 minutes), any failure (404, 401, 403, 5xx, transport error,
/// probe timeout) for <c>failureTtl</c> (default 30 seconds), so a flapping server is neither hammered nor trusted while
/// it fails: annotations go off at the first failed probe and come back at the first successful one.</item>
/// <item>Answers are kept per model id (up to <see cref="MaxModels"/>, oldest dropped first), so two models on one
/// server (a main and a sub-agent model) do not re-probe on every switch. A request naming no model uses the most
/// recently used answer; the first request to name a model adopts an answer probed without one.</item>
/// <item>After <see cref="MarkNotCapable"/> (the server refused a manifest), every call answers "not capable" for the
/// failure TTL without probing.</item>
/// <item>Concurrent callers of one model share one probe. The probe runs without the first caller's execution context
/// (no chunk scope, no tracing <c>Activity</c> parent), has its own timeout (default 5 seconds) and is not cancelled
/// by a caller giving up; a caller's cancellation cancels only its own wait.</item>
/// </list>
/// </remarks>
public sealed class ChunkCapabilityCache : IChunkCapabilitySource
{
    /// <summary>Default lifetime of a successful probe.</summary>
    public static readonly TimeSpan DefaultSuccessTtl = TimeSpan.FromMinutes(5);

    /// <summary>Default lifetime of a failed probe.</summary>
    public static readonly TimeSpan DefaultFailureTtl = TimeSpan.FromSeconds(30);

    /// <summary>Default probe timeout.</summary>
    public static readonly TimeSpan DefaultProbeTimeout = TimeSpan.FromSeconds(5);

    /// <summary>The most model ids answered at once.</summary>
    public const int MaxModels = 16;

    private const string NoModel = "";

    private readonly IChunkCacheClient _client;
    private readonly TimeSpan _successTtl;
    private readonly TimeSpan _failureTtl;
    private readonly TimeSpan _probeTimeout;
    private readonly TimeProvider _time;
    private readonly object _gate = new();

    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TaskCompletionSource<ChunkCacheProbeResult>> _inFlight = new(StringComparer.Ordinal);
    private string? _lastKey;
    private long _stamp;
    private DateTimeOffset _rejectedUntil = DateTimeOffset.MinValue;
    private ChunkCacheProbeResult? _last;
    private int _probes;

    /// <summary>Creates the cache.</summary>
    /// <param name="client">The control-plane client whose <see cref="IChunkCacheClient.GetCapabilitiesAsync"/> probes.</param>
    /// <param name="successTtl">How long a success is trusted (default <see cref="DefaultSuccessTtl"/>).</param>
    /// <param name="failureTtl">How long a failure or a refusal is remembered (default <see cref="DefaultFailureTtl"/>).</param>
    /// <param name="probeTimeout">Probe timeout (default <see cref="DefaultProbeTimeout"/>; <see cref="Timeout.InfiniteTimeSpan"/> disables it).</param>
    /// <param name="timeProvider">Clock for the TTLs (default <see cref="TimeProvider.System"/>).</param>
    public ChunkCapabilityCache(IChunkCacheClient client, TimeSpan? successTtl = null, TimeSpan? failureTtl = null,
        TimeSpan? probeTimeout = null, TimeProvider? timeProvider = null)
    {
        ArgumentNullException.ThrowIfNull(client);
        _client = client;
        _successTtl = successTtl ?? DefaultSuccessTtl;
        _failureTtl = failureTtl ?? DefaultFailureTtl;
        _probeTimeout = probeTimeout ?? DefaultProbeTimeout;
        _time = timeProvider ?? TimeProvider.System;
        ArgumentOutOfRangeException.ThrowIfLessThan(_successTtl, TimeSpan.Zero, nameof(successTtl));
        ArgumentOutOfRangeException.ThrowIfLessThan(_failureTtl, TimeSpan.Zero, nameof(failureTtl));
        if (_probeTimeout != Timeout.InfiniteTimeSpan && _probeTimeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(probeTimeout));
    }

    /// <summary>The last completed probe (its <see cref="ChunkCacheProbeResult.Failure"/> says why it failed).</summary>
    public ChunkCacheProbeResult? LastResult
    {
        get
        {
            lock (_gate) return _last;
        }
    }

    /// <summary>Probes started so far.</summary>
    public int ProbeCount => Volatile.Read(ref _probes);

    /// <inheritdoc />
    public async ValueTask<ChunkCacheCapabilities?> GetAsync(string? modelId, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        TaskCompletionSource<ChunkCacheProbeResult> pending;
        bool start = false;
        string key;
        lock (_gate)
        {
            DateTimeOffset now = _time.GetUtcNow();
            if (now < _rejectedUntil)
                return null;

            key = modelId ?? _lastKey ?? NoModel;
            if (_entries.TryGetValue(key, out Entry? entry) && now < entry.Expires)
            {
                _lastKey = key;
                return entry.Result.Capabilities;
            }

            // The first request to name a model adopts an answer probed without one (the decorator often does not know it).
            if (modelId is not null && !_entries.ContainsKey(modelId) && _entries.TryGetValue(NoModel, out Entry? anon) && now < anon.Expires)
            {
                _entries.Remove(NoModel);
                _entries[modelId] = anon;
                _lastKey = modelId;
                return anon.Result.Capabilities;
            }

            if (_inFlight.TryGetValue(key, out TaskCompletionSource<ChunkCacheProbeResult>? running)
                || (modelId is not null && _inFlight.TryGetValue(NoModel, out running)))
            {
                pending = running;
            }
            else
            {
                pending = _inFlight[key] = new TaskCompletionSource<ChunkCacheProbeResult>(TaskCreationOptions.RunContinuationsAsynchronously);
                start = true;
            }
        }

        if (start)
        {
            // The shared probe must not inherit this caller's execution context (its chunk scope, its tracing Activity).
            if (ExecutionContext.IsFlowSuppressed())
            {
                _ = Task.Run(() => ProbeAsync(pending, key));
            }
            else
            {
                using (ExecutionContext.SuppressFlow())
                    _ = Task.Run(() => ProbeAsync(pending, key));
            }
        }

        ChunkCacheProbeResult result = await pending.Task.WaitAsync(ct).ConfigureAwait(false);
        lock (_gate)
        {
            if (_time.GetUtcNow() < _rejectedUntil)
                return null;
        }

        return result.Capabilities;
    }

    /// <inheritdoc />
    public void Invalidate()
    {
        lock (_gate)
        {
            _entries.Clear();
            _rejectedUntil = DateTimeOffset.MinValue;
        }
    }

    /// <inheritdoc />
    public void MarkNotCapable()
    {
        lock (_gate)
            _rejectedUntil = _time.GetUtcNow() + _failureTtl;
    }

    private async Task ProbeAsync(TaskCompletionSource<ChunkCacheProbeResult> pending, string key)
    {
        Interlocked.Increment(ref _probes);
        ChunkCacheProbeResult result;
        using (var timeout = _probeTimeout == Timeout.InfiniteTimeSpan ? new CancellationTokenSource() : new CancellationTokenSource(_probeTimeout, _time))
        {
            try
            {
                ChunkCacheCapabilities? caps = await _client.GetCapabilitiesAsync(timeout.Token).ConfigureAwait(false);
                result = new ChunkCacheProbeResult(caps is not null && caps.SupportsVersion(1) ? caps : null, null);
            }
            catch (Exception ex)
            {
                // Ruling 16: any exception (including this probe's own timeout) means "not capable".
                result = new ChunkCacheProbeResult(null, ex);
            }
        }

        lock (_gate)
        {
            if (_inFlight.TryGetValue(key, out TaskCompletionSource<ChunkCacheProbeResult>? current) && ReferenceEquals(current, pending))
                _inFlight.Remove(key);

            _last = result;
            _entries[key] = new Entry(result, _time.GetUtcNow() + (result.IsCapable ? _successTtl : _failureTtl), ++_stamp);
            _lastKey = key;
            while (_entries.Count > MaxModels)
            {
                string oldest = _entries.MinBy(kv => kv.Value.Stamp).Key;
                _entries.Remove(oldest);
            }
        }

        pending.SetResult(result);
    }

    private sealed record Entry(ChunkCacheProbeResult Result, DateTimeOffset Expires, long Stamp);
}
