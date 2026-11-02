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
    /// <param name="modelId">The model the request names, when known; a change of model triggers a new probe (spec 8.3).</param>
    /// <param name="ct">The caller's cancellation.</param>
    ValueTask<ChunkCacheCapabilities?> GetAsync(string? modelId, CancellationToken ct);

    /// <summary>Forgets any cached answer, so the next call probes again (e.g. after the server refused a manifest).</summary>
    void Invalidate();
}

/// <summary>A fixed answer, for a provider probed once when it was built (<see langword="null"/> = not capable).</summary>
public sealed class FixedChunkCapabilities(ChunkCacheCapabilities? capabilities) : IChunkCapabilitySource
{
    /// <inheritdoc />
    public ValueTask<ChunkCacheCapabilities?> GetAsync(string? modelId, CancellationToken ct) => ValueTask.FromResult(capabilities);

    /// <summary>No-op: a fixed answer has nothing to forget.</summary>
    public void Invalidate()
    {
    }
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
/// <item>A request naming a different model than the last known one probes again at once; a request naming no model, or
/// the first to name one, keeps the current answer.</item>
/// <item>Concurrent callers share one probe. The probe has its own timeout (default 5 seconds) and is not cancelled by a
/// caller giving up; a caller's cancellation cancels only its own wait.</item>
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

    private readonly IChunkCacheClient _client;
    private readonly TimeSpan _successTtl;
    private readonly TimeSpan _failureTtl;
    private readonly TimeSpan _probeTimeout;
    private readonly TimeProvider _time;
    private readonly object _gate = new();

    private ChunkCacheProbeResult? _last;
    private string? _lastModel;
    private DateTimeOffset _expires;
    private TaskCompletionSource<ChunkCacheProbeResult>? _inFlight;
    private string? _inFlightModel;
    private int _probes;

    /// <summary>Creates the cache.</summary>
    /// <param name="client">The control-plane client whose <see cref="IChunkCacheClient.GetCapabilitiesAsync"/> probes.</param>
    /// <param name="successTtl">How long a success is trusted (default <see cref="DefaultSuccessTtl"/>).</param>
    /// <param name="failureTtl">How long a failure is remembered (default <see cref="DefaultFailureTtl"/>).</param>
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
        string? model;
        lock (_gate)
        {
            model = modelId;
            if (_last is not null && _time.GetUtcNow() < _expires && SameModel(model, _lastModel))
            {
                _lastModel ??= model;
                return _last.Capabilities;
            }

            if (_inFlight is not null && SameModel(model, _inFlightModel))
            {
                _inFlightModel ??= model;
                pending = _inFlight;
            }
            else
            {
                pending = _inFlight = new TaskCompletionSource<ChunkCacheProbeResult>(TaskCreationOptions.RunContinuationsAsynchronously);
                _inFlightModel = model;
                start = true;
            }
        }

        if (start)
            _ = ProbeAsync(pending, model);

        ChunkCacheProbeResult result = await pending.Task.WaitAsync(ct).ConfigureAwait(false);
        return result.Capabilities;
    }

    /// <inheritdoc />
    public void Invalidate()
    {
        lock (_gate)
            _last = null;
    }

    /// <summary>
    /// Only a change between two known model ids is a change: a call that names no model (the decorator often does not
    /// know it) and the request that names it describe the same server.
    /// </summary>
    private static bool SameModel(string? a, string? b) => a is null || b is null || string.Equals(a, b, StringComparison.Ordinal);

    private async Task ProbeAsync(TaskCompletionSource<ChunkCacheProbeResult> pending, string? model)
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
            if (ReferenceEquals(_inFlight, pending))
            {
                _inFlight = null;
                _inFlightModel = null;
            }

            _last = result;
            _lastModel = model ?? _lastModel;
            _expires = _time.GetUtcNow() + (result.IsCapable ? _successTtl : _failureTtl);
        }

        pending.SetResult(result);
    }
}
