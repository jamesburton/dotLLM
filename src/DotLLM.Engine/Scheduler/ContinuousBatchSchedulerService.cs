using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Telemetry;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Async-driven wrapper around <see cref="ContinuousBatchScheduler"/> implementing the existing
/// <see cref="IScheduler"/> contract (<c>EnqueueAsync</c> + <c>RunLoopAsync</c>).
/// </summary>
/// <remarks>
/// <para>This is the engine-integration seam. The server (or any host) wires a single instance
/// per loaded model, calls <see cref="RunLoopAsync"/> on a background task at startup, and
/// awaits <see cref="EnqueueAsync"/> from each request handler.</para>
///
/// <para>The run loop polls <see cref="ContinuousBatchScheduler.Step"/> repeatedly. When the
/// scheduler is idle (no queued or active sequences) the loop waits on a semaphore that
/// <see cref="EnqueueAsync"/> releases — so an idle server consumes zero CPU until a request
/// arrives. While the scheduler has work, the loop spins without yielding to maximize
/// per-iteration throughput (the bottleneck is the model's forward pass, not the loop body).</para>
///
/// <para>Backpressure: if the underlying scheduler queue is full (configurable via
/// <see cref="ContinuousBatchSchedulerOptions.MaxActiveSequences"/> +
/// pool free-block check), Submit still accepts the request — admission happens later
/// when capacity returns. Callers wanting an immediate 429 should gate before calling Enqueue
/// (the rate-limiting layer in roadmap step 38).</para>
/// </remarks>
public sealed class ContinuousBatchSchedulerService : IScheduler, IDisposable
{
    private readonly ContinuousBatchScheduler _inner;
    private readonly SemaphoreSlim _wakeup = new(initialCount: 0, maxCount: int.MaxValue);
    private readonly KvBlockPool? _pagedPool;
    private readonly bool _ownsTelemetryProviders;
    private bool _disposed;

    /// <summary>The underlying step-driven scheduler. Exposed for advanced callers and tests.</summary>
    public IBatchScheduler Inner => _inner;

    /// <summary>
    /// Creates a new async scheduler service.
    /// </summary>
    /// <param name="model">Loaded transformer model.</param>
    /// <param name="tokenizer">Tokenizer used for response detokenization.</param>
    /// <param name="kvCacheFactory">Per-sequence KV-cache factory.</param>
    /// <param name="options">Scheduler options. Null = defaults.</param>
    /// <param name="pagedPool">Optional paged-block pool, used for admission gating and as the
    /// source for the <c>dotllm.engine.kvcache.utilization</c> observable gauge.</param>
    /// <param name="prefixCache">Optional prefix-cache hook (Step 37). Consulted on admission.</param>
    /// <param name="registerTelemetryProviders">When <see langword="true"/> (default), registers
    /// engine-telemetry providers (<see cref="EngineTelemetry.SetQueueDepthProvider"/> and
    /// <see cref="EngineTelemetry.SetKvCacheUtilizationProvider"/>) that report this scheduler's
    /// live state. Cleared back to <c>null</c> on <see cref="Dispose"/>. Set to <see langword="false"/>
    /// for multi-scheduler hosts that want to wire the providers themselves.</param>
    public ContinuousBatchSchedulerService(
        IModel model,
        ITokenizer tokenizer,
        Func<ModelConfig, int, IKvCache> kvCacheFactory,
        ContinuousBatchSchedulerOptions? options = null,
        KvBlockPool? pagedPool = null,
        PrefixTrieManager? prefixCache = null,
        bool registerTelemetryProviders = true)
    {
        _inner = new ContinuousBatchScheduler(model, tokenizer, kvCacheFactory, options, pagedPool, prefixCache);
        _pagedPool = pagedPool;
        _ownsTelemetryProviders = registerTelemetryProviders;

        if (registerTelemetryProviders)
        {
            EngineTelemetry.SetQueueDepthProvider(() => _inner.QueueDepth + _inner.ActiveCount);
            if (_pagedPool is not null)
            {
                int total = _pagedPool.TotalBlocks;
                if (total > 0)
                {
                    EngineTelemetry.SetKvCacheUtilizationProvider(
                        () => 1.0 - (double)_pagedPool.FreeBlocks / total);
                }
            }
        }
    }

    /// <inheritdoc/>
    public Task<InferenceResponse> EnqueueAsync(InferenceRequest request, CancellationToken cancellationToken = default)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchSchedulerService));

        var handle = _inner.Submit(request, cancellationToken);
        _wakeup.Release(); // wake the run loop if it's idle
        return handle.Completion;
    }

    /// <inheritdoc/>
    public async Task RunLoopAsync(CancellationToken cancellationToken)
    {
        // Sustained low-latency GC mode mirrors the recommendation in CLAUDE.md for inference
        // hot paths. Saved/restored so the caller's prior latency mode is preserved.
        var priorMode = System.Runtime.GCSettings.LatencyMode;
        try
        {
            System.Runtime.GCSettings.LatencyMode = System.Runtime.GCLatencyMode.SustainedLowLatency;

            while (!cancellationToken.IsCancellationRequested)
            {
                if (_inner.IsIdle)
                {
                    try
                    {
                        await _wakeup.WaitAsync(cancellationToken).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        return;
                    }
                }

                // Drain pending work in a tight loop; yield only when idle.
                while (!cancellationToken.IsCancellationRequested && !_inner.IsIdle)
                {
                    bool didWork = _inner.Step();
                    if (!didWork)
                    {
                        // Defensive yield — Step returning false on a non-idle scheduler shouldn't
                        // happen, but avoid pegging a core if it ever does.
                        await Task.Yield();
                    }
                }
            }
        }
        finally
        {
            try { System.Runtime.GCSettings.LatencyMode = priorMode; }
            catch { /* restore best-effort */ }
        }
    }

    /// <inheritdoc/>
    public SchedulerMetrics GetMetrics() => _inner.GetMetrics() with
    {
        // Metrics shape is the engine's existing public contract — pass through unchanged.
        // PreemptionCount is reserved for the eventual preempt-on-pressure path (Step 59).
    };

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsTelemetryProviders)
        {
            EngineTelemetry.SetQueueDepthProvider(null);
            EngineTelemetry.SetKvCacheUtilizationProvider(null);
        }
        _inner.Dispose();
        _wakeup.Dispose();
    }
}
