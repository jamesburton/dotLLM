using System.Collections.Concurrent;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Prefill/decode-disaggregated scheduler: a <b>prefill worker</b> and a <b>decode worker</b> run as
/// two independent <see cref="ContinuousBatchScheduler"/> instances (separate model replicas of the
/// same weights) over a <b>shared paged KV pool</b>. A sequence is prefilled by the prefill replica,
/// then its KV-cache is <em>handed off</em> (no copy — the cache lives in the shared pool) to the
/// decode replica, which runs it to completion.
/// </summary>
/// <remarks>
/// <para>This builds on the <see cref="ContinuousBatchScheduler.StepPrefill"/> /
/// <see cref="ContinuousBatchScheduler.StepDecode"/> seam. Because each replica is a distinct
/// <see cref="IModel"/> instance with its own forward scratch, the two phases can overlap on separate
/// threads (genuine parallelism on multi-core CPU / future multi-device) — unlike a single scheduler,
/// whose model forward is single-threaded.</para>
/// <para><b>KV handoff is by reference.</b> The prefill replica allocates the KV-cache from the shared
/// factory/pool and populates it; the decode replica attends over the same cache object and appends to
/// it. Both replicas must be the same architecture (identical KV layout) — the contract for a replica
/// pair.</para>
/// <para><b>Driving.</b> <see cref="Step"/> drives one synchronous iteration (prefill → handoff →
/// decode) on the calling thread — deterministic, used by tests and simple hosts.
/// <see cref="RunLoopAsync"/> drives the two phases on separate worker tasks with a lock-free handoff
/// queue — the production overlap path.</para>
/// <para><b>Backpressure</b> comes from the shared KV pool: when blocks run low, the prefill replica's
/// admission gate (<see cref="ContinuousBatchSchedulerOptions.ReserveBlocksPerSequence"/>) stops
/// admitting until the decode replica frees blocks on completion.</para>
/// </remarks>
public sealed class DisaggregatedScheduler : IScheduler, IBatchScheduler, IDisposable
{
    private readonly ContinuousBatchScheduler _prefill;
    private readonly ContinuousBatchScheduler _decode;

    // KV-handoff strategy + the decode replica's cache factory. For the default (reference) transfer the
    // factory is the shared one and the strategy is a no-op pass-through; for the copy transfer the
    // factory allocates from the decode replica's own pool and the strategy copies the K/V across.
    private readonly IKvHandoffTransfer _handoffTransfer;
    private readonly Func<ModelConfig, int, IKvCache> _decodeKvCacheFactory;
    private readonly ModelConfig _config;

    // Lock-free handoff channel: the prefill worker enqueues post-prefill sequences; the decode worker
    // dequeues and injects them. Only used by the async RunLoopAsync path (Step does the move inline).
    private readonly ConcurrentQueue<SchedulerRequest> _handoff = new();
    private readonly List<SchedulerRequest> _extractBuf = new();

    private readonly SemaphoreSlim _prefillWakeup = new(initialCount: 0, maxCount: int.MaxValue);
    private readonly SemaphoreSlim _decodeWakeup = new(initialCount: 0, maxCount: int.MaxValue);
    private bool _disposed;

    /// <summary>The prefill-worker scheduler (admits + prefills). Exposed for advanced callers/tests.</summary>
    public ContinuousBatchScheduler Prefill => _prefill;

    /// <summary>The decode-worker scheduler (decodes handed-off sequences). Exposed for advanced callers/tests.</summary>
    public ContinuousBatchScheduler Decode => _decode;

    /// <summary>
    /// Creates a disaggregated scheduler over two model replicas sharing one KV pool.
    /// </summary>
    /// <param name="prefillModel">Model replica driven by the prefill worker.</param>
    /// <param name="decodeModel">Model replica driven by the decode worker. Must be the same
    /// architecture/weights as <paramref name="prefillModel"/> (so it can attend the prefill replica's KV).</param>
    /// <param name="tokenizer">Tokenizer for response detokenization (shared).</param>
    /// <param name="sharedKvCacheFactory">KV-cache factory backed by the shared pool — both replicas use it.</param>
    /// <param name="options">Scheduler options applied to both replicas.</param>
    /// <param name="sharedPagedPool">The shared paged-block pool (admission gating + handoff backing).</param>
    /// <param name="prefixCache">Optional prefix cache — wired to the prefill replica (admission) only.</param>
    /// <param name="handoffTransfer">KV-handoff strategy. Defaults to <see cref="ReferenceKvHandoffTransfer"/>
    /// (zero-copy, shared pool). Pass <see cref="CopyKvHandoffTransfer"/> together with a distinct
    /// <paramref name="decodeKvCacheFactory"/>/<paramref name="decodePagedPool"/> to copy the KV across
    /// separate pools (the cross-device-transfer simulation).</param>
    /// <param name="decodeKvCacheFactory">KV-cache factory for the <em>decode</em> replica. When null, the
    /// decode replica shares <paramref name="sharedKvCacheFactory"/> (the by-reference default). Provide a
    /// factory backed by a separate pool to give the decode replica its own pool (required for copy transfer).</param>
    /// <param name="decodePagedPool">The decode replica's own paged-block pool. When null, the decode
    /// replica shares <paramref name="sharedPagedPool"/>. Provide a separate pool for copy transfer.</param>
    public DisaggregatedScheduler(
        IModel prefillModel,
        IModel decodeModel,
        ITokenizer tokenizer,
        Func<ModelConfig, int, IKvCache> sharedKvCacheFactory,
        ContinuousBatchSchedulerOptions? options = null,
        KvBlockPool? sharedPagedPool = null,
        PrefixTrieManager? prefixCache = null,
        IKvHandoffTransfer? handoffTransfer = null,
        Func<ModelConfig, int, IKvCache>? decodeKvCacheFactory = null,
        KvBlockPool? decodePagedPool = null)
    {
        ArgumentNullException.ThrowIfNull(prefillModel);
        ArgumentNullException.ThrowIfNull(decodeModel);
        _handoffTransfer = handoffTransfer ?? ReferenceKvHandoffTransfer.Instance;
        _config = decodeModel.Config;
        // The decode replica uses its own factory/pool when supplied (copy transfer), else the shared ones.
        _decodeKvCacheFactory = decodeKvCacheFactory ?? sharedKvCacheFactory;
        var decodePool = decodePagedPool ?? sharedPagedPool;
        _prefill = new ContinuousBatchScheduler(prefillModel, tokenizer, sharedKvCacheFactory, options, sharedPagedPool, prefixCache);
        // The decode replica never admits from a queue (it only receives handoffs), so it gets no prefix cache.
        _decode = new ContinuousBatchScheduler(decodeModel, tokenizer, _decodeKvCacheFactory, options, decodePool, prefixCache: null);
    }

    /// <inheritdoc/>
    public ISchedulerRequest Submit(InferenceRequest request, CancellationToken cancellationToken = default)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DisaggregatedScheduler));
        var handle = _prefill.Submit(request, cancellationToken);
        _prefillWakeup.Release();
        return handle;
    }

    /// <inheritdoc/>
    public Task<InferenceResponse> EnqueueAsync(InferenceRequest request, CancellationToken cancellationToken = default)
        => Submit(request, cancellationToken).Completion;

    /// <summary>
    /// Drives one synchronous iteration: a prefill step on the prefill replica, an inline handoff of
    /// every just-prefilled sequence to the decode replica, then a decode step on the decode replica.
    /// Returns whether any work was performed.
    /// </summary>
    public bool Step()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DisaggregatedScheduler));

        bool didWork = _prefill.StepPrefill();

        _extractBuf.Clear();
        _prefill.ExtractDecodable(_extractBuf);
        for (int i = 0; i < _extractBuf.Count; i++)
        {
            ApplyHandoff(_extractBuf[i]);
            _decode.InjectDecodable(_extractBuf[i]);
        }
        if (_extractBuf.Count > 0) didWork = true;

        didWork |= _decode.StepDecode();
        return didWork;
    }

    /// <summary>
    /// Applies the configured <see cref="IKvHandoffTransfer"/> to a just-prefilled sequence, swapping its
    /// KV-cache for the one the decode replica should use. For the reference transfer this is a no-op
    /// (same shared cache object); for the copy transfer it replaces the prefill-pool cache with a fresh
    /// decode-pool copy and disposes the old one. A sequence without a KV-cache (e.g. recurrent-only) is
    /// left untouched.
    /// </summary>
    private void ApplyHandoff(SchedulerRequest seq)
    {
        var cache = seq.KvCache;
        if (cache is null) return;
        seq.KvCache = _handoffTransfer.Transfer(cache, _config, _decodeKvCacheFactory);
    }

    /// <inheritdoc/>
    public bool IsIdle => _prefill.IsIdle && _decode.IsIdle && _handoff.IsEmpty;

    /// <inheritdoc/>
    public int ActiveCount => _prefill.ActiveCount + _decode.ActiveCount + _handoff.Count;

    /// <inheritdoc/>
    public int QueueDepth => _prefill.QueueDepth;

    /// <inheritdoc/>
    public SchedulerMetrics GetMetrics()
    {
        var p = _prefill.GetMetrics();
        var d = _decode.GetMetrics();
        return new SchedulerMetrics(
            ActiveSequences: p.ActiveSequences + d.ActiveSequences + _handoff.Count,
            QueueDepth: p.QueueDepth,
            PreemptionCount: p.PreemptionCount + d.PreemptionCount);
    }

    /// <summary>Merged per-API-key generated-token usage across both replicas (sequences that stop
    /// during prefill complete on the prefill replica; the rest complete on the decode replica).</summary>
    public IReadOnlyDictionary<string, long> GetPerKeyTokenUsage()
    {
        var merged = new Dictionary<string, long>(_prefill.GetPerKeyTokenUsage());
        foreach (var kv in _decode.GetPerKeyTokenUsage())
            merged[kv.Key] = merged.GetValueOrDefault(kv.Key) + kv.Value;
        return merged;
    }

    /// <summary>
    /// Runs the prefill and decode workers on two background tasks until <paramref name="cancellationToken"/>
    /// fires. The prefill worker admits + prefills and hands sequences to the decode worker via the
    /// lock-free queue; the decode worker injects and decodes them. Each worker sleeps on its wakeup
    /// semaphore when it has no work, so an idle driver consumes no CPU.
    /// </summary>
    public async Task RunLoopAsync(CancellationToken cancellationToken)
    {
        var prefillTask = Task.Run(() => PrefillWorkerAsync(cancellationToken), cancellationToken);
        var decodeTask = Task.Run(() => DecodeWorkerAsync(cancellationToken), cancellationToken);
        await Task.WhenAll(prefillTask, decodeTask).ConfigureAwait(false);
    }

    private async Task PrefillWorkerAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            // Idle when nothing is queued and nothing is mid-prefill.
            if (_prefill.IsIdle)
            {
                try { await _prefillWakeup.WaitAsync(ct).ConfigureAwait(false); }
                catch (OperationCanceledException) { return; }
            }

            while (!ct.IsCancellationRequested && !_prefill.IsIdle)
            {
                _prefill.StepPrefill();
                _extractBuf.Clear();
                _prefill.ExtractDecodable(_extractBuf);
                if (_extractBuf.Count > 0)
                {
                    for (int i = 0; i < _extractBuf.Count; i++)
                    {
                        ApplyHandoff(_extractBuf[i]);
                        _handoff.Enqueue(_extractBuf[i]);
                    }
                    _decodeWakeup.Release();
                }
            }
        }
    }

    private async Task DecodeWorkerAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            if (_decode.ActiveCount == 0 && _handoff.IsEmpty)
            {
                try { await _decodeWakeup.WaitAsync(ct).ConfigureAwait(false); }
                catch (OperationCanceledException) { return; }
            }

            while (!ct.IsCancellationRequested && (!_handoff.IsEmpty || _decode.ActiveCount > 0))
            {
                while (_handoff.TryDequeue(out var seq))
                    _decode.InjectDecodable(seq);
                if (!_decode.StepDecode())
                    break; // no active decoders left; wait for the next handoff
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Drain any in-flight handoffs into the decode scheduler so its Dispose cancels them.
        while (_handoff.TryDequeue(out var seq))
            _decode.InjectDecodable(seq);
        _prefill.Dispose();
        _decode.Dispose();
        _prefillWakeup.Dispose();
        _decodeWakeup.Dispose();
    }
}
