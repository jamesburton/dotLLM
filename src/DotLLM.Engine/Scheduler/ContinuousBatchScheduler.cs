using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Telemetry;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Continuous-batching scheduler — MVP implementation.
/// </summary>
/// <remarks>
/// <para>This MVP performs <em>iteration-level</em> scheduling: every <see cref="Step"/> processes
/// admissions and one decode token across all active sequences. The KV-cache pool is shared
/// across sequences (via a caller-supplied factory such as <see cref="PagedKvCacheFactory"/>),
/// so sequences can be admitted as soon as blocks become free.</para>
///
/// <para><b>What this MVP does NOT do</b> (deferred to follow-on roadmap steps):</para>
/// <list type="bullet">
///   <item>Kernel-level batched forward across sequences. We currently call
///   <see cref="IModel.Forward(System.ReadOnlySpan{int},System.ReadOnlySpan{int},int,IKvCache?)"/>
///   once per sequence per iteration. The model still benefits from KV-cache pooling and
///   request-admission overlap, but per-sequence GEMV is not yet batched. Step 59
///   (advanced scheduling) is the right place for that.</item>
///   <item>Chunked prefill. A long prompt is prefilled in a single forward pass during admission.
///   See <see cref="ContinuousBatchSchedulerOptions.MaxPrefillTokensPerStep"/> for the partial
///   admission cap that mitigates head-of-line blocking at admission time only.</item>
///   <item>Host-memory KV swap. When <see cref="ContinuousBatchSchedulerOptions.EnablePreemption"/>
///   is set, the scheduler preempts lower-priority active sequences under block pressure and
///   <em>recomputes</em> their KV-cache on resume (Step 59). CPU-offload of KV blocks (faster resume,
///   uses host RAM) is the remaining swap strategy.</item>
///   <item>Streaming yield. Generated tokens accumulate inside the scheduler and surface only
///   when <see cref="ISchedulerRequest.Completion"/> resolves. Streaming through a
///   <c>ChannelWriter&lt;GenerationToken&gt;</c> per request is straightforward to add but
///   out of scope for the MVP.</item>
/// </list>
///
/// <para>Thread-safety: <see cref="Submit"/> may be called from any thread (queue is
/// <see cref="ConcurrentQueue{T}"/>). <see cref="Step"/> must be driven by a single thread —
/// typically the server's run-loop task.</para>
/// </remarks>
public sealed class ContinuousBatchScheduler : IBatchScheduler, IDisposable
{
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly Func<ModelConfig, int, IKvCache> _kvCacheFactory;
    private readonly ContinuousBatchSchedulerOptions _options;
    private readonly KvBlockPool? _pagedPool;
    private readonly PrefixTrieManager? _prefixCache;
    // True when the model carries per-sequence recurrent state the scheduler allocates and threads
    // (Mamba-3 / Qwen3-MoE-Hybrid GDN). Such models dispatch ALL forwards (even single-sequence) via
    // ForwardBatch — the only IModel entrypoint that carries the per-seq state — which both enables
    // batched recurrent decode/prefill and fixes cross-sequence corruption from a shared default state.
    private readonly bool _supportsThreadedState;
    private long _cachedPromptTokens;
    private long _prefilledPromptTokens;

    // Priority-ordered queue. Element key is (-(int)Priority, submissionOrder), so the min-heap
    // pops the highest RequestPriority first (Critical < High < Normal < Low by enum value, negated)
    // and FIFO-orders among the same priority via submissionOrder ascending. PriorityQueue is not
    // thread-safe; Submit/Step both serialise mutations through _queueLock — the lock is uncontended
    // in the steady-state single-Step-loop pattern, and Submit happens once per request.
    private readonly PriorityQueue<SchedulerRequest, (int PriorityRank, long FairTag, long Order)> _pendingQueue = new();
    private readonly Lock _queueLock = new();
    private readonly List<SchedulerRequest> _active = new();

    // Start-time fair-queuing (SFQ) state, all guarded by _queueLock. _keyFinishTag holds each fairness
    // key's running cumulative cost (finish tag); _virtualTime is the virtual clock advanced to the
    // start tag of the most-recently-admitted request. Only used when _options.EnableFairness. See
    // ContinuousBatchSchedulerOptions.EnableFairness for the algorithm.
    private readonly Dictionary<string, long> _keyFinishTag = new();
    private long _virtualTime;
    private const string AnonymousFairnessKey = "\0anonymous";

    // Per-API-key cumulative generated-token accounting (observability). Updated once per completed
    // request that carries an ApiKey, under _keyTokensLock; snapshot via GetPerKeyTokenUsage(). Also
    // emitted to the EngineTelemetry.TokensByKey counter (zero-overhead when no listener).
    private readonly Dictionary<string, long> _keyTokensServed = new();
    private readonly Lock _keyTokensLock = new();

    // Reusable scratch for batched decode (Step 59). Cleared/refilled each Step; persisted as
    // fields to avoid per-iteration allocation on the decode hot path. _decodeReady holds the
    // active decoders that passed the capacity gate; _decodeBatch is the ForwardBatch request
    // list; _decodeTokens/_decodePositions back each request's 1-element TokenIds/Positions.
    private readonly List<SchedulerRequest> _decodeReady = new();
    private readonly List<SequenceForwardRequest> _decodeBatch = new();
    private int[] _decodeTokens = Array.Empty<int>();
    private int[] _decodePositions = Array.Empty<int>();

    // Reusable scratch for batched prefill (Step 59, follow-up to batched decode). The admission
    // loop prepares each newly-admitted (non-resuming) sequence — KV-cache alloc + prefix seeding +
    // forward-token range — WITHOUT forwarding, collecting them in _prefillReady; after the loop a
    // single fused ForwardBatch runs over all of them when the model is stateless and ≥2 are pending.
    // _prefillBatch is the ForwardBatch request list; _prefillPositions packs every pending prefill's
    // contiguous position range (forwardStart .. forwardStart+forwardLen-1) so each request can hand
    // ForwardBatch a stable slice without per-sequence allocation.
    private readonly List<PendingPrefill> _prefillReady = new();
    private readonly List<SequenceForwardRequest> _prefillBatch = new();
    private int[] _prefillPositions = Array.Empty<int>();

    private long _submissionCounter;
    // Number of preemptions performed (Step 59 advanced scheduling). Incremented when a lower-priority
    // active sequence is evicted under block pressure to admit a higher-priority request; the victim is
    // re-queued and resumes via recompute (see TryPreemptForPressure). Surfaced via GetMetrics().
    private long _preemptionCount;
    private bool _disposed;

    /// <summary>Number of sequences ever submitted to this scheduler.</summary>
    public long TotalSubmitted => Interlocked.Read(ref _submissionCounter);

    /// <inheritdoc/>
    public int ActiveCount => _active.Count;

    /// <inheritdoc/>
    public int QueueDepth
    {
        get { lock (_queueLock) { return _pendingQueue.Count; } }
    }

    /// <inheritdoc/>
    public bool IsIdle
    {
        get { lock (_queueLock) { return _active.Count == 0 && _pendingQueue.Count == 0; } }
    }

    /// <summary>
    /// Creates a new continuous-batch scheduler.
    /// </summary>
    /// <param name="model">The transformer model to run. Forward is invoked once per active
    /// sequence per <see cref="Step"/>.</param>
    /// <param name="tokenizer">Tokenizer for decoding generated tokens into the final response text.</param>
    /// <param name="kvCacheFactory">Factory returning a fresh per-sequence KV-cache. For paged
    /// caching, pass <see cref="PagedKvCacheFactory.Create(int)"/> wrapped in a delegate.</param>
    /// <param name="options">Optional scheduler options.</param>
    /// <param name="pagedPool">Optional reference to the underlying paged-block pool. When provided,
    /// admission uses pool free-block count for capacity gating in addition to
    /// <see cref="ContinuousBatchSchedulerOptions.MaxActiveSequences"/>.</param>
    /// <param name="prefixCache">Optional cross-request prefix cache. When provided, admission
    /// seeds new KV-caches from the trie and routes completions back to it (Step 37).</param>
    public ContinuousBatchScheduler(
        IModel model,
        ITokenizer tokenizer,
        Func<ModelConfig, int, IKvCache> kvCacheFactory,
        ContinuousBatchSchedulerOptions? options = null,
        KvBlockPool? pagedPool = null,
        PrefixTrieManager? prefixCache = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(kvCacheFactory);
        _model = model;
        _tokenizer = tokenizer;
        _kvCacheFactory = kvCacheFactory;
        _options = options ?? new ContinuousBatchSchedulerOptions();
        _pagedPool = pagedPool;
        _prefixCache = prefixCache;
        _supportsThreadedState = model.SupportsThreadedSequenceState;
    }

    /// <summary>
    /// Whether a forward over <paramref name="count"/> sequences should be dispatched as a single
    /// <see cref="IModel.ForwardBatch"/> rather than per-sequence <c>IModel.Forward</c>:
    /// recurrent threaded-state models always batch (count ≥ 1 — only <c>ForwardBatch</c> carries the
    /// per-seq state); dense models fuse only at ≥2; recurrent hosts without threadable state never batch.
    /// </summary>
    private bool ShouldBatch(int count) =>
        _supportsThreadedState ? count >= 1 : (!_model.RequiresPerSequenceState && count >= 2);

    /// <summary>Maximum sequences that may be active at once — <see cref="ContinuousBatchSchedulerOptions.MaxActiveSequences"/>,
    /// further clamped by <see cref="ContinuousBatchSchedulerOptions.MaxRecurrentSequences"/> for a
    /// threaded-state recurrent model (bounds aggregate per-seq recurrent-state memory).</summary>
    private int EffectiveMaxActive =>
        _supportsThreadedState && _options.MaxRecurrentSequences > 0
            ? Math.Min(_options.MaxActiveSequences, _options.MaxRecurrentSequences)
            : _options.MaxActiveSequences;

    /// <summary>Cumulative prompt tokens served from the prefix cache (no prefill needed).</summary>
    public long CachedPromptTokens => Interlocked.Read(ref _cachedPromptTokens);

    /// <summary>Cumulative prompt tokens that required prefill compute.</summary>
    public long PrefilledPromptTokens => Interlocked.Read(ref _prefilledPromptTokens);

    /// <inheritdoc/>
    public ISchedulerRequest Submit(InferenceRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));

        // Validate prompt non-empty (mirrors TextGenerator guard but throws here — scheduler
        // callers are server endpoints that have already validated input).
        var promptIds = request.TokenIds;
        if (promptIds is null || promptIds.Length == 0)
            throw new ArgumentException("Request prompt must contain at least one token.", nameof(request));

        var options = request.Options;
        int maxTokens = Math.Max(1, options.MaxTokens);

        // Sampler pipeline (per-sequence — pipelines hold RNG state, so we can't share).
        var pipeline = new SamplerPipeline(options);

        // Build stop conditions. Mirrors TextGenerator: explicit list if set, else EOS + MaxTokens + stop strings.
        IReadOnlyList<IStopCondition> stops;
        if (options.StopConditions is not null)
        {
            stops = options.StopConditions;
        }
        else
        {
            var list = new List<IStopCondition>(capacity: 2 + options.StopSequences.Count)
            {
                new EosStopCondition(_tokenizer.EosTokenId),
                new MaxTokensStopCondition(maxTokens),
            };
            foreach (var stopSeq in options.StopSequences)
                list.Add(new StopStringCondition(stopSeq));
            stops = list;
        }

        // Decoding constraint for structured output (JSON / schema / regex / grammar).
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
            ResponseFormat.Regex rx => new RegexConstraint(_tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(_tokenizer, gr.GbnfGrammar),
            _ => null,
        };

        var tcs = new TaskCompletionSource<InferenceResponse>(
            TaskCreationOptions.RunContinuationsAsynchronously);

        var seq = new SchedulerRequest(
            request,
            new InferenceOptionsLike(maxTokens, options.Logprobs, options.TopLogprobs),
            promptLength: promptIds.Length,
            maxTokens: maxTokens,
            samplerPipeline: pipeline,
            stopConditions: stops,
            constraint: constraint,
            submissionOrder: Interlocked.Increment(ref _submissionCounter),
            tcs: tcs);

        if (cancellationToken.CanBeCanceled)
        {
            seq.CancellationRegistration = cancellationToken.Register(static state =>
            {
                var s = (SchedulerRequest)state!;
                // We cannot safely free the KV-cache here — the scheduler thread may be in the
                // middle of decoding this sequence. Mark for cancellation; Step() will collect it.
                s.State = SequenceState.Cancelled;
                s.CompletionSource.TrySetCanceled();
            }, seq);
        }

        // Priority key: most-negative RequestPriority pops first (Critical=3 → -3, Low=0 → 0); within a
        // tier, order by the SFQ start tag (0 for everyone when fairness is off ⇒ pure FIFO), then by
        // submissionOrder as the final FIFO tie-break. Fairness state is computed under _queueLock.
        lock (_queueLock)
        {
            if (_options.EnableFairness)
                seq.FairnessTag = ComputeFairnessStartTag(request.ApiKey, cost: promptIds.Length + maxTokens);
            _pendingQueue.Enqueue(seq, (-(int)request.Priority, seq.FairnessTag, seq.SubmissionOrder));
        }
        return seq;
    }

    /// <summary>
    /// Computes a request's start-time-fair-queuing start tag and charges its key's running finish tag.
    /// Must be called under <see cref="_queueLock"/>. An idle key starts at the current virtual clock
    /// (recent-usage forgiveness); a backlogged key's tags grow by <paramref name="cost"/> each request.
    /// </summary>
    private long ComputeFairnessStartTag(string? apiKey, long cost)
    {
        string key = apiKey ?? AnonymousFairnessKey;
        long start = _keyFinishTag.TryGetValue(key, out long finish) ? Math.Max(_virtualTime, finish) : _virtualTime;
        _keyFinishTag[key] = start + Math.Max(1, cost);

        // Opportunistic prune: drop keys that have fully caught up to the virtual clock (idle/served),
        // bounding the map for long-running multi-tenant servers. Only when it has grown past a cap.
        if (_keyFinishTag.Count > 1024)
        {
            var stale = new List<string>();
            foreach (var kv in _keyFinishTag)
                if (kv.Value <= _virtualTime && kv.Key != key) stale.Add(kv.Key);
            foreach (var k in stale) _keyFinishTag.Remove(k);
        }
        return start;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// One full iteration = the prefill phase then the decode phase, run in sequence (so a sequence
    /// admitted this iteration also decodes once this iteration, as before). The two phases are also
    /// exposed individually as <see cref="StepPrefill"/> / <see cref="StepDecode"/> — the
    /// prefill/decode-disaggregation seam: a disaggregated/multi-worker driver can run them on
    /// separate thread pools (or model replicas, with KV transfer between), while this in-process
    /// single-GPU driver runs both here because the model forward is single-threaded.
    /// </remarks>
    public bool Step()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));

        bool didWork = SweepCancelled();
        didWork |= AdmitAndPrefillPhase();
        didWork |= DecodePhase();
        return didWork;
    }

    /// <summary>
    /// The <b>prefill</b> half of <see cref="Step"/>: sweeps cancellations, then admits queued requests
    /// from the prefill queue (subject to capacity / block pressure) and runs their deferred prompt
    /// prefill. Exposed for prefill/decode disaggregation — a prefill worker drives this; the in-process
    /// driver calls <see cref="Step"/> which runs prefill then decode.
    /// </summary>
    /// <returns><see langword="true"/> if any admission/prefill/sweep work was performed.</returns>
    public bool StepPrefill()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));
        bool didWork = SweepCancelled();
        didWork |= AdmitAndPrefillPhase();
        return didWork;
    }

    /// <summary>
    /// The <b>decode</b> half of <see cref="Step"/>: sweeps cancellations, then advances one decode
    /// token for every actively-decoding sequence (the decode set is the implicit decode queue).
    /// Exposed for prefill/decode disaggregation — a decode worker drives this.
    /// </summary>
    /// <returns><see langword="true"/> if any decode/sweep work was performed.</returns>
    public bool StepDecode()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));
        bool didWork = SweepCancelled();
        didWork |= DecodePhase();
        return didWork;
    }

    /// <summary>Sweeps caller-cancelled active sequences (cancellation may have flipped state),
    /// releasing their KV-cache. Idempotent — safe to run at the start of either phase.</summary>
    private bool SweepCancelled()
    {
        bool didWork = false;
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var s = _active[i];
            if (s.State == SequenceState.Cancelled)
            {
                ReleaseKvCache(s);
                _active.RemoveAt(i);
                didWork = true;
            }
        }
        return didWork;
    }

    /// <summary>
    /// Admits queued requests (priority-ordered, capacity/block-pressure gated) and runs their deferred
    /// prompt prefill. Newly-admitted sequences are PREPARED (KV-cache alloc + prefix seeding + forward
    /// range + per-seq recurrent state) but their prefill forward is DEFERRED to <see cref="PrefillReadySequences"/>,
    /// where all sequences admitted this call fuse into one <see cref="IModel.ForwardBatch"/> when batchable.
    /// Resuming (preempted) sequences recompute inline. Queue draining is priority-ordered: highest
    /// <see cref="RequestPriority"/> first, FIFO within tier (see <c>_pendingQueue</c> key shape).
    /// </summary>
    private bool AdmitAndPrefillPhase()
    {
        bool didWork = false;
        _prefillReady.Clear();
        int admittedThisStep = 0;
        int prefillTokensThisStep = 0;
        // Blocks tentatively reserved for prepared-but-not-yet-forwarded prefills this step. Each
        // prepared prefill's blocks are not consumed from the pool until its deferred forward runs,
        // so the per-iteration block gate must subtract this running reservation from FreeBlocks to
        // keep its tight-pressure behaviour (≤1 admit/step when pool-constrained) intact.
        int reservedBlocksThisStep = 0;
        int maxActive = EffectiveMaxActive;
        while (_active.Count + _prefillReady.Count < maxActive)
        {
            SchedulerRequest? head;
            lock (_queueLock)
            {
                if (!_pendingQueue.TryPeek(out head, out _)) break;
            }

            // Cancelled while queued — drop without admission.
            if (head.State == SequenceState.Cancelled)
            {
                lock (_queueLock) { _pendingQueue.TryDequeue(out _, out _); }
                continue;
            }

            // Bound prefill cost per step. Once we've started admitting, finishing the head is fine
            // unless this step has already done meaningful prefill work.
            if (_options.MaxPrefillTokensPerStep > 0 &&
                admittedThisStep > 0 &&
                prefillTokensThisStep + head.PromptLength > _options.MaxPrefillTokensPerStep)
            {
                break;
            }

            // Block-pool gating: refuse admission if the paged pool can't fit the worst-case
            // footprint, accounting for blocks already reserved by this step's pending prefills.
            // Try to relieve pressure by evicting zero-refcount trie blocks first, then — when
            // preemption is enabled — by evicting lower-priority active sequences.
            if (_pagedPool is not null && _options.ReserveBlocksPerSequence > 0)
            {
                int needed = reservedBlocksThisStep + _options.ReserveBlocksPerSequence;
                if (_pagedPool.FreeBlocks < needed)
                {
                    int short_ = needed - _pagedPool.FreeBlocks;
                    if (_prefixCache is not null)
                        _prefixCache.TryEvict(short_);

                    if (_pagedPool.FreeBlocks < needed)
                    {
                        // Preempt the lowest-priority active sequence(s) strictly below this request's
                        // priority to free blocks. Victims are re-queued and resume via recompute.
                        if (_options.EnablePreemption &&
                            TryPreemptForPressure(head, needed))
                        {
                            // Freed enough — re-evaluate the (possibly mutated) queue head and capacity.
                            continue;
                        }
                        break;
                    }
                }
            }

            SchedulerRequest? seq;
            lock (_queueLock)
            {
                if (!_pendingQueue.TryDequeue(out seq, out _)) break;
                // Admitting a request advances the SFQ virtual clock to its start tag, so subsequently
                // submitted requests from idle keys are forgiven up to here (no-op when fairness is off).
                if (_options.EnableFairness && seq.FairnessTag > _virtualTime)
                    _virtualTime = seq.FairnessTag;
            }

            try
            {
                if (seq.IsResuming)
                {
                    // Re-admitted after preemption: rebuild KV from prompt + generated tokens.
                    // No first-token sample (output already has tokens); goes straight to Decoding.
                    // Recompute is inline (not batched) — preemption is an off-by-default edge path.
                    AdmitAndResume(seq);
                    _active.Add(seq);
                    didWork = true;
                }
                else
                {
                    // Prepare only; the forward is fused in step 2b.
                    var pending = PreparePrefill(seq);
                    _prefillReady.Add(pending);
                    reservedBlocksThisStep += _options.ReserveBlocksPerSequence;
                }
                admittedThisStep++;
                prefillTokensThisStep += seq.PromptLength;
            }
            catch (OperationCanceledException)
            {
                seq.State = SequenceState.Cancelled;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetCanceled();
            }
            catch (Exception ex)
            {
                seq.State = SequenceState.Completed;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetException(ex);
            }
        }

        // Run the deferred prefill forwards prepared above — fused via ForwardBatch when batchable,
        // else per-sequence. Samples each first token, checks stop conditions, and routes each
        // sequence to active / completion.
        if (_prefillReady.Count > 0)
        {
            PrefillReadySequences();
            didWork = true;
        }

        return didWork;
    }

    /// <summary>
    /// Advances one decode token for each actively-decoding sequence (the decode set is the implicit
    /// decode queue). When batchable (<see cref="ShouldBatch"/>) the ready decoders fuse into a single
    /// <see cref="IModel.ForwardBatch"/>; otherwise each decodes per-sequence. Capacity-gated sequences
    /// finish here without a forward.
    /// </summary>
    private bool DecodePhase()
    {
        bool didWork = false;
        _decodeReady.Clear();
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var seq = _active[i];
            if (seq.State != SequenceState.Decoding) continue;

            if (!TryStartDecode(seq, out _))
            {
                // Length / max-tokens cap: finish without a forward.
                seq.State = SequenceState.Completed;
                CompleteSequence(seq);
                _active.RemoveAt(i);
                didWork = true;
                continue;
            }
            _decodeReady.Add(seq);
        }

        int readyCount = _decodeReady.Count;
        if (readyCount > 0)
            didWork = true;

        if (ShouldBatch(readyCount))
        {
            DecodeBatched(readyCount);
        }
        else
        {
            // Single dense decoder, or a recurrent host without threadable state (Nemotron-H) whose
            // ForwardBatch can't batch without per-seq state. Decode each sequence independently via
            // the per-seq Forward (model-owned default state).
            for (int j = 0; j < readyCount; j++)
                DecodeSingleAndFinish(_decodeReady[j]);
        }

        return didWork;
    }

    /// <summary>Fuses the <paramref name="readyCount"/> ready decoders in <see cref="_decodeReady"/>
    /// into one <see cref="IModel.ForwardBatch"/> call, then samples / stops each independently.</summary>
    private void DecodeBatched(int readyCount)
    {
        if (_decodeTokens.Length < readyCount)
        {
            _decodeTokens = new int[readyCount];
            _decodePositions = new int[readyCount];
        }
        _decodeBatch.Clear();
        for (int j = 0; j < readyCount; j++)
        {
            var seq = _decodeReady[j];
            _decodeTokens[j] = seq.GeneratedTokens[^1];
            _decodePositions[j] = seq.Position - 1;
            _decodeBatch.Add(new SequenceForwardRequest
            {
                TokenIds = _decodeTokens.AsMemory(j, 1),
                Positions = _decodePositions.AsMemory(j, 1),
                KvCache = seq.KvCache!,
                MambaState = seq.RecurrentState as IMambaState,
                GdnState = seq.RecurrentState as IGdnState,
            });
        }

        IReadOnlyList<ITensor> results;
        long fwdStart = Stopwatch.GetTimestamp();
        try
        {
            results = _model.ForwardBatch(_decodeBatch, deviceId: -1);
        }
        catch (OperationCanceledException)
        {
            FailReadyDecoders(readyCount, ex: null);
            return;
        }
        catch (Exception ex)
        {
            FailReadyDecoders(readyCount, ex);
            return;
        }
        long perSeqTicks = (Stopwatch.GetTimestamp() - fwdStart) / readyCount;

        for (int j = 0; j < readyCount; j++)
        {
            var seq = _decodeReady[j];
            ITensor logits = results[j];
            seq.DecodeTicks += perSeqTicks;
            try
            {
                bool finished = ProcessDecodeLogits(seq, logits);
                if (finished)
                {
                    seq.State = SequenceState.Completed;
                    CompleteSequence(seq);
                    _active.Remove(seq);
                }
            }
            catch (OperationCanceledException)
            {
                seq.State = SequenceState.Cancelled;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetCanceled();
                _active.Remove(seq);
            }
            catch (Exception ex)
            {
                seq.State = SequenceState.Completed;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetException(ex);
                _active.Remove(seq);
            }
            finally
            {
                logits.Dispose();
            }
        }
    }

    /// <summary>Fails every ready decoder when the batched forward itself threw (a batch-scoped
    /// model error — e.g. device-lost / OOM — cannot be isolated to one sequence).</summary>
    private void FailReadyDecoders(int readyCount, Exception? ex)
    {
        for (int j = 0; j < readyCount; j++)
        {
            var seq = _decodeReady[j];
            ReleaseKvCache(seq);
            if (ex is null)
            {
                seq.State = SequenceState.Cancelled;
                seq.CompletionSource.TrySetCanceled();
            }
            else
            {
                seq.State = SequenceState.Completed;
                seq.CompletionSource.TrySetException(ex);
            }
            _active.Remove(seq);
        }
    }

    /// <summary>Per-sequence decode for one ready sequence, with the same finish / cancel / error
    /// handling as the batched path.</summary>
    private void DecodeSingleAndFinish(SchedulerRequest seq)
    {
        try
        {
            bool finished = DecodeOneStep(seq);
            if (finished)
            {
                seq.State = SequenceState.Completed;
                CompleteSequence(seq);
                _active.Remove(seq);
            }
        }
        catch (OperationCanceledException)
        {
            seq.State = SequenceState.Cancelled;
            ReleaseKvCache(seq);
            seq.CompletionSource.TrySetCanceled();
            _active.Remove(seq);
        }
        catch (Exception ex)
        {
            seq.State = SequenceState.Completed;
            ReleaseKvCache(seq);
            seq.CompletionSource.TrySetException(ex);
            _active.Remove(seq);
        }
    }

    /// <inheritdoc/>
    public SchedulerMetrics GetMetrics()
    {
        int queueDepth;
        lock (_queueLock) { queueDepth = _pendingQueue.Count; }
        return new(
            ActiveSequences: _active.Count,
            QueueDepth: queueDepth,
            PreemptionCount: Interlocked.Read(ref _preemptionCount));
    }

    // ── Admission & prefill ──

    /// <summary>One prepared-but-not-yet-forwarded prefill: the sequence plus the contiguous range
    /// of its prompt to forward (<paramref name="ForwardStart"/> .. <c>+ForwardLen-1</c>). For a
    /// normal prefill the range is the uncached suffix; on a 100% prefix-cache hit it is the single
    /// last prompt token (re-forwarded to obtain its logits). Positions equal token indices.</summary>
    private readonly record struct PendingPrefill(SchedulerRequest Seq, int ForwardStart, int ForwardLen);

    /// <summary>
    /// Prepares a freshly-admitted sequence for prefill WITHOUT forwarding: allocates (or
    /// prefix-seeds) its KV-cache, accounts prompt tokens, sets <see cref="SequenceState.Prefilling"/>,
    /// and computes the contiguous prompt range its deferred forward will run over. The forward and
    /// first-token sample happen later in <see cref="PrefillReadySequences"/> so multiple admits in
    /// one Step can fuse into a single <see cref="IModel.ForwardBatch"/>.
    /// </summary>
    private PendingPrefill PreparePrefill(SchedulerRequest seq)
    {
        Debug.Assert(seq.State == SequenceState.Queued);

        int promptLen = seq.PromptLength;
        int cacheSize = Math.Min(promptLen + seq.MaxTokens, _model.Config.MaxSequenceLength);
        var promptIds = seq.PromptTokenIds;

        // Prefix-cache-aware admission: when the manager can seed a prefix, only the
        // suffix is run through the model. Falls back to the configured factory when
        // the cache is disabled, missed, or no manager is wired.
        int cachedTokens = 0;
        if (_prefixCache is not null)
        {
            var admission = _prefixCache.Admit(promptIds, cacheSize);
            seq.KvCache = admission.Cache;
            seq.IsPrefixCached = true;
            cachedTokens = admission.CachedTokens;
        }
        else
        {
            seq.KvCache = _kvCacheFactory(_model.Config, cacheSize);
        }

        seq.PrefixCachedTokens = cachedTokens;
        Interlocked.Add(ref _cachedPromptTokens, cachedTokens);
        Interlocked.Add(ref _prefilledPromptTokens, promptLen - cachedTokens);

        // Recurrent hosts: allocate this sequence's own SSM/GDN state so its prefill (and later decode)
        // runs against an isolated container — both the batched-throughput enabler and the fix for
        // concurrent sequences otherwise sharing the model-owned default state.
        if (_supportsThreadedState)
            seq.RecurrentState = _model.CreateSequenceState();

        seq.State = SequenceState.Prefilling;

        int prefillStart = cachedTokens;
        int prefillLen = promptLen - prefillStart;

        // 100% cache hit (prefillLen == 0): re-forward the last prompt token to obtain its logits.
        return prefillLen > 0
            ? new PendingPrefill(seq, prefillStart, prefillLen)
            : new PendingPrefill(seq, promptLen - 1, 1);
    }

    /// <summary>
    /// Runs the deferred prompt-prefill forwards for every sequence in <see cref="_prefillReady"/>:
    /// fused via <see cref="IModel.ForwardBatch"/> when the model is stateless and ≥2 are pending,
    /// else per-sequence. Each sequence's first token is sampled from its last-position logits, stop
    /// conditions are checked, and the sequence is routed to the active set or completed.
    /// </summary>
    private void PrefillReadySequences()
    {
        int count = _prefillReady.Count;

        // Pack contiguous position ranges for every pending prefill into the shared scratch.
        int totalTokens = 0;
        for (int i = 0; i < count; i++) totalTokens += _prefillReady[i].ForwardLen;
        if (_prefillPositions.Length < totalTokens)
            _prefillPositions = new int[totalTokens];

        _prefillBatch.Clear();
        int off = 0;
        for (int i = 0; i < count; i++)
        {
            var p = _prefillReady[i];
            for (int t = 0; t < p.ForwardLen; t++)
                _prefillPositions[off + t] = p.ForwardStart + t;
            _prefillBatch.Add(new SequenceForwardRequest
            {
                TokenIds = p.Seq.PromptTokenIds.AsMemory(p.ForwardStart, p.ForwardLen),
                Positions = _prefillPositions.AsMemory(off, p.ForwardLen),
                KvCache = p.Seq.KvCache!,
                MambaState = p.Seq.RecurrentState as IMambaState,
                GdnState = p.Seq.RecurrentState as IGdnState,
            });
            off += p.ForwardLen;
        }

        if (ShouldBatch(count))
        {
            IReadOnlyList<ITensor> results;
            long fwdStart = Stopwatch.GetTimestamp();
            try
            {
                results = _model.ForwardBatch(_prefillBatch, deviceId: -1);
            }
            catch (OperationCanceledException)
            {
                FailReadyPrefills(ex: null);
                return;
            }
            catch (Exception ex)
            {
                FailReadyPrefills(ex);
                return;
            }
            long perSeqTicks = (Stopwatch.GetTimestamp() - fwdStart) / count;

            for (int i = 0; i < count; i++)
            {
                var seq = _prefillReady[i].Seq;
                ITensor logits = results[i];
                seq.PrefillTicks += perSeqTicks;
                try
                {
                    // A per-sequence failure (sampler / constraint) must not abort the rest of the
                    // batch — the sequence is not yet in _active, so just release its KV and fail it.
                    FinishPrefill(seq, logits);
                }
                catch (OperationCanceledException)
                {
                    seq.State = SequenceState.Cancelled;
                    ReleaseKvCache(seq);
                    seq.CompletionSource.TrySetCanceled();
                }
                catch (Exception ex)
                {
                    seq.State = SequenceState.Completed;
                    ReleaseKvCache(seq);
                    seq.CompletionSource.TrySetException(ex);
                }
                finally
                {
                    logits.Dispose();
                }
            }
        }
        else
        {
            // Single prefill, or a recurrent model whose ForwardBatch needs per-seq state the
            // scheduler doesn't thread: forward each sequence independently. Positions are already
            // packed contiguously in _prefillPositions in _prefillReady order (offset tracked below).
            int posOff = 0;
            for (int i = 0; i < count; i++)
            {
                var p = _prefillReady[i];
                var seq = p.Seq;
                long fwdStart = Stopwatch.GetTimestamp();
                ITensor logits;
                try
                {
                    logits = _model.Forward(
                        seq.PromptTokenIds.AsSpan(p.ForwardStart, p.ForwardLen),
                        _prefillPositions.AsSpan(posOff, p.ForwardLen),
                        deviceId: -1, seq.KvCache);
                }
                catch (OperationCanceledException)
                {
                    seq.State = SequenceState.Cancelled;
                    ReleaseKvCache(seq);
                    seq.CompletionSource.TrySetCanceled();
                    posOff += p.ForwardLen;
                    continue;
                }
                catch (Exception ex)
                {
                    seq.State = SequenceState.Completed;
                    ReleaseKvCache(seq);
                    seq.CompletionSource.TrySetException(ex);
                    posOff += p.ForwardLen;
                    continue;
                }
                seq.PrefillTicks += Stopwatch.GetTimestamp() - fwdStart;
                try
                {
                    FinishPrefill(seq, logits);
                }
                catch (OperationCanceledException)
                {
                    seq.State = SequenceState.Cancelled;
                    ReleaseKvCache(seq);
                    seq.CompletionSource.TrySetCanceled();
                }
                catch (Exception ex)
                {
                    seq.State = SequenceState.Completed;
                    ReleaseKvCache(seq);
                    seq.CompletionSource.TrySetException(ex);
                }
                finally { logits.Dispose(); }
                posOff += p.ForwardLen;
            }
        }
    }

    /// <summary>
    /// Post-forward processing for one prefilled sequence: samples the first token from its
    /// last-position logits (<c>[N,vocab]</c> CPU or <c>[1,vocab]</c> GPU — both read row
    /// <c>Shape[0]-1</c>), advances any constraint, appends it, checks stop conditions, and
    /// transitions the sequence to <see cref="SequenceState.Decoding"/> (added to the active set) or
    /// <see cref="SequenceState.Completed"/> (response finished). Throws are propagated to the
    /// caller, which fails just this sequence.
    /// </summary>
    private void FinishPrefill(SchedulerRequest seq, ITensor logits)
    {
        int vocabSize = _model.Config.VocabSize;
        unsafe
        {
            float* logitPtr = (float*)logits.DataPointer;
            int logitRows = logits.Shape[0];
            var logitSpan = new Span<float>(logitPtr + (long)(logitRows - 1) * vocabSize, vocabSize);

            if (seq.Constraint is not null)
                TokenMaskApplier.Apply(logitSpan, seq.Constraint.GetAllowedTokens());

            long sStart = Stopwatch.GetTimestamp();
            int firstToken = seq.SamplerPipeline.Sample(logitSpan, seq.GeneratedTokens);
            seq.SamplerTicks += Stopwatch.GetTimestamp() - sStart;

            seq.Constraint?.Advance(firstToken);
            seq.GeneratedTokens.Add(firstToken);
        }

        // Check stop conditions on the first generated token. If satisfied, the sequence completes
        // without entering the decoding phase.
        if (CheckStopAfterAppend(seq, out var result))
        {
            seq.FinishReason = result == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
            seq.State = SequenceState.Completed;
            CompleteSequence(seq);
            return;
        }

        seq.State = SequenceState.Decoding;
        _active.Add(seq);
    }

    /// <summary>Fails every ready prefill when the batched prefill forward itself threw (a
    /// batch-scoped model error — e.g. device-lost / OOM — cannot be isolated to one sequence).</summary>
    private void FailReadyPrefills(Exception? ex)
    {
        for (int i = 0; i < _prefillReady.Count; i++)
        {
            var seq = _prefillReady[i].Seq;
            ReleaseKvCache(seq);
            if (ex is null)
            {
                seq.State = SequenceState.Cancelled;
                seq.CompletionSource.TrySetCanceled();
            }
            else
            {
                seq.State = SequenceState.Completed;
                seq.CompletionSource.TrySetException(ex);
            }
        }
    }

    /// <summary>
    /// Re-admits a preempted sequence by recomputing its KV-cache (swap strategy (i)).
    /// Re-forwards the retained prompt + already-generated tokens (excluding the most recent one,
    /// which the decode loop re-forwards) to rebuild KV, then transitions straight to
    /// <see cref="SequenceState.Decoding"/>. No token is sampled here — the generated output is
    /// preserved exactly, so resume is observationally identical to never having been preempted.
    /// </summary>
    private void AdmitAndResume(SchedulerRequest seq)
    {
        Debug.Assert(seq.State == SequenceState.Queued);
        Debug.Assert(seq.IsResuming);

        int promptLen = seq.PromptLength;
        int genCount = seq.GeneratedTokens.Count;
        Debug.Assert(genCount >= 1, "A preempted sequence has always sampled its first token.");

        // KV must cover positions [0, promptLen + genCount - 2]; the last generated token is
        // re-forwarded by DecodeOneStep. rebuildLen >= promptLen >= 1.
        int rebuildLen = promptLen + genCount - 1;
        int cacheSize = Math.Min(promptLen + seq.MaxTokens, _model.Config.MaxSequenceLength);

        // Always a fresh, non-prefix-cached cache: the recompute rebuilds every block from scratch.
        seq.KvCache = _kvCacheFactory(_model.Config, cacheSize);
        seq.IsPrefixCached = false;
        seq.PrefixCachedTokens = 0;

        // Recurrent host: the recompute must also rebuild the SSM/GDN state from scratch — allocate a
        // fresh container (the preempt freed the old one) so the re-forward repopulates it.
        if (_supportsThreadedState)
            seq.RecurrentState = _model.CreateSequenceState();

        seq.State = SequenceState.Prefilling;

        int[] ctx = ArrayPool<int>.Shared.Rent(rebuildLen);
        int[] positionsArray = ArrayPool<int>.Shared.Rent(rebuildLen);
        try
        {
            Array.Copy(seq.PromptTokenIds, ctx, promptLen);
            for (int i = 0; i < genCount - 1; i++)
                ctx[promptLen + i] = seq.GeneratedTokens[i];

            var positions = positionsArray.AsSpan(0, rebuildLen);
            for (int i = 0; i < rebuildLen; i++)
                positions[i] = i;

            long ts0 = Stopwatch.GetTimestamp();
            // Logits are discarded — the next token comes from the decode loop re-forwarding the
            // retained last generated token. We forward only to repopulate the KV-cache (and, for a
            // recurrent host, the per-seq state). Threaded-state models must route through ForwardBatch
            // (the only entrypoint that carries the state); others use the plain Forward.
            if (_supportsThreadedState)
            {
                var req = new SequenceForwardRequest
                {
                    TokenIds = ctx.AsMemory(0, rebuildLen),
                    Positions = positionsArray.AsMemory(0, rebuildLen),
                    KvCache = seq.KvCache!,
                    MambaState = seq.RecurrentState as IMambaState,
                    GdnState = seq.RecurrentState as IGdnState,
                };
                var res = _model.ForwardBatch(new[] { req }, deviceId: -1);
                res[0].Dispose();
            }
            else
            {
                using ITensor _ = _model.Forward(ctx.AsSpan(0, rebuildLen), positions, deviceId: -1, seq.KvCache);
            }
            seq.PrefillTicks += Stopwatch.GetTimestamp() - ts0;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(ctx);
            ArrayPool<int>.Shared.Return(positionsArray);
        }

        seq.IsResuming = false;
        seq.State = SequenceState.Decoding;
    }

    // ── Preemption (Step 59) ──

    /// <summary>
    /// Frees blocks under pressure by preempting active sequences whose priority is strictly below
    /// <paramref name="incoming"/>. Preempts greedily until the pool holds at least
    /// <paramref name="neededFreeBlocks"/> free blocks or no eligible victim remains.
    /// </summary>
    /// <returns><see langword="true"/> if the pool now has at least <paramref name="neededFreeBlocks"/>
    /// free blocks.</returns>
    private bool TryPreemptForPressure(SchedulerRequest incoming, int neededFreeBlocks)
    {
        Debug.Assert(_pagedPool is not null);
        while (_pagedPool!.FreeBlocks < neededFreeBlocks)
        {
            int victimIdx = SelectVictim(incoming);
            if (victimIdx < 0) return false;
            PreemptSequence(_active[victimIdx]);
            _active.RemoveAt(victimIdx);
        }
        return true;
    }

    /// <summary>
    /// Selects the preemption victim: the lowest-priority active <see cref="SequenceState.Decoding"/>
    /// sequence strictly below <paramref name="incoming"/>'s priority. Among equal-priority candidates
    /// the most-recently-submitted (largest <see cref="SchedulerRequest.SubmissionOrder"/>) is chosen so
    /// older sequences keep running — anti-starvation within a tier. Returns -1 when none qualify.
    /// </summary>
    private int SelectVictim(SchedulerRequest incoming)
    {
        int incomingPriority = (int)incoming.Request.Priority;
        int victimIdx = -1;
        int victimPriority = int.MaxValue;
        long victimOrder = long.MinValue;

        for (int i = 0; i < _active.Count; i++)
        {
            var s = _active[i];
            if (s.State != SequenceState.Decoding) continue;

            int p = (int)s.Request.Priority;
            if (p >= incomingPriority) continue; // never preempt a same-or-higher tier (incl. Critical)

            if (p < victimPriority || (p == victimPriority && s.SubmissionOrder > victimOrder))
            {
                victimPriority = p;
                victimOrder = s.SubmissionOrder;
                victimIdx = i;
            }
        }
        return victimIdx;
    }

    /// <summary>
    /// Preempts a single active sequence: frees its KV-cache (without recording a trie completion —
    /// the sequence is not finished), retains its generated tokens, and re-queues it at its original
    /// priority and submission order so it resumes in place via <see cref="AdmitAndResume"/>.
    /// </summary>
    private void PreemptSequence(SchedulerRequest seq)
    {
        FreeKvCacheOnly(seq);
        seq.State = SequenceState.Queued;
        seq.IsResuming = true;
        seq.IsPrefixCached = false;
        seq.PrefixCachedTokens = 0;
        Interlocked.Increment(ref _preemptionCount);

        // Re-queue with the ORIGINAL key (priority + fairness start tag + submission order) so a
        // repeatedly-preempted request keeps its place ahead of newer same-tier requests (anti-starvation).
        lock (_queueLock)
        {
            _pendingQueue.Enqueue(seq, (-(int)seq.Request.Priority, seq.FairnessTag, seq.SubmissionOrder));
        }
    }

    /// <summary>
    /// Disposes a sequence's KV-cache and frees its blocks, <em>without</em> routing a completion
    /// back to the prefix trie (used for preemption, where the sequence is not finished).
    /// </summary>
    private static void FreeKvCacheOnly(SchedulerRequest seq)
    {
        // Recurrent state is freed alongside the KV — a resumed sequence recomputes both from scratch.
        DisposeRecurrentState(seq);
        var cache = seq.KvCache;
        if (cache is null) return;
        seq.KvCache = null;
        try { cache.Dispose(); }
        catch
        {
            // Disposal failures must not derail the scheduler loop.
        }
    }

    /// <summary>Disposes and clears a sequence's per-seq recurrent state, if any. Safe to call
    /// repeatedly and on dense sequences (no-op when the state is already null).</summary>
    private static void DisposeRecurrentState(SchedulerRequest seq)
    {
        var state = seq.RecurrentState;
        if (state is null) return;
        seq.RecurrentState = null;
        try { state.Dispose(); }
        catch
        {
            // Disposal failures must not derail the scheduler loop.
        }
    }

    // ── Decode ──

    /// <summary>
    /// Pre-forward capacity gate for one decoding sequence. Returns <c>false</c> (and sets
    /// <see cref="FinishReason.Length"/>) when the sequence has hit the KV-cache size or its
    /// max-tokens cap and should finish without another forward; otherwise returns <c>true</c>
    /// with <paramref name="nextPos"/> = the position of the last appended token (whose successor
    /// the next forward generates).
    /// </summary>
    private static bool TryStartDecode(SchedulerRequest seq, out int nextPos)
    {
        Debug.Assert(seq.State == SequenceState.Decoding);
        int cacheSize = seq.KvCache!.MaxLength;
        // Position is appended at PromptLength + GeneratedCount - 1; the next forward consumes that
        // token at that position. nextPos == seq.Position - 1.
        nextPos = seq.Position - 1;
        if (nextPos >= cacheSize || seq.GeneratedCount >= seq.MaxTokens)
        {
            seq.FinishReason = FinishReason.Length;
            return false;
        }
        return true;
    }

    /// <summary>
    /// Decodes one token for a single sequence (the non-batched path). Runs the capacity gate, one
    /// <c>IModel.Forward</c>, then <see cref="ProcessDecodeLogits"/>. Returns whether the
    /// sequence finished.
    /// </summary>
    private bool DecodeOneStep(SchedulerRequest seq)
    {
        if (!TryStartDecode(seq, out int nextPos))
            return true;

        int lastToken = seq.GeneratedTokens[^1];
        Span<int> tokenSpan = stackalloc int[1] { lastToken };
        Span<int> posSpan = stackalloc int[1] { nextPos };

        long fwdStart = Stopwatch.GetTimestamp();
        using ITensor logits = _model.Forward(tokenSpan, posSpan, deviceId: -1, seq.KvCache);
        seq.DecodeTicks += Stopwatch.GetTimestamp() - fwdStart;
        return ProcessDecodeLogits(seq, logits);
    }

    /// <summary>
    /// Per-sequence post-forward processing shared by the single and batched decode paths:
    /// apply the constraint token-mask, sample the next token, advance the constraint, append it,
    /// and check stop conditions. <paramref name="logits"/> is this sequence's decode-token logits
    /// (<c>[1, vocab]</c>). Returns whether the sequence finished.
    /// </summary>
    private bool ProcessDecodeLogits(SchedulerRequest seq, ITensor logits)
    {
        int vocabSize = _model.Config.VocabSize;
        int nextTokenId;
        unsafe
        {
            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
            if (seq.Constraint is not null)
                TokenMaskApplier.Apply(logitSpan, seq.Constraint.GetAllowedTokens());

            long sStart = Stopwatch.GetTimestamp();
            nextTokenId = seq.SamplerPipeline.Sample(logitSpan, seq.GeneratedTokens);
            seq.SamplerTicks += Stopwatch.GetTimestamp() - sStart;
        }

        seq.Constraint?.Advance(nextTokenId);
        seq.GeneratedTokens.Add(nextTokenId);

        if (CheckStopAfterAppend(seq, out var result))
        {
            seq.FinishReason = result == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
            return true;
        }

        return false;
    }

    // ── Helpers ──

    /// <summary>
    /// Runs stop conditions over the latest appended token. If the result is <see cref="StopResult.Stop"/>,
    /// removes the trailing token from the output (matching TextGenerator semantics).
    /// </summary>
    private static bool CheckStopAfterAppend(SchedulerRequest seq, out StopResult result)
    {
        result = StopResult.Continue;
        int last = seq.GeneratedTokens[^1];

        // MVP: we do not pass a decoded-text tail. Stop-string conditions therefore won't fire.
        // EOS and MaxTokens both work on tokenId / count alone, which covers the contract
        // documented in CLAUDE.md. Tail-aware stop strings are a near-term enhancement —
        // we'd need a per-sequence IncrementalDetokenizer (see DEFERRED note in the test class).
        ReadOnlySpan<char> emptyTail = ReadOnlySpan<char>.Empty;

        for (int i = 0; i < seq.StopConditions.Count; i++)
        {
            var r = seq.StopConditions[i].ShouldStop(last, seq.GeneratedTokens, emptyTail);
            if (r != StopResult.Continue)
            {
                result = r;
                if (r == StopResult.Stop)
                {
                    // Stop semantics exclude the triggering token from output.
                    seq.GeneratedTokens.RemoveAt(seq.GeneratedTokens.Count - 1);
                }
                return true;
            }
        }
        return false;
    }

    private void CompleteSequence(SchedulerRequest seq)
    {
        Debug.Assert(seq.State == SequenceState.Completed);

        // Build response.
        string text = seq.GeneratedTokens.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(seq.GeneratedTokens), stripBosSpace: false)
            : string.Empty;

        long kvBytes = seq.KvCache is not null ? TextGenerator.GetKvCacheBytes(seq.KvCache) : 0;

        var timings = BuildTimings(
            seq.PromptLength,
            seq.GeneratedTokens.Count,
            seq.PrefillTicks,
            seq.DecodeTicks,
            seq.SamplerTicks,
            kvBytes);

        var response = new InferenceResponse
        {
            GeneratedTokenIds = seq.GeneratedTokens.ToArray(),
            Text = text,
            FinishReason = seq.FinishReason,
            PromptTokenCount = seq.PromptLength,
            GeneratedTokenCount = seq.GeneratedTokens.Count,
            Timings = timings,
        };

        RecordPerKeyTokens(seq.Request.ApiKey, seq.GeneratedTokens.Count);

        ReleaseKvCache(seq);
        seq.CancellationRegistration.Dispose();
        seq.CompletionSource.TrySetResult(response);
    }

    /// <summary>
    /// Attributes a completed request's generated tokens to its fairness key (per-key token
    /// observability). No-op when the request carries no <see cref="InferenceRequest.ApiKey"/> or
    /// generated nothing. Updates the in-process snapshot map and emits the
    /// <see cref="EngineTelemetry.TokensByKey"/> counter (zero-overhead when no listener).
    /// </summary>
    private void RecordPerKeyTokens(string? apiKey, int generatedCount)
    {
        if (apiKey is null || generatedCount <= 0) return;

        lock (_keyTokensLock)
        {
            _keyTokensServed[apiKey] = _keyTokensServed.GetValueOrDefault(apiKey) + generatedCount;
            // Bound the map for long-running multi-tenant servers; the telemetry counter is the
            // durable record, this map is only a cheap live snapshot.
            if (_keyTokensServed.Count > 4096)
                _keyTokensServed.Clear();
        }

        if (EngineTelemetry.TokensByKey.Enabled)
            EngineTelemetry.TokensByKey.Add(generatedCount, new KeyValuePair<string, object?>("key", apiKey));
    }

    /// <summary>
    /// Returns a snapshot of cumulative generated tokens per API key since startup (requests carrying
    /// an <see cref="InferenceRequest.ApiKey"/> only). Intended for admin/observability surfaces; the
    /// authoritative time-series is the <c>dotllm.engine.tokens.by_key</c> meter counter.
    /// </summary>
    public IReadOnlyDictionary<string, long> GetPerKeyTokenUsage()
    {
        lock (_keyTokensLock)
        {
            return new Dictionary<string, long>(_keyTokensServed);
        }
    }

    private static InferenceTimings BuildTimings(
        int promptLen, int generatedCount,
        long prefillTicks, long decodeTicks, long samplerTicks, long kvBytes)
    {
        double tickFreq = Stopwatch.Frequency;
        int decodeSteps = generatedCount > 1 ? generatedCount - 1 : 0;
        return new InferenceTimings
        {
            PrefillTimeMs = prefillTicks / tickFreq * 1000.0,
            DecodeTimeMs = decodeTicks / tickFreq * 1000.0,
            SamplingTimeMs = samplerTicks / tickFreq * 1000.0,
            PrefillTokenCount = promptLen,
            DecodeTokenCount = decodeSteps,
            KvCacheBytes = kvBytes,
        };
    }

    private void ReleaseKvCache(SchedulerRequest seq)
    {
        // Free the per-seq recurrent state alongside the KV (before the cache-null early-out, so a
        // sequence that somehow holds state without a cache still releases it).
        DisposeRecurrentState(seq);

        var cache = seq.KvCache;
        if (cache is null) return;
        seq.KvCache = null;

        // Push back into the prefix trie before disposal so freshly-computed blocks
        // can be reused by future requests.
        if (seq.IsPrefixCached && _prefixCache is not null && cache is PagedKvCache paged)
        {
            try
            {
                // Build the full token sequence (prompt + generated) covered by the cache.
                int promptLen = seq.PromptLength;
                int genCount = seq.GeneratedTokens.Count;
                int totalLen = promptLen + genCount;
                var full = ArrayPool<int>.Shared.Rent(totalLen);
                try
                {
                    Array.Copy(seq.PromptTokenIds, full, promptLen);
                    for (int i = 0; i < genCount; i++)
                        full[promptLen + i] = seq.GeneratedTokens[i];
                    _prefixCache.RecordCompletion(paged, full.AsSpan(0, totalLen));
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(full);
                }
            }
            catch
            {
                // Telemetry-only failure; never block the scheduler loop.
            }
        }

        try { cache.Dispose(); }
        catch
        {
            // Disposal failures must not derail the scheduler loop. Future: telemetry hook.
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Cancel everything in flight.
        foreach (var active in _active)
        {
            ReleaseKvCache(active);
            active.CompletionSource.TrySetCanceled();
            active.CancellationRegistration.Dispose();
        }
        _active.Clear();

        lock (_queueLock)
        {
            while (_pendingQueue.TryDequeue(out var pending, out _))
            {
                pending.CompletionSource.TrySetCanceled();
                pending.CancellationRegistration.Dispose();
            }
        }
    }
}
