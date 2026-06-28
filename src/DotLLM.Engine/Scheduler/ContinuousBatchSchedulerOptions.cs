namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Configuration for <see cref="ContinuousBatchScheduler"/>.
/// </summary>
public sealed record ContinuousBatchSchedulerOptions
{
    /// <summary>
    /// Maximum number of sequences that may be actively prefilling or decoding at once.
    /// Acts as a soft upper bound — KV-cache capacity is the hard constraint.
    /// Default 64 matches small-server throughput; servers with large pools can raise this.
    /// </summary>
    public int MaxActiveSequences { get; init; } = 64;

    /// <summary>
    /// Maximum number of prompt tokens admitted (across all newly-admitted sequences) in
    /// a single scheduler iteration. Bounds the worst-case prefill latency before a decode
    /// iteration is allowed to run. Set to 0 to disable the bound.
    /// </summary>
    /// <remarks>
    /// MVP: this is a per-iteration admission cap, not chunked prefill. A long prompt that
    /// exceeds the cap simply waits for a later iteration to be admitted in full. Chunked
    /// prefill (splitting one sequence's prefill across iterations) is a future enhancement.
    /// </remarks>
    public int MaxPrefillTokensPerStep { get; init; } = 0;

    /// <summary>
    /// Optional cap on KV-cache blocks reserved for in-flight sequences. When non-zero,
    /// admission is gated on (free blocks ≥ blocks-required) so a single oversize prompt
    /// can't drain the pool.
    /// </summary>
    /// <remarks>
    /// MVP gates only on raw free-block count; the pool itself is the source of truth via
    /// <see cref="DotLLM.Engine.KvCache.KvBlockPool.FreeBlocks"/>. <see cref="ContinuousBatchScheduler"/>
    /// works with non-paged caches too — in that case, admission is governed by
    /// <see cref="MaxActiveSequences"/> alone.
    /// </remarks>
    public int ReserveBlocksPerSequence { get; init; } = 0;

    /// <summary>
    /// When <see langword="true"/>, the scheduler may <em>preempt</em> an active, lower-priority
    /// sequence to admit a strictly-higher-priority queued request under KV-cache block pressure.
    /// The preempted sequence is returned to the admission queue (at its original priority and
    /// submission order) and later <em>resumes by recomputing</em> its KV-cache from
    /// prompt + already-generated tokens — no host-memory swap is required. Default
    /// <see langword="false"/> (sequences run to completion once admitted, as in the MVP).
    /// </summary>
    /// <remarks>
    /// <para>Preemption only engages when a paged pool is wired and
    /// <see cref="ReserveBlocksPerSequence"/> &gt; 0 (those are what surface block pressure to the
    /// admission loop). The victim is the lowest-priority active sequence strictly below the
    /// incoming request's priority; among equal-priority candidates the most-recently-submitted is
    /// chosen so older sequences keep running (anti-starvation within a tier). <c>Critical</c>
    /// sequences are never selected as victims, and a request never preempts a same-or-higher tier.</para>
    /// <para><b>Recompute, not swap.</b> This is swap-strategy (i) from <c>docs/SCHEDULING.md</c> § Preemption:
    /// the victim's KV blocks are freed immediately and rebuilt on resume by re-running the model over
    /// the retained prompt + generated tokens. Already-generated output is preserved; only the KV
    /// tensors are recomputed. Host-memory KV offload (strategy (ii)) is a future enhancement.</para>
    /// </remarks>
    public bool EnablePreemption { get; init; } = false;

    /// <summary>
    /// Optional cap on the number of concurrently-active sequences for a recurrent model whose
    /// per-sequence state the scheduler threads (<see cref="DotLLM.Core.Models.IModel.SupportsThreadedSequenceState"/>).
    /// When non-zero, admission of such sequences is gated so no more than this many run at once.
    /// Default 0 (unlimited).
    /// </summary>
    /// <remarks>
    /// Each threaded recurrent sequence owns a full per-sequence recurrent-state container (Mamba SSM /
    /// GDN), which can be tens of MB for a large model — far heavier than a decode-step KV slice. On a
    /// memory-constrained (e.g. UMA) host this cap bounds the aggregate recurrent-state footprint the
    /// same way <see cref="ReserveBlocksPerSequence"/> bounds KV-block usage. Ignored for dense models
    /// and recurrent hosts without threadable state.
    /// </remarks>
    public int MaxRecurrentSequences { get; init; } = 0;

    /// <summary>
    /// When <see langword="true"/>, admission within each priority tier is ordered by per-API-key
    /// <b>start-time fair queuing (SFQ)</b> instead of pure FIFO, so a single high-volume client can't
    /// starve others sharing the same tier. Default <see langword="false"/> (FIFO by submission order).
    /// </summary>
    /// <remarks>
    /// <para>Each request is charged an estimated cost (<c>promptLength + maxTokens</c>). The scheduler
    /// keeps a per-key running finish tag and a global virtual clock; a request's intra-tier ordering key
    /// is its SFQ <em>start tag</em> = <c>max(virtualClock, keyFinish)</c>, and admitting a request
    /// advances the virtual clock to that tag. A backlogged key's start tags grow (its requests fall
    /// behind lighter keys'); a key that goes idle resets to the virtual clock (recent-usage forgiveness).
    /// Priority still dominates across tiers — fairness only reorders <em>within</em> a tier.</para>
    /// <para>The fairness identity is <see cref="InferenceRequest.ApiKey"/> (typically the resolved API
    /// key); a <see langword="null"/> key shares one "anonymous" bucket. With fairness off the start tag
    /// is 0 for every request, so admission is byte-identical to FIFO-by-submission-order.</para>
    /// </remarks>
    public bool EnableFairness { get; init; } = false;
}
