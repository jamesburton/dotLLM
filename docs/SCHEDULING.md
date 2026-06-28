# Scheduling & Continuous Batching — dotLLM

## Overview

The scheduler operates at **iteration granularity**, not request granularity. This enables continuous batching: as sequences finish, new ones are admitted immediately, keeping the hardware batch always full.

## IScheduler Interface

```
IScheduler:
  Enqueue(request: InferenceRequest) → Task<InferenceResponse>
  RunLoop(cancellation) → Task    // Main scheduling loop
  GetMetrics() → SchedulerMetrics
```

Concrete implementation: `ContinuousBatchScheduler` (step-driven, exposed via `IBatchScheduler`) wrapped by `ContinuousBatchSchedulerService` (async, exposed via `IScheduler`).

## Iteration-Level Scheduling

Each `ContinuousBatchScheduler.Step()` call:

1. **Sweep cancelled sequences** — caller-side `CancellationToken` may have flipped state; release their KV-cache.
2. **Admit** new sequences up to `MaxActiveSequences` and (when paged) sufficient free blocks. Admission allocates the KV-cache, consults the optional `ISchedulerPrefixCache` for reuse, and transitions to `Prefilling`. **Actual prefill work happens in step 3** — admission is purely a slot/cache assignment. Because the prefill forward is deferred, the per-iteration block gate subtracts the blocks already reserved by this step's other pending prefills from the pool's free count, preserving its tight-pressure behaviour (≤1 admit/step when pool-constrained).
3. **Prefill admitted sequences** — the sequences admitted this Step run their prompt prefill, sized by
   `MaxPrefillTokensPerStep` (0 = unlimited, single shot), then each samples its first token →
   `Decoding` (or `Completed` if the first token stops). When the model is **stateless**
   (`IModel.RequiresPerSequenceState == false`) and **≥2** sequences were admitted together, their
   prefills are fused into a single `IModel.ForwardBatch` (variable per-seq token counts — the dense
   CPU/Vulkan hosts pack them into one batched dispatch); otherwise each prefills per sequence via
   `Forward`. Resuming (preempted) sequences always recompute their KV inline, per sequence.
4. **Decode the active sequences.** Each `Decoding` sequence contributes its last sampled token. When
   the model is **stateless** (KV-only — `IModel.RequiresPerSequenceState == false`) and **≥2** sequences
   are decoding, the scheduler fuses them into a single `IModel.ForwardBatch(requests, deviceId)` call
   (one `SequenceForwardRequest` per sequence, each with its own KV-cache) — a backend overrides
   `ForwardBatch` to fuse the per-sequence GEMVs into batched GEMMs (the dense CPU/Vulkan hosts do; the
   default interface impl loops `Forward`). For a single decoder, or a model that needs per-sequence
   recurrent state (Mamba-3 / Qwen3-MoE-Hybrid GDN / Nemotron-H — see `RequiresPerSequenceState`), it
   decodes per sequence via `Forward`, keeping single-tenant decode latency unchanged.
5. **Process results** per sequence: apply the constraint token-mask, sample the next token, advance the
   constraint, check stop conditions (EOS, max-tokens) → `Completed` when fired; then sweep
   completed/cancelled — build the `InferenceResponse`, release the KV-cache, complete the task.

```
while (!cancelled):
  SweepCancelled()
  admitted = Admit(pendingQueue, MaxActiveSequences, prefixCache?)   # KV/prefix only — forward deferred
  if !model.RequiresPerSequenceState and admitted.Count >= 2:
    results = Model.ForwardBatch(admitted)               # fused batched prefill
    for i, seq in admitted: FinishPrefill(seq, results[i])
  else:
    for seq in admitted: FinishPrefill(seq, Model.Forward(seq))
  ready = active.Where(Decoding)                          # capacity-gated finish here
  if !model.RequiresPerSequenceState and ready.Count >= 2:
    results = Model.ForwardBatch(ready)                   # fused batched decode
    for i, seq in ready: ProcessDecodeLogits(seq, results[i])
  else:
    for seq in ready: ProcessDecodeLogits(seq, Model.Forward(seq))
  SweepCompleted()
```

## Chunked Prefill

`MaxPrefillTokensPerStep` controls how many prompt tokens a single Step iteration may push through the model in aggregate. When non-zero, a prompt longer than the cap is split across multiple Step iterations: the sequence stays in `Prefilling` state until its `PrefilledTokens == PromptLength`, advancing one chunk per Step. **Decode tokens of already-decoding sequences keep running every step** regardless of the prefill budget — this is the head-of-line-blocking property that lets a 4096-token user prompt land without freezing every other concurrent chat session.

The trade-off: a very small chunk size raises per-step overhead (lots of small kernel dispatches); a very large chunk size lets one long prompt dominate the GPU for several steps before decode catches up. Production setups tune chunk size against expected prompt-length distribution and decode-batch size.

## Kernel-Batched Forward (`IModel.ForwardBatch`)

`IModel.ForwardBatch(IReadOnlyList<SequenceForwardRequest>, int deviceId)` is the seam for true batched compute across sequences:

```csharp
readonly record struct SequenceForwardRequest
{
    public required ReadOnlyMemory<int> TokenIds { get; init; }   // 1 (decode) or N (prefill chunk)
    public required ReadOnlyMemory<int> Positions { get; init; }
    public required IKvCache KvCache { get; init; }                // independent per sequence
    public ILoraAdapter? Adapter { get; init; }
}
```

The default interface implementation loops over `Forward` per request — backends pay the per-sequence kernel-dispatch overhead until they override with a fused implementation. Current state:

- **CPU (`TransformerModel.ForwardBatch`)**: shipped. Phase 5a fuses the lm_head GEMM at `seqLen = Σ N_i` (commit `479c23f`); Phase 5b fuses the intra-block matmuls (Q/K/V/O/gate/up/down) across the simple subgroup — GQA non-MLA non-MoE non-LoRA-active (commit `92c1345`, ~2.09× speedup at 4× decode batch on Strix Halo / SmolLM-135M Q8_0). Attention stays per-sequence; complex requests fall through to the per-seq loop.
- **Vulkan dense host (`VulkanTransformerModel.ForwardBatch`)**: shipped. Phase 5f path-1 — same intra-block matmul fusion pattern; attention dispatches per-seq via slice copy into shared scratch (commit `1c04887`). Phase 5e (lm_head-only fusion) was skipped on Vulkan because Vulkan's lm_head runs only on the last token (seqLen=1, returns `[1, vocab]`), making the saving ~150-350 µs per step — noise-floor.
- **Vulkan other hosts (Qwen3MoeHybrid / NemotronH / Mamba3)**: per-seq dispatch. Per-sequence recurrent-state isolation is shipped — `Qwen3MoeHybrid` via `IGdnState` + `SequenceForwardRequest.GdnState` (commits `03f7ab9`/`a3ad719`/`0f3e4ce`), `Mamba3` via `IMambaState` + `SequenceForwardRequest.MambaState` (session 7); each host's `ForwardBatch` threads the per-seq state through and rejects null-state multi-seq dispatch with a clear diagnostic. Intra-block matmul fusion to mirror Phase 5f's dense-host pattern is the remaining follow-up — every layer in these hosts is per-token recurrent (GDN scan, Mamba SSD scan) or sparse MoE routing, so the fusion target is lm_head fan-out only.
- **CUDA**: per-seq fallback. Same mirror needed when a CUDA host is available.
- **Vulkan block-table attention (Phase 5g)**: deferred — vLLM-style single attention kernel reading per-seq block tables.

The acceptance test (`FourConcurrentSchedulerTests`) drives 4 distinct prompts concurrently through the scheduler and verifies each gets its own per-request response — the API contract is in place across all backends.

## Prefill/Decode Separation

Different compute characteristics:
- **Prefill**: Process N prompt tokens. Compute-bound (GEMM). High arithmetic intensity.
- **Decode**: Process 1 token per sequence. Memory-bandwidth-bound (GEMV). Low arithmetic intensity.

The scheduler can separate these into micro-batches within one iteration for optimal utilization. Prefill benefits from large batch GEMM; decode benefits from batching many sequences together.

## Request Priority

Each request carries a `RequestPriority` enum (`Critical`, `High`, `Normal` (default), `Low`) on `InferenceRequest.Priority`. The scheduler's admission queue is a `PriorityQueue` keyed by `(-(int)Priority, submissionOrder)`, so higher priorities are dequeued before lower priorities and FIFO holds within a tier.

| Level | Admission behavior | Preemption (Step 59, pending) |
|-------|--------------------|-------------------------------|
| `Critical` | Admitted first, ahead of all other tiers | Never preempted |
| `High` | Admitted ahead of `Normal` and `Low` | Preempts `Normal` and `Low` |
| `Normal` | Default tier | — |
| `Low` | Admitted last | Preempted first |

**Admission ordering shipped (Phase 9 Step 59 first piece).** Verified by 4 unit tests in `ContinuousBatchSchedulerTests`:
- `Priority_HighAfterLowQueue_AdmittedFirst` — High submitted after queued Lows admits first.
- `Priority_SameTier_DrainsInSubmissionOrder` — FIFO within tier.
- `Priority_InferenceRequest_DefaultsToNormal` — guards default.
- `Priority_CriticalBeatsHighBeatsNormalBeatsLow` — strict tier ordering.

**Preemption shipped (Phase 9 Step 59, recompute-on-resume)** — see § Preemption below. `SchedulerMetrics.PreemptionCount` now reflects real preemptions.

## Preemption

When KV-cache block pressure builds and a higher-priority request is waiting, the scheduler can
preempt a lower-priority active sequence to admit the higher-priority one. Enabled via
`ContinuousBatchSchedulerOptions.EnablePreemption` (default **off**); it engages only when a paged
pool is wired and `ReserveBlocksPerSequence > 0` (those are what surface block pressure to the
admission loop).

The flow, inside the admission loop of `Step()` (after zero-refcount trie eviction fails to relieve pressure):

1. **Select victim** — the lowest-priority active `Decoding` sequence **strictly below** the
   incoming request's priority. Among equal-priority candidates the **most-recently-submitted** is
   chosen, so older sequences keep running (anti-starvation within a tier). `Critical` (and any
   same-or-higher tier) is never selected.
2. **Swap out (recompute strategy)** — free the victim's KV blocks immediately (`PreemptSequence` →
   `FreeKvCacheOnly`, no trie-completion recording since the sequence is unfinished). The victim's
   already-generated tokens are **retained**.
3. **Re-queue** the victim at its **original priority and submission order** so a repeatedly-preempted
   request keeps its FIFO place ahead of newer same-tier work. Its state returns to `Queued` with an
   `IsResuming` flag, and `SchedulerMetrics.PreemptionCount` is incremented.
4. **Repeat** until the pool holds at least `ReserveBlocksPerSequence` free blocks or no eligible
   victim remains, then admit the incoming request.
5. **Swap in (resume)** — when the preempted sequence is later re-admitted, `AdmitAndResume` allocates
   a fresh cache and **recomputes** the KV by re-forwarding `prompt + generated[0 .. n-2]` (the last
   generated token is re-forwarded by the normal decode step). **No token is sampled on resume**, so
   the output is token-for-token identical to a sequence that was never preempted.

Swap options:
- **Recompute** (shipped): Discard KV, re-forward prompt + generated tokens on resume. Simple, no host
  memory needed. This is the implemented strategy.
- **CPU offload** (future): Copy KV blocks to host memory. Faster resume but uses host RAM.

Unit tests (in `ContinuousBatchSchedulerTests`):
- `Preemption_LowEvictedForHigh_UnderBlockPressure` — a Low sequence is evicted so a later High can be
  admitted under block pressure; the victim is re-queued, `PreemptionCount == 1`, and all blocks return
  to the pool after completion.
- `Preemption_ResumedSequence_MatchesUnpreemptedOutput` — the preempted/resumed sequence reproduces the
  unpreempted control output exactly, token-for-token.
- `Preemption_Disabled_HigherPriorityWaitsInsteadOfPreempting` — with `EnablePreemption = false` the
  High request waits; `PreemptionCount == 0`.
- `Preemption_NeverEvictsCriticalActiveSequence` — a High request cannot preempt a `Critical` active
  sequence; it waits instead.

**Batched decode (shipped, Step 59):** the scheduler's steady-state decode is fused via
`IModel.ForwardBatch` whenever the model is stateless and ≥2 sequences are decoding (gated on
`IModel.RequiresPerSequenceState`; recurrent hosts keep the per-sequence loop because their
`ForwardBatch` requires per-sequence Mamba/GDN state the scheduler does not yet thread). Single-tenant
decode and recurrent-model decode are unchanged.

**Batched prefill (shipped, Step 59):** the admission loop now *prepares* each newly-admitted sequence
(KV-cache alloc + prefix seeding + forward range) without forwarding, then fuses the prefills admitted
in the same Step into one `IModel.ForwardBatch` — same stateless/≥2 gate as decode. The dense CPU and
Vulkan hosts accept variable per-seq token counts in one batched dispatch, so multi-admit prefill pays
one kernel-dispatch overhead instead of N. Tests: `BatchedPrefill_MultipleSequences_MatchPerSeqBaseline`
(token-for-token parity vs the `MaxActiveSequences=1` per-seq baseline),
`BatchedPrefill_FirstTokenStop_CompletesWithoutDecode`, `BatchedPrefill_StatefulModel_FallsBackToPerSeq`,
`BatchedPrefill_SingleAdmission_NotBatched`. Resuming (preempted) sequences keep their inline per-seq
recompute.

**Still pending in Step 59**: recurrent batched decode/prefill (thread per-seq Mamba/GDN state through
the scheduler, then drop the gate); separate prefill/decode queues/pools; and fairness constraints
(per-API-key accounting to prevent starvation under a continuous higher-priority stream).

## Sequence State Machine

```
QUEUED → PREFILLING → DECODING → COMPLETED
                ↕                    ↓
           PREEMPTED ←──────── (memory pressure)
```

## Scheduling Policies

The `IScheduler` interface allows different policies:
- **FCFS with priority**: Default. Priority queue ordered by (priority, arrival_time).
- **Shortest-job-first**: Estimate remaining tokens, prioritize short generations.
- **Fair-share**: Balance token throughput across API keys/users.

## Prefix Cache Integration (Step 37)

`ContinuousBatchScheduler` takes an optional `PrefixTrieManager` constructor
argument. When supplied:

1. **Admission**: `PreparePrefill` calls `manager.Admit(promptTokens, cacheSize)`
   to mint the per-sequence cache; the longest matching trie prefix is seeded
   (no prefill compute), and only the suffix's forward range is recorded for the
   deferred (per-seq or batched) prefill pass.
2. **Eviction pressure**: before refusing admission on block-pool exhaustion,
   the scheduler calls `manager.TryEvict(shortBy)` to recover zero-refcount
   trie blocks. Active sequences are never preempted in this step — that's
   the Step 59 surface.
3. **Completion**: `ReleaseKvCache` calls `manager.RecordCompletion(cache, fullTokens)`
   so the new blocks become available to future admissions before the cache
   is disposed.

The scheduler exposes `CachedPromptTokens` and `PrefilledPromptTokens` counters
so callers can verify the trie is delivering reuse. See
`PrefixCachedSchedulerTests.FourConcurrentSequences_SharedPrompt_PrefillCounts`
for the acceptance probe.

See [docs/KV_CACHE.md § Advanced Prompt Caching](KV_CACHE.md#advanced-prompt-caching--prefix-sharing-step-37)
for the data structure and refcount-lifecycle details.
## Prefix-cache hook (`ISchedulerPrefixCache`)

`ContinuousBatchScheduler` takes an optional `ISchedulerPrefixCache?` parameter, consulted on admission. The interface is intentionally minimal — a single `TryGetReusableBlocks(promptTokens, out reusedBlocks, out reusedTokenCount)` call — so step 37's prefix-cache trie can plug in without an API change. The scheduler:

1. Calls `TryGetReusableBlocks` after allocating the new KV-cache.
2. When the cache returns a non-zero `reusedTokenCount`, advances `PrefilledTokens` to that count (clamped to `PromptLength - 1` so at least the last token still goes through the model to produce sampling logits).
3. The actual block-splicing — wiring `reusedBlocks` into the new sequence's `KvBlockTable` — is the prefix cache's responsibility; the scheduler treats the cache as a black-box prefix oracle.

## Engine telemetry providers

`ContinuousBatchSchedulerService` registers two providers on construction and clears them on `Dispose`:

```csharp
EngineTelemetry.SetQueueDepthProvider(() => Inner.QueueDepth + Inner.ActiveCount);
EngineTelemetry.SetKvCacheUtilizationProvider(() => 1.0 - pagedPool.FreeBlocks / (double)pagedPool.TotalBlocks);
```

Tests that share the EngineTelemetry static state (`Engine.Scheduler.EngineTelemetryCollection`) opt into a non-parallel xUnit collection to avoid register/clear race flakes.
