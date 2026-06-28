---
topic: Phase 9 Step 59 — advanced scheduling, scope plan
status: research
date: 2026-05-20
sequence_item: phase-9-step-59
---

# Phase 9 Step 59 — Advanced scheduling: scope + first-piece deliverable

> **Progress (update 2026-06-26):**
> - ✅ Sub-piece 1 — priority-ordered admission (commit `feb4c8c`).
> - ✅ Sub-piece 2 — **preemption-on-pressure (recompute-on-resume, strategy (i))**. Added
>   `ContinuousBatchSchedulerOptions.EnablePreemption` (default off), victim selection
>   (lowest-priority active sequence strictly below the incoming request; newest-first tie-break;
>   never `Critical`/same-tier), `PreemptSequence`/`FreeKvCacheOnly` (free KV, retain generated
>   tokens, re-queue at original priority+order), and `AdmitAndResume` (rebuild KV from
>   prompt + generated[0..n-2], no re-sample). `SchedulerMetrics.PreemptionCount` now live.
>   4 new unit tests in `ContinuousBatchSchedulerTests`; `docs/SCHEDULING.md § Preemption` refreshed.
> - ✅ Sub-piece 3a — **batched decode** (#348). The "mixed batch / BuildBatch" framing below was
>   STALE: the scheduler decoded one `Forward(seqLen=1)` per active sequence, while the fused
>   `IModel.ForwardBatch` (dense CPU/Vulkan, Phase 5a/5b/5f) went unused. Wired decode through
>   `ForwardBatch` when the model is stateless (`IModel.RequiresPerSequenceState == false`) and ≥2
>   sequences decode; recurrent hosts keep the per-seq loop (their `ForwardBatch` needs per-seq
>   Mamba/GDN state the scheduler doesn't thread yet). 4 new tests; `docs/SCHEDULING.md` refreshed.
> - ✅ Sub-piece 3b — **batched prefill** (#349). The admission loop was refactored to *prepare*
>   each newly-admitted sequence (KV alloc + prefix seed + forward range) via `PreparePrefill`
>   without forwarding, then fuse the prefills admitted in one Step into a single `IModel.ForwardBatch`
>   (`PrefillReadySequences` → `FinishPrefill` per seq) under the same stateless/≥2 gate as decode.
>   The deferred forward required a `reservedBlocksThisStep` term in the per-iteration block gate so
>   tight-pressure admission (≤1/step) and all preemption tests stay intact. Resuming (preempted)
>   sequences keep their inline per-seq recompute. 4 new tests; `docs/SCHEDULING.md` refreshed.
> - ✅ Sub-piece 3c — **recurrent batched decode/prefill** (#350, Mamba3 + Qwen3-MoE-Hybrid, CPU+Vulkan).
>   New Core `IRecurrentSequenceState` marker (`IMambaState`/`IGdnState` derive from it); `IModel`
>   gains `SupportsThreadedSequenceState` + `CreateSequenceState()`. The scheduler allocates one
>   per-seq state at admission (`SchedulerRequest.RecurrentState`), threads it through prefill/decode/
>   resume, disposes on release, and dispatches threaded-state models via `ForwardBatch` for ALL counts
>   (incl. 1 — the only entrypoint carrying the state) via a unified `ShouldBatch(count)` helper. Also a
>   **correctness fix**: the prior per-seq `Forward` loop shared the model-owned default state across
>   concurrent recurrent sequences. `MaxRecurrentSequences` option caps concurrency (state memory).
>   Nemotron-H deferred (its `SsmStateCache` has no interface/factory — own follow-up). 4 new tests
>   (mock recurrent model throws on null/shared state ⇒ correct output proves threading); 29 scheduler
>   tests + 203 model/forwardbatch tests green. `docs/SCHEDULING.md` refreshed.
> - ⏳ Remaining: Nemotron-H recurrent batching (needs `ISsmState`); separate prefill/decode
>   queues/pools; and fairness constraints (sub-piece 4).

## What Step 59 covers

From `docs/ROADMAP.md:169`:

> **Advanced scheduling** | Prefill/decode disaggregation — separate queues and
> thread pools for prefill-heavy vs decode-heavy workloads. Priority-based
> scheduling with preemption (swap lower-priority sequences to CPU when
> VRAM-constrained). Fairness constraints to prevent starvation. Chunked
> prefill for long prompts to avoid head-of-line blocking. | 35, 36 |

Four sub-items:

1. **Prefill/decode disaggregation** — separate queues + thread pools.
2. **Priority-based scheduling with preemption** — swap lower-priority
   sequences to CPU when VRAM-constrained.
3. **Fairness constraints** — prevent starvation.
4. **Chunked prefill** — for long prompts.

## What's already in tree

Survey of `src/DotLLM.Engine/Scheduler/` (commit `d78bc4a`, post-housekeeping):

| Sub-item | Status | Where |
|---|---|---|
| Chunked prefill | **SHIPPED.** `MaxPrefillTokensPerStep` clamps how many prompt tokens a single Step pushes; long prompts split across iterations and decode tokens keep running between chunks. | `ContinuousBatchSchedulerOptions.MaxPrefillTokensPerStep`; `docs/SCHEDULING.md` § Chunked Prefill |
| Priority scheduling | **NOT SHIPPED — scaffolding only.** `SchedulerRequest.SubmissionOrder` exists for FIFO tie-breaking "among same-priority requests", but no priority field on the request; admission is pure FCFS via `ConcurrentQueue`. | `SchedulerRequest.cs:75` (comment is forward-looking); `ContinuousBatchScheduler.cs` `_pendingQueue` is unordered |
| Preemption | **NOT SHIPPED — counter only.** `_preemptionCount` field exists with explicit `#pragma warning disable CS0649` and the comment "Preemption is intentionally not implemented in the MVP (deferred to Step 59)". `BatchSchedulerMetrics.PreemptionCount` is wired up to always read 0. | `ContinuousBatchScheduler.cs:66-71`; `ContinuousBatchSchedulerService.cs:143` |
| Prefill/decode disaggregation | **NOT SHIPPED.** Today every Step builds one batch that mixes prefill chunks and decode tokens (line `ContinuousBatchScheduler.Step`); no separate queues, no separate thread pools, no separate Step loops. Phase 5a/5b (CPU) and Phase 5f (Vulkan dense) intra-batch fusion treats the mixed batch as one ForwardBatch dispatch. | `ContinuousBatchScheduler.Step()`, `BuildBatch`; `docs/SCHEDULING.md` § Prefill/Decode Separation is design-only |
| Fairness | **NOT SHIPPED.** No per-API-key or per-user accounting in the scheduler. Phase 9 Step 38 (rate limiting) does token-bucket per API key at the *server admission* layer (`HeaderApiKeyResolver`), but the scheduler doesn't see those buckets — it processes whatever the server lets through. | `IBatchScheduler` interface has no fairness hooks; `DotLLM.Server.RateLimiting` is the orthogonal gate |

So of the four sub-items, chunked prefill is done; the other three are either
scaffolding-only or missing entirely. The scheduler MVP code carries forward
comments naming Step 59 as the destination for preemption, which is helpful —
the surface is documented as the right place to land this work.

## First-piece deliverable for a session

**Priority field on `InferenceRequest` + priority-ordered admission.**

### Why this first

- **Small + scoped.** ~200 LoC of net change. The whole change fits in one PR
  and one session.
- **Unblocks preemption.** Preemption needs a victim selection rule; the
  natural rule is "lowest-priority active sequence wins". Without a priority
  field, preemption has nothing to sort against.
- **Clean external API surface.** Adds a `Priority` enum (`Critical`, `High`,
  `Normal`, `Low`) to `InferenceRequest`. The server passes it through from
  the API. Default `Normal` keeps every existing caller unchanged.
- **Testable in isolation.** Verify a `High` request submitted *after* a
  queue of `Low` requests gets admitted first; verify FIFO order among same
  priority via `SubmissionOrder`.
- **No hardware dependency.** Pure CPU + scheduler code; runs on any host.

### Out of scope for the first piece

- Preemption itself (active-sequence eviction). That's the next session.
- Prefill/decode disaggregation. That's a bigger architectural change —
  separate Step() loops, separate thread pools, ForwardBatch routing
  changes. ~1-2 days alone.
- Fairness. Requires the scheduler to see API-key identity; cross-cuts with
  Step 38's rate limiter.
- Wiring `Priority` into the OpenAI-compatible API. Initial pass surfaces
  the field through `InferenceRequest` only; HTTP server can route it from
  `X-Priority` header in a follow-up.

### Concrete plan

1. **Add `Priority` enum to `DotLLM.Core.Inference`** (or wherever
   `InferenceRequest` lives): `Critical`, `High`, `Normal` (default), `Low`.
2. **Add `Priority` field to `InferenceRequest`** with default `Normal`.
3. **Replace `_pendingQueue` (ConcurrentQueue) with a priority structure.**
   Two options:
   - (a) Keep ConcurrentQueue, sort on each `Admit` scan. Simpler, O(n log n)
     per admission but n is bounded by `MaxActiveSequences` so this is
     fine for the MVP.
   - (b) Use `PriorityQueue<TElement, TPriority>` with `(priority,
     submissionOrder)` tuple. Faster but needs locking around mutation
     since `PriorityQueue` is not thread-safe.
   Recommendation: **(a)** — keep the simple shape, swap if benchmarks
   later show admission as a hot path.
4. **Add `Submit(request, priority)` overload** on `ContinuousBatchScheduler`
   that records the priority on `SchedulerRequest`.
5. **Update `Admit` scan** to drain queued requests in
   `(priority desc, submissionOrder asc)` order.
6. **Tests**:
   - `Priority_HighSubmittedAfterLowQueue_AdmittedFirst` — submit 4 Low
     requests, then 1 High; verify the High request enters the active set
     before any remaining Low requests on the next Admit call.
   - `Priority_SamePriority_FifoOrder` — submit 3 Normal requests; verify
     they admit in submission order.
   - `Priority_DefaultIsNormal` — request without explicit priority gets
     `Normal`.
   - `Priority_CriticalAlwaysAdmittedFirst` — mix Critical + High + Normal,
     verify Critical wins ties.
7. **Doc update**: refresh `docs/SCHEDULING.md` § Request Priority — convert
   from design-only to "shipped". Note that preemption (the second half of
   the existing § Preemption section) is still pending.

Estimated effort: **2-4 hours**. Single commit, single PR. No code outside
`DotLLM.Engine.Scheduler` + `DotLLM.Core.Inference` + the unit test surface.

## Sequence: full Step 59 plan

The first piece above is followed by these in order:

1. **(this session)** Priority field + priority-ordered admission.
2. **Preemption-on-pressure** — when `_pagedPool.FreeBlocks` drops below an
   incoming request's predicted need AND a higher-priority request is in
   the queue, swap out the lowest-priority active sequence's KV. Two swap
   strategies: (i) recompute on resume (simple, no host memory); (ii) host
   memory-mapped persistence. Start with (i). ~1 day.
3. **Prefill/decode disaggregation** — separate `_pendingPrefill` and
   `_pendingDecode` lists, separate sub-batches per Step. Phase 5a/5b
   intra-block fusion already supports mixed batches, so disaggregation
   is purely a routing improvement (decode-only batches dispatch differently
   from prefill-heavy ones). ~1-2 days.
4. **Fairness constraints** — per-API-key token accounting, weight queued
   requests by recent usage so a hammer-client can't starve normal traffic.
   ~1 day. Needs API-key visibility plumbed into the scheduler from the
   server's `IApiKeyResolver`.

Total Step 59 estimate from this scope plan: **~5-7 working days**,
spread across 4-5 sessions.

## Open questions

- **Q1.** Should `Priority` be carried by `InferenceOptions` (per-request
  inference parameter, like temperature) or `InferenceRequest` (transport-
  level metadata)? Recommendation: `InferenceRequest`. Priority is a
  *scheduling* concern not an *inference* one; the model and sampler don't
  care. Mirrors the placement of `Adapter` (which is on `InferenceRequest`).
- **Q2.** Should the API key map to a default priority? E.g. `Critical`
  API key always has Critical-level requests by default. Recommendation:
  **Yes, but in a follow-up.** Keep the first piece scoped to in-process
  priority handling; the API-key → priority mapping is a server-layer
  concern that can land later (Step 38's `IApiKeyResolver` already returns
  a record that can carry a default priority).
- **Q3.** Should preemption be allowed within the same priority tier? E.g.
  can a Normal request preempt another Normal request? Recommendation:
  **No.** Preemption strictly serves higher priority over lower; among
  the same tier, FIFO. This avoids preemption-thrash during steady-state
  multi-tenant load.
