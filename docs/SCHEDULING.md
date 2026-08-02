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
   `Decoding` (or `Completed` if the first token stops). The same `ShouldBatch(count)` rule as decode
   (step 4) selects the dispatch: dense ≥2 fuse into one `IModel.ForwardBatch` (variable per-seq token
   counts — the CPU/Vulkan hosts pack them into one dispatch); recurrent threaded-state models always
   batch (carrying each sequence's freshly-allocated `IMambaState`/`IGdnState`); otherwise per-seq
   `Forward`. Resuming (preempted) sequences recompute their KV (and, for recurrent hosts, their state)
   per sequence — threaded-state hosts route that single recompute through `ForwardBatch` too.
4. **Decode the active sequences.** Each `Decoding` sequence contributes its last sampled token. The
   dispatch is chosen by `ShouldBatch(count)`:
   - **Dense** models (`RequiresPerSequenceState == false`) fuse into a single
     `IModel.ForwardBatch(requests, deviceId)` at **≥2** decoders (each `SequenceForwardRequest` carries
     its own KV-cache); a single decoder uses `Forward`, keeping single-tenant latency unchanged.
   - **Recurrent threaded-state** models (`SupportsThreadedSequenceState == true` — Mamba-3,
     Qwen3-MoE-Hybrid GDN, **and Nemotron-H** (CPU + Vulkan) via `ISsmState`) dispatch **everything via
     `ForwardBatch`, even a single sequence**, because `ForwardBatch` is the only `IModel` entrypoint
     that carries the per-seq `IMambaState`/`IGdnState`/`ISsmState` the scheduler allocates and threads.
     This both batches recurrent decode and fixes the latent corruption of running >1 concurrent
     recurrent sequence against a shared model-owned default state.
   - All recurrent hosts now expose a threadable state container, so none fall back to the per-seq
     model-owned-state loop for >1 concurrent sequence.
5. **Process results** per sequence: apply the constraint token-mask, sample the next token, advance the
   constraint, check stop conditions (EOS, max-tokens) → `Completed` when fired; then sweep
   completed/cancelled — build the `InferenceResponse`, release the KV-cache, complete the task.

```
while (!cancelled):
  SweepCancelled()
  admitted = Admit(pendingQueue, EffectiveMaxActive, prefixCache?)   # KV/prefix(+recurrent state) only — forward deferred
  if ShouldBatch(admitted.Count):                        # dense>=2, or recurrent threaded-state (any count)
    results = Model.ForwardBatch(admitted)               # fused batched prefill (per-seq state threaded)
    for i, seq in admitted: FinishPrefill(seq, results[i])
  else:
    for seq in admitted: FinishPrefill(seq, Model.Forward(seq))
  ready = active.Where(Decoding)                          # capacity-gated finish here
  if ShouldBatch(ready.Count):
    results = Model.ForwardBatch(ready)                   # fused batched decode (per-seq state threaded)
    for i, seq in ready: ProcessDecodeLogits(seq, results[i])
  else:
    for seq in ready: ProcessDecodeLogits(seq, Model.Forward(seq))
  SweepCompleted()
```

## Chunked Prefill

`MaxPrefillTokensPerStep` controls how many prompt tokens a single Step iteration may push through the model in aggregate. When non-zero, a prompt longer than the cap is split across multiple Step iterations: the sequence stays in `Prefilling` state until its `PrefilledTokens == PromptLength`, advancing one chunk per Step. **Decode tokens of already-decoding sequences keep running every step** regardless of the prefill budget — this is the head-of-line-blocking property that lets a 4096-token user prompt land without freezing every other concurrent chat session.

The trade-off: a very small chunk size raises per-step overhead (lots of small kernel dispatches); a very large chunk size lets one long prompt dominate the GPU for several steps before decode catches up. Production setups tune chunk size against expected prompt-length distribution and decode-batch size.

**CLI/server knob (#141).** `MaxPrefillTokensPerStep` is user-facing via `--prefill-chunk-size`
(alias `--ubatch-size`, llama.cpp `-ub` analog) on `run`, `chat`, and `serve`, and via
`ServerOptions.PrefillChunkSize` (bindable from `appsettings.json`). `ServerStartup.ResolveSchedulerOptions`
maps the flag onto this scheduler's `MaxPrefillTokensPerStep` when the continuous-batch scheduler is
active and no `Scheduler` section already sets an explicit cap — see [docs/SERVER.md](SERVER.md) for
the exact precedence rule and [docs/SPECULATIVE.md](SPECULATIVE.md#cli--server-usage) for the sibling
`--speculative-model`/`--speculative-k` (`--draft-model`/`--draft-tokens`) flags. On the single-request
`TextGenerator` path (no scheduler) the same flag drives true intra-prompt chunking instead of an
admission-level cap — see the honest per-path semantics note in `docs/SERVER.md`.

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
- **Vulkan other hosts (Qwen3MoeHybrid / NemotronH / Mamba3)**: per-seq dispatch. Per-sequence recurrent-state isolation is shipped for all three — `Qwen3MoeHybrid` via `IGdnState` + `SequenceForwardRequest.GdnState` (commits `03f7ab9`/`a3ad719`/`0f3e4ce`), `Mamba3` via `IMambaState` + `SequenceForwardRequest.MambaState` (session 7), `NemotronH` via `ISsmState` + `SequenceForwardRequest.SsmState` (CPU #355 / Vulkan #356); each host's `ForwardBatch` threads the per-seq state through and rejects null-state multi-seq dispatch with a clear diagnostic. Intra-block matmul fusion to mirror Phase 5f's dense-host pattern was evaluated (issue #359) and deliberately **not** adopted; the fusion target stays lm_head fan-out only. Note the rationale is *not* "every layer is recurrent" — that is inaccurate for the hybrids: Nemotron-H interleaves a minority of GQA-attention + FFN layers among its Mamba2 SSM stack, and Qwen3-MoE-Hybrid has full GQA-attention layers, and those non-recurrent layers' projection GEMMs (Q/K/V/O, up/down) **are** batchable across the sub-batch in isolation. The win is nonetheless noise-floor and not worth the rewrite, because: (1) the residual stream is shared across the interleaved recurrent + non-recurrent layers, so batching the non-recurrent minority while the dominant SSM/GDN layers stay per-seq forces a per-seq slice/write-back of the stacked hidden buffer at every recurrent layer — i.e. a full restructure of each host's per-seq `RunForwardCore` into a dense-style batched-layer loop with N-way SSM/KV state threading; (2) cost is dominated by the per-token recurrent scan, which is the large majority of layers (~90%+ for Nemotron-H), is unfusable, and is memory-bandwidth-bound on gfx1151; (3) at decode each sequence contributes `seqLen = 1`, so the batchable projection GEMMs are tiny GEMV-batch dispatches (`N = batch`) whose saved dispatch/barrier overhead is a few percent of total dispatches confined to the non-dominant layer minority. The dense host's full layer-loop fusion transfers there only because *every* dense layer is batchable; in the hybrids it is not. See the `ForwardBatch` doc-comment in `VulkanNemotronHTransformerModel` for the same conclusion at the code level.
- **CUDA**: per-seq fallback. Same mirror needed when a CUDA host is available.
- **Vulkan block-table attention (Phase 5g)**: deferred — vLLM-style single attention kernel reading per-seq block tables.

The acceptance test (`FourConcurrentSchedulerTests`) drives 4 distinct prompts concurrently through the scheduler and verifies each gets its own per-request response — the API contract is in place across all backends.

## Prefill/Decode Separation

Different compute characteristics:
- **Prefill**: Process N prompt tokens. Compute-bound (GEMM). High arithmetic intensity.
- **Decode**: Process 1 token per sequence. Memory-bandwidth-bound (GEMV). Low arithmetic intensity.

The scheduler separates these into distinct dispatches within one iteration for optimal utilization: prefill benefits from large batch GEMM, decode benefits from batching many sequences together (see steps 3–4 above — each issues its own `ForwardBatch`).

**Disaggregation seam (shipped, Step 59).** `ContinuousBatchScheduler` exposes the two phases of a
`Step()` individually — `bool StepPrefill()` (sweep + admit + deferred prompt prefill) and
`bool StepDecode()` (sweep + decode the active set) — with `Step()` simply running prefill then decode
in sequence (byte-identical to before; a sequence admitted this iteration still decodes the same
iteration). The two phases are the **separate-queue / separate-thread-pool seam**: the *prefill queue*
is the priority admission queue (`_pendingQueue`); the *decode queue* is the active-decoder set (in
continuous batching, decoders are not separately queued — they are the active sequences). A future
**disaggregated / multi-worker deployment** drives `StepPrefill` and `StepDecode` on separate thread
pools — or separate model replicas / devices with KV transfer between — so prefill-heavy and
decode-heavy work scale independently. The **in-process single-GPU** driver
(`ContinuousBatchSchedulerService.RunLoopAsync`) keeps calling `Step()`, because the model forward is
single-threaded (instance-scoped forward state) — running both phases concurrently against one model
instance would corrupt state. Tests: `Disaggregated_AdmittedInStepPrefill_DecodesOnlyInStepDecode`,
`Disaggregated_SeparatePhases_MatchCombinedStepOutput` (phase-split is token-identical to combined
`Step()`), `Disaggregated_DecodePhaseAloneIsNoOp_WhenNothingAdmitted`.

**Disaggregated driver (shipped, Step 59).** `DisaggregatedScheduler` is the driver built on that seam:
a **prefill worker** and a **decode worker** run as two `ContinuousBatchScheduler` instances over
*separate model replicas* of the same weights, sharing one paged KV pool. A sequence is prefilled by
the prefill replica, then its KV-cache is **handed off by reference** (no copy — it lives in the shared
pool) to the decode replica, which runs it to completion. Because each replica is a distinct `IModel`
with its own forward scratch, the two phases overlap on separate threads (real parallelism on
multi-core CPU / future multi-device), sidestepping the single-threaded-forward constraint. The handoff
is two internal hooks on `ContinuousBatchScheduler` — `ExtractDecodable` (remove post-prefill `Decoding`
sequences) and `InjectDecodable` (admit a pre-prefilled sequence into the decode set). `Step()` drives
one synchronous iteration (prefill → handoff → decode); `RunLoopAsync()` runs the two workers on
separate tasks with a lock-free handoff queue. Backpressure comes from the shared pool's block gate.
Completion, cancellation, and per-key token accounting all ride on the migrated `SchedulerRequest`.
Tests in `DisaggregatedSchedulerTests` prove token-parity with a single scheduler, that decode runs on
the decode replica (the prefill replica never issues a single-token forward), per-seq stop/max-tokens
across the handoff, and async end-to-end parity.

**Pluggable KV-handoff transfer (shipped).** The handoff mechanism is abstracted behind
`IKvHandoffTransfer.Transfer(source, config, destinationFactory)`, which returns the KV-cache the decode
replica should use for a just-prefilled sequence. Two implementations:

- `ReferenceKvHandoffTransfer` (**default**) — both replicas share one pool, so it returns the *same*
  cache object by reference (zero copy; the original shared-pool behaviour). Unchanged.
- `CopyKvHandoffTransfer` — the prefill and decode replicas use **separate** pools. It allocates a fresh
  cache from the decode pool, copies the source's per-layer K/V contents across (via the public
  `IKvCache` surface — `GetKeysRef`/`GetValuesRef` → `Update`, position-preserving, see internal
  `KvCacheCopy`), disposes the prefill-pool cache, and returns the new one. This is the **in-process
  stand-in for a cross-process / cross-device KV transfer**: on a single box the transferred state is a
  byte-for-byte copy between two CPU pools, so decode output is *token-identical* to the by-reference
  path. `DisaggregatedScheduler` takes optional `handoffTransfer`, `decodeKvCacheFactory`, and
  `decodePagedPool` parameters (all default to the shared-pool/reference behaviour) and applies the
  transfer in both `Step()` and the async prefill worker before injecting into the decode replica.
  `DisaggregatedKvTransferTests` proves copy-transfer output equals reference-transfer output across
  separate pools, plus a byte-for-byte `CopyKvHandoffTransfer` content check.

**Still future (needs multi-device hardware):** true cross-device transfer routes the copy over a
device→host→device (or NCCL / RDMA / NVLink) path with explicit placement on both ends — a third
`IKvHandoffTransfer` implementation behind the same seam, validatable only on a multi-GPU box — and
multi-GPU replica placement. The seam and the content-copy correctness are landed in-process now.

## Request Priority

Each request carries a `RequestPriority` enum (`Critical`, `High`, `Normal` (default), `Low`) on `InferenceRequest.Priority`. The scheduler's admission queue is a `PriorityQueue` keyed by `(-(int)Priority, fairnessStartTag, submissionOrder)`, so higher priorities are dequeued first; within a tier, requests order by the fairness start tag (0 for all when fairness is off ⇒ pure FIFO) then by submission order.

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

## Fairness (per-API-key admission, Step 59)

With `ContinuousBatchSchedulerOptions.EnableFairness` (default off), admission **within each priority tier** is ordered by **start-time fair queuing (SFQ)** instead of pure FIFO, so a single high-volume client cannot starve others sharing the tier. The fairness identity is `InferenceRequest.ApiKey` (the resolved API key; `RateLimitMiddleware` stashes it in `HttpContext.Items` and `CompletionEndpoint` copies it onto the request); a `null` key shares one "anonymous" bucket.

Each request is charged an estimated cost (`promptLength + maxTokens`). The scheduler keeps a per-key running finish tag and a global virtual clock (both under `_queueLock`): a request's intra-tier ordering key is its SFQ **start tag** = `max(virtualClock, keyFinish)`, and admitting a request advances the virtual clock to that tag. A backlogged key's start tags grow so its requests fall behind lighter keys'; a key that goes idle resets to the virtual clock (recent-usage forgiveness). **Priority still dominates across tiers** — fairness only reorders within a tier. With fairness off, every start tag is 0, so admission is byte-identical to FIFO-by-submission-order.

**Per-key fairness weights.** By default every key is charged its raw cost (weight 1.0 — equal shares). Supply `ContinuousBatchSchedulerOptions.FairnessWeightProvider` (`Func<string, double>`) to give a key a weight `w`: it is then charged `cost / w` into its finish tag, so a higher-weight key's tags grow more slowly and it earns a proportionally larger share of admissions **within its priority tier** under contention (e.g. weight 4 drains a backlog roughly 4× faster than an equal-priority weight-1 key). The provider is consulted once per `Submit` under `_queueLock`, so it must be cheap and side-effect-free; a `null` provider, a returned weight ≤ 0 / non-finite, or the anonymous (`null`-key) bucket all resolve to weight 1.0 — keeping admission byte-identical to the unweighted SFQ path. On the server, weights are sourced per-key from `RateLimitPolicy.Weight` (default 1.0): when fairness is enabled and a `RateLimit` config is present, `ServerStartup` wires `FairnessWeightProvider = apiKey => RateLimit.PolicyFor(apiKey)?.Weight ?? 1.0`.

Tests (`ContinuousBatchSchedulerTests`): `Fairness_Enabled_LightKeyInterleavesAheadOfHammerBacklog` (a light client interleaves right after the first of a 5-request hammer flood, vs. waiting behind all 5 with `Fairness_Disabled_..._Fifo`), `Fairness_PriorityStillDominatesAcrossTiers`, `Fairness_HigherWeightKey_AdmittedProportionallyAhead` (a weight-4 key drains its whole backlog before more than one equal-priority weight-1 request slips through), `Fairness_UniformWeightProvider_MatchesUnweightedSfq` (a uniform weight scales all charges equally ⇒ ordering unchanged), `InferenceRequest_ApiKey_DefaultsNull`.

**Config + telemetry.** Scheduler options (including `EnableFairness` and the other limits) are configurable via `ServerOptions.Scheduler` — a host binding `ServerOptions` from `appsettings.json` sets the whole `ContinuousBatchSchedulerOptions` section; `ServerStartup` passes it to the scheduler (the CLI also takes `--scheduler-fairness`). Per-key token accounting: `ContinuousBatchScheduler.GetPerKeyTokenUsage()` returns a live snapshot of cumulative generated tokens per `ApiKey`, and each completed keyed request increments the `dotllm.engine.tokens.by_key` meter counter (tagged by `key`; zero-overhead when no listener).

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

**Recurrent batched decode/prefill (shipped, Step 59):** recurrent hosts that expose a threadable
per-sequence state — `IModel.SupportsThreadedSequenceState == true` (Mamba-3, Qwen3-MoE-Hybrid GDN) —
are no longer gated off. The scheduler allocates one `IModel.CreateSequenceState()` container per
sequence at admission, threads it through that sequence's prefill / decode / resume, and disposes it on
release; every dispatch (incl. single-sequence) routes through `ForwardBatch` so the per-seq
`IMambaState`/`IGdnState` rides along. This also fixes the prior latent corruption of running >1
concurrent recurrent sequence against the model-owned default state. `MaxRecurrentSequences` caps
concurrency to bound aggregate per-seq state memory. Recurrent hosts **without** a threadable container
(Nemotron-H — `SsmStateCache` has no `IRecurrentSequenceState` yet) keep the per-seq loop. Tests:
`RecurrentBatched_MultipleSequences_ThreadsPerSeqStateAndMatchesBaseline` (a mock recurrent model that
throws on null/shared state — so correct output proves per-seq threading; asserts one state alloc/free
per sequence), `RecurrentBatched_SingleSequence_UsesForwardBatchWithState`,
`RecurrentBatched_MaxRecurrentSequences_CapsConcurrency`, `RecurrentBatched_PerSeqMaxTokens_Honored`.

**Still pending in Step 59** (tail follow-ups; the four roadmap sub-items — chunked prefill, priority +
preemption, prefill/decode disaggregation, fairness — are all shipped, as is the in-process
`DisaggregatedScheduler` driver, config/telemetry wiring, and Nemotron-H recurrent batching on both
CPU and Vulkan via `ISsmState`): cross-process / cross-device KV transfer + multi-GPU replica placement
for the disaggregated driver. (Per-key fairness *weights* are now shipped via
`FairnessWeightProvider` / `RateLimitPolicy.Weight`.)

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
