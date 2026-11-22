# GenerationSession — dotLLM

## Purpose

`GenerationSession` (`DotLLM.Engine`, issue #1919) is a resumable, single-sequence generation: one prefill, then any number of
*generate until a boundary*, *append observation tokens*, *probe*, and *roll back*, without re-prefilling the committed prefix.
It exists for orchestrators that pause a model mid-reply, feed it external observations (tool or expert output) and resume.

```csharp
var gen = new TextGenerator(model, tokenizer, kvCacheFactory);
await using var s = gen.CreateSession(options, new CacheProvenanceTag(modelDigest, tokRev, precision, modsDigest, snapshotId),
                                      currentProvenance: () => liveTag);
await s.PrefillAsync(promptIds);
var seg = await s.GenerateUntilBoundaryAsync(stopCondition);     // seg.TokenIds, seg.Text, seg.FinishReason, Start/EndPosition
await s.AppendObservationTokensAsync(observationIds);            // full-layer prefill, never sampled or constrained
int top = s.NextTokenArgMax();                                    // raw model logits probe
s.CopyNextLogits(buffer);
s.SetConstraint(new JsonSchemaConstraint(tokenizer, schema));     // mask sampled tokens from now on; null detaches
if (s.CanRollbackTo(k)) s.RollbackTo(k);
```

## Invariants

- After every pause, **KV length == `CurrentPosition`** and the **next-token logits are valid**. A boundary token (`StopInclude`, the
  per-segment `MaxTokens` limit, a stop-string match) is committed *and* forwarded, costing one extra forward at the segment end.
  An end-of-generation or `Stop` token is not committed (same as `TextGenerator`).
- **Sampler history is session-wide**: the repetition / frequency / DRY processors and the Gumbel generation index see every committed
  token (prompt, generated and observation tokens, as the model saw them), not just the current segment. Two segments of N sample what one
  segment of 2N does. This differs from `TextGenerator.Generate`, whose history is the generated tokens only.
- One operation at a time (a concurrent call throws `InvalidOperationException`). Operations complete synchronously; cancellation is
  checked per token and per prefill chunk. A failed or cancelled forward is undone to where it started when the cache and model can; otherwise
  the session is faulted and only `RollbackTo(0)` or disposal remain.
- **Constraints** (#1921): `ActiveConstraint` / `SetConstraint(IDecodingConstraint?)` swap the decoding constraint at a pause without touching KV or recurrent state; `options.ResponseFormat` initialises it. See "Mid-Session Constraint Swapping" in [CONSTRAINED_DECODING.md](CONSTRAINED_DECODING.md). A complete constraint that permits only end-of-generation tokens ends a segment with `FinishReason.ConstraintComplete`. `RollbackTo(k)` resets the constraint and re-advances it over the tokens the session sampled in `[attachPos, k)` (observation tokens excluded), so a paused mid-document span survives rollback exactly; the session never disposes a constraint (caller owns it).
- **Faulted sessions recover with `RollbackTo(0)`** (and `CanRollbackTo(0)`), which resets KV and recurrent state completely and clears the fault; every other operation throws until then, or disposal.
- **A forward that restored itself is not a fault.** An exception implementing `DotLLM.Core.Models.IForwardStateRestored` ("thrown after the model restored KV and recurrent state to their values at forward entry") is rethrown with the session intact: no `Recover`, no fault. Mid-segment, the tokens committed before it stay committed (`CurrentPosition`) and `GenerateUntilBoundaryAsync` continues from there. In an attention rollback re-forward the KV cut is part of the rollback and cannot be undone, so the session sits cleanly one token short (KV length == position, no next-token logits): append that token again or roll back further.

## `InferenceOptions` audit

| Field | On a session |
|---|---|
| Temperature, TopK, TopP, MinP, TopNSigma, Repetition/Frequency/Presence penalties, LogitBias, DRY*, Seed, GumbelSampling, SamplerSteps, LogitProcessors, IgnoreEos, MinTokens, ThinkingBudget | honoured (the shared `SamplerPipeline`; history is session-wide) |
| MaxTokens | honoured, per segment |
| StopSequences, ReasoningStopGate, StopSequencesUngated | honoured per segment (a match keeps its token, trims the text, `FinishReason.Stop`); stop strings cannot span a pause |
| ResponseFormat | honoured: builds the initial constraint (same factory as `Generate`); `SetConstraint` swaps it later |
| StopConditions, RepeatGuardLimit, Logprobs, TopLogprobs | rejected with `NotSupportedException` at `CreateSession` (use the per-call stop condition, `CopyNextLogits`) |
| GenerationPromptSuffix, SavePointPrefix, Threading | ignored (prefix-cache and threading knobs; a session bypasses the generator's prefix caches) |

## Capacity

`maxContextTokens = 0` (default) starts the KV-cache at 4096 tokens (`TextGenerator.DefaultInitialContextTokens`, capped at the model maximum) and grows it with `IKvCache.TryGrowTo` (x2, up to the model's `MaxSequenceLength`) when a prefill, append or generated token needs the room; admission against free device memory is whatever the GPU cache does inside `TryGrowTo`. If growth is refused (the cache cannot grow, device admission, ceiling): an append throws `InvalidOperationException`, a segment ends with `FinishReason.Length`, and a warning is traced. A positive `maxContextTokens` is a fixed capacity allocated at once (use it for a cache that cannot grow); above the model maximum it throws `ArgumentOutOfRangeException`.

## Provenance

`CacheProvenanceTag(ModelDigest, TokenizerRevision, Precision, ActiveModifiersDigest, KnowledgeSnapshotId)` is the identity the cached KV and
recurrent state were built under (`ActiveModifiersDigest.Compute` hashes overlays/adapters/LoRAs order-stably). When a `currentProvenance`
provider is given, *every* operation (including `RollbackTo`, `CanRollbackTo` and the probes) compares it with the creation tag and throws
`InvalidOperationException` naming exactly the differing fields; `ValidateProvenance(tag)` does the same explicitly. Stale state is never reused.

## Rollback

| Model kind | `RollbackTo(k)` | Cost |
|---|---|---|
| Attention-only (Llama etc.) | any `0 <= k <= CurrentPosition`; `kv.Rollback(k-1)`, token k-1 re-forwarded for logits | one decode row |
| Recurrent (GDN hybrid) | only `0`, the current position, or a **recorded boundary** (`RollbackBoundaries`); state, KV and logits restored | no forward |
| Recurrent, no checkpoint support | only `0` or the current position | n/a |

Recurrent state is model-owned and has no position addressing, so at every pause the session stores a boundary record: position,
`CheckpointRecurrentState()` object, and a copy of the next-token logits. At most 16 boundaries (and a 1 GiB host/checkpoint budget) are kept,
oldest evicted. Unsupported rollbacks throw `NotSupportedException` and change nothing; a position outside `[0, CurrentPosition]` throws
`ArgumentOutOfRangeException`.

### KV-cache gating

The session cuts a KV-cache only if it declares an exact length cut: `SimpleKvCache`, `IPrefixReusableKvCache.CanTruncateTo`, or
`IKvLengthRollback.CanTruncateTo` (a length-only cache whose rows below the new length stay valid). Any other cache makes `RollbackTo`
refuse with `NotSupportedException` naming the cache type. Matrix (`GenerationSessionKvRollbackMatrixTests` pins it by type):

| Backend | Cache | Rollback | Notes |
|---|---|---|---|
| CPU | `SimpleKvCache` | yes | refuses where a compact sliding window already dropped rows |
| CPU | `QuantizedKvCache` | yes | windowed instance refuses cuts past its evicted region |
| CPU | `PagedKvCache`, `TurboQuantKvCache` | refused | no exact-cut declaration |
| CUDA | `CudaKvCache`, `CudaPagedKvCache` | yes | full history |
| CUDA | `CudaQuantizedKvCache` | yes | windowed + evicted rows: refuses (`CanTruncate`) |
| CUDA | `CudaHybridKvCacheHandle` (Qwen3 hybrid dense / MoE hybrid, Lfm2Moe) | yes (`IKvLengthRollback`) | length-only; the owning model shrinks its F16 cursor to the handle length on the next forward (`HybridKvLengthBookkeeping.SyncToCommitted`) |
| CUDA | `CudaNemotronHKvCache` | yes (`IKvLengthRollback`) | length-only bookkeeping |
| CUDA | `CudaTurboQuantKvCache`, pipeline / split / hybrid-device caches | refused | |
| Vulkan | `VulkanKvCache`, `VulkanQuantizedKvCache` | yes | |
| Vulkan | `VulkanNemotronHKvCache` (Qwen3 hybrid dense / MoE hybrid) | yes (`IKvLengthRollback`) | lowers the visible length; later writes overwrite |
| Vulkan | `VulkanTurboQuantKvCache`, `VulkanPipelineKvCache` | refused | |

## One session per recurrent model

Recurrent state is model-owned, so a second `CreateSession` on the same model (from any generator), or `TextGenerator.Generate*` on the
generator, throws while a session is open. Attention-only models have no such limit. `CreateSession` resets the model's recurrent state.

## Ownership and disposal

The session owns its KV-cache (allocated through the generator's `kvCacheFactory`) and every boundary checkpoint. `DisposeAsync` (idempotent)
disposes the KV-cache and all checkpoints, resets the model's recurrent state and detaches from the generator, after which a new session may open.
Checkpoints are pooled by the model (one spare) and released with it.

`DisposeAsync` is serialised with the one-operation guard: a dispose from another thread during an in-flight operation refuses new operations at once and performs the release when that operation ends (the returned task completes immediately), so no forward ever runs on freed buffers. The generator is detached even if a release step throws.

### Checkpoint budget and failures

`CreateSession(..., checkpointBudgetBytes)` bounds the recurrent-state checkpoints (plus a logits copy per boundary) a session holds; default 1 GiB, at most 16 boundaries. **On a GPU these are device memory (VRAM), not host RAM.** A checkpoint that fails (for example device admission) never throws after the segment committed: the oldest boundary is freed and the checkpoint retried once, otherwise the boundary is skipped (that position is not a rollback target, `SkippedBoundaries`) and a warning is traced.

## Known limits

- **Sampler RNG is not rewound by a rollback** (M-2): seeded stochastic sampling uses a sequential `Random`, so a continuation after `RollbackTo` differs from the original run and from a fresh session. Only `GumbelSampling` is position-seeded; use it for sessions that roll back.
- **The segment detokenizer is new per segment** (M-4): a UTF-8 character split across a pause decodes as U+FFFD in both segments' `Text`, and stop strings cannot span segments. `TokenIds` are authoritative.
- **Model-attached modifiers are not rolled back** (M-8): a stateful `IResidualModifier` (for example one counting consecutive triggers) keeps its state across `RollbackTo`.
- **A forgotten dispose** (M-3) keeps a recurrent model locked to the session until the model is collected; `CreateSession` then throws naming the cause. Always dispose.
- **Provenance is caller-asserted** (M-5): the tag is compared, never derived from the model. Prefix-cache provenance stays with #1915; a session bypasses the generator's prefix caches.
- `StopResult.StopInclude` reports `FinishReason.Length` (mirrors `TextGenerator`; see `GenerationSegmentResult`).

## Backend status

| Backend | Families | Status |
|---|---|---|
| CPU | Llama, Qwen3 hybrid dense, Qwen3 MoE hybrid | all session tests run and pass |
| Vulkan | same three | `GenerationSessionVulkanTests` run and pass on the Strix Halo box (incl. a device-memory leak check across sessions) |
| CUDA | same three | `GenerationSessionCudaTests` written and built; **not yet run** (needs an NVIDIA GPU: framework / T5500, with the owner's go). `CudaQwen3MoeHybridTransformerModel` gained `CheckpointRecurrentState` / `RestoreRecurrentState` / `RecurrentStateCheckpointBytes` for this (#1919); before, the session refused every rollback on that model |

The same assertions (`GenerationSessionBackendSuite`) run on every backend. The logit tolerance versus a fresh prefill is **derived, not fixed** (James 2026-11-22, consistent with #1928), and its control does **not** run through the code under test: it is raw `IModel.Forward` on a fresh cache from the generator's KV factory. Arms, against one batched prefill of the same tokens: (1) a genuine **1-ULP arm** (one F32 `BitIncrement` on the largest element of token 0's residual after layer 0, through the residual-modifier hook; CPU models only, so GPU backends have no such arm), which must be observable (> 0) or the control is inert; (2) the prefill-vs-decode path split (prefill of half, the rest one row at a time). Control = the larger; bound = **k = 2** x control (the control is one realisation of the noise, the compared pair another); bit-exact when the control is exactly 0; a control above 1e-1 fails the run as a broken instrument. Exactness of an attention rollback is asserted separately against a path-matched oracle (prefill k-1 tokens then append one) at 5e-4, and a recurrent rollback must restore the logits captured at the boundary bit-exactly and decode one further token like the oracle within the derived bound.
