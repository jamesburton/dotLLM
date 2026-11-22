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
- **Constraints** (#1921): `ActiveConstraint` / `SetConstraint(IDecodingConstraint?)` swap the decoding constraint at a pause without touching KV or recurrent state; `options.ResponseFormat` initialises it. See "Mid-Session Constraint Swapping" in [CONSTRAINED_DECODING.md](CONSTRAINED_DECODING.md). A complete constraint that permits only EOS ends a segment with `FinishReason.ConstraintComplete`; `RollbackTo` resets the constraint.

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

## Backend status

| Backend | Families | Status |
|---|---|---|
| CPU | Llama, Qwen3 hybrid dense, Qwen3 MoE hybrid | all session tests run and pass |
| Vulkan | same three | `GenerationSessionVulkanTests` run and pass on the Strix Halo box (incl. a device-memory leak check across sessions) |
| CUDA | same three | `GenerationSessionCudaTests` written and built; **not yet run** (needs an NVIDIA GPU: framework / T5500, with the owner's go). `CudaQwen3MoeHybridTransformerModel` gained `CheckpointRecurrentState` / `RestoreRecurrentState` / `RecurrentStateCheckpointBytes` for this (#1919); before, the session refused every rollback on that model |

The same assertions (`GenerationSessionBackendSuite`, incl. the `SetConstraint` tests: end-to-end multi-phase, prefix preserved / no re-prefill, multi-turn isolation, rollback reset, re-entrancy guard, observation tokens never advance) run on every backend. GPU logit tolerance versus a fresh prefill is 1e-2 absolute
(CPU 1e-4): measured Vulkan Llama prefill-vs-decode differences reach 5.6e-3 with identical tokens. Exactness of the rollback itself is
asserted separately against a path-matched oracle (prefill k-1 tokens then append one) at 5e-4 on every backend.
