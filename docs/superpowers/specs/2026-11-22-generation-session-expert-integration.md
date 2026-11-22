# Spec: GenerationSession, hidden-state taps, mid-session constraints (#1919, #1920, #1921)

Consumer: `neuro-symbolic-ai` (`Base.Host/DotLlmResidualHost.cs`). Today it calls `TextGenerator.Generate` once per prompt and
re-prefills per expert call; residual hooks go through its own bridge. It needs (a) pause/observe/resume without re-prefill,
(b) read-only multi-layer residual probes on every backend, (c) constraint swaps between phases. Decisions below were made
without user dialogue; each carries its reason.

## Findings that shape the design
- `IInferenceHook` is not wired into ANY model. Only `IResidualModifier` (PostLayer, in-place; CPU, and Vulkan dense/hybrid-dense) exists.
  So #1920 is new wiring, not a uniformization.
- Recurrent (GDN) state is model-owned (one per model instance) and advanced in place per layer. `CheckpointRecurrentState` /
  `RestoreRecurrentState` exist on CPU x3, CUDA dense-hybrid, Vulkan dense-hybrid and MoE-hybrid. **CUDA MoE-hybrid has none** (gap; filled in #1919).
- KV `CurrentLength` advances during the forward (record time on Vulkan, up front on the CUDA hybrid handle), so an abort needs `Rollback`.
- CUDA decode graphs and Vulkan fused decode record the whole forward with no per-layer host visibility.
- `TextGenerator` inlines sampling/constraint/stop logic four times and keeps `ForwardPrefill`/`AllocateKvCache` private.

## #1919 GenerationSession (DotLLM.Engine)
API as in the issue, plus:
- `TextGenerator.CreateSession(InferenceOptions, CacheProvenanceTag, Func<CacheProvenanceTag>? currentProvenance = null)`.
  Reason for the provider: enforcement must be automatic when adapters/LoRA/knowledge snapshot change. The orchestrator passes a
  live view; every session operation compares it to the creation tag and throws `InvalidOperationException` naming the differing
  fields. `session.ValidateProvenance(tag)` does the same explicitly.
- `CacheProvenanceTag`: sealed record of 5 strings (the issue's fields). `ActiveModifiersDigest.Compute(IEnumerable<(kind,id,contentDigest)>)`
  is order-stable (ordinal sort, length-prefixed, SHA-256 hex). Strings are the contract; Engine does not reference DotLLM.ChunkReuse, so
  callers build `ModelDigest` / `TokenizerRevision` / `Precision` with `GgufIdentity` there.
- Invariant: **KV length == CurrentPosition and next-token logits are valid after every pause.** A boundary token is committed AND
  forwarded (one extra forward per segment end for StopInclude/length stops). EOS / `Stop` tokens are not committed (as in TextGenerator).
  Tokens of a multi-token stop string stay committed (they are in KV); `Text` is trimmed, `TokenIds` lists committed tokens.
  Reason: observation-append parity with single-pass needs no hidden pending state.
- Segment stops: `IStopCondition` (Stop/StopInclude), EOS, constraint terminal, `options.MaxTokens` per segment, KV capacity (`FinishReason.Length`).
- `GenerationSegmentResult(Text, TokenIds, FinishReason, StartPosition, EndPosition)`. Probe accessors: `int NextTokenArgMax()`,
  `void CopyNextLogits(Span<float>)`. Methods return `ValueTask` but complete synchronously (model Forward is synchronous);
  cancellation is checked per token and per prefill chunk.
- Prefill reuses the generator's chunking (`ForwardPrefill` made internal). Observation tokens are never constrained, never sampled.
- **RollbackTo(k):** legal for `0 <= k <= CurrentPosition`; k == 0 resets KV and calls `ResetSequenceState`.
  - Attention-only models: `kv.Rollback(k-1)`, then re-forward token k-1 for logits. The cache must satisfy
    `IPrefixReusableKvCache.CanTruncateTo(k-1)` (or `SimpleKvCache`'s internal check, same assembly), else `NotSupportedException` naming the cache type. Never re-prefill.
  - Recurrent models: at every pause the session stores a boundary record (position, `CheckpointRecurrentState()` object, copy of the next-token logits).
    `RollbackTo(k)` is legal only at a recorded boundary (`CanRollbackTo(k)`, `RollbackBoundaries`); it restores state, KV and logits with no forward. Otherwise `NotSupportedException`.
    Bounded: 16 boundaries and a byte budget from `RecurrentStateCheckpointBytes`, oldest evicted. The logits copy is boundary-only (not hot path); F32 because model logits are F32.
  - A recurrent model without checkpoint support: rollback (0 < k < CurrentPosition) refused loudly. No backend is deferred: CUDA MoE-hybrid gets checkpoint support in #1919.
- One live session per recurrent model (state is model-owned): creating a second, or calling `TextGenerator.Generate*` on the same generator while it is open, throws.
  `DisposeAsync` releases owned KV and checkpoints and detaches.

## #1920 Hidden-state taps (Core, Models, Cuda, Vulkan)
- `DotLLM.Core.Diagnostics`: `HiddenStateTapSubscription(int[]? Layers, HookPoint[] Points, TapRows Rows = Last, BufferProvider?)`.
  Supported points: `PostEmbedding` (layer -1), `PostLayer`, `PreLmHead` (**post-final-norm**: the exact input of the LM head, what a logit lens needs; layer -1) and a new
  `PreFinalNorm` appended to `HookPoint` (the residual after the last layer, before the final norm; what residual probes need). Both are exposed because the consumer
  (neuro-symbolic-ai) does not pin it down: its residual host works on the raw residual stream (`ForwardPassResult.FinalHiddenState`, layer snapshots), its research notes
  ask for 25/50/75% depth sites, and nothing there says which side of the norm. Rows for the two final points are the rows fed to the LM head (normally the last row).
  Other `HookPoint`s throw `NotSupportedException` at registration (fused kernels; the consumer needs depth probes + PreLmHead; stated, never silently ignored).
  `TapRows.Last` (default, decode-friendly, prefill copies one row) or `All`. `Layers` null = all; indices validated against the model.
- `HiddenStateSample` readonly record struct `(HookPoint Point, int LayerIndex, int RowIndex, int TokenPosition, int SequenceId, int Step, ReadOnlyMemory<float> Data)`;
  `Data` is pooled or caller-owned and valid only during the callback. Caller-owned buffers via `Func<int, Memory<float>>? BufferProvider`.
  F32 because the public hook contract is `float`; the GPU F16 residual is converted at the host boundary only (no device F32 store).
- `IHiddenStateTapHost { IDisposable RegisterTap(sub, Action<HiddenStateSample>); IDisposable RegisterAbortableTap(sub, Func<HiddenStateSample, HookResult>); }`.
  Abortable callbacks may return only `Continue` or `AbortForward` (`Replace` rejected: read-only; injection stays with `IResidualModifier`).
- `HookResult.AbortForward` singleton. Two surfaces, one implementation (each model has an internal forward core returning an abort outcome):
  `IHiddenStateTapHost.TryForward(tokenIds, positions, kvCache, out ITensor? logits, out ForwardAbortInfo abort)` returns `bool` with no exception (hot-path early exit per token),
  and the legacy `IModel.Forward` throws `ForwardAbortedException(Layer, Point)` when an abort happens (a null/sentinel return would be silently misused). Consumer check: neuro-symbolic-ai
  has no per-token early-exit code today (its "routing" is concept routing), but the issue targets mid-layer routing, which can be per token, so the non-throwing path is the primary one. Before throwing, the model rolls KV back to its length at entry and restores recurrent state (checkpoint taken at entry
  **only when an abortable tap is attached**; read-only taps never checkpoint). Layers after L and the LM head are skipped.
- Registry: `HiddenStateTapRegistry` (Core) holds an immutable snapshot swapped atomically; a model reads it once per forward into a local that is `null` when nothing
  is attached, so the unattached per-layer cost is one local null check. Delivery: CPU inline at the point. GPU: read-only taps deliver after the forward's work
  completes (before `Forward` returns), in (point, layer) order; abortable taps force a sync at the subscribed point (Vulkan: split submit; CUDA: stream sync + D2H).
- With any tap attached the backend leaves its fast path: CUDA decode graphs bypassed (DFlash/profiling precedent), Vulkan fused decode bypassed, packed `ForwardBatch`
  falls back to per-sequence (the `ResidualModifier` precedent). Unattached paths are unchanged.
- Path counters: `HiddenStateTapRegistry.PathCounters` (Interlocked counters: `TapForwards`, `CudaGraphBypassed`, `VulkanFusedDecodeBypassed`, `PackedBatchFallbacks`, `AbortedForwards`,
  `AbortCheckpoints`) plus a one-time Information log when a fast path is bypassed. Tests assert the counters (proves which path ran, and that unattached runs never move them).
  The cost of the eager fallbacks is documented in `docs/DIAGNOSTICS.md`.
- Wiring scope (three families x CPU/CUDA/Vulkan = 9 classes), main forward loops: `TransformerModel`, `CudaTransformerModel`, `VulkanTransformerModel`, the three
  `Qwen3HybridDense*` and the three `Qwen3MoeHybrid*`. Out of scope and NOT implementing the interface (registration fails by type, loudly): Gemma loops, pipeline and CPU-GPU split
  models, CloverPreview, other architectures. Listed in the PR.
- Non-perturbation: tap code only reads (CPU: span over hidden; GPU: copy to staging). Tests compare logits and KV bytes with and without a tap; mutant: tap writes into the residual.

## #1921 SetConstraint (Engine, stacked on #1919)
- `ActiveConstraint`, `SetConstraint(IDecodingConstraint?)`, legal between operations (a call during an in-flight operation throws).
- Calls `constraint.Reset()` on the supplied instance and takes ownership (documented; cloning per set would cost grammar caches). `null` detaches. KV, recurrent state and boundaries untouched.
- The constraint advances only on tokens the session samples; observation and prefill tokens never advance it.
- `RollbackTo` resets the active constraint (its parse state described discarded tokens).
- A complete constraint ends the segment (`FinishReason.ConstraintComplete`), as TextGenerator does.

## Test strategy
- Fixtures: `SyntheticDenseGguf` (Llama), `SyntheticQwen35HybridDenseMtpGguf`, `SyntheticQwen35MoeGguf`; byte-level tokenizer. No downloads.
- Per backend: CPU (here, `cpu` lock), Vulkan (here, `gpu` lock + gpu-lock.sh), CUDA (framework: builds fine, runs only after James's go).
- Mutants (each shown to die): boundary token not forwarded; rollback skips KV truncate; rollback skips recurrent restore; provenance ignores one field (one mutant per field);
  tap writes back; tap fires on unsubscribed layer; abort skips KV rollback; abort skips recurrent restore; SetConstraint keeps old parser state; SetConstraint flushes KV (prefill-count assertion).
- Zero overhead: allocation test (`GC.GetAllocatedBytesForCurrentThread` equal with and without the tap registry present but empty, over N decode forwards) and a counting fake proving no dispatch.
