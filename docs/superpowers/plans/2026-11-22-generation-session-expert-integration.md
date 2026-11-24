# Plan: GenerationSession, hidden-state taps, SetConstraint (#1919, #1920, #1921)

Spec: `docs/superpowers/specs/2026-11-22-generation-session-expert-integration.md`. TDD per task: write the failing test, see it fail, implement, see it pass, then run the listed mutants.
Branches (all PRs target `dev`, do not merge): `issue/1919-generation-session` (worktree `gen-session-1919`), `issue/1920-hidden-state-taps` (`taps-1920`, independent),
`issue/1921-session-set-constraint` (stacked on 1919). Commits end with `(#N)` and the Co-Authored-By trailer.
Locks: owner `expert-integ`; `cpu` for builds and CPU tests, `gpu` (+ `scripts/gpu-lock.sh`) for Vulkan, CUDA tests only on framework after James's go. One implementer at a time.
Test filters: use narrow `--filter` (never `~Quant`, see memory).

## Branch #1919
- **T1 CacheProvenanceTag.** `src/DotLLM.Engine/CacheProvenanceTag.cs` (+ `ActiveModifiersDigest`). Tests: equality per field, `Diff(other)` lists differing field names, digest order-stable, content-sensitive. Mutants: Diff skips a field; digest order-dependent.
- **T2 Session core (CPU Llama).** `GenerationSession.cs`, `GenerationSegmentResult.cs`, `TextGenerator.CreateSession`; make `ForwardPrefill`/`AllocateKvCache` internal-callable. Prefill, Append, GenerateUntilBoundary, NextTokenArgMax/CopyNextLogits, DisposeAsync. Test: multi-segment parity vs single pass on synthetic dense GGUF (tokens identical, logits <= 1e-4), prefill-count assertion (no re-prefill of prefix; use a counting IModel wrapper). Mutant: boundary token not forwarded must fail parity.
- **T3 RollbackTo + recurrent boundaries.** `CanRollbackTo`, `RollbackBoundaries`, boundary records, cache gating (IPrefixReusableKvCache / SimpleKvCache), refusal tests (non-truncatable cache, unrecorded boundary on hybrid). Tests on CPU Llama, Qwen3HybridDense, Qwen3MoeHybrid: after RollbackTo(k) logits equal a fresh prefill of tokens[0..k) and generated tokens match. Mutants: skip KV truncate; skip recurrent restore; skip logits restore.
- **T4 Provenance enforcement.** Provider func, `ValidateProvenance`, check at every public operation and RollbackTo. Tests: one per field mutated. Mutants: one per field ignored.
- **T5 Backend parity.**
  - Add `SupportsRecurrentStateCheckpoint` / `CheckpointRecurrentState` / `RestoreRecurrentState` / `RecurrentStateCheckpointBytes` to `CudaQwen3MoeHybridTransformerModel` (mirror the dense CUDA hybrid).
  - Audit Rollback + CanTruncateTo for CUDA/Vulkan caches the session can get (plain, paged, quantized); refuse loudly where unsupported.
  - Vulkan tests for all three families here (synthetic GGUFs + `VulkanDevice.IsAvailable` skip); CUDA test classes written, built, run only on framework after James's go.
- **T6 Disposal + docs + PR.** Disposal test (KV disposed, checkpoints disposed, second session allowed, generator usable again). `docs/GENERATION_SESSION.md` (+ index row in CLAUDE.md docs table is NOT edited; add link from `docs/KV_CACHE.md`). PR "Closes #1919".

## Branch #1920 (independent)
- **T7 Core API.** `HiddenStateTapSubscription`, `HiddenStateSample`, `TapRows`, `IHiddenStateTapHost`, `HiddenStateTapRegistry`, `HookResult.AbortForward`, `ForwardAbortedException`. Unit tests with a fake: filtering by layer/point, validation, dispose unregisters, Replace rejected, unsupported points throw.
- **T8 CPU x3.** `TransformerModel`, `Qwen3HybridDense`, `Qwen3MoeHybrid`: PostEmbedding / PostLayer / PreLmHead, abort with KV rollback + recurrent restore, packed batch fallback. Tests per model: 20 greedy steps identical with tap, logits bit-identical, KV bytes identical, captured shapes `[hidden]` and positions, zero-overhead alloc test, abort leaves model equal to unstarted. Mutants: tap writes back; fires on unsubscribed layer; abort skips KV rollback; abort skips recurrent restore.
- **T9 Vulkan x3.** Same tests, run here; split submit for abortable, post-submit staging download for read-only, bypass fused decode when attached. Add `IHiddenStateTapHost` to the three classes.
- **T10 CUDA x3.** Same; graph bypass, D2H after stream sync (F16 residual to float on host; undo Flint scale). Build here/framework; tests on framework after James's go.
- **T11 Docs + PR.** `docs/DIAGNOSTICS.md` section. PR "Closes #1920".

## Branch #1921 (stacked on 1919)
- **T12 SetConstraint.** `ActiveConstraint`, `SetConstraint`, reset on rollback, terminal-constraint segment end, re-entrancy guard. Tests on CPU x3 + Vulkan (+ CUDA after go): free -> JsonSchemaConstraint -> null -> observation -> resume; Json/Grammar/null alternation with no leaked state; prefill-count unchanged and position monotonic. Mutants: keep old parser state (no Reset); SetConstraint flushes KV; constraint advanced on observation tokens. Docs in `docs/CONSTRAINED_DECODING.md`. PR "Closes #1921", base `issue/1919-generation-session`.

## Gates before each PR
Build warnings clean for touched projects, new tests green on CPU (and Vulkan here), mutant table in PR body, `.docs/progress-expert-integ.md` updated, no model files in git.
