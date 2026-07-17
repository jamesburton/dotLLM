# Issue #369 — Model keep-alive / idle-unload / multi-model concurrent serving

**FOR THE COORDINATOR: review, fold into `docs/SERVER.md` if not already done inline, then delete this file.**

## What was built

`ServerState` gained multi-model residency on top of the existing single-model hot-swap:

- **`src/DotLLM.Server/ModelResidency.cs`** (new): `ResidentModelSnapshot` (a stashed-but-loaded
  model's live objects + keep-alive bookkeeping) and `ModelResidencyManager` (LRU eviction under a
  count/byte budget, keep-alive expiry sweeping, observability snapshot).
- **`ServerState`** (`src/DotLLM.Server/ServerState.cs`): at any time, its flat properties
  (`Model`, `Generator`, `Tokenizer`, …) mirror exactly one "active" model — unchanged from before
  #369, so every existing endpoint keeps compiling and working unmodified. New:
  - `Residency` (always non-null `ModelResidencyManager`), `LastUsedUtc`, `KeepAliveSecondsOverride`,
    `EstimatedBytes` (GGUF file size, symlink-resolved — see Gotchas below).
  - `EnsureActiveAsync(requestedModel, keepAliveOverride, ct)`: the money method. No-ops if the
    requested model is already active; else reactivates a stashed snapshot (cheap field-swap, no
    I/O), lazily reloads the same model if it idled out, or does a fresh load — evicting LRU
    stashed models first if the budget requires it.
  - `SwapModelAsync` (pre-existing, used by `POST /v1/models/load`) now **stashes** the outgoing
    model into `Residency` instead of unconditionally disposing it, then immediately calls
    `Residency.EnforceBudget`. With the default `MaxResidentModels = 1` this evicts (disposes) it
    on the spot — bit-for-bit the old behavior.
  - `SweepIdleAsync` / `RunIdleSweepLoopAsync`: background sweep (default every 5s). Disposes
    expired *stashed* models unconditionally (nothing is using them). For the *active* model it
    only acts if `_requestGate.WaitAsync(0)` succeeds immediately — an in-flight generation is
    never interrupted; the sweep just retries next tick.
- **`ServerOptions`**: `KeepAliveSeconds` (default 300, ollama parity), `MaxResidentModels`
  (default 1), `ResidentMemoryBudgetBytes` (default 0 = unlimited byte budget, count still bounds
  it), `IdleSweepInterval` (default 5s). CLI flags `--keep-alive`, `--max-resident-models`,
  `--resident-memory-budget` wired in `ServeCommand`.
- **API surface**:
  - `keep_alive` (seconds, float) added to `ChatCompletionRequest`, `CompletionRequest`, and
    `ModelLoadRequest`. `0` = unload after this use, negative = never, null = server default.
  - `model` field on chat/completion requests now actually routes — previously it was accepted but
    ignored. `ChatCompletionEndpoint` / `CompletionEndpoint` call `state.EnsureActiveAsync(request.Model, request.KeepAlive, ct)`
    before the existing readiness check, returning 400 with a message on load failure.
  - `GET /v1/models` now lists every resident model (active + stashed), each with `is_active`,
    `idle_seconds`, `keep_alive_seconds`, `expires_in_seconds`, `size_bytes` — previously it only
    ever reported one synthetic row from `Options.ModelId`.

## Concurrency scoping (explicitly flagged per the issue's point 5)

`ContinuousBatchScheduler`/`ContinuousBatchSchedulerService` are constructed **per single `IModel`**
— there is no multi-model batching/dispatch infrastructure anywhere in `DotLLM.Engine`. Building
that (a scheduler-of-schedulers, cross-model fairness, etc.) is a materially larger project than
this issue's scope.

**Decision: kept fully serialized for v1.** All requests — even to different resident models —
continue to funnel through the single `SemaphoreSlim(1,1)` request gate on `ServerState`
(`ExecuteAsync` / the gate inside `EnsureActiveAsync`/`SwapModelAsync`). Concretely:

- Two resident models **cannot** run inference concurrently. A request to model B while model A is
  mid-generation queues behind it (same as today's single-model behavior).
- What #369 *does* deliver despite this: **N models can be resident and hot-swappable between them
  at near-zero cost** — reactivating a stashed model is a field-copy (microseconds), not a reload
  (seconds, disk I/O, GGUF re-parse). That's the ollama-parity value proposition: "loaded but idle"
  models don't pay reload latency on their next request, even though only one executes at a time.
- This matches the issue text's own fallback: *"serializing cross-model requests but keeping N
  models resident and hot-swappable without reload cost is still a big improvement over today,
  even without concurrent execution."*

If true concurrent cross-model execution is wanted later, it needs its own issue: either N
independent `ContinuousBatchSchedulerService` instances each with their own run-loop (straightforward
but multiplies background threads and KV-cache memory pressure per resident model), or a
scheduler-of-schedulers that time-slices a shared thread/CPU budget across models.

## Other scope decisions

- **Memory accounting proxy**: no existing VRAM/RAM budget-tracking utility was found in the
  codebase to reuse (searched for `VramBudget`, `MemoryBudget`, `EstimateVram`/`EstimateMemory` —
  none exist; Vulkan weight residency doesn't pre-account a budget either). Used GGUF file size on
  disk as the estimate for both host-RAM (mmap) and VRAM-resident models. This is an
  over-approximation for quantized-KV/paged setups and an under-approximation once KV-cache and
  activation buffers are counted, but it's the same order of magnitude and requires no new
  instrumentation. A follow-up could wire real `IBackend`-reported allocation counters once one
  exists.
- **`keep_alive: 0` ("unload after each use")** is enforced by the periodic sweep (default 5s
  interval), not synchronously at the end of the request. Synchronous unload-in-request-path was
  rejected — it would add unload latency (and, for the just-completed request, GGUF-unmap latency)
  to every response when `keep_alive: 0` is set. Ollama's own daemon polls similarly rather than
  unloading synchronously.
- **Scheduler rebuild on reactivation**: `ContinuousBatchSchedulerService` is intentionally *not*
  captured in `ResidentModelSnapshot` — it's in-memory-only (no GGUF/weights inside it), so it's
  cheap to reconstruct via `ServerState.StartSchedulerLoop()` on every activation. This also fixed
  a **pre-existing, unrelated bug**: `ModelManagementEndpoint`'s `POST /v1/models/load` never
  transferred `state.Scheduler` from the freshly-loaded state, so every explicit model swap
  silently downgraded the server to the single-request gate path until process restart. Fixed in
  the same change since it directly blocked correct multi-model scheduler behavior (see
  `ModelManagementEndpoint.cs`).
- **LoRA registry / rate limiter / sampling defaults** stay process-wide (not per-model), matching
  pre-#369 behavior — `ModelManagementEndpoint` already special-cased LoRA registry preservation
  across swaps; that comment/behavior is unchanged.

## Gotcha found and fixed: `FileInfo.Length` is 0 for HF-cache symlinks on Windows

While writing the eviction-budget test (two SmolLM-135M-Instruct GGUF quants from the local HF hub
cache), `new FileInfo(path).Length` returned **0** for every file — the HF hub cache links models
into snapshot directories via real NTFS symbolic links (`mklink /D` style), and `FileInfo.Length`
reports the **reparse point's own size** (0), not the resolved target's size, unless you explicitly
follow the link. Confirmed via `Get-Item | Format-List LinkType,Target` (`LinkType: SymbolicLink`)
and `File.ResolveLinkTarget`.

Fixed with a shared `ServerStartup.SafeFileLength(path)` (resolves via
`File.ResolveLinkTarget(path, returnFinalTarget: true)` before falling back to `FileInfo.Length`),
used everywhere a GGUF size is estimated (`ServerStartup.LoadModel`, `ServerState.EnsureActiveAsync`,
`ModelManagementEndpoint`). Without this fix, every model loaded from the default HF cache layout
(which is the **documented default for this whole project** per the user's global model-storage
rule) would silently report `EstimatedBytes = 0`, defeating the memory-budget eviction entirely for
the common case. This would have been a footgun for exactly the setups #369 is meant to serve.

## Tests

New: `tests/DotLLM.Tests.Unit/Server/ModelResidencyManagerTests.cs` (pure unit, no model weights —
LRU/count/byte eviction, keep-alive sweep incl. `0` and negative sentinels, `TryTake`/`Stash`
round-trip) and `tests/DotLLM.Tests.Unit/Server/ServerStateResidencyTests.cs` (real-weight,
CPU-only, using two quants of the cached `mradermacher/SmolLM-135M-Instruct-i1-GGUF` as stand-ins
for two distinct models — skipped via `[SkippableFact]`/`Skip.IfNot` if that fixture isn't present
locally, e.g. a fresh clone without the model downloaded):

- `MultiModelResidency_LoadingSecondModel_KeepsFirstResidentAndReactivatableWithoutReload` — loads
  A then B without unloading A (acceptance criterion #1), then reactivates A and confirms B is now
  the stashed one.
- `IdleUnload_PastKeepAlive_UnloadsActiveModel_NextRequestLazilyReloads` — 50ms keep-alive, sleeps
  past it, sweeps, asserts unloaded-but-`LoadedModelPath`-preserved, then asserts a follow-up
  `EnsureActiveAsync(null, …)` reloads it (acceptance criterion #2, lazy-reload UX).
- `IdleUnload_NeverExpiresWhenKeepAliveNegative` — `keep_alive: -1` pin survives a sweep past the
  server default.
- `EvictionUnderBudgetPressure_EvictsLruStashedModelWhenBudgetExceeded` — byte budget sized for one
  model forces immediate eviction of the LRU stash even though `MaxResidentModels` would otherwise
  allow both.
- `SingleModelHotSwap_DefaultConfiguration_RegressionUnchanged` — exercises `SwapModelAsync`
  exactly as `ModelManagementEndpoint` does, with default `MaxResidentModels = 1`; asserts the
  outgoing model is evicted immediately (acceptance criterion #3 / "no regression").

### Results

```
dotnet build -c Release                                    -> 0 errors (full solution)
dotnet test tests/DotLLM.Tests.Unit -c Release
    --filter "FullyQualifiedName~ServerStateResidencyTests|FullyQualifiedName~ModelResidencyManagerTests"
    -> 15/15 passed
dotnet test tests/DotLLM.Tests.Unit -c Release
    --filter "FullyQualifiedName!~Vulkan&FullyQualifiedName!~Cuda"
    -> see final report from the agent that ran this; matched the pre-existing baseline
       (5 pre-existing Q8_0-interleaved WeightRepackingTests failures noted in project memory
       are unrelated to this change and were not touched).
```

## GPU validation

Not performed — this feature is CPU-path-agnostic (it operates on whichever `IModel`/backend was
already selected per model; no new GPU code paths were added). The two test models above load on
CPU (`Device` defaults to `"cpu"`), so no GPU lock was needed. VRAM-specific validation (confirming
Vulkan-resident weights are actually freed on eviction) would exercise the same `Model.Dispose()`
codepath the pre-#369 hot-swap already used — unchanged by this issue — so it was not re-validated
under the GPU lock; flagging this as a reasonable follow-up if the coordinator wants explicit VRAM
before/after numbers for a Vulkan multi-model scenario.
