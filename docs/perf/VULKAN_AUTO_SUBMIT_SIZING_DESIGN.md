# Vulkan automatic submit sizing by GPU time — design

**Issue:** #1760 (also #1748). **Opt-in precursor:** PR #1761 (`DOTLLM_VULKAN_PREFILL_SPLIT_LAYERS`, merged, default 0).
**Status:** design only. No code, no GPU run, no ResourceLock taken. Every number is quoted from the #1760 / #1748 /
#1761 threads (with the source) or derived from them by arithmetic shown here (marked *derived*), or is *unmeasured*.
Layer counts of the test models are not in the threads; where a derivation needs one it is flagged.

## 0. Summary of decisions

| # | Decision | Section |
|---|---|---|
| D1 | The unit to bound is **GPU time per `vkQueueSubmit`** (what the WDDM watchdog sees), not rows and not layers. A **submit governor** inside `VulkanDevice.SubmitContext` owns the budget; hosts only call `MaybeSplit(...)` at their natural boundaries. | 3, 4 |
| D2 | Default per-device budget **1000 ms**, adaptive, `DOTLLM_VK_SUBMIT_BUDGET_MS` to change, `=0` to disable. | 4, 7 |
| D3 | Preference: **row-chunk first** (perf and the only thing that scales past one layer), **layer-split second** (guarantee for any caller, bit-exact), **per-stage split last** (rows so large one layer exceeds the budget). | 5 |
| D4 | Control loop is **measured**, seeded by a cheap per-layer cost model; cold start is conservative (split every layer until the first measurement). | 4 |
| D5 | On **by default for discrete GPUs**; **measure-only on UMA** (Strix, Arc iGPU) until a TDR is observed there; explicit user settings always win. | 6 |
| D6 | No new stored tensors. Timestamp pool = 8 B per query. Layer splitting is bit-exact; chunking is not, and says so. | 8 |
| D7 | Cheap experiment first: a **fenceless** split (existing `SplitSubmit` mechanism) may avoid the sync cost entirely. | 3.3 |

## 1. Problem and evidence

RTX 3060 (Framework eGPU, Gen3 x4, Windows, Vulkan 0x10de). A forward that is one monolithic command buffer dies with
`VK_ERROR_DEVICE_LOST` at `vkWaitForFences` / `vkQueueSubmit` once it runs long enough; every failure adds one `nvlddmkm`
event 153, no 4101 (TDR) event is logged, TdrDelay is unset (default).

| Observation (source) | Reading |
|---|---|
| TwIL-8B Q4_K_M: 149 tok 8/8 pass, 299 tok 3/8, >= 530 tok always fails (#1760) | Probabilistic, rises with rows. |
| Chunk 64/128 pass at 1.8K tok (91 / 80-85 tok/s); 192/256/384/512 fail (#1760) | Onset between a 128- and a 192-row pass. |
| `PREFILL_SPLIT_LAYERS` every 1 or 2: pass (50.6-50.7 tok/s); every 5, 10, 20: fail, at 1,799 tok (#1761) | Tracks submit duration. |
| 299 tok, split every 10: 6/6 pass (#1761) | Same. |
| `PREFILL_PROFILE=1` (per-stage host-synced submits): 4/4 pass; GPU-timestamp profile mode, tracker barriers, fused-norm off, MMQ/flash/coopmat/split-KV toggles: all still fail (#1760) | Not one kernel family; not a tracked hazard. |
| VRAM peak 6.6-7.1 of 12 GiB (#1760) | Not memory. |
| Llama-3.2-1B Q8_0 and Tev1-4B pass single ~3 s submits (#1761) | **Counter-evidence**: duration alone is not the whole story. |
| Intel Arc iGPU: one 9,200-token monolithic prefill ran 337 s without device loss (#1723) | An iGPU on this class of driver is not subject to the same limit. |

**Derived budget evidence.** 1,799 tok at 84.7 tok/s (chunk 128) = 21.2 s over ~14 chunks = **~1.5 s per whole-model
128-row forward**, which passes as ONE submit. Chunk 192 would be ~2.3 s if rate-linear (*derived, optimistic*: attention
grows with depth) and fails. That brackets the limit between ~1.5 s and ~2.3 s, matching a ~2 s watchdog.
Split-every-2 at 51 tok/s is 35 s for the prompt, so ~1 s/layer-pair of submit time: right at the edge and passing. Split
every 5 (~2.5-5 s, depending on the layer count, not in the threads) fails.

**Status of the cause.** "Watchdog on long submits" is the best-supported hypothesis, not proven (no 4101; the 1B/Tev1
counter-evidence). The design below does not depend on the mechanism: it bounds the observable (submit time), and Open
Question 1 asks for the one experiment (raise `TdrDelay`) that would confirm it. If the mechanism turns out to be
something else, the knob that fixes it is still "shorter submits", which is what #1761 already demonstrated.

#1748 (`Bench_LlamaProj_64x4096x4096`: 50 dispatches per submit x 10 reps; and
`VulkanQ8_0PrefillDefaultTests.Llama32_1B_Q8_0_CoopmatDefault_PathProof...`) is the same signature on the same device;
the governor covers the model-driven one. The bench issues raw dispatches through its own `SubmitContext` use, so it also
needs the test-side `Bench` change already proposed in #1748 triage (opt-in, fewer dispatches per submit).

## 2. Where submits are formed today

All forms go through `VulkanDevice.SubmitContext` (`src/DotLLM.Vulkan/VulkanDevice.cs`): `Begin()`, `SubmitAndWait()`
(end, `vkQueueSubmit` with fence, adaptive spin/block wait, reset), and `SplitSubmit()` (fenceless submit on a second
command buffer; **at most once per forward**, two-buffer ring; used by decode record/execute overlap, #143).
`DOTLLM_VK_SPIN_WAIT` governs the wait; the adaptive wait already keeps an EMA of fence-wait wall time (`_emaWaitMs`).

| Path | Where | Submit shape today |
|---|---|---|
| Dense AR forward (prefill and decode), `VulkanTransformerModel.Forward` | layer loop near `VulkanTransformerModel.cs:4640-4670` | **One** command buffer per forward. Decode: one fenceless split at layer `DOTLLM_VULKAN_DECODE_SPLIT_LAYER` (default 8). Prefill (`seqLen > 1`): optional host-synced split every `PrefillSplitLayers` (#1761; `_submit.SubmitAndWait(); Begin(); HostToComputeBarrier`). `PREFILL_PROFILE=1` splits per stage. |
| `ForwardBatch` (dense) | `VulkanTransformerModel.cs` ~3942, ends in `SubmitAndWait` ~4431 | One submit for the fused batch (decode of N sequences; prefill-ish members are peeled off to the per-seq `Forward`). Rows = sum of members. |
| Hybrid recurrent hosts (Qwen3.5/3.6 `VulkanQwen3HybridDenseTransformerModel`) | own record loops, many `SubmitAndWait` sites | Engine-side chunking: `PreferredPrefillChunkTokens` = `DefaultPrefillChunkTokens` = **1024** (#1100). Each chunk is one forward = one submit (plus recurrent-state transitions). |
| Other hosts (grep of `SubmitAndWait` under `src/DotLLM.Vulkan`; **not audited per host**) | `VulkanQwen3MoeHybrid*`, `VulkanQwen4Exp*`, `VulkanNemotronH*`, `VulkanLfm2Moe`, `VulkanLfm3`, `VulkanMamba3`, Bert/Gemma embedding, DFlash, pipeline | Per-forward single submit, some with category splits for profiling. Dual-device pipeline (#366) is one `SubmitContext` per device. |
| Engine prefill chunking | `TextGenerator.EffectivePrefillChunkSize` = `--prefill-chunk-size` else `IModel.PreferredPrefillChunkTokens` (default 0 = one pass) | Static per model; cancel and progress only between chunks. #1723 (open) asks for default chunking on Vulkan precisely for progress/cancel. |

Consequences: (a) a dense model on Vulkan has **no default chunk**, so a 1.8K prompt is one ~35 s submit on the 3060;
(b) the only existing mid-forward host-synced boundary is the layer loop of the dense host, and it is not shared;
(c) callers that bypass the engine (the #1756 `VulkanGraniteF16KvTests` does 512-row chunks itself, which still fail)
need protection **inside** the model, so an engine-only fix is insufficient.

## 3. Estimating and measuring per-submit GPU time

### 3.1 Signals

* **Host fence wall time** (free; already measured in `WaitForFenceAdaptive`): submit-to-signal wall time. This is the
  closest observable to what a watchdog counts (it includes queueing behind other contexts, e.g. a display compositor).
* **GPU timestamps** (`vkCmdWriteTimestamp`; `VulkanDevice.TimestampPeriodNs` exists; `VulkanLfm2MoeTransformerModel`
  and the `PREFILL_PROFILE=gpu` mode already use query pools): two stamps per submit (top and bottom of pipe) give GPU
  busy span; per-layer stamps in a **calibration forward** give a per-layer-kind cost. Validity: check
  `timestampValidBits` of the queue family; if 0, fall back to wall time. Cost: 8 B per query, negligible.

Controller input is `max(timestamp span, fence wall)`: wall dominates under contention (the watchdog does too), timestamps
dominate when the host is slow to observe the fence. Timestamps' main job is the **breakdown** (cost per layer kind) and
the contention diagnostic (wall >> span means someone else is using the GPU).

### 3.2 Cost model (seed, then measure)

Seed, used only before the first measurement and to compare candidate plans:

`t_layer(rows) ~ max( W_layer / BW_eff , rows * 2 * P_layer / F_eff ) + t_attn(rows, ctx)`

with `W_layer` = quantized weight bytes of the layer (known from `_weights`, read in place: rule 2), `P_layer` = active
parameter count (MoE: active experts only), `F_eff` and `BW_eff` **calibrated, not datasheet** (the first measured forward
fits them; datasheet numbers are wrong by 2-3x on an eGPU over Gen3 x4 and across the coopmat/MMQ arm split), and
`t_attn` linear in `rows * ctx` (flash/split-KV paths). The threads give one fitted point per model; the model is a seed
precisely because **superlinearity is visible in the data**: 128 rows -> whole forward ~1.5 s, 1,799 rows -> ~35 s
(23x per layer for 14x rows, layer count cancels *derived*). Do not trust extrapolation beyond ~2x of the measured rows.

Runtime plan per submit: after each submit, update an EMA of `ms per (row * layer)` per layer kind (dense / MoE / GDN /
sliding) and of the attention term versus depth. Predict the next submit as `t_last * (rows_next/rows_last) * (layers_next/layers_last) * g(ctx)`
with `g` the observed slope (cap growth at 1.25x per step, shrink immediately).

### 3.3 Cost of host-synced splits

* Measured (#1760/#1761): split every 1 vs 2 layers = 50.7 vs 50.6 tok/s, i.e. **a fence round trip per layer is below
  measurement noise** next to ~1 s of GPU time per layer at 1.8K rows. The expected sync cost is milliseconds per split
  (drain, re-record barrier state, `HostToComputeBarrier`, restart the queue; ~1.3 ms wake-up on Strix, per the existing
  spin-wait comment); on 36 layers that is <= ~50 ms against 35 s.
* So **sync cost is not why 51 tok/s is slower than 84.7 tok/s** (chunk 128). The 1.65x gap (35 s vs 21 s for the same
  1,799 tokens, *derived*) is a per-row efficiency effect of one huge row count versus 128-row passes (attention is
  quadratic within a pass; scratch/cache footprint; which GEMM arm is chosen; all *unmeasured*). That is the actual
  argument for row-chunking in section 5, and it is a hypothesis that the 3060 window should test with the GPU-timestamp
  profile at rows 128 vs 1,799 (Open Question 9).
* **Experiment E2 (cheap, high value).** `SplitSubmit` is a fenceless submit (queue-order serialized, no host wait). If the
  watchdog operates per submitted packet, fenceless splits every k layers pass with **zero** sync cost and with no change in
  the host's recording overlap. #1760 only tested host-synced splits. The mechanism needs generalizing from "at most once per
  forward, two-buffer ring" to an N-buffer ring (or reset after the oldest fence). Gate: E2 on the 3060 at 1.8K tokens
  with split every 2 and every 5; if it passes at every 5, prefer fenceless splits as the governor's mechanism and keep
  host-synced only for the final fence.

## 4. Governor design

```
SubmitGovernor (per VulkanDevice.SubmitContext; one per queue)
  BudgetMs        device default, env override
  Calibrated      bool; cost EMAs per layer kind
  MaybeSplit(layerIndex, rowsInSubmit)  -> bool       // host calls at layer boundaries
  PlanRows(requestedRows, ctxLen)       -> int        // chunk-size advice for the engine
  OnSubmitCompleted(ms, rows, layers)                // from SubmitAndWait
```

* **Layer boundary split** (D3 second level): maintain `predictedSinceLastSubmit`; before layer `i`, if
  `predictedSinceLastSubmit + t_layer(rows) > BudgetMs`, submit-and-wait (or fenceless submit if E2 passes), `Begin()`,
  `HostToComputeBarrier`, continue. Same ordering semantics and bit-exactness as #1761, just placed by prediction instead
  of `layer % N`. A host with no layer loop (a flat stage list) calls it at stage boundaries.
* **Cold start**: before the first measurement on a device, split **every layer** (the proven-safe #1761 shape for 1.8K rows
  on the 3060), stamp every layer, fit the model from that one forward, then stop over-splitting. Cost: one forward of a
  few dozen extra syncs (milliseconds).
* **Per-stage split** (third level): if a single layer's predicted time exceeds the budget (rows large), the governor
  requests the engine chunk (below) and, for callers that cannot chunk, falls back to the `PREFILL_PROFILE=1` per-stage
  boundaries inside that layer. (Evidence this is safe: PREFILL_PROFILE=1 passes 4/4.)
* **Row advice** (first level): `PlanRows` returns the largest multiple of 64 whose predicted whole-forward time is
  within budget, clamped to `[64, 1024]` (1024 is the existing hybrid default; 64 is the lowest size evidenced to pass).
  Exposed to the engine through an `IModel` hook (section 5.2).
* **Adaptation**: AIMD-like. Overrun (`measured > budget`) shrinks the next advice multiplicatively to
  `0.8 * budget / measured`; ramp up by at most 25 % per submit while `measured < 0.6 * budget`.
* **Device lost is not recoverable** (the logical device is gone), so the controller must never *learn from* a failure; it
  must only avoid it. Errors still surface as today, but the exception gains a diagnostic suffix naming the last submit
  size and `DOTLLM_VK_SUBMIT_BUDGET_MS` / `PREFILL_SPLIT_LAYERS`.

## 5. Preference order and why chunk 64/128 passed

### 5.1 Order

1. **Row-chunk** (engine side, advised by the governor). Reasons: (a) 84.7 vs 51 tok/s on the same prompt (#1760): it
   is also a throughput win on this device; (b) it is the only lever that scales past one layer: at 1.8K rows one layer is
   already ~1 s (*derived*) and at 8K rows one layer is several seconds, beyond what layer splitting can fix; (c) it gives
   progress and cancel (#1723) for free. Cost: not bit-identical to single-pass prefill (row batching changes the GEMM arm
   above/below the coopmat threshold and reduction order); acceptable because chunked prefill is already the established
   engine mode and the repository's tests for it use KL/greedy parity, but it must be stated in the PR.
2. **Layer-split inside the forward** (model side, always on under the governor). Reasons: protects every caller that
   does not go through `TextGenerator` chunking (tests, `ForwardBatch`, embedding encoders, the scheduler); bit-exact versus
   unsplit; the #1761 mechanism, now placed by prediction.
3. **Per-stage split** only when one layer exceeds the budget and the caller cannot chunk.

### 5.2 Why chunk 64/128 passed and 192+ failed

A 128-row forward is ~1.5 s whole-model (*derived*): below a ~2 s limit; 192 rows is ~2.3 s or more (*derived*): above. The
threshold is therefore a **time** threshold that happens to coincide with a row count on TwIL-8B and Llama-3.2-3B Q8_0 on this
device, which is why a fixed row constant (`--prefill-chunk-size 128`) is the wrong fix: it is wrong for a 1B model
(1,024 rows would be fine and 8x faster per weight read), wrong for a 70B-class model on the same card (64 rows may already
exceed the budget), and wrong at depth (attention cost grows, so the same 128 rows take longer at 8K context). The
1B/Tev1 single ~3 s submits that passed are the unexplained residue; a budget below the limit still protects them (they
would simply be split earlier than strictly necessary: a 5-10 % cost at worst, to be measured, section 7).

Engine hook (proposal): `IModel.NextPrefillChunkTokens(int remaining, int contextLength)` with the default implementation
`Math.Max(0, PreferredPrefillChunkTokens)`, and `TextGenerator` calling it per chunk instead of once. An explicit
`--prefill-chunk-size N` is an **upper bound** (the governor may lower the chunk, never raise it); layer splitting still
applies inside each chunk, which is exactly what the #1756 512-row test needs.

## 6. Defaults, switches, device classes

| Switch | Meaning |
|---|---|
| `DOTLLM_VK_SUBMIT_BUDGET_MS` | Unset: device default (below). `0`: governor fully off (today's behaviour, including no cold-start splitting). `N>0`: budget in ms, applies on every device class. |
| `DOTLLM_VULKAN_PREFILL_SPLIT_LAYERS=N` (existing, #1761) | Explicit fixed split. When set it **overrides** the governor's layer plan (explicit beats auto); the governor still reports measurements. |
| `--prefill-chunk-size`, `DOTLLM_VULKAN_PREFILL_CHUNK` (existing) | Explicit chunk: upper bound as above. |

| Device class | Default |
|---|---|
| Discrete GPU (`DeviceType == 2`), any OS | **On**, budget 1000 ms. The only class with a reproduced failure, and a healthy discrete GPU pays nothing: the governor allows rows up to the budget (e.g. a card that does 1,024 rows in 0.3 s keeps 1,024). |
| UMA iGPU as compute-only or secondary (Strix Halo, Arc iGPU) | **Measure-only**: collect timestamps/EMA, expose them in the CLI summary, split nothing. Evidence: 337 s monolithic submit on Arc iGPU without loss (#1723) and Strix long prefills run monolithic today. |
| UMA/iGPU that **drives the display** (TDR risk is real there: the compositor must be scheduled) | Measure-only by default, but one switch (`DOTLLM_VK_SUBMIT_BUDGET_MS=2000`) turns enforcement on. Note the Arc evidence above was a display-capable iGPU, so there is no failing example to justify enforcing. |
| Software/virtual devices | Off. |

Why 1000 ms: 2x margin under the default 2 s watchdog, and slack for GPU sharing with the display. The largest submit proven safe (~1.5 s, a 128-row whole forward) exceeds 1 s, so the budget splits it
in two ~0.75 s submits, a small cost (Open Question 2 asks about 1500 ms). Linux discrete GPUs have different timeouts
(amdgpu lockup timeout 10 s, NVIDIA without a display attached: none) so the 1 s default is conservative there; keep it on
for consistency, revisit with Linux data.

## 7. Interactions

* **Paged KV (`docs/perf/VULKAN_PAGED_KV_DESIGN.md`).** Chunk sizes are multiples of 64 = the default block size `B`, so
  chunk ends land on block boundaries (trie hits, recurrent checkpoints #1286, CoW unaffected). The block table is uploaded
  once per forward and shared by all layers (D2 there): a layer-boundary split sits **after** that upload and CoW copies,
  and the post-split `HostToComputeBarrier` is the same as #1761's; no extra table traffic. Block append/CoW must never
  straddle a governor-induced chunk reduction (the governor only changes the chunk **size**, never splits a row range
  mid-block-append; the engine already rounds). The paged in-place prefill is "per-token = slow" on the hybrids in S3, which
  makes it the likeliest consumer of per-submit budgeting (a slow kernel inflates `ms/row`; the controller handles it, a
  fixed constant would not).
* **#1723 (progress/cancel).** Auto chunking is precisely the progress/cancel unit it asks for; it should land together or
  right before the default chunking decision there (one mechanism, not two). Cancel stays **between forwards** only:
  aborting mid-forward would leave KV/recurrent state half-written. Layer-split points can report finer progress (rows x
  layers done) but never cancel.
* **Scheduler batched decode (#348) / latency.** A bounded submit also bounds how long concurrent decode steps stall behind
  a prefill chunk (side benefit). ForwardBatch rows = total tokens across members; same governor.
* **Pipeline (#366).** One governor per device; stage budgets independent.
* **Precision rules.** No stored tensor is added or widened (rule 1); no weight expansion (rule 2); timestamps are an
  8 B/query pool. Layer splitting changes no kernel. Chunking may select a different GEMM arm, which is already normal.
* **Cross-backend rule.** CUDA has no equivalent watchdog failure on the same card (the same quants run on the 3060 under
  CUDA); no propagation required. Record that in the PR; HIP/Windows is untested.

## 8. Test plan

### 8.1 CPU-only / no-GPU unit tests (run anywhere, fast)

Make the governor a pure class over an injected clock and cost callback, so decisions are testable without a device.

* Budget respected: for synthetic per-layer costs, no planned submit exceeds the budget unless a single layer does.
* Cold start splits every layer, then relaxes after one measurement.
* Shrink-on-overrun and bounded ramp-up (<= 25 %/step); chunk always a multiple of 64 within `[64, 1024]`.
* Explicit `PREFILL_SPLIT_LAYERS` / `--prefill-chunk-size` precedence; `BUDGET_MS=0` is exactly today's behaviour (zero splits).
* Device-class gate table (discrete on, UMA measure-only, override).
* **Split-count oracle**: expose `SubmitContext.SubmitCount` (diagnostic counter) so a test can assert how many submits a
  forward produced; combined with the bit-exactness test below this fixes the real behaviour (a mere "output matches"
  cannot tell a split from a no-op; per the "prove a fast path ran" rule).
* **Bit-exact split test** (Strix/any Vulkan device, CI-eligible when Vulkan is present): greedy text and logits identical
  with the governor forced to split every layer versus unsplit (layer splitting must be bit-exact; this is the discriminator
  for a lost barrier at the split boundary). Chunking is covered by KL/top-1 parity, not bit identity.
* Add `ForwardBatch` and one hybrid host (Qwen3.5 small) to the bit-exact split test, so the shared helper is exercised at
  every host shape it is wired into.

### 8.2 Named substitution mutants (must each fail at least one test; list killed/survived in the PR)

| Mutant | Expected killer |
|---|---|
| M1 `MaybeSplit` never returns true (budget ignored) | split-count oracle; 3060 gate (DEVICE_LOST) |
| M2 predicted cost independent of rows | budget-respected unit test with varying rows |
| M3 split placed **before** layer 0 / skipping the last boundary (off-by-one, cf. #1761's `layer > 0`) | split-count oracle with exact boundary assertions |
| M4 post-split `HostToComputeBarrier` removed | bit-exact split test on a device (barrier hazard); on a fake only the call-order test |
| M5 controller never shrinks after an overrun | shrink unit test |
| M6 UMA/discrete gate inverted | device-class table test; Strix ABBA (extra syncs would show) |
| M7 explicit env ignored in favour of auto | precedence test |
| M8 chunk not rounded to 64 (breaks block alignment with paged KV) | rounding test + paged-KV S3 block-boundary test |
| M9 wall time used instead of `max(wall, span)` in a contention fake (wall large, span small) | contention unit test using the fake clock |

A mutant that **survives** is a finding; fixtures must be able to tell right from wrong (a synthetic cost function with
non-uniform layers, not a uniform one, or M3 cannot die).

### 8.3 Performance gate (same-session ABBA, Strix, no regression)

On the Strix box, governor on (measure-only) vs `BUDGET_MS=0`, TwIL-8B Q4_K_M and one 1B Q8_0 model, prefill 1.8K and decode
64 tokens, order A B B A, a warm-up pass first (clock-ramp confound, 2-3x phantom deltas), GPU quiet (`Get-Process`
before/after; UMA contention swings ~40 %). Gate: |delta| inside the measured noise of the A-vs-A control pair **recorded in the same session**; a
control arm that cannot disagree is not a control. Also verify the timestamp stamps cost nothing visible in decode.

### 8.4 What can only be validated on the RTX 3060 (framework laptop): explicit GPU-window request

Nothing may be launched until the owner grants a window and the lock is taken at execution time (3060 is shared; use the
ResourceLock claim for that machine's GPU, not now). **Estimate: ~2.5 h with a hard cap of 3 h**; ordered so the cheap,
decisive items come first and the long ones can be dropped:

1. Sanity (~10 min): verify the TwIL GGUF SHA-256 (`17BDA82D...`, the earlier corrupt copy was `6C91B507...`); RED arm:
   unsplit 1.8K prompt must fail (otherwise the device state does not reproduce and the rest is meaningless).
2. TdrDelay experiment (~20 min, needs admin and an explicit owner OK, revert afterwards): `TdrDelay=10`, unsplit 1.8K prompt.
3. E2 fenceless split every 2 / every 5 at 1.8K (~15 min).
4. Auto governor on vs the three manual arms on the same prompt: auto, chunk 128, `SPLIT_LAYERS=2`; N=3 each at 299 / 530 /
   1.8K tokens (~40 min). Pass: 0 `DEVICE_LOST`, and auto >= 0.95 x chunk-128 tok/s (84.7 on the dev build).
5. 8K prompt, auto (~10 min; ~3-4 min per run at 40-50 tok/s) and Llama-3.2-3B Q8_0, 1B Q8_0, Tev1-4B (counter-evidence
   models; measure the cost of conservative splitting on those, target <= 5 %).
6. `VulkanGraniteF16KvTests` (the #1756 acceptance test) with the governor, and #1748's `Bench_LlamaProj_64x4096x4096` /
   `...CoopmatDefault_PathProof...` (~30 min). Note the #1756 KL result on the 3060 (top-1 97.0 % vs Strix >= 99.8 %) is a
   **separate** finding and is out of scope here; do not let it block this issue's close-out.
7. GPU-timestamp profile at rows 128 vs 1,799 to explain the 1.65x per-row efficiency gap (Open Question 9).

Not 3060-only (can run on Strix): everything in 8.1 and 8.3.

## 9. Slices

| Slice | Content | Needs GPU |
|---|---|---|
| A | Pure governor + unit tests + `SubmitCount` + device-class gate (measure-only everywhere) | No |
| B | Wire `MaybeSplit` into the dense host layer loop (replaces `layer % N` when auto), ForwardBatch, and each host that has a layer loop; timestamps in `SubmitAndWait` | Strix for bit-exact |
| C | `IModel.NextPrefillChunkTokens` hook + `TextGenerator` per-chunk call; coordinate with #1723 | Strix for parity |
| D | Enable enforcement for discrete GPUs; 3060 window (8.4); docs/GPU.md section | **3060** |
| E | Optional: fenceless N-buffer splits (E2), if it passes | **3060** |

## 10. Open questions for the owner

1. **Confirm the mechanism with a TdrDelay experiment?** It needs an admin registry edit on the Framework laptop (revert
   after) and settles watchdog vs something else. **Recommendation: yes**, as step 2 of the window; the design is the same
   either way, but a negative result would change the budget rationale and the Linux default.
2. **Budget default: 1000 ms or 1500 ms?** 1.5 s is the largest submit proven to pass; 1 s leaves a 2x margin and splits the
   128-row TwIL forward in two. **Recommendation: 1000 ms**, adjustable; revisit with the 8.4 step-5 cost numbers.
3. **Enforcement scope.** Discrete on; UMA measure-only; a display-driving iGPU needs the explicit switch. **Recommendation:
   accept as in section 6**; revisit if a TDR is ever reproduced on Strix or Arc.
4. **Controller signal: `max(fence wall, timestamp span)`** or timestamps only? **Recommendation: the max**, since the
   watchdog counts wall time under contention.
5. **API change: add `IModel.NextPrefillChunkTokens` (engine chunks adaptively) in addition to the model-internal
   layer splitting?** Alternative is model-internal only (slice B), which fixes correctness but leaves the 1.65x throughput on
   the table. **Recommendation: both**, B first (correctness for every caller), C second (performance and #1723).
6. **Allow the fenceless-split experiment (E2) and an N-buffer ring in `SubmitContext`?** **Recommendation: yes**,
   conditional on the 3060 result; it is the only route that could remove the split cost entirely.
7. **Cold start**: split every layer until the first measurement (cheap, proven), versus persisting a per-(device, model)
   calibration file. **Recommendation: cold-start split; no persistence in v1.**
8. **Failure handling**: no automatic retry after `DEVICE_LOST` (the device is gone); improve the message only.
   **Recommendation: accept.**
9. **Investigate the 1.65x per-row efficiency gap (single 1.8K pass 35 s vs chunk 128 21 s)?** If it is real on every
   backend, chunked prefill might be a default for every device, not only a TDR fix. **Recommendation: a separate issue** with a
   Strix measurement (UMA), so it does not gate #1760.
10. **Link to #1723**: ship the auto chunking as that issue's mechanism (one design) rather than a fixed per-device chunk.
    **Recommendation: yes**; close #1723 through slice C after the progress-bar and cancel behaviour are demonstrated.
11. **GPU window**: grant ~2.5 h (cap 3 h) on the 3060 per section 8.4, and the SHA-256 guard on the TwIL GGUF.
    **Recommendation: grant; schedule after slices A-C are green on Strix** so the window is spent only on 3060-specific facts.

## 11. References

`src/DotLLM.Vulkan/VulkanDevice.cs` (`SubmitContext`, `SubmitAndWait`, `SplitSubmit`, `TimestampPeriodNs`);
`src/DotLLM.Vulkan/VulkanTransformerModel.cs` (`PrefillSplitLayers`, layer loop); `src/DotLLM.Vulkan/VulkanQwen3HybridDenseTransformerModel.Profiling.cs`
(`DefaultPrefillChunkTokens`); `src/DotLLM.Engine/TextGenerator.cs` (`EffectivePrefillChunkSize`);
`src/DotLLM.Core/Models/IModel.cs` (`PreferredPrefillChunkTokens`); `docs/perf/VULKAN_PAGED_KV_DESIGN.md`; `docs/GPU.md`;
#1760, #1748, #1761, #1756, #1723, #1100, #914/#1649, #143, #348, #366, #1286.
