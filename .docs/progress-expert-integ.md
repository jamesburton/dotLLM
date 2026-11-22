# Expert integration progress (#1920 T7/T8, CPU taps)

Mutants applied to the committed implementation on all three CPU models (TransformerModel, Qwen3HybridDense, Qwen3MoeHybrid),
suite `FullyQualifiedName~HiddenStateTap` (68 tests). Each was reverted with `git checkout --`.

| mutant | where | result |
|---|---|---|
| tap writes back into the residual (`*rows += 1f` after dispatch) | each model's DispatchTap | KILLED, 18 failures (perturbation, abort-control, ForwardBatch) |
| tap fires on unsubscribed layer (per-entry layer mask ignored) | Registry Entry.Wants | SURVIVED the single-tap test (union filter hides it); added `TwoTaps_OnDifferentLayers_EachSeeOnlyTheirOwn`, then KILLED (3) |
| abort skips KV rollback | each model | KILLED, 9 (last-layer, pre-lm-head, prefill abort; the embedding-site abort writes no KV so it cannot see it) |
| abort skips recurrent restore | both hybrids | KILLED, 6 (hybrids only; llama unaffected as expected) |
| wrong row index (Last = second-to-last row) | Registry Dispatch | KILLED, 20 |
| PreLmHead delivers pre-norm data | each model | KILLED, 3 (`Rows_LastVsAll_AndFinalPointsRelate`) |
- T9 dense VulkanTransformerModel taps wired: llama suite 15/15 on Vulkan (staging scheme in src/DotLLM.Vulkan/VulkanHiddenStateTaps.cs).

## T9 Vulkan taps (#1920)

Hosts: VulkanTransformerModel, VulkanQwen3HybridDenseTransformerModel, VulkanQwen3MoeHybridTransformerModel (shared staging helper `VulkanHiddenStateTaps`).
Suite `HiddenStateTapVulkanTests` (49 tests: 45 abstract-suite + deferred/immediate plan equivalence x3 + MoE fused-decode bypass) green, CPU suite still green.
Tolerance: tapped vs untapped is bit-identical (same path) everywhere except hybrid-dense ForwardBatch (untapped = fused batch, tapped = per-sequence): 1e-2 abs + identical argmax (`BatchBitExact`).

| mutant (committed code, reverted with git checkout --) | result |
|---|---|
| tap writes back into the device residual (reverse copy, shifted) | KILLED, 14 (llama + hybrid-dense; MoE reads back synchronously so cannot be hit) |
| fires on unsubscribed layer (PostLayer dispatched as last layer) | KILLED, 7 |
| abort skips KV rollback (all 3 hosts) | KILLED, 9 |
| abort skips recurrent restore (both hybrids) | KILLED, 6 |
| MoE fused decode not bypassed with a tap | KILLED, 5 (counter test + MoE capture/abort) |

## T10 CUDA taps (#1920)

Hosts: CudaTransformerModel (dense/GQA/MoE), CudaQwen3HybridDenseTransformerModel, CudaQwen3MoeHybridTransformerModel; shared helper
`src/DotLLM.Cuda/CudaHiddenStateTaps.cs` (pinned staging, deferred vs immediate plan, host F16->F32 and unscale).
Suite `HiddenStateTapCudaTests` (45 abstract + CPU parity x3 + deferred/immediate x3 + graph bypass x2 + tiled hybrid-dense + MoE refusal/accept).
No NVIDIA on this box: BUILT and DISCOVERED only; locally every CUDA test SKIPS (0 failed). CPU suite 68/68 and Vulkan suite 49/49 green after the change.
NEVER RUN on a GPU: the whole CUDA path is unverified until run on the T5500 (ask the user first).

### Mutants to run on a GPU box: all NOT RUN

| mutant | where | killing tests |
|---|---|---|
| tap writes back on device (e.g. cuMemcpyHtoD of a +1 row into HiddenState after the site) | helper / each host | `MultiLayerTap_DoesNotPerturb_TokensLogitsOrKv`, `Rows_LastVsAll...`, `Abort_TryForward...` (all three families) |
| fires on an unsubscribed layer (TapSite ignores snap.Wants / Plan returns true) | `CudaHiddenStateTaps.Plan` | `Tap_FiresOnlyOnSubscribedLayers`, `TwoTaps_OnDifferentLayers_EachSeeOnlyTheirOwn` |
| graph not bypassed (drop the `ts is null` guard on the graph branch) | `CudaTransformerModel.ForwardInternalCore`, hybrid `ForwardTile` | `DecodeGraph_IsBypassed_WhenATapIsAttached` (counter and sample count); samples would also vanish |
| abort skips KV rollback | dense `ForwardInternal`, hybrid `ForwardCoreTapped`, MoE `ForwardTapped` | `Abort_TryForward_LeavesModelAsIfNeverStarted`, `Abort_DuringPrefill_RollsBackToEmpty`, `HybridDense_TiledPrefill_...AbortUndoesEveryTile` |
| abort skips recurrent restore | hybrid dense (MoE once #1919 lands) | `Abort_TryForward...` (hybrid-dense) |
| wrong source buffer: last layer reads `Residual` instead of `HiddenState` | dense PostLayer site | `TappedRows_MatchTheCpuModel_AtEveryPointAndLayer` (llama); NOT the perturbation tests |
| PreLmHead All reads row 0 after the one-row norm overwrote it | dense | `Rows_LastVsAll_AndFinalPointsRelate`, CPU-parity test |
| Flint unscale skipped (1/resScale -> 1) | dense | NO KILLING TEST: no Flint (or BitNet) synthetic fixture exists; needs a tiny Flint GGUF builder |
| BitNet ResidualF32 not used | dense | NO KILLING TEST (same reason) |

### Deviations / decisions
- CUDA `ExpectsPackedBatchFallback = false` and `PackedBatchFallbacks` is never bumped on CUDA (no packed matmul batch exists there); a tapped batch runs per request synchronously.
- CUDA fixtures switch the decode graphs off (untapped control = eager) so comparisons are bit-exact; bypass tests switch them on.
- Hybrid-dense and dense: an every-row tap makes a long hybrid-dense prefill untiled (VRAM cost, documented).
- MoE hybrid: abortable tap refused (NotSupportedException citing #1919) until `SupportsRecurrentStateCheckpoint`; test passes either way. When #1919 merges, its checkpoint members implicitly implement the IModel ones used through `((IModel)this)`.
- Suite change: `HiddenStateTapFixture.SupportsAbortableTaps` (default true) gates the three abort tests.
