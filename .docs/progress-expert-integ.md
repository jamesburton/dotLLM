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
