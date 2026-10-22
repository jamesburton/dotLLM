# IQ3 per-host parity — deferred work surfaced by host-level tests

Created: 2026-05-18 (audit H3 follow-up)
Updated: 2026-05-18 (session 6 — Qwen3MoeHybrid IQ3 fixture-builder shipped, audit-H3 fully closed)
Status: **CLOSED.** All 4 Vulkan transformer hosts have IQ3 upload-path predicates,
        prebuilt-weights entrypoints, and host-level forward parity tests PASSING
        on Strix Halo (16 tests total: 12 from session 5 + 4 Qwen3MoeHybrid from
        session 6). Audit-H3 trap-the-bug discriminator (kernel-level ✓ ≠
        dispatch-level ✓) is now closed for IQ3 across the entire Vulkan host
        matrix. Note: this file is retained as a historical record of the
        4-host audit closure; new IQ-family work should not extend it.

## Summary

Audit `.planning/notes/2026-05-18-state-audit.md` §H3 flagged that the IQ3
dispatch arms wired into all 4 Vulkan transformer hosts (commits `07f391f` /
`48d65fe` / `146d747` / `ad6b853`) had only kernel-level parity coverage (16
IQ3 kernel tests). No host-level test verified that the dispatch arm actually
fires during a model forward — a miswired case label or wrong codebook handle
would compile, pass kernel tests, and silently mis-decode at inference.

This deferred-notes file documents the **non-test gaps** the new host-level
tests surfaced. The tests themselves ship in this PR:

- `tests/DotLLM.Tests.Unit/Vulkan/VulkanTransformerModelIq3ForwardTests.cs`
  (dense host — PASSES on Strix Halo, exercises IQ3_S + IQ3_XXS through the
  full Vulkan dispatch).
- `tests/DotLLM.Tests.Unit/Vulkan/VulkanMamba3TransformerModelIq3ForwardTests.cs`
  (Mamba3 host — skip-gated, see below).
- `tests/DotLLM.Tests.Unit/Vulkan/VulkanNemotronHTransformerModelIq3ForwardTests.cs`
  (NemotronH host — skip-gated, see below).
- `tests/DotLLM.Tests.Unit/Vulkan/VulkanQwen3MoeHybridTransformerModelIq3ForwardTests.cs`
  (Qwen3MoeHybrid host — skip-gated on missing prerequisites, see below).

## Audit H3 trap-the-bug finding

The trap-the-bug discriminator inside the Mamba3 and NemotronH tests asserts
the device-side quant type is IQ3 after upload. Without this assertion the
parity check would still pass — both backends would consume F32 (the CPU
dequant path and the Vulkan F32-fallback both produce identical F32 outputs)
— but the IQ3 dispatch arm would never fire. **The Mamba3 / NemotronH tests
surface the bug exactly this way** — they fail closed (skip) because the
upload predicate routes the IQ3 source down to F32 dequant at upload time.

In other words: the kernel-level test ✓ does not prove the model-level ✓.
Confirmed in tree on 3 of the 4 hosts. Dense host (the one with the
correctly wired upload-path predicates) passes the parity test.

## What's missing — three concrete predicate adds

The dense host's `src/DotLLM.Vulkan/VulkanWeights.cs` already has:

```csharp
private static bool KeepIq3XxsOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
    => !dequantToFp32 && qt == QuantizationType.IQ3_XXS && (inputDim % 256) == 0;

private static bool KeepIq3SOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
    => !dequantToFp32 && qt == QuantizationType.IQ3_S && (inputDim % 256) == 0;
```

And they're listed in `DeviceQuantTypeFor`. **These predicates need to be
mirrored into:**

1. `src/DotLLM.Vulkan/VulkanMamba3Weights.cs` — add `KeepIq3XxsOnDevice` /
   `KeepIq3SOnDevice` to the file's predicate set and list them in
   `KeepQuantOnDevice`. Each is a ~3-line method. Bonus: `MaxStagingBytes`
   accounting via `Dequantize.RowByteSize(..., IQ3_S)` for the staging-buffer
   sizing path (already auto-correct because the existing
   `KeepQuantOnDevice` returns `false` for IQ3 today; once the predicates
   land, the staging-bytes branches need IQ3 rowstride lookups).

2. `src/DotLLM.Vulkan/VulkanNemotronHWeights.cs` — same shape. Add the
   `KeepIq3*OnDevice` predicates, list them in `DeviceQuantTypeFor` and in
   `KeepNative`, and extend `ProjectionUploadBytes` to compute IQ3 row-stride
   bytes.

3. `src/DotLLM.Vulkan/VulkanQwen3MoeHybridWeights.cs` — same shape. Add the
   `KeepIq3*` predicates, list them in `DeviceQuantTypeFor` and `KeepNative`,
   extend `ProjectionUploadBytes`. Plus the additional Qwen3MoeHybrid
   prerequisites below.

Each of (1) / (2) is roughly a 15-line patch. The companion `VulkanWeights`
predicate is the authoritative reference.

## Qwen3MoeHybrid additional prerequisites

The Qwen3MoeHybrid IQ3 host test file is skip-gated on a SECOND missing
piece on top of (3) above:

4. `src/DotLLM.Vulkan/VulkanQwen3MoeHybridTransformerModel.cs` `BuildFromPrebuiltWeights`
   factory — **SHIPPED in session 6** (this commit). Mirrors the
   `VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights` signature and
   delegates to `VulkanQwen3MoeHybridWeights.Upload` with the caller-owned
   pointers. Uses `gguf: null, cpuModel: null` — disposal frees only device-side
   weights, forward scratch, the GDN state cache, and kernels; the caller owns
   the unmanaged pointers and the `Qwen3MoeLayerWeights[]` array.

## Re-enabling the skip-gated tests

Once the predicate adds in (1), (2), (3) land, the corresponding host's
IQ3 tests will auto-promote from skip to active parity tests (the
discriminator's `Skip.IfNot` check unblocks once
`vkWeights.LmHeadDeviceQuantType == iq3Type` is true after upload).

For Qwen3MoeHybrid, after (3) AND (4) land, the skip-only test class needs
its actual fixture builder fleshed out — see the class XML doc for the
prescription. Until then it ships as a place-holder pinning the gap so
nobody quietly forgets.

## Why ship the tests now if 3 of 4 are skip-gated?

- The dense host IQ3 test PASSES — that's a real new parity assertion in
  tree, proving commit `07f391f`'s dispatch wiring works end-to-end.
- The Mamba3 + NemotronH skip messages diagnose the exact next step
  (file + predicate name + the dense-host reference). A future
  contributor adding the predicates will see the test auto-promote to
  parity-passing as soon as they're correct.
- The Qwen3MoeHybrid skip-only class pins the gap so it doesn't get
  silently lost when the audit notes get archived.
- Even in skip state, the tests catch one regression class: a future
  loader change that quietly drops the IQ3 source quant type on the
  CPU-side (rerouting it through Q8_0 or F32 at load time) would cause
  the discriminator to fail differently — the diagnostic stays useful.

## Estimated work to clear the deferred bin

- (1) Mamba3 predicates: SHIPPED (session 5).
- (2) NemotronH predicates: SHIPPED (session 5).
- (3) Qwen3MoeHybrid predicates: SHIPPED (session 5).
- (4) Qwen3MoeHybrid `BuildFromPrebuiltWeights` factory: SHIPPED (session 6).
- (5) Qwen3MoeHybrid IQ3 fixture builder + parity assertions: SHIPPED (session 6).
  All 4 tests pass on Strix Halo at abs 0.15 / rel 0.15 IQ3 envelope; routed
  MoE banks stay F32 (Vulkan `moe_indexed_matmul` is F32-only); GDN + full-attn
  layers + lm_head exercise the IQ3 dispatch end-to-end. The dense host pattern
  + the existing CPU `Qwen3MoeHybridFixtureBuilder` (in
  `Qwen3MoeHybridTransformerModelTests.cs`) were the structural templates;
  dimensions bumped to satisfy IQ3's `% 256 == 0` contraction-axis rule
  (NVHead×DState=256 for GDN OutWeight, NumAttentionHeads×HeadDim=256 for
  full-attn OWeight).

Total: complete. Audit-H3 IQ3 coverage matrix is full.
