# IQ3 per-host parity — deferred work surfaced by host-level tests

Created: 2026-05-18 (audit H3 follow-up)
Status: 3 of 4 Vulkan transformer hosts have a missing IQ3 upload-path predicate;
        all 4 hosts now have host-level forward tests but 3 are skip-gated.

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

4. `src/DotLLM.Vulkan/VulkanQwen3MoeHybridTransformerModel.cs` has no
   `BuildFromPrebuiltWeights` factory — only `BuildFromGguf`. The other 3
   IQ3 tests use `BuildFromPrebuiltWeights` on both backends to skip the GGUF
   loader; for Qwen3MoeHybrid we'd need either:
   - (a) Add a `BuildFromPrebuiltWeights` factory mirroring the dense /
     NemotronH equivalents (the CPU-side
     `Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights` already
     exists). Estimate: 30 lines.
   - (b) Synthesise an IQ3 GGUF in-test via the existing `GgufTestData`
     writer. Qwen3MoeHybrid carries 40+ tensors per layer (GDN recurrence
     + sparse MoE FFN + optional shared expert); the GGUF-writer wiring is
     substantial. Estimate: half a session.

Path (a) is the clean fix and unblocks Option A or Option B in the task
statement — recommended.

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

- (1) Mamba3 predicates: 15 min.
- (2) NemotronH predicates: 15 min.
- (3) Qwen3MoeHybrid predicates: 15 min.
- (4) Qwen3MoeHybrid `BuildFromPrebuiltWeights` factory: 30 min.
- Qwen3MoeHybrid fixture builder (the test-side scaffolding) and asserting
  parity: 1-2 hours.

Total: half a session to clear all 4 hosts to PASS-or-fail-loud.
