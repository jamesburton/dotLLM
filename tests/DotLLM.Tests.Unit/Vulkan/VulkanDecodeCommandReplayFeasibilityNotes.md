# Vulkan decode command-buffer replay — feasibility notes (2026-07-18)

Scope: investigate whether a CUDA-graph-style "record once, replay every token"
mechanism is worth building for the Vulkan decode path, per
`.docs/KERNEL_MAP.md` §10's "SUSPECTED-minor ~1000+ fresh commands per token"
note. No source files were edited; this is a scoping note only.

## 1. Actual command count per decode token (SmolLM-135M, 30 layers)

Measured directly by the #143 campaign's `DOTLLM_VULKAN_DECODE_PROFILE=1`
census (`.docs/KERNEL_MAP.md` §12, experiment 1, baseline):

- **573 `vkCmdDispatch`**, **546 `vkCmdPipelineBarrier`**, **62
  `vkCmdCopyBuffer`** per decode token (pre-#150 baseline; #150 later cut
  ~140-224 dispatches/barriers per token on larger mixed-quant models by
  sharing activation-quantize ops across MMVQ groups, so current SmolLM
  counts are somewhat lower, but same order of magnitude).
- Each dispatch is additionally preceded by 1 `vkCmdBindPipeline`, 1
  `vkCmdBindDescriptorSets`, and (for most kernels) 1+ `vkCmdPushConstants`
  call (`KernelSupport.cs`, `AllocateDescriptorSet`/`WriteBufferBindings`
  pattern used by all ~130 kernel `Record` methods under
  `src/DotLLM.Vulkan/Kernels/*.cs`), plus one `vkAllocateDescriptorSets` +
  one `vkUpdateDescriptorSets` host-side call per dispatch.
- Total host-side Vulkan API calls per decode token: dispatch(573) +
  barrier(546) + copy(62) + bind-pipeline(~573) + bind-descriptor-set(~573)
  + descriptor-alloc(~573) + descriptor-write(~573) + push-constants(~500+)
  ≈ **3,800-4,000 calls/token**.

**Verdict: the KERNEL_MAP §10 claim of "~1000+ fresh commands per token" is
correct and, on a strict `vkCmd*`-only count (dispatch+barrier+copy =
~1,180), actually understates it — the full host-side API-call count
(including descriptor churn) is closer to 3,800-4,000/token.**

## 2. VK_KHR/EXT_device_generated_commands driver support (best-effort, no runtime check)

- The relevant extension for GPU-driven command generation is
  `VK_EXT_device_generated_commands` (the industry-converged EXT form; there
  is no shipping `VK_KHR_device_generated_commands` — KHR variant does not
  exist in the registry, only `VK_NV_device_generated_commands` (NVIDIA,
  legacy) and `VK_EXT_device_generated_commands` (cross-vendor, ratified
  2024).
- **AMD Windows (Adrenalin) driver**: `VK_EXT_device_generated_commands` was
  added starting with **Adrenalin 25.8.1** (August 2025 release notes:
  "AMD Software: Adrenalin Edition ... Expanded Vulkan Extension Support"),
  alongside `VK_KHR_depth_clamp_zero_one` and `VK_KHR_robustness2`. Verified
  present on RX 9070 / RX 7900 (RDNA3/4 discrete) in that release's notes.
- **gfx1151 (Strix Halo iGPU) specifically: NOT independently confirmed.**
  Strix Halo shares the same Adrenalin driver stack as discrete RDNA, so it
  is plausible the extension is exposed there too on a current (2026)
  driver, but this needs a **runtime `vkEnumerateDeviceExtensionProperties`
  / `VkPhysicalDeviceFeatures2` check on this actual box** before relying on
  it — flagging as unverified rather than assuming.
- Even where present, DGC is a heavyweight, indirect-buffer-driven
  mechanism (GPU reads an "indirect commands layout" from a buffer to
  generate binds/dispatches) designed for large draw/dispatch-count scenes
  (thousands of draws) with token-level branching. It is not a drop-in
  "replay this fixed sequence" primitive — adopting it would mean building
  and maintaining an indirect-commands-layout buffer format alongside the
  existing direct-recording kernels, a large structural investment for a
  workload (30 layers × ~19 dispatches/layer, entirely static shape) that
  doesn't need per-dispatch GPU-side branching at all.

## 3. Secondary command buffer replay — push-constant compatibility issue

The simpler mechanism (record decode once into a `SIMULTANEOUS_USE`
secondary command buffer, `vkCmdExecuteCommands` it from a thin primary each
token) hits a real constraint: **push constants are baked into the
recording, not read from a bound buffer.** Decode's per-token push constants
include the KV-cache write offset and RoPE position — both of which change
every single token. A secondary buffer recorded once cannot encode a
different position/offset per replay; `vkCmdPushConstants` calls would have
to be re-issued from the *primary* buffer wrapping the secondary, which:

- Only works because push constants happen to be visible to subsequently
  bound pipelines in the *same* command buffer — but the compute pipeline
  binds and dispatches themselves are inside the secondary buffer, so the
  primary would need to push constants for ~19 ops/layer × 30 layers = ~570
  separate push regions *before* invoking the secondary, which defeats most
  of the host-recording savings (the primary buffer's per-dispatch
  `vkCmdPushConstants` calls dominate the same way the direct-recording
  path's calls do today).
- Alternative: move the changing values (KV offset, position) out of push
  constants into a small SSBO/UBO updated once per token via
  `vkCmdUpdateBuffer` or a host-visible mapped write — architecturally
  viable, but a nontrivial kernel-interface change touching every one of the
  ~130 kernel files' descriptor/push-constant layout, not a localized
  optimization.

## 4. Does this save anything BEYOND what #143's split-submit overlap already captures?

This is the crux question, and the answer is **no, not meaningfully.**

From `.docs/KERNEL_MAP.md` §12's residual attribution (SmolLM @545 tok/s,
1.835 ms/tok wall):

- Total recording cost (pre-#143 overlap): **~0.23-0.29 ms/token** (8
  µs/layer × 30 layers ≈ 0.24 ms, matches the profiler's `record` bucket).
- #143's `SplitSubmit()` (mid-forward split at layer 8, no-fence submit,
  continue recording while GPU executes the first chunk) already hides
  **~2/3 of that** behind GPU execution. The KERNEL_MAP explicitly records
  the residual: **"~0.08 ms/tok — record of pre-split chunk (8 layers ≈ 65
  µs) + host loop — framework — closable with a 3-buffer multi-split ring."**
- The KERNEL_MAP's own "Remaining follow-up levers" list (§12, item 3)
  already names the next increment: a **3-buffer split-submit ring** to hide
  the *remaining* exposed recording (~0.05-0.08 ms/tok), not a full
  record-once/replay-forever mechanism.

So the ceiling for ANY recording-elimination technique — replay, DGC,
secondary buffers, or a deeper split ring — is bounded by that **~0.05-0.08
ms/tok remaining exposed recording cost**, which is **~3-4% of the current
1.835 ms/tok decode wall time**. A full replay mechanism (DGC or secondary
buffers) would have to:
1. Solve the push-constant-per-token problem (non-trivial kernel-interface
   change, §3 above), which itself likely re-introduces host-side work
   proportional to the layer count — eating back most of the theoretical
   saving.
2. Do all of this to compete with a **cheaper, already-shipped, already
   bit-exact mechanism** (#143's split-submit) that captures the large
   majority of the available win for a fraction of the complexity, and that
   the codebase's own backlog already earmarks for a small further
   increment (3-buffer ring) rather than a rewrite.

Given the noise floor on this box for perf claims is documented at up to
~40% swing from UMA memory-bandwidth contention (Strix Halo memory notes),
a ≤4%-of-wall-time theoretical ceiling is very unlikely to be measurable as
a real win at all, let alone justify the invasiveness of DGC/secondary-buffer
adoption plus a push-constant-to-UBO kernel-interface migration across ~130
kernel files.

## 5. Conclusion / priority recommendation

- **Do not pursue VK_EXT_device_generated_commands or secondary-command-buffer
  replay for Vulkan decode.** The remaining recording-cost headroom
  (~0.05-0.08 ms/tok, ~3-4% of wall time) is already the *documented next
  increment* on the existing split-submit mechanism (3-buffer ring, KERNEL_MAP
  §12 lever 3) — cheap, incremental, bit-exact-proven pattern — not a new
  mechanism.
- If the 3-buffer ring is ever built and the residual is still deemed worth
  chasing further, the correct next step is a **runtime capability check**
  (`vkGetPhysicalDeviceFeatures2` for `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`)
  on this specific gfx1151 device/driver before any implementation work,
  since driver support for the extension on this iGPU (as opposed to
  discrete RDNA) is unverified.
- Recommend downgrading the KERNEL_MAP §10 "SUSPECTED-minor" note to
  **"CONFIRMED-minor, superseded by #143's split-submit; no further action
  planned"** rather than leaving it open as a distinct investigation
  target — it is not an independent lever, it is the same lever #143
  already picked most of the fruit from.
