# Issue #144 ledger — hazard-scoped barriers (2026-07-16, branch `issue/144-hazard-barriers`)

**MERGE TARGET: append as §13 to the main repo's `.docs/KERNEL_MAP.md`; also apply the two
edits at the bottom (env index §0b + §12 lever-1 status).** Written from the worktree because
the agent sandbox blocks direct main-checkout edits.

## 13. Issue #144 — hazard-scoped barriers (2026-07-16, `issue/144-hazard-barriers`)

Mission: replace the full-pipeline barrier after essentially every dispatch (~573/token SmolLM
decode) with per-buffer RAW/WAR/WAW hazard tracking so independent ops overlap (llama.cpp's
ggml-vulkan model). Target was 546 → ≥625 tok/s SmolLM decode.
**Result: implemented, exact-token-parity-proven on all gate models, but PERF-NEUTRAL on gfx1151
(±0.5–1% on every model, decode AND prefill). The #143 premise "wall = sum because of blanket
barriers" was WRONG for dotLLM's op chain — see verdict.**

### Design (as landed on the branch)
- `VulkanHazardTracker`: per-buffer last-write/last-read epochs + ONE `_lastBarrier` watermark
  (a barrier is a global memory barrier, so one watermark captures it exactly). Op *i* barriers
  iff it reads a buffer with write-epoch > watermark, or writes one with write- OR read-epoch >
  watermark. Single batched `vkCmdPipelineBarrier` (COMPUTE|TRANSFER → COMPUTE|TRANSFER,
  SHADER_W|TRANSFER_W → SHADER_R|W + TRANSFER_R|W) — superset of every legacy non-host shape.
- **Access sets can't drift from shaders**: per-binding writes mask reflected from SPIR-V
  `NonWritable` decorations at module load (`SpirvReflection`), threaded VulkanModule →
  ComputePipeline → DescriptorSetCache. `readonly buffer` ⇒ read; unqualified ⇒ conservative
  read+write. Parse anomaly ⇒ all-writable (degenerates to blanket = safe).
- **Choke point**: `DescriptorSetCache.GetOrCreate` is the ONLY descriptor path (audited: 1:1
  GetOrCreate↔vkCmdDispatch in all 128 kernel classes), so every dispatch is guarded exactly
  once, immediately before recording. Forward-path `vkCmdCopyBuffer` sites (embed gather,
  KV/MLA cache appends, LM-head row copy, MoE fold, LoRA delta) carry explicit
  `OnTransfer(src,dst)` guards. No vkCmdFillBuffer/UpdateBuffer exist in the backend.
- Tracked scope: `VulkanTransformerModel.Forward` (dense + Gemma-4 MoE + generic MoE + MLA +
  diffusion tail). Batched forward, diagnostics paths, Mamba3/NemotronH/Qwen3MoeHybrid models
  (separate IModel classes) stay on bit-identical legacy blanket barriers. Host barriers
  unconditional. SubmitContext arms/disarms across Begin/SplitSubmit/SubmitAndWait; epoch state
  survives SplitSubmit (barriers sync across same-queue submission order).
- Kill-switch `DOTLLM_VULKAN_LEGACY_BARRIERS=1`; debug `DOTLLM_VULKAN_HAZARD_VALIDATE=1`
  (barrier at every guard), `DOTLLM_VULKAN_HAZARD_DEBUG=1` (log offending buffer).

### Experiment ledger
| # | Change | Result | Verdict |
|---|---|---|---|
| 1 | Naive tracker (2-pass dict guard, per-forward Clear, splitkv internal barriers kept) | SmolLM decode barriers 546→**576**/tok (+30!), 536→531 tok/s | REGRESSION: splitkv's 2 internal blanket barriers doubled with the guards (+2/layer); record +0.15 ms/tok guard cost |
| 2 | Hazard-debug trace audit | guard logic CORRECT: skips exactly K/V/up GEMVs + V-cache-copy; every other decode edge is a true dependency | the "overlap pool" at decode is ~empty — shared-quant groups already ran Q/K/V + gate/up unbarriered |
| 3 | Tracker-aware kernel-internal barriers (splitkv, fused-LoRA suppress when armed) + single-pass guard (1 dict lookup/buffer, check+stamp) + warm epochs (Begin keeps dict, advances watermark) | barriers 576→**516**/tok, record 0.31→**0.24** ms/tok (below legacy!), SmolLM 539.8 median | fixed; net barrier count now < legacy |
| 4 | Full perf matrix (below) | neutral everywhere | hazard scoping is NOT a decode/prefill lever on gfx1151 |

### Perf matrix (same-day back-to-back process pairs, `dotllm bench -p 512 -n 128`, medians; legacy = kill-switch on)
| Model | Decode hazard | Decode legacy | Prefill hazard | Prefill legacy |
|---|---|---|---|---|
| SmolLM-135M Q8_0 (r3/r5, 3 pairs) | 539.96 / 535.61 / 539.78 tok/s (best 544.6) | 538.86 / 536.25 / 528.09 (best 541.1) | 12000–12260 pp | 11407–12260 pp |
| Llama-3.2-3B IQ4_XS (2 pairs) | 75.81 / 75.89 | 75.79 / 76.30 | 1742 / 1879 | 1783 / 1840 |
| Llama-3.1-8B Q4_K_M | 28.95 (min 4413.7 ms) | 28.79 (min 4438.7 ms) | 641.7 | 635.4 |
| gemma-4-26B-A4B Q4_K_M (2 pairs) | 35.36 / 35.5 | 35.19 / 35.0 | 21.7–21.9 | 22.4–23.2 |

Gates: exact-token parity (greedy 128 @p512, hazard vs legacy dump diff) PASS on SmolLM Q8_0,
3B IQ4_XS, gemma-4-26B-A4B (MoE), re-run after every tracker change; prefill covered (any
prefill divergence would flip the greedy tokens). Vulkan unit suite 879/0 passed ×2 (hazard
active on all tracked-model forward tests). Integration Vulkan filter: only pre-existing
DeepSeekV2Lite load-OOM + the box's transient DEVICE_LOST flake (see below).

### Verdict / why no win
llama.cpp's per-op sum (2.56 ms) > wall (1.60 ms) observation does NOT translate into
recoverable overlap for dotLLM: our shared-activation-quant groups had ALREADY removed the
barriers between the only wide-independent ops (Q/K/V, gate/up — llama.cpp gets its overlap in
exactly those spots), and the rest of the decode chain (rmsnorm→quant→GEMV→rope→KV→attention→…)
is a true dependency chain where hazard tracking correctly emits ~17 barriers/layer vs legacy's
18. GPU wall is unchanged; the measured gains are host-side only (record 0.31→0.24 ms/tok since
fewer barrier commands are recorded and guards are 1 warm dict-lookup/buffer). Residual gap vs
llama.cpp SmolLM decode stays ~0.87× (540 vs 625): per the #143 attribution the remainder is
per-op GPU time + fence/submit cost, NOT barrier serialization. Value of keeping #144:
structural correctness base for future async/multi-queue/batched-overlap work + it removes the
barrier-recording overhead. Default = hazard ON; `DOTLLM_VULKAN_LEGACY_BARRIERS=1` restores.

### Box-instability note (NOT mode-correlated)
During the perf campaign the box entered a ~15-min spell of `VK_ERROR_DEVICE_LOST` at
vkQueueSubmit (SplitSubmit AND plain SubmitContext) that hit **pure-legacy runs**
(command-stream identical to dev baseline) as often as hazard runs, then cleared on its own;
AddKernel smoke passed throughout, no Event-ID-4101 TDR logged. Same family as the documented
`VK_ERROR_MEMORY_MAP_FAILED` back-to-back flake (agent-146 was reproducing map-flakes on this
box the same day). Treat one-off DEVICE_LOST during heavy back-to-back sessions as
environmental; A/B evidence requires the paired run to pass in the same window.

---

### ALSO APPLY to main `.docs/KERNEL_MAP.md`:

1. **§0b env index** — add rows:
```
| `DOTLLM_VULKAN_LEGACY_BARRIERS` | Kill-switch (#144): restore blanket barrier-per-dispatch instead of hazard-scoped barriers. `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_HAZARD_VALIDATE` | Debug (#144): tracker armed but a barrier is forced at every guard (legacy-equivalent ordering through the tracked path). |
| `DOTLLM_VULKAN_HAZARD_DEBUG` | Debug (#144): print the buffer that forced each hazard barrier. `VulkanHazardTracker.cs` |
```
2. **§12 "Remaining follow-up levers" item 1** — mark: DONE (#144) but PERF-NEUTRAL on gfx1151;
   do not re-chase decode overlap via barrier scoping (see §13). §10 "Barriers" paragraph:
   note barriers are now hazard-scoped by default on the tracked forward.
