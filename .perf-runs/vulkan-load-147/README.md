# Issue #147 — Vulkan load-path host-allocation elimination (ledger)

Machine: Strix Halo (gfx1151, UMA, **32 GiB** RAM), Windows, AMD proprietary driver.
Harness: `profile-vulkan-load` (commit `10c5715b`) — one load per process; peak commit =
`Process.PeakPagedMemorySize64`. GPU runs under `scripts/gpu-lock.sh` as `agent-147-loadpath`.

## Before / after (dev baseline `c443891f` vs this branch)

| Model | Load ms (warm) before → after | Peak commit MiB before → after | Logits SHA-256 (first forward) |
|---|---|---|---|
| Llama-3.2-3B IQ4_XS (Q6_K embed) | 1657 → **1096** | 4861 → **3445** (−1.4 GiB) | `0230545FB73C85AA` = identical |
| Llama-3.1-8B Q4_K_M (Q4_K embed) | 2969 → **2285** | 8821 → **6900** (−1.9 GiB) | `37997DF35ABEDF05` = identical |
| gemma-4-26B-A4B Q4_K_M (Q6_K embed) | 59.7 s cold / 119.9 s pressured → **43.5–46.9 s** | 24484 → **21773** (−2.7 GiB) | `98946D9E0CEB4F70` = identical |

Notes on the numbers:
- 26B wall time is noise-dominated on this 32 GiB box (16 GiB model + ~21 GiB commit ⇒ chronic
  paging; baseline itself swung 59.7→119.9 s run-to-run). Peak commit is the stable metric.
- Peak-commit deltas ≈ the old staging allocation (vocab×hidden×4: 1.5 / 2.0 / 2.75 GiB).
- Staging-cap sweep on the 8B (warm, `DOTLLM_VULKAN_STAGING_MB` = 32/64/128/256):
  load 2283 / 2334 / 2695 / 3514 ms, peak commit 6871 / 6901 / 6965 / 7071 MiB — 32 and 64 are
  equivalent within noise, larger caps mildly WORSE on both metrics (bigger resident map, longer
  per-chunk fence waits). 64 MiB kept as the default (headroom for wide rows / expert slabs).

## Stress gate (the #146 trigger)

`DOTLLM_VULKAN_STRESS_LOAD_CYCLES=20` × 2 runs with the 3B IQ4_XS, under a detached 8 GiB
touched-commit hog on the 32 GiB box: **40/40 cycles green, zero `[vulkan-mem]` transient
retries** (baseline flaked ~2/15 with retry-rescue). The #146 retry stays as a safety net.

## What was eliminated

1. **Giant staging buffer + per-upload remap** (commit `be991b41`): staging is now a single
   `VulkanStagingBuffer` capped at `DOTLLM_VULKAN_STAGING_MB` (default 64 MiB), persistently
   mapped once; large tensors stream in chunks; a dedicated 256 KiB vec-staging buffer carries
   norm/bias/scale vectors. Converted: `VulkanWeights` (dense/MLA/MoE/Gemma4),
   `VulkanNemotronHWeights`, `VulkanQwen3MoeHybridWeights`, `VulkanMamba3Weights`,
   `VulkanQwen3MoeMoeUpload` (its whole-bank staging write — ~8 GiB at qwen35moe-A3B fp32
   scale — became per-expert / contiguous streaming).
2. **CPU F32 dequant of the token-embed table** (commit `f5e8ddb7`): Q4_K/Q6_K embeds (all
   three gate models) now upload raw quantized bytes and dequant ON DEVICE via new
   `q4_k/q6_k_dequant_f32` shaders — bit-identical to the CPU oracle (`precise` math, same op
   order; 0-ULP kernel tests incl. >32768-block chunked dispatch; discriminating end-to-end
   table comparison vs `DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT=1`).
3. **Per-expert slot copies for contiguous raw-quant MoE banks** (commit for item 3): whole-bank
   zero-copy import attempt, falling back to ONE streamed copy.

## Zero-copy import audit (item 3)

**Driver regression found:** on this box's CURRENT AMD driver, EVERY mmap import fails at
`vkAllocateMemory` with `VK_ERROR_INVALID_EXTERNAL_HANDLE` (−1000072003) for BOTH handle types
(`HOST_MAPPED_FOREIGN_MEMORY` and `HOST_ALLOCATION`) — `HostVisibleBuffer`'s docs (and
`docs/GPU.md`) record that HOST_MAPPED_FOREIGN previously worked here. All import paths
therefore fall back to bounded staging today; the import code (incl. the new bank/embed
imports) is exercised in fallback mode only on this box. Re-validate after a driver update or
on Linux radv.

Per-tensor-class audit (UMA, assuming the import works):

| Tensor class | Import status | Reason |
|---|---|---|
| Raw-quant dense matrices | implemented (pre-existing) | contiguous mmap range |
| Token-embed quantized source (Q4_K/Q6_K) | **implemented (#147)** | contiguous; feeds device dequant |
| Routed MoE raw-quant banks (Q8_0/F16 kept-native) | **implemented (#147)** | GGUF fused-expert tensor is expert-contiguous with the bank's exact stride |
| Qwen3-MoE Q6_K resident banks (`VulkanQwen3MoeMoeUpload`) | follow-up | same contiguity argument; per-forward streamed path, wants separate validation |
| Gemma-4 fused gate_up banks (Q4_K) | not importable | gate/up rows interleave per expert — packed W1/W3 layout ≠ source; would need stride-aware indexed-matmul shaders |
| Gemma-4 down banks (Q5_0→Q5_1 repack, Q8_0 scale-fold) | not importable | bytes transformed en route (Q5_1-verbatim sub-case is importable but rare in the wild; follow-up) |
| F32-dequant fallback matrices | not importable | transformed host-side |
| Norm/bias/scale vectors | not importable, justified | managed `float[]` (GC-movable), KB-scale |
| MLA F32 projections | unsafe today | loader-owned upcast buffers with early-free semantics (`TryReleaseOwnedHostAllocation`); import would require lifetime changes |
| LoRA adapter tensors | justified as-is | per-tensor one-shot staging, MB-scale, one-time per adapter |

Safety notes for import-in-place: alignment handled by `HostVisibleBuffer` (page-round +
bindOffset); lifetime contract = the `GgufFile` mmap must outlive the model (already the
documented contract for dense weights); page-cache: imported pages are the OS file cache —
first-touch faults stream from disk during the first forward instead of at load.

## Host-materialisation ledger (item 4 sweep)

| Site | Size | Verdict |
|---|---|---|
| Weight staging buffer (was vocab×hidden×4) | 1.5–2.9 GiB → ≤64 MiB + 256 KiB | **fixed** (item 1) |
| Token-embed CPU F32 dequant (Q4_K/Q6_K) | vocab×hidden×4 through staging | **fixed** (item 2, GPU dequant) |
| Token-embed CPU dequant, other types (Q8_0/F16/F32; NemotronH/Qwen3-hybrid/Mamba3 embeds) | streamed, host-bounded | follow-up: reuse `UploadTokenEmbedding` + add q8_0/f16 dequant shaders; tables ≤8× smaller than the K-quant gate models' |
| `VulkanQwen3MoeMoeUpload` whole-bank staging write | ~8 GiB at A3B fp32 scale | **fixed** (item 1) |
| DeepSeek-MoE loader F32 host dequant of all experts (`TransformerWeights.LoadFromGguf` without `skipF32MoeDequant`) | ~2.2 GiB/layer (V2-Lite Q4_K_M) | follow-up: `VulkanTransformerModel.LoadFromGguf` does NOT pass `skipF32MoeDequant: true` (CUDA does). Passing it needs an audit that every Vulkan MoE fallback consumes raw views — multi-GiB win for DeepSeek-family Vulkan loads |
| MLA loader F32 upcasts (F16/BF16→F32, loader-owned) | per-projection | justified today (CPU MLA oracle is F32-only); revisit with native F16 MLA kernels |
| Gemma-4 F32 expert-dequant fallback (`UploadGemma4MoeLayer`) | streamed, host-bounded | justified — synthetic-fixture/correctness path; quantized path is the production default |
| I2_S force-dequant full-size one-off staging (`UploadMatrix`) | tensor-sized, rare | justified — single trailing per-tensor scale prevents block-aligned chunking (BitNet lm_head force-dequant only) |
| Norm/bias `float[]` in loader | KB-scale | justified |
| Q5_0→Q5_1 / Q8_0-scale-fold repacks (Gemma-4 down) | streamed through bounded staging | justified — bit-exact transform must happen somewhere; now chunked |

## Pre-existing baseline failure (NOT from this branch)

`VulkanPipelineParityTests.PipelinedForwardBatch_MatchesPerSequenceForward` fails on the dev
baseline (`c443891f`, verified by stashing all #147 changes and rebuilding):
`[single-device] seq 1 col 0: serial=0.045437 vs batched=0.004787` (seq 0 matches 0.000E+000).
Batched-decode cross-talk shape, unrelated to the load path — and FLAKY: it failed on the
stashed dev baseline and on the item-1 full-suite run, then passed on the final-state re-run.

Suite results on the final branch state (split into <10-min chunks): 218 + 142 + 224 + 294 =
878 passed, 0 failed, 41 skipped across the full `~Vulkan` population. One earlier
single-process 13-minute run of the A–Ma chunk reported "Failed: 3" with the failing names lost
to output truncation; the identical population re-run green in splits immediately after —
consistent with cumulative-pressure flakiness on this 32 GiB box (2.3 GiB free RAM during the
campaign), not with a deterministic regression.

## KERNEL_MAP.md updates to fold in (worktree cannot edit the main repo's `.docs/`)

- §1 embedding: token-embed table is now GPU-dequanted at load for Q4_K/Q6_K (bit-exact,
  `q4_k/q6_k_dequant_f32.comp`); host F32 image eliminated; gather unchanged.
- Env-var section: add `DOTLLM_VULKAN_STAGING_MB`, `DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT`.
- §12 / load notes: staging bounded + persistently mapped (#147); #146 retry is now a safety
  net; "Recurring transient VK_ERROR_MEMORY_MAP_FAILED at load" note can be retired.
- Zero-copy import: current driver rejects ALL mmap imports (`VK_ERROR_INVALID_EXTERNAL_HANDLE`
  at vkAllocateMemory, both handle types) — regression vs the documented earlier behaviour.

## Addendum: RTX 3060 dGPU verification (2026-07-27)

Re-verified this already-merged branch state (`origin/dev` @ `ab853eb`, #147 content landed via
`be991b4`/`f5e8ddb`/`7e65e4a`/`4ae4db3`/`e506d45`/`46d31dd`, already an ancestor of `origin/dev` —
no PR was ever opened for it, which is presumably why #147 was still open with 0 comments) on a
**discrete** GPU (RTX 3060 12 GiB, NVIDIA proprietary driver, PCIe, non-UMA) to see how the
findings generalize off the original Strix Halo iGPU box. GPU confirmed idle before each run
(`nvidia-smi`, <=557 MiB / 12288 MiB, 0-5% util).

**Item 1+2 before/after, apples-to-apples (both warm OS page cache), Qwen3-8B Q4_K_M
(`unsloth/Qwen3-8B-GGUF`, real weights, Q4_K token-embed):**

| Config | Load ms | Peak commit MiB | Logits SHA-256 |
|---|---|---|---|
| Simulated pre-#147 (`DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT=1` + `DOTLLM_VULKAN_STAGING_MB=3000`, i.e. CPU embed dequant + one giant staging buffer sized for the largest upload) | 15492 | 9444.3 | `A3A077163737523D` |
| Current default (64 MiB staging cap, GPU Q4_K embed dequant) | 6292 | 7233.9 | `A3A077163737523D` (identical) |

Load time **-59%** (2.46x), peak commit **-2210 MiB** (-23.4%) — same direction and similar
magnitude to the Strix Halo numbers above, on completely different hardware/driver. Logits
bit-identical between configs, confirming item 2's GPU dequant is a true no-op on output.
`VulkanEmbedGpuDequantParityTests.EmbedTable_GpuDequant_BitIdenticalToCpuPath` (previously SKIP
on this box — its default fixture path is Strix-only) passes green when pointed at this model via
`DOTLLM_VULKAN_EMBED_PARITY_MODEL`.

Cold-cache (first read off disk, OS cache empty) load for the same model: 54.6 s — dominated by
disk I/O for the 4.8 GiB file, not by the load-path allocator; not used for the before/after
comparison above for that reason.

SmolLM-135M Q4_K_M (`token_embd.weight` is Q8_0 in this GGUF, confirmed by direct tensor-info
parse, dims `[576, 49152]`, ggml_type 8): `embedDequant='cpu'` as expected — Q8_0/F16/F32 embed
tables correctly fall through to the still-CPU streamed path per the existing scope (item 4 row
"Token-embed CPU dequant, other types").

**Item 3 (zero-copy import) on this dGPU:** `device.HasExternalMemoryHost` reports true and the
handle-type query succeeds, but every import **still falls back to staging** —
`lastFallbackReason='import_rejected'`, `import-reject: stage='vkAllocateMemory' vkResult=-2`
(`VK_ERROR_OUT_OF_DEVICE_MEMORY`) on both the 100 MiB SmolLM and 4.8 GiB Qwen3-8B models (i.e. not
an actual memory-pressure condition — 12 GiB free). This is a **different rejection point and
code** than the Strix Halo amdvlk finding (`VK_ERROR_INVALID_EXTERNAL_HANDLE` at the same call) —
plausibly the NVIDIA driver accepting the host-pointer query but refusing to back a
`DEVICE_LOCAL`-usable allocation with arbitrary (non-pinned) `MemoryMappedFile` pages over PCIe,
which is the expected/principled outcome for non-UMA hardware (there is no page a dGPU can access
without *some* PCIe hop; "zero-copy" only has a real payoff when the pages are already GPU-local
DRAM, i.e. UMA). Net: on both hardware classes verified so far (AMD iGPU/UMA and NVIDIA dGPU), the
whole-import path is real, tested code but dormant in production — the bounded-staging fallback
(item 1) is what's actually carrying the load-path today everywhere. **Could not test the
UMA-specific zero-copy win itself** — this machine has no iGPU; that requires Strix Halo or
similar (see docs/GPU.md Future Work: Mesa radv validation is the next real chance to see it fire).

**Item 4 follow-up re-audited — the `skipF32MoeDequant` row above is NOT a safe drop-in as
worded.** Traced `VulkanWeights.UploadMoeLayer` -> `MoeRoutedRawDeviceQuantType`
(`src/DotLLM.Vulkan/VulkanWeights.cs`): for the routed-expert banks (`W1`/`W2`/`W3`) this only
keeps the raw quantized view for **Q8_0** (`MoeRoutedRawKeepsQ8`) or **F16 with cooperative-matrix
support**; every other quant type — including **Q4_K/Q5_K/Q6_K**, the common case for
`*_K_M`-quantized DeepSeek-family GGUFs, e.g. the V2-Lite Q4_K_M example this very ledger uses —
falls through to `QuantizationType.F32` and reads `moe.W1/W2/W3` (the F32 host-dequant arrays).
`LoadDeepSeekMoeLayer`'s `skipRoutedDequant` path leaves those arrays as `new nint[numExperts]`
(all-null pointers) when `skipF32MoeDequant: true`. So passing that flag to
`VulkanTransformerModel.LoadFromGguf` today, unconditionally, would silently corrupt DeepSeek-MoE
Vulkan inference for any K-quant routed-expert bank — exactly the "silently wrong, not a crash"
failure mode this task explicitly warns against. NOT shipping this.

The K-quant indexed-matmul kernels this would need already exist and are already wired up — just
for the *other* MoE loader path: `VulkanQwen3MoeHybridKernels`/`VulkanQwen3MoeHybridTransformerModel`
has `MoeIndexedMatmulQ4_KF32Kernel`/`Q5_K`/`Q6_K` (plus MMQ variants) with a `useRawQuantView`
dispatch. The generic/DeepSeek path (`VulkanWeights`/`VulkanTransformerModel`) has no equivalent
dispatch on the routed-bank quant type — it always issues the F32 indexed matmul. Making
`skipF32MoeDequant` safe for Vulkan therefore means porting that dispatch (recognize Q4_K/Q5_K/Q6_K
routed banks when the contraction axis is a multiple of 256, upload raw, and record the matching
indexed kernel instead of `MoeIndexedMatmulF32Kernel`) to the generic path first — real, scoped,
but a distinct chunk of work with its own correctness-parity test needs, not a one-line flag flip.
Filed as a new, separate issue rather than attempted here under time/risk constraints.
