# Usable 128k continuing sessions on Bonsai-2 — Vulkan (Strix Halo, Meteor Lake Arc) — design addendum

Date: 2026-10-22. Status: proposed. Extends `2026-10-21-bonsai-128k-session-design.md` (the CUDA design; C1–C7 there).

## Intent

The same continuing Bonsai-2 27B PQ2_0-MTP chat / agent session should grow turn by turn to 128k on the **Vulkan**
backend (`VulkanQwen3HybridDenseTransformerModel`), with each turn prefilling only what it adds. The user wants this both
as a **real usable deployment on Strix** (Ryzen AI Max+ 395, Radeon 8060S, RDNA 3.5, ~96 GB GPU-usable UMA) and because
the same Vulkan code serves the **local Meteor Lake Arc iGPU** (Core Ultra 7 155H, 96 GB shared RAM). The 2026-10-21
non-goal "Vulkan / HIP quantized-KV prefill" is lifted for Vulkan by this addendum (user decision, 2026-10-22).

Strix is the primary Vulkan validation and performance target (memory is not the constraint there, bandwidth and
kernels are); Arc is the portability check (no cooperative-matrix units, display-driving iGPU → TDR risk, see memory
note on Arc TDR).

## Measured baseline (Strix, dev `e2219f5a`, 2026-10-22)

`dotllm serve <bonsai> --device vulkan --no-mtp --ctx-size 131072`, `scripts/session-growth.py` with 8k-token turns. Logs
are in `.docs/ctx128k/strix-baseline/`. Two gpt-oss-20b servers from another session shared the GPU during the run.

| request | result |
|---|---|
| growth turn 1, 7,823 tokens | prefill 56.6 s (138 tok/s), decode 16.1 tok/s |
| growth turn 2, 17,037 tokens | **refused**: "The KV cache could not grow to the 17101 positions … it holds 8143" |
| fresh 4,006-token prompt | prefill 23.4 s (171 tok/s), decode 19.6 tok/s |
| fresh ~16k-token prompt | **`VK_ERROR_DEVICE_LOST`** in the prefill `Forward` after ~40 s |
| fresh ~33k-token prompt | same device loss |

So on Vulkan a growing session cannot pass turn 1 (#1093 turns the non-growable miss into a clean refusal), and **no prompt
much over 8k can be prefilled at all**. The Vulkan Bonsai model reports `PreferredPrefillChunkTokens = 0`, so the whole prompt
is one `Forward`. Long prefills submit one layer per command buffer, so a single submission covers every row of one layer:
the projections plus the attention over the whole history. Past ~8k rows that exceeds the Windows GPU watchdog (TDR, 2 s).

## What reading the code shows (to be confirmed by the baseline)

1. **The Vulkan KV can never grow for a prefix hit.** `VulkanNemotronHKvCache` keeps the default
   `IKvCache.TryGrowTo` (`maxLength <= MaxLength`), and the Bonsai factory (`CreateKvCache(size)`) builds an eager cache of
   exactly the size the engine asked for: `needed + 256` on a miss. In a growing session nearly every turn needs more
   than the previous allocation, so `EnsureRecurrentPrefixKvCapacity` fails and (after #1093) either misses and
   re-prefills the whole history (< 4096 tokens) or refuses with `context_length_exceeded` (≥ 4096). #1093's goals G1/G2
   therefore do not hold on Vulkan at all.
2. **The lazy, layer-by-layer growth already exists.** `VulkanNemotronHKvCache(lazy: true)` + `EnsureCapacity` (25 %
   steps, one layer's K/V transient) is used by Qwen4-Exp (`VulkanQwen4ExpTransformerModel.EnsureContext`), which also
   shows the descriptor-cache invalidation the Bonsai forward would need.
3. **KV is F32: 128 KiB/token** for Bonsai (16 attention layers × 4 KV heads × 256 × 2 × 4 B) — 16 GiB at 128k.
   Capacity is fine on both machines; **decode bandwidth is not**: every decoded token reads the whole history. At 128k,
   F32 KV alone is ~16 GiB/token of reads → on the order of 60–80 ms/token on Strix and well over 100 ms/token on
   Meteor Lake, before any weights. F16 halves that; q8_0 roughly quarters it. `VulkanKvStorage.F16` exists but only the
   Qwen4-Exp QSA gather reads it today.
4. **Prefill attention at hd256** (#441): the F32 flash kernel caps at headDim 128; a cooperative-matrix hd256 pipeline
   exists (`VulkanFlashAttentionCoopmatKernel.WideMaxHeadDim`). Strix (coopmat) may take it; Arc (no coopmat) falls to the
   naive kernel, which is O(seqQ × seqKv) per head with no tiling.

## Goals

- V-G1. On Vulkan, a growing session's every turn is a prefix hit that prefills only the new turn (same acceptance as
  #1093: cached ≥ previous prompt − 64), all the way to the configured window.
- V-G2. A growth that cannot be satisfied is a clean `context_length_exceeded`, never a silent full re-prefill and never a
  device-lost.
- V-G3. Long-context decode on Strix is bandwidth-reasonable: KV narrower than F32 on the Bonsai path, read directly by the
  decode kernel (no widen-to-F32 copy of the history — the user's no-expansion rule applies to KV on every backend).
- V-G4. Prefill attention at hd256 is a tiled (flash) kernel on both Strix and Arc, with a one-time diagnostic when a
  long-context prefill falls back to the naive kernel.
- V-G5. Step tables (TTFT, cached, prefill/decode tok/s, memory) to 128k on Strix and to the largest practical window on
  Arc, recorded in `docs/SERVER.md` beside the CUDA tables.

## Non-goals

- Raw prefill GEMM throughput (separate work; coopmat GEMM tiles are already tracked).
- HIP/ROCm.
- The window-hint server work (C4 in the CUDA design) beyond making its ladder accept a Vulkan model as "explicit
  `--ctx-size` equivalent" — already the CUDA design's stated behaviour for models without a fit record.
- MTP on Vulkan at long context (measure only; tune later).

## Anti-goals

- Any full-history widen of quantized/narrow KV into F32 scratch, on any Vulkan path the Bonsai shape reaches.
- Re-creating KV buffers without invalidating every descriptor cache that names them (silent zeros — documented hazard).
- A growth that takes the Arc display GPU past the 2 s TDR window in one submission (split long copies / dispatches).

## Changes (proposed; my calls unless marked)

### V1. Growable Vulkan KV for the Bonsai path (engine-visible; do first)

- `VulkanNemotronHKvCache.TryGrowTo(n)`: for a lazy cache, raise the logical limit (no allocation; buffers grow in the
  forward via `EnsureCapacity`). For an eager cache, grow by copying, layer by layer (existing `GrowTo`), and raise the limit.
  Cap at the model's window.
- `VulkanQwen3HybridDenseTransformerModel.CreateKvCache` creates a **lazy** cache; its forward calls `EnsureCapacity`
  for `positions[^1] + 1` before recording and invalidates its descriptor caches when buffers were re-created (pattern:
  Qwen4-Exp `EnsureContext`). Out-of-device-memory during growth maps to `DeviceMemoryAdmissionException` (→ HTTP 400).
- Snapshots (`SnapshotSequencePrefix` / restore) and `Rollback` keep working across a growth; covered by tests.
- Shared: same applies to `VulkanQwen3MoeHybrid` / LFM / Nemotron-H Vulkan hosts that build the same cache type — check
  each and fix where the same factory pattern exists (cross-backend rule for correctness bugs; this is a functional bug).

### V1b. Bounded prefill tiles on Vulkan (device-loss fix; with V1)

- The Vulkan hybrid hosts report a `PreferredPrefillChunkTokens` (my call: 1024, overridable by an environment variable
  like CUDA's `DOTLLM_CUDA_PREFILL_CHUNK`; measure 512 and 1024 on Strix and Arc), so `TextGenerator` tiles every prefill.
- A tile's attention over a deep history must also fit the watchdog. Until V3 lands, measure the per-layer submission time at
  128k depth on Strix. If it is too close to 2 s, split the attention dispatch along the KV axis (or submit fewer rows per
  tile at depth).
- Acceptance: fresh 16k, 33k, 66k and 128k prompts prefill without a device loss on Strix (and up to the practical window
  on Arc), with prefill tok/s recorded per depth.

### V2. Narrow KV on the Bonsai Vulkan path, read directly by attention

- Builder's choice between F16 (existing storage enum; extend split-KV decode, flash prefill and the snapshot copy to read
  it) and q8_0 (new storage; dequant in the kernels' tile loads, as CUDA #956/#1094 do). Recommended order: F16 first
  (smallest kernel change, halves bandwidth), q8_0 second if Strix decode at 128k is still KV-bound. Measure each.
- No kernel may convert the whole history to F32 scratch.

### V3. hd256 tiled prefill attention on every Vulkan device (#441)

- Strix: confirm the coopmat hd256 pipeline engages for Bonsai prefill (diagnostic + test).
- Non-coopmat devices (Arc): an F32/F16 scalar flash variant for headDim 256 within 32 KiB shared memory (Strix limit;
  Arc's is larger), tiled so one dispatch stays well inside the TDR window.
- One-time warning when a prefill with seqKv above a threshold takes the naive kernel.

### V5. YaRN beyond the native 262,144 window (user request, 2026-10-22)

The user wants to know how far context can be pushed, especially on Strix. Qwen3.8 supports 262,144 tokens natively and
up to ~1M with YaRN (factor 4, `original_max_position_embeddings` 262,144). Bonsai-2 is built from Qwen3.8 and keeps its
attention and RoPE unchanged: GGUF `qwen35.context_length = 262144`, `rope.freq_base = 1e7`, partial RoPE 64 of 256 head
dims, and no rope-scaling keys, the same as Qwen3.8's opt-in YaRN. The RoPE math does not touch the ternary weights, so
**YaRN applies to Bonsai-2 directly** and the work stays on Bonsai-2 (user decision: only fall back to Qwen3.8 if Bonsai
needs model-specific tweaking).

- Today the hybrid-dense hosts (CPU reference, CUDA, Vulkan) read only `RoPEConfig.Theta` / `DimensionCount`. A
  `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` override is plumbed into `RoPEConfig` but silently ignored by
  these hosts, and the window stays capped at 262,144.
- Wire the existing YaRN pieces into all three hosts: the CPU `RoPE.PrecomputeFrequencyTableYarn`, CUDA's inverse-frequency
  table upload, and Vulkan's `rope_invfreq_f32` / `rope_yarn_f32`. The ramp is over the 64 rotary dims, with NeoX pairing.
  Raise `MaxSequenceLength` to `factor × original`.
- Attention-scale convention: HF Qwen YaRN uses `attention_factor = 0.1·ln(factor) + 1` applied to cos/sin (the llama.cpp
  `ggml_rope_ext` convention that `RoPEConfig.ComputeYarnMscaleMultiplier` today gives only gpt-oss). Qwen3.5/3.8 must use it
  too. **Verify against HF `_compute_yarn_parameters` numbers before any backend work.**
- YaRN changes short-context numerics too (mscale at every position), which is why Qwen recommends enabling it only when
  needed. The window hint (CUDA design C4) should therefore turn YaRN on automatically only for a hint above the native
  window, e.g. `bonsai[512k]`.
- Memory on Strix (~96 GB usable): F32 KV is 128 KiB/token, so 512k = 64 GiB and 1M = 128 GiB (does not fit). F16 halves
  that (1M = 64 GiB, fits); q8_0 roughly 34 GiB at 1M. **V2 is a prerequisite for 1M on Strix.** Prefill time at those
  depths is the next limit: at ~100–140 tok/s a fresh 1M prompt is hours, so the realistic use is a session grown turn by turn
  through prefix reuse.
- Acceptance:
  - YaRN tables match an independent reference (HF formula) for Bonsai's config.
  - A fixed short prompt's greedy output is unchanged with YaRN off.
  - A needle-in-a-haystack retrieval past 262,144 tokens on Strix succeeds with YaRN on.
  - Record the furthest working window and the prefill/decode tok/s at each step (256k, 384k, 512k, then 1M once V2 lands).

### V4. Measure and document

- Baseline (above), then after V1, V2, V3 on Strix; a shorter run on Arc (display-driving iGPU — use the TDR guidance).
- `docs/SERVER.md` Vulkan rows; `docs/KV_CACHE.md` Vulkan growth and storage; `docs/ATTENTION.md` hd256.

## Order and issues

V1 + V1b (functional; unblock every measurement; #1100) → V3 (tiled hd256 prefill attention, #441) → V5 (YaRN) → V2 (narrow
KV; needed for decode speed at depth and for 1M on Strix). Each gets its own issue/PR from `dev`, like the CUDA PRs. Vulkan
PRs can proceed in parallel with CUDA PRs C/D/E because they touch different files (the shared engine code already merged in
#1097).

## Next target after Bonsai-2: Qwen3.8-Flash-Next (user, 2026-10-22)

The user flagged https://qwen.ai/blog?id=qwen3.8-flash-next as the key target beyond these, because it previews the
architecture Qwen4 is expected to use. It is `qwen4_exp` (HF `Qwen/Qwen3.8-Flash-Next`):

- 125B total / 6B active, plus 51B n-gram embedding and a 4B MTP head.
- 48 layers: 12 × (3 Gated DeltaNet + 1 Qwen Sparse Attention), each followed by MoE (512 experts, 10 routed + 1 shared).
- QSA: 24 Q / 2 KV heads, head dim 256, rotary 64, indexer budget 2048 tokens.
- 262,144 native context, extensible to 1M.

dotLLM's qwen4_exp work is tracked in epic #814 (CPU oracle, Vulkan bring-up, n-gram offload, expert tiering, Strata
benchmark). The long-context pieces here — growable lazy KV, bounded prefill tiles, YaRN, narrow KV — are kept generic so
they carry over to the qwen4_exp host. That host already uses the lazy Vulkan KV, and with only 2 KV heads × 12 layers its
per-token KV is small enough that 1M on Strix is a realistic goal.
