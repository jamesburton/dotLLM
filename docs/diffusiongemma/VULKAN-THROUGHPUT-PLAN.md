# Vulkan Throughput Plan — Strix Halo (Radeon 8060S, gfx1151)

**Status:** Planning doc (not an implementation). Captures the in-flight decode-GEMV
throughput campaign (issues #45–#54), the current code state on `dev`, and the ranked
remaining Vulkan kernel targets. The single recommended next work item is in §5.

> **STATUS UPDATE (2026-06-21):** The kernel-side campaign is now **banked on `dev`**.
> Verified by commit ancestry against `origin/dev` @ `dd53670`:
> - **#52 Q4_K dp4a MMVQ decode — MERGED** (`92a84c8`; `matmul_q4_k_mmvq.comp` present). The
>   old §5 "recommended next item" is therefore **DONE**. Validated +56 % min-latency (memory
>   `vulkan-decode-bench-uma-contention`).
> - **#54 wave32 subgroup control — MERGED** (`30ab725`; +33 % min).
> - **#45 / #46 / #50 — MERGED** (vectorised GEMV, Q8_0 dp4a MMVQ decode + prefill).
> - **#53 sub-block thread mapping — REJECTED** on gfx1151 (no win); correctly not on `dev`.
> - **#49 Q8_0 MMVQ occupancy remap — REJECTED (2026-06-23).** Replanted onto current `dev`
>   (`8c363f1`) and A/B'd clean (two worktrees off the identical base, sandwiched alternating order,
>   3B Q8_0). **Regresses decode ~8–10%:** base `decode_min_ms` 26.27/26.25/26.59 (avg ~28.06,
>   35.6 tok/s) vs #49 28.19/29.15 (avg ~30.9, 32.4 tok/s) — both #49 runs outside the stable
>   baseline bracket. Kernel itself is correct (7/7 F32-oracle + 3/3 shared-quant tests). The
>   subgroup-per-row remap (32 lanes/row on wave32) loses to the 128-thread/row mapping on this
>   bandwidth-bound GEMV. Confirms the #53-family skepticism. **NOT on `dev`, do not merge.**
>
> **The genuine next lever is the rank-2 forward-graph fusion (§6, NEW).** The Q4_K_M decode
> gap §5 targets is already closed by #52.

**Box:** AMD Strix Halo, Radeon 8060S iGPU (gfx1151, RDNA 3.5), 128 GB UMA (96 GB VRAM
split). Vulkan is the GPU path (ROCm iGPU kernel-load broken, no NVIDIA). Device advertises
`fp16:1`, **`bf16:0`**, `VK_KHR_cooperative_matrix`, and `VK_KHR_shader_integer_dot_product`
(dp4a). See `STRIX_HALO.md`.

---

## 1. Current measured state (the gap)

Decode is **memory-bandwidth-bound** — one full sweep of the quantized weights per token —
so the lever is bytes-per-token efficiency of the quantized weight-GEMV (`mul_mat_vec`).

**Decode tok/s, dotLLM Vulkan vs llama.cpp Vulkan, same box (`STRIX_HALO.md` Benchmarks):**

| Model | dotLLM | llama.cpp | ratio |
|---|--:|--:|--:|
| SmolLM-135M Q8_0 | 357 | 487 | 0.73× |
| Llama-3.2-1B Q8_0 | 72 | 130 | 0.55× |
| Llama-3.2-3B Q8_0 | 28.6 | 54.7 | 0.52× |
| Llama-3.1-8B Q4_K_M | 9.91 | 42.96 | **0.23×** (older measurement, pre-campaign) |

The ~0.52–0.55× gap **does not amortise** (it is the GEMV kernel, not a fixed submit cost —
the ~2 ms/token submit tax only dominates the 135M toy). Root cause: dotLLM's quantized
weight-GEMV reached **~34 % of the ~256 GB/s** memory bandwidth vs llama.cpp's **~61 %** →
**~1.8× shared headroom across all models**. Profile any model with:

```
DOTLLM_VULKAN_PERF=1 DOTLLM_VULKAN_PERF_MODEL=<path-to.gguf>
```
(harness: `tests/DotLLM.Tests.Integration/Vulkan/VulkanGemma4DecodePerfHarness.cs`,
`VulkanForwardPerfHarness.cs`).

### Important nuance from the diagnostics (memory note `vulkan-decode-barrier-bottleneck`)
A later by-elimination diagnosis (Llama-8B Q4_K_M) showed the **unchanged** Q4_K MMVQ kernel
streams at **270–423 GB/s when dispatches OVERLAP**, but collapses to **22–204 GB/s when each
dispatch runs alone** (submit+fence+global barrier), worst for small matmuls (k/v proj M=1024
→ 22 GB/s). The real forward is ~225 dependent, individually-barriered GEMVs/token, so each
runs in isolation and can't keep enough memory requests in flight. Ruled out (each <5 %):
thread count (#49 +4.6 %), wave32 (#54 +0.9 %), barriers-as-noops (+1.7 %), CPU record/submit.

**Two complementary levers therefore exist:**
- **(A) Kernel byte-efficiency** — finish the dp4a/integer-dot MMVQ path for the *remaining*
  quant types so each GEMV reads fewer bytes / does less ALU per byte (issues #52/#53/#54).
- **(B) Execution-model overlap/fusion** — issue independent GEMVs (q/k/v, gate/up) without
  intervening global barriers, and **fuse** small projections (qkv → one matmul, gate+up → one)
  so latency-bound small GEMVs become bandwidth-bound large ones. This is *not* an issue branch
  yet and is the larger structural win once (A) lands. NO_BARRIERS alone gave only +1.7 % because
  the dependency chain still serialises — fusion/concurrency is required, not just barrier removal.

---

## 2. Issue campaign status (#45–#54)

Merged-into-`dev` determined by commit-SHA ancestry (`git merge-base --is-ancestor <sha> dev`),
not by `(#N)` grep (several N collide with unrelated historical PRs).

| # | Branch | Status | One-line | Key files |
|---|--------|--------|----------|-----------|
| 45 | issue/45-vectorize-quant-gemv | **Merged (dev)** | Vectorise Q8_0 + K-quant decode-GEMV weight loads (vec4 nibble unpack, read-once slabs) | `native/vulkan/shaders/matmul_q4_k_gemv_f32.comp`, `matmul_q8_0.comp`; perf harness env override |
| 46 | issue/46-dp4a-mmvq-decode | **Merged (dev)** | Q8_1 activation quant + dp4a Q8_0 MMVQ GEMV; probe/enable `VK_KHR_shader_integer_dot_product` | `quantize_q8_1.comp`, `matmul_q8_0_mmvq.comp`; `QuantizeQ8_1Kernel`, `MatMulQ8_0MmvqKernel` |
| 47 | issue/47-cpu-vnni-gemm | **Merged (dev)** | CPU AvxVnni (`vpdpbusd`) fused int8 Q8_0 matmul (not Vulkan) | `src/DotLLM.Cpu/Kernels/MatMul.cs` |
| 48 | issue/48-iq-gemv-vectorize | **Merged (dev)** | Vectorise IQ1_S / IQ2_S / IQ3_S decode-GEMV qs/sign/scale loads | `matmul_iq{1_s,2_s,3_s}_f32_gemv.comp` |
| 49 | issue/49-decode-thread-mapping | **REJECTED** (not in dev; 2026-06-23 clean A/B) | Subgroup-per-row MMVQ decode GEMV (occupancy remap) — **regresses ~8–10%** on gfx1151 3B Q8_0 (the old "+4.6%" was a contended/invalid run). Correct but slower; do not merge. | `matmul_q8_0_mmvq.comp`, `MatMulQ8_0MmvqKernel.cs`, `VulkanTransformerModel.cs` |
| 50 | issue/50-prefill-mmq | **Merged (dev)** | Route Q8_0 prefill through dp4a MMQ (Q8_1 row quant) with FP-GEMM fallback | `quantize_q8_1_rows.comp`, `matmul_q8_0_mmq.comp`; `QuantizeQ8_1RowsKernel`, `MatMulQ8_0MmqKernel` |
| 51 | issue/51-net11-vnni512-bf16 | **Open** (not in dev) | CPU AVX512-BF16 (`VDPBF16PS`) dequant-accumulate Q8_0 outer-product (.NET 11 preview track) | `src/DotLLM.Cpu/Kernels/MatMul.cs`, `global.json`, `Directory.Build.props` |
| 52 | issue/52-kquant-mmvq | **Open** (not in dev) | **Q4_K MMVQ dp4a decode shader** (integer-dot K-quant GEMV) + kernel wiring + parity test | `matmul_q4_k_mmvq.comp` (new), `MatMulQ4KMmvqKernel.cs` (new), `VulkanTransformerModel.cs` |
| 53 | issue/53-decode-gemv-occupancy | **Open** (not in dev, builds on #52) | Sub-block thread mapping for Q4_K MMVQ decode GEMV (occupancy) | `matmul_q4_k_mmvq.comp` |
| 54 | issue/54-wave32-subgroup-control | **Open** (not in dev, builds on #52) | Targeted per-kernel wave32 (`VK_EXT_subgroup_size_control`) for decode MMVQ pipelines | `Wave32SubgroupControl.cs` (new), `VulkanStructs.cs`, `VulkanDevice.cs`, MMVQ kernels |

Branch lineage: #52 → #53 → #54 are a stacked chain (each branch contains the previous). #49
is an independent Q8_0-MMVQ occupancy tweak.

### What is already DONE (do not redo)
- **Q8_0 decode** runs through dp4a MMVQ on `dev` (#46). Gated on `device.HasIntegerDotProduct`
  and SPV presence; falls back to FP GEMV. Q8_1 activation quant on device.
- **Q8_0 prefill** runs through dp4a MMQ on `dev` (#50) with FP-GEMM fallback.
- **K-quant + IQ decode FP-GEMV weight loads are already vectorised** (#45, #48) — vec4
  nibble unpack, read-each-slab-once (~8× fewer qs loads). This is the *FP-dequant* GEMV; the
  *integer-dot* Q4_K MMVQ (#52) is a separate, faster path and is NOT yet on dev.
- **Coopmat GEMM** exists for prefill: `matmul_f16_gemm_coopmat.comp`,
  `matmul_q8_0_gemm_coopmat.comp`, `moe_grouped_matmul_f16_coopmat.comp`.
- **Gemma-4 MoE quant kernels exist** (`moe_indexed_matmul_q4_k_f32.comp`,
  `..._q5_1_f32.comp`, `..._q6_k_f32.comp`, `..._q8_0_f32.comp`). NOTE: `GEMMA4-GPU-GAPS.md`
  §1 row 15 / §2 says Q4_K + Q5_1 indexed-MoE kernels are missing — **that is now stale**;
  they have since landed. Update that doc when convenient.

### Current decode dispatch reality on `dev` (`VulkanTransformerModel.cs`)
- `weightQt == Q8_0`, seqLen==1, MMVQ live → `MatMulQ8_0MmvqKernel` (dp4a). ✅ fast path.
- `weightQt == Q4_K`, seqLen==1 → `_matmulQ4K.Record` = **`matmul_q4_k_gemv_f32.comp`**
  (FP dequant GEMV, vectorised by #45). ❌ **No dp4a here** — #52 would replace this.
- Q5_K / Q6_K / IQ* decode → their respective FP-dequant GEMVs (vectorised, no dp4a).

**This is the headline gap:** the most common real quant for the target models is
**Q4_K_M**, and on `dev` its decode GEMV still does on-the-fly FP dequant rather than the
integer-dot dp4a path. The dp4a Q4_K shader and kernel already exist on `issue/52`.

---

## 3. Remaining highest-value Vulkan targets (ranked by expected speedup)

| Rank | Target | Kernel / file | Technique | Expected gain | Risk |
|---|---|---|---|---|---|
| 1 | **Land Q4_K dp4a MMVQ decode** | `matmul_q4_k_mmvq.comp` + `MatMulQ4KMmvqKernel.cs` (on `issue/52`) | Quantize activation to Q8_1 on device; `dotPacked4x8AccSatEXT` integer dot over Q4_K nibbles vs Q8_0-MMVQ already shipped | Brings Q4_K decode to the Q8_0 MMVQ regime; closes a large part of the 0.23–0.55× gap on Q4_K_M models (the common case). #52 A/B previously cited **+26.5 %** for the Q4_K dp4a MMVQ | **Low–Med.** Shader + parity test already written; gated + FP fallback. Must verify argmax-exact parity (Q8_1 activation quant is lossy by design). |
| 2 | **Execution-model overlap / projection fusion** (no issue yet) | `VulkanTransformerModel.cs` forward graph; new fused qkv / gate_up matmul dispatch | Issue independent GEMVs (q/k/v; gate/up) without intervening global `ComputeToComputeBarrier`; **fuse** qkv→1 matmul and gate+up→1 so small latency-bound GEMVs (M≈1024 @ 22 GB/s isolated) become large bandwidth-bound ones (200+ GB/s even serialised) | Diagnostics say this is the **largest structural win** — the kernel already streams 270–423 GB/s when overlapped; the forward serialises it to ~48 GB/s. Plausibly the bulk of the remaining ~1.8× | **High.** Touches the forward graph + weight layout (fused QKV/gate_up banks), descriptor sets, and barrier scoping. Needs op-swap parity guarding (see `debug-gpu-divergence-op-swap`). Biggest, riskiest — do AFTER #52. |
| 3 | **Q4_K MMVQ occupancy + wave32** | `matmul_q4_k_mmvq.comp` (#53), `Wave32SubgroupControl.cs` (#54) | Sub-block thread remap (#53) + targeted wave32 subgroup size on the MMVQ pipelines (#54) | Small per-item (#49 +4.6 %, #54 +0.9 % on Q8_0) but **free once #52 is in** — they are the same stacked branch | **Low.** Already implemented on the stack; mostly re-validation. Note wave32 gave noise on Q8_0 — measure on Q4_K before keeping. |
| 4 | **Q5_K / Q6_K dp4a MMVQ decode** | new `matmul_q5_k_mmvq.comp`, `matmul_q6_k_mmvq.comp` | Same integer-dot pattern as #52, one extra qh-byte (Q5) / 6-bit (Q6) read per element | Same class of win as #52 but for less-common quants; only worth it for models actually shipped in Q5_K/Q6_K | **Med.** New shaders (no branch yet); follow the #52 template once it lands. |
| 5 | **#49 subgroup-per-row Q8_0 MMVQ** | `matmul_q8_0_mmvq.comp` | Subgroup-per-row occupancy remap on the *already-shipped* Q8_0 MMVQ | +4.6 % on Q8_0 decode | **Low.** Standalone, mergeable; modest, independent of #52. |
| 6 | **Coopmat on Q4_K prefill** | new `matmul_q4_k_gemm_coopmat.comp` | Use `VK_KHR_cooperative_matrix` for Q4_K prefill GEMM where dims qualify (Q8_0 coopmat already exists) | Prefill-only (pp), not decode; helps long-prompt latency | **Med.** Coopmat dequant-feed plumbing; lower priority than decode. |

CPU-track (#47 merged, #51 open) is a separate `.NET 11`/AVX512-BF16 lever and not on the
Vulkan critical path — keep #51 on its preview-runtime branch.

---

## 4. Suggested sequencing

1. **#52** — land Q4_K dp4a MMVQ decode (rank 1). Highest value, lowest risk, code exists.
2. **#53 + #54** — fold in the stacked occupancy + wave32 tweaks (rank 3); re-measure, keep
   only what beats noise on Q4_K.
3. **#49** — merge the standalone Q8_0 MMVQ occupancy remap (rank 5) if it still measures +4 %.
4. **New issue: forward-graph overlap + qkv/gate_up fusion** (rank 2) — the big structural win,
   scoped after the kernel-side wins are banked so the A/B baseline is clean.
5. Opportunistic: Q5_K/Q6_K MMVQ (rank 4) and Q4_K coopmat prefill (rank 6) per shipped quants.

---

## 5. Recommended next concrete work item

### Issue: "Land Q4_K dp4a MMVQ decode path (close the Q4_K_M decode gap)"

Q4_K_M is the dominant real quant; on `dev` its decode GEMV still does FP dequant
(`matmul_q4_k_gemv_f32.comp`) while Q8_0 already uses dp4a MMVQ (#46). The dp4a Q4_K shader,
kernel, and parity test already exist on `issue/52-kquant-mmvq` — this item is to validate,
benchmark, and merge it forward into `dev`.

**Scope / files:**
- `native/vulkan/shaders/matmul_q4_k_mmvq.comp` (new) + compiled `spv/` (re-run
  `native/vulkan/build.ps1`).
- `src/DotLLM.Vulkan/Kernels/MatMulQ4KMmvqKernel.cs` (new).
- `src/DotLLM.Vulkan/VulkanTransformerModel.cs` — route `weightQt == Q4_K && seqLen == 1`
  through the MMVQ kernel when `device.HasIntegerDotProduct` + SPV present, else fall back to
  the existing `_matmulQ4K` FP GEMV (mirror the Q8_0 MMVQ gate at lines ~651–667 / ~3833).
- `tests/.../Vulkan/VulkanMatMulQ4KMmvqKernelTests.cs` (parity test, already on the branch).

**Acceptance criteria:**
1. Build clean; SPV recompiled and resolved at runtime.
2. Parity test passes: Q4_K MMVQ output matches the FP-dequant GEMV oracle within tolerance
   **and is argmax-exact** on the validation vector (Q8_1 activation quant is intentionally
   lossy — argmax-exact is the real bar).
3. End-to-end generation on a Q4_K_M model produces the **same tokens** as the FP path
   (op-swap / known-good-output guard per `debug-gpu-divergence-op-swap` — don't trust
   per-layer cosine).
4. `DOTLLM_VULKAN_MMVQ` disable flag still falls back to the FP GEMV cleanly.
5. Decode tok/s on a Q4_K_M model improves measurably vs `dev` (target: parity with the Q8_0
   MMVQ bandwidth regime; the #52 A/B previously cited +26.5 % for the Q4_K dp4a kernel).

**How to benchmark:**
```
# Baseline (dev, FP-dequant Q4_K GEMV):
DOTLLM_VULKAN_PERF=1 DOTLLM_VULKAN_PERF_MODEL=C:/Development/gguf-cache/<model>.Q4_K_M.gguf \
  dotnet test tests/DotLLM.Tests.Integration -c Release \
  --filter "FullyQualifiedName~VulkanGemma4DecodePerfHarness"

# After merging #52, same command; compare decode tok/s and the effective GB/s line.
# Cross-check against llama.cpp Vulkan -ngl 99 tg on the same gguf for the ratio.
```
Run a small (Llama-3.2-1B Q4_K_M) and a mid (3B/8B Q4_K_M) model — the gap widens with size,
so the win should too.

**Then:** stack #53 (sub-block thread mapping) and #54 (wave32) on top, re-measure, keep only
what beats noise. After that, open the rank-2 forward-graph overlap/fusion issue for the larger
structural win.

---

## 6. NEXT WORK ITEM (2026-06-21): forward-graph projection fusion

§5 (Q4_K dp4a MMVQ) is **done** (#52 merged). The remaining structural lever, and the largest
one per `vulkan-decode-barrier-bottleneck`, is **fusing independent projection GEMVs** so the
small latency-bound matmuls (M≈1024 @ **22 GB/s isolated**) become large bandwidth-bound ones.

### 6.1 Why fusion, not just barrier removal
On a single compute queue, dispatches execute in submission order; `NO_BARRIERS` alone gave only
**+1.7 %** because the dependency chain still serialises and the driver does not pipeline adjacent
GEMVs. The reliable lever is to **merge independent GEMVs into one larger matmul** (one dispatch,
larger M, more memory requests in flight) — the kernel already streams 270–423 GB/s once M is big
enough that it is bandwidth- rather than latency-bound.

### 6.2 The two independent pairs on the decode path (`VulkanTransformerModel.cs`)
Decode (`seqLen==1`) records, per layer:
- RMSNorm **+ Q proj** — **already fused** (`TryRecordFusedRmsNormMatmul`, line ~2006). Leave it.
- **K proj** (line ~2021) ‖ **V proj** (line ~2024) — independent, both read `_state.NormOutput`,
  write `_state.K` / `_state.V`. For Llama-3.1-8B GQA these are **M=1024 each** → the exact
  22 GB/s case. **← fuse these (slice 1, highest value).**
- **Gate proj** (line ~1770/decode equiv) ‖ **Up proj** — independent, both read post-FFN-norm,
  M=intermediate each, consumed by SwiGLU. **← fuse these (slice 2).**

### 6.3 Implementation approach — load-time concatenated weight bank
Weights upload one-buffer-each via `UploadMatrix` (`VulkanWeights.cs` ~557–572). GGUF tensors are
row-major `[out_features, in_features]` in quant blocks, so **stacking K's rows then V's rows along
the output dim is a valid `[K_out+V_out, hidden]` weight** — readable by the *existing* MMVQ/GEMV
kernels at M=K_out+V_out with **no new shader**.

- **Slice 1 (KV):** add `UploadFusedKV` — require `KDeviceQt == VDeviceQt` (else fall back to
  separate upload + the current two-dispatch path); concatenate the two quant byte-blocks into one
  staging→device buffer; store a new `KV` buffer + `KVSplitRow` (= K_out). Forward: one
  `RecordMatmul(KV, …, out=_state.KV, M=K_out+V_out)`, then K = `_state.KV[0 : K_out]`,
  V = `_state.KV[K_out : K_out+V_out]` (contiguous at seqLen=1). RoPE touches K (offset 0); cache
  update copies K@0 and V@K_out. Drops one dispatch + one `ComputeToComputeBarrier`.
- **Slice 2 (gate_up):** identical pattern over Gate|Up → one matmul → SwiGLU reads the two halves.
  (Note Gemma-4 *already* ships a fused gate_up bank for its MoE experts — reuse that layout idea.)

### 6.4 Guards / fallbacks / parity
- Gate the fused upload on `KDeviceQt == VDeviceQt` (and gate_up likewise); fall back to the
  current separate buffers + two-dispatch path otherwise. MLA / V-from-K (Gemma-4 global) layers
  stub K/V — **skip fusion there** (they already bypass the standard K/V matmul).
- Env opt-out (`DOTLLM_VULKAN_DISABLE_FUSED_KV` / `…_GATEUP`) for clean back-to-back A/B and a
  safety hatch.
- **Parity test:** fused-KV output sliced into K,V must be **bit-identical** to two separate
  `RecordMatmul` calls (same kernel, same bytes, just concatenated) — this is a strong, cheap
  oracle. Plus an **op-swap e2e** guard (same generated tokens vs `dev`), per
  `debug-gpu-divergence-op-swap` — do not trust per-layer cosine.
- **Benchmark:** back-to-back same-session A/B via the env flag, report `decode_min_ms`, under the
  GPU lock (`vulkan-decode-bench-uma-contention`). Expect the win to grow with model size; the
  M=1024 KV pair is the clearest target on 8B-class GQA.

### 6.5 Risk / sequencing
- **Slice 1 (KV) first** — highest value (directly attacks the 22 GB/s pair), self-contained,
  bit-exact oracle, low blast radius (one new buffer + forward routing). GPU-gated for the A/B.
- **Slice 2 (gate_up)** after slice 1 lands — same pattern, smaller relative win (M already large)
  but removes a barrier and a dispatch per layer.
- Each slice: own `issue/<n>-…` branch → parity (CPU-buildable) → GPU A/B → `--no-ff` merge to `dev`.
- The CPU-only prep (upload concat, forward routing, parity test) can be written now; the A/B and
  e2e-token guard wait for the GPU lock.

---

## 7. PIVOT (2026-06-22): §6 concat-fusion REJECTED → async/overlap of independent projections

**§6 concat-fusion was implemented, proven bit-exact, and REJECTED on measurement** (issue #335,
branch `issue/335-kv-proj-fusion`, NOT merged). Decisive A/B on Llama-3.2-3B Q8_0 (uniform ⇒ all
layers fuse), alternating order, `decode_min_ms`: fused **69.56/75.03** vs separate **66.06/73.73**
— fused *slower* in both pairs (~2–5%). The two transfer split-copies + COMPUTE↔TRANSFER barriers
per layer cost more than fusing the K+V matmuls saves. (8B Q4_K_M's "+1–2%" was noise — it only
HALF-fuses: attn_k=Q4_K but attn_v=Q6_K in 16/32 layers. And always ALTERNATE A/B order: a naive
sep-then-fused run showed a FAKE 9% regression from UMA contention drift.) Memory:
`vulkan-kv-projection-fusion-finding`.

### 7.1 Hardware feasibility (gfx1151)
`vulkaninfo`: **family 0 = GRAPHICS+COMPUTE, queueCount 8; family 1 = dedicated COMPUTE, queueCount
8**; transfer-only family; video families. So multi-queue async compute is hardware-supported. The
engine today is **single-queue, one-command-buffer-per-forward, synchronous (`vkQueueWaitIdle`)**
(`VulkanDevice` `queueCount=1`).

### 7.2 The real reason independent ops don't overlap (key finding)
The MMVQ bandwidth microbench (`MeasureMmvqBandwidth`, spike `4328e8b`) hit 270–423 GB/s by issuing
**back-to-back IDENTICAL dispatches reading the same immutable weights** (no hazards → the GPU
pipelines them on one queue). The real forward's `NO_BARRIERS` gained only +1.7% — because the
"independent" q/k/v **share one activation-quant scratch `Q8_1Xq`**: each re-quantizes `NormOutput`
into the *same* buffer (WAR/RAW), serialising them no matter the barriers. Same for gate/up.

### 7.3 Recommended PoC — shared-activation-quant + no-barrier overlap (SINGLE queue first)
q/k/v all read the **same** activation (`NormOutput`); gate/up read the same post-FFN-norm. So:
1. **Quantize the activation ONCE** (Q8_1) into a shared read-only buffer.
2. Issue q, k, v MMVQ matmuls **back-to-back with NO barriers** — all read the shared Q8_1 scratch
   (RAR, safe), write separate Q/K/V (no WAW). One barrier after all three.
3. Same for gate‖up.
This removes the scratch hazard that hobbled `NO_BARRIERS`, lets the GPU pipeline the three GEMVs
(the microbench's proven regime), AND drops 2 redundant activation-quantizes — all on the existing
single queue, **no multi-queue refactor**. Cost: hoist the Q8_1 quantize out of `RecordMatmul` for
the q/k/v + gate/up groups; give the MMVQ matmul a "pre-quantized activation" entry point.
Complication: the decode path's `TryRecordFusedRmsNormMatmul` computes Q inside the rmsnorm kernel —
the PoC likely bypasses that fusion (rmsnorm → quantize-once → q/k/v) and must A/B against keeping it.

### 7.4 If 7.3 shows headroom but caps → multi-queue async (heavier)
Only then: a second compute queue (family 1) + per-op command buffers + timeline semaphores at the
q/k/v→attention and gate/up→swiglu joins. Major change to the single-queue synchronous engine; do
NOT start before 7.3 quantifies the overlap ceiling. On bandwidth-bound UMA the multi-queue win over
7.3 may be small (same CUs/bus), so 7.3 is the gating experiment.

### 7.5 Sequencing
1. **PoC 7.3** (shared-quant + scoped no-barrier on q/k/v and gate/up), env-gated, bit-exact parity
   (`VulkanFusedKvParityTests` pattern), alternating A/B `decode_min_ms` on 3B Q8_0 + 8B Q4_K_M.
2. Keep iff it beats the noise floor; `--no-ff` merge. Else record and stop (decode is already
   ~0.5–0.7× llama.cpp post kernel-campaign).
3. Multi-queue (7.4) only if 7.3 is positive but clearly bandwidth-underutilised.

### 7.6 RESULT (2026-06-23): PoC 7.3 MEASURED — bit-exact but PERFORMANCE-NEUTRAL, NOT MERGED
The §7.3 PoC was built (issue #336, branch `issue/336-proj-overlap`, commit `808e4f6` — K‖V overlap:
quantize `NormOutput`→Q8_1 once, then K,V MMVQ back-to-back NO barrier, barrier after; env opt-in
`DOTLLM_VULKAN_ENABLE_PROJ_OVERLAP=1`) and **GPU-validated on Strix Halo gfx1151**:

- **Parity: bit-exact (L∞ == 0)** on SmolLM-135M Q8_0, Llama-3.2-3B Q8_0, Llama-3.1-8B Q4_K_M —
  all 8 decode steps each (`VulkanProjOverlapParityTests`). The overlap wiring is correct.
- **A/B `decode_min_ms` — NO measurable speedup.** Llama-3.2-3B Q8_0 (uniform ⇒ *all* layers
  eligible, the clean ceiling), two alternating-order rounds:
  - OFF: 26.29, 26.45, 26.42, 27.09 (mean 26.56) · ON: 26.21, 26.20, 26.66, 26.75 (mean 26.46).
  - Round 1 (OFF-first) showed ON ~0.6% faster; round 2 (ON-first) reversed it. **Net ≈ 0.4% =
    within the UMA noise floor.** No signal.
  - 8B Q4_K_M (partial — only K=Q4_K/V=Q4_K layers overlap; 16/32 have V=Q6_K and skip): OFF
    93.16/94.53 vs ON 93.11/95.49 — flat, pure noise.
- **Why neutral (not a regression like §6):** removing the shared-scratch hazard + 2 redundant
  activation-quantizes lets the GPU *issue* the GEMVs back-to-back, but on bandwidth-bound UMA the
  GEMVs are already memory-starved (~48 GB/s, ~19% of peak — memory `vulkan-decode-barrier-bottleneck`).
  Pipelining bandwidth-starved dispatches on a shared bus adds no effective bandwidth, so latency is
  unchanged. The microbench's 270–423 GB/s came from *identical* hazard-free dispatches hitting cached
  weights; distinct K/V weight banks don't get that.
- **DECISION: do NOT merge** (same disposition as #335 — branch kept, not merged). Per §7.5 step 2:
  does not beat the noise floor → record and stop.
- **Implication for §7.4 (multi-queue async): DO NOT PURSUE.** §7.3 is the gating experiment and it is
  neutral, not "positive-but-underutilised". The bottleneck is shared bus bandwidth, which multiple
  queues share on UMA — multi-queue cannot beat a single queue that is already bandwidth-saturated.
  The forward-graph projection-overlap line of attack is **exhausted** on gfx1151.

Memory: `vulkan-proj-overlap-finding`.
