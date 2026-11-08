# KV-Cache — dotLLM

## Purpose

The KV-cache stores previously computed key and value vectors for all layers, so each decode step computes and appends K/V only for the new token. Attention still reads the cached history: a decode step takes work proportional to the context length, and processing a full causal sequence has quadratic attention work in its length.

## Memory Consumption

Llama 3 8B, FP16, 2048 tokens:
```
2 (K+V) × 32 layers × 8 KV heads × 128 head_dim × 2048 tokens × 2 bytes
= 268,435,456 bytes = 256 MiB
```
For uniform full-attention layers, this scales linearly with sequence length and batch size. Per-layer KV widths and sliding windows change the total. The CPU `SimpleKvCache` stores separate K and V FP32 buffers per layer in position-major rows `[capacity, kvStride]`, where `kvStride = numKvHeads × headDim` for that layer; windowed layers retain only their configured recent rows. Other cache implementations can use different layouts and precision.

### Quantized KV-Cache Memory

Quantizing the KV-cache to Q8_0 or Q4_0 reduces memory by 3.8-7.1x vs FP32,
enabling long-context inference on memory-constrained systems.

Effective bytes per element (including 2-byte scale per 32-element block):

| Format | Bytes/element | Compression vs FP32 |
|--------|---------------|---------------------|
| FP32   | 4.000         | 1.0x                |
| FP16   | 2.000         | 2.0x                |
| Q8_0   | 1.0625        | 3.8x                |
| Q4_0   | 0.5625        | 7.1x                |

KV bytes per token (K+V, one layer, `kvStride = numKvHeads * headDim`):

| Model shape       | KV heads | head_dim | FP32   | Q8_0   | Q4_0   | K8V4   |
|--------------------|----------|----------|--------|--------|--------|--------|
| Qwen3.5 27B Dense  | 4        | 128      | 4096 B | 1088 B | 576 B  | 832 B  |
| Llama 3 8B/70B     | 8        | 128      | 8192 B | 2176 B | 1152 B | 1664 B |

Example: Qwen3.5 27B, 64 layers, 64K context:
- FP32: 16.4 GB
- Q8_0: 4.4 GB (3.8x savings)
- K8V4: 3.3 GB (5.0x savings)
- Q4_0: 2.3 GB (7.1x savings)

See `.docs/kv_quant_memory_estimates.md` for per-hardware breakdowns (hybrid
GPU/CPU split, T5500/Framework/Strix targets).

## Simple KV-Cache (Phase 1)

Pre-allocated contiguous buffer per sequence:
```
K_cache[layer][kv_head][max_seq_len][head_dim]  — FP16
V_cache[layer][kv_head][max_seq_len][head_dim]  — FP16
```
Simple indexing: `K_cache[layer][head][pos] = new_K`. Wastes memory for short sequences.

### Hybrid-aware geometry (issue #839)

For hybrid models (`ModelConfig.HybridLayout` set: Qwen3.5/3.6/3.8 GDN, Nemotron-H, Qwen4-Exp) only the attention layers hold K/V.
`KvGeometry.FromConfig` therefore allocates **slots for attention layers only**, in layer order (slot k = k-th attention layer — the
`kvSlotForLayer` convention every hybrid forward pass, CPU/Vulkan/CUDA, already uses). The geometry exposes the map:
`HasLayerSlotMap`, `SlotOfLayer(modelLayer)` (-1 for a recurrent layer), `ModelLayerCount`; `LayerCount` is the **slot** count and
`KvStrideOf(slot)` takes a slot. `KvGeometry.SlotCount(config)` is the allocation-free slot count (use it instead of
`config.NumLayers` anywhere a cache or handoff is sized). A 1-in-4 hybrid (Qwen4-Exp: 12 of 48) allocates 4x less KV; dense models
are unchanged (`HasLayerSlotMap == false`, identity map). Vulkan (`VulkanNemotronHKvCache`) and CUDA (`CudaNemotronHKvCache`) hybrid
caches were already sparse; the CPU `SimpleKvCache` / `QuantizedKvCache` / `TurboQuantKvCache`, the hybrid CPU-tail cache of the
split Qwen3-hybrid model, the KV handoff paths and the CLI memory estimate now agree with them.

### Vulkan hybrid KV: element format and lazy growth (issue #898)

`VulkanNemotronHKvCache` (shared by the Vulkan Nemotron-H, Qwen3.5/3.6/3.8 hybrids and Qwen4-Exp) takes two opt-in constructor arguments, both OFF by default so existing models are
untouched: `storage: VulkanKvStorage.F16` (K/V rows written by a convert kernel, 2 bytes per element; read by the Qwen4-Exp QSA gather kernel and,
since #1102, by the `_kvf16` attention twins the dense hybrid selects; see the next section) and `lazy: true` (buffers start empty and `EnsureCapacity(positions)` reallocates layer by layer, 25 % / >= 1024 positions at a time, copying the live prefix).
`MaxLength` is the logical limit, `AllocatedLength` the physical one. After `EnsureCapacity` returns true the CALLER must drop every descriptor cache that may name the freed buffers
(handles are recycled; a stale cached set reads zeros without any error). Qwen4-Exp does this in `ForwardCore` before recording (`EnsureContext`), for the sequence's K/V and its indexer
state. Switches and measurements: [ATTENTION.md](ATTENTION.md), "Lazy growth and F16 K/V".

#### Growable caches for a continuing session (issue #1100)

The engine grows a sequence's KV with `IKvCache.TryGrowTo` when a recurrent-prefix hit needs more positions than the cache
was made with (a growing chat session needs that on almost every turn). The Vulkan cache used to keep the interface default
(never grows), so on Vulkan every turn after the first missed its prefix, and after #1093 a long one was refused.

- **`TryGrowTo(n)`** raises `MaxLength` up to the `growthLimit` the cache was created with (the model's window; 0, the
  default, means never). A lazy cache only raises the limit, and its buffers grow in the forward. An eager cache re-creates its
  buffers at once, copying the live rows layer by layer, so the transient cost is one layer's K and V. Out of device memory
  returns `false` and leaves the cache as it was.
- **Hosts.** The Bonsai / Qwen3.5 dense hybrid (`VulkanQwen3HybridDenseTransformerModel`) and LFM2
  (`VulkanLfm2MoeTransformerModel`) create **lazy** caches capped at the window. Before recording, each forward calls
  `VulkanNemotronHKvCache.PrepareForForward(kv, positionsEnd, ref seenEpoch)`. It grows the buffers to cover the forward and maps
  out-of-device-memory to `DeviceMemoryAdmissionException` (HTTP 400, not a crash). It returns true when the caller must drop
  its descriptor caches.
- **Descriptor epoch.** Kernels cache descriptor sets by buffer handle, and Vulkan recycles a freed handle into the next
  allocation. Any cache of this type that frees buffers (a growth, per layer, or a dispose) bumps the process-wide
  `BufferEpoch`. A model drops every descriptor cache when the epoch it last saw has moved. The check covers another sequence's
  growth and a snapshot's dispose, not only this cache's. The Arc driver does not recycle a handle soon enough for a stale set
  to change the logits. The tests therefore pin the invalidation itself (`KvBufferInvalidationsForTest`), as well as
  bit-identical logits against a cache allocated in one step.
- **Snapshots.** `SnapshotPrefix` copies into an exactly-sized eager cache. `RestorePrefixFrom` grows a lazy destination first.
  A snapshot taken before a growth restores exactly after it (`VulkanQwen3HybridDenseKvGrowthTests`).
- **Cost.** Growth, admission and snapshot copies size everything from the cache's `ElementBytes` / `BytesPerPosition`, never an
  assumed F32. Since #1102 the storage is F16 by default: 64 KiB per token for Bonsai-2 (16 attention layers x 4 KV heads x 256 x
  K and V x 2 B), so 8 GiB at 128k (F32, the A/B baseline: 128 KiB, 16 GiB). A growth step is 25 % of what is held (at least
  1,024 positions), plus one layer's K and V in transit.

### Prefix reuse through a device-owned cache: `IPrefixReusableKvCache` (issue #1591)

Vulkan's single-request path has no paged pool, so the Step 54 `PrefixCache` retains **whole `IKvCache` objects**. A cache that implements
`IPrefixReusableKvCache` (`VulkanKvCache`, `VulkanQuantizedKvCache`) is adoptable by a later request when it `HasExclusiveStorage` (#1025:
rows valid only until a forward runs on a different cache otherwise), `CanTruncateTo(n)` (compact sliding-window layers refuse a cut past
their margin) and is large enough for the next request. `TextGenerator.ResolveKvCache` then clamps the match to
`min(common prefix, cache.CurrentLength, promptLen - 1)` and `Rollback`s. The `CurrentLength` clamp matters: the stored sequence ends with
the last sampled token, which was never forwarded, so its row does not exist. The `promptLen - 1` clamp keeps one token to prefill.
`StoreInPrefixCache` retains only reusable cache types (others were retained until evicted without ever matching). Device caches cannot
grow in place, so `prefixCacheHeadroomTokens` (1,024 on Vulkan) sizes the retained allocation for the next turn. Unsafe hosts (sinks,
recurrent state, non-exclusive storage, TurboQuant/MLA/pipeline caches) take a miss. Tests: `PrefixCacheReusableKvTests` (CPU model behind a
delegating cache; mutants: no CurrentLength clamp, ignored exclusivity flag, ignored `CanTruncateTo`, wrong cached count, no cap, no headroom
all fail) and `VulkanPrefixReuseTests` (greedy continuation equals a cold prefill on the real model).

### Vulkan dense-hybrid (Bonsai 2) KV: F16 storage read in place (issue #1102)

`VulkanQwen3HybridDenseTransformerModel` (Bonsai 2, Qwen3.5 / 3.6 dense hybrids) keeps its KV in a lazy, growable
`VulkanNemotronHKvCache` (#1100) whose element format is selectable:

```bash
dotllm serve <bonsai> --device vulkan            # F16 KV by default
dotllm bench <bonsai> --device vulkan --kv-quant-k f32 --kv-quant-v f32   # explicit F32 A/B baseline
```

| storage | bytes / element | Bonsai 2 per token (16 attn layers × 4 KV heads × 256 × K+V) | 128k | 262,144 | 1M (YaRN, #1101) |
|---|---|---|---|---|---|
| f32 (A/B baseline only) | 4 | 128 KiB | 16 GiB | 32 GiB | 128 GiB |
| **f16 (default)** | 2 | 64 KiB | 8 GiB | 16 GiB | 64 GiB |

- **Read in place, no widening.** Every attention kernel the model dispatches has an F16 twin (`*_kvf16.comp`) that
  loads its K / V elements from the packed halves (`kv_load.glsl`, `unpackHalf2x16` in the load). No kernel converts the
  history to an F32 scratch, so the device holds exactly the per-token cost above plus the existing per-forward
  scratch. Decode reads 64 KiB per cached token instead of 128 KiB. Details and the kernel list:
  [ATTENTION.md](ATTENTION.md), "Narrow (F16) KV on Vulkan".
- **Writes.** The F32 K / V rows the projections produce are converted on the device by `convert_f32_to_f16_range.comp`
  (round-to-nearest-even in the integer domain, saturating at ±65504), from contiguous positions only. That is every
  forward the engine issues: prefill tiles, decode, MTP verify and each `ForwardBatch` sequence.
- **Snapshots, restore, rollback, growth** copy whole rows byte for byte, so they keep the format and round-trip exactly
  (`ValidateSnapshotLayout` no longer requires F32).
- **Selection.** `ConfigureKvStorage(VulkanKvStorage)` sets the format of caches created afterwards; the forward picks
  kernels per cache. The CLI and server accept `f16` / `fp16` and `f32`. q8_0, q4_0, TurboQuant and mixed K / V are
  refused at startup with a message, never stored as something else. **F16 is the default** (#1107: no F32 KV on hot
  paths). F32 is the explicit A/B and parity baseline only: `--cache-type-k/-v f32` on `run` / `bench`, or `DOTLLM_VK_KV=f32`
  (the server cannot tell an explicit `f32` from its option default).
- **Measured on Strix** (`serve --device vulkan --no-mtp`, one session grown by ~8k-token turns with `scripts/session-growth.py`,
  64 generated tokens per turn; prefix reuse on every turn after the first; no device loss to 128k with #1100's submit budget):

  | context | decode tok/s, F32 KV | decode tok/s, F16 KV | prefill tok/s (new turn), F16 |
  |---|---|---|---|
  | 7.8k | 19.6 | 21.2 | 179 |
  | 32.9k | 12.6 | 15.9 | 66 |
  | 64.3k | 8.7 | 12.0 | 38 |
  | 128.3k | 5.4 | **8.1** | 21 |

  Process private bytes grew 134 KiB/token with F32 and 70 KiB/token with F16: the cache is 128 vs 64 KiB/token, plus the lazy
  growth headroom. Both arms ran the GQA-grouped decode pass. In the per-layer kernel bench (ATTENTION.md) the per-query-head
  pass is within 5 % of it for F32, so the F32 column is close to the old default.

  The F16 arm ran the paired-load variant of that pass (#1113). The shipped per-element pass measures 6.13 instead of 5.37 ms
  per layer at 128k, about 12 ms more per token (16 layers), so expect ~7.4 tok/s at 128k with the shipped kernels.
- **Accuracy.** F16 rounds each K / V element once (relative error ≤ 2⁻¹¹). On the synthetic fixture the logits stay
  within the tolerance the Vulkan forward is held to against the CPU reference (abs 5e-3 + rel 1e-3).

### Vulkan dense host: F16 KV storage (issues #1151, #1549)

`VulkanTransformerModel` (Llama, Mistral, Qwen incl. Qwen3, Phi: no MLA, no sinks, no shared-KV layers, head dim <= 128) creates an
**F16** `VulkanKvCache` by default (CLAUDE.md precision rule 1). `DOTLLM_VK_KV=f32` is the explicit A/B and oracle baseline, and
`CreateKvCache(maxSeqLen, compactSlidingWindows, VulkanKvStorage.F32)` gives an F32 cache in-process. KV bytes per token halve
(Qwen3-4B: 288 -> 144 KiB). Every attention dispatch reads the halves in place with F32 accumulation (no F32 copy of the history):

- **Writes.** Contiguous rows: the fused RoPE + KV write twin `rope_kv_write_f32_kvf16.comp` (decode and prefill), else the
  unfused `convert_f32_to_f16_range.comp`. Rows at arbitrary positions (selective recompute, scattered batches):
  `convert_f32_to_f16_rows.comp` (one dispatch per tensor, positions read from the device buffer). All round to nearest-even in
  the integer domain and saturate at +-65504 (the hardware conversion may round toward zero on AMD).
- **Reads.** Decode: split-KV `_kvf16` twin, else the per-token twin. Prefill: GQA flash, coopmat flash, scalar flash, per-token
  (all `_kvf16` twins). Selective recompute: `attention_f32_qpos_kvf16` and the tiled GQA qpos `_kvf16` twin. Query-relevance
  probe: built per storage. `ForwardBatch` (fused batched decode) takes F16 caches.
- **Rope shift.** `RopeShiftKv` has an F16 pipeline (rotation F32 in registers, one rounding at the store).
- **Host IO.** `IPositionedHostKvCache.WriteRows/ReadRows` and the host-staged `UploadLayer/DownloadLayer` take and return **F32 host
  rows**; an F16 cache rounds on write and widens on read, so write -> read is exact only to F16, and writing the rows just read
  back is bit-idempotent.
- **Architectures not switched** (still F32 until their dispatch is audited): Gemma-2/3/4 (soft-cap, hd256, shared KV), gpt-oss and
  MiMo (sinks; Flint is already F16), MLA (DeepSeek), BitNet/Granite, and every host with its own cache (Nemotron-H, LFM2/LFM3,
  Qwen3.5/3.6 MoE hybrids). Bonsai-2 dense-hybrid and Qwen4-Exp were already F16.

**Accuracy (Qwen3-4B-Instruct Q4_K_M, Strix, wikitext-2 LF, 512-row prefill chunks through the cache, KL(F32 KV || F16 KV)).**

| band | rows | KL median | p99 | max | top-1 | NLL F32 / F16 |
|---|---:|---:|---:|---:|---:|---|
| prefill [0, 2K) | 2,048 | 3.5e-7 | 8.2e-6 | 4.3e-5 | 100% | 2.0548 / 2.0547 |
| prefill [2K, 8K) | 6,144 | 7.2e-7 | 1.4e-5 | 0.226 (1 row) | 99.98% | 2.4779 / 2.4778 |
| decode @ 8K (32 steps) | 32 | 2.0e-7 | 2.5e-6 | 2.5e-6 | 100% | 1.8194 / 1.8188 |

The one outlier (position 5224, token "1") is a chaotic row of the model, not an F16 defect: swapping only the F32 prefill attention
kernel (GQA flash vs per-head coopmat, both F32 KV) moves that row's layer-17 K by 0.35 relative, the same as F16 does (0.36); the
K/V of that position diverge from layer 16 on whatever the KV format. Median KL 4e-7 matches the issue's ~1e-6 for Q4_0-class models;
Llama-1B-class models are the sensitive end (5e-4, #1109) and were not re-run here.

**Speed and memory (same process, F32/F16 caches alternating A B B A, 3 rounds, Qwen3-4B Q4_K_M, Strix; min / median / max):**

| metric | F32 KV | F16 KV |
|---|---|---|
| prefill 512 rows, tok/s | 1705 / 2109 / 2242 | 2063 / 2136 / 2232 |
| decode @ ctx 512, tok/s | 58 / 60.0 / 60.7 (2 contended outliers) | 59 / 60.5 / 61 |
| decode @ ctx 8K, tok/s | 27.0 / 27.2 / 27.3 (1 outlier) | 26.6 / 27.7 / 28.4 (1 outlier) |
| KV bytes / token | 288 KiB | 144 KiB |

Speed is a wash (the dense decode is weight-bound; the attention kernels are the same code with a different load). The win is
memory: twice the context in the same VRAM, and the F32 prefix copy that chunk reuse needed goes away. (Cross-process CLI runs
on this box swung 5 to 60 tok/s on identical settings from background load; only the in-process paired numbers are quoted.)

**RoPE-shift error accumulation (`RopeShiftKv`, real Qwen3-4B K rows, 512 rows x 8 heads, relative head-norm error vs an f64
rotation by the summed delta; mean / worst over rows and heads, layers 0, 18, 35):**

| shifts | F32 cache | F16 cache |
|---|---|---|
| one, delta 1 .. 1024 | <= 2.7e-6 / 5.2e-6 | 1.8-2.8e-4 / 3.9-6.1e-4 |
| one, delta 8192 ; 32768 | <= 2.1e-5 / 4.1e-5 ; 8.6e-5 / 1.7e-4 | <= 2.9e-4 / 7.8e-4 ; 2.9e-4 / 8.7e-4 |
| 4096 x 2 ; x 3 ; x 4 (mean) | <= 2.1e-5 ; 3.2e-5 ; 4.3e-5 | 3.9e-4 ; 5.0e-4 ; 6.1e-4 (worst 1.4e-3 at x4) |

The F16 floor is the stored rounding (~2.2e-4 mean at the first store); each re-shift adds about one more rounding in quadrature
(up to ~0.3e-3 per shift in the worst layer). The F32 cache's own error is the float angle and grows with |delta|. Shifting an
already shifted F16 chunk four times costs 1.4e-3 worst-case relative head norm. The test asserts
`mean_F16 < 4.5e-4 * sqrt(1 + shifts) + mean_F32` and the matching worst-case bound.

**Selective recompute on the F16 cache (#1549).** Qwen3-4B, 96 rows with 8 scattered, and 2048 rows with 64 scattered through the
tiled kernel: the recomputed rows match the full prefill as closely as the contiguous tail control (96/8: mean K 8.6e-3, V 1.6e-2 vs
control 8.9e-3, 1.6e-2; F32 cache 8.3e-3, 1.6e-2), rows outside the subset are byte-identical, and the qpos kernel ran once per layer.
The floor is the activation-quantised Q4_K_M forward shape (#1534), not the KV format.

Mutants (all die): scatter write treated as contiguous -> untouched/recomputed rows off by 2.7 (expected 0); F32 qpos kernels fed
an F16 cache -> refused by the size guard; F32 split-KV kernel on an F16 cache at decode -> batched-vs-F32-oracle KL 26 (the
batch test compares against an F32-cache oracle, because a wrong kernel is wrong identically in both F16 arms).

#### For chunk-reuse and other external consumers

- **Use F16 KV (or Q8_0 once #1550 lands). Do not require exact F32 prefix copies.** The F32 cache is an oracle for tests, not a
  storage format to preserve. A chunk's K/V captured from an F16 cache is exactly what an F16 cache holds; moving it into another F16
  cache with `CopyRowsFrom` (device to device) or `ReadRows` -> `WriteRows` (host F32 in between) is bit-exact, because the host F32
  rows are widened halves and re-rounding a value that is already a half is the identity.
- **Rows.** `IPositionedHostKvCache.WriteRows(layer, startPosition, count, keys, values)` / `ReadRows(...)` take host F32
  `[count, kvStride(layer)]`. K is stored post-RoPE, so a chunk placed at a different position needs `RopeShiftKv` by the position
  delta (device, F16 pipeline exists). `CurrentLength` becomes `max(CurrentLength, start + count)`.
- **Tolerance guidance.** Expect KL vs an F32-KV run ~4e-7 median on Qwen3-4B Q4_K_M (5e-4 on Llama-1B Q8_0), top-1 agreement
  >= 99.9%, a stored-row relative error ~2.2e-4 (one rounding), plus up to ~0.3e-3 per RoPE re-shift in quadrature. Do not compare a
  reused chunk bit-for-bit with a fresh prefill: even two F32 prefill kernels differ by ~1e-2 relative on late-layer K/V of individual
  rows (the recompute floor above). Compare with KL / top-1 / PPL, and budget the occasional chaotic row (one in 8,224 here, KL 0.23).
- **Do not** write F32 into a Vulkan KV cache through raw buffers: only the cache's own writers know the element format.

### Flint Q8_0 KV on Vulkan (opt-in, issue #1605)

`DOTLLM_VK_KV=q8_0` (Flint only) stores K / V as 36-byte blocks (`[float scale][32 x int8]`, 1.125 B/element) written by
`convert_f32_to_q8_0_kv_range` and **read in place** by the `_kvq8` twins of the split-KV decode (per-head pass via `kv_load.glsl`, GQA v2
pass with its own loader), the reference sink kernel and the coopmat sink-flash prefill. No history dequant scratch. No small-row (2..4)
twin, no host row IO, prefix snapshots or handoff (they refuse Q8_0). It stays opt-in; F16 is the Flint default.

- **Bytes:** K+V per token per layer 1,152 B vs F16 2,048 B (0.5625x); 14 full layers grow at 15.75 KiB/token vs 28 KiB. Caches measured:
  8K 180 vs 320 MiB, 128K 2,073 vs 3,686 MiB.
- **Speed:** NOT faster. Decode equal to F16 at 2K, slower at depth (128K median 20.2 vs 26.0 tok/s); prefill equal at 8K.
- **Quality (8K, 512 scored rows at depth 7.7K, three corpus windows, F32 KV reference):** KL mean q8_0 0.068 / 0.076 / 0.054 vs F16
  0.019 / 0.034 / 0.013 (2.3-4.3x the F16 floor; medians 3.8-4.9x); top-1 93.6-94.5% vs 95.1-97.5%; NLL within the run-to-run spread of
  F16 (no PPL loss). Needle recall 6/6 at 8K and 3/3 at 32K for both formats.
- **128K:** the q8_0 needle prefill arm coincided with a hard system lock-up (unexpected shutdown, no TDR event) on 2026-11-06; the F16 arm
  of the same run passed. Not retried. Earlier 128K q8_0 decode sweeps and a PPL run completed, so the cause is unconfirmed.
- **Proof the path ran:** halving the writer's stored scale moves q8_0 NLL 2.34 -> 9.53 while F16 stays bit-identical (2.3242); zeroing
  the GQA v2 decode kernel's K scale turns a q8_0 needle HIT into a MISS with F16 unchanged.

### Vulkan quantized KV and the per-query-position attention (feasibility, issue #1550)

Question: can the per-query-position attention (`attention_flash_gqa_qpos*`, #1540; used for selective recompute and chunk-KV
reuse) read Q8_0 / Q4_0 KV in place on Vulkan? Read-only study, nothing implemented.

**What the standard Qwen3 / Llama Vulkan path supports today.**
- `VulkanTransformerModel.CreateKvCache` gives an **F16** `VulkanKvCache` for the plain dense architectures since #1151 (see
  "Vulkan dense host: F16 KV storage" above; F32 is `DOTLLM_VK_KV=f32` only), so F16 qpos is reachable from a dense model (#1549).
- `--kv-quant-k/-v q8_0|q4_0` selects `VulkanQuantizedKvCache` (split layout: one `float` scale per 32-element block plus
  packed quants in a separate buffer; Q8_0 = 8 uints of int8, Q4_0 = 4 uints of nibbles, adjacent pairs per byte). It is
  **not read in place**: every layer of every forward dequantizes `[0, CurrentLength)` of both K and V into one shared F32
  scratch (`RecordDequant`) and the ordinary F32 kernels read that. The scratch is allocated up front at
  `2 x maxSeqLen x stride x 4 B` (Qwen3-4B, 32K: 256 MiB, i.e. twice an F32 cache of one layer, on top of the quantized
  store), and each forward re-dequantizes the full history of 36 layers. That is the expansion CLAUDE.md precision rule 2
  calls a defect (the Bonsai-2 Vulkan path refuses q8_0/q4_0 rather than do it).
- The qpos dispatch arm takes `VulkanKvCache` in F32 or F16 (#1549), not the quantized cache, and `VulkanQuantizedKvCache.RecordUpdate` accepts
  contiguous ascending positions only. So no quantized qpos path exists; a recompute over a quantized Vulkan cache is
  refused or mis-masked.

**Bytes per token, Qwen3-4B (36 layers x K+V x 1024):** F32 288 KiB, F16 144 KiB, Q8_0 split layout 81 KiB (1.125 B/element;
76.5 KiB in standard 34-byte blocks), Q4_0 split 45 KiB (0.625 B/element).

**What an in-place qpos variant takes.**
- *Read side.* The qpos kernels already stage K / V tiles global -> registers -> LDS in `fetchTile` / `storeTile`
  (`attention_flash_gqa_cm.glsl`, `attention_flash_gqa.glsl`), converting F16 or F32 to the LDS element there. A quantized
  twin changes only that load: one `uint` of quants (Q8_0: 4 int8 = one vec4; Q4_0: 4 nibbles = 16 bits of a uint) times the
  block scale, dequantized in registers inside the tile load, so no wide history is ever stored and the LDS tile is
  unchanged. Cost: two more bindings (K and V scales) on top of q, kq, vq, out, qpos (7 total), and a per-tile scale fetch
  (one scale per 32 elements, so a 128-wide head is 4 scales per row).
- *Variants.* K x V dtype pairs (`q8q8`, `q4q4`, `q8q4`, `q4q8`) x {scalar, coopmat hd256, coopmat hd128} = 12 pipelines if
  done as separate wrappers; fewer with `KV_K_DTYPE` / `KV_V_DTYPE` specialization constants (the repo's rule: spec constant,
  not runtime uniform, for register-pressure-sensitive choices; needs a third arm in any A/B). Start with `q8q8` and `q4q4`
  (6 pipelines). Extra LDS: none. VGPRs: the dequant adds a few registers to the prefetch stage `pf[]` (`KVREG` becomes a
  `uint` pair plus a scale), so re-check the VGPR granularity cliff on gfx1151 with a ballast arm.
- *Write side.* `KvQuantizeKernel` takes a contiguous block range. Chunk recompute writes arbitrary positions, so it needs a
  position-indexed quantize (or per-run dispatches) and `VulkanQuantizedKvCache` row IO (`WriteRows` / `ReadRows` /
  `CopyRowsFrom` like `VulkanKvCache`), plus the routing arm widened from `VulkanKvCache` to the quantized cache.
- *Precision/oracle.* The parity oracle is "dequant to the F32 scratch, then the F32 qpos kernel" on the same cache, which
  must be bit-identical for Q8_0 (same arithmetic, same order), plus the f64 reference at 2K..32K and mutants (wrong
  scale index, nibble order, off-by-block at the tile edge). Re-verify the Q4_0 nibble layout against the CUDA / GGUF
  convention first: the Vulkan quant shader comment says "matches GGUF" while the byte order it describes (adjacent pairs)
  is not the GGUF low/high-halves layout.
- *Effort.* qpos-only, Q8_0 + Q4_0 same-type pairs, with the write path, routing, tests and one measured A/B: about 3-5 days.
  Removing the full-history scratch from the ordinary prefill / decode / split-KV paths (the part that actually fixes rule 2
  for ordinary serving) is a separate 2-3 week item: every attention kernel family gains quantized twins, as the CUDA
  hybrid-dense path did in #988 / #1094.

Tracked in #1550 (implementation).

### CUDA hybrid-dense KV: staging, quantized storage, fused decode (issue #956)

`CudaQwen3HybridDenseTransformerModel` (Bonsai-2 27B and other qwen35 dense hybrids) owns its KV buffers behind the
length-only `CudaHybridKvCacheHandle`. Per full-attention slot (16 of 64 layers for Bonsai-2: 4 KV heads x 256 dim,
32,768 elements/token across slots):

| storage | per token | at 64k |
|---|---|---|
| F16 (default) | 64 KiB | 4.0 GiB |
| Q8_0 (`DOTLLM_HYBRID_KV=q8_0`) | ~34 KiB | ~2.2 GiB |
| K8/V4 (`k8v4`) | ~26 KiB | ~1.7 GiB |
| Q4_0 (`q4_0`) | ~18 KiB | ~1.15 GiB |

- **One slot, growable (#1025).** The storage is single-slot: a forward on a different handle overwrites it from position 0,
  so the handle reports `HasExclusiveStorage = false`, and the engine's recurrent prefix snapshot is dropped when another
  request runs. Growing a handle (`TryGrowTo`, used when a longer follow-up prompt reuses that snapshot) keeps the live
  rows. Since #1080 it grows **in place**: each slot's K and V address range is reserved for the context window
  (`cuMemAddressReserve`) and physical pages are mapped in as it grows (`CudaGrowableDeviceBuffer`), so nothing is copied
  and a growth admits only the added 2 MiB pages. Without the CUDA VMM API (or with `DOTLLM_CUDA_KV_VMM=0`) it falls back
  to reallocating and copying, where old and new storage coexist during the copy.
- **F32 staging is shared and on demand.** Only the F32-consuming tiers (prefill `attention_f32`, opt-in/fallback decode
  tiers) read F32 K/V. They get ONE staging pair shared by all layers, fully reconverted per use (512 MiB at 64k). The
  default mma decode kernel reads the F16 cache directly and stages nothing. Before #956 every slot kept a persistent F32
  mirror (128 KiB/token, power-of-two growth): 8 GiB at 64k, 16 GiB past 65,536.
- **Quantized storage** (`--kv-quant-k/--kv-quant-v` or `DOTLLM_HYBRID_KV=f16|q8_0|q4_0|k8v4|k4v8`): rows are quantized on write
  (`quant_f16_to_q{8,4}_0`, standard 32-element blocks) and read either by the **fused** decode kernel
  (`attention_flash_mma_decode_gqa_split_kvq_*`, dequant inside the K/V tile load, bit-identical to the scratch path;
  **default ON since #988's v2 kernels** — v1 fused measured slower than scratch, Q4_0 at ctx 16k 12.4 vs 13.3 tok/s, v2
  reads whole blocks per quad and holds 18.5 tok/s flat from ctx 512 to 16k; `DOTLLM_HYBRID_KV_FUSED=0` opts out), by the
  **fused flash prefill** (`attention_flash_mma_prefill_hd256_kvq_*`, issue #1094: prefill tiles and MTP verifies wider than
  the 8-row per-row tier, dequantizing one 16-row K/V tile at a time in shared memory, bit-identical to the scratch path), or
  by a dequant into one shared F16 scratch pair (the fallback tiers, shapes the hd256 kernels do not cover, and decode when
  fused is off).
- **No full-history F16 copy on the Bonsai path (#1094).** Before #1094, every quantized-KV prefill tile dequantized the
  whole history of every attention layer into that scratch, which grows with the context: 2 bytes x 2 sides x
  `numKvHeads * headDim` per token, **4 KiB/token on Bonsai-2, ~0.5 GiB at 128k**. It was what held a 12 GB RTX 3060 at a
  110k fit (q4_0, no MTP, 512 tile). The served fit no longer counts it when the fused prefill applies: q4_0/q4_0 without
  MTP is 18 KiB/token (KV rows only). `CudaQwen3HybridDenseTransformerModel.FlashPrefillHd256KvqForTest` turns the tier off
  for A/B tests. Remaining `ResolveKvF16` callers (decode with `DOTLLM_HYBRID_KV_FUSED=0`, decode below the mma split
  minimum, the F32 fallback tiers) still allocate the scratch on demand, admitted against free VRAM.
- **The MTP head's KV follows the trunk's type (#1095).** `CudaMtpState` used to keep the head's one attention layer in F32
  (8 KiB/token on Bonsai-2) and grow it by copying, so the fit counted 16 KiB/token, more than the q4_0 trunk's 16 layers
  together (18 KiB). The head now stores K/V in the trunk's type per side: Q8_0 / Q4_0 (and the mixed pairs) when the fused
  quantized-KV decode kernel takes the head's shape, otherwise F16, never wider (#1107). Rows are written exactly like the
  trunk's (F32 -> F16 staging -> `quant_f16_to_q{8,4}_0`), byte for byte. A draft step reads the history in place with
  `attention_flash_mma_decode_gqa_split_kvq_*` (quantized) or the F16 tensor-core decode kernel; below their 256-key minimum
  the short history goes through the dequant scratch and the F32 kernel, which that minimum bounds. The head's attention reads
  are all single-row draft steps (the absorb writes K/V only), so the prefill and per-row verify tiers never see its cache.
  K, V and the position iota grow **in place** with `CudaGrowableDeviceBuffer` (address range reserved for the ceiling, pages
  mapped on growth); without the VMM API the copying growth remains. Per token on Bonsai-2: q4_0 **1,156 B** (2 x 576 + 4 iota),
  q8_0 2,180 B, f16 4,100 B, against 16,392 B counted before. `DOTLLM_CUDA_MTP_KV=f32` restores F32 storage for A/B runs only.
- **KV Q4_0 is not the GGUF Q4_0 layout.** KV blocks pack ADJACENT PAIRS (low nibble = even element), matching CPU
  `KvQuantize` and `quant_kv.cu`; the weight dequant (`dequant_q4_0_f16`) unpacks low/high HALVES. Always use
  `CudaKernels.LaunchKvDequantToF16` for KV data. (The eager `CudaQuantizedKvCache` path used the weight dequant until
  #956, permuting elements inside every 32-block.)
- **CLI selection (issue #967).** `run`, `bench` and `serve` accept `--kv-quant-k` / `--kv-quant-v` (`f16`, `q8_0`, `q4_0`);
  on `run` and `serve` they are synonyms of `--cache-type-k/-v` (giving both with different values is an error).
  `dotllm run --device gpu --kv-quant-k q8_0 --kv-quant-v q4_0` selects K8/V4. Precedence is **per side**:
  explicit flag, then `DOTLLM_HYBRID_KV`, then F16 (so `--kv-quant-k q4_0` with `DOTLLM_HYBRID_KV=q8_0` gives K4/V8, and
  `--kv-quant-k f16 --kv-quant-v f16` forces unquantized over the env var). `serve`'s `--cache-type-*` are unset by default (#1284: unset reads as "not given", leaving the env var in charge, and on CPU paged selects Q8_0 blocks); an explicit `f32` is also treated as "not given" by the CUDA hybrid-dense model; pass `f16` to force unquantized.
  The pure logic lives in `HybridKvStorage` (Core); the model exposes `ConfigureKvStorage(key, value)` (call before the
  first forward), `KvStorageLabel`, `ComputeKvBytes(maxSeq)` and `KvAllocatedBytes`; the VRAM ledger and the `run`
  summary share that byte math. `run` prints the effective format in the Memory box
  (`KV Cache  ... (N slots, KV k8v4 (K:Q8_0 V:Q4_0), 1664 MiB)`), `bench` prints a `KV ...` line after the table and
  records the flags in the results.csv settings column, `serve` logs `KV storage: ...` at startup.
  - A flag the selected model/backend cannot honour prints a `WARNING` naming the flag, model and backend, never silence:
    Vulkan (no quantized KV), split GPU/CPU hybrids, Qwen3MoeHybrid and Nemotron-H (own cache, no quantized storage), and
    `bench` on CPU / plain CUDA models (its `SimpleKvCache` ignores them). `--cache-window` is ignored (warned) for the
    hybrid-dense model: one format for the whole cache, no mixed-precision window.
  - Rejected with an error (exit 1 / startup failure): TurboQuant on this model, and quantized storage when
    `numKvHeads * headDim` is not a multiple of 32.
- `DOTLLM_HYBRID_VRAM_LEDGER=1` logs the model-owned KV/staging/scratch bytes once after the first attention call.

### Compact sliding-window KV on CPU (issue #1169)

A sliding-window layer with window `W` only ever reads the last `W` positions, so the CPU caches keep only those rows.

- **Opt-in by geometry.** `KvGeometry.FromConfig(config, compactSlidingWindows: true)` marks each sliding-window slot with its
  window (`ModelConfig.SlidingWindowOf(layer)`: `PerLayerSlidingWindow`, else `SlidingWindowSize` with `SlidingWindowPattern`).
  The plain `FromConfig(config)` is unchanged. Callers pass `model.SupportsCompactSlidingKv`, which only `TransformerModel`
  returns true for; other CPU models and GPU upload paths (`HybridPrefillDecodeStrategy.KeysSpan`) read rows by absolute
  position and never see a compact cache. TextGenerator's default cache, `run` / `chat` / `serve` quantized factories, `bench`
  and `perplexity` all pass it.
- **Layout.** A windowed layer gets `2 * (W - 1 + 64)` rows. When an append would overflow, the last `W - 1 + 64` committed rows
  move to the front and the layer's row base advances (`SlidingWindowRows`). A chunk larger than the buffer grows it for that
  chunk; the next compaction that fits shrinks it back. Full layers are untouched.
- **Attention.** `IWindowedKvCache.RowBaseOf(layer)` gives the position in row 0. Callers pass `positionOffset - base` and
  `seqKv - base`: masks are relative, K is stored after RoPE and sinks have no position, so the kernels are unchanged and the
  output is bit-identical to a full cache. The quantized kernel applies the base itself; `TransformerModel.AttendCachedF32`
  does it for F32.
- **Formats.** `SimpleKvCache` (F32) and `QuantizedKvCache` in pure mode (`--cache-window 0`, Q8_0 / Q4_0). The FP32
  mixed-precision window mode keeps full rows (rule 1: do not grow FP32 windows). There is no F16 CPU KV yet (#1107).
- **Rollback and prefix reuse.** The 64-row margin lets speculative rollback and prefix truncation step back that far.
  `CanTruncateTo(length)` is false once a windowed layer has dropped rows a query at `length` needs; `Rollback` then throws and
  the prompt cache treats the hit as a miss. Truncating to 0 always works. `SupportsExactF32PrefixCopy` is false.
  `KvCacheCopy.CopyContents` copies a windowed layer from its row base.
- **Savings.** At 262K context Flint (35 of 49 layers windowed at 513) needs ~14 GiB F32 / ~3.7 GiB Q8_0 for its 14 full layers
  plus ~0.15 GiB / ~0.04 GiB for the windowed ones, against ~49 GiB / ~13 GiB with full rows everywhere.
- Tests: `SlidingWindowKvCacheTests` (bit-identical attention through compactions, an oversized chunk, rollback in and past the
  margin, F32 / Q8_0 / Q4_0). Real weights: `NemotronProbe swa` runs one stream through a full and a compact cache.

### Compact sliding-window KV on Vulkan (issue #1176)

The same ring on `VulkanKvCache` (F16 for Flint, F32 also works). `SlidingWindowRows` now lives in `DotLLM.Core.Attention` and
only plans (`TryPlan` / `Commit`); each backend carries out the move.

- **Opt-in.** `VulkanTransformerModel.SupportsCompactSlidingKv` is true for Flint on a single device only, and `CreateKvCache(n)`
  follows it; `CreateKvCache(n, compactSlidingWindows: false)` is the full-length A/B baseline (`NemotronProbe swa ... vulkan`,
  `STEPS_FULLKV=1` for `steps`). Gemma-2/3/4, gpt-oss and Mistral-SWA stay on full rows: they also reach the fused RoPE+KV write,
  the batched forward or Gemma-4 shared-KV donors, which address rows by absolute position. A pipeline stage keeps full rows.
- **Compaction on the device.** `RecordUpdate` records the move into the forward's command buffer before the F16 write. When the
  kept rows do not overlap their destination (every decode compaction on the default ring) it is one `vkCmdCopyBuffer` per K / V;
  otherwise (a prefill chunk crossing the end of the ring) the rows stage through a cache-owned scratch sized for the largest compaction, for
  K and V, since `vkCmdCopyBuffer` forbids overlapping regions. Barriers come from the hazard tracker, or a blanket
  compute/transfer barrier on the untracked path. No host round trip.
- **Oversized chunks grow the ring before recording.** A chunk longer than the ring's free half needs a bigger buffer. Growing
  swaps buffer handles, so `Forward` calls `EnsureWindowCapacity(positions)` before it records and drops the kernels' cached
  descriptor sets when anything grew (as on a scratch resize); the kept rows move with a one-off device copy. The ring is
  grow-only on Vulkan (shrinking would invalidate descriptors again), so it settles at `W - 1 + 64 + chunk` rows.
- **Bit-identical by alignment.** The sink prefill kernel tiles the KV range in 256-row tiles from row 0, so an arbitrary base
  changes where a window crosses a tile boundary and the online-softmax rounding with it (measured ~1e-2 on synthetic logits).
  Vulkan therefore keeps every base a multiple of 256 (`SlidingWindowRows(..., baseAlignment: 256)`): a compaction keeps
  `W - 1 + 64` to `W - 1 + 64 + 255` rows and the default ring is `2 * (W - 1 + 64) + 255` rows (1,407 for Flint). The decode
  and small-row kernels split relative to the window start and are shift-invariant anyway. Prefill now runs the GQA flash sink
  twins (#1200), whose 16 / 32-row tiles divide 256 and whose window-start tile skip is exact, so the ring stays bit-identical.
- **Attention.** The forward passes `positions[0] - base` / `CurrentLength - base`; the sink, split-KV and small-row kernels are
  unchanged. The fused RoPE+KV write and the host-staged paths (`IngestFromHost`, `DownloadLayer`) refuse a compact cache.
- **Rollback / truncation** behave as on the CPU (`CanTruncateTo`, `Rollback`, `SetCurrentLength`, `Reset`).
- **Bytes** (Flint, F16, 2 KiB per token per layer; 14 full layers + 35 rings of 1,407 rows): 4K 392 → 208 MiB,
  32K 3.06 → 0.97 GiB, 262K 24.5 → 7.09 GiB.
- Tests: `FlintVulkanCompactKvTests` (synthetic Flint, window 20: an overlapping in-ring compaction, a grown ring and direct decode
  compactions against the full-length F16 cache and the CPU compact cache; rollback within / past the margin; reset and reuse).

## Paged KV-Cache

Inspired by OS virtual memory paging. This is the **memory management** half of PagedAttention (vLLM): block-based allocation, ref counting, CoW. The **kernel** half (attention reading non-contiguous blocks directly) is the default on CPU since #1268 (the attention kernel reads F16/Q8_0 blocks in place, prefill and decode) and opt-in on CUDA as of issue #200 (`DOTLLM_ATTN_PAGED_NATIVE=1`, decode only — see the CUDA section below; #1270 tracks the CUDA follow-up). Any future Vulkan paged cache still needs it.

### Why

**Batch serving memory efficiency**: `SimpleKvCache` pre-allocates `maxSeqLen × layers` per sequence. With 10 concurrent requests averaging 200 tokens but `maxSeqLen=4096`, that's 40K tokens worth of buffers for 2K tokens of actual data — ~95% waste. A shared block pool allocates on demand and reclaims blocks on completion, keeping waste <4%.

**Foundation for downstream steps**: The ref counting and CoW infrastructure enables:
- **Step 37 (advanced prefix sharing)** — hard requirement. Cross-sequence block sharing (e.g., shared system prompt) is impossible without block tables and ref counting.
- **Step 43 (speculative decoding)** — benefits from cheap block-level rollback on draft rejection and CoW fork for draft branching, though `SimpleKvCache.SetCurrentLength()` truncation works as a fallback.
- **Beam search** (future) — beams share prefix blocks via `Fork`, diverge via CoW.

**When it doesn't help**: Single-sequence CLI inference. One sequence uses most of its allocation. This is why paged is opt-in for `run`/`chat` and default-on only for `serve`.

### Design
- Divide cache into fixed-size **blocks** of B tokens (B = 16).
- **Block table** per sequence: maps logical positions to physical blocks (page table).
- **Free pool**: blocks allocated on demand, returned on completion.
- Memory waste: <4% (vs ~60% for static pre-allocation).

### Operations
- **Allocate**: When sequence needs more blocks, pop from free pool.
- **Free**: On sequence completion/eviction, return all blocks to pool.
- **Copy-on-write**: For beam search — beams share prefix blocks (ref-counted). On divergence, copy the shared block.
- **Fork**: For prompt caching — new sequence references existing prefix blocks.

### Attention Integration (CPU: direct block reads, #1268)

`PagedKvCache` implements `IPagedKvCache` (Core): block size, uniform `KvStride`, the block table (`BlockIds`) and
per-layer views of the physical K/V block storage, whose `DType` is the row type. `TransformerModel.AttendCachedF32`
sends a paged cache to `Attention.Execute(float*, IPagedKvCache, ...)`, which reads the blocks in place: no gather and no
staging. It is the same kernel as the quantized-cache path (`ExecuteHeadGroup`), generalised over row addressing
(contiguous / ring / paged, a paged block resolved once per run of rows, not per row) and row type (F32 / F16 / Q8_0 /
Q4_0, one head slice decoded into a `headDim` scratch by the existing per-row helpers). It supports a custom score scale
(Gemma `query_pre_attn_scalar`, Granite), softcap, sinks and sliding windows, and partitions heads across workers with
`PartitionRange` like the contiguous path; up to 4 query heads sharing a KV head run together so each cached row is
decoded once for the group. Rows with ≤ 2048 visible keys take the contiguous path's single-shot softmax,
longer rows the same tiled online softmax, so **F32 blocks reproduce the contiguous kernel bit for bit** (#530; tested).

**Block row type** (`KvBlockPool` / `PagedKvCacheFactory` `elementType`): **Q8_0 when no KV type is specified** (#1284), F16 when one is. Q8_0 was faster than F16 at every
context in the #1268 decode benchmark (ctx 512: 172 vs 125 us; 4096: 1.12 vs 0.85 ms; 16384: 4.55 vs 3.18 ms) at about half the bytes, with
comparable logit drift. The default applies on the CPU paged path (`serve`, `run --paged`, `chat --paged`) for uniform non-hybrid models, no
mixed-precision window, head dim a multiple of 32; startup logs `INFO: KV cache: no type specified; using Q8_0 paged blocks ...`. Any other model
keeps F16 blocks (or its own cache). Explicit choices are honoured: `--cache-type-k f16 --cache-type-v f16` (or `f32`) gives F16 blocks, and
`q8_0` for both gives Q8_0 blocks. Mixed (only K or only V given) is unchanged: a lone `f16`/`f32` side gives F16 blocks, a lone `q8_0`/`q4_0`
side uses the quantized simple cache. F32 only as a reference for bit-identity tests. Bytes per token (K+V, all layers), Llama-3.2-1B (16 L × 8 KV heads × 64):

| rows | bytes/token | Llama-3.1-8B (32 L × 8 × 128) |
|---|---|---|
| F32 (before) | 64 KiB | 256 KiB |
| F16 (explicit `f16`) | 32 KiB | 128 KiB |
| Q8_0 (default, #1284) | 17 KiB | 68 KiB |

The old per-sequence F32 staging (`2 × maxSeqLen × kvStride × 4` bytes, allocated eagerly: 64 MiB for Llama-3.1-8B at 8k)
is gone from the attention path: `GetKeysRef`/`GetValuesRef` still exist for callers without a paged kernel (other
architectures, KV handoff) and decode into staging that is allocated **lazily**, on first use. `DownloadLayer` decodes
straight into the caller's buffers. `PagedKvCache.AllocatedBytes` is the bytes of the blocks this sequence holds plus any
staging (shared prefix blocks are counted by every holder); the pool's reservation is `KvBlockPool.AllocatedBytes`.

Decode attention, one token, 16 q / 4 KV heads, head dim 64, fragmented pages (`PagedAttentionBenchmarks`, 8 threads,
Zen5): before = F32 blocks gathered into staging + contiguous kernel; after = blocks read in place.

| ctx | before (F32 gather) | after F16 | after Q8_0 | after F32 |
|---|---|---|---|---|
| 512 | 256 µs | 172 µs | 125 µs | 93 µs |
| 4096 | 2.00 ms | 1.12 ms | 0.85 ms | 0.78 ms |
| 16384 | 10.7 ms | 4.55 ms | 3.18 ms | 4.81 ms |

F16/Q8_0 blocks change greedy output only at near-ties: on SmolLM-135M, teacher-forced logits differ from the F32 simple
cache by at most 0.94 (F16) / 0.90 (Q8_0), against 0.84 for the existing Q8_0 `QuantizedKvCache`, with top-1 agreement
14/15 for all three (`PagedKvCache_Blocks_TrackSimpleCacheLogits`).

### Key Classes

- `KvBlockPool` — Fixed-size pool of KV blocks. Per-layer contiguous storage, 64-byte aligned. Free list with `Interlocked` ref counting for thread safety.
- `KvBlockTable` — Per-sequence mapping from logical positions to physical block IDs. Supports `Fork` (shared prefix) and `EnsureWritable` (copy-on-write).
- `PagedKvCache : IPagedKvCache, IHostStagedKvCache` — Paged cache read in place by the CPU attention kernel; legacy contiguous views decode into lazy staging. Drop-in replacement for `SimpleKvCache`.
- `PagedKvCacheFactory` — Creates `PagedKvCache` instances backed by a shared `KvBlockPool`.

### CUDA Paged KV-Cache (issue #252)

GPU-resident mirror of the above, in `DotLLM.Cuda`:

- `CudaKvBlockPool` — Same `[totalBlocks, blockSize, kvStride]` shape as `KvBlockPool`, but block
  storage lives in device memory (FP16, matching `CudaKvCache`'s on-device element type) instead of
  host `NativeMemory`. Allocation/free-list/refcount bookkeeping stays host-side (a C# `lock` +
  `Interlocked`, identical to the CPU pool) — only block storage and copy-on-write duplication
  (`CopyBlock`) touch the device. Built around `KvGeometry` (per-layer stride) rather than a single
  scalar stride, matching `CudaKvCache`/`CudaQuantizedKvCache`'s existing generalization for
  Gemma-4's non-uniform layer shapes.
- `CudaKvBlockTable` — Per-sequence block table, same semantics as `KvBlockTable` (`Fork`,
  `EnsureWritable`/CoW, `SeedSharedBlocks`, `SnapshotFullBlocks`), except `EnsureWritable` drives a
  device-to-device async copy on a caller-supplied CUDA stream instead of `Buffer.MemoryCopy`.
- `CudaPagedKvCache : IKvCache, IPerLayerKvCache` — Device-pointer-only cache (mirrors
  `CudaKvCache`/`CudaQuantizedKvCache`'s convention: `Update(ITensor/TensorRef)`,
  `GetKeys`/`GetValues`, and `GetKeysRef`/`GetValuesRef` all throw `NotSupportedException`). Driven
  via `UpdateDevice` (block-boundary-batched D2D writes, allocating/CoW'ing blocks as needed) and
  `PrepareAttentionScratch` (gathers this layer's blocks into a contiguous FP16 scratch buffer via
  D2D copies — the GPU-side equivalent of `PagedKvCache`'s staging-buffer gather). The existing
  CUDA attention kernels are unmodified; they read the gathered scratch exactly like a plain
  `CudaKvCache`'s buffer.
- `CudaPagedKvCacheFactory` — Creates `CudaPagedKvCache` instances backed by a shared
  `CudaKvBlockPool`, mirroring `PagedKvCacheFactory`.
- Wired into `CudaTransformerModel` via `CreateKvBlockPool`/`CreatePagedKvCache`, and into `--paged`
  for the `run`/`chat`/`serve` CLI commands.
- `CudaTransformerModel.ForwardBatch` (issue #251) is a real, fused-launch-sequence override — no
  longer the `IModel` per-sequence-loop default — so `ContinuousBatchScheduler`-driven multi-sequence
  decode on CUDA runs through this cache's dispatch too, not just single-sequence `run`/`chat`.

**Issue #200 (opt-in, default OFF)**: a direct block-table-read decode attention kernel,
`attention_f16_paged`, now exists alongside the default gather-into-scratch path — see
`docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md` for the full design/implementation note. Set
`DOTLLM_ATTN_PAGED_NATIVE=1` to enable it. It reads K/V rows through a small per-layer array of
block base device pointers (`CudaPagedKvCache.PrepareNativeBlockPtrs`) instead of gathering KV
bytes into a contiguous scratch buffer first (`PrepareAttentionScratch`), eliminating that D2D copy
on the decode hot path. Restricted to decode (`seqLen == 1`); prefill still uses the gather-based
path (G3/flash prefill kernels need a contiguous buffer regardless). Default remains the
gather-based dispatch — this kernel is unmeasured on real hardware as of this writing (see the
design doc's "What's still unverified" section) and is deliberately opt-in per this project's
"don't ship an unvalidated kernel as the default" convention.

**Still not wired**: `ContinuousBatchScheduler`'s own paged-pool-aware preemption/prefix-trie logic
(`pagedPool:` parameter) is CPU-`KvBlockPool`-typed and does not drive `CudaPagedKvCache` — CUDA
requests still get scheduled without paged-cache-aware preemption, even though the cache itself is
now paged. This is a separate scheduler-integration gap, not part of #200's scope.

### CLI

```
# CLI (run/chat): paged is opt-in
dotllm chat model.gguf --paged

# Server (serve): paged is on by default, opt-out with --no-paged
dotllm serve model.gguf                 # paged KV-cache active
dotllm serve model.gguf --no-paged      # simple KV-cache

# API-only server (no web UI)
dotllm serve model.gguf --no-ui
```

### Limitations (v1)

- CUDA now has a paged KV-cache (issue #252, see above) via `CudaPagedKvCache`. Hybrid (CPU+GPU
  split-layer) models still fall back to their native KV-cache — `CudaKvBlockPool`'s per-layer
  shape assumes every layer's storage lives on one device, which doesn't fit the hybrid split.
- CPU: q8_0 K+V selects Q8_0 blocks (uniform, non-hybrid models, no mixed-precision window); any other quantized KV
  request falls back to `QuantizedKvCache` with a warning. CUDA: falls back to `CudaQuantizedKvCache`.
- CPU paged caches need a uniform per-layer KV stride: Gemma-4 / MiMo-V2 (per-layer KV widths) use the simple cache.
- The legacy `GetKeysRef`/`GetValuesRef` (CPU) staging results are only valid until the next call
  (shared buffer). CUDA's `CudaPagedKvCache` doesn't implement `GetKeysRef`/`GetValuesRef` at all —
  there's no single contiguous device pointer to hand back without a gather — callers must use
  `PrepareAttentionScratch(layerIndex, stream)` instead.
- CUDA paged KV-cache is not wired into `ContinuousBatchScheduler`'s paged-pool-aware preemption —
  see the CUDA section above.
- CUDA has an opt-in direct block-table-read decode attention kernel now (issue #200,
  `DOTLLM_ATTN_PAGED_NATIVE=1`, see the CUDA section above and
  `docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md`), but it defaults OFF pending real-hardware
  measurement, and it only covers CUDA decode (#1270). CPU's `PagedKvCache` reads blocks directly
  since #1268; a future Vulkan paged cache would need its own kernel (see `docs/ROADMAP.md`'s "Paged
  attention kernels" row).

## KV-Cache Quantization

Compress cached K/V to extend context capacity:

| Format | CPU (vs FP32) | GPU (vs FP16) | Quality Impact |
|--------|---------------|---------------|----------------|
| FP32 (CPU) / FP16 (GPU) | 1× (baseline) | 1× (baseline) | None |
| Q8_0 | 3.76× | 1.88× | Minimal |
| Q4_0 | 7.11× | 3.56× | Small (for older tokens) |

### Implementation

**Dual-region storage** with quantize-on-evict:
1. **Quantized buffer** (append-only): Q8_0/Q4_0 blocks for positions outside the window
2. **Full-precision ring buffer**: FP32 (CPU) or FP16 (GPU) for the most recent W tokens

On each new token write, the oldest window entry is quantized and appended to the quantized buffer. Separate key/value quantization types are supported (e.g., Q8_0 keys + Q4_0 values).

### Key Classes
- `QuantizedKvCache` — CPU implementation in `DotLLM.Engine`
- `CudaQuantizedKvCache` — GPU implementation in `DotLLM.Cuda`
- `IQuantizedKvCache` — Interface extending `IKvCache` for quantized access
- `KvCacheConfig { KeyDType, ValueDType, MixedPrecisionWindowSize }` — Configuration

### Attention Integration
- **CPU**: Per-tile dequantization inside tiled attention kernel (`Attention.ExecuteTiledQuantizedHead`). Phase 1 processes quantized region with on-the-fly dequant, Phase 2 reads FP32 window directly.
- **GPU**: Scratch-buffer approach — dequant quantized region + ring-ordered window copy into temporary FP16 buffer, then standard attention kernel.

### CLI
```
--cache-type-k q8_0    # key quantization (f32, q8_0, q4_0); synonym --kv-quant-k (also on bench)
--cache-type-v q4_0    # value quantization (f32, q8_0, q4_0); synonym --kv-quant-v (also on bench)
--cache-window 64      # recent tokens in full precision (0 = all quantized)
```

Orthogonal to weight quantization — Q4_K_M model can use Q8_0 KV-cache.

## MLA KV-Cache (DeepSeek-V2 / V3)

MLA decouples Q head-dim from V head-dim (V2-Lite: qk=192, v=128) and adds
a shared MQA-style rope-K that broadcasts across heads — neither fits the
per-head-uniform `IKvCache` shape that GQA/MHA caches assume. A dedicated
`MlaExpandedKvState` lives next to `TransformerModel` for this reason.

**Loader default**: HF and GGUF config extractors both set `MlaConfig.UseHybridMlaCache = true` for `Architecture.DeepSeekV2` / `Architecture.DeepSeekV3`, so production code paths get **Phase C** (hybrid latent + absorbed decode) without needing per-call configuration. Phase A remains active and is the numerical oracle; tests that build `MlaConfig` directly (bypassing the loader) still default to Phase A. The default flip lives at commits `4b54a72` (HF) and `4724397` (GGUF) — pre-flip, V2-Lite at `max_position_embeddings=163840` allocated ~68 GB and OOM'd on most hosts.

### Phase A — expanded reference cache

`src/DotLLM.Models/Architectures/MlaExpandedKvState.cs`. Per layer:

- `K_nope[layer]` : `[maxSeqLen, numHeads * qkNopeHeadDim]` — per-head non-rope K
- `V[layer]` : `[maxSeqLen, numHeads * vHeadDim]` — per-head V
- `KPe[layer]` : `[maxSeqLen, qkRopeHeadDim]` — shared rope-K (post-rotation)

All 64-byte aligned native memory, lazily constructed on the first MLA
forward and reset when the caller signals a fresh sequence by passing
`positions[0] == 0`. Not re-entrant; single-stream only (beam search or
batching needs per-sequence instances). Caller-supplied `IKvCache` is
ignored for MLA layers.

This layout is the PoC scalar kernel's scratch layout made persistent —
storage is 1:1 with what the kernel already computes, so zero shape
translation. Memory: ~16.6 KB per token per layer at F32 for V2-Lite;
on a 27-layer 8K context that's ~3.6 GB. The purpose of Phase A is
**correctness oracle**: generation works end-to-end and a split call
(prefill + step-by-step decode) produces logits that match a single-call
forward over the combined range within 1e-4.

### Phase B — pure latent + W_UK absorbed (landed)

The production memory win (per the DeepSeek-V2 paper, §2.1.2): store
the *compressed* latent `c_kv[kv_lora_rank]` per token (512 floats for
V2-Lite) alongside the shared `k_pe[qk_rope_head_dim]` (64 floats) —
a single `[kv_lora_rank + qk_rope_head_dim] = 576` value per token per
layer, 7.2× smaller than Phase A at F32. Attention math absorbs `W_UK`
into Q on-the-fly: `Q_latent[h] = Q_nope[h] @ W_UK_T[h]` (size
kv_lora_rank) and `score[h, t, s] = Q_latent[h] · c_kv[s] + Q_pe[h]
· k_pe[s]`. Output uses absorbed `W_UV`: `out[h] = W_UV[h] @ (softmax
· c_kv)`. vLLM's MLA backend is the reference implementation.

Lives at `src/DotLLM.Models/Architectures/MlaLatentKvState.cs` +
`src/DotLLM.Cpu/Kernels/MlaAttention.ExecuteLatent`. Selected via
`MlaConfig.UseLatentCache = true` (mutually exclusive with
`UseHybridMlaCache`).

### Phase C — hybrid: latent persistence + Phase A-equivalent prefill expand + absorbed decode (landed, default)

The production-shipping path mirrors vLLM's MLA backend: prefill
(`seqLen > 1`) expands cached latents through `W_UK` / `W_UV` into
local scratch and runs the standard 192-dim per-head MHA loop
(compute-bound at long seqKv); decode (`seqLen == 1`) delegates to
`ExecuteLatent` — the absorbed 576-dim MQA-style read of the compact
latent cache (bandwidth-bound at decode). Both paths persist the
SAME latent form (c_kv + k_pe per token) to `MlaLatentKvState` —
Phase A's expanded per-head K_nope/V is local prefill scratch and is
discarded. A decode step therefore consumes exactly the latents a
pure-Phase-B prefill would have written, so the absorbed kernel can
run over them without re-expansion.

Lives at `src/DotLLM.Cpu/Kernels/MlaAttention.ExecuteLatentHybrid`.
Selected via `MlaConfig.UseHybridMlaCache = true` (mutually
exclusive with `UseLatentCache`). **Default for DeepSeek-V2/V3 from
the loaders.**

Phase A is the numerical oracle for Phase B / Phase C: oracle tests
prove split-call match against the expanded-cache reference at 1e-3
drift on real-weight prompts.

## Simple Prompt Caching (Step 54)

Live KV-cache reuse for multi-turn conversations. No paged attention required — works with `SimpleKvCache`.

### Design: Live Reuse (not Snapshots)

After each generation, the KV-cache is transferred to a `PrefixCache` instead of being disposed. On the next call:

1. `PrefixCache.FindMatch(promptTokenIds)` scans entries (max 1–4) with element-wise prefix comparison (`MemoryExtensions.CommonPrefixLength`, SIMD-vectorized).
2. On hit: `SimpleKvCache.SetCurrentLength(matchedTokens)` truncates visible length. Suffix tokens are prefilled at positions `[matchedLen..promptLen)`.
3. On miss: fresh KV-cache allocated, full prefill as usual.

No data copying on cache hit. The same KV-cache object persists across calls.

### Key Classes

- `PrefixCache` — LRU cache with configurable max entries. Owns cached KV-cache instances.
- `PrefixCacheEntry` — Token sequence + live KV-cache + LRU timestamp.
- `SimpleKvCache.SetCurrentLength(int)` — Truncates visible length for prefix reuse.

### Multi-turn Chat Pattern

Each turn's prompt = previous prompt + assistant response + new user message. The stored token sequence (prompt + generated) shares a prefix with the new prompt. Typical cache hit rate: near 100%.

### CLI

```
--no-prompt-cache      # Disable (enabled by default in chat/serve)
--prompt-cache-size 1  # Max cached sessions (1 for chat, 4 for serve)
```

### Scope

- CPU `SimpleKvCache` only. QuantizedKvCache and GPU caches fall back to no caching.
- Cache cleared on model swap, `/clear` command (CLI), or `POST /v1/cache/clear` (server).

### Stats

`InferenceTimings.CachedTokenCount` reports how many prompt tokens were served from cache. Displayed in CLI output, API `timings.cached_tokens`, and Chat UI stats bar.

## Advanced Prompt Caching / Prefix Sharing (Step 37)

Cross-request prefix sharing on top of paged KV-cache. Shared system prompts
are computed once and reused by every subsequent request, regardless of which
sequence first filled the trie.

### Problem

Many requests share the same system prompt (e.g., all chat requests in a deployment).
Recomputing KV-cache for the shared prefix is wasteful.

### Solution: Prefix Trie (RadixAttention-style)

A block-granular radix trie of computed KV blocks. Each path from the root
corresponds to a sequence of physical blocks in the shared `KvBlockPool`; the
tokens that filled each block are the "label" on that edge.

- Walks one step per `BlockSize` (16) tokens of prompt.
- FNV-1a 64-bit hash keys the child map; full-token comparison guards against
  collisions (the trie reuses **full** blocks only).
- Per-node `RefCount` + `LastTouchedTicks` so zero-refcount LRU leaves are
  evictable on block-pool pressure.

### Key Classes

- `PrefixTrie` — the radix structure itself. `Lookup` / `Insert` /
  `Release` / `EvictOneLru` / `RegisterNamedPrefix` / `UnpinNamedPrefix`.
- `PrefixTrieManager` — single integration seam in front of `PagedKvCache`.
  Owns the trie + the paged-cache factory; mints per-sequence caches seeded
  with the longest matching prefix and routes completion back into the trie.
- `PrefixCacheConfig` — `Enabled` (default true when paged is active),
  `MaxPrefixDepth` (token cap, 0 = unbounded), `EvictionEnabled`.

### Refcount Lifecycle

For each block in the trie:

1. **Trie ref**: created when `Insert` links a freshly-computed block.
   Released when the node is evicted.
2. **Sequence ref**: created when `Lookup` matches a block (one per
   `Admit` call). Released when the sequence's `PagedKvCache.Dispose` runs.
3. **Named-prefix pin**: created when `RegisterNamedPrefix` pins a path.
   Released by `UnpinNamedPrefix`.

LRU eviction picks zero-refcount nodes (no active sequence, no pin); the
trie ref is then released and the block returns to the pool.

### Scheduler Integration

`ContinuousBatchScheduler` accepts an optional `PrefixTrieManager`. When
present, admission seeds a new sequence's cache from the trie (only the
suffix runs through `Forward`) and completion pushes new blocks back so
later requests reuse them. The admission loop calls `TryEvict` to relieve
block-pool pressure before refusing admission — active sequences are NOT
preempted (that's Step 59).

`TextGenerator` accepts an optional `PrefixTrieManager` too; it takes
precedence over the simple `PrefixCache` from Step 54. Both engine paths
(scheduler + single-session generator) benefit from the trie.

### Server API

- `GET /v1/prompt-cache` — stats for the active trie.
- `GET /v1/prompt-cache/{id}` — inspect a named prefix.
- `POST /v1/prompt-cache/{id}` — pre-warm + pin a named prefix
  (body: `{ "prompt": "..." }` or `{ "token_ids": [...] }`).
- `DELETE /v1/prompt-cache/{id}` — unpin a named prefix.
- `DELETE /v1/prompt-cache` — clear the entire trie.
- `prefix_id` field on chat/completion requests — best-effort hint that
  validates the named prefix exists before the request runs.

### Limitations (Step 37 MVP)

- Block-level (16-token) match granularity — partial trailing blocks
  are NOT shared.
- Paged KV-cache only (CPU). GPU caches fall back to the per-sequence
  Step 54 behaviour. Quantized KV-cache fall back too.
- Eviction only frees zero-refcount blocks. Preempting active sequences
  to make room is deferred to Step 59 (priority-based scheduling).
- The trie walks linearly to find a node by block id during `Release`;
  good enough for typical depths (< 1024 blocks) but a future tweak.

### Snapshot lineage: exact vs approximate rows (chunk-cache L7)

Chunk-level KV reuse (#1542) will put **approximate** K/V into a sequence: rows assembled from
position-independent chunk entries, or selectively recomputed. A prefix snapshot taken afterwards must
never be restored by a request that did not ask for approximation, and must never be reported as
`cached_tokens`. Both prefix stores carry the lineage; nothing in the engine creates approximate rows
yet, so every existing caller sees unchanged behaviour (all rows exact, all defaults).

- **Rule.** A snapshot is approximate from the first row whose K/V came from assembly or recompute;
  every later row attended to it and is approximate too. Rows before it are exact.
- **`PrefixCache` (Step 54).** `PrefixCacheEntry.FirstApproxRow` (`int?`, `null` = all exact).
  `FindMatch(prompt)` and `FindMatch(prompt, allowApprox: false, out _)` cap the match at
  `FirstApproxRow` (and pick the best entry by the capped length). Only with `allowApprox: true` does
  `FindMatch(prompt, allowApprox, out approxRowsReused)` restore the full match, reporting the approximate rows
  separately; on equal match length it prefers the entry with more exact rows.
  `Store(tokens, cache, firstApproxRow)` sets the tag; on the live-reuse path it **replaces** the old tag,
  so an exact recompute after an approximate snapshot is stored exact. A caller that restored approximate
  rows passes the earlier of the restored entry's `FirstApproxRow` and its own first approximate row.
- **`PrefixTrie` / `PrefixTrieManager` (Step 37).** Each node has `IsApprox`; a block holding any
  approximate row is approximate (mid-block rows taint their block). Approximate nodes sit in the child
  map under a salted key, so an exact and an approximate node for the same tokens coexist as siblings.
  `Admit(prompt, maxSeqLen, allowApprox = false)`: an exact lookup never enters an approximate node; with
  `allowApprox` the walk prefers the exact child and falls back to the approximate one.
  `RecordCompletion(cache, tokens, firstApproxRow = null)` inserts blocks from the earlier of the first
  approximate block restored at admission and `firstApproxRow / BlockSize` as approximate.
  `RegisterNamedPrefix` pins exact paths only.
- **Counting.** `PrefixAdmission.CachedTokens` (→ `cached_tokens` / `cache_n`) and `PrefixTrie.HitTokens`
  count exact rows only. Approximate restored rows are `PrefixAdmission.ApproxRowsReused` (→
  `chunk_cache.tokens.prefix_reused_approx`). Prefill starts at `PrefixAdmission.RestoredTokens`
  (= `CachedTokens + ApproxRowsReused`; equal to `CachedTokens` without `allowApprox`). With approximate
  restores, `HitTokens + MissTokens` is less than the prompt length: approximate rows are in neither.
- **Limitations.** An approximate trie node is not upgraded in place when an exact recompute arrives;
  the exact sibling is linked beside it and the approximate one ages out by LRU. With `allowApprox` the walk
  is greedy exact-first and never backtracks, so it can stop on a shorter exact branch when the approximate
  sibling's subtree is longer (lost reuse, not a correctness issue). Recurrent prefix
  snapshots (`RecurrentPrefixSnapshotCache`) carry no tag: chunk reuse rejects recurrent and hybrid
  models (`chunk_model_unsupported`).

### Bonsai-2 quality gate and recommended format (#973)

RTX 3060 12 GB, Bonsai-2 27B PQ2_0, flash prefill (#962), v2 decode + fused quantized decode (#988).
Perplexity: `DOTLLM_HYBRID_KV=<fmt> dotllm perplexity <gguf> --device cuda --kv-cache f32 --context 512 --max-tokens 8192`
(wikitext-2 test; without `--kv-cache` the CUDA hybrid-dense model scores cacheless over exact F32 K/V and is blind to KV format).
Needles: 3 secrets at 10/50/90% depth, greedy; token counts are 18,011 / 35,888 / 71,811.

| KV | PPL | vs F16 | needles 18k / 36k / 72k | peak VRAM 36k / 72k | decode tok/s @72k |
|---|---:|---:|---|---|---:|
| F16 | 10.2350 | - | 3/3, 3/3, does not fit | 11.7 GB / - | - |
| Q8_0 | 10.2346 | -0.004% | 3/3, 3/3, 3/3 | 10.7 GB / **12.08 GB** | 18.1 |
| K8V4 | 10.2529 | +0.18% | 3/3, 3/3, 3/3 | 10.5 GB / 11.5 GB | 17.1 |
| Q4_0 | 10.2661 | +0.32% | 3/3, 3/3, 3/3 | 10.2 GB / 10.95 GB | 18.8 |

Decode sweep (Q4_0, 64 tokens per point): 18.9 tok/s at ctx 512, 19.5 at 4k, 19.7 at 16k, 18.9 at 32k, 18.4 at 64k.

**Recommendation (12 GB card):** Q8_0 up to ~36k (indistinguishable from F16, 1 GB headroom). Beyond that
prefer K8V4 (+0.18% PPL, 11.5 GB at 72k) or Q4_0 (10.95 GB) - Q8_0 at 72k peaks at 12.08 GB of 12.29 and is
one allocation from WDDM paging. Needle retrieval is a coarse test; the PPL deltas are the fine-grained signal.
# Recurrent prefix snapshots

## LFM2 convolution checkpoints and sequence prefixes (#1028)

CPU and Vulkan LFM2 models implement model-owned recurrent checkpoint/restore and
independent per-sequence prefix snapshots. A prefix snapshot owns both the attention
KV prefix and the short-convolution history; restoring it into another sequence leaves
the KV cache at the captured length. The caller must capture recurrent state immediately
after exactly that prefix has been consumed. Both backends support interleaved sequences
through serialized `ForwardBatch` calls; this does not make simultaneous calls on one
model instance thread-safe.

The initial CPU path supports `SimpleKvCache` host F32 storage. The
`IKvCache.SupportsExactF32PrefixCopy` capability opts a cache into complete, lossless
F32 copying with explicit per-layer geometry. Compressed and paged caches have not
opted in: capture returns null and restore rejects them before changing the destination.
Vulkan snapshots retain F32 KV and convolution buffers on the same device. Snapshot
ownership, state/cache geometry, device identity and disposed resources are validated.
Callers own and dispose snapshots; they can restore an intact snapshot repeatedly.

These APIs do not implement per-row recurrent snapshots, add an LFM MTP head, or add
the host/disk transport described below to LFM models.

## Host tier for models with snapshot transport (#1033)

Recurrent and hybrid models need both attention KV rows and recurrent state to reuse a
conversation prefix. Retaining only KV rows is insufficient. Multi-slot snapshots keep
independent copies of both, match the longest token prefix, and evict least-recently-used
entries when the configured entry or memory limits are reached. A request still forwards
at least one suffix token to produce its next-token logits.

`--prefix-snapshots 20 --prefix-snapshot-host-budget 8GiB` enables bounded multi-session
retention on models with snapshot transport support. The default remains one snapshot and
no host tier. The host budget counts snapshot buffers, including recurrent state; it does
not include model weights or the active device KV cache. Pinned host memory cannot be
paged out, so configure a budget appropriate to the machine's available RAM.

The active snapshot uses device storage when admitted by the backend's memory policy.
Cold snapshots can use page-locked host memory and restore their exact stored bytes instead
of prefilling the conversation again. A model/configuration change invalidates snapshots;
failed restores are discarded as misses. Transfers and generation remain serialized around
the live state. `DELETE /v1/prompt-cache` clears recurrent snapshots as well as the attention
prefix trie.

For a representative F16 layout, 16 attention layers × two K/V tensors × 1,024 elements ×
two bytes gives 64 KiB per token. Use the loaded model's reported KV geometry and
snapshot byte counts for budgeting. That layout needs about 1.83 GiB for
30,000 tokens, plus recurrent state (about 150 MiB in the issue's example), so an 8 GiB
host budget holds roughly four such snapshots. Quantized KV retains its raw packed format
and reduces this storage proportionally to the actual format block size.

Streaming timing extensions expose `cached_tokens`, `cache_tier` and `cache_restore_ms`.
`GET /v1/prompt-cache` includes recurrent occupancy, host bytes and tier-specific hit
counters even when no paged prefix trie is active. The NVMe spill tier is tracked separately
in #1034; configuring the host tier does not enable disk persistence.

CUDA transport retains raw F16, Q8 or Q4 KV rows and independent convolution/GDN state;
it does not expand quantized rows to F32. Device admission reserves at least 512 MiB of
free VRAM beyond the snapshot allocation. Multi-slot mode captures a first-seen prompt
at its penultimate token so alternating conversations can reuse their first visit.
Multi-slot hits leave MTP disabled, because the snapshot copies only the trunk, and rerun a suffix token to regenerate logits.
The default single slot keeps the MTP head with its snapshot, so its hits do speculate (#1088, below).
Snapshot ownership is tied to the model instance and forward settings. Entries do not
survive a process restart; persistent storage belongs to #1034.

The initial independent CUDA transport covers the Qwen3 hybrid dense model family,
including Bonsai. Models without the snapshot-provider capability retain their existing
single-slot behavior even when a larger entry count is requested; these flags do not add
transport support to another backend or architecture.

## MTP on a single-slot prefix hit (#1088)

A recurrent-prefix hit skips the cached prefix's forward, and the MTP head drafts from its own KV, which must cover the
whole sequence (#469). Before #1088 a hit therefore decoded without speculation. The single-slot snapshot shares the
live KV handle rather than copying it, and the head now gets the same treatment:

- **What the snapshot keeps.** The request's head state (`IMtpState`) stays alive with the entry, plus a mark taken at
  the snapshot boundary (`IMtpState.MarkPrefix`): the head's length and the trunk hidden row its next absorb pairs with.
  The head's rows below the mark stay valid for the same reason the trunk's KV rows do: every later write is at or
  beyond the boundary.
- **What a hit does.** It rolls the head back to the mark with the trunk (`IMtpState.TryResumeAt`), raising the head's
  ceiling to the trunk KV's length when the hit grew it. The suffix prefill then absorbs from the mark exactly as a cold
  run would. The draft logits match a cold run's: bit for bit on the CPU, and within 1.2e-7 on CUDA.
- **Plain turns keep the head in step while MTP may be used.** A request that takes the snapshot carries the head
  through its prefill even when it decodes plain (the adaptive gate picked plain, or logprobs were requested), as long as
  the gate forecasts MTP for a coming request (`MtpAdaptiveGate.LikelyUsesMtpNext`: an arm still unexplored, MTP the
  measured winner, or a re-probe due). Otherwise a plain first turn would leave every later hit without a head, and the
  gate could never try MTP. Once plain decode measures faster, plain turns stop feeding the head and the session's later
  hits stay plain; the next miss starts a fresh head.
- **Why not copy or recompute it.** Copying the head's F32 KV into the snapshot costs about 8 KiB/token of extra memory
  and a copy on every turn, and the single slot already shares its storage. Recomputing it needs the trunk's hidden
  state at every cached position, which is the cost of the prefill the hit exists to skip.
- **What it costs.** Memory: the head's KV now lives between requests and during hit turns, which the context fit
  already counts when MTP is on. Compute: a turn that feeds the head pays its absorb in the prefill, so in time to first
  token. On Bonsai-2 27B on an RTX 3060 the CUDA absorb measured about 0.2 ms per prompt token (3% of the trunk's
  prefill, 0.7-1.0 s on a 4k-token agent turn); hence the forecast above.
- **Limits.** The multi-slot tiers (`--prefix-snapshots` > 1) copy only the trunk, so their hits stay plain. Backends
  whose head state cannot mark (the Vulkan states) keep no head, so their hits stay plain too, and their plain turns
  never feed one (the first head made shows the states cannot resume).

With no snapshot held, a prompt that shares nothing usable with the previous one (the first turn of a conversation)
now snapshots at its own end. The second agent turn then hits instead of re-prefilling the first.

## Mid-history resume: step checkpoints (#1286)

An agent that trims an old tool result resends a prompt that diverges from the cached history in the middle. The
single-slot snapshot sits at the end of the last prompt, so before #1286 that was a miss and a full re-prefill.

`--prompt-cache-step N` (chat and serve; `RecurrentPrefixCacheOptions.StepTokens`) checkpoints the recurrent state every
N prefill tokens, at absolute positions that are multiples of N. Idea from azllm's `--prompt-cache-step`.

- **Resume.** A new request finds where it diverges from the cached history and resumes from the deepest checkpoint at
  or before that point. It rolls the KV back there and re-prefills only the rest.
- **KV is shared.** Attention KV is per token, so every checkpoint along one history shares the single slot's KV
  handle. A step copies only the recurrent (GDN/conv) state, in its own dtype
  (`IModel.RecurrentStateCheckpointBytes`).
- **Invalidation.** Checkpoints past the resume point are dropped, because the request overwrites those rows.
- **Bounds.** The newest (deepest) `--prompt-cache-step-keep K` steps are kept, default 8; `StepBudgetBytes` adds an
  optional byte cap. When steps are on, the boundary a hit moves past is kept as a step too, so a retried turn (#1090)
  resumes there.
- **Default off.** Each step is a full recurrent-state copy, which needs sign-off under the avoid-state-copies rule.
  Bonsai-2 27B holds about 150 MiB of state per checkpoint, so eight steps cost about 1.2 GiB (in VRAM on CUDA/Vulkan).
  Enable it where agent traces trim history.
- **MTP.** A mid-history resume decodes plain, because the MTP head is marked only at the snapshot boundary.
- **Scope.** Steps live on the single-slot path only (CPU, and CUDA/Vulkan hosts with `CheckpointRecurrentState`). The
  tiered cache (`--prefix-snapshots` > 1) and the continuous-batch scheduler copy KV rows into every snapshot, so
  periodic snapshots there would copy the history again and again. They wait for KV rows shared with the snapshot
  (#1111). A model that cannot size its checkpoints takes no steps.

Measured on CPU (Bonsai-2 2B PQ2_0; `HybridMidHistoryResumeProbe`, `DOTLLM_HYBRID_RESUME_PROBE=1`), step 512:

- **Trace.** A 3,523-token agent trace. The second request trims the middle tool result, giving 2,392 tokens that share
  1,222 with the first.
- **Resume and output.** The resumed request restored the step at 1,024 and re-prefilled 1,368 tokens. Its greedy output
  is identical to a cold prefill.
- **TTFT.** 127 s resumed against 467 s cold. Absolute times are inflated by other workloads on the box, so read the ratio,
  not the seconds.
- **Size.** One checkpoint is 19.3 MiB (F32 GDN + conv state); the six steps held cost 115.6 MiB.

## Agent-aware prefix policy (#1654)

Agent harnesses (Claude Code, Codex) interleave one long conversation with many short requests: subagents that repeat the
project instructions, title or summary side requests over the same history, and unrelated quick calls. Every prefix cache
above was LRU, and a hit on the live-KV `PrefixCache` or the single recurrent slot *truncates or replaces* the entry it hits.
So a 3k-token subagent that shared only the system prompt destroyed a 100k-token conversation for 2k tokens of saving,
and the conversation's next turn re-prefilled everything. Idea from the halogen-flash review (`.docs/HALOGEN_FLASH_REVIEW.md`
T6; ideas only, its engine is closed). Opt-in: `--prompt-cache-policy agent` (`DOTLLM_PREFIX_POLICY=agent`); default `lru`
keeps the pre-#1654 behaviour exactly. The policy decides only what is *reused*, never what is computed: output is unchanged.

One rule, applied by every cache through `PrefixRetention` (`src/DotLLM.Engine/PromptCache/PrefixCachePolicy.cs`):

- **Token clock.** Each cache counts the prompt tokens it has been asked about (hit or miss). An entry's *age* is the tokens
  read since its last use divided by its own length. At age >= 1 it has **expired**: about its own length of other prompt
  tokens has been read since it was last used.
- **Own-length eviction.** When a cache is over capacity it gives up the entry with the largest age (LRU breaks ties), not the
  least recently used. A 100k conversation used a moment ago (age 0.09 after 9k of side traffic) outlives a 3k side request
  stored after it (age 2). Applied to: `PrefixCache` (live KV: Vulkan, CUDA, `--no-paged`), `PrefixTrie` zero-ref leaf eviction
  (paged CPU; a leaf's length is its path in tokens, so long conversations shorten from the tail last),
  `RecurrentPrefixSnapshotCache` (`--prefix-snapshots` tiers, all four removal sites) and the scheduler's recurrent entries.
- **Protected conversations.** A request that would *destroy* an entry (shares only part of it: a hit would truncate the rows
  past the share, a miss would replace the slot) and is **not the same conversation** runs cold, on its own KV, without a
  snapshot and without touching the entry, while the entry is unexpired. Counted as `protected_misses` (`/v1/prompt-cache`,
  `dotllm.engine.prefix_cache.lookups{result="protected"}`). Once the entry has expired the request takes over as before, so
  equal-sized peers (a classifier's questions) still churn exactly as under LRU. Applied to `PrefixCache` and the single
  recurrent slot; the tiered and trie caches never destroy an entry on a hit, so they only need the eviction rule.
- **Save point = conversation identity.** The server passes `InferenceOptions.SavePointPrefix`: the prompt up to the start of
  the last *non-assistant* message's content, a user message or the tool result an agent loop ends with (derived from a marked
  re-render, because on Qwen3 templates a head render is not a prefix of the full one). With a system prompt and one user
  message the save point is the end of the system prompt, so any request with the same system prompt may destroy a one-turn
  entry: by design, a one-turn entry is cheap to rebuild and the shared prefix is what both want. A later request that shares everything before that message is the **same conversation** (a retry, an
  edited or compacted last message, or a side request over the same history) and may resume destructively; one that shares
  less is another conversation and is protected. Without the hint (raw completions), sharing at least half of the entry stands
  in. The single slot also **checkpoints the recurrent state at the save point** (a step checkpoint: state bytes only, KV
  shared), so the same-conversation branch resumes there instead of re-prefilling the history (halogen's SNAP3).
- **End-of-reply checkpoint.** After a request on the slot's KV finishes, the single slot checkpoints the state at the end of
  the reply (the rows the KV holds: prompt + every forwarded generated token, at least `RecurrentPrefixMinTokens` past the
  boundary). The next turn repeats the reply, so it resumes after it rather than before it. Skipped when the entry keeps a
  resumable MTP head (a resume past the head's mark would decode plain) and when the client's re-tokenisation of the reply
  diverges (the prompt-end boundary is still there, so that case costs nothing extra). Both checkpoints share the step budget
  (`--prompt-cache-step-keep`, default 8) and need no `--prompt-cache-step`.
- **Not implemented, deliberately.** Cached last logits for an exact repeat (halogen caches them so a retry costs zero
  prefill): one row of F32 logits per entry is FP32 storage (`docs/PRECISION.md` rule 1), an F16 copy could flip greedy near
  ties, and the saving is one decode-step forward per exact repeat. The exact-repeat path re-runs the last prompt token as
  before. A "lead" entry for the tiered cache (a second capture at the shared system-prompt boundary) is also left for a
  follow-up: it is a KV copy per capture there.

What to expect (synthetic session in `AgentPrefixPolicyReplayTests`: 6 main turns, two 2-turn subagents sharing the project
instructions, 6 unrelated requests; token counts, so contention-proof): under LRU the main conversation's first turn after
each side block reuses only the shared system prompt; under the agent policy it reuses the whole previous turn, and the
subagents' second turns pay for it on the single slot (they re-prefill their own first turn, since the slot is held). The
tiered cache (`--prefix-snapshots N`) is where both win. Numbers are in PR for #1654.

Where it meets the chunk cache (#1542, `DotLLM.ChunkReuse`): untouched. Chunk reuse decides which rows can be *assembled*
(lineage, `FirstApproxRow`); this policy decides which whole entries are *kept* and which requests may truncate them. An
assembled entry carries its lineage through `PrefixCache.Store` unchanged.

## Disk spill below the host tier (#1034)

Use all three limits to retain more conversations than fit in pinned memory:

```shell
dotllm serve --model MODEL --device gpu --prefix-snapshots 20 --prefix-snapshot-host-budget 8GiB --prefix-snapshot-disk-dir ./prefix-cache --prefix-snapshot-disk-budget 40GiB
```

The disk directory and a positive disk budget must be supplied together. Cold snapshots
spill as raw packed KV plus recurrent state; their token sequences remain indexed in
memory, so prefix matching performs no disk reads. Successful reloads report
`cache_tier: disk`. Recurrent statistics include `disk_entries`, `disk_bytes`,
`disk_budget_bytes`, `disk_hits` and the implemented `disk_transport`.

The current transport is `file-stream`: overlapped file handles with provider-owned
pinned CUDA transfers. Native DirectStorage/CUDA external-memory and cuFile integration
remain unavailable in this implementation. Startup reports the selected disk transport
or that disk caching is unavailable for the provider or directory. This path does not
claim a CPU-free GPU read.

Disk accounting includes metadata and temporary writes. Leave enough disk budget for
an atomic spill while a protected reload is in progress; a capacity calculation based
only on final snapshot payloads can require additional evictions. Host accounting also
reserves incoming snapshot bytes before import. Corrupt, truncated, mismatched or failed
reads become cache misses and trigger ordinary prefill.

Each cache owns a private session subdirectory. Files are written to temporary names,
flushed and atomically renamed; checksums and bounded metadata are validated before
state import. Clear, eviction and model changes remove owned files. Restart persistence
is disabled: a new process creates a new session directory and does not load old files.
An abrupt process termination can leave an orphaned session directory; after confirming
its process has stopped, remove that directory manually. Live-instance budgets do not
include orphaned files from earlier processes.
