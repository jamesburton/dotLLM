# KV-Cache — dotLLM

## Purpose

The KV-cache stores previously computed key and value vectors for all layers, avoiding O(n²) recomputation during autoregressive generation. At each decode step, only the new token's K/V are computed and appended.

## Memory Consumption

Llama 3 8B, FP16, 2048 tokens:
```
2 (K+V) × 32 layers × 8 KV heads × 128 head_dim × 2048 tokens × 2 bytes
= ~1 GB
```
Scales linearly with sequence length and batch size. Dominant memory consumer in production.

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
- **Cost.** F32 storage (the default; F16 halves it, next section): 128 KiB per token for Bonsai-2 (16 attention layers x 4 KV heads x 256 x K and V x 4 B), so
  16 GiB at 128k. A growth step is 25 % of what is held (at least 1,024 positions), plus one layer's K and V in transit.

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
- **Accuracy.** F16 rounds each K / V element once (relative error ≤ 2⁻¹¹). On the synthetic fixture the logits stay
  within the tolerance the Vulkan forward is held to against the CPU reference (abs 5e-3 + rel 1e-3).

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
  reads whole blocks per quad and holds 18.5 tok/s flat from ctx 512 to 16k; `DOTLLM_HYBRID_KV_FUSED=0` opts out) or
  by a dequant into one shared F16 scratch pair (used by prefill, the fallback tiers, and decode when fused is off).
- **KV Q4_0 is not the GGUF Q4_0 layout.** KV blocks pack ADJACENT PAIRS (low nibble = even element), matching CPU
  `KvQuantize` and `quant_kv.cu`; the weight dequant (`dequant_q4_0_f16`) unpacks low/high HALVES. Always use
  `CudaKernels.LaunchKvDequantToF16` for KV data. (The eager `CudaQuantizedKvCache` path used the weight dequant until
  #956, permuting elements inside every 32-block.)
- **CLI selection (issue #967).** `run`, `bench` and `serve` accept `--kv-quant-k` / `--kv-quant-v` (`f16`, `q8_0`, `q4_0`);
  on `run` and `serve` they are synonyms of `--cache-type-k/-v` (giving both with different values is an error).
  `dotllm run --device gpu --kv-quant-k q8_0 --kv-quant-v q4_0` selects K8/V4. Precedence is **per side**:
  explicit flag, then `DOTLLM_HYBRID_KV`, then F16 (so `--kv-quant-k q4_0` with `DOTLLM_HYBRID_KV=q8_0` gives K4/V8, and
  `--kv-quant-k f16 --kv-quant-v f16` forces unquantized over the env var). `serve`'s `--cache-type-*` default is `f32`,
  which cannot be told apart from "not given" and so leaves the env var in charge; pass `f16` to force unquantized.
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

## Paged KV-Cache

Inspired by OS virtual memory paging. This is the **memory management** half of PagedAttention (vLLM): block-based allocation, ref counting, CoW. The **kernel** half (attention reading non-contiguous blocks directly) is opt-in on CUDA as of issue #200 (`DOTLLM_ATTN_PAGED_NATIVE=1`, decode only — see the CUDA section below); it remains a future step for CPU and any future Vulkan paged cache, where current kernels still see contiguous buffers via staging-buffer gather.

### Why

**Batch serving memory efficiency**: `SimpleKvCache` pre-allocates `maxSeqLen × layers` per sequence. With 10 concurrent requests averaging 200 tokens but `maxSeqLen=4096`, that's 40K tokens worth of buffers for 2K tokens of actual data — ~95% waste. A shared block pool allocates on demand and reclaims blocks on completion, keeping waste <4%.

**Foundation for downstream steps**: The ref counting and CoW infrastructure enables:
- **Step 37 (advanced prefix sharing)** — hard requirement. Cross-sequence block sharing (e.g., shared system prompt) is impossible without block tables and ref counting.
- **Step 43 (speculative decoding)** — benefits from cheap block-level rollback on draft rejection and CoW fork for draft branching, though `SimpleKvCache.SetCurrentLength()` truncation works as a fallback.
- **Beam search** (future) — beams share prefix blocks via `Fork`, diverge via CoW.

**When it doesn't help**: Single-sequence CLI inference. One sequence uses most of its allocation, and the staging-buffer gather adds overhead. This is why paged is opt-in for `run`/`chat` and default-on only for `serve`.

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

### Attention Integration (v1: Staging Buffer)

Current implementation uses a staging buffer approach: `GetKeysRef()`/`GetValuesRef()` gather block data into a pre-allocated contiguous buffer before each attention call. This avoids modifying attention kernels. The gather cost is O(seqLen × headDim) — negligible compared to attention's O(seqLen² × headDim).

Future optimization: specialized paged attention kernels that read blocks directly via block table indirection, eliminating the staging copy.

### Key Classes

- `KvBlockPool` — Fixed-size pool of KV blocks. Per-layer contiguous storage, 64-byte aligned. Free list with `Interlocked` ref counting for thread safety.
- `KvBlockTable` — Per-sequence mapping from logical positions to physical block IDs. Supports `Fork` (shared prefix) and `EnsureWritable` (copy-on-write).
- `PagedKvCache : IKvCache` — Paged cache with staging-buffer gather. Drop-in replacement for `SimpleKvCache`.
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
- Not compatible with quantized KV-cache (`--cache-type-k`/`--cache-type-v`) on either backend.
  Falls back to `QuantizedKvCache`/`CudaQuantizedKvCache` with a warning.
- Staging buffer means `GetKeysRef`/`GetValuesRef` (CPU) results are only valid until the next call
  (shared buffer). CUDA's `CudaPagedKvCache` doesn't implement `GetKeysRef`/`GetValuesRef` at all —
  there's no single contiguous device pointer to hand back without a gather — callers must use
  `PrepareAttentionScratch(layerIndex, stream)` instead.
- CUDA paged KV-cache is not wired into `ContinuousBatchScheduler`'s paged-pool-aware preemption —
  see the CUDA section above.
- CUDA has an opt-in direct block-table-read decode attention kernel now (issue #200,
  `DOTLLM_ATTN_PAGED_NATIVE=1`, see the CUDA section above and
  `docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md`), but it defaults OFF pending real-hardware
  measurement, and it only covers CUDA decode — CPU's `PagedKvCache` and any future Vulkan paged
  cache still only have the staging-buffer-gather shape (see `docs/ROADMAP.md`'s "Paged attention
  kernels" row for the remaining backends/prefill scope).

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
