# Vulkan paged KV-cache with in-place paged attention — design

**Epic:** #1724. **Slices:** #1725 (S1) to #1730 (S6) and #1733 (S7), section 6.
**Owner decisions (2026-11-09) are folded in; see section 8 (RESOLVED / provisional).** First hosts are the Qwen3.5/3.6 hybrids (Bonsai-2 style), not the dense host.
**Status:** design only (2026-11-09). No code, no measurements in this document: every number is either read
from the repository (with the file) or labelled *unmeasured*.
**Scope rule:** docs only. Nothing here was built or run; a benchmark was using the GPU while this was written.

## 0. Summary of decisions

| # | Decision | Why (section) |
|---|---|---|
| D1 | Per-layer K slab and V slab, addressed by **block id** (same ids in every layer). No pointers, no per-block buffers. | 3.1 |
| D2 | **One block-id table per sequence, shared by all layers**, in a single SSBO arena; updated inside the command buffer, and only when a block is added, replaced (CoW) or released. | 3.3 |
| D3 | Kernel change is **address-only** (`row -> physRow`). Arithmetic and reduction order untouched, so the oracle is **bit-exact** against the contiguous F16 cache. | 3.7, 4 |
| D4 | **No gather/expansion fallback.** A host or cache type without an in-place paged kernel is refused with a named reason and keeps its own cache. | 3.8 |
| D5 | Block size `B = 64` tokens by default (power of two, specialization constant), chosen by a decode sweep in S3. | 3.2 |
| D6 | v1 element type is **F16** (default, owner decision; the Vulkan default since #1151/#1102). Q8_0 is opt-in, uses the Flint 36-byte interleaved layout, later (S7). F32 only as a test oracle. | 3.6 |
| D7 | The pool grows by **realloc with stable block ids** at forward boundaries (the #1100 contract), capped per slab by the device allocation limit. No chunked multi-binding in v1. | 3.1 |
| D8 | Scheduler admission and the prefix trie move onto a backend-neutral allocator interface (S1), which also closes the same gap for CUDA. | 3.9 |
| D9 | **First hosts: the Qwen3.5/3.6 hybrids** (`VulkanQwen3HybridDenseTransformerModel`, hd256, G = 6, F16 KV). Pool/cache, decode and prefill kernels target their attention layers first; trie hits on hybrids only at block boundaries that also have a recurrent checkpoint (#1286). Dense host, other hybrids, Q8_0 and sliding windows follow. | 3.5, 6 |

"Parity with CUDA" is **not** the target. It is a floor that is too low (section 1).

## 1. Problem, and what "CUDA parity" actually is today

Facts, each read from the tree:

- `dotllm serve` has `UsePaged = true` by default (`ServerOptions.cs:179`, `--no-paged` opts out). On Vulkan
  `ServerStartup.cs:~580` prints `Paged KV-cache not supported on Vulkan; using the model's own KV-cache.` and the
  Vulkan branch has no pool, no trie and (for the dense host) no scheduler.
- Vulkan prefix reuse is whole-`IKvCache` retention (`IPrefixReusableKvCache`, #1591): at most two retained caches,
  adoption only when the cache has exclusive storage, `prefixCacheHeadroomTokens = 1024`, no sharing between
  concurrent requests. Block-granular sharing is #1595.
- Every Vulkan request creates fresh KV buffers sized by `maxSeqLen` (lazy only on the hybrid hosts, #898/#1100).
  Fresh buffers recycle `VkBuffer` handles, the mechanism behind the stale descriptor-set bug (#467, fixed by the
  destroy-log epoch, but the churn remains).
- CUDA is only partly paged (`docs/KV_CACHE.md` "CUDA Paged KV-Cache", `docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md`):
  - `CudaKvBlockPool` / `CudaKvBlockTable` / `CudaPagedKvCache` exist (#252), FP16 only, per-layer storage,
    host-side free list and refcounts, `SeedSharedBlocks` / `SnapshotFullBlocks` / `EnsureWritable`.
  - Default attention **gathers each layer's blocks into a contiguous scratch** (`PrepareAttentionScratch`). That is
    the expansion CLAUDE.md rule 2 calls a defect.
  - `attention_f16_paged` (#200) is opt-in (`DOTLLM_ATTN_PAGED_NATIVE=1`), decode only, plain kernel only, and does two
    synchronous H2D pointer-array copies per layer per token (#1270). Prefill always gathers.
  - Serving is still single-request: `ServerStartup.cs` passes `pagedFactory = null` for CUDA, because
    `ContinuousBatchScheduler(pagedPool:)` and `PrefixTrieManager` are typed to the CPU `KvBlockPool` /
    `PagedKvCache`.

So the Vulkan target is: pool and table (what CUDA has), **plus** in-place attention in decode **and** prefill (CUDA has
neither as default), **plus** scheduler and trie integration (CUDA has neither). Lessons taken from CUDA's gaps:
share one block table across layers (#1270), never gather, and decouple the trie from the CPU pool.

## 2. Constraints inventory (what the design must live with)

### 2.1 Attention kernels that the F16 dense host can reach

Built from the dispatch in `VulkanTransformerModel.cs` (~3225-3350) and the narrow-KV kernel set
(`VulkanNarrowKvAttentionKernels.cs`), not from shader file names. Many `attention_flash_f32_coopmat_*` files
(`pre543`, `v1scalarqk`, `v2scalarpv`, `v3duppad`) are A/B arms and are **not** in the paged scope.

| Kernel family | Used for | Shaders (F16 twin) | Paged twin needed | Cost drivers |
|---|---|---|---|---|
| Split-KV GQA v2 partial + merge | decode, `WouldSplit` (seqKv >= 17) | `attention_f32_splitkv_gqa_v2_kvf16`, `_kvq8`; merge reads no KV | partial: yes (dense host: S6; hybrid hd256 uses the v1 GQA pass: S3). merge: **unchanged** | 16 lanes/position, 8 columns/lane, TILE_KV 128, 4 KiB Q LDS + 4 KiB scores; hd <= 128 |
| Split-KV per-head (non-GQA) | decode, `G == 1` or fallback | `attention_f32_splitkv_kvf16` | yes | `kv_load.glsl` element loads |
| Split-KV rows (small-row 2..4) | MTP/spec verify (MiMo, gpt-oss) | `attention_f32_splitkv_rows_kvf16` | yes | window-clipped span |
| Sink variants (Flint, gpt-oss) | decode and prefill with sinks | `attention_sinks_f32_kvf16/_kvq8`, `attention_flash_gqa_cm_sinks_*` | S3 (decode), S7 (windowed models) | 256-row base alignment for bit-identity (#1176) |
| GQA coopmat flash hd128/hd256 | prefill, G >= 2 | `attention_flash_gqa_cm_kvf16{,_hd128}` | yes (S4 for hd256; hd128 in S6) | BC 32, K/V tile 32 x 264 f16 + scores + P = 30.5 KiB LDS of 32 KiB |
| GQA flash scalar | prefill fallback | `attention_flash_gqa_kvf16` | yes (S4) | |
| Coopmat hd64 | prefill, seqKv >= 640, hd <= 64 | `attention_flash_f32_coopmat_f16act_hd64` family | yes (S6) | **21,504 B LDS, 341 B headroom to stay at 3 workgroups/WGP** |
| Flash / coopmat / naive per-token | prefill fallbacks, short seqKv | `attention_flash_f32_kvf16`, `attention_f32_kvf16` | naive: yes (S3, doubles as the slow in-place prefill); others only if reachable | |
| qpos (per-query-position) | selective recompute, chunk reuse | `attention_f32_qpos_kvf16`, `attention_flash_gqa_cm_qpos_kvf16` | yes (S4) | |
| Not in v1 | MLA (`attention_mla_*`, own latent cache), QSA (`qsa_attention_*f16kv`), dflash, TurboQuant | | **refused** (section 3.8) | |

**v1 scope after the owner decision (D9).** The first host is the Qwen3.5/3.6 dense hybrid, which reaches only a subset
of the table: decode through the split-KV GQA pass at hd256 (it calls `SplitKvAttention.Record` for `seqLen == 1` when
`WouldSplit`, `headDim <= 512`; the v2 pass is limited to hd <= 128), small-row split-KV for MTP verify, prefill through
`EnsureGqaFlash` (GQA coopmat hd256 / scalar GQA flash / hd256 br flash) and the per-token fallbacks. **S3 and S4 cover
exactly that subset.** The v2 decode pass, the hd128 and hd64 coopmat variants and the sink kernels belong to the dense
host and later slices (S6, S7). The hybrid's writes are contiguous `convert_f32_to_f16_range` calls; its prefix snapshots
currently copy the KV rows and clone the GDN state (`SequencePrefixSnapshot`), which the paged pool replaces with
refcounted blocks plus the GDN clone.

The KV **write** side also needs twins: the fused RoPE + KV write (`rope_kv_write_f32_kvf16`), the range converter
(`convert_f32_to_f16_range`) and the row-scatter converter (`convert_f32_to_f16_rows`, which already takes positions from
a device buffer and is the precedent for a table-indirected write).

### 2.2 Device limits

| Limit | Strix Halo (gfx1151) | Intel Arc | RTX 3060 (Vulkan) |
|---|---|---|---|
| `maxMemoryAllocationSize` | 2 GiB (user-global `STRIX_HALO.md`, not in the repo) | *unmeasured, query* | *unmeasured, query* |
| `maxStorageBufferRange` | 4 GiB - 1 (user-global `STRIX_HALO.md`, not in the repo) | *unmeasured, query* | *unmeasured, query* |
| Memory model | UMA; per-process resident cap ~81-83 GiB that drifts (#905 canary) | UMA | discrete; `VulkanDevice.Download` needs staging |
| Subgroup | 64 (kernels baked for wave64 and host-gated) | 16/32 | 32 |

The two unmeasured columns are **not** to be filled from memory. S2 adds a device-limits probe (`vulkaninfo` or the
existing `tools/vk-alloc-probe`) and records the values in `STRIX_HALO.md` / `OTHER_MACHINES.md` style notes.
Shaders index with `uint`, every kernel binds `VK_WHOLE_SIZE` at offset 0 (`KernelSupport.cs`), and no kernel enables
64-bit integers in shaders, so a slab binding must stay under `min(maxStorageBufferRange, 4 GiB)` and the
allocation under `maxMemoryAllocationSize`.

### 2.3 Existing machinery to reuse, not reinvent

- `KvGeometry` slot map (hybrid attention slots, per-layer strides): `VulkanKvCache` already takes it.
- `VulkanNemotronHKvCache.PrepareForForward` + `BufferEpoch` + `DeviceMemoryAdmissionException`: growth admission and
  descriptor invalidation (#1100).
- `VulkanMemoryCapacity`, `VulkanGpuMemoryPressure`, `VulkanHeadroomCanary`: pool sizing (section 3.1).
- Hazard tracker / blanket barriers for the copy-then-read ordering inside one command buffer.
- `IPositionedHostKvCache` (`WriteRows` / `ReadRows` / `CopyRowsFrom`): chunk reuse and tests use it, so the paged cache
  implements it.

## 3. Design

### 3.1 Block pool memory layout

**Choice: per layer, one K slab and one V slab; slab row `p` is physical row `p = blockId * B + offset`.** The slab is
a plain `[totalBlocks * B, kvStride(layer)]` array in the cache's element type. This is exactly today's contiguous
cache layout with the logical row replaced by a physical row, so every shader's `KV_K(i)` / `loadK8(i)` element index
keeps its meaning (`i = physRow * stride + col`).

Alternatives considered:

| Option | Verdict |
|---|---|
| One buffer per block | Rejected. Thousands of descriptors per dispatch, or descriptor indexing everywhere. |
| One slab for all layers (layer offset in the index) | Rejected. 36 layers x 2 KiB x blocks exceeds the 2 GiB allocation cap on Strix at modest pools, and layers have different strides on Gemma-4/MiMo (`KvGeometry` non-uniform). |
| Chunked slabs per layer, several bindings | Rejected for v1. Needs a data-dependent choice of binding inside the K phase (descriptor arrays / `runtimeDescriptorArray`, divergent addressing) and a different shader for every twin. Revisit only if a model cannot fit one slab. |
| Buffer device address + pointer table (the CUDA design) | Rejected for v1. Needs `bufferDeviceAddress`, 64-bit address math in the innermost bandwidth-bound loop (the repo avoids int64 in shaders), and gives up bit-identical `KV_K(i)` indexing. Its only advantage is unbounded slab size, which D7 handles differently. |
| **Per-layer K/V slabs + block id (chosen)** | Same descriptors as today (K, V per layer), stable handles across requests, block ids layer-independent. |

**Slab capacity.** `slabCapBytes = min(maxMemoryAllocationSize, maxStorageBufferRange, 2 GiB)` per slab (the 2 GiB is
the Strix value and the floor for the unmeasured devices, until probed). Worked capacity per layer slab, F16:

| Model | K row | tokens / 2 GiB slab | B = 64 blocks | table ints per max sequence |
|---|---|---|---|---|
| Qwen3-4B (8 KV x 128) | 2 KiB | 1,048,576 | 16,384 | 16,384 (64 KiB) |
| Bonsai-2 (4 KV x 256, 16 attn layers) | 2 KiB | 1,048,576 | 16,384 | 16,384 |
| 16 KV x 256 | 8 KiB | 262,144 | 4,096 | 4,096 |

A pool needs `totalBlocks <= slabCapBytes / (B * rowBytes)`, so the *model window* rarely binds; the per-process memory
budget does. If a model's window cannot fit, the pool caps and the scheduler admits fewer tokens; it never silently
chunks.

**Sizing and lazy growth (D7).** The pool is sized from a byte budget, not from `maxSeqLen x sequences`:

1. Initial blocks = enough for `initialTokens` (default one window or 32K, whichever is smaller; *open question OQ3*).
2. Before any forward that needs more blocks than are free, `PrepareForForward` grows each layer's slabs by 25 %
   (at least 1,024 positions, the #1100 rule) up to the budget: allocate new slab, `vkCmdCopyBuffer` the used prefix
   `[0, highWaterBlock * B)` layer by layer, free the old slab, bump `BufferEpoch`, and return "drop descriptor caches" to
   the caller. Block ids do not change, so **no table is rewritten**. Transient cost: one layer's K and V (old + new).
3. Growth runs only between forwards (the model gate already serializes forwards; the fence-pipelined graph must have
   drained). Out-of-memory returns false and the scheduler queues; direct `Forward` callers get
   `DeviceMemoryAdmissionException`.
4. The budget is admitted with `VulkanHeadroomCanary` (allocate-and-touch on a second `VulkanDevice`), because the Strix
   per-process limit drifts (81.4 -> 73.9 GiB measured in #905) and a failed submit poisons the device.

UMA vs discrete: on UMA the slab is `DEVICE_LOCAL | HOST_VISIBLE` memory and counts against the per-process resident cap
1:1; on a discrete GPU it is strictly device-local, so every host-side access (table updates, `ReadRows`, tests) must
stage (`VulkanDevice.Download` already does for reads; uploads must be guarded the same way, see 3.3).

### 3.2 Block size

`B` must be a power of two (shift/mask addressing). Constraints and effects:

- Kernel tiles: split-KV TILE_KV = 128, coopmat BC = 32 (F16) or 16 (F32), scalar flash tiles 16/32, sink prefill needs
  256-row-aligned bases for bit-identity (#1176). Because every kernel resolves **logical** row -> physical row, tile
  boundaries stay in logical space and are unaffected by `B`. A tile may straddle blocks; correctness never depends on
  alignment. Alignment only decides how many table entries a tile needs (1 if `B >= tile` and aligned, else 2-3).
- Trie granularity: the trie shares **full** blocks only, so a hit recomputes up to `B - 1` tokens. At `B = 64` that is
  <= 63 tokens, about the cost of one decode batch at prefill speed; the CPU uses 16 (`KvBlockPool` default).
- Waste: expected half a block per sequence, 4.5 MiB at 144 KiB/token (Qwen3-4B F16): negligible against `maxSeqLen`
  preallocation.
- CoW unit: one block across all layers = `B x 144 KiB` = 9 MiB at `B = 64`. Rare (fork, rollback into a shared block);
  a device copy at tens of GB/s is sub-millisecond.
- Table size: `ceil(window / B)` ints.

**Default `B = 64`, as `BLOCK_SHIFT` specialization constant (pipeline built after the pool exists).** Using a
specialization constant rather than a push constant keeps shift/mask constant-folded, in line with the repo's rule that
register-pressure-sensitive choices are spec constants, not runtime uniforms. S3 sweeps B in {16, 32, 64, 128} on decode
(same-session ABBA); the chosen value is recorded here. The trie and `KvBlockPool` stay parametric.

### 3.3 Block table: layout, indirection, upload

**Layout.** One `int` SSBO arena `tables[slot * maxBlocksPerSeq + logicalBlock] = blockId`. A sequence owns a fixed
slot for its lifetime; `maxBlocksPerSeq = ceil(window / B)`. `tableBase = slot * maxBlocksPerSeq` is a push constant;
`BLOCK_SHIFT` is a specialization constant. Block id 0 is a reserved **null block** (holes, sliding-window releases).

**Why an SSBO, not push constants.** The guaranteed push-constant minimum is 128 bytes. The split-KV v2 block is 14
`uint`/`float` fields (56 B) and the GQA-cm block 12 (48 B); `tableBase` adds 4 B each, so the scalars fit, but even a
16K-entry table obviously does not. The other split-KV and sink variants append stamp/sink fields and must be counted in
S3 (acceptance item: every twin's push block <= 128 B on the minimum-limit device). The table is therefore binding N+1,
a `readonly` int buffer, read through the scalar/uniform path where the compiler can prove the index is wave-uniform.

**Why shared across layers.** Block ids are layer-independent (D1), so one 64 KiB table serves all 36 layers. CUDA's
per-layer pointer arrays cost two synchronous H2D copies per layer per token (#1270); here the table changes only when a
block is added (every `B` decode tokens), CoW'd, or released.

**Update path (UMA and discrete alike).** The host mutates its `int[]` mirror, records the dirty range, and at the *start
of the next forward's command buffer* records `vkCmdUpdateBuffer` (<= 64 KiB, 4-byte aligned) or a small
`vkCmdCopyBuffer` from a persistently mapped staging ring, then a transfer-to-compute barrier before the first attention
dispatch. Writing the table from the host directly would race an in-flight forward (a CoW'd entry is still read by it);
recording it in the command buffer orders it by construction and is identical on a discrete GPU. A forward that appends
no block records no table update at all.

**Descriptor stability.** The arena and the per-layer slabs keep their handles for the process lifetime except across
growth, so descriptor sets are reused across requests. This removes per-request KV buffer churn, which is what made
#467 reproducible, and makes the descriptor-aliasing class of bug structurally rarer (the epoch guard stays).

**Batch (S5).** `ForwardBatch` passes one `tableBase` per sequence; attention remains per-sequence dispatch as today. A
fused multi-sequence decode dispatch (batch dimension on workgroup z, vLLM `paged_attention_v2` style) is *possible*
because of the arena layout, but is out of scope (the Vulkan `ForwardBatch` fuses only the GEMMs; the CUDA design doc
reached the same per-sequence-launch decision for #251).

### 3.4 Refcount, CoW, append, rollback, truncate

All bookkeeping is host-side (a lock plus plain arrays, as in `KvBlockPool` / `CudaKvBlockPool`); block allocation
happens at most once per `B` tokens per sequence, never per token.

- **Allocate:** `EnsureCapacity(newLength)` before recording a forward pops blocks from the free stack, sets the table
  mirror, marks it dirty. Fails -> `false` -> growth (3.1) or queue.
- **Append:** the write twins resolve `physRow = table[tableBase + (pos >> SHIFT)] * B + (pos & MASK)`; a batch of rows
  crossing a block boundary is just per-row resolution.
- **Share (trie / fork):** `AddRef` per block; the sequence's table points at the shared ids. The trie shares full
  blocks only, so a seeded sequence's first write is into a *fresh* block; CoW is needed only for fork/beam and for
  rollback into a shared block.
- **CoW:** `EnsureWritable(pos)`: if `refcount(block) > 1`, allocate a block, record `vkCmdCopyBuffer` of that block's
  `B * rowBytes` from the old to the new slab region **for every layer, K and V**, record the barrier, swap the table
  entry (dirty), drop one ref. Recorded in the same command buffer before the write twins.
- **Rollback / truncate:** `Rollback(n)` sets the length; blocks entirely beyond `ceil(n / B)` whose refcount drops to
  zero return to the pool (keeping one spare block avoids free/alloc thrash for speculative decode). Stale rows inside
  the last block are never read (attention masks at `seqKv`). If the cut lands inside a *shared* block, the next append
  CoWs first (the CPU/CUDA rule). `CanTruncateTo` is always true for full-attention layers; windowed groups refuse a cut
  past what they still hold (3.5).
- **Free:** `Dispose` releases every ref; a sequence that fails or is cancelled returns its blocks and inserts nothing in
  the trie (`RecordCompletion` only on success), which is the structural fix for #1589's stale-row concern.
- **Hole sentinel:** released-by-window entries point at block 0 so a stray read is a defined (masked) read.

### 3.5 Sliding-window and hybrid layers

**Layer groups.** A block id is shared by all layers of a group. Layers with different needs must not share a table:
Flint has 14 full layers and 35 windowed layers; releasing a block that the windowed layers no longer need would
corrupt the full layers. So the cache has *KV groups*: `(rowStride, windowed|full)`. v1 has one group (uniform
full-attention models). S7 adds the windowed group: its own table (hence its own arena region), its own block ids from
the same pool or a second pool, and **block release**: a block wholly below `pos - W + 1` is released (refcount drop) and
its table entry set to block 0. Existing kernels already compute `kvStart = max(0, pos - W + 1)` and skip earlier tiles
(split-KV `kvStart`, flash window-start tile skip), so they never read the hole entries. This replaces the compact ring
(`SlidingWindowRows`, base-alignment 256) for the paged path; the ring stays for the non-paged cache. Prefix reuse across
released windowed blocks is not possible and the trie must treat a windowed-group prefix as match-able only while its
blocks are alive (mirrors `CanTruncateTo`).

**Hybrid (Qwen3.5/3.6 first; then Qwen3MoeHybrid / Nemotron-H / LFM2). First-class, not an extension (D9).** KV exists
only on attention layers: `KvGeometry.SlotOfLayer` gives the slot, and the pool allocates slabs for attention slots only
(as `VulkanNemotronHKvCache` already does; Bonsai-2: 16 attention layers, 64 KiB/token F16). Recurrent state
(`VulkanGdnStateCache`, SSM, conv state) stays per-sequence fixed-size and is **not paged**, but its lifetime is tied to
the paged cache's: a sequence owns one table slot and one GDN slot, released together on completion, failure or
cancellation. Consequences:

- **Trie hit rule.** A block-trie match is usable only at a block boundary for which the recurrent state is also
  available (a checkpoint from the #1286 step checkpoints / `RecurrentPrefixCacheEntries`). KV blocks up to the boundary
  are shared by refcount; the GDN state is restored from the checkpoint (`Clone`). A deeper KV match with no checkpoint
  is a miss at that depth; the lookup falls back to the deepest boundary that has one. S1 gives the trie an optional
  boundary-validity predicate for this; with no predicate (dense models) behaviour is unchanged.
- **Snapshots.** Today `SequencePrefixSnapshot` copies the KV prefix (`SnapshotPrefix`) and clones GDN state. With paged
  blocks the KV half becomes refcounts (no copy, no transient bytes); only the GDN clone (tens of MB on a 4B hybrid)
  remains, so checkpointing every K-th block is much cheaper than today.
- **Rollback / MTP.** KV rollback is length-only; GDN state cannot roll back, so speculative/MTP verify keeps using its
  existing state checkpoint (unchanged), and the paged cache must not free blocks a checkpointed prefix still references.
- **Failure mode to test.** Restoring KV blocks without the matching GDN state produces plausible but wrong text; the S5
  identity test has a mutant for exactly that.
- **Memory.** Concurrency on a hybrid is bounded by GDN slots as well as KV blocks; admission checks both.

**Other cache types stay as they are in v1:** MLA latent cache, Qwen4-Exp QSA gather caches, dflash ring, pipeline
(two-device) caches.

### 3.6 KV element types

| Type | v1 | Notes |
|---|---|---|
| F16 | **yes** | The Vulkan dense default since #1151. Every kernel in 2.1 has an F16 twin; the paged twin changes only the row index. Bytes/token Qwen3-4B 144 KiB. |
| F32 | oracle only | Allowed as a reference/baseline per CLAUDE.md rule 1, used through the *existing contiguous* F32 cache, not a paged F32 pool. No new F32 hot-path store. |
| Q8_0 (opt-in) | S7 | Must use the Flint interleaved layout (`kv_load.glsl` `KV_Q8`: 36-byte blocks `[float scale][32 x int8]`, rows a multiple of 32). A paged row is `stride/32 * 36` bytes, so `KV_K(physRow * stride + col)` works unchanged. Must **not** use `VulkanQuantizedKvCache`'s split scale/quant layout, which dequantizes the full history into an F32 scratch (the defect described in KV_CACHE.md "feasibility, #1550"). Twins exist today only for split-KV decode, sink kernels and the sink coopmat prefill; the generic GQA flash/coopmat/qpos paths have none, so S7 includes writing them. |
| Q4_0 | later | Blocked on the #1550 nibble-order check. |
| TurboQuant | **refused** | `VulkanTurboQuantKvCache` dequantizes into a shared F32 scratch; paged TurboQuant needs an in-place decode-and-attend kernel first. |

Q8_0 measured *slower and worse* than F16 on Vulkan (#1605: decode equal at 2K, slower at depth, KL 2.3-4.3x the F16
floor), while the CPU paged default is Q8_0 (#1284). Vulkan paged defaults to F16; see OQ2.

### 3.7 Kernel changes, per family

Common transformation, applied in the loads of every twin:

```
physRow(r) = tables[tableBase + (r >> BLOCK_SHIFT)] * B + (r & (B - 1))
KV_K(i) -> KV_K(physRow(i / stride) * stride + i % stride)      // or: pass physRow where rows are indexed
```

- **Split-KV GQA v2 (decode):** each wave covers 4 positions of a 128-position tile (16 lanes/position). The tile spans at
  most 3 blocks at `B = 64` (start unaligned) and 2 at B = 128. Resolve in two stages: lanes that own a position read
  `tables[...]` once for their position (a wave-uniform broadcast load for same-block lanes) *before* the K loads, so
  the table read overlaps the previous tile's softmax. Hoist when `B >= 128`. The split partition (`kvStart`,
  `numSplits`, per-split span) is **identical** to the contiguous kernel, so results are bit-exact; rounding splits to a
  multiple of `B` (which would make each tile single-block) is a possible later optimization, but it would change the
  reduction order and needs its own measured decision, so it is not in v1.
- **Split-KV per-head, rows (small-row), sink variants:** same row remap; the sink merge and window-clipped span are
  untouched.
- **GQA coopmat flash (hd128/hd256) and scalar GQA flash:** the tile fetch (`fetchTile`/`storeTile`: global -> registers
  -> LDS) already addresses a row per thread; replace the row by `physRow`. The LDS tile, scores and P are unchanged.
  The prefetch stage `pf[]` gains one `uint` per distinct block touched by the thread's rows, i.e. registers only.
- **Coopmat hd64:** 21,504 B LDS and 341 B of headroom (65,536 / 3 = 21,845). **Rule: no table staging in LDS in any
  twin.** Table entries are held in registers/SGPRs. A single occupancy step (3 -> 2 workgroups) costs 21 % at p2048, so
  S4 verifies the compiled LDS size of the paged hd64 twin equals the contiguous one.
- **Per-token naive F16 twin:** simplest change; becomes the in-place (slow) prefill path between S3 and S4 so S3 is
  usable on its own without any gather.
- **qpos kernels:** rows are already scattered by position; they gain the table binding.
- **Merge kernels, RoPE, norm, GEMMs:** unchanged.

**VGPR/LDS risk.** Zero-LDS and low-LDS kernels sit on wave64 VGPR granularity boundaries (one extra live float cost
11-17 % on `moe_indexed_matmul_q4_k_q8_1`, a different kernel but the same hardware class). Each paged twin therefore gets
(a) a shader-info VGPR/LDS/spill report next to its contiguous twin, (b) a *ballast arm* (the contiguous kernel plus one
live uint, no table) in the A/B so a regression can be attributed to register pressure rather than to the extra load,
and (c) if a twin crosses a granularity boundary, the spec-constant route (a separate pipeline) rather than a runtime
uniform.

### 3.8 No gather-into-scratch fallback

A paged cache plus a gather/dequant scratch is the CUDA default today and the CPU pre-#1268 design; CLAUDE.md rule 2
classifies it as a defect to remove, not a cost to budget. Therefore:

1. Paged mode is only enabled for a (host, cache type, kernel path) combination for which **every** attention dispatch it
   can reach has an in-place paged twin. The admission check lives in one function
   (`VulkanPagedKvSupport.Check(model, kvType, config)`) that returns `ok` or a named reason (`"MLA latent cache"`,
   `"attention sinks on a full-history group (S7)"`, `"Q8_0 twin for qpos missing"`, ...).
2. A refused combination uses the model's own contiguous cache and **logs the reason** (never the current silent text).
3. The in-place per-token prefill kernel (S3) is slow but correct and bounded; it is not an "expansion fallback" because
   it allocates nothing. If a shape has no in-place twin at all, the request is refused at admission, not served via a
   scratch.
4. If a future shape ever truly needs an expansion fallback, it must be bounded, admitted against free device memory
   (never spill), logged, and its byte cost listed in the PR, as rule 2 requires. None is planned.

### 3.9 Scheduler and prefix-trie integration

- **Allocator seam (S1).** `PrefixTrie` and `PrefixTrieManager` take `KvBlockPool` / `PagedKvCache` concretely;
  `ContinuousBatchScheduler` takes `KvBlockPool? pagedPool`. S1 extracts `IKvBlockAllocator` (block size, free/total
  blocks, bytes per block, allocate, addref, release) and `IBlockSeededKvCache` (`SeedSharedBlocks`,
  `SnapshotFullBlocks`, ids). `CudaKvBlockPool` / `CudaPagedKvCache` already have the methods, so S1 also removes CUDA's
  "not wired to the scheduler" gap (a separate CUDA issue consumes it).
- **Admission by bytes (S5).** Today the Vulkan scheduler (opt-in, hybrids only, which is why the hybrid host goes first) bounds concurrency at
  `MaxActiveSequences = 8` ("no byte-budget admission without a paged pool"). With the pool, admission is
  `FreeBlocks >= blocksNeeded(prompt) + ReserveBlocksPerSequence`, with eviction of zero-ref trie leaves first (the
  existing `TryEvict` logic), then queue. An idle sequence holds no device bytes beyond its table slot.
- **Dense-host `ForwardBatch` (S6; the hybrid host already runs sequences through its per-sequence `ForwardBatch` and the opt-in Vulkan scheduler).** `ForwardBatchSimple` currently peels off caches that are not `VulkanKvCache`, and copies
  each sequence's K/V slice into its own cache. For paged caches it scatter-writes straight from the batched K/V buffers
  through the table, removing the per-sequence slice copy. Attention stays per-sequence. `SupportsThreadedSequenceState`
  is not set on the dense host today; S6 audits whether the shared scratch needs per-sequence state or whether the
  existing model gate suffices.
- **Trie vs `PrefixCache` (#1591).** On the paged path the block trie supersedes the whole-cache `PrefixCache`
  (the engine already gives the trie precedence). The #1591 path remains for non-paged Vulkan. `cached_tokens` becomes
  `matchedBlocks * B`, exact.
- **Stats and API.** `/v1/prompt-cache` reports trie stats; named-prefix pins work unchanged (they are refcounts).
- **Precision rule 3 (disk <-> device direct).** Out of scope: nothing here spills paged blocks to disk. If the trie ever
  spills to disk, UMA imports mmap pages zero-copy and a discrete GPU needs DirectStorage; a host bounce needs sign-off.

### 3.10 Server / CLI wiring

- New `VulkanPagedKvCacheFactory` + pool creation after model load in the Vulkan branch of `ServerStartup`, plus the
  corresponding `RunCommand` / `ChatCommand` branches (which today print other "not supported" warnings).
- `--paged` / `UsePaged` semantics: serve's `UsePaged` defaults to true, so it cannot distinguish "user asked" from
  "default". Introduce a tri-state (Auto / On / Off): `--no-paged` = Off (always wins); `--paged` = On (error or warning
  with the named reason if unsupported, not silent); neither = Auto. **Auto on Vulkan stays contiguous until S5 passes and an RTX 3060 run is done (RESOLVED, OQ1)**, then becomes paged when `Check` says ok. `DOTLLM_VK_PAGED=1|0` overrides for A/B and
  rollback.
- Startup logs (replace the current line):
  - `[dotllm] Using Vulkan paged KV-cache (f16 blocks of 64 tokens, pool 6.0 GiB (41,984 blocks), in-place attention, max 4 sequences)`
  - `[dotllm] Paged KV-cache not used on Vulkan for <architecture>: <named reason>; using the model's own KV-cache.`
- `--cache-type-k/-v`: unset or `f16` -> F16 blocks; `f32` -> contiguous F32 cache (the oracle), with a note; `q8_0` ->
  S7 (refused with reason before); others -> refused with reason.
- `--kv-pool` sizing flags (name and default are OQ3).

### 3.11 Failure modes to design against

| Hazard | Mitigation |
|---|---|
| Stale descriptor after slab growth | Growth bumps `BufferEpoch`; the model drops its descriptor caches (`PrepareForForward` contract); a test pins `KvBufferInvalidationsForTest`. |
| Table update races an in-flight forward | Updates are recorded inside the command buffer with a barrier; never host-written. |
| Table entry beyond `seqKv` read | Kernels mask at `seqKv`; appended entries are only read after the update barrier; hole sentinel is block 0. |
| dGPU host access to device-local memory | All pool/table host access goes through staged copies; test on the RTX 3060 before default-on (validate on discrete first rule). |
| Device memory limit drifts (Strix) | Canary admission before growth; `OUT_OF_DEVICE_MEMORY` maps to `DeviceMemoryAdmissionException`; scheduler queues. |
| Spec-decode rollback leaves rows | Length-only truncation; reads masked at `seqKv`; block-level free only on refcount zero. |
| Wave64-baked kernels on Arc/NVIDIA | Inherit existing `SubgroupSize` gates; paged twins add no new subgroup assumptions. |

## 4. Test plan

**General fixture rule (the trap):** if blocks are allocated in identity order, a kernel that ignores the table passes.
Every paged test builds its tables from a **seeded shuffle** over a pool shared with at least two other live sequences
whose blocks sit between the test sequence's blocks, with a partial tail block, and with block id 0 never used as data.
Fixture content is random per row (distinct K and V), so swapping K/V tables or rows is observable.

**4.1 Allocator and trie (host only, S1/S2).** Refcount, CoW, exhaustion, free/alloc thrash, abandon, trie insert/lookup/
evict against a fake device allocator. Mutants: refcount not dropped on abandon, wrong `BlockSize`, admission ignoring
the allocator, trie hit counted in tokens not blocks.

**4.2 Write path and row IO (S2).** Scatter write through the table, then `ReadRows` vs a contiguous F16
`VulkanKvCache` given the same logical rows: **bit-identical**. Positions crossing blocks, rewrites, rollback and
re-append, CoW with a second sequence reading the shared block (its attention input must stay bitwise unchanged),
growth mid-sequence (content identical before and after; `KvBufferInvalidationsForTest` > 0).

**4.3 Kernel parity per twin (S3/S4), address-only so exact equality, not a tolerance.** Grid: seqKv in
{1, 15, 16, 17, B-1, B, B+1, 2B+3, 4096, 33000}; G in {1, 2, 4, 6, 8, 12}; hd in {64, 128, 256}; window on/off;
`kvStart != 0`; sinks; masks (causal, bidirectional, hybrid); seqQ in {1, 2..4, 17, 512}. Oracle: the contiguous twin on
the same content. Where a future change alters reduction order (split rounding), switch that case to an f64 reference and
a bound that scales with sqrt(seqKv), not a fixed one.

**4.4 Substitution mutants (all must die; survivors are findings).**

| Mutant | Expected |
|---|---|
| Kernel ignores the table (identity map) | differs (table is a non-identity permutation) |
| Table rotated by one entry; off-by-one block (`pos>>SHIFT` +1) | differs |
| Wrong `BLOCK_SHIFT` | differs |
| `tableBase` ignored for the second sequence | differs |
| K and V tables swapped (needs distinct K/V content) | differs |
| Stale table after CoW / growth (update skipped) | differs |
| CoW skipped | the other sequence's output changes |
| Window: hole entry mapped to a live block | no change expected for in-window reads; catches reads below `kvStart` (this one is a negative control: it must *survive* in decode, proving the skip is real) |

Record in each PR which mutants were applied and which died. Commit before applying mutants (a `git checkout --` wipes the
implementation).

**4.5 End to end.** Greedy text identity, paged vs contiguous F16, on Bonsai-2 (S3-S5, MTP verify included) and, from S6, Qwen3-4B Q4_K_M and Llama-3.2-1B, at ctx 512 and 8K,
256 tokens x several prompts, both A/B orders, plus logits equality on the first N steps (perplexity cannot see
decode-path changes). Under the scheduler (S5) the oracle is the **contiguous cache under the same `ForwardBatch`
schedule**, not a solo run: batched GEMM and GEMV rows are not bit-identical. Prefix reuse: `cached_tokens` exact and a
multiple of `B`, warm output equal to cold, mutant "off by one block in the match". Cancellation mid-prefill and mid-decode
leaves the pool at its pre-request free count.

**4.6 Proof the path ran.** Perturb one stored row through the table and confirm the output moves; assert a
`PagedDispatchCount` test hook (as `ForwardBatchFusedCount` does) so a silent fall back to the contiguous cache is a test
failure, not a pass.

**4.7 Hardware.** All GPU tests take the ResourceLock `Strix:GPU` lock in the implementing PRs. Run the parity grid and one
e2e on the discrete RTX 3060 (via the Framework box, contended, coordinate with James) before default-on; the Intel Arc
leg is optional but listed.

## 5. Performance plan and gates

- **Method:** same-session, alternating ABBA, >= 3 rounds, report min / median / max; never compare cross-process runs
  (the box swings 5-60 tok/s under load). Check `Get-Process` for human GPU use before and after. Warm the GPU clocks
  (cold vs warm is a 2-3x phantom delta).
- **Metrics:** decode tok/s at ctx 512 / 8K / 32K; prefill pp512 / pp2048 / 8K; for S5, aggregate tok/s at N = 1/4/8
  concurrent, first-token latency with a prefix hit, bytes held per idle sequence, peak device bytes vs `maxSeqLen`
  preallocation.
- **Gates:** paged median >= 0.97x contiguous with ranges not disjoint-worse for decode and prefill at every depth; or an
  explained regression signed off by James. LDS bytes unchanged for every twin; VGPR counts reported with a ballast arm;
  hd64 coopmat within its 341 B headroom.
- **Expected effects (hypotheses, not claims):** decode ~neutral (one L1-hit table load per tile); prefill ~neutral for
  the coopmat kernels (load-stage index math); big wins are memory (no `maxSeqLen` preallocation, concurrency admission
  by bytes) and prefix reuse (block-granular, concurrent) rather than single-stream speed.
- **Record which code path produced each figure** (fixture is not path): the dispatch-count hooks from 4.6 go in the
  benchmark output.

## 6. Slices

| Slice | Issue | Scope | Depends on | Effort | Risk |
|---|---|---|---|---|---|
| S1 | #1725 | Backend-neutral `IKvBlockAllocator` / `IBlockSeededKvCache`; trie, manager and scheduler off the CPU concrete types; trie boundary-validity predicate (hybrid checkpoints). CPU-testable. Also unblocks CUDA's scheduler gap. | none | M, 3-4 d | low-medium |
| S2 | #1726 | `VulkanKvBlockPool` (slabs for attention slots, budget, growth, canary), table arena + in-cmdbuf updates, `VulkanPagedKvCache`, write twins, CoW, rollback, snapshots by refcount, row IO, GDN-slot pairing; device-limit probe. No attention. | none (S1 optional) | L, 1.5 w | medium |
| S3 | #1727 | Hybrid-host decode twins (split-KV GQA hd256, small-row, per-token = slow in-place prefill); `VulkanQwen3HybridDenseTransformerModel` wiring; `DOTLLM_VK_PAGED` opt-in; block-size sweep. | S2 | L, 1-1.5 w | high (VGPR cliff) |
| S4 | #1728 | Hybrid-host prefill twins (GQA coopmat hd256, scalar GQA flash, hd256 br flash, qpos if reached); chunked prefill through paged caches. | S2; S3 first recommended | L-XL, 1.5-2 w | highest (LDS cliff, hd256 at 30.5 of 32 KiB) |
| S5 | #1729 | Hybrid-host scheduler with paged caches + GDN slots, byte-budget admission, Vulkan block trie with recurrent-checkpoint hits (#1286), `/v1/prompt-cache`, tri-state paged mode, **default flip after S5 + RTX 3060 run**, `DOTLLM_VK_PAGED=0` rollback. | S1, S3, S4 | XL, 2 w | high (KV/recurrent consistency) |
| S6 | #1730 | Dense host (split-KV v2, hd128/hd64 coopmat, `ForwardBatch` scatter-write) and other hybrid hosts (Qwen3MoeHybrid, Nemotron-H, LFM2). | S3, S4, S5 | L-XL, 2 w | high (hd64 LDS cliff) |
| S7 | #1733 | Q8_0 blocks (opt-in, interleaved layout) + twins; sliding-window layer groups with block release; sink twins. Split into issues when started. | S3, S4 (S6 for windowed dense) | XL | high |

Merge order: S1 and S2 in parallel, then S3 and S4 (parallelizable after the ABI is frozen in S2 and the block size is fixed
in S3), then S5 (default flip for the hybrid host), then S6, then S7. S3 alone is usable (hybrid decode in place, slow
in-place prefill) behind the opt-in switch, so each slice leaves `dev` green and the default unchanged until S5.

Totals: roughly 8-10 engineer-weeks for S1-S5 (default-on for the Qwen3.5/3.6 hybrids; the recurrent-checkpoint work in S5 and the hd256 kernels in S4 are the added cost versus a dense-first plan), about 2 more for S6, S7 open-ended.

## 7. Risks

1. **VGPR granularity cliff** on the decode twins (3.7). Mitigation: ballast arm, spec-constant pipelines.
2. **hd64 coopmat LDS cliff** (341 B headroom): registers-only table, compiled-size assertion.
3. **Descriptor aliasing / stale descriptors** on growth: epoch contract and pinned tests; fewer buffers churn than today.
4. **dGPU behaviour** differs from UMA (staged host access, subgroup sizes, allocation limits): RTX 3060 run is a hard
   gate; the Framework box is contended and flaky (eGPU drops), so schedule it.
5. **Strix memory cap drift** can fail a submit after a successful allocation: canary admission, queue not crash.
6. **Scheduler/trie refactor (S1)** touches CPU code that is currently well-tested; keep it behaviour-preserving.
7. **Default flip** changes `serve` behaviour for every Vulkan user: gated, env rollback.
8. **Bit-exact claim** holds only while kernels stay address-only; any later partition change needs its own oracle.

## 8. Decisions and open questions

**RESOLVED (owner, 2026-11-09):**

1. **OQ1 default-flip policy: RESOLVED.** Flip Vulkan `serve` to paged (Auto) after S5 **and** an RTX 3060 run, with
   `DOTLLM_VK_PAGED=0` as the rollback.
2. **OQ2 default block type: RESOLVED.** F16 is the default; Q8_0 is opt-in (S7).
3. **OQ4 scope of v1 hosts: RESOLVED.** The Qwen3.5/3.6 hybrids (Bonsai-2 style) come **first**, not the dense host. The
   slices were re-cut: S2 builds the pool for attention slots, S3/S4 target the hybrid's hd256 kernels, S5 owns the hybrid
   scheduler and trie with recurrent checkpoints (#1286); the dense host and other hybrids are S6; Q8_0 and sliding
   windows are S7.
4. **OQ5 CUDA follow-up: RESOLVED (approved).** S1 unblocks CUDA's scheduler/trie wiring and #1270, and the CUDA gather
   default (a rule 2 violation) should be removed; the coordinator opens that issue separately.

**Provisional, awaiting James (recommendations only):**

5. **OQ3 pool sizing (provisional, awaiting James).** Flag name (`--kv-pool-gib`? fraction of post-load headroom?) and
   initial size. Recommendation: initial = min(one window, 32K tokens) x 2 sequences, grow to 40 % of canary-admitted
   headroom, cap by slab limit. On the hybrids the GDN slots also bound concurrency.
6. **OQ6 discrete-GPU access (provisional, awaiting James).** The Framework RTX 3060 is the only dGPU Vulkan box, is
   flaky/contended, and is now a hard gate for the default flip. Recommendation: schedule a dedicated window before S5
   completes; no better alternative identified.
7. **OQ7 split rounding (provisional, awaiting James).** Allow a later, separately measured change that rounds split
   boundaries to multiples of `B` (faster, changes reduction order, drops bit-exactness for that kernel).
   Recommendation: not in v1; revisit after S3 numbers.
8. **OQ8 Arc (provisional, awaiting James).** Recommendation: best-effort, not a sign-off gate.

## 9. References

- `docs/KV_CACHE.md`: Paged KV-Cache, CUDA Paged KV-Cache (#252), Prefix reuse (#1591), Vulkan dense F16 (#1151),
  Flint Q8_0 (#1605), quantized KV feasibility (#1550), compact sliding window (#1176), Prefix trie (Step 37).
- `docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md` (#200); CUDA #1270; CPU #1268; Vulkan #898, #1100, #1102, #1595, #1286, #1589.
- `src/DotLLM.Cuda/CudaKvBlockPool.cs`, `CudaKvBlockTable.cs`, `CudaPagedKvCache.cs`.
- `src/DotLLM.Vulkan/VulkanKvCache.cs`, `VulkanQuantizedKvCache.cs`, `VulkanNemotronHKvCache` (lazy growth),
  `Kernels/VulkanSplitKvAttentionKernel.cs`, `VulkanNarrowKvAttentionKernels.cs`; `VulkanTransformerModel.cs` attention
  dispatch (~3225) and `ForwardBatch` (~3896).
- `native/vulkan/shaders/kv_load.glsl`, `attention_f32_splitkv_gqa_v2.glsl`, `attention_flash_gqa_cm.glsl`.
- `src/DotLLM.Engine/PromptCache/PrefixTrie.cs`, `PrefixTrieManager.cs`; `Scheduler/ContinuousBatchScheduler.cs`.
- Device limits: user-global `STRIX_HALO.md` (max allocation 2 GiB, max storage range 4 GiB - 1, per-process cap drift).
- Prior art (cited, not redistributed): vLLM PagedAttention v1/v2 block tables.
