# Q8_1-rows 4x-interleaved packing (llama.cpp `block_q8_1_x4_packed128`) — feasibility notes

Follow-up to #376's "unexplored" item (`.docs/KERNEL_MAP.md` §3 item 2): does dotLLM's
`quantize_q8_1_rows.comp` / `matmul_q8_0_mmq.comp` activation staging benefit from 4-row
interleaved packing the way llama.cpp's Vulkan MMQ path does?

## 1. Current dotLLM per-row layout

`QuantizeQ8_1RowsKernel` writes two flat, row-major buffers for `N` rows × `K` cols
(`blocksPerRow = K/32`):

- `xq : uint[N * K/4]` — row `r` starts at `r * (K/4)`; within a row, 4 signed int8 lanes
  packed per uint, `K/4` uints laid out contiguously in block order (block `b`'s 8 uints at
  `[b*8 .. b*8+7]`).
- `xds: vec2[N * K/32]` — row `r` starts at `r * blocksPerRow`; one `(d, s)` pair per 32-block.

This is a **plain, non-interleaved** per-row layout — one row's data is fully contiguous, no
cross-row interleaving. Contrast with llama.cpp's `block_q8_1_x4` (`types.glsl`): 4 *rows* are
packed into a single struct (`f16vec2 ds[4]; int32_t qs[32];`, i.e. rows 0-3's blocks
interleaved together), later reinterpreted as `block_q8_1_x4_packed128` (`ivec4 qs[8]`) for
128-bit-aligned loads.

## 2. `matmul_q8_0_mmq.comp` staging loop — is it already coalesced?

The relevant loop (Stage sharedXq[64][8]):

```glsl
for (uint s = 0u; s < 2u; s++) {
    uint idx = tid + s * 256u;      // 0..511, tid = ty*16+tx, 0..255
    uint rLocal = idx >> 3u;        // 0..63 token within tile
    uint lane   = idx & 7u;         // 0..7 uint within block
    ...
    v = xq[xqRowBase + kBlock * UINTS_PER_BLOCK + lane];
    sharedXq[rLocal * CHUNK_STRIDE + lane] = v;
}
```

`tid` runs 0..255 linearly (`lane = tid & 7`, `rLocal = tid >> 3`). So for 8 **consecutive**
`tid` values, `rLocal` is constant and `lane` sweeps 0..7 — i.e. 8 consecutive threads read 8
**consecutive uints** (32 contiguous bytes) of the *same* token row. This is already coalesced
*within* a row: each 8-thread segment issues one contiguous 32-byte burst.

The catch: the *next* 8-thread segment (`tid` 8..15) jumps to `rLocal=1`, whose base address is
`xqRowBase(row+1) = xqRowBase(row) + K/4` uints away — for any realistic `K` (e.g. 2048-4096)
this is a large stride, not adjacent to the previous segment's 32 bytes. So across a whole wave
the accesses look like **N/8 separate 32-byte bursts with large gaps between them**, not one
merged burst. On AMD RDNA (wave32 or wave64 depending on driver/compute-mode), a wave spans 4-8
of these 8-thread row-segments, so the hardware coalescer/L1 sees multiple discrete 32B
transactions per wave instead of one wider one — this is the concrete gap 4-row interleaving
would close (see §3 diagram).

**Verdict on the crux question:** the staging load is coalesced *at 32-byte (8-uint)
granularity, per row*, but **not merged across rows** into a single wider transaction. It is
"already effectively coalesced" in the narrow sense that no thread reads a scattered/strided
individual element (the #338/#339 decode bug pattern) — there is no correctness-adjacent
uncoalescing here. But it is *not* maximally coalesced in the sense llama.cpp's packed128
format achieves (single 128-byte contiguous transaction per wave-segment of 4 rows).

## 3. What llama.cpp's interleaving actually optimizes for

`quantize_q8_1.comp` (`QBLOCK_X4` path) packs 4 blocks (= 4 independent 32-element quantization
groups, from up to 4 different rows when `ne` spans multiple rows) into one `block_q8_1_x4`,
then `mul_mmq.comp` reinterprets it as `block_q8_1_x4_packed128` (`ivec4 qs[8]`) and reads it via
`data_b_packed128[...]` — an **`ivec4` (16-byte/128-bit) load**, i.e. `LOAD_VEC_B = 16` bytes,
one instruction per 16 bytes instead of one instruction per 4 bytes.

This is the real driver: llama.cpp's `mul_mmq.comp` warptile design (`BLOCK_SIZE`/`WARP`
constant-specialized, `WMITER`/`TM`/`TN` register tiling with an explicit `block_b_cache`
struct, see `mul_mmq_shmem_types.glsl`) is built around **vectorized 128-bit global/shared loads**
as the unit of transfer — merging 4 rows' worth of `int32_t` lanes into one struct is what makes
a *single* `ivec4` load fetch 4 rows' first 4 lanes at once. It is a vectorization technique
(fewer, wider load instructions) as much as a coalescing technique.

dotLLM's `matmul_q8_0_mmq.comp` staging loop does **not** use vectorized loads at all — it is a
scalar-per-thread `uint` load (`xq[xqRowBase + kBlock*UINTS_PER_BLOCK + lane]`), one thread one
uint, relying purely on the memory system's cross-thread coalescer, not on `ivec4`/`vec4`
GLSL vector loads. Adopting llama.cpp's *exact* struct layout without also rewriting the staging
loop to issue `ivec4` loads would only get part of the benefit (transaction merging, not
instruction-count reduction) — and merging still requires 4 threads' worth of `lane`-values to
map onto one interleaved 128B block, which does not fit dotLLM's current `tid` → (`rLocal`,
`lane`) assignment (8 threads per row, not 4) without also changing that mapping.

## 4. Does it transfer to dotLLM's WG=16x16 layout? Concrete before/after

Current (`xq` row-major, K/4=…):

```
row r:      [ blk0.lane0 blk0.lane1 ... blk0.lane7 | blk1.lane0 ... ]   (contiguous, K/4 uints)
row r+1:    [ blk0.lane0 ... ]   <- starts K/4 uints after row r (big stride)
```
8-thread segment (tid 0-7) reads row r, lanes 0-7 → 32B contiguous. Next segment (tid 8-15)
jumps to row r+1's base → new, distant 32B burst.

Hypothetical llama.cpp-style interleave (pack 4 rows per 32-block, packed128 = 128B per
kBlock for 4 rows):

```
block (rows r..r+3, kBlock b):
  [ r.lane0 r+1.lane0 r+2.lane0 r+3.lane0 | r.lane1 r+1.lane1 r+2.lane1 r+3.lane1 | ... x8 lanes ]
  = 32 uints = 128 bytes, ivec4-loadable in 8x 16-byte loads
```
With this layout, 32 consecutive threads (tid 0-31, i.e. one RDNA wave32) mapped as
(`lane` 0..7, `rowInGroup` 0..3) would read the *entire* 128-byte block contiguously — one merged
transaction instead of 4 separate 32-byte ones. That is a genuine, mechanically distinct
improvement over the current layout, IF the staging loop's `tid → (rLocal, lane)` assignment is
also rewritten to match (currently 8 threads/row; would need to become "4 rows × 8 lanes,
lane-major" to get the coalescing benefit, or `ivec4` vector loads to get the instruction-count
benefit).

## 5. Recommendation: LOW priority, unconfirmed bottleneck

- The current per-row layout is **not the #338/#339-style uncoalesced-scatter bug pattern** —
  it already reads 32 contiguous bytes per row-segment, so this is not a "silently broken"
  coalescing gap like the ones that campaign fixed.
- The theoretical gap that remains (4 separate 32B transactions per wave vs. 1 merged 128B
  transaction) is real on paper, but:
  - `matmul_q8_0_mmq.comp`'s own header comment states prefill is **compute-bound** (dp4a-bound,
    "~62% of prefill time" was about kernel *dispatch coverage*, not a measured
    memory-bandwidth bottleneck in this specific staging load).
  - The `sharedXq` staging happens once per `kBlock` iteration and is amortized over
    `TM*TN=16` dp4a-heavy inner-product work per thread before the next barrier; per-#366 the
    LDS-load-to-dp4a ratio was already improved 0.5→8, suggesting staging bandwidth is not the
    dominant cost after that rewrite.
  - No profiling data exists (in this investigation or #376) isolating the `xq` staging load
    as a measured bottleneck; the case here is architectural reasoning only.
- Adopting llama.cpp's exact struct is not a drop-in port: it would require (a) rewriting
  `quantize_q8_1_rows.comp` to emit the 4-row-interleaved layout, (b) rewriting the
  `matmul_q8_0_mmq.comp` staging loop's `tid` → (row, lane) mapping, and (c) doing the same for
  every sibling MMQ kernel that shares this activation format (Q4_K/Q5_K/Q6_K/IQ4_XS MMQ per
  `docs/QUANTIZATION.md`) — a moderate-size, cross-kernel change for a benefit that isn't yet
  measured.

**Conclusion: LOW priority.** Not "not applicable" (unlike a refuted hypothesis) — there is a
real, identifiable structural difference between dotLLM's and llama.cpp's activation layouts,
and a plausible mechanism (transaction merging across the row-stride jump) by which interleaving
could help. But it targets a narrower and unconfirmed problem than the decode-side coalescing
bugs #338/#339 fixed (those were genuinely-scattered element-level reads measured at ~48GB/s /
19% of peak; this is a "4x fewer already-32B-aligned bursts" question with no profiling evidence
of being on the critical path). Recommend: **do not implement without first profiling** whether
`matmul_q8_0_mmq.comp` prefill is memory-bandwidth-bound on the `sharedXq` staging phase
specifically (e.g. via a microbench that varies only the K-stride between rows, or a GPU counter
capture); if profiling shows staging bandwidth matters, revisit this note for the concrete
layout change in §4.

No bench scaffold was added for this investigation — the analysis above is architectural/
byte-layout reasoning, not a validated-enough hypothesis to justify a GPU bench file per the
task's own guidance (only add one "if you're confident enough in a specific, concrete kernel
change"). A future bench, if profiling justifies proceeding, should follow
`VulkanMmvqSharedQuantBench.cs`'s convention (a same-session, order-reversed A/B).
