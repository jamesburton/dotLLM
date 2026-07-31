// PQ2_0 (PrismML Bonsai ternary) GEMV for dotLLM's decode path.
// y[row] = sum_g group_scale[row,g] * sum_{i in group g} (code(W[row,i]) - 1) * x[i]
//
// dotLLM PQ2_0 on-disk layout (must match dequant_pq2_0.cu / MatMul.PQ2S.cs / Dequantize.DequantizePQ2_0):
//   * Row-major W[n,k], k a multiple of 128. Row stride = (k/128)*34 bytes.
//   * 128-element GROUP = 34 bytes: scale(Half, 2 bytes) THEN codes[32] (4 codes/byte, 2 bits
//     each). Scale is PER-GROUP (not per-tensor like I2_S) — a group's contribution must be
//     scaled BEFORE summing into the row total, not once at the very end (see the per-group
//     `acc[rr] += group_acc * scale[rr]` below).
//   * Byte gp in [0,31] within a group's codes holds elements {gp, gp+32, gp+64, gp+96} at bit
//     offsets {6,4,2,0}. Code mapping value = code - 1 (0->-1, 1->0, 2->+1). This interleave is
//     BYTE-FOR-BYTE IDENTICAL to i2_s_gemv.cu's — only the scale granularity differs.
//
// ───────────────────────── F16 kernel v2: shared-x staging + warp-per-row ─────────────────────────
// v1 (still used by pq2_0_gemv_f32in below, kept as the CPU-vs-GPU parity reference) read x
// straight from global on every element, reasoning that k can be up to 17408 for real
// Bonsai-27B FFN rows, which at 4 bytes/float would need ~68 KB of static shared memory — over
// sm_86's 48 KB static cap. That reasoning holds for FLOAT staging but not HALF staging: 17408
// halfs is 34 KB, comfortably under the cap. v2 stages x as __half (converted to float only on
// read, same as I2_S's xs[] but half-width) and reuses I2_S's proven v2 warp-per-row scheme
// (see i2_s_gemv.cu's history comment for the full rationale): PQ2_0_ROWS_PER_BLOCK output rows
// per block, ONE WARP PER ROW pair, x staged into shared ONCE by all 256 threads and reused
// across every row in the block instead of re-read (and re-converted from half) per row. The
// grid is also uncapped (grid.x = ceil(n / PQ2_0_ROWS_PER_BLOCK)) — v1's grid-stride loop was
// capped at MaxDequantGridSize=256 blocks, serializing large-n projections (e.g. n=17408 FFN
// gate/up) across many grid-stride iterations per warp.
//
// Not yet ported from I2_S: vectorized (uint4/uint2) coalesced weight loads. PQ2_0's per-group
// 2-byte scale prefix means a group's 32 code bytes begin at a generally-unaligned offset
// (group_base+2), unlike I2_S's fully contiguous k/4-byte rows — porting the wide-load scheme
// needs either lane-splits-a-group's-codes restructuring or a weight repack, deferred as a
// follow-up. Numerics are unchanged from v1 (same per-group scale-then-accumulate order).
//
// TRIED AND REVERTED (2026-07-22): batching 8 groups per warp into one 32-byte-aligned
// shared-memory staging window (per-(warp,row) `groupBuf[320]`, two `__syncwarp()`s per batch)
// raised `ncu`-measured load-sector-efficiency from ~51% to a simulated ~94% and passed all
// correctness tests bit-exact, but MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS DROPPED
// FROM 10.95 TO 4.67 TOK/S (RTX 3060) — a ~57% regression, not an improvement. Root cause:
// sector-efficiency modeling doesn't capture the real cost of the added shared-memory round-trip
// (write-then-read instead of straight-to-register) or the two `__syncwarp()` barriers per batch
// (10/warp/row for k=5120, 34 for k=17408), which also blocks the compiler from pipelining
// successive unsynchronized loads the way the plain per-group read allowed. Superseded by the
// weight-repack approach below, which gets the same alignment win with none of that overhead.
//
// ───────────────────────── Split-layout addressing (weight repack follow-up) ─────────────────────────
// The "weight repack" deferred above is now implemented: `pq2_0_repack.cu`'s
// `pq2_0_repack_split_f16` reorders each tensor, ONCE at load time (see
// CudaQwen3HybridDenseTransformerModel.UploadRawTensor), from the interleaved on-disk layout into
// a SPLIT layout — all `n * groups_per_row` group scales first (contiguous Halfs), then all
// `n * groups_per_row * 32` code bytes (contiguous). `pq2_0_gemv_f16in` and `pq2_0_gemv2_f16in`
// below consume this split layout directly: a flat group index `g = row*groups_per_row + gi`
// addresses `scales[g]` and `codesBase[g*32 + lane]`, and since `g*32` is trivially a multiple of
// 32, EVERY group's 32-lane code read is unconditionally 32-byte-aligned — no per-group variation,
// no shared-memory staging, and critically no added `__syncwarp()`/`__syncthreads()` in this hot
// loop (see `pq2_0_codes_base_offset`'s doc comment in this file, and pq2_0_repack.cu's file
// header, for the offset derivation and its alignment-robustness rounding). This is strictly
// cheaper than the batched-staging approach reverted above — the alignment fix is entirely
// amortized into a one-time host-side repack, so the decode kernel itself gains coalescing with
// ZERO new per-iteration cost. Measure before trusting this comment, though — see the batched-
// staging note above for why "the math predicts a win" isn't sufficient on its own.
//
// `pq2_0_gemv_f32in` below is NOT touched — it is the CPU-vs-GPU correctness reference and no
// production caller ever passes it split-layout data.
//
// ───────────────────────── Round 4 (#157): residual "20.0/32 bytes" investigated, NOT changed ─────────────────────────
// A follow-up `ncu --set full` pass (post all three fixes above) still flagged
// MemoryCacheAccessPattern on global loads at ~62.5% sector efficiency ("only 20.0 of the 32 bytes
// transmitted per sector are utilized", Estimated Speedup ~21-22%). Since the split-layout repack
// makes `codesBase[gFlat*32 + lane]` unconditionally 32-byte-aligned and fully coalesced (32 lanes,
// 32 consecutive bytes, 100% of that sector used), the residual number can't be coming from the
// code read — it was traced by inspection to the per-group scale read,
// `scale = __half2float(scales[gFlat]);` (`pq2_0_gemv_f16in`/`pq2_0_gemv2_f16in`, main loop).
// `gFlat` here depends only on `rows[rr]` (a function of `blockIdx.x`/`wid`, uniform per warp) and
// `g` (the loop counter, also uniform per warp) — NOT on `lane`. Confirmed by compiling this file
// to a `-cubin -arch=sm_86` and inspecting the disassembly (`cuobjdump --dump-sass`, no GPU
// execution involved): every lane in the warp issues its OWN `LDG.E.U16.CONSTANT` for this read
// (not hoisted into a `ULDC`/uniform-register-file load — nvcc's automatic uniform-datapath
// promotion doesn't reach this case), but all 32 lanes compute the IDENTICAL address. This is a
// genuine single-address broadcast, not a stride/misalignment problem: the GPU's load/store unit
// coalesces same-address warp-wide requests into ONE sector fetch (broadcast to all 32 threads)
// regardless of whether the compiler recognized the uniformity statically — this collapsing
// happens on the actual runtime addresses, at the memory pipeline, not in the compiler. So the
// real transaction count for this read is ~1 sector per (row-pair, group) instance, not 32; the
// "20.0/32 bytes" / ~62.5% figure is `ncu`'s byte-utilization accounting (2 of 32 bytes in that one
// fetched sector are the ones actually asked for) rather than evidence of redundant traffic. This
// matches a documented false-positive pattern for the MemoryCacheAccessPattern rule on
// uniform/broadcast reads (cheap in practice, penalized by a metric that doesn't distinguish
// "many threads, one real transaction" from "many threads, many transactions").
//
// Considered and REJECTED, both without writing code that could only be evaluated by a real
// `ncu`/benchmark run this session couldn't perform (GPU execution was off-limits — see the perf
// investigation's own "TRIED AND REVERTED" note above for why an untested modeled win here would
// be exactly the wrong kind of bet):
//   1. Bulk-stage all of a block's group scales into shared memory once, mirroring `xs[]`'s
//      staging pattern. Arithmetic: current static shared usage is `xs[PQ2_0_MAX_K]` (34816 bytes)
//      + `rowOut[PQ2_0_ROWS_PER_BLOCK]` (32 bytes) = 34848 bytes against sm_86's 49152-byte static
//      cap — only 304 bytes of headroom. A block's needed scales are
//      `PQ2_0_ROWS_PER_BLOCK * groups_per_row * sizeof(half)`: 16*136*2 = 4352 bytes for k=17408
//      (the FFN down-proj shape), or 16*40*2 = 1280 bytes even for the smallest real shape,
//      k=5120. Both blow the 304-byte headroom by roughly an order of magnitude — this does not
//      fit without shrinking `PQ2_0_ROWS_PER_BLOCK` (== reducing occupancy, which the
//      already-landed ROWS_PER_WARP=4 experiment above showed costs more than it's worth) or some
//      other structural cut. Ruled out on the arithmetic alone.
//   2. Explicit `__shfl_sync` broadcast (one lane loads the scale, shuffles it to the rest of the
//      warp) instead of every lane loading it independently. This needs no new shared memory and
//      no new barrier (all lanes reaching this line are already warp-convergent — `warpActive` is
//      uniform per warp), so it doesn't carry the batch-staging experiment's specific risk
//      (synchronization overhead swamping a modeled bandwidth win). But per the SASS evidence
//      above, the hardware is already collapsing the per-lane loads into one broadcast transaction
//      — a manual shuffle wouldn't remove a real memory transaction that exists today, only trade
//      31 redundant-but-cheap per-lane LDG issues for one LDG + one shuffle instruction. Expected
//      effect was modeled as a wash-to-marginal at best.
//
//      TESTED (#159 follow-up) — implemented as `if (lane == 0) scale = __half2float(scales[gFlat]);
//      scale = __shfl_sync(0xFFFFFFFF, scale, 0);` in both `pq2_0_gemv_f16in` and
//      `pq2_0_gemv2_f16in` (numerics preserved exactly — __half2float is a pure bit-pattern
//      conversion, lane-order-independent, so shuffling the converted float is bit-identical to the
//      old per-lane load). Compile-time check first, per this file's standing rule: `-Xptxas -v`
//      showed register usage went UP slightly (pq2_0_gemv_f16in 41->43, pq2_0_gemv2_f16in 45->46;
//      zero spill in both before/after), but not enough to change either kernel's occupancy-binding
//      constraint (both were already shared-mem-bound at 5 blocks/SM; registers at 43/46 still give
//      floor(65536/(43*256))=5 and floor(65536/(46*256))=5 — no occupancy regression predicted).
//
//      MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS DROPPED FROM A FRESH 15.89-15.93 TO
//      12.42-12.65 TOK/S (RTX 3060, `bench -p 64 -n 16`, 3 reps, reproduced twice) — a ~20-22%
//      regression, not the modeled wash-to-marginal. This is the FOURTH compile-time-clean,
//      arithmetically-modeled change in this investigation to regress real throughput (see the
//      batch-8 and tail-wave-grid-resize entries above for the first two, and this file's own
//      "Round 4" section for a THIRD case — the small-K specialization — that for once held up).
//      Reverted in full (`git reset`/`git checkout` back to the pre-experiment commit); no
//      `__shfl_sync` broadcast exists in the kernel as shipped.
//
//      Root cause NOT confirmed by profiling (no `ncu` counter access in the session that ran this
//      experiment, matching the tail-wave entry's constraint) — offered as a REASONED HYPOTHESIS: an
//      explicit `if (lane==0) ... __shfl_sync(...)` forces the compiler to treat `scale` as
//      thread-divergent-then-reconverged at the source level (one lane takes a real branch, then a
//      real cross-lane data movement instruction), which likely defeats whatever the compiler/
//      hardware were doing implicitly to already treat the uniform-address load as cheap (the SASS
//      evidence in the "Round 4" section above showed each lane issuing its own `LDG`, but that LDG
//      could still be scheduled/pipelined by the compiler alongside neighboring independent loads;
//      an explicit `__shfl_sync` is a synchronizing warp-collective instruction that the scheduler
//      cannot reorder past as freely, and the lane-0-only branch adds real predication overhead 32x
//      per warp per group — group count is large, 68-136 iterations per row here). In short: the
//      "same address, many redundant-but-cheap loads" pattern the hardware already collapses for
//      free was, in practice, cheaper than one load plus one explicit shuffle instruction — this
//      closes out the file header's own prediction that "the hardware already broadcasts, so a
//      manual shuffle just trades cheap redundancy for an explicit instruction" with a real,
//      reproducible number: NOT a wash, a clear loss. Don't re-try this specific shuffle-broadcast
//      shape for this kernel without new evidence (e.g. a future restructuring that makes the
//      surrounding loop divergence-free in some other way this reasoning doesn't anticipate).
//
// ───────────────────────── Vectorized activation staging (#157, latency follow-up) ─────────────────────────
// The two coalescing fixes above (output-write staging, split-layout weight repack) targeted
// memory-bandwidth/sector-efficiency and delivered far less real speedup than predicted. `ncu`'s
// SpeedOfLight rule flagged this from the first profiling pass: "Achieved compute throughput
// and/or memory bandwidth below 60% of peak typically indicate latency issues." A source-
// correlated pass then found the single largest per-instruction stall in the entire kernel is the
// `xs[i] = x[i]` activation-staging loop at the top of `pq2_0_gemv_f16in`/`pq2_0_gemv2_f16in` —
// not any weight-read PC. With blockDim.x=256 and k up to PQ2_0_MAX_K, each thread issued up to
// k/256 SEQUENTIAL single-half (2-byte) loads before `__syncthreads()` released any compute — the
// whole block sat idle on this one-time-per-block, once-per-layer prologue. Fix: stage via uint4
// (16 bytes = 8 halfs) per iteration instead of one half, cutting the sequential load COUNT ~8x —
// a different lever from the two fixes above (transaction count, not per-transaction byte
// efficiency). See each kernel's staging block below for the alignment/divisibility reasoning.

// ───────────────────────── Small-K specialization (occupancy follow-up, round 4) ─────────────────────────
// A fresh `ncu --set full` pass on the kernel above (post all three fixes documented above) still
// flags it latency-bound (Memory/Compute Throughput 56-59%, still < 60%) with achieved occupancy
// stuck at 31.8-31.9% against a 33.3% THEORETICAL ceiling — `Block Limit Shared Mem = 2` is the
// binding occupancy constraint (vs `Block Limit Registers = 5`, `Block Limit Warps = 6`), and none
// of the three landed fixes touched it. The L1TEX scoreboard stall is still the single largest
// stall category (~48% of cycles) and the scheduler has an eligible warp only ~0.85-0.88/cycle out
// of ~3.85 active/scheduler (~44% of cycles with nothing to issue) — textbook "not enough resident
// warps to hide latency behind," which raising occupancy directly addresses (unlike the reverted
// 8-groups-batching experiment above, which targeted a *different* metric — sector efficiency — and
// didn't pay off; this fix targets the metric ncu's own SOL rule names: occupancy/latency-hiding).
//
// Root cause of the shared-mem ceiling: `xs[PQ2_0_MAX_K]` is sized for the LARGEST real call shape
// (dense FFN gate/up/down, k=17408) and costs 34816 bytes of the 48 KB static cap on EVERY launch,
// including the much smaller attention/GDN-projection call sites (k=5120 — Bonsai-27B's
// qwen35.embedding_length; see CudaPQ2_0GemvTest's file header for the same real dims), which only
// need 10240 bytes and are paying for 3/4 of a buffer they never touch.
//
// Occupancy arithmetic (sm_86, RTX 3060: 100 KB/SM shared-mem budget, 65536 registers/SM, 1536
// threads/SM max, 16 resident blocks/SM max; both kernels below share BlockSize=256 = 8 warps/block
// and the unchanged rowOut[PQ2_0_ROWS_PER_BLOCK]=16 halfs=32-byte staging buffer):
//   Current (xs[17408]):  34816 + 32 = 34848 B/block.  floor(102400/34848) = 2  -> matches the
//     ncu-reported "Block Limit Shared Mem = 2" exactly, cross-checking this model against the
//     actual profiler numbers before trusting it for the new case below.
//   New, k<=5120 (xs[5120]): 10240 + 32 = 10272 B/block. floor(102400/10272) = 9.
//   Registers: 48 regs/thread * 256 threads/block = 12288 regs/block; floor(65536/12288) = 5 ->
//     matches the ncu-reported "Block Limit Registers = 5" (unaffected by the shared-mem shrink —
//     same code, same per-thread register pressure; only the compile-time array *size* constant
//     changes, not the instruction stream, which is unchanged and still driven by the runtime `k`).
//   Warps: 1536 threads/SM / 256 threads/block = 6 -> matches ncu's "Block Limit Warps = 6".
//   New binding constraint = min(9 [shared mem], 5 [registers], 6 [warps], 16 [max blocks]) = 5 —
//     REGISTERS, not shared mem. New theoretical occupancy = 5*8/48 warps = 40/48 = 83.3% (up from
//     33.3%) — a real, arithmetic-grounded ~2.5x increase in resident warps available to the
//     scheduler to hide the still-dominant L1TEX latency behind, for the k=5120 attention/GDN call
//     sites specifically (QKV/gate/alpha-beta projections, K+V fused GEMV — see
//     CudaQwen3HybridDenseTransformerModel's ForwardGdnBody/attention body). The k=17408 FFN call
//     sites are UNCHANGED (still routed through the kernels above) since a k=17408 launch cannot fit
//     in a 5120-sized xs[] buffer and gains nothing from this fix anyway (already near its own
//     register ceiling's neighborhood, and FFN's compute-per-byte ratio is higher, per the file's
//     own history of "compute already dominates there" for k=17408 fusion decisions).
//
// Measure before trusting this, per this file's own standard (see the reverted batching experiment
// above) — theoretical occupancy is not measured throughput. This specialization is a plausible,
// arithmetically-motivated candidate, not a proven win; #157's remaining work is confirming it on
// real Bonsai-27B decode.
//
// ───────────────────────── Windowed activation staging (#159 — occupancy for the k=17408 FFN path) ─────────────────────────
// The small-K fix above (#157 round 4) only helps the attention/GDN call sites (k<=5120) — the
// FFN gate/up/down call sites (k=17408, routed through pq2_0_gemv_f16in/pq2_0_gemv2_f16in below,
// and the larger share of decode time) are UNCHANGED by it: still `xs[PQ2_0_MAX_K]` = 34816
// bytes, still `Block Limit Shared Mem = 2`, still 31.8-31.9% achieved occupancy against a 33.3%
// ceiling, still latency-bound per the same ncu SpeedOfLight signal as before that fix. Shrinking
// xs[] outright (like the small-K fix did) isn't available here — the whole point of this kernel
// is serving k up to 17408, and a single upfront stage-then-reuse design fundamentally needs a
// buffer sized for the full row.
//
// Fix: stage x in WINDOWS instead of all at once — a fixed-size xs[] buffer holding only
// PQ2_0_WINDOW_GROUPS groups' worth of activations (8704 elements = 17408 bytes) at a time. Every
// warp/row in the block still walks every group of the row exactly once, in the same
// group-then-row nesting order as before (preserving the established access pattern) — only now
// the outer loop is windows-of-groups, with a stage+sync bracketing each window instead of one
// stage+sync bracketing the whole row.
//
// Sync-count modeling (the metric that actually mattered in the reverted batch-8 experiment above
// — see that note for why "the occupancy math looks right" isn't sufficient on its own): the
// ORIGINAL kernel already has TWO `__syncthreads()`s, not one (easy to miscount) — one after the
// single stage (RAW), one before the block-coalesced rowOut write (protects THAT write — a
// hazard on the `rowOut[]` shared array, unrelated to `xs[]`/staging). A single-buffered R-window
// version needs, per window: one sync after staging (RAW — safe to read xs) and one sync after
// compute (WAR — safe to overwrite xs with the next window's data) — except the LAST window never
// needs its trailing WAR sync, because nothing stages into xs again after it. That gives
// R (RAW) + (R-1) (WAR) = 2R-1 syncs inside/around the window loop, PLUS the pre-existing
// rowOut-protecting final sync, which still runs exactly as before — it protects a DIFFERENT
// hazard (rowOut, not xs) and is unaffected by R; it is not "reused" as a WAR sync, it simply
// keeps doing its own unrelated job. Total: 2R `__syncthreads()`s for R windows (vs 2 today, i.e.
// R=1 in this formula too — 2*1=2, consistent).
//
// Double-buffering (stage window r+1 into a second buffer while computing window r, so the WAR
// sync collapses into the same barrier as the next window's RAW sync) would cut the loop-internal
// syncs from 2R-1 to roughly R, i.e. total from 2R to R+1 — NOT implemented here: with the window
// size chosen below, R is at most 2 for any k this kernel actually serves (see below), so
// double-buffering would only save ONE barrier (3 syncs vs 4 for the one real k=17408 shape) at
// the cost of real complexity (a second static buffer, current/other bookkeeping, more live
// registers) — judged not worth it for a single-barrier saving. Left as a candidate if a real
// `ncu` pass on this change (which this session's no-GPU-execution constraint can't run) shows the
// remaining stage/sync overhead is still material.
//
// `cp.async` (Ampere/sm_86-native asynchronous global->shared copy, bypassing the register file,
// letting the copy be in flight without blocking the issuing warp until an explicit
// `cp.async.wait_group`) targets the same latency-exposure problem double-buffering does, without
// needing a second static buffer. Also considered and rejected for now: at R<=2 windows there is
// very little staging latency left to hide — the dominant cost is the per-group global-load-bound
// compute loop, not the staging loop (per this file's own #157 "vectorized activation staging"
// finding above, which already cut the FULL k=17408 stage from 68 to ~9 sequential loads; this
// windowed version stages a strict subset per round, so it's cheaper still) — and hand-written
// inline-PTX `cp.async` is real added correctness surface this session cannot validate on real
// hardware. A documented candidate for a future round that CAN measure it, not committed on
// modeled reasoning alone (matching this file's established standard).
//
// Window size: PQ2_0_WINDOW_GROUPS=68 groups (8704 elements, 17408 bytes). Chosen by the same
// arithmetic the small-K fix used, applied to the SHARED-MEMORY BUFFER SIZE (k itself stays a
// runtime value covering everything from just above Pq2_0MaxKSmall up to PQ2_0_MAX_K):
//   xs[8704] (17408 B) + rowOut[16] (32 B) = 17440 B/block. floor(102400/17440) = 5. Unlike the
//   small-K fix (a pure constant-size change with no register impact), this one restructures the
//   loop nesting, so registers were NOT assumed unaffected — confirmed via `-Xptxas -v` instead
//   (see the "Confirmed via -Xptxas -v" paragraph below): both kernels bind on shared mem at 5
//   blocks/SM (registers turn out to allow 6 for pq2_0_gemv_f16in, 5 for pq2_0_gemv2_f16in — see
//   below for the actual numbers). New theoretical occupancy = 5*8/48 = 83.3%, the same ceiling
//   the small-K fix reached for k<=5120.
//   79 groups is the true max window that still fits this budget (floor((20480-32)/256)=79); 68
//   was picked instead because it is EXACTLY HALF of k=17408's groups_per_row=136, giving exactly
//   2 equal windows with no remainder for the one real production shape this kernel serves, at the
//   cost of a few hundred bytes of unused headroom — a deliberate simplicity-over-headroom trade,
//   not an oversight.
//
// Round count for every k this kernel actually needs to support: this file's own dispatch
// convention caps this kernel's callers at k in (Pq2_0MaxKSmall, PQ2_0_MAX_K] = (5120, 17408],
// i.e. groups_per_row in (40, 136]. ceil(groups_per_row / 68) is 1 for groups_per_row<=68
// (k<=8704) and 2 for groups_per_row in (68,136] (k in (8704,17408]) — NEVER more than 2 windows
// for any in-range k. The synthetic dispatch-boundary test shape k=5248 (groups_per_row=41) hits
// the ceil(41/68)=1 case: it degenerates to exactly today's single-stage behavior (one window
// covering the whole row, same 2 total syncs as before, R=1 in the formula above) — zero added
// overhead for that shape. Only the real k=17408 FFN shape (groups_per_row=136, R=2) pays the
// extra syncs, going from 2 total to 2*2=4.
//
// Tail-window correctness: the window loop's bound is `wStart < groups_per_row` with
// `wGroups = min(PQ2_0_WINDOW_GROUPS, groups_per_row - wStart)` — the last window is a genuine
// partial window whenever groups_per_row isn't a multiple of 68 (NOT the case for k=17408:
// 136 = 68+68 exactly; IS the case for k=5248: 41 = 41, one partial-by-construction window
// covering everything). `wElems = wGroups * PQ2_0_GROUP_SIZE` stays a multiple of 128 (hence of 8)
// for ANY wGroups >= 1, since PQ2_0_GROUP_SIZE=128 is itself a multiple of 8 — so the vectorized
// uint4 staging loop never needs scalar-tail handling regardless of window/row-tail interaction,
// mirroring the whole-row staging loop's existing k%8==0 argument (file header, #157). The group
// index used to address `scales[]`/`codesBase[]` (both full-row-length, NOT windowed) is always
// the GLOBAL group index `g = wStart + gi`; only the xs[] read (`out_base = gi *
// PQ2_0_GROUP_SIZE`, LOCAL to the window) is relative to the window's own base — the two index
// spaces are kept in separate variables (`g` vs `gi`) specifically so this can't be confused.
//
// NOT measured on real hardware (GPU execution off-limits this session, per the task constraint)
// — theoretical occupancy is not measured throughput, per this file's own standing rule. But the
// compile-time signal that IS available (`nvcc -cubin -arch=sm_86 -Xptxas -v`, no GPU execution)
// confirms the arithmetic above against the real compiler output, not just hand math:
//   pq2_0_gemv_f16in:  48 regs, 34848 B smem (baseline) -> 41 regs, 17440 B smem (this change).
//   pq2_0_gemv2_f16in: 55 regs, 34848 B smem (baseline) -> 45 regs, 17440 B smem (this change).
//   Zero spill stores/loads, zero stack frame, in both the baseline and windowed builds — the
//   loop restructuring did not push either kernel into register spilling, which would have been
//   the main way this change could quietly sabotage itself (a spill is a per-iteration global
//   memory round-trip, i.e. exactly the kind of cost this whole investigation is trying to hide).
//   Smem exactly matches the 17440 B predicted above. Registers dropped MORE than "unaffected" —
//   both kernels now need fewer registers than before (48->41, 55->45), plausibly because the
//   68-group inner loop gives the compiler a shorter loop body to keep live values across (vs
//   unrolling/scheduling across a 136-iteration loop) — a real, measured bonus on top of the
//   shared-mem shrink, not something this comment predicted going in.
//   Recomputing occupancy from these ACTUAL numbers (not the 48-register assumption used above,
//   which turned out conservative): pq2_0_gemv_f16in registers -> floor(65536/(41*256))=6;
//   pq2_0_gemv2_f16in registers -> floor(65536/(45*256))=5. Shared mem for both ->
//   floor(102400/17440)=5. min(shared=5, registers=6 or 5, warps=6, maxblocks=16) = 5 for BOTH
//   kernels -> occupancy = 5*8/48 = 83.3%, matching the predicted ceiling (shared mem and
//   registers are now co-binding for pq2_0_gemv2_f16in, and shared mem alone binds
//   pq2_0_gemv_f16in since its registers headroom is even better than predicted).
// Confidence is grounded in this confirmed arithmetic (matching the small-K fix's own
// methodology, independently confirmed against real ncu numbers for that case) plus a sync-count
// argument showing this design adds far fewer, and far more heavily-amortized, barriers than the
// reverted batch-8 experiment (2-3 block-wide syncs total here, vs that experiment's 34 warp-wide
// syncs, which ALSO broke compiler-level overlap of the weight reads themselves — unlike this
// change, which leaves that loop's structure untouched). This is a plausible candidate, not a
// proven win — actual ACHIEVED occupancy and decode throughput still need a real `ncu`/benchmark
// pass before trusting this further, per this file's own standing rule.
//
// ───────────────────────── TRIED AND REVERTED: tail-wave-quantization grid resize (#159 continued) ─────────────────────────
// A fresh `ncu --set full` pass on the windowed kernel above (real shape n=5120/k=17408, the FFN
// down-projection, grid=320=ceil(5120/16)) found SOLBottleneck had flipped to "Compute and Memory
// are well-balanced" (~69-71% both) but ncu's TOP-ranked remaining finding was tail-wave
// quantization, Est. Speedup 33.33%: achieved occupancy 70.9-71.0% vs 83.33% theoretical, bound by
// Block Limit Registers=5/Shared Mem=5 (tied) -> 5 blocks/SM x 28 SMs = 140 blocks/wave. 320 blocks
// = 2 full waves (280) + a 40-block (28.6%-utilized) partial tail wave.
//
// n=5120 is fixed (Bonsai-27B's qwen35.embedding_length) and confirmed to be the ONLY production
// shape ever routed to pq2_0_gemv_f16in/pq2_0_gemv2_f16in (every other Gemm/TryFusedPQ2_0Gemm2 call
// site in CudaQwen3HybridDenseTransformerModel.cs has k<=5120, routed to `_small`) — so a fix only
// needed to work for this one (n,k) pair. `nvcc -cubin -arch=sm_86 -Xptxas -v` sweeps over
// PQ2_0_ROWS_PER_WARP (no GPU execution, before writing any kernel change, per this file's own
// standing rule) found RPW=5 (rows-per-block=40) was the largest rows-per-warp that kept
// pq2_0_gemv_f16in's blocks/SM UNCHANGED at 5 (48 regs -> floor(65536/12288)=5; shared mem
// floor(102400/17488)=5 — same ceiling as the RPW=2 baseline) while shrinking grid to
// ceil(5120/40)=128, which is <= 140 — a SINGLE wave, structurally eliminating the tail-wave effect
// entirely rather than just shrinking its relative share (RPW=3/4 still needed 2 waves each). A
// simple per-block-time model (T(RPW) ~= RPW*c_row + c_stage, total = waves * T) predicted RPW=5
// strictly dominates both the baseline and RPW=3/4 in both terms of that model. Implemented as a
// SEPARATE constant (PQ2_0_ROWS_PER_WARP_LARGE, used only by the two large-K kernels) specifically
// to avoid regressing the `_small` kernels, which the same `-Xptxas -v` sweep showed would drop
// from 6 blocks/SM (100% occupancy) to 5 (83.3%) if the shared RPW were simply raised for
// everyone — that decoupling worked correctly (confirmed via a second `-Xptxas -v` pass on the
// modified file: `_small` kernels stayed at 39 regs/10272B, bit-identical to baseline).
//
// MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS DROPPED FROM 15.71-15.91 TO 13.19-13.35
// TOK/S (RTX 3060, `bench -p 64 -n 16`, 3-5 reps) — a ~15-16% regression, not an improvement, and
// clearly reproducible (baseline re-confirmed at 15.71-15.81 immediately before AND after this
// experiment, on the same machine in the same session, ruling out thermal/contention drift). This
// is the THIRD time in this investigation that a compile-time-clean, arithmetically-modeled,
// occupancy-preserving change has regressed real throughput (see the batch-8 sector-efficiency
// experiment above for the first) — the recurring lesson is that this kernel's actual bottleneck,
// once it stops being a pure "not enough resident warps" latency problem (as ncu's own SOL rule
// confirmed it had, right before this experiment), stops responding to occupancy-ceiling arithmetic
// at all, in either direction.
//
// Root cause NOT confirmed by profiling — `ncu`'s hardware performance counters are unavailable in
// this environment (`ERR_NVGPUCTRPERM`, no admin rights to grant `NVreg_RestrictProfilingToAdminUsers`
// access), so this is a REASONED HYPOTHESIS, not a measured one, offered for whoever picks this up
// next with profiler access: raising ROWS_PER_WARP from 2 to 5 didn't just shrink the grid, it also
// cut the number of INDEPENDENT thread blocks launched for the whole kernel from 320 to 128 while
// increasing each warp's own sequential row count 2.5x (2 -> 5). ncu's own SOL readout said this
// kernel was ALREADY compute/memory-balanced (~70% both) before this experiment, i.e. no longer
// starved for resident-warp-level latency hiding (the exact condition the windowed-staging and
// small-K fixes above successfully exploited) — so shrinking the grid bought nothing there, while
// the fewer/larger blocks may have reduced whatever request-level parallelism the memory system was
// still extracting from having 320 independently-schedulable blocks in flight across the kernel's
// lifetime (even though blocks/SM and total resident-warp COUNT were unchanged by this model, the
// GRANULARITY of how work arrives at the GPU's block scheduler and L2/DRAM queues was not, and that
// dimension isn't captured by a blocks/SM or waves-count calculation at all). This is offered as a
// plausible mechanism, not a confirmed one — flagged explicitly so a future session with working
// `ncu` counters can check SOL/L2 hit-rate/scheduler-issue metrics before trusting it further,
// rather than accepting this paragraph as settled fact.
//
// Reverted in full (`git reset --hard` back to the pre-experiment commit) — no PQ2_0_ROWS_PER_WARP_LARGE
// constant, no CudaKernels.cs grid-sizing split, exists in the kernel as shipped. Left as a
// documented negative result, matching the batch-8 precedent above: don't re-try a grid-only resize
// for this shape without new evidence that the kernel has gone latency-bound again (e.g. a future
// architecture change lowering compute/memory utilization back under ncu's ~60% latency-bound
// threshold), since this round's SOL readout says it currently hasn't.
//
// ───────────────────────── F32-native activations (#161 — eliminate surrounding convert launches) ─────────────────────────
// Everything above targets the GEMV kernels themselves. A separate advisor pass (see the
// `prismml-bonsai-model` project memory, "2026-07-22 continued — advisor review") found a
// different class of waste one level UP the call stack: dotLLM's activation pipeline for this
// model is F32 end-to-end (see CudaQwen3HybridDenseTransformerModel's class doc, "F32 activations
// throughout"), but every kernel below is F16-in/F16-out only. Each decode-time call therefore
// bracketed a GEMV launch with a `convert_f32_to_f16` launch before it and a `convert_f16_to_f32`
// launch after it (see convert.cu), round-tripping through dedicated `_activF16InScratch`/
// `_activF16OutScratch` scratch buffers — pure plumbing, no compute. Counted directly at the real
// call sites (CudaQwen3HybridDenseTransformerModel.Gemm/TryFusedPQ2_0Gemm2): ~1,088 PQ2_0-related
// kernel launches per decode token, ~65% (~704) of which were this conversion overhead.
//
// Fix: four new kernels below — `pq2_0_gemv_f32io`, `pq2_0_gemv2_f32io`, `pq2_0_gemv_f32io_small`,
// `pq2_0_gemv2_f32io_small` — read `const float* x` and write `float* y` DIRECTLY, doing the
// F32<->F16 conversion inline inside the existing vectorized stage/store steps instead of via a
// separate launch. Internal precision is UNCHANGED: `xs[]` is still `half` (the whole point of the
// v2 shared-x-staging design was fitting a full FFN row in 48 KB static shared memory — going
// float would double that footprint and blow the budget again, see the file's own v1->v2
// rationale at the top), so the only change is WHERE the F32<->F16 conversion happens (fused into
// the existing per-window/per-row load and store) versus a dedicated elementwise kernel launch
// immediately before/after. Naming: deliberately NOT `_f32in` (that name is already taken by the
// v1 CPU-vs-GPU correctness-reference kernel below, which must stay untouched per this file's own
// standing rule) — `_f32io` denotes "both directions are F32-native", to avoid any ambiguity with
// that reference kernel's name.
//
// Activation-staging load: was `uint4` (16 bytes = 8 halfs/iteration) reading directly from a
// `const half* x` with a straight copy. Now reads `const float4` (16 bytes = 4 floats/iteration,
// since input is 2x wider per element) from `const float* x`, then converts each of the 4 lanes'
// floats to half via `__float2half` before four scalar stores into the SAME `half xs[]` buffer.
// Alignment: `x` here is one of the model's persistent F32 activation buffers, always allocated via
// `cuMemAlloc_v2` (CUDA's documented minimum 256-byte alignment) with no sub-buffer byte-offset
// arithmetic at any real call site (decode's seqLen=1 uses the whole buffer) — comfortably 16-byte
// aligned for `float4`. The windowed kernels' window offset (`x + wStart*PQ2_0_GROUP_SIZE`
// elements) is a multiple of `PQ2_0_GROUP_SIZE`=128 floats = 512 bytes, also 16-byte aligned.
// `wElems`/`k` remain multiples of 128 (hence of 4) by the same argument this file already makes
// for the F16 uint4 staging, so no scalar tail path is needed here either.
//
// Output store: was staging each warp's `__float2half`-rounded reduced result into
// `__shared__ half rowOut[]`, then one block-coalesced `half` write to `y`. Now `rowOut` is
// `float` (64 bytes total instead of 32 — negligible next to the 34/17 KB `xs[]` buffer, does not
// move the occupancy ceiling) and stores the raw float accumulator with NO half rounding at all —
// strictly MORE precise than the old convert-launch path (which rounded to half in
// `_activF16OutScratch`, then widened back to float via `convert_f16_to_f32`), while still being a
// single coalesced block-wide write (4 bytes/lane × up to 16 lanes instead of 2 bytes/lane, still
// far cheaper than 16 independent lane-0 scatter stores per the #157 output-write-coalescing fix
// above, which this design reuses unchanged).
//
// Correctness/precision note for anyone re-running the existing F16In-vs-CPU tolerance bars: the
// F32-native kernels' `y[]` output is expected to be AT LEAST as close to the CPU F32 reference as
// the equivalent F16In kernel's output (one fewer intermediate rounding step on the output side;
// the input side and internal accumulation are numerically identical either way, since `xs[]`
// still holds `half` and the accumulate loop is unchanged) — see the new
// `PQ2_0GemvF32Native_MatchesCpuFloatReference` test below the existing F16In test in
// CudaPQ2_0GemvTest.cs.
//
// Call-site scope: EVERY real production call to `pq2_0_gemv_f16in`/`pq2_0_gemv2_f16in`/their
// `_small` siblings goes through `CudaQwen3HybridDenseTransformerModel.Gemm`'s single PQ2_0 branch
// or `TryFusedPQ2_0Gemm2` — both are centralized dispatchers reached from every GDN/attention/FFN
// call site in that file, and both always pass the model's F32 activation buffers on both sides (no
// call site was found needing F16 in with F32 out, or vice versa — the model has no F16-only
// downstream consumer for a PQ2_0 GEMV's output on the decode path). So exactly one F32-native
// variant per existing F16-native kernel was needed; the old F16In kernels are NOT deleted (kept
// for the CPU-vs-GPU F16 tolerance test and as a documented fallback shape, but no longer reached by
// any production code path after this change).
//
// I2_S's `Gemm()` branch has the byte-for-byte identical convert-launch-bracketing pattern
// (flagged by the same advisor pass) — deliberately NOT touched in this change; out of scope for
// issue #161, left as a documented, equally-mechanical follow-up.
//
// ───────────────────────── TRIED AND REVERTED: SwiGLU epilogue fusion (#161 continued, advisor candidate #2) ─────────────────────────
// A follow-up advisor pass on `ForwardDenseFfnBody` (CudaQwen3HybridDenseTransformerModel.cs)
// proposed folding the dense FFN's SwiGLU epilogue (`silu(gate)*up`, currently a wholly separate
// `swiglu_f32.cu` launch reading `_state.FfnGate`/`_state.FfnUp` and writing `_state.SiluOutput`)
// into the tail of the gate+up fused GEMV2 kernel — the same class of fusion as the
// already-proven residual-copy+RmsNorm and GDN-decay+sigmoid epilogue fusions from #157 (both real
// wins, +1.6%/+0.8%). The advisor's own estimate for this one was modest (<2%).
//
// Implemented as two new kernels, `pq2_0_gemv2_f32io_swiglu`/`pq2_0_gemv2_f32io_swiglu_small`
// (mirroring `pq2_0_gemv2_f32io`/`_small`'s windowed/small-K pair): a MATCHED-ROW design, not the
// existing kernels' virtual row-concatenation (rows [0,n0) from weight0 then [0,n1) from weight1)
// — SwiGLU needs gate[i] and up[i] for the SAME i combined together, but the virtual-concat
// kernels compute gate rows and up rows in entirely different blocks, so the epilogue can't be
// grafted onto that structure. The matched-row kernels instead compute BOTH weight0's row i (gate)
// and weight1's row i (up) in the SAME warp for every row i (n0==n1 required — true for dense FFN
// gate+up, both project to intermediateSize), then write `y[i] = silu(gate_i)*up_i` directly — one
// combined output buffer instead of two raw ones plus a follow-on elementwise launch.
//
// Compile-time check first, per this file's own standing rule (`nvcc -cubin -arch=sm_86 -Xptxas
// -v`, no GPU execution): the `_small` variant — the ONLY one actually reached on Bonsai-27B's
// production path, since dense FFN gate/up input dim = hidden_size = 5120 <= PQ2_0_MAX_K_SMALL —
// compiled to a bit-for-bit IDENTICAL register/shared-mem footprint as the kernel pair it replaces
// (`pq2_0_gemv2_f32io_small`): 39 registers, 10304 bytes smem, zero spill, in BOTH. Occupancy
// arithmetic from those numbers is unchanged in both directions (shared mem floor(102400/10304)=9,
// registers floor(65536/(39*256))=6, min(...)=6 -> 100% theoretical occupancy either way). The
// large-K windowed variant (not on Bonsai-27B's hot path, kept only for dispatch-family symmetry)
// showed a small register increase (45 -> 48) with unchanged occupancy binding (still 5 blocks/SM
// either way) and zero spill. By every compile-time signal this file's own standing rule asks for,
// this looked like a clean, low-risk change — exactly the profile the batch-8/shfl_sync/tail-wave
// entries above warn is NOT sufficient on its own.
//
// Correctness: validated bit-for-bit against a CPU reference (two separate `MatMul.GemvPQ2_0`
// calls + host-side `silu(gate)*up`) across the same shape/tail-clamp/dispatch-boundary coverage as
// the other GEMV tests (n=512/37/3, k=5120/17408/5248) — all within the established F16-internal-
// precision tolerance bar (max abs diff <= 5e-2, observed <= 1.4e-3 across all shapes). The full
// CUDA test suite passed (312/313 excluding one pre-existing, unrelated Q4_K_M flaky failure and
// the pre-existing #162 prefill-inf skip/failure, both confirmed unaffected by this change).
//
// MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS DROPPED FROM A FRESH BASELINE OF 16.74-
// 16.95 (median ~16.85, 6 reps across 2 independent 3-rep `bench -p 64 -n 16` runs, RTX 3060) TO
// 15.82-16.09 (median ~15.94, same 6-rep/2-run protocol) — a reproducible ~5.4% REGRESSION, not the
// advisor's predicted <2% improvement, and the two distributions do not overlap at all (baseline's
// worst rep, 16.74, still beats the fused kernel's best rep, 16.09). This is the FIFTH
// compile-time-clean, arithmetically/occupancy-modeled change in this investigation to regress real
// throughput (see the batch-8, shfl_sync-broadcast, and tail-wave-grid-resize entries above for the
// first three; #159's windowed-staging fix is the one clean occupancy-model win, by contrast).
//
// Root cause NOT confirmed by profiling (no `ncu` counter access in the session that ran this
// experiment, matching several entries above) — offered as a REASONED HYPOTHESIS: the virtual
// row-concatenation kernels this replaces process ALL of weight0's rows first (blocks 0 through
// n0/16-1), then ALL of weight1's rows (the remaining blocks) — for n0==n1==17408 that's exactly
// half the grid streaming sequentially through ONE tensor's memory, then the other half streaming
// through the OTHER tensor, each phase enjoying strong spatial locality within a single
// (large, contiguous) weight allocation. The matched-row fused kernel instead has EVERY block
// alternate between weight0 and weight1 — two independently-`cuMemAlloc`'d, generally
// widely-separated device allocations — every single group iteration, for the kernel's entire
// lifetime. Register/shared-mem/occupancy accounting (this file's usual compile-time check) cannot
// see this: it has no model for L2/DRAM locality or the GPU memory controller's page-open/close
// behavior across two simultaneously-hot, disjoint address ranges, which is exactly the kind of
// real-hardware effect this file's own standing rule (measure, don't trust the model) exists to
// catch. Total bytes read and total instruction count are IDENTICAL between the two designs (same
// FLOPs, same launches after accounting for the eliminated `swiglu_f32` launch) — the regression
// has to be a scheduling/locality effect, not a work-volume one.
//
// Reverted in full (`git checkout` back to this commit's pre-experiment state) — no
// `pq2_0_gemv2_f32io_swiglu`/`pq2_0_gemv2_f32io_swiglu_small` kernels, no
// `LaunchPQ2_0Gemv2F32NativeSwiGLU`/`TryFusedPQ2_0Gemm2SwiGLU`, exist in the kernel/dispatcher as
// shipped. Left as a documented negative result, matching this file's established precedent: don't
// re-try a matched-row (interleaved-tensor-access) fusion across two independently-allocated weight
// tensors for this kernel family without new evidence — e.g. a future change that co-locates
// gate/up weights in one contiguous allocation at load time, which would remove the hypothesized
// locality cost this round's numbers point to, or real `ncu` L2 hit-rate/DRAM-throughput counters
// confirming (or refuting) the hypothesis above.
//
// ───────────────────────── Algebraic ALU reduction (#161 continued, advisor candidate #4/5) ─────────────────────────
// A fresh advisor review (see the `prismml-bonsai-model` project memory) found the two largest
// GEMV kernels (FFN gate+up fusion, FFN down-proj) are no longer latency/occupancy-bound — an
// `ncu` pass captured before the F32-native (#161) work reported ALU as the highest-utilized
// pipeline (78.2%/68.5%). This section targets that: the shared per-code decode helper (formerly
// `pq2_0_accum_byte`, now split into `pq2_0_load_group_x`/`pq2_0_code_dot` below) computed
// `(code - 1) * x` per element, i.e. an explicit int subtract (`IADD3 ..., -0x1`) on every one of
// the 4 codes packed per byte, once PER ROW (called once per `rr` in the `PQ2_0_ROWS_PER_WARP=2`
// unrolled loop at every call site) — even though `x` (the shared activation) does not depend on
// the row at all. Algebraic identity: `Sum_group (code_i-1)*x_i = Sum_group code_i*x_i - Sum_group
// x_i`. The `Sum x_i` term is row-independent, so it can be loaded+summed ONCE per (warp, group)
// and reused across both rows that warp owns, instead of being implicitly recomputed (via the
// per-code `-1` bias and a fresh `__half2float(xs[...])` load) once per row as before.
//
// Compile-time check FIRST, per this file's own standing rule (`nvcc -cubin -arch=sm_86 -Xptxas
// -v` + `cuobjdump --dump-sass`, no GPU execution) — and the result was a genuinely MIXED signal,
// not the clean win a naive "removing an instruction reduces instruction count" argument would
// predict, reported here honestly per the task's own directive to do so even when it dampens
// expectations:
//   * Registers/shared-mem: the `_small` kernels (k<=5120, attention/GDN path) picked up exactly
//     +1 register (39->40) with zero spill; the windowed large-K kernels (k=17408, FFN path) were
//     UNCHANGED (41/45 registers, identical smem). Neither shift changes any kernel's occupancy-
//     binding constraint (`_small`: floor(65536/(40*256))=6, same floor as 39 registers gave;
//     windowed kernels were already shared-mem-bound at 5, untouched by register count either way).
//   * SASS instruction count (`cuobjdump --dump-sass`, `pq2_0_gemv_f32io_small` as the
//     representative _small kernel): the `IADD3 ..., -0x1, RZ` per-code subtract (18 occurrences
//     in the compiled body) went to EXACTLY ZERO, confirming `ptxas` had NOT already collapsed the
//     shift+mask+subtract into something free (the concern this file's standing rule asks to check
//     first) — it was a genuine, separate instruction. But total instruction count went 360 -> 368
//     (+2.2%), NOT down: `FFMA` dropped 30->24 (-6, the redundant per-row x*code multiply-adds) but
//     `FADD` rose 10->25 (+15, the new shared-sum computation plus the `code_dot - gx.sum` step,
//     which the GPU ISA implements via `FADD` with a negated operand rather than a dedicated
//     `FSUB`). For the windowed large-K kernel (`pq2_0_gemv_f32io`), the same trade went the OTHER
//     direction: 392 -> 384 (-2.0%, a real reduction) with the identical IADD3 18->0 elimination.
//   * Net: a mixed, small-magnitude, DIRECTION-DEPENDENT compile-time signal (+2.2% instructions
//     for the k<=5120 kernels, -2.0% for the k=17408 kernels) — not the clean, unambiguous win this
//     section's opening rationale hoped for. Flagged explicitly, per the task's instruction to
//     report this honestly BEFORE trusting a real-hardware measurement, since the two kernel
//     families point in different directions and neither is dramatic.
//
// MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS: fresh baseline (pre-change, this commit's
// parent) 16.85-17.06 tok/s across 6 reps/2 runs (mean 16.98, `bench -p 64 -n 16`, RTX 3060) ->
// 17.12-17.52 tok/s across 11 reps/3 runs post-change (mean 17.30; one clear outlier rep at 15.45
// tok/s excluded — that single rep's decode time, 1035ms, was ~12% higher than every neighboring
// rep in the same run, 913-934ms, consistent with a transient system hiccup rather than a real
// regression, and was not reproduced when the same run configuration was repeated immediately
// after). A reproducible **+1.9% mean improvement**, smaller than the mixed/ambiguous compile-time
// signal above would have predicted in either direction, but real and clearly separated from the
// baseline distribution (baseline max 17.06 vs post-change min-excluding-outlier 17.12) across
// three independent `bench` invocations. Matches this file's established pattern: SASS/occupancy
// signals are directionally suggestive at best, never a substitute for a real end-to-end
// measurement — this is one of the rare cases in this investigation where a small, genuinely mixed
// compile-time signal still translated to a small, genuinely positive real-hardware result (contrast
// with the batch-8/shfl-broadcast/tail-wave/SwiGLU-fusion entries above, where clean-looking
// compile-time signals regressed real throughput).
//
// Scope decision: this file's own task framing explicitly allows stopping here if candidate (A)
// (this algebraic identity) delivers without "room to spare" for candidate (B) — a broader
// LUT/wider-bit-trick ternary decode reducing per-code instruction count further. Given (A)'s real
// win came in smaller (+1.9%) than the advisor's original 2-20% estimate for this pair of
// candidates, and given (B) carries the advisor's own flagged risk (per-lane divergent constant-
// memory addresses potentially serializing across LUT cache banks — NOT modeled or measured this
// session), (B) is deliberately NOT attempted here. Left as a documented, unimplemented follow-up
// candidate for a future session with fresh `ncu` access to first confirm how much ALU headroom (A)
// actually left before spending further risk budget on the higher-risk LUT approach.
//
// Granularity note for anyone revisiting this: `pq2_0_load_group_x` is hoisted to ONCE per (warp,
// group) — i.e. shared across the PQ2_0_ROWS_PER_WARP=2 rows one warp owns — NOT once per (block,
// group) across all 8 warps/16 rows in a block, even though the value is identical for every warp
// in the block (x does not depend on row at all). A block-level version was considered but
// deliberately NOT implemented: it would need a separate precompute pass (spreading the
// `PQ2_0_WINDOW_GROUPS`-worth of group sums across warps into a new `__shared__` array) plus at
// least one new `__syncthreads()` per window, which is exactly the "add synchronization for a
// modeled-but-unmeasured win" shape that has regressed real throughput five separate times earlier
// in this file's history (batch-8, register-hint, tail-wave-resize, scale-shuffle, SwiGLU-fusion).
// The warp-level version implemented here needed NO new shared memory and NO new synchronization at
// all — it only reorders which loop level a row-independent computation lives at — which is why it
// was chosen as this session's risk-appropriate scope. A true block-level version (8x less
// redundant computation instead of 2x) remains a real, larger, higher-risk follow-up candidate.
//
// ───────────────────────── TRIED AND REVERTED: F32-native staging half2 packing (#164, shared-memory bank-conflict "fix") ─────────────────────────
// A fresh `ncu --set full` pass (2nd advisor review round, see the `prismml-bonsai-model` project
// memory) flagged a real, SASS-reproducible shared-memory bank conflict on the F32-native staging
// store (`pq2_0_gemv_f32io`/`pq2_0_gemv2_f32io`/`_small` siblings, #161's `xs[base+0..3] =
// __float2half(v.x..w)` block): "2.1-way bank conflict across all shared store requests", Est.
// Speedup ~9.3-9.6%, concentrated on the FFN down-proj kernel (k=17408). The advisor's hypothesis,
// read from the SOURCE, was that the four separate scalar `half` stores per thread were each their
// own 2-byte store instruction, four times the necessary store-instruction count.
//
// Bank-index arithmetic worked through explicitly, per this file's standing rule, BEFORE writing any
// fix (thread i, one warp, lane = i mod 32; base = i*4 half-elements; byte offset for xs[base+0] is
// 8*lane, i.e. thread-to-thread stride = 8 bytes = 2 shared-memory words on sm_86's 32-bank/4-byte-
// wide layout): bank(lane) = floor(8*lane / 4) mod 32 = (2*lane) mod 32. For lane in [0,31] this
// hits only the 16 EVEN banks, each hit by exactly 2 lanes (lane and lane+16, e.g. bank 0 <- lanes 0
// and 16) — a genuine 2-way conflict, matching ncu's "2.1-way" (the ~0.1 excess plausibly averaging
// in partial tail-window iterations with fewer active threads). This DOES confirm a real conflict
// exists on the actual generated store address pattern — but critically, per-store-instruction
// arithmetic for xs[base+1]/[+2]/[+3] (the same formula, byte offsets 8*lane+2/+4/+6) gives the
// IDENTICAL bank set and lane-pairing, just shifted to odd banks for [+2]/[+3] — i.e. packing two of
// these four scalar stores into one wider store (`__floats2half2_rn` -> one `half2` write) targets
// the SAME two banks with the SAME lane pairing as the two stores it replaces, because the
// underlying stride between adjacent threads (2 words) — not the store's own width — is what
// determines the conflict. Working this through explicitly (as the task instructions demanded, "a
// naive repack might only halve instruction count without eliminating the conflict") predicted
// exactly that outcome before any code was written: halving instruction count from 4 to 2 stores,
// while the residual 2-way conflict on each remaining instruction is UNCHANGED.
//
// `cuobjdump --dump-sass` on the UNMODIFIED (pre-#164) kernel, checked before writing the fix per
// this file's own standing rule, revealed something the source-level reading above could not: ptxas
// had ALREADY auto-vectorized the four scalar `xs[base+k] = __float2half(...)` assignments into a
// SINGLE `STS.64` instruction per thread (three `F2FP.PACK_AB` float->half2 pack ops plus two `PRMT`
// byte-permutes to reassemble a 64-bit register pair, then one 8-byte store) — for BOTH
// `pq2_0_gemv_f32io_small` and the windowed `pq2_0_gemv_f32io`. The advisor's foundational premise
// (four separate 2-byte store instructions in the compiled kernel) was already false at the SASS
// level before this session started; the compiler had independently arrived at exactly the kind of
// wide-store vectorization a hand-written fix would aim for.
//
// Implemented anyway (`__floats2half2_rn(v.x, v.y)`/`__floats2half2_rn(v.z, v.w)` into two explicit
// `half2` stores) to test whether an EXPLICIT packing, rather than one ptxas happened to find, would
// still shift anything — numerically it is bit-identical to two independent `__float2half` calls
// (both round-to-nearest-even; `cuobjdump` shows the same `F2FP.PACK_AB` instruction class either
// way, confirmed by all correctness tests passing at the same ~1e-4-to-1e-3 tolerance band as the
// pre-existing F32-native tests). Compile-time result: register/shared-mem footprint UNCHANGED (40/
// 40/41/45 registers, 10304/10304/17472/17472 bytes smem — bit-identical to the unmodified kernel,
// confirmed via a diffed `nvcc -cubin -arch=sm_86 -Xptxas -v` pass before vs after), and — the
// decisive check — the generated `STS.64` for `xs[]` is the EXACT SAME single instruction, same
// operand pattern, in both versions (only two `F2FP.PACK_AB` instead of three, zero `PRMT`, replacing
// the compiler's own three-pack-plus-permute sequence with a cleaner two-pack sequence). Total SASS
// instruction count for the whole kernel body was IDENTICAL before/after (368 for `_small`, 384 for
// the windowed kernel) — the handful of real instructions removed (2 `PRMT`, 1 `F2FP.PACK_AB`, plus
// a few `IMAD`/`CALL.REL.NOINC`) were exactly offset by additional `NOP`/`MOV`/`LEA` the scheduler
// inserted elsewhere, a net wash at the static-instruction-count level, not merely "mixed" like the
// algebraic-ALU-reduction round above — the compiled kernel is, for all practical purposes, THE SAME
// kernel with different register naming.
//
// MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS: fresh fixed baseline (this commit's parent,
// `bench -p 64 -n 16`, RTX 3060) 15.98-17.65 tok/s across 9 reps/3 runs (one low outlier at 15.98,
// otherwise 16.97-17.65; median ~17.36) -> 17.20-17.65 tok/s across 6 reps/2 runs post-change (median
// ~17.48). The post-change distribution sits entirely INSIDE the baseline's own observed range (not
// separated from it the way every real win in this file's history has been) — a ~0.7% median shift
// that is fully explained by ordinary run-to-run noise, not a reproducible effect. This matches the
// SASS finding exactly: since the compiled `STS.64` and total instruction count are unchanged, no
// real speedup was ever mechanistically possible from this specific change, and none beyond noise was
// observed.
//
// Reverted in full (`git checkout -- native/kernels/pq2_0_gemv.cu native/ptx/pq2_0_gemv.ptx` back to
// this commit's parent) — no `half2`/`__floats2half2_rn` staging exists in the kernel as shipped.
//
// Root cause of why the flagged conflict survives untouched: the 2-way conflict is intrinsic to the
// THREAD-TO-SHARED-ADDRESS MAPPING (thread i owns shared elements [4i, 4i+4), a direct positional
// copy of global elements [4i, 4i+4), preserving the simple 1:1 correspondence every downstream
// reader — `pq2_0_load_group_x(xs, xb)` — relies on), not to how many store instructions realize that
// mapping. Any repacking of the SAME four values into fewer, wider stores keeps the same per-lane
// word stride (2 words) and therefore the same bank collisions; eliminating the conflict for real
// would require an INTERLEAVED/transposed shared layout (e.g. thread i's four loaded values scattered
// to xs[i], xs[i+w4], xs[i+2*w4], xs[i+3*w4] instead of xs[4i..4i+3]), which breaks the direct
// positional correspondence between `xs[]` and the windowed slice of `x[]` that every read site in
// the accumulate loop currently assumes — a restructuring that reaches into `pq2_0_load_group_x` and
// every one of its call sites across all 4 F32-native kernels (8 counting both windowed/`_small`
// variants), not a local, low-risk change. Left as a documented, NOT-recommended follow-up candidate:
// the underlying kernel was already reported "well-balanced" (compute/memory ~69-71%) by the same
// advisor round that flagged this conflict, this file's own calibration note says `ncu`'s `Est.
// Speedup` figures have overstated real gains by roughly 3-10x on every prior round, and the fix
// itself is large/invasive relative to the (likely low-single-digit-at-best) real payoff — don't
// re-attempt the interleaved-layout restructuring without new evidence that materially changes this
// cost/benefit balance (e.g. a future round finding this kernel latency-bound again, the one
// condition under which occupancy/access-pattern levers have actually paid off in this investigation).
//
// ───────────────────────── TRIED AND REVERTED: FP32 FMA-fusion reassociation (#164 candidate #1) ─────────────────────────
// The same round-2 advisor pass that flagged the bank-conflict "fix" above (SASS-proven no-op) also
// flagged `ncu`'s "Compute Workload Analysis" rule reporting an Est. Speedup of ~7-9% on every PQ2_0
// GEMV variant: "This kernel executes N fused and M non-fused FP32 instructions. Converting pairs of
// non-fused instructions to fused... could increase FP32 performance up to 20-24%." The hypothesis was
// that this is a side effect of #161's own algebraic ALU reduction, which reassociated
// `acc += (code-1)*x` into `acc += (Sum code*x - Sum x) * scale` — the final `(dot - sum) * scale`
// expression is a subtract feeding a multiply, which the advisor's premise (read from source, not SASS)
// assumed was NOT already fused into the accumulate.
//
// Per this file's own standing rule (verify SASS on the UNMODIFIED kernel before writing any fix,
// reinforced hard by the immediately-preceding bank-conflict entry above), `cuobjdump --dump-sass` on
// the two flagged highest-value targets (`pq2_0_gemv_f32io`, grid=320/k=17408, and
// `pq2_0_gemv2_f32io_small`, grid=2176) was inspected FIRST. Finding: `acc[rr] += (dot - gx.sum) * scale`
// was ALREADY compiled to exactly 2 instructions per row — `FADD R4, -R7, R4` (the subtract, dot-sum,
// using the GPU's free negated-operand add) immediately followed by `FFMA R14, R5, R4, R14` (multiply by
// scale AND accumulate into acc, in ONE fused instruction). The advisor's premise — that the final
// multiply-then-accumulate was NOT fused — was false: `ptxas` (nvcc's default `fmad=true`, unchanged for
// this file) had already fused it. The only "non-fused" FP32 instruction in this expression is the
// subtract itself, which has no adjacent multiply of its own to fuse with in the CURRENT algebraic form.
//
// Reassociating to two chained FMAs — `acc = fmaf(dot, scale, acc); acc = fmaf(-gx.sum, scale, acc);`
// (matching the advisor's suggested pattern, and this investigation's "safe" shape: pure arithmetic, no
// new sync/shared-mem) — was implemented and checked via SASS on a throwaway compile BEFORE touching the
// real file, per this file's standing rule. Result: the fused-multiply-accumulate step trades 1 FADD +
// 1 FFMA (2 instructions total) for 2 FFMAs (2 instructions total) — NO reduction in FP32-pipe
// instruction count. This is not a modeling error: on this architecture (and on NVIDIA GPUs generally,
// since Fermi) FADD/FMUL/FFMA execute at the IDENTICAL throughput per CUDA core (FADD and FMUL are
// hardware special-cases of the same FMA datapath, not separate cheaper units) — "fusing" an FADD+FFMA
// pair that's already down to 2 instructions into 2 FFMAs cannot reduce cycle count; it only changes
// which opcode is chosen for the same number of pipe slots. Confirmed directly: total FFMA+FADD count
// for the accumulate expression's contribution to the whole kernel was IDENTICAL before/after (49 either
// way: 24 FFMA + 25 FADD baseline vs 30 FFMA + 19 FADD reassociated, for both `pq2_0_gemv_f32io` and
// `pq2_0_gemv2_f32io_small`) — a straight FADD<->FFMA relabeling with zero net FP32-pipe cost, exactly
// the condition under which `ncu`'s FMA-fusion heuristic does not correspond to a real opportunity (the
// heuristic assumes any executed FADD/FMUL could halve into a shared FFMA slot with SOME multiply, which
// is not true when — as here — the nearby multiply is already spoken for by an existing FFMA).
//
// Total per-kernel SASS instruction count DID drop for 6 of the 8 production kernel variants (-8 to -16
// instructions, ~2-4%) — `pq2_0_gemv_f32io_small`/`pq2_0_gemv_f16in_small` (single-projection, small-K):
// 357->341 / 355->339; `pq2_0_gemv_f32io`/`pq2_0_gemv_f16in` (single-projection, windowed): 370->362 /
// 368->360; `pq2_0_gemv2_f32io_small`/`pq2_0_gemv2_f16in_small` (fused 2-way, small-K): 384->376 /
// 381->373. This came ENTIRELY from incidental scheduler-level side effects unrelated to the FP32 pipe
// (fewer LOP3.LUT/IMAD/IMAD.WIDE/BRA/NOP — plausibly a cheaper register-liveness/scheduling graph once
// `dot` no longer needs to survive past an explicit subtract), NOT from the FMA-fusion mechanism `ncu`
// named — confirmed by the FFMA+FADD sum being exactly unchanged (49=49) in every case, including these
// six. The two `pq2_0_gemv2_f32io`/`pq2_0_gemv2_f16in` (fused 2-way, WINDOWED — the dense FFN gate+up
// kernel, one of this file's largest decode-time contributors) showed ZERO total instruction change
// (389->389, 386->386) — even the incidental scheduling side effect did not materialize there. Registers
// only improved or stayed flat (single-projection kernels: -1 to -3 registers; fused 2-way: unchanged),
// zero spills throughout (`ptxas -v` before/after), and no occupancy-binding constraint moved in either
// direction (shared-mem/absolute-warp-count ceilings, not registers, still bind every affected kernel —
// unchanged from this file's earlier occupancy analysis).
//
// MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS: fresh baseline (this commit's parent, `bench -p
// 64 -n 16`, RTX 3060) 16.37-17.67 tok/s across 6 reps/2 runs (mean 17.28) -> 16.86-17.54 tok/s across 6
// reps/2 runs post-change (mean 17.29). The two distributions FULLY OVERLAP (baseline max 17.67 exceeds
// post-change max 17.54; baseline min 16.37 is below post-change min 16.86) and the means are
// indistinguishable (17.28 vs 17.29) — unlike every real win in this file's history, which showed
// non-overlapping distributions. This matches the SASS finding precisely: the actual FMA-fusion
// mechanism `ncu` named is provably throughput-neutral on the FP32 pipe (49=49 instructions either way),
// and the small (~2-4%, ~8-16 instructions out of 340-390) incidental instruction-count reduction that
// DID occur in 6 of 8 kernels is far below this benchmark's observed noise floor (this file's own prior
// rounds have repeatedly needed changes an order of magnitude larger, e.g. #159's 7.6% or #161's
// F32-native fusion's 8%, to clear the noise floor with a visibly separated distribution).
//
// Reverted in full (`git checkout` back to this commit's parent) — no `fmaf()` calls exist in the
// accumulate expression as shipped; `acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;` is unchanged
// at all 8 call sites.
//
// **Verdict for whoever revisits this**: `ncu`'s FMA-fusion "Est. Speedup" rule is NOT a real,
// actionable opportunity for this kernel family — it is provably mischaracterizing an already-fused
// multiply-accumulate (confirmed at the SASS level, not inferred) as a fusible pair, the same class of
// false positive this file has now documented twice in a row (see the bank-conflict entry immediately
// above — ptxas already auto-vectorized the flagged stores; here, ptxas already auto-fused the flagged
// multiply-add). Any future PQ2_0 GEMV work chasing `ncu`'s raw "Est. Speedup" percentages should
// disassemble the CURRENT compiled kernel first, every time, rather than trusting the metric's premise
// about what the compiler has or hasn't already done — the premise has now been wrong twice
// consecutively on this exact kernel family.
//
// ───────────────────────── TRIED AND REVERTED: software-pipelined weight-byte prefetch (#166) ─────────────────────────
// A fresh round-2 advisor pass (see the `prismml-bonsai-model` project memory) found the LAST
// untried candidate on `pq2_0_gemv2_f32io_small` (the FFN gate+up fused kernel, grid=2176): a
// genuine, repeated L1TEX scoreboard-stall flag ("5.2-5.4 cycles stalled waiting on a scoreboard
// dependency on L1TEX") DESPITE 96.4-96.8% achieved occupancy against a 100% theoretical ceiling —
// i.e. occupancy-based latency hiding is exhausted here, yet the stall persists. Proposed lever:
// manual double-buffering of each row's per-group scale+code global loads, so group g+1's loads are
// ISSUED before group g's already-loaded values are consumed by the accumulate FFMA, giving the
// L1TEX round-trip a full loop iteration to complete instead of stalling immediately before use.
//
// Step 1, per this file's own standing rule: `cuobjdump --dump-sass` on the CURRENT, unmodified
// kernel BEFORE writing any fix. Finding: `ptxas` had ALREADY 2x-unrolled the `for (g...)` group
// loop (confirmed by the loop-counter decrement `IADD3 R22, R22, -0x2` and a single backward branch
// covering BOTH groups' worth of loads+compute in one straight-line block) and, within that unrolled
// pair, issues most of both groups' scale/code `LDG` loads clustered early in the block, well before
// the bulk of the `FFMA`/`FADD` chain that consumes them — a real, if shallow (~1-group-deep),
// software-pipelining effect the compiler was already producing on its own via unroll+list-schedule,
// not a stall the compiler was doing nothing about.
//
// Step 3, occupancy modeling BEFORE implementing in earnest: `nvcc -Xptxas -v` on the unmodified
// kernel confirmed the premise's numbers were current — `pq2_0_gemv2_f32io_small` compiles to 40
// registers, 10304 bytes smem, zero spill. Occupancy arithmetic (sm_86: 65536 registers/SM, 102400
// bytes shared mem/SM, 1536 threads/SM max, 256-thread blocks = 8 warps/block): registers ->
// floor(65536/(40*256))=6; shared mem -> floor(102400/10304)=9; warps (ABSOLUTE ceiling, 1536/256)
// -> 6. Binding constraint = min(6,9,6,16) = 6 blocks/SM, TIED between registers and the absolute
// warp-count ceiling — i.e. this kernel is already at the hard maximum (48 warps/SM = 100%
// theoretical occupancy), matching the advisor's 96.4-96.8% achieved figure. Headroom before a
// register increase would drop this to 5 blocks/SM (83.3%, a real regression): only 2 registers
// (40 -> 42 stays at 6; 43 drops to 5) — a thin margin, exactly the risk this task's framing flagged
// up front. This alone did not conclusively rule the candidate out (2 registers isn't obviously zero
// headroom), so a throwaway compile check followed rather than stopping on hand arithmetic alone.
//
// Throwaway compile check (a standalone `.cu` file, NOT this file, compiled with `nvcc -Xptxas -v`
// only — no GPU execution, matching the FMA-fusion candidate's precedent for de-risking a change
// before wiring it in for real): a minimal manual double-buffer of `scaleBuf[2][ROWS_PER_WARP]`/
// `codeBuf[2][ROWS_PER_WARP]`, preloading group 0 before the loop and prefetching group g+1 at the
// top of each iteration before consuming group g's buffered values, compiled to **39 registers** (one
// FEWER than baseline, not more — the naive "doubling live state roughly doubles registers" estimate
// was WRONG, in the opposite direction from what the task's framing worried about) with zero spill,
// unchanged 10304 bytes smem. But total SASS instruction count for the whole kernel body rose 400 ->
// 440 (+10%), and the loop structure changed qualitatively: a genuine backward-branching loop
// handling ONE group per iteration (not ptxas's own 2x-unrolled pair), confirmed via `cuobjdump
// --dump-sass` on the throwaway compile. A MIXED signal — registers/occupancy fine or better, but
// real added instruction count and a materially different (single-group, real-branch) loop shape —
// not a clean rule-out by modeling alone, so per this task's own decision tree the candidate was
// implemented for real rather than stopped here.
//
// Wired into `pq2_0_gemv2_f32io_small` itself (the exact kernel the advisor flagged, confirmed by
// grid=2176 = ceil(2*17408/16), Bonsai-27B's dense FFN gate+up projection). `nvcc -Xptxas -v` on the
// REAL file after the change reproduced the throwaway compile's numbers exactly: 39 registers, 10304
// bytes smem, zero spill — occupancy-binding constraint UNCHANGED at 6 blocks/SM (100% theoretical,
// same as baseline; if anything 1 more register of headroom than before, 3 vs 2). Every OTHER kernel
// variant in the file (`pq2_0_gemv_f32io_small`, the two windowed large-K kernels, the F16-native
// kernels, the F32 CPU-reference kernel) compiled to BIT-IDENTICAL register/smem footprints before
// and after — confirming the change's compile-time footprint was correctly scoped to only the one
// targeted kernel.
//
// Correctness: full CUDA test suite (312 passed / 0 failed / 39 skipped) including
// `CudaQwen3HybridDenseRealGgufSmokeTest` (the #162 prefill regression guard, run for real against
// the Bonsai-27B GGUF fixture, not skipped) and the pre-existing `CudaGraphCaptureEquivalenceTest`
// isolation flake (passed clean in this run, no recurrence) — no correctness regression from this
// change at any point.
//
// MEASURED DECODE THROUGHPUT ON REAL BONSAI-27B WEIGHTS: fresh baseline (this commit's parent,
// `bench -p 64 -n 16`, RTX 3060) 17.19-17.36 tok/s across 6 reps/2 runs (median ~17.3, consistent
// with the ~17.2-17.65 tok/s this investigation has held at since #164) -> **15.32-15.75 tok/s across
// 6 reps/2 runs post-change (median ~15.6)** — a reproducible **~9-10% REGRESSION**, and the two
// distributions do NOT overlap at all (baseline's worst rep, 17.19, still beats the prefetch
// version's best rep, 15.75). This is the NINTH negative result in this investigation's history, and
// notably a DIFFERENT failure shape than the two immediately preceding it (bank-conflict and
// FMA-fusion, both proven SASS-level NO-OPS with zero measurable throughput change either way): this
// change was a real, substantive SASS-level restructuring (confirmed non-no-op via the instruction-
// count and loop-shape differences above) that nonetheless made real throughput measurably WORSE, not
// neutral — closer in shape to the batch-8/SwiGLU-fusion class of failure (a real change, real cost,
// negative payoff) than to the bank-conflict/FMA-fusion class (no real change at all).
//
// Root cause hypothesis (not `ncu`-counter-confirmed — no profiler access in the session that ran
// this experiment): the manual double-buffer's explicit `cur`/`next` index swap forces a true,
// single-group-per-iteration loop with one real backward branch, REPLACING the 2x-unrolled,
// straight-line-scheduled loop `ptxas` had already built on its own (see the Step 1 SASS finding
// above) — which was ALREADY clustering both unrolled groups' loads early and interleaving them with
// the compute chain, a shallower but branch-free form of the same latency-hiding idea this fix tried
// to add explicitly. Forcing an explicit, deeper (1-iteration-ahead) prefetch distance also forces a
// REAL loop (double the branch count vs the unrolled baseline, since branches are now paid per group
// instead of per group-pair) and a loop-carried buffer-index dependency the scheduler has less
// freedom to hide across than the unrolled body's independent, staticaly-scheduled instruction
// stream. In other words: this is the same "the compiler had already captured most of the achievable
// benefit via a cheaper mechanism (unrolling), and an explicit manual version paid a real structural
// cost (branch count, loop-carried state) to chase the SAME latency-hiding effect less efficiently" —
// a genuinely new variant of this file's now-familiar lesson, distinct from the SASS-no-op cases:
// here the compiler's existing solution wasn't merely equivalent to the proposed fix, it was
// ACTIVELY BETTER, and replacing it with an explicit version cost real performance.
//
// Reverted in full (`git checkout -- native/kernels/pq2_0_gemv.cu native/ptx/pq2_0_gemv.ptx`) — no
// `scaleBuf`/`codeBuf`/manual double-buffering exists in `pq2_0_gemv2_f32io_small` as shipped; the
// original single-buffered `for (g...) { gx = ...; for (rr...) { scale = ...; p = ...; acc += ...; } }`
// loop is unchanged.
//
// **Status after this candidate**: this was explicitly the LAST remaining, previously-unexplored
// candidate flagged by the round-2 advisor review on this kernel family (see the earlier
// bank-conflict and FMA-fusion entries' own "closest thing to unexplored territory left" framing) —
// every `ncu`-flagged item from that round is now tried-and-reverted (bank-conflict, FMA-fusion,
// this prefetch candidate) or previously deferred with documented reasoning (tail-wave resize on this
// same kernel family, LUT-style deeper decode). This is a strong, well-substantiated stopping point
// for the PQ2_0 GEMV kernel-level investigation on `pq2_0_gemv2_f32io_small` specifically — a future
// session would need genuinely new profiling evidence (real `ncu` L1TEX/scheduler counters, not
// available in this session) before re-attempting any occupancy/latency-hiding lever on this kernel.
//
// **Fresh `ncu --set full` evidence obtained 2026-07-27 (elevated, RTX 3060, real Bonsai-27B decode
// launches) — CONFIRMS the stopping point above, no reopening warranted.** The large/dominant
// decode-path GEMV launches (`pq2_0_gemv_f32io` grid=320, `pq2_0_gemv2_f32io_small`/
// `pq2_0_gemv_f32io_small` grid=384-2176 — i.e. the dense-FFN gate/up/down and attention K/V/O
// projections that actually dominate decode time) are already well-occupied: 71-97% Achieved
// Occupancy, 54-82% Compute/Memory Throughput, Waves Per SM 2.29-12.95. This is a healthy profile —
// nothing here flags a "grid too small" or latency-bound pathology, matching every prior negative
// result's conclusion that this kernel family is close to its practical ceiling at these shapes.
// One genuinely new, narrow finding: a `pq2_0_gemv2_f32io_small` launch with grid=(6,1,1) — a much
// smaller output-row count than the dominant launches above, almost certainly one of the small
// per-layer GDN gating projections (alpha/beta, NVHead-wide output) rather than a dense FFN/attention
// bank — DOES show the familiar "grid too small" shape (Waves Per SM 0.04, Compute/Memory Throughput
// 4.15%, matching `attention_f32`'s pathology documented in `docs/CUDA.md`'s Future Work). Given its
// absolute duration (21.79 us per launch) is tiny next to the dominant GEMV launches' 77-408 us, any
// fix here has a small absolute ceiling — worth a look only as a low-effort, low-risk follow-up
// (e.g. batching multiple small per-layer projections into fewer, larger launches), not worth
// reopening the broader kernel-level investigation over.
//
// ───────────────────────── #244: re-investigated, found ALREADY IMPLEMENTED — no code change ─────────────────────────
// Issue #244 (filed 2026-07-31, seeded by a fresh re-profile at
// `.perf-runs/ncu-2026-07-30-ffn-gemv-post161/README.md`) proposed the identity
// `acc = Sum(raw_code * x) - Sum(x)` for the ternary unpack/accumulate loop, citing the
// 2026-07-22 advisor review's "algebraic ALU reduction" candidate as still untried post-#161.
// That premise was WRONG: this is the exact same candidate as the "Algebraic ALU reduction
// (#161 continued, advisor candidate #4/5)" section above, same identity, same file — and it was
// already implemented and shipped on 2026-07-22, commit 7c7101c ("perf(cuda): algebraic ALU
// reduction in PQ2_0 GEMV decode loop (#161)"), nine days before #244 was filed. Every production
// kernel in this file (`pq2_0_gemv_f16in`/`_small`, `pq2_0_gemv2_f16in`/`_small`,
// `pq2_0_gemv_f32io`/`_small`, `pq2_0_gemv2_f32io`/`_small` — 8 entry points, confirmed by
// grepping for `pq2_0_code_dot(p, gx)`) already calls `pq2_0_load_group_x`/`pq2_0_code_dot`
// exactly as described above; only `pq2_0_gemv_f32in` (the deliberately-untouched CPU-vs-GPU
// exact-reference kernel, see the file's v1/v2 note near the top) still uses the older per-code
// `code - 1` decode. The `.perf-runs` README that seeded #244 re-ran `ncu` against these ALREADY-
// REDUCED kernels and correctly measured 75.10% compute-bound headroom on the dominant FFN
// kernel — but that number describes headroom REMAINING AFTER this identity, not evidence the
// identity itself was still unapplied; nobody checked the kernel source before writing the issue.
//
// No functional change was made for #244 — instead, the already-shipped implementation was
// independently re-verified fresh:
//   * Correctness: full `CudaPQ2_0GemvTest` suite, 21/21 passed on real hardware (RTX 3060, driver
//     present, nothing skipped), including `PQ2_0GemvF32Native_MatchesCpuFloatReference` and
//     `PQ2_0GemvFusedDecodeF32Native_MatchesSeparateLaunches` at real Bonsai-27B dims (n=512/37/3,
//     k=5120/17408/5248) — max abs diff stays within the existing 5e-2/1e-2 tolerance bar. That
//     bar is NOT new and is NOT specific to the algebraic identity: it was set by `xs[]` staying
//     `half`-precision internally (unchanged either way), the same bar `PQ2_0GemvF16In_...`
//     already used before commit 7c7101c existed. This session did not need to characterize a new
//     tolerance because none was introduced.
//   * Benchmark: fresh `bench --device cuda -p 64 -n 16 -r 3` against the real
//     `Ternary-Bonsai-27B-Q2_0.gguf`, RTX 3060, GPU otherwise idle (~800 MiB baseline VRAM) ->
//     18.30-18.40 tok/s decode (median 18.37, best 18.40; prefill 101-102 tok/s). Consistent with,
//     and modestly ahead of, the ~17.2-18.4 tok/s range this file's history has held at since the
//     algebraic reduction landed (commit 7c7101c's own recorded +1.9% mean win, 16.98 -> 17.30
//     tok/s, persisted through every subsequent negative-result round in this file without
//     regressing) — no throughput regression, nothing to revert.
//
// Recommendation for whoever triages #244: close as a duplicate of the work already merged under
// #161 (commit 7c7101c), referencing this note. The only remaining untried candidate in this
// specific vein is candidate (B) from the "Algebraic ALU reduction" section's own "Scope decision"
// paragraph above (a broader LUT/wider-bit-trick ternary decode) — deliberately not attempted
// here either, for the same reasons already given there (advisor-flagged per-lane divergent
// constant-memory risk, not modeled/measured, and identity (A)'s real win already came in under
// the advisor's original estimate). Also worth doing: updating the `prismml-bonsai-model` project
// memory to record this candidate as DONE rather than pending, so a third session doesn't
// rediscover the same "untried candidate" framing a second time.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2_0_GROUP_SIZE  128
#define PQ2_0_GROUP_BYTES 34

// Rows handled per warp / per block for the v2 F16 kernel. Mirrors I2_S's tuned choice
// (I2S_ROWS_PER_WARP=2) — amortizes the shared-x stage and grid size 16x vs one-row-per-warp.
// TUNING EXPERIMENT (2026-07-21): tried 4 after v3's warp-cooperative group reads made the
// weight reads coalesced, hypothesizing the ILP-vs-occupancy tradeoff that picked 2 for I2_S
// might shift. Measured WORSE on real Bonsai-27B weights (decode 10.52 -> 9.91 tok/s, -5.8%,
// 3-rep/16-token benchmark) — reverted to 2. Fewer warps resident per SM at ROWS_PER_WARP=4
// apparently costs more in occupancy/latency-hiding than it gains in per-warp reduction
// overhead amortization. Left as documented negative result — don't re-try without new
// evidence.
#define PQ2_0_ROWS_PER_WARP  2
#define PQ2_0_ROWS_PER_BLOCK (8 * PQ2_0_ROWS_PER_WARP)   // 8 warps/block × rows-per-warp

// Largest K across Bonsai-27B's PQ2_0 projections (the FFN down-projection's input dim =
// intermediateSize = 17408). 17408 halfs = 34 KB, under sm_86's 48 KB static shared cap.
// Mirrors I2S_MAX_K's precedent (i2_s_gemv.cu) — a future PQ2_0 model with larger K would need
// this raised (no runtime bounds check, matching I2_S's existing convention).
#define PQ2_0_MAX_K 17408

// Small-K specialization bound (occupancy follow-up, #157 round 4) — see the file-header section
// above for the full occupancy-arithmetic rationale. Exactly Bonsai-27B's attention/GDN input dim
// (qwen35.embedding_length=5120); the `_small` kernel variants below are only valid for k <= this
// value (no runtime bounds check, same convention as PQ2_0_MAX_K/I2S_MAX_K above).
#define PQ2_0_MAX_K_SMALL 5120

// Window size for the windowed-activation-staging fix (#159) applied to pq2_0_gemv_f16in /
// pq2_0_gemv2_f16in below — see the file-header "Windowed activation staging" section for the
// full derivation. Exactly half of k=17408's groups_per_row=136, giving 2 equal windows for the
// one real production FFN shape this kernel pair serves.
#define PQ2_0_WINDOW_GROUPS 68
#define PQ2_0_WINDOW_ELEMS  (PQ2_0_WINDOW_GROUPS * PQ2_0_GROUP_SIZE)   // 8704 elements = 17408 bytes

// Byte offset from a split-layout tensor's base to the start of its codes region. Must match the
// identical helper in pq2_0_repack.cu and dequant_pq2_0.cu — see pq2_0_repack.cu's file header
// for the round-up-to-32 rationale (guarantees alignment regardless of totalGroups' parity).
__device__ __forceinline__ size_t pq2_0_codes_base_offset(long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

__device__ __forceinline__ float pq2_0_warp_reduce(float acc)
{
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    return acc;
}

// ───────────────────────── Algebraic ALU reduction (#161 continued, advisor candidate #4/5) ─────────────────────────
// See the file-header "Algebraic ALU reduction" section for the full derivation. Replaces the old
// `pq2_0_accum_byte` (per-row: decode `code-1` via an explicit shift+mask+IADD3, load+convert x
// from `xs[]`, accumulate) with two pieces split by what actually varies per row:
//   * `pq2_0_load_group_x` loads and sums the 4 activations {xb,xb+32,xb+64,xb+96} for byte-lane
//     `xb` — this depends ONLY on `xs`/`xb` (the shared activation staging buffer and the
//     lane/group position within it), NEVER on a row's weight bits or scale. Call ONCE per
//     (warp, group) and reuse across all PQ2_0_ROWS_PER_WARP rows that warp owns, instead of once
//     per (row, group) as the old code implicitly did (each row's own accum_byte call reloaded
//     and re-summed the SAME xs[] elements).
//   * `pq2_0_code_dot` decodes byte `p`'s 4 RAW (unbiased, 0/1/2) codes and computes their
//     dot-product against the already-loaded x values — this DOES vary per row (each row has its
//     own weight byte `p`), but no longer needs a `-1` bias subtract per code: the algebraic
//     identity `Sum (code_i - 1)*x_i = Sum code_i*x_i - Sum x_i` moves the `-1` term out to the
//     row-independent `gx.sum` computed above, applied ONCE per row as a single `code_dot - gx.sum`
//     subtract instead of 4 per-code IADD3 instructions.
// Net effect per (warp, group, lane), confirmed via `cuobjdump --dump-sass` instruction counts
// (see file header): removes 4 IADD3 (per-code `-1`) and 4 redundant `__half2float` conversions
// per extra row beyond the first, at the cost of 3 FADD (the shared sum) + 1 FSUB per row — a net
// SASS instruction reduction for the PQ2_0_ROWS_PER_WARP=2 case used by every kernel below.
struct Pq2_0GroupX
{
    float x0, x1, x2, x3, sum;
};

__device__ __forceinline__ Pq2_0GroupX pq2_0_load_group_x(const half* xs, int xb)
{
    Pq2_0GroupX gx;
    gx.x0 = __half2float(xs[xb]);
    gx.x1 = __half2float(xs[xb + 32]);
    gx.x2 = __half2float(xs[xb + 64]);
    gx.x3 = __half2float(xs[xb + 96]);
    gx.sum = gx.x0 + gx.x1 + gx.x2 + gx.x3;
    return gx;
}

__device__ __forceinline__ float pq2_0_code_dot(unsigned int p, const Pq2_0GroupX& gx)
{
    unsigned int c0 = (p >> 6) & 0x3;
    unsigned int c1 = (p >> 4) & 0x3;
    unsigned int c2 = (p >> 2) & 0x3;
    unsigned int c3 =  p       & 0x3;
    return (float)c0 * gx.x0 + (float)c1 * gx.x1 + (float)c2 * gx.x2 + (float)c3 * gx.x3;
}

// ───────────────────────── F32 activations/output — exact-match CPU-vs-GPU validation twin ─────────────────────────
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_f32in(
    const uint8_t* __restrict__ weight,   // [n x rowBytes] rowBytes = (k/128)*34
    const float*   __restrict__ x,        // [k]
    float*         __restrict__ y,        // [n]
    const int n,
    const int k)
{
    const int groups_per_row = k / PQ2_0_GROUP_SIZE;
    const int row_bytes      = groups_per_row * PQ2_0_GROUP_BYTES;

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;

    for (int row = blockIdx.x * warps_per_block + wid; row < n; row += gridDim.x * warps_per_block)
    {
        const uint8_t* row_ptr = weight + (size_t)row * row_bytes;
        float acc = 0.0f;

        for (int g = lane; g < groups_per_row; g += warpSize)
        {
            const uint8_t* group_base = row_ptr + (size_t)g * PQ2_0_GROUP_BYTES;
            const float scale = __half2float(*reinterpret_cast<const half*>(group_base));
            const uint8_t* codes = group_base + 2;
            const int out_base = g * PQ2_0_GROUP_SIZE;

            float group_acc = 0.0f;
            #pragma unroll
            for (int gp = 0; gp < 32; gp++)
            {
                uint8_t p = codes[gp];
                int c0 = ((p >> 6) & 0x3) - 1;
                int c1 = ((p >> 4) & 0x3) - 1;
                int c2 = ((p >> 2) & 0x3) - 1;
                int c3 = ( p       & 0x3) - 1;
                group_acc += (float)c0 * x[out_base + gp]
                           + (float)c1 * x[out_base + gp + 32]
                           + (float)c2 * x[out_base + gp + 64]
                           + (float)c3 * x[out_base + gp + 96];
            }
            acc += group_acc * scale;
        }

        acc = pq2_0_warp_reduce(acc);
        if (lane == 0) y[row] = acc;
    }
}

// ───────────────────────── F16 activations/output — production decode path (v3) ─────────────────────────
// v2 (above rationale) fixed shared-x staging and grid sizing but still distributed GROUPS
// across lanes (`for g = lane; g < groups_per_row; g += warpSize`) — each lane then decoded
// its OWN group's 32 code bytes as a private scalar loop. That means at any instruction, the
// 32 lanes of a warp are reading 32 DIFFERENT groups, `PQ2_0_GROUP_BYTES` (34) apart — the same
// "34-byte stride between lanes" uncoalesced pattern the v1 kernel had, just now applied once
// per group instead of once per element.
//
// v3 restructures the loop nesting so the WARP cooperates on one group at a time instead of
// each lane owning whole groups: the group loop is now a plain sequential loop every lane
// executes together, and within each iteration lane L reads code byte `L` of that group
// (`group_base[2 + lane]`) — 32 lanes reading 32 CONSECUTIVE bytes, a single coalesced
// transaction. Byte `L`'s decode target in dotLLM's PQ2_0 bit-interleave is elements
// `{L, L+32, L+64, L+96}` of the group (see the file-header layout note) — i.e. exactly
// `xb = out_base + lane`, so the per-byte decode helper is called with `lane` in place of
// the old per-lane `gp` loop variable (see this file's later "Algebraic ALU reduction" section,
// #161, for that helper's current form — originally `pq2_0_accum_byte`, since replaced by
// `pq2_0_load_group_x`/`pq2_0_code_dot`). The redundant per-lane read of the group's 2-byte scale
// (same address for all 32 lanes) is a hardware broadcast, not a coalescing concern. Total
// weight-byte traffic per warp is unchanged (`groups_per_row * 32` either way) — this is a pure
// access-pattern reorganization, not a change to total bytes read. The warp reduction moves
// from "not needed" (v2 had none — each lane fully owned its groups) to a single reduction at
// the very end of the whole row (not per-group), keeping shuffle overhead low.
// ───────────────────────── Output write: staged block-coalesced store (#157) ─────────────────────────
// ncu (--set full) on the 3060 flagged the tail write of the (pre-fix) kernel as the single
// biggest inefficiency in the whole profile: each warp reduced PQ2_0_ROWS_PER_WARP values and
// wrote them via lane 0 only — a single-thread 2-byte scalar store per row, occupying a full
// 32-byte global-memory sector transaction for 2 useful bytes (MemoryCacheAccessPattern: ~6%
// sector efficiency, Estimated Speedup ~51.56%, the largest single number in the profile).
// PQ2_0_ROWS_PER_BLOCK (16) such lane-0 stores were scattered per block instead of one
// block-wide write.
//
// Fix: each warp stages its reduced half results into a small shared buffer (rowOut[16] = 32
// bytes total — negligible next to the 34 KB xs[] staging buffer, does not move the occupancy
// ceiling). After a block-wide __syncthreads(), the first PQ2_0_ROWS_PER_BLOCK threads (lanes
// 0..15 of warp 0) perform ONE coalesced write of up to 16 contiguous halfs to y[] — a single
// 32-byte sector, except on the tail block where n isn't a multiple of 16 (guarded per-lane with
// `row < n`, which only breaks perfect coalescing on that last partial block).
//
// Correctness note: the early "skip this warp" path used to be a `return` guarding the whole
// warp from ANY out-of-range row. That can no longer be a `return` — every thread in the block
// must reach the new __syncthreads() below, so out-of-range warps instead skip only the
// accumulate/stage step via the `warpActive` guard and fall through to the sync + write. Any
// shared rowOut[] slot that stays unwritten (because its owning warp was entirely inactive)
// corresponds to a row >= n, which the final `row < n` check guarantees is never read.
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_f16in(
    const uint8_t* __restrict__ weight,   // split layout — see file header's "Split-layout addressing" note
    const half*    __restrict__ x,
    half*          __restrict__ y,
    const int n,
    const int k)
{
    // Windowed activation staging (#159) — see file header's "Windowed activation staging"
    // section for the full derivation/sync-count modeling. xs[] now holds only
    // PQ2_0_WINDOW_GROUPS groups' worth of x (8704 elements/17408 bytes) instead of the whole row
    // (up to PQ2_0_MAX_K=17408 elements/34816 bytes) — staged and re-staged once per window in
    // the loop below, which needs groups_per_row/rowBase/warpActive computed up front (unlike the
    // pre-#159 version, which staged before any of that was known). `x`/`xs` alignment reasoning
    // for the uint4 staging is unchanged from the pre-#159 version (see git history of this file
    // for the full argument): `xs` via __align__(16) below, `x` via CUDA's device-allocation
    // minimum 256-byte alignment plus the window offset being a multiple of PQ2_0_GROUP_SIZE*2=256
    // bytes (128 halfs), comfortably 16-byte aligned.
    __shared__ __align__(16) half xs[PQ2_0_WINDOW_ELEMS];

    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;

    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2_0_codes_base_offset(total_groups);

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < n;

    __shared__ half rowOut[PQ2_0_ROWS_PER_BLOCK];

    // Declared unconditionally (unlike the pre-#159 version's warpActive-scoped locals): the
    // window loop's stage step below must run on EVERY thread regardless of warpActive (staging
    // distributes x across all 256 threads, not just active warps' threads), so rows[]/acc[] need
    // to survive across window iterations outside any warpActive-only scope. `min(rowBase+rr,
    // n-1)` is safe even when warpActive is false (n is always >= 1) — only ever READ below
    // inside the `if (warpActive)` guard, matching the pre-#159 version's actual values exactly
    // when warpActive is true.
    int   rows[PQ2_0_ROWS_PER_WARP];
    float acc[PQ2_0_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
    {
        rows[rr] = min(rowBase + rr, n - 1);   // clamp tail rows; their result is discarded below
        acc[rr] = 0.0f;
    }

    for (int wStart = 0; wStart < groups_per_row; wStart += PQ2_0_WINDOW_GROUPS)
    {
        const int wGroups = min(PQ2_0_WINDOW_GROUPS, groups_per_row - wStart);
        const int wElems  = wGroups * PQ2_0_GROUP_SIZE;   // always a multiple of 8 (128 | wElems)

        // Vectorized staging (#157), applied per-window: stage only this window's x slice.
        {
            const uint4* x4  = reinterpret_cast<const uint4*>(x + (size_t)wStart * PQ2_0_GROUP_SIZE);
            uint4*       xs4 = reinterpret_cast<uint4*>(xs);
            const int w8 = wElems >> 3;
            for (int i = threadIdx.x; i < w8; i += blockDim.x)
                xs4[i] = x4[i];
        }
        __syncthreads();   // RAW — this window's stage must finish before any read below

        if (warpActive)
        {
            for (int gi = 0; gi < wGroups; gi++)
            {
                const int g        = wStart + gi;           // global group index (scales/codesBase)
                const int out_base = gi * PQ2_0_GROUP_SIZE; // LOCAL to this window's xs

                // Loaded/summed ONCE per (warp, group), reused across both rows below — see the
                // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
                const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

                #pragma unroll
                for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
                {
                    const long gFlat = (long)rows[rr] * groups_per_row + g;
                    float scale = __half2float(scales[gFlat]);   // lane-independent address — warp broadcast, see "Round 4" file-header note
                    uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced — see file header

                    acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
                }
            }
        }

        // WAR — before the NEXT window overwrites xs, every thread must be done reading this
        // window's contents. Skipped on the last window: nothing stages into xs again, so no
        // WAR hazard exists there (see file header's sync-count derivation — the separate,
        // pre-existing sync below protects a DIFFERENT hazard, on rowOut, not xs).
        if (wStart + PQ2_0_WINDOW_GROUPS < groups_per_row)
            __syncthreads();
    }

    if (warpActive)
    {
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = __float2half(a);
        }
    }

    __syncthreads();   // RAW on rowOut — unrelated to xs/windowing, unchanged from the pre-#159 kernel

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int row = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (row < n) y[row] = rowOut[threadIdx.x];
    }
}

// ───────────────────────── F16 fused 2-way projection — shared x read across both ─────────────────────────
// Virtual row-concatenation of weight0/weight1 (rows [0,n0) then [0,n1)), same k for both. Used for
// any decode-time PQ2_0 pair sharing one input: dense FFN gate+up, or full-attention K+V. Mirrors
// i2_s_gemv2_f16in — see that kernel's comments for the row-selection / tail-clamp rationale.
// Same staged block-coalesced write fix as pq2_0_gemv_f16in above (#157). Here the block's 16
// virtually-concatenated rows can straddle the n0/n1 boundary between the two output arrays, so
// the final write routes each lane to y0 or y1 based on its global row index — same routing as
// before, just performed as part of the batched write instead of 8 separate lane-0 stores. A
// block that straddles the boundary splits into two smaller coalesced writes (one run into y0,
// one into y1) instead of one — still a large improvement over independent scalar stores, and
// correctness (not maximal coalescing) is what matters here.
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv2_f16in(
    const uint8_t* __restrict__ weight0,   // split layout — each of weight0/weight1 has its OWN codesBase (own n)
    const uint8_t* __restrict__ weight1,
    const half*    __restrict__ x,
    half*          __restrict__ y0,
    half*          __restrict__ y1,
    const int n0,
    const int n1,
    const int k)
{
    // Windowed activation staging (#159) — see pq2_0_gemv_f16in's identical comment above and the
    // file header's "Windowed activation staging" section for the full derivation.
    __shared__ __align__(16) half xs[PQ2_0_WINDOW_ELEMS];

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1;
    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < totalN;

    const int groups_per_row = k / PQ2_0_GROUP_SIZE;

    // Each virtually-concatenated array is a physically separate split-layout tensor, so each
    // gets its own scales/codes split point derived from its OWN row count (n0 vs n1) — see
    // pq2_0_gemv_f16in's "Split-layout addressing" file-header note.
    const half*    scales0    = reinterpret_cast<const half*>(weight0);
    const half*    scales1    = reinterpret_cast<const half*>(weight1);
    const uint8_t* codesBase0 = weight0 + pq2_0_codes_base_offset((long)n0 * groups_per_row);
    const uint8_t* codesBase1 = weight1 + pq2_0_codes_base_offset((long)n1 * groups_per_row);

    __shared__ half rowOut[PQ2_0_ROWS_PER_BLOCK];

    // Declared unconditionally — same reasoning as pq2_0_gemv_f16in above (must survive across
    // window iterations; staging needs every thread regardless of warpActive). `min(rowBase+rr,
    // totalN-1)` is safe even when warpActive is false (totalN = n0+n1 >= 1 for any real call).
    const half*    rowScales[PQ2_0_ROWS_PER_WARP];
    const uint8_t* rowCodesBase[PQ2_0_ROWS_PER_WARP];
    int            localRows[PQ2_0_ROWS_PER_WARP];
    float          acc[PQ2_0_ROWS_PER_WARP];

    #pragma unroll
    for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
    {
        int globalRow = min(rowBase + rr, totalN - 1);   // clamp tail; discarded below via row<n check
        if (globalRow < n0)
        {
            rowScales[rr] = scales0; rowCodesBase[rr] = codesBase0; localRows[rr] = globalRow;
        }
        else
        {
            rowScales[rr] = scales1; rowCodesBase[rr] = codesBase1; localRows[rr] = globalRow - n0;
        }
        acc[rr] = 0.0f;
    }

    for (int wStart = 0; wStart < groups_per_row; wStart += PQ2_0_WINDOW_GROUPS)
    {
        const int wGroups = min(PQ2_0_WINDOW_GROUPS, groups_per_row - wStart);
        const int wElems  = wGroups * PQ2_0_GROUP_SIZE;   // always a multiple of 8 (128 | wElems)

        // Vectorized staging (#157), applied per-window: stage only this window's x slice.
        {
            const uint4* x4  = reinterpret_cast<const uint4*>(x + (size_t)wStart * PQ2_0_GROUP_SIZE);
            uint4*       xs4 = reinterpret_cast<uint4*>(xs);
            const int w8 = wElems >> 3;
            for (int i = threadIdx.x; i < w8; i += blockDim.x)
                xs4[i] = x4[i];
        }
        __syncthreads();   // RAW — this window's stage must finish before any read below

        // v3 coalescing: warp cooperates on one group at a time (lane L reads code byte L),
        // instead of each lane owning whole groups — see pq2_0_gemv_f16in's file comment.
        if (warpActive)
        {
            for (int gi = 0; gi < wGroups; gi++)
            {
                const int g        = wStart + gi;
                const int out_base = gi * PQ2_0_GROUP_SIZE;

                // Loaded/summed ONCE per (warp, group), reused across both rows below — see the
                // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
                const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

                #pragma unroll
                for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
                {
                    const long gFlat = (long)localRows[rr] * groups_per_row + g;
                    float scale = __half2float(rowScales[rr][gFlat]);   // lane-independent address — warp broadcast, see "Round 4" file-header note
                    uint8_t p = rowCodesBase[rr][(size_t)gFlat * 32 + lane];

                    acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
                }
            }
        }

        // WAR — see pq2_0_gemv_f16in's identical comment above.
        if (wStart + PQ2_0_WINDOW_GROUPS < groups_per_row)
            __syncthreads();
    }

    if (warpActive)
    {
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = __float2half(a);
        }
    }

    __syncthreads();

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int globalRow = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (globalRow < totalN)
        {
            if (globalRow < n0) y0[globalRow]      = rowOut[threadIdx.x];
            else                 y1[globalRow - n0] = rowOut[threadIdx.x];
        }
    }
}

// ───────────────────────── Small-K specialization kernels (#157 round 4) ─────────────────────────
// Byte-for-byte identical to pq2_0_gemv_f16in/pq2_0_gemv2_f16in above except for `xs`'s static size
// (PQ2_0_MAX_K_SMALL=5120 instead of PQ2_0_MAX_K=17408) — see the file-header "Small-K
// specialization" section for the occupancy-arithmetic motivation (Block Limit Shared Mem 2 -> 9,
// Block Limit Registers unaffected at 5, so registers become the new binding constraint at 5 blocks
// -> 83.3% theoretical occupancy vs 33.3% today). Deliberately duplicated rather than templated:
// this codebase's existing kernel-variant convention (i2_s_gemv_f16in/_gemv2_f16in/_gemv3_f16in in
// i2_s_gemv.cu) is near-duplicate `extern "C" __global__` functions, since `extern "C"` forecloses
// C++ template instantiation across the P/Invoke boundary — a `#define`-parameterized generation
// was considered but rejected as strictly harder to read/debug (macro-expanded compiler errors)
// for a two-instance case with no near-term third size tier.
//
// Callers MUST only launch these for k <= PQ2_0_MAX_K_SMALL (no runtime bounds check, matching
// PQ2_0_MAX_K/I2S_MAX_K's existing convention) — see CudaKernels.LaunchPQ2_0GemvF16In/
// LaunchPQ2_0Gemv2F16In for the dispatch-by-k routing (transparent to callers of those wrappers;
// the model-layer call sites are unchanged).
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_f16in_small(
    const uint8_t* __restrict__ weight,   // split layout — see pq2_0_gemv_f16in's file header
    const half*    __restrict__ x,
    half*          __restrict__ y,
    const int n,
    const int k)
{
    __shared__ __align__(16) half xs[PQ2_0_MAX_K_SMALL];
    {
        const uint4* x4 = reinterpret_cast<const uint4*>(x);
        uint4* xs4 = reinterpret_cast<uint4*>(xs);
        const int k8 = k >> 3;   // k is always a multiple of 8 (k is a multiple of 128)
        for (int i = threadIdx.x; i < k8; i += blockDim.x)
            xs4[i] = x4[i];
    }
    __syncthreads();

    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;

    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2_0_codes_base_offset(total_groups);

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < n;

    __shared__ half rowOut[PQ2_0_ROWS_PER_BLOCK];

    if (warpActive)
    {
        int   rows[PQ2_0_ROWS_PER_WARP];
        float acc[PQ2_0_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            rows[rr] = min(rowBase + rr, n - 1);   // clamp tail rows; their result is discarded below
            acc[rr] = 0.0f;
        }

        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;

            // Loaded/summed ONCE per (warp, group), reused across both rows below - see the
            // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
            const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const long gFlat = (long)rows[rr] * groups_per_row + g;
                float scale = __half2float(scales[gFlat]);
                uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced

                acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
            }
        }

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = __float2half(a);
        }
    }

    __syncthreads();

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int row = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (row < n) y[row] = rowOut[threadIdx.x];
    }
}

extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv2_f16in_small(
    const uint8_t* __restrict__ weight0,   // split layout — each of weight0/weight1 has its OWN codesBase (own n)
    const uint8_t* __restrict__ weight1,
    const half*    __restrict__ x,
    half*          __restrict__ y0,
    half*          __restrict__ y1,
    const int n0,
    const int n1,
    const int k)
{
    __shared__ __align__(16) half xs[PQ2_0_MAX_K_SMALL];
    {
        const uint4* x4 = reinterpret_cast<const uint4*>(x);
        uint4* xs4 = reinterpret_cast<uint4*>(xs);
        const int k8 = k >> 3;   // k is always a multiple of 8 (k is a multiple of 128)
        for (int i = threadIdx.x; i < k8; i += blockDim.x)
            xs4[i] = x4[i];
    }
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1;
    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < totalN;

    const int groups_per_row = k / PQ2_0_GROUP_SIZE;

    const half*    scales0    = reinterpret_cast<const half*>(weight0);
    const half*    scales1    = reinterpret_cast<const half*>(weight1);
    const uint8_t* codesBase0 = weight0 + pq2_0_codes_base_offset((long)n0 * groups_per_row);
    const uint8_t* codesBase1 = weight1 + pq2_0_codes_base_offset((long)n1 * groups_per_row);

    __shared__ half rowOut[PQ2_0_ROWS_PER_BLOCK];

    if (warpActive)
    {
        const half*    rowScales[PQ2_0_ROWS_PER_WARP];
        const uint8_t* rowCodesBase[PQ2_0_ROWS_PER_WARP];
        int            localRows[PQ2_0_ROWS_PER_WARP];
        float          acc[PQ2_0_ROWS_PER_WARP];

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            int globalRow = min(rowBase + rr, totalN - 1);   // clamp tail; discarded below via row<n check
            if (globalRow < n0)
            {
                rowScales[rr] = scales0; rowCodesBase[rr] = codesBase0; localRows[rr] = globalRow;
            }
            else
            {
                rowScales[rr] = scales1; rowCodesBase[rr] = codesBase1; localRows[rr] = globalRow - n0;
            }
            acc[rr] = 0.0f;
        }

        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;

            // Loaded/summed ONCE per (warp, group), reused across both rows below - see the
            // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
            const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const long gFlat = (long)localRows[rr] * groups_per_row + g;
                float scale = __half2float(rowScales[rr][gFlat]);
                uint8_t p = rowCodesBase[rr][(size_t)gFlat * 32 + lane];

                acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
            }
        }

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = __float2half(a);
        }
    }

    __syncthreads();

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int globalRow = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (globalRow < totalN)
        {
            if (globalRow < n0) y0[globalRow]      = rowOut[threadIdx.x];
            else                 y1[globalRow - n0] = rowOut[threadIdx.x];
        }
    }
}

// ───────────────────────── F32-native activations - production decode path, no convert launches (#161) ─────────────────────────
// See the file-header "F32-native activations" section above for the full design rationale. Every
// kernel below is byte-for-byte identical to its `_f16in` counterpart except: (a) `x` is
// `const float*` and the staging loop converts `float4` -> 4 halfs on load instead of copying
// `uint4` halfs verbatim, and (b) `rowOut`/`y` are `float`, storing the raw accumulator with no
// `__float2half` rounding. `xs[]` type/size, the weight-read/accumulate loop, and the warp
// reduction are all UNCHANGED from the `_f16in` kernels - deliberately duplicated rather than
// templated, matching this file own "Small-K specialization" precedent (extern "C" forecloses
// C++ template instantiation across the P/Invoke boundary; near-duplicate explicit functions read
// and debug more easily than a macro-generated family for a fixed, small variant count).
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_f32io(
    const uint8_t* __restrict__ weight,   // split layout - see file header "Split-layout addressing" note
    const float*   __restrict__ x,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    __shared__ __align__(16) half xs[PQ2_0_WINDOW_ELEMS];

    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;

    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2_0_codes_base_offset(total_groups);

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < n;

    __shared__ float rowOut[PQ2_0_ROWS_PER_BLOCK];

    int   rows[PQ2_0_ROWS_PER_WARP];
    float acc[PQ2_0_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
    {
        rows[rr] = min(rowBase + rr, n - 1);   // clamp tail rows; their result is discarded below
        acc[rr] = 0.0f;
    }

    for (int wStart = 0; wStart < groups_per_row; wStart += PQ2_0_WINDOW_GROUPS)
    {
        const int wGroups = min(PQ2_0_WINDOW_GROUPS, groups_per_row - wStart);
        const int wElems  = wGroups * PQ2_0_GROUP_SIZE;   // always a multiple of 4 (128 | wElems)

        // F32-native vectorized staging: float4 load (4 elements/iteration), convert-on-store
        // into the same half xs[] the accumulate loop below reads - see file header.
        {
            const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)wStart * PQ2_0_GROUP_SIZE);
            const int w4 = wElems >> 2;
            for (int i = threadIdx.x; i < w4; i += blockDim.x)
            {
                float4 v = x4[i];
                int base = i * 4;
                xs[base + 0] = __float2half(v.x);
                xs[base + 1] = __float2half(v.y);
                xs[base + 2] = __float2half(v.z);
                xs[base + 3] = __float2half(v.w);
            }
        }
        __syncthreads();   // RAW - this window stage must finish before any read below

        if (warpActive)
        {
            for (int gi = 0; gi < wGroups; gi++)
            {
                const int g        = wStart + gi;
                const int out_base = gi * PQ2_0_GROUP_SIZE;

                // Loaded/summed ONCE per (warp, group), reused across both rows below - see the
                // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
                const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

                #pragma unroll
                for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
                {
                    const long gFlat = (long)rows[rr] * groups_per_row + g;
                    float scale = __half2float(scales[gFlat]);   // lane-independent address - warp broadcast, see "Round 4" file-header note
                    uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced - see file header

                    acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
                }
            }
        }

        if (wStart + PQ2_0_WINDOW_GROUPS < groups_per_row)
            __syncthreads();   // WAR - see pq2_0_gemv_f16in identical comment above
    }

    if (warpActive)
    {
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = a;   // no half rounding - see file header
        }
    }

    __syncthreads();   // RAW on rowOut - unrelated to xs/windowing

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int row = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (row < n) y[row] = rowOut[threadIdx.x];
    }
}

extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv2_f32io(
    const uint8_t* __restrict__ weight0,   // split layout - each of weight0/weight1 has its OWN codesBase (own n)
    const uint8_t* __restrict__ weight1,
    const float*   __restrict__ x,
    float*         __restrict__ y0,
    float*         __restrict__ y1,
    const int n0,
    const int n1,
    const int k)
{
    __shared__ __align__(16) half xs[PQ2_0_WINDOW_ELEMS];

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1;
    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < totalN;

    const int groups_per_row = k / PQ2_0_GROUP_SIZE;

    const half*    scales0    = reinterpret_cast<const half*>(weight0);
    const half*    scales1    = reinterpret_cast<const half*>(weight1);
    const uint8_t* codesBase0 = weight0 + pq2_0_codes_base_offset((long)n0 * groups_per_row);
    const uint8_t* codesBase1 = weight1 + pq2_0_codes_base_offset((long)n1 * groups_per_row);

    __shared__ float rowOut[PQ2_0_ROWS_PER_BLOCK];

    const half*    rowScales[PQ2_0_ROWS_PER_WARP];
    const uint8_t* rowCodesBase[PQ2_0_ROWS_PER_WARP];
    int            localRows[PQ2_0_ROWS_PER_WARP];
    float          acc[PQ2_0_ROWS_PER_WARP];

    #pragma unroll
    for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
    {
        int globalRow = min(rowBase + rr, totalN - 1);   // clamp tail; discarded below via row<n check
        if (globalRow < n0)
        {
            rowScales[rr] = scales0; rowCodesBase[rr] = codesBase0; localRows[rr] = globalRow;
        }
        else
        {
            rowScales[rr] = scales1; rowCodesBase[rr] = codesBase1; localRows[rr] = globalRow - n0;
        }
        acc[rr] = 0.0f;
    }

    for (int wStart = 0; wStart < groups_per_row; wStart += PQ2_0_WINDOW_GROUPS)
    {
        const int wGroups = min(PQ2_0_WINDOW_GROUPS, groups_per_row - wStart);
        const int wElems  = wGroups * PQ2_0_GROUP_SIZE;   // always a multiple of 4 (128 | wElems)

        {
            const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)wStart * PQ2_0_GROUP_SIZE);
            const int w4 = wElems >> 2;
            for (int i = threadIdx.x; i < w4; i += blockDim.x)
            {
                float4 v = x4[i];
                int base = i * 4;
                xs[base + 0] = __float2half(v.x);
                xs[base + 1] = __float2half(v.y);
                xs[base + 2] = __float2half(v.z);
                xs[base + 3] = __float2half(v.w);
            }
        }
        __syncthreads();   // RAW - this window stage must finish before any read below

        if (warpActive)
        {
            for (int gi = 0; gi < wGroups; gi++)
            {
                const int g        = wStart + gi;
                const int out_base = gi * PQ2_0_GROUP_SIZE;

                // Loaded/summed ONCE per (warp, group), reused across both rows below - see the
                // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
                const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

                #pragma unroll
                for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
                {
                    const long gFlat = (long)localRows[rr] * groups_per_row + g;
                    float scale = __half2float(rowScales[rr][gFlat]);   // lane-independent address - warp broadcast, see "Round 4" file-header note
                    uint8_t p = rowCodesBase[rr][(size_t)gFlat * 32 + lane];

                    acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
                }
            }
        }

        if (wStart + PQ2_0_WINDOW_GROUPS < groups_per_row)
            __syncthreads();   // WAR - see pq2_0_gemv_f16in identical comment above
    }

    if (warpActive)
    {
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = a;   // no half rounding - see file header
        }
    }

    __syncthreads();

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int globalRow = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (globalRow < totalN)
        {
            if (globalRow < n0) y0[globalRow]      = rowOut[threadIdx.x];
            else                 y1[globalRow - n0] = rowOut[threadIdx.x];
        }
    }
}

// ───────────────────────── F32-native small-K specialization kernels (#161) ─────────────────────────
// Byte-for-byte identical to pq2_0_gemv_f16in_small/pq2_0_gemv2_f16in_small above except for the
// F32-native input staging / output store described in this file "F32-native activations"
// header section - same relationship the large-K pq2_0_gemv_f32io/pq2_0_gemv2_f32io kernels above
// have to pq2_0_gemv_f16in/pq2_0_gemv2_f16in.
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_f32io_small(
    const uint8_t* __restrict__ weight,   // split layout - see pq2_0_gemv_f16in file header
    const float*   __restrict__ x,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    __shared__ __align__(16) half xs[PQ2_0_MAX_K_SMALL];
    {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const int k4 = k >> 2;   // k is always a multiple of 128, hence of 4
        for (int i = threadIdx.x; i < k4; i += blockDim.x)
        {
            float4 v = x4[i];
            int base = i * 4;
            xs[base + 0] = __float2half(v.x);
            xs[base + 1] = __float2half(v.y);
            xs[base + 2] = __float2half(v.z);
            xs[base + 3] = __float2half(v.w);
        }
    }
    __syncthreads();

    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;

    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2_0_codes_base_offset(total_groups);

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < n;

    __shared__ float rowOut[PQ2_0_ROWS_PER_BLOCK];

    if (warpActive)
    {
        int   rows[PQ2_0_ROWS_PER_WARP];
        float acc[PQ2_0_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            rows[rr] = min(rowBase + rr, n - 1);   // clamp tail rows; their result is discarded below
            acc[rr] = 0.0f;
        }

        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;

            // Loaded/summed ONCE per (warp, group), reused across both rows below - see the
            // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
            const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const long gFlat = (long)rows[rr] * groups_per_row + g;
                float scale = __half2float(scales[gFlat]);
                uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced

                acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
            }
        }

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = a;   // no half rounding - see file header
        }
    }

    __syncthreads();

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int row = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (row < n) y[row] = rowOut[threadIdx.x];
    }
}

extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv2_f32io_small(
    const uint8_t* __restrict__ weight0,   // split layout - each of weight0/weight1 has its OWN codesBase (own n)
    const uint8_t* __restrict__ weight1,
    const float*   __restrict__ x,
    float*         __restrict__ y0,
    float*         __restrict__ y1,
    const int n0,
    const int n1,
    const int k)
{
    __shared__ __align__(16) half xs[PQ2_0_MAX_K_SMALL];
    {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const int k4 = k >> 2;   // k is always a multiple of 128, hence of 4
        for (int i = threadIdx.x; i < k4; i += blockDim.x)
        {
            float4 v = x4[i];
            int base = i * 4;
            xs[base + 0] = __float2half(v.x);
            xs[base + 1] = __float2half(v.y);
            xs[base + 2] = __float2half(v.z);
            xs[base + 3] = __float2half(v.w);
        }
    }
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1;
    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < totalN;

    const int groups_per_row = k / PQ2_0_GROUP_SIZE;

    const half*    scales0    = reinterpret_cast<const half*>(weight0);
    const half*    scales1    = reinterpret_cast<const half*>(weight1);
    const uint8_t* codesBase0 = weight0 + pq2_0_codes_base_offset((long)n0 * groups_per_row);
    const uint8_t* codesBase1 = weight1 + pq2_0_codes_base_offset((long)n1 * groups_per_row);

    __shared__ float rowOut[PQ2_0_ROWS_PER_BLOCK];

    if (warpActive)
    {
        const half*    rowScales[PQ2_0_ROWS_PER_WARP];
        const uint8_t* rowCodesBase[PQ2_0_ROWS_PER_WARP];
        int            localRows[PQ2_0_ROWS_PER_WARP];
        float          acc[PQ2_0_ROWS_PER_WARP];

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            int globalRow = min(rowBase + rr, totalN - 1);   // clamp tail; discarded below via row<n check
            if (globalRow < n0)
            {
                rowScales[rr] = scales0; rowCodesBase[rr] = codesBase0; localRows[rr] = globalRow;
            }
            else
            {
                rowScales[rr] = scales1; rowCodesBase[rr] = codesBase1; localRows[rr] = globalRow - n0;
            }
            acc[rr] = 0.0f;
        }

        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;

            // Loaded/summed ONCE per (warp, group), reused across both rows below - see the
            // "Algebraic ALU reduction" note above pq2_0_load_group_x's definition.
            const Pq2_0GroupX gx = pq2_0_load_group_x(xs, out_base + lane);

            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const long gFlat = (long)localRows[rr] * groups_per_row + g;
                float scale = __half2float(rowScales[rr][gFlat]);
                uint8_t p = rowCodesBase[rr][(size_t)gFlat * 32 + lane];

                acc[rr] += (pq2_0_code_dot(p, gx) - gx.sum) * scale;
            }
        }

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float a = pq2_0_warp_reduce(acc[rr]);
            if (lane == 0)
                rowOut[wid * PQ2_0_ROWS_PER_WARP + rr] = a;   // no half rounding - see file header
        }
    }

    __syncthreads();

    if (threadIdx.x < PQ2_0_ROWS_PER_BLOCK)
    {
        int globalRow = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + threadIdx.x;
        if (globalRow < totalN)
        {
            if (globalRow < n0) y0[globalRow]      = rowOut[threadIdx.x];
            else                 y1[globalRow - n0] = rowOut[threadIdx.x];
        }
    }
}
