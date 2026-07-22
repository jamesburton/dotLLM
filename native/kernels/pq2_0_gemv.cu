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
//      effect is a wash-to-marginal at best, and NOT verifiable without a real benchmark run (which
//      this session's hard "no GPU execution" constraint ruled out) — left as a documented
//      candidate for a future round that CAN measure it, rather than committed on modeled reasoning
//      alone.
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

// Decode the 4 codes packed in byte `p` (elements {gp,+32,+64,+96}) and accumulate into `acc`
// against the four shared (half-precision) activations at base `xb` + {0,32,64,96}.
__device__ __forceinline__ void pq2_0_accum_byte(float& acc, unsigned int p, const half* xs, int xb)
{
    int c0 = ((p >> 6) & 0x3) - 1;
    int c1 = ((p >> 4) & 0x3) - 1;
    int c2 = ((p >> 2) & 0x3) - 1;
    int c3 = ( p       & 0x3) - 1;
    acc += (float)c0 * __half2float(xs[xb]);
    acc += (float)c1 * __half2float(xs[xb + 32]);
    acc += (float)c2 * __half2float(xs[xb + 64]);
    acc += (float)c3 * __half2float(xs[xb + 96]);
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
// `xb = out_base + lane`, so `pq2_0_accum_byte` (unchanged) is called with `lane` in place of
// the old per-lane `gp` loop variable. The redundant per-lane read of the group's 2-byte scale
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
                #pragma unroll
                for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
                {
                    const long gFlat = (long)rows[rr] * groups_per_row + g;
                    float scale = __half2float(scales[gFlat]);   // lane-independent address — warp broadcast, see "Round 4" file-header note
                    uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced — see file header

                    float group_partial = 0.0f;
                    pq2_0_accum_byte(group_partial, p, xs, out_base + lane);
                    acc[rr] += group_partial * scale;
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
                #pragma unroll
                for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
                {
                    const long gFlat = (long)localRows[rr] * groups_per_row + g;
                    float scale = __half2float(rowScales[rr][gFlat]);   // lane-independent address — warp broadcast, see "Round 4" file-header note
                    uint8_t p = rowCodesBase[rr][(size_t)gFlat * 32 + lane];

                    float group_partial = 0.0f;
                    pq2_0_accum_byte(group_partial, p, xs, out_base + lane);
                    acc[rr] += group_partial * scale;
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
            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const long gFlat = (long)rows[rr] * groups_per_row + g;
                float scale = __half2float(scales[gFlat]);
                uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced

                float group_partial = 0.0f;
                pq2_0_accum_byte(group_partial, p, xs, out_base + lane);
                acc[rr] += group_partial * scale;
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
            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const long gFlat = (long)localRows[rr] * groups_per_row + g;
                float scale = __half2float(rowScales[rr][gFlat]);
                uint8_t p = rowCodesBase[rr][(size_t)gFlat * 32 + lane];

                float group_partial = 0.0f;
                pq2_0_accum_byte(group_partial, p, xs, out_base + lane);
                acc[rr] += group_partial * scale;
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
