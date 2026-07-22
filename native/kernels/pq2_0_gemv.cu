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
    // Stage x[k] into shared memory once per block (kept half-width — see file header for why
    // this fits under the static cap where a float stage would not). Reused by all
    // PQ2_0_ROWS_PER_BLOCK rows' warps in this block.
    //
    // Vectorized staging (#157 follow-up): a per-thread scalar-half loop here issues up to
    // PQ2_0_MAX_K/blockDim.x = 17408/256 = 68 SEQUENTIAL 2-byte loads before the __syncthreads()
    // below releases any compute — ncu's source-correlated stall counters showed this loop as the
    // single largest per-instruction latency stall in the whole kernel (dwarfing every weight-read
    // PC), despite the underlying x[] reads already being perfectly coalesced. The fix targets
    // ROUND-TRIP COUNT, not bytes moved: read one uint4 (8 halfs = 16 bytes) per iteration instead
    // of one half, cutting the sequential load chain ~8x (68 -> 9 iterations for k=17408). k is a
    // multiple of PQ2_0_GROUP_SIZE (128) by the on-disk-layout invariant (file header), hence
    // always a multiple of 8 — no scalar tail path is needed for any shape that actually occurs.
    // `x` (the model's activation-staging scratch buffer) and `xs` both need >=16-byte alignment
    // for the uint4 reinterpret to be valid: `xs` gets it via the __align__(16) below (shared-
    // memory arrays are not 16-byte aligned by default); `x` is always the base pointer of a
    // dedicated `cuMemAlloc_v2`-backed device allocation (CudaQwen3HybridDenseTransformerModel's
    // _activF16InScratch, or a full CudaForwardState buffer such as NormOutput/AttnOutput/
    // SiluOutput, or a test-owned buffer) — never a sub-offset into a larger allocation — and
    // CUDA's device allocator guarantees a minimum 256-byte alignment, comfortably satisfying the
    // 16-byte requirement here.
    __shared__ __align__(16) half xs[PQ2_0_MAX_K];
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
                float scale = __half2float(scales[gFlat]);   // lane-independent address — warp broadcast, see "Round 4" file-header note
                uint8_t p = codesBase[(size_t)gFlat * 32 + lane];   // unconditionally aligned+coalesced — see file header

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
    // Vectorized staging (#157 follow-up) — see pq2_0_gemv_f16in's file-header comment above for
    // the full rationale (latency-bound prologue, k%8==0 invariant, x/xs alignment reasoning).
    // Identical loop, just applied to this kernel's shared xs[].
    __shared__ __align__(16) half xs[PQ2_0_MAX_K];
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

    // Each virtually-concatenated array is a physically separate split-layout tensor, so each
    // gets its own scales/codes split point derived from its OWN row count (n0 vs n1) — see
    // pq2_0_gemv_f16in's "Split-layout addressing" file-header note.
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

        // v3 coalescing: warp cooperates on one group at a time (lane L reads code byte L),
        // instead of each lane owning whole groups — see pq2_0_gemv_f16in's file comment.
        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;
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
