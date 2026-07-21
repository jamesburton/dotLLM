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
    const uint8_t* __restrict__ weight,
    const half*    __restrict__ x,
    half*          __restrict__ y,
    const int n,
    const int k)
{
    // Stage x[k] into shared memory once per block (kept half-width — see file header for why
    // this fits under the static cap where a float stage would not). Reused by all
    // PQ2_0_ROWS_PER_BLOCK rows' warps in this block.
    __shared__ half xs[PQ2_0_MAX_K];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = x[i];
    __syncthreads();

    const int groups_per_row = k / PQ2_0_GROUP_SIZE;
    const int row_bytes      = groups_per_row * PQ2_0_GROUP_BYTES;

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < n;

    __shared__ half rowOut[PQ2_0_ROWS_PER_BLOCK];

    if (warpActive)
    {
        const uint8_t* row_ptrs[PQ2_0_ROWS_PER_WARP];
        float acc[PQ2_0_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            int r = min(rowBase + rr, n - 1);   // clamp tail rows; their result is discarded below
            row_ptrs[rr] = weight + (size_t)r * row_bytes;
            acc[rr] = 0.0f;
        }

        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;
            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const uint8_t* group_base = row_ptrs[rr] + (size_t)g * PQ2_0_GROUP_BYTES;
                float scale = __half2float(*reinterpret_cast<const half*>(group_base));
                uint8_t p = group_base[2 + lane];   // coalesced: 32 lanes read 32 consecutive bytes

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
    const uint8_t* __restrict__ weight0,
    const uint8_t* __restrict__ weight1,
    const half*    __restrict__ x,
    half*          __restrict__ y0,
    half*          __restrict__ y1,
    const int n0,
    const int n1,
    const int k)
{
    __shared__ half xs[PQ2_0_MAX_K];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = x[i];
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1;
    const int rowBase = blockIdx.x * PQ2_0_ROWS_PER_BLOCK + wid * PQ2_0_ROWS_PER_WARP;
    const bool warpActive = rowBase < totalN;

    const int groups_per_row = k / PQ2_0_GROUP_SIZE;
    const int row_bytes      = groups_per_row * PQ2_0_GROUP_BYTES;

    __shared__ half rowOut[PQ2_0_ROWS_PER_BLOCK];

    if (warpActive)
    {
        const uint8_t* row_ptrs[PQ2_0_ROWS_PER_WARP];
        float          acc[PQ2_0_ROWS_PER_WARP];

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            int globalRow = min(rowBase + rr, totalN - 1);   // clamp tail; discarded below via row<n check
            const uint8_t* w;
            int localRow;
            if (globalRow < n0) { w = weight0; localRow = globalRow; }
            else                { w = weight1; localRow = globalRow - n0; }

            row_ptrs[rr] = w + (size_t)localRow * row_bytes;
            acc[rr]      = 0.0f;
        }

        // v3 coalescing: warp cooperates on one group at a time (lane L reads code byte L),
        // instead of each lane owning whole groups — see pq2_0_gemv_f16in's file comment.
        for (int g = 0; g < groups_per_row; g++)
        {
            const int out_base = g * PQ2_0_GROUP_SIZE;
            #pragma unroll
            for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
            {
                const uint8_t* group_base = row_ptrs[rr] + (size_t)g * PQ2_0_GROUP_BYTES;
                float scale = __half2float(*reinterpret_cast<const half*>(group_base));
                uint8_t p = group_base[2 + lane];

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
