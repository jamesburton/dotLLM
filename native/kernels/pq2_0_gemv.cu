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

// ───────────────────────── F16 activations/output — production decode path (v2) ─────────────────────────
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
    if (rowBase >= n) return;

    const uint8_t* row_ptrs[PQ2_0_ROWS_PER_WARP];
    float acc[PQ2_0_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
    {
        int r = min(rowBase + rr, n - 1);   // clamp tail rows; their result is discarded below
        row_ptrs[rr] = weight + (size_t)r * row_bytes;
        acc[rr] = 0.0f;
    }

    for (int g = lane; g < groups_per_row; g += warpSize)
    {
        float scale[PQ2_0_ROWS_PER_WARP];
        const uint8_t* codes[PQ2_0_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            const uint8_t* group_base = row_ptrs[rr] + (size_t)g * PQ2_0_GROUP_BYTES;
            scale[rr] = __half2float(*reinterpret_cast<const half*>(group_base));
            codes[rr] = group_base + 2;
        }

        const int out_base = g * PQ2_0_GROUP_SIZE;

        #pragma unroll
        for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
        {
            float group_acc = 0.0f;
            #pragma unroll
            for (int gp = 0; gp < 32; gp++)
                pq2_0_accum_byte(group_acc, codes[rr][gp], xs, out_base + gp);
            acc[rr] += group_acc * scale[rr];
        }
    }

    #pragma unroll
    for (int rr = 0; rr < PQ2_0_ROWS_PER_WARP; rr++)
    {
        float a = pq2_0_warp_reduce(acc[rr]);
        int row = rowBase + rr;
        if (lane == 0 && row < n) y[row] = __float2half(a);
    }
}
