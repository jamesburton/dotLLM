// PQ2_0 (PrismML Bonsai ternary) GEMV for dotLLM's decode path.
// y[row] = sum_g group_scale[row,g] * sum_{i in group g} (code(W[row,i]) - 1) * x[i]
//
// dotLLM PQ2_0 on-disk layout (must match dequant_pq2_0.cu / MatMul.PQ2S.cs / Dequantize.DequantizePQ2_0):
//   * Row-major W[n,k], k a multiple of 128. Row stride = (k/128)*34 bytes.
//   * 128-element GROUP = 34 bytes: scale(Half, 2 bytes) THEN codes[32] (4 codes/byte, 2 bits
//     each). Scale is PER-GROUP (not per-tensor like I2_S) — this is the key structural
//     difference from i2_s_gemv.cu, and is why this kernel can't reuse I2_S's uint4-load /
//     shared-x-staging scheme unmodified: each group's contribution must be scaled BEFORE
//     summing into the row total, not once at the very end.
//   * Byte gp in [0,31] within a group's codes holds elements {gp, gp+32, gp+64, gp+96} at bit
//     offsets {6,4,2,0}. Code mapping value = code - 1 (0->-1, 1->0, 2->+1).
//
// Correctness-first only (mirrors MatMul.PQ2S.cs's CPU scalar tier — no AVX2-equivalent SIMD
// widening here, and no shared-memory x staging: k can be up to 17408 for real Bonsai-27B FFN
// rows, which at 4 bytes/float would need ~68KB of static shared memory — over sm_86's 48KB
// static cap. Reading x directly from global relies on L1/L2 caching for reuse across the
// warps in a block, which all read the SAME x vector for their (different) rows.
//
// One warp per output row; a block's warpsPerBlock rows are handled in parallel, and the grid
// strides over rows so any grid size is correct (not tuned to a fixed rows-per-block constant
// like I2_S's launch contract).

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2_0_GROUP_SIZE  128
#define PQ2_0_GROUP_BYTES 34

__device__ __forceinline__ float pq2_0_warp_reduce(float acc)
{
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    return acc;
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

// ───────────────────────── F16 activations/output — production decode path ─────────────────────────
extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_f16in(
    const uint8_t* __restrict__ weight,
    const half*    __restrict__ x,
    half*          __restrict__ y,
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
                group_acc += (float)c0 * __half2float(x[out_base + gp])
                           + (float)c1 * __half2float(x[out_base + gp + 32])
                           + (float)c2 * __half2float(x[out_base + gp + 64])
                           + (float)c3 * __half2float(x[out_base + gp + 96]);
            }
            acc += group_acc * scale;
        }

        acc = pq2_0_warp_reduce(acc);
        if (lane == 0) y[row] = __float2half(acc);
    }
}
