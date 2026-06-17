// I2_S (BitNet b1.58 ternary) GEMV for dotLLM's decode path.
// y[row] = scale * sum_k (code(W[row,k]) - 1) * x[k]
//
// dotLLM I2_S on-disk layout (must match MatMul.I2S.cs / Dequantize.DequantizeI2_S):
//   * Row-major W[n,k], k a multiple of 128. Row stride = k/4 bytes (4 codes/byte, 2 bits each).
//   * 128-element block = 32 bytes. Byte gp in [0,31] holds elements {gp, gp+32, gp+64, gp+96}
//     at bit offsets {6,4,2,0}.
//   * Code mapping value = code - 1  (0->-1, 1->0, 2->+1). NOTE: differs from BitNet GPU's
//     offset-binary {1,2,3}; the decode here subtracts 1.
//   * ONE per-tensor float32 scale at the tensor tail, byte offset (size_t)n*(k/4).
//
// Variant A (W2A16): decode ternary, multiply by half/float activations, fp32 accumulate.
// Numerically matches the CPU float reference (MatMul.GemvI2_S) to fp32 rounding.
// Control structure (grid=n, block=256, thread-stride over 128-blocks, two-stage warp reduction)
// is identical to quantized_gemv_q8_0.

#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float i2s_block_reduce(float acc)
{
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    }
    return acc;
}

// ───────────────────────── Variant A: W2A16, FP16 activations ─────────────────────────
extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_f16in(
    const uint8_t* __restrict__ weight,   // packed codes [n × k/4 bytes] + trailing f32 scale
    const half*    __restrict__ x,        // [k]
    half*          __restrict__ y,        // [n]
    const int n,
    const int k)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int blocks    = k / 128;
    const uint8_t* w_row = weight + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int blk = threadIdx.x; blk < blocks; blk += blockDim.x)
    {
        const uint8_t* bp = w_row + blk * 32;
        const int x_base  = blk * 128;
        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = bp[gp];
            int c0 = ((p >> 6) & 0x3) - 1;
            int c1 = ((p >> 4) & 0x3) - 1;
            int c2 = ((p >> 2) & 0x3) - 1;
            int c3 = ( p       & 0x3) - 1;
            acc += (float)c0 * __half2float(x[x_base + gp]);
            acc += (float)c1 * __half2float(x[x_base + gp + 32]);
            acc += (float)c2 * __half2float(x[x_base + gp + 64]);
            acc += (float)c3 * __half2float(x[x_base + gp + 96]);
        }
    }

    acc = i2s_block_reduce(acc);
    if (threadIdx.x == 0) y[row] = __float2half(acc * scale);
}

// ───────────────────────── Variant A twin: FP32 activations ─────────────────────────
// Exact-match reference for CPU-vs-GPU validation and any F32 activation path.
extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_f32in(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int blocks    = k / 128;
    const uint8_t* w_row = weight + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int blk = threadIdx.x; blk < blocks; blk += blockDim.x)
    {
        const uint8_t* bp = w_row + blk * 32;
        const int x_base  = blk * 128;
        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = bp[gp];
            int c0 = ((p >> 6) & 0x3) - 1;
            int c1 = ((p >> 4) & 0x3) - 1;
            int c2 = ((p >> 2) & 0x3) - 1;
            int c3 = ( p       & 0x3) - 1;
            acc += (float)c0 * x[x_base + gp];
            acc += (float)c1 * x[x_base + gp + 32];
            acc += (float)c2 * x[x_base + gp + 64];
            acc += (float)c3 * x[x_base + gp + 96];
        }
    }

    acc = i2s_block_reduce(acc);
    if (threadIdx.x == 0) y[row] = acc * scale;
}
