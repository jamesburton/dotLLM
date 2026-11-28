// Upstream ggml Q2_0 packed GEMV (issue #2074). Weights are read in place - never expanded:
//   block (64 elements, 18 bytes) = fp16 d + qs[16]; element j = code (qs[j/4] >> 2*(j%4)) & 3; value = d * (code - 1) (codes 0..3 = -d, 0, +d, +2d).
// y[c, row] = sum_k W[row, k] * x[c, k] for c in [0, cols), cols <= 8; F32 activations in, F32 out, F32 accumulation.
// One warp per output row. Lane l handles code BYTES l, l+32, ... of the row's qs stream (4 consecutive elements per byte): within one
// block 16 consecutive lanes read 16 consecutive bytes and 64 consecutive floats of x (float4 each), so both streams are coalesced.
// 8 columns share one read of the packed row (the MTP / small-batch verify shape); wider batches go through dequant + cuBLAS (bounded scratch).
#include <cuda_fp16.h>
#include <stdint.h>

#define Q2_0_GROUP 64
#define Q2_0_BYTES 18
#define Q2_0_MAX_COLS 8

extern "C" __global__ void __launch_bounds__(256) q2_0_gemv_f32io(
    const uint8_t* __restrict__ weight,   // [n, k] packed, row stride (k/64)*18
    const float*   __restrict__ x,        // [cols, k]
    float*         __restrict__ y,        // [cols, n]
    const int n, const int k, const int cols)
{
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n) return;

    const int blocks = k / Q2_0_GROUP;
    const uint8_t* wrow = weight + (size_t)row * blocks * Q2_0_BYTES;
    float acc[Q2_0_MAX_COLS];
#pragma unroll
    for (int c = 0; c < Q2_0_MAX_COLS; ++c) acc[c] = 0.f;

    const int total_bytes = blocks * 16;
    for (int q = lane; q < total_bytes; q += 32)
    {
        const int blk = q >> 4, within = q & 15;
        const uint8_t* wb = wrow + (size_t)blk * Q2_0_BYTES;
        const float d = __half2float(*reinterpret_cast<const half*>(wb));
        const uint32_t byte = wb[2 + within];
        const float w0 = d * (float)((int)( byte        & 3) - 1);
        const float w1 = d * (float)((int)((byte >> 2)  & 3) - 1);
        const float w2 = d * (float)((int)((byte >> 4)  & 3) - 1);
        const float w3 = d * (float)((int)((byte >> 6)  & 3) - 1);
        const int col0 = blk * Q2_0_GROUP + within * 4;
#pragma unroll
        for (int c = 0; c < Q2_0_MAX_COLS; ++c)
        {
            if (c < cols)
            {
                const float4 xv = *reinterpret_cast<const float4*>(x + (size_t)c * k + col0);
                acc[c] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Q2_0_MAX_COLS; ++c)
    {
        float v = acc[c];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0 && c < cols) y[(size_t)c * n + row] = v;
    }
}
