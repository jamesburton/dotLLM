// RMS Normalization kernel for dotLLM.
// out[i] = (input[i] / rms) * weight[i], rms = sqrt(mean(input^2) + eps)
// FP16 in/out, FP32 accumulation for numerical stability.
// One block per row.
//
// Optimizations vs the naive scalar version:
//   * half2 vectorized loads/stores (one transaction per 2 elements)
//   * __shfl_xor_sync warp reduction (symmetric — every lane holds the sum)
//   * Pre-folds 1/n into the rsqrt argument via fmaf
//   * No dynamic shared memory; only a tiny static warp-scratch buffer

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) rmsnorm_f16(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const half* x = input + (size_t)row * n;
    half* y = output + (size_t)row * n;
    const int tid = threadIdx.x;
    const int n2 = n >> 1;

    const half2* __restrict__ x2 = reinterpret_cast<const half2*>(x);
    const half2* __restrict__ w2 = reinterpret_cast<const half2*>(weight);
    half2* __restrict__       y2 = reinterpret_cast<half2*>(y);

    // ── Pass 1: sum of squares via vectorized half2 loads ──
    float sum_sq = 0.0f;
    for (int i = tid; i < n2; i += blockDim.x)
    {
        half2 v = x2[i];
        float v0 = __low2float(v), v1 = __high2float(v);
        sum_sq = fmaf(v0, v0, sum_sq);
        sum_sq = fmaf(v1, v1, sum_sq);
    }
    if ((n & 1) && tid == 0)
    {
        float v = __half2float(x[n - 1]);
        sum_sq = fmaf(v, v, sum_sq);
    }

    // ── Warp reduction (symmetric) ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[32]; // max 32 warps per block
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane == 0)
            warp_sums[0] = rsqrtf(fmaf(sum_sq, 1.0f / (float)n, eps));
    }
    __syncthreads();
    const float rms_inv = warp_sums[0];

    // ── Pass 2: vectorized normalize and scale ──
    for (int i = tid; i < n2; i += blockDim.x)
    {
        half2 v  = x2[i];
        half2 wh = w2[i];
        float v0 = __low2float(v),  v1 = __high2float(v);
        float w0 = __low2float(wh), w1 = __high2float(wh);
        y2[i] = __floats2half2_rn(v0 * rms_inv * w0, v1 * rms_inv * w1);
    }
    if ((n & 1) && tid == 0)
    {
        int last = n - 1;
        float v = __half2float(x[last]);
        float w = __half2float(weight[last]);
        y[last] = __float2half(v * rms_inv * w);
    }
}
