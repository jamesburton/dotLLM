// RMS Normalization with FP32 input, FP32 weight, FP16 output.
// Used when the residual stream is FP32 but downstream GEMM needs FP16 input.
// out[i] = FP16((FP32_input[i] / rms) * FP32_weight[i])
//
// Optimizations:
//   * float2 vectorized loads
//   * __shfl_xor_sync warp reduction
//   * Pre-folds 1/n via fmaf

#include <cuda_fp16.h>  // needed for half output type

extern "C" __global__ void __launch_bounds__(256) rmsnorm_f32in_f16out(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    half* y = output + (size_t)row * n;
    const int tid = threadIdx.x;
    const int n2 = n >> 1;

    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    const float2* __restrict__ w2 = reinterpret_cast<const float2*>(weight);
    half2* __restrict__        y2 = reinterpret_cast<half2*>(y);

    // ── Pass 1: sum of squares via float2 loads ──
    float sum_sq = 0.0f;
    for (int i = tid; i < n2; i += blockDim.x)
    {
        float2 v = x2[i];
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
    }
    if ((n & 1) && tid == 0)
    {
        float v = x[n - 1];
        sum_sq = fmaf(v, v, sum_sq);
    }

    // ── Warp reduction (symmetric) ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[32];
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

    // ── Pass 2: vectorized scale + half2 store ──
    for (int i = tid; i < n2; i += blockDim.x)
    {
        float2 v  = x2[i];
        float2 wh = w2[i];
        y2[i] = __floats2half2_rn(v.x * rms_inv * wh.x, v.y * rms_inv * wh.y);
    }
    if ((n & 1) && tid == 0)
    {
        int last = n - 1;
        y[last] = __float2half(x[last] * rms_inv * weight[last]);
    }
}
