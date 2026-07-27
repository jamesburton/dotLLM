// Fused residual-add + RMSNorm with an FP32 residual stream.
//
// Identical to fused_add_rmsnorm_f16 except the residual buffer is FP32. BitNet b1.58's residual
// stream grows past FP16's ~65504 ceiling in the deepest layers (observed ~65000 by layer 27),
// so keeping the residual in FP32 prevents the overflow→Inf→NaN cascade. The layer output `x` and
// the norm `weight`/`output` stay FP16; only the long-lived residual accumulator is FP32.
//
//   1. sum = residual_f32[i] + FP32(x_f16[i])      (FP32)
//   2. residual_f32[i] = sum                        (FP32 store — no truncation)
//   3. output_f16[i] = FP16(sum * rsqrt(mean(sum^2) + eps) * w)

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) fused_add_rmsnorm_f32res(
    float*       __restrict__ residual,   // [n] FP32 in/out: updated with sum
    const half*  __restrict__ x,          // [n] FP16 layer output to add
    const half*  __restrict__ weight,     // [n] FP16 norm weights
    half*        __restrict__ output,     // [n] FP16 normalized output
    const int   n,
    const float eps)
{
    const int row = blockIdx.x;
    float*       res_row = residual + (size_t)row * n;
    const half*  x_row   = x + (size_t)row * n;
    half*        out_row = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float sum = res_row[i] + __half2float(x_row[i]);
        res_row[i] = sum;
        sum_sq += sum * sum;
    }

    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();

    if (wid == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    }

    __shared__ float rms_inv;
    if (threadIdx.x == 0)
        rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = res_row[i];
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * rms_inv * w);
    }
}

// Seed the FP32 residual from an FP16 source (the embedding output) at layer 0.
extern "C" __global__ void __launch_bounds__(256) copy_f16_to_f32(
    const half* __restrict__ src, float* __restrict__ dst, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// Final residual add → FP16 hidden state for the output norm:
//   hidden_f16[i] = FP16(residual_f32[i] + x_f16[i])
extern "C" __global__ void __launch_bounds__(256) add_f32res_f16(
    const float* __restrict__ residual, const half* __restrict__ x,
    half* __restrict__ output, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __float2half(residual[idx] + __half2float(x[idx]));
}

// RMSNorm with FP32 input, FP16 weight, FP16 output. The model-wide norm weights are uploaded as
// FP16, so the existing rmsnorm_f32in (which reads FP32 weight) cannot be used for the final norm
// over the FP32 residual. One block per row.
extern "C" __global__ void __launch_bounds__(256) rmsnorm_f32in_f16w(
    const float* __restrict__ input,
    const half*  __restrict__ weight,
    half*        __restrict__ output,
    const int   n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    half* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();

    if (wid == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    }

    __shared__ float rms_inv;
    if (threadIdx.x == 0)
        rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        float w = __half2float(weight[i]);
        y[i] = __float2half(v * rms_inv * w);
    }
}
