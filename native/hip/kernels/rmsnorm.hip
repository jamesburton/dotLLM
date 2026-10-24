// RMS Normalization kernel for dotLLM — HIP port of native/kernels/rmsnorm.cu and rmsnorm_f32.cu.
// out[i] = (input[i] / rms) * weight[i], rms = sqrt(mean(input^2) + eps)
// One block per row, wavefront shuffle reduction.
//
// Mechanical hipify translation: cuda_fp16.h -> hip/hip_fp16.h, __shfl_down_sync works
// on HIP with the same signature. __syncthreads() is unchanged. warpSize is the
// wavefront size (32 on RDNA in wave32 mode / CDNA, 64 on older/wave64).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP32 variant — matches CPU reference exactly for the scaffold test.
extern "C" __global__ void __launch_bounds__(256) rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    float* y = output + (size_t)row * n;

    // Step 1: sum of squares, FP32 accumulation
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    // Wavefront-level reduction. On AMD, warpSize is 32 (wave32) or 64 (wave64).
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down(sum_sq, offset);

    // Cross-wavefront reduction via shared memory. Oversize to 64 slots so wave64
    // GPUs with 256-thread blocks (= 4 waves) still fit; wave32 fits trivially.
    __shared__ float warp_sums[64];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down(sum_sq, offset);
    }

    __shared__ float rms_inv;
    if (threadIdx.x == 0)
        rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    // Step 2: normalize and scale
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = x[i] * rms_inv * weight[i];
}

// FP16 variant — kept for parity with the CUDA kernel catalog.
extern "C" __global__ void __launch_bounds__(256) rmsnorm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const __half* x = input + (size_t)row * n;
    __half* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down(sum_sq, offset);

    __shared__ float warp_sums[64];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down(sum_sq, offset);
    }

    __shared__ float rms_inv;
    if (threadIdx.x == 0)
        rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half(v * rms_inv * w);
    }
}
