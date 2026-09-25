// Fused squared-ReLU GLU + RMSNorm for dotLLM's BitNet b1.58 FFN.
//
// BitNet's gated FFN intermediate is  t[i] = relu(gate[i])^2 * up[i],  immediately followed by a
// Sub-LN RMSNorm before the down projection. The un-normalized intermediate is LARGE (squaring an
// O(50) pre-activation yields O(1e5+)), which overflows FP16 if materialized. The CPU path keeps it
// in FP32; on GPU we must do the same. Fusing relu^2-GLU with the RMSNorm lets us hold the large
// value in FP32 registers, compute the RMS in FP32, and write only the normalized (O(1)) result to
// FP16 — so the large intermediate is never truncated.
//
//   out[i] = (relu(gate[i])^2 * up[i]) * rsqrt(mean_j(t[j]^2) + eps) * weight[i]
//
// One block per token row (length n). gate/up/weight/out are FP16. Two passes over the FP16 inputs
// (recompute t[i] for sum-of-squares, then for the normalized output) — cheap vs an FP32 buffer.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) relu2_glu_rmsnorm_f16(
    const half*  __restrict__ gate,
    const half*  __restrict__ up,
    const half*  __restrict__ weight,
    half*        __restrict__ output,
    const int   n,
    const float eps,
    const int   seq_len)   // unused; one block per row, grid = seq_len
{
    const int row = blockIdx.x;
    if (row >= seq_len) return;

    const half* g = gate   + (size_t)row * n;
    const half* u = up     + (size_t)row * n;
    half*       y = output + (size_t)row * n;

    // Pass 1: sum of squares of t[i] = relu(g)^2 * u, FP32 accumulation.
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float gv = __half2float(g[i]);
        float r  = gv > 0.0f ? gv : 0.0f;
        float t  = r * r * __half2float(u[i]);
        sum_sq += t * t;
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

    // Pass 2: normalize and scale by the Sub-LN weight, write FP16 (now O(1)).
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float gv = __half2float(g[i]);
        float r  = gv > 0.0f ? gv : 0.0f;
        float t  = r * r * __half2float(u[i]);
        float w  = __half2float(weight[i]);
        y[i] = __float2half(t * rms_inv * w);
    }
}
