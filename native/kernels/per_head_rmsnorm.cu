// Per-head RMS normalization kernel (QK-norm, Qwen3-style) for dotLLM.
// Normalizes each head vector independently: qk[t, h, :headDim]
//
// Optimizations:
//   * half2 vectorized loads/stores
//   * __shfl_xor_sync warp reduction
//   * Pre-folds 1/head_dim via fmaf

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) per_head_rmsnorm_f16(
    half* __restrict__ qk,
    const half* __restrict__ weight,
    const float eps,
    const int num_heads,
    const int head_dim,
    const int seq_len)
{
    // One block per (token, head) pair
    int block_id = blockIdx.x;
    int t = block_id / num_heads;
    int h = block_id % num_heads;
    if (t >= seq_len) return;

    int stride = num_heads * head_dim;
    half* vec = qk + (size_t)t * stride + h * head_dim;
    const int tid = threadIdx.x;
    const int hd2 = head_dim >> 1;

    const half2* __restrict__ v2_in = reinterpret_cast<const half2*>(vec);
    half2* __restrict__       v2_out = reinterpret_cast<half2*>(vec);
    const half2* __restrict__ w2 = reinterpret_cast<const half2*>(weight);

    // ── Pass 1: sum of squares ──
    float sum_sq = 0.0f;
    for (int i = tid; i < hd2; i += blockDim.x)
    {
        half2 v = v2_in[i];
        float v0 = __low2float(v), v1 = __high2float(v);
        sum_sq = fmaf(v0, v0, sum_sq);
        sum_sq = fmaf(v1, v1, sum_sq);
    }
    if ((head_dim & 1) && tid == 0)
    {
        float v = __half2float(vec[head_dim - 1]);
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
            warp_sums[0] = rsqrtf(fmaf(sum_sq, 1.0f / (float)head_dim, eps));
    }
    __syncthreads();
    const float rms_inv = warp_sums[0];

    // ── Pass 2: vectorized normalize + scale ──
    for (int i = tid; i < hd2; i += blockDim.x)
    {
        half2 v  = v2_in[i];
        half2 wh = w2[i];
        float v0 = __low2float(v),  v1 = __high2float(v);
        float w0 = __low2float(wh), w1 = __high2float(wh);
        v2_out[i] = __floats2half2_rn(v0 * rms_inv * w0, v1 * rms_inv * w1);
    }
    if ((head_dim & 1) && tid == 0)
    {
        int last = head_dim - 1;
        float v = __half2float(vec[last]);
        float w = __half2float(weight[last]);
        vec[last] = __float2half(v * rms_inv * w);
    }
}
