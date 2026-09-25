// Fused residual-add + RMS normalization kernel for dotLLM.
// Numerical contract: avoids the FP16 truncation of `sum` between Add and RmsNorm
// (the original benefit of fusing) AND now also avoids the FP16 round-trip in
// the second pass — the FP32 sum is cached in shared memory.
//
//   1. sum = FP32(residual) + FP32(x)         — keep in FP32
//   2. residual[i] = FP16(sum)                 — update residual buffer (FP16 store)
//   3. shmem[i] = sum                          — cache FP32 sum for pass 2
//   4. output[i] = FP16(sum * rsqrt(rms) * w)  — normalize from cached FP32 sum
//
// Optimizations vs the original two-pass shmem-tree-reduction kernel:
//   * half2 vectorized loads/stores (one transaction per 2 elements)
//   * __shfl_xor_sync warp reduction (symmetric — every lane holds the sum)
//   * Pre-folds 1/n into the rsqrt argument via fmaf
//   * Pass 2 reads FP32 sums from shared memory — eliminates the second global
//     read of residual AND the extra __half2float conversion
//
// Shared memory layout (dynamic): [n floats sum cache] [up to 32 floats warp scratch / rms_inv]
// Caller MUST pass sharedBytes = (n + 32) * sizeof(float) (rounded up internally).

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) fused_add_rmsnorm_f16(
    half* __restrict__ residual,       // [n] in/out: updated with sum
    const half* __restrict__ x,        // [n] layer output to add
    const half* __restrict__ weight,   // [n] norm weights
    half* __restrict__ output,         // [n] normalized output (may alias `x`)
    const int n,
    const float eps)
{
    extern __shared__ float smem[];
    // smem[0 .. n-1]            : FP32 sum cache
    // smem[smem_scratch_off ..] : warp-sum scratch (max 32 floats); slot [0] also stores rms_inv
    const int smem_scratch_off = (n + 1) & ~1; // align to 2 floats (8 bytes)
    float* warp_scratch = smem + smem_scratch_off;

    const int row = blockIdx.x;
    half* res_row = residual + (size_t)row * n;
    const half* x_row = x + (size_t)row * n;
    half* out_row = output + (size_t)row * n;
    const int tid = threadIdx.x;
    const int n2 = n >> 1;

    // ── Pass 1: add, store FP16 sum to residual, cache FP32 sum in shmem, accumulate sum_sq ──
    float sum_sq = 0.0f;

    const half2* __restrict__ res2_in = reinterpret_cast<const half2*>(res_row);
    const half2* __restrict__ x2      = reinterpret_cast<const half2*>(x_row);
    half2* __restrict__       res2_out = reinterpret_cast<half2*>(res_row);

    for (int i = tid; i < n2; i += blockDim.x)
    {
        half2 r = res2_in[i];
        half2 xi = x2[i];
        float r0 = __low2float(r),  r1 = __high2float(r);
        float x0 = __low2float(xi), x1 = __high2float(xi);
        float s0 = r0 + x0;
        float s1 = r1 + x1;
        res2_out[i] = __floats2half2_rn(s0, s1);
        smem[2 * i]     = s0;
        smem[2 * i + 1] = s1;
        sum_sq = fmaf(s0, s0, sum_sq);
        sum_sq = fmaf(s1, s1, sum_sq);
    }

    // Tail element if n is odd
    if ((n & 1) && tid == 0)
    {
        int last = n - 1;
        float r = __half2float(res_row[last]);
        float xi = __half2float(x_row[last]);
        float s = r + xi;
        res_row[last] = __float2half(s);
        smem[last] = s;
        sum_sq = fmaf(s, s, sum_sq);
    }

    // ── Warp reduction (symmetric: all lanes hold the warp sum) ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) warp_scratch[warp_id] = sum_sq;
    __syncthreads();

    // First warp aggregates the per-warp sums and publishes rms_inv
    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        sum_sq = (lane < num_warps) ? warp_scratch[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

        if (lane == 0)
            warp_scratch[0] = rsqrtf(fmaf(sum_sq, 1.0f / (float)n, eps));
    }
    __syncthreads();

    const float rms_inv = warp_scratch[0];

    // ── Pass 2: read FP32 sum from shmem, scale by rms_inv * weight, write FP16 ──
    const half2* __restrict__ w2 = reinterpret_cast<const half2*>(weight);
    half2* __restrict__       y2 = reinterpret_cast<half2*>(out_row);

    for (int i = tid; i < n2; i += blockDim.x)
    {
        float s0 = smem[2 * i];
        float s1 = smem[2 * i + 1];
        half2 wh = w2[i];
        float w0 = __low2float(wh), w1 = __high2float(wh);
        y2[i] = __floats2half2_rn(s0 * rms_inv * w0, s1 * rms_inv * w1);
    }

    if ((n & 1) && tid == 0)
    {
        int last = n - 1;
        float s = smem[last];
        float w = __half2float(weight[last]);
        out_row[last] = __float2half(s * rms_inv * w);
    }
}
