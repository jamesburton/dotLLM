// Tiled attention with FP32 Q/K/V/output and online softmax.
//
// Softmax uses Schraudolph's IEEE-754 bit-trick approximation of expf, matching the
// CPU oracle's DotLLM.Cpu.Kernels.FastMath.ExpSumAndStore. The CPU side has used the
// fast-exp path since the kernel's inception; switching CUDA to precise expf made the
// two backends disagree by ~1% (5e-3 abs on attention output) on synthetic-fixture
// parity. The bit-trick keeps both backends bit-near-equivalent without a CPU-side
// accuracy regression. Constants C0/C1 must stay in sync with FastMath.cs.
//
//   exp(x) ≈ bitcast_int_to_float((int)(x * C0 + C1)),   x ≤ 0 only (no overflow guard)
//
// C0 = 2^23 / ln(2), C1 = (127 - 0.0579) * 2^23. Applied only to softmax `expf` calls
// where the argument is always ≤ 0 by construction (max-subtracted scores).

#include <float.h>
#include <math.h>

#define TILE_KV 256

// Schraudolph fast-exp constants (mirror FastMath.cs).
#define FASTEXP_C0 12102203.0f
#define FASTEXP_C1 1064866805.0f
#define FASTEXP_MIN_CLAMP -87.3f

__device__ __forceinline__ float fast_exp_neg(float x)
{
    // Caller contract: x ≤ 0 (max-subtracted softmax scores). Clamp the lower bound
    // to keep the integer cast inside the IEEE-754 normal range. Use float-to-int
    // truncation (toward zero) to match the C# scalar `(int)x` and the SIMD
    // ConvertToVector*Int32WithTruncation paths in FastMath.cs — round-to-nearest
    // would introduce a sub-ULP bias.
    x = fmaxf(x, FASTEXP_MIN_CLAMP);
    int bits = __float2int_rz(fmaf(x, FASTEXP_C0, FASTEXP_C1));
    return __int_as_float(bits);
}

extern "C" __global__ void __launch_bounds__(256) attention_f32(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    const int seq_q, const int seq_kv,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;
    int hkv = hq / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);
    int pos_q = position_offset + tq;

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Load Q into shared memory
    const float* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = q_vec[d];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // Compute scores
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv > sliding_window))
            { score_tile[t] = -FLT_MAX; continue; }

            const float* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * k_vec[d];
            score_tile[t] = score * scale;
        }
        __syncthreads();

        // Tile max reduction
        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        // Online softmax rescale
        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? fast_exp_neg(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        __syncthreads();

        // Attention weights
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? fast_exp_neg(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // Accumulate weighted V
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
                if (score_tile[t] > 0.0f)
                    v_acc += score_tile[t] * (v + (size_t)(t_start + t) * kv_stride + hkv * head_dim)[d];
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Normalize and write
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;
    float* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = out_accum[d] * sum_inv;
}
