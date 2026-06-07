// Multi-head Latent Attention (MLA) Phase A naive forward kernel — FP32.
//
// Implements the equivalent of CPU MlaAttention.Execute's attention loop.
// Each block computes attention for one (query_token, head) pair, scanning
// the cached K_nope (per-head), shared K_pe (broadcast), and V (per-head)
// over [0..seqKv) with a causal mask.
//
// Distinctive vs the standard FP32 attention kernel:
//   - Per-token Q is split into nope (qkNope) + rope (qkRope) sub-dims.
//     Q lives at [seqLen, numHeads, qkNope + qkRope] row-major. The kernel
//     reads both halves and dot-products against:
//         K_nope[s, h, :qkNope]   (per-head, [seqKv, numHeads, qkNope])
//         K_pe[s, :qkRope]        (shared across heads, [seqKv, qkRope])
//   - Per-head V dim (vHead) may differ from qkHead (DeepSeek-V2 typical:
//     qk_head_dim = 192, v_head_dim = 128). Output per (token, head) has
//     vHead floats.
//   - Softmax scale is the explicit "softmax_scale" passed in (callers
//     fold YaRN mscale^2 into it).
//
// Matches MlaAttention.Execute byte-for-byte algorithmically; FP32 throughout
// for bit-near-equivalence with the CPU oracle.

#include <float.h>
#include <math.h>

#define TILE_KV 128

extern "C" __global__ void __launch_bounds__(128) attention_mla_f32(
    const float* __restrict__ q,         // [seqLen, numHeads, qkNope + qkRope]
    const float* __restrict__ k_nope,    // [seqKv, numHeads, qkNope]
    const float* __restrict__ k_pe,      // [seqKv, qkRope]              (shared)
    const float* __restrict__ v,         // [seqKv, numHeads, vHead]
    float* __restrict__ output,          // [seqLen, numHeads, vHead]
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int qk_nope_head_dim,
    const int qk_rope_head_dim,
    const int v_head_dim,
    const int position_offset,
    const float softmax_scale)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;
    int pos_q = position_offset + tq;

    int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int q_stride = num_heads * qk_head_dim;
    int k_nope_stride = num_heads * qk_nope_head_dim;
    int v_stride = num_heads * v_head_dim;

    // Shared layout:
    //   q_nope_shared [qk_nope_head_dim]
    //   q_pe_shared   [qk_rope_head_dim]
    //   score_tile    [TILE_KV]
    //   out_accum     [v_head_dim]
    //   warp_scratch  [32]
    extern __shared__ float smem[];
    float* q_nope_shared = smem;
    float* q_pe_shared   = q_nope_shared + qk_nope_head_dim;
    float* score_tile    = q_pe_shared + qk_rope_head_dim;
    float* out_accum     = score_tile + TILE_KV;
    float* warp_scratch  = out_accum + v_head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Load Q for this (tq, hq) into shared.
    const float* q_vec_nope = q + (size_t)tq * q_stride + hq * qk_head_dim;
    const float* q_vec_pe   = q_vec_nope + qk_nope_head_dim;
    for (int d = threadIdx.x; d < qk_nope_head_dim; d += blockDim.x)
        q_nope_shared[d] = q_vec_nope[d];
    for (int d = threadIdx.x; d < qk_rope_head_dim; d += blockDim.x)
        q_pe_shared[d] = q_vec_pe[d];
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // Compute scores for tokens [t_start, t_end)
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            const float* k_nope_vec = k_nope
                + (size_t)tkv * k_nope_stride
                + hq * qk_nope_head_dim;
            const float* k_pe_vec = k_pe + (size_t)tkv * qk_rope_head_dim;

            float dot = 0.0f;
            for (int d = 0; d < qk_nope_head_dim; d++)
                dot += q_nope_shared[d] * k_nope_vec[d];
            for (int d = 0; d < qk_rope_head_dim; d++)
                dot += q_pe_shared[d] * k_pe_vec[d];

            score_tile[t] = dot * softmax_scale;
        }
        __syncthreads();

        // ── Tile max reduction ──
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

        // ── Online softmax rescale ──
        float new_max = fmaxf(running_max, tile_max);
        // Avoid expf(very-negative) on the first tile when running_max == -FLT_MAX.
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        __syncthreads();

        // ── Attention weights for this tile ──
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? expf(score_tile[t] - running_max) : 0.0f;
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

        // ── Accumulate weighted V ──
        // Each thread handles a subset of v_head_dim, scanning the tile.
        for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                if (score_tile[t] > 0.0f) {
                    const float* v_vec = v
                        + (size_t)(t_start + t) * v_stride
                        + hq * v_head_dim;
                    v_acc += score_tile[t] * v_vec[d];
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Normalize and write
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;
    float* out_vec = output + (size_t)tq * v_stride + hq * v_head_dim;
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
        out_vec[d] = out_accum[d] * sum_inv;
}
