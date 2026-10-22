// Multi-head Latent Attention (MLA) Phase B absorbed-attention kernel — FP32.
//
// Implements the equivalent of CPU MlaAttention.ExecuteLatent's attention loop:
//   score[h, t, s] = Q_absorbed[h, t] · c_kv[s] + Q_pe[h, t] · k_pe[s]
//   softmax over causal mask (s <= cachedLength + t == position_offset + t)
//   c_v_out[h, t] = Σ_s softmax · c_kv[s]      (shape [kv_lora_rank])
//
// The key Phase B win: K and V are NOT materialised per head. Instead the
// kernel reads the compact latent c_kv[seq_kv, kv_lora_rank] (one shared row
// per token, no head dim) and the shared k_pe[seq_kv, qk_rope_head_dim] —
// 8-16× smaller than Phase A's expanded per-head K_nope/V/K_pe cache.
//
// Q-side absorption (W_UK^T @ Q_nope) is precomputed by the caller into
// Q_absorbed[seq_q, num_heads, kv_lora_rank]. The post-attention W_UV
// expansion is also caller-side.
//
// Per-(query_token, head) one CUDA block. Block size 128. Online-softmax
// loop tiled over seq_kv in TILE_KV-sized chunks, identical structure to
// the Phase A kernel — only the per-tile dot product changes.
//
// F32 throughout for bit-near-equivalence with the CPU oracle.

#include <float.h>
#include <math.h>

#define TILE_KV 128

extern "C" __global__ void __launch_bounds__(128) attention_mla_latent_f32(
    const float* __restrict__ q_absorbed, // [seq_q, num_heads, kv_lora_rank]
    const float* __restrict__ q_pe,       // [seq_q, num_heads, qk_rope_head_dim]
    const float* __restrict__ c_kv,       // [seq_kv, kv_lora_rank]            (shared across heads)
    const float* __restrict__ k_pe,       // [seq_kv, qk_rope_head_dim]        (shared across heads)
    float* __restrict__ c_v_out,          // [seq_q, num_heads, kv_lora_rank]  (latent V output)
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int kv_lora_rank,
    const int qk_rope_head_dim,
    const int position_offset,
    const float softmax_scale)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;
    int pos_q = position_offset + tq;

    int q_abs_stride = num_heads * kv_lora_rank;
    int q_pe_stride = num_heads * qk_rope_head_dim;

    // Shared layout:
    //   q_abs_shared  [kv_lora_rank]
    //   q_pe_shared   [qk_rope_head_dim]
    //   score_tile    [TILE_KV]
    //   out_accum     [kv_lora_rank]   (latent-dim output)
    //   warp_scratch  [32]
    extern __shared__ float smem[];
    float* q_abs_shared = smem;
    float* q_pe_shared  = q_abs_shared + kv_lora_rank;
    float* score_tile   = q_pe_shared + qk_rope_head_dim;
    float* out_accum    = score_tile + TILE_KV;
    float* warp_scratch = out_accum + kv_lora_rank;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Load Q_absorbed and Q_pe for this (tq, hq) into shared memory.
    const float* q_abs_vec = q_absorbed + (size_t)tq * q_abs_stride + hq * kv_lora_rank;
    const float* q_pe_vec  = q_pe       + (size_t)tq * q_pe_stride  + hq * qk_rope_head_dim;
    for (int d = threadIdx.x; d < kv_lora_rank; d += blockDim.x)
        q_abs_shared[d] = q_abs_vec[d];
    for (int d = threadIdx.x; d < qk_rope_head_dim; d += blockDim.x)
        q_pe_shared[d] = q_pe_vec[d];
    for (int d = threadIdx.x; d < kv_lora_rank; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // ── Compute scores for tokens [t_start, t_end) ──
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            const float* c_kv_vec = c_kv + (size_t)tkv * kv_lora_rank;
            const float* k_pe_vec = k_pe + (size_t)tkv * qk_rope_head_dim;

            float dot = 0.0f;
            for (int d = 0; d < kv_lora_rank; d++)
                dot += q_abs_shared[d] * c_kv_vec[d];
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
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < kv_lora_rank; d += blockDim.x)
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

        // ── Accumulate weighted c_kv into out_accum (latent-dim output) ──
        // Each thread handles a subset of kv_lora_rank, scanning the tile.
        // c_kv is shared across heads — same row for every head's block,
        // good L2 reuse across the (h, t) blocks running concurrently.
        for (int d = threadIdx.x; d < kv_lora_rank; d += blockDim.x) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                if (score_tile[t] > 0.0f) {
                    const float* c_kv_vec = c_kv + (size_t)(t_start + t) * kv_lora_rank;
                    v_acc += score_tile[t] * c_kv_vec[d];
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // ── Normalize and write the latent output ──
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;
    float* out_vec = c_v_out + (size_t)tq * q_abs_stride + hq * kv_lora_rank;
    for (int d = threadIdx.x; d < kv_lora_rank; d += blockDim.x)
        out_vec[d] = out_accum[d] * sum_inv;
}

// ── Q absorption: Q_absorbed[h, t] = W_UK[h]^T @ Q_nope[h, t] ─────────────
//
// W_UK[h] is the per-head slice of kv_b_proj: rows [h * (qkNope + vHead) ..
// h * (qkNope + vHead) + qkNope) of the full row-major kv_b_proj weight,
// each row being kv_lora_rank wide.
//   W_UK[h][j][k] = kv_b_proj[(h * (qkNope + vHead) + j) * kv_lora_rank + k]
//
// Q_absorbed[h, t][k] = Σ_j W_UK[h][j][k] · Q_nope[h, t][j]
//
// Q layout: [seq_q, num_heads, qkNope + qkRope] — Q_nope[h, t] sits at
// offset t * (num_heads * qkHead) + h * qkHead, length qkNope.
// Output layout: [seq_q, num_heads, kv_lora_rank].
//
// Grid: blocks = seq_q * num_heads, each block handles one (t, h) and
// computes kv_lora_rank outputs. Block size 128, threads cooperate on
// emitting the kv_lora_rank-wide output via a strided loop.
extern "C" __global__ void __launch_bounds__(128) mla_q_absorb_uk_f32(
    const float* __restrict__ q,           // [seq_q, num_heads, qkNope + qkRope]
    const float* __restrict__ kv_b_proj,   // [num_heads * (qkNope + vHead), kv_lora_rank]
    float* __restrict__ q_absorbed,        // [seq_q, num_heads, kv_lora_rank]
    const int seq_q,
    const int num_heads,
    const int qk_nope_head_dim,
    const int qk_rope_head_dim,
    const int v_head_dim,
    const int kv_lora_rank)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int t = block_id / num_heads;
    int h = block_id % num_heads;

    int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int per_head_kvb_out = qk_nope_head_dim + v_head_dim;
    int q_stride = num_heads * qk_head_dim;
    int q_abs_stride = num_heads * kv_lora_rank;

    // Q_nope[h, t] — first qk_nope_head_dim of the (t, h) head block.
    const float* q_nope = q + (size_t)t * q_stride + h * qk_head_dim;
    // W_UK[h] starts at row (h * per_head_kvb_out), spans qk_nope_head_dim rows.
    const float* w_uk_base = kv_b_proj + (size_t)(h * per_head_kvb_out) * kv_lora_rank;
    // Output Q_absorbed[h, t].
    float* q_abs = q_absorbed + (size_t)t * q_abs_stride + h * kv_lora_rank;

    for (int k = threadIdx.x; k < kv_lora_rank; k += blockDim.x)
    {
        float acc = 0.0f;
        for (int j = 0; j < qk_nope_head_dim; j++)
            acc += w_uk_base[(size_t)j * kv_lora_rank + k] * q_nope[j];
        q_abs[k] = acc;
    }
}

// ── V expansion: out[h, t] = W_UV[h] @ c_v_out[h, t] ─────────────────────
//
// W_UV[h] is the per-head slice of kv_b_proj: rows [h * (qkNope + vHead) +
// qkNope .. h * (qkNope + vHead) + qkNope + vHead) of kv_b_proj.
//   W_UV[h][v][k] = kv_b_proj[(h * (qkNope + vHead) + qkNope + v) * kv_lora_rank + k]
//
// out[h, t][v] = Σ_k W_UV[h][v][k] · c_v_out[h, t][k]
//
// c_v_out layout: [seq_q, num_heads, kv_lora_rank].
// Output layout: [seq_q, num_heads, v_head_dim] (= attention output before
// o_proj — same shape Phase A's attn_out has).
//
// Grid: blocks = seq_q * num_heads, each block emits v_head_dim outputs.
extern "C" __global__ void __launch_bounds__(128) mla_v_expand_uv_f32(
    const float* __restrict__ c_v_out,     // [seq_q, num_heads, kv_lora_rank]
    const float* __restrict__ kv_b_proj,   // [num_heads * (qkNope + vHead), kv_lora_rank]
    float* __restrict__ attn_out,          // [seq_q, num_heads, v_head_dim]
    const int seq_q,
    const int num_heads,
    const int qk_nope_head_dim,
    const int v_head_dim,
    const int kv_lora_rank)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int t = block_id / num_heads;
    int h = block_id % num_heads;

    int per_head_kvb_out = qk_nope_head_dim + v_head_dim;
    int c_v_stride = num_heads * kv_lora_rank;
    int out_stride = num_heads * v_head_dim;

    const float* c_v_in = c_v_out + (size_t)t * c_v_stride + h * kv_lora_rank;
    // W_UV[h] starts at row (h * per_head_kvb_out + qk_nope_head_dim).
    const float* w_uv_base = kv_b_proj
        + (size_t)(h * per_head_kvb_out + qk_nope_head_dim) * kv_lora_rank;
    float* out_vec = attn_out + (size_t)t * out_stride + h * v_head_dim;

    for (int v = threadIdx.x; v < v_head_dim; v += blockDim.x)
    {
        float acc = 0.0f;
        const float* w_row = w_uv_base + (size_t)v * kv_lora_rank;
        for (int k = 0; k < kv_lora_rank; k++)
            acc += w_row[k] * c_v_in[k];
        out_vec[v] = acc;
    }
}
