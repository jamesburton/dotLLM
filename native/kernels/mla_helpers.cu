// Helper kernels for the MLA Phase A forward path. All F32 to match the
// CPU oracle (DotLLM.Cpu.Kernels.MlaAttention.Execute) byte-for-byte
// algorithmically.

#include <math.h>

// ── Split kv_b_proj output into per-head K_nope and per-head V ──────────
//
// Input layout (per token): [numHeads, qkNopeHeadDim + vHeadDim] contiguous.
// Per head h, the first qkNopeHeadDim floats go to kNope[t, h, :], the
// remaining vHeadDim floats go to v[t, h, :].
//
// Grid: blocks = seqLen * numHeads, one block per (token, head) pair.
// Threads = 128 (cover qkNope + vHead with a strided loop — typical
// V2-Lite has qkNope=128, vHead=128 → 256 elems/block).
extern "C" __global__ void __launch_bounds__(128) mla_split_kv_b_f32(
    const float* __restrict__ kv_b_expanded,   // [seqLen, numHeads * (qkNope + vHead)]
    float* __restrict__ k_nope_dst,            // [seqLen, numHeads * qkNope]
    float* __restrict__ v_dst,                 // [seqLen, numHeads * vHead]
    const int seq_len, const int num_heads,
    const int qk_nope_head_dim, const int v_head_dim)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_len * num_heads) return;

    int t = block_id / num_heads;
    int h = block_id % num_heads;

    int per_head = qk_nope_head_dim + v_head_dim;
    int kv_b_stride = num_heads * per_head;
    int k_stride = num_heads * qk_nope_head_dim;
    int v_stride = num_heads * v_head_dim;

    const float* src_block = kv_b_expanded + (size_t)t * kv_b_stride + h * per_head;
    float* k_dst = k_nope_dst + (size_t)t * k_stride + h * qk_nope_head_dim;
    float* v_out = v_dst + (size_t)t * v_stride + h * v_head_dim;

    for (int d = threadIdx.x; d < qk_nope_head_dim; d += blockDim.x)
        k_dst[d] = src_block[d];
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x)
        v_out[d] = src_block[qk_nope_head_dim + d];
}

// ── Apply RoPE (Norm-pair convention) to the per-head rope portion of Q ──
//
// For each (t, h, i) with i in [0, halfRope):
//   q_pe = q[t, h, qkNope + 2i .. qkNope + 2i + 2)
//   v0 = q_pe[0] * cos[t,i] - q_pe[1] * sin[t,i]
//   v1 = q_pe[1] * cos[t,i] + q_pe[0] * sin[t,i]
//
// Cos/Sin tables: [maxSeq, halfRope] row-major. positionOffset shifts the
// row index so token t reads row (positionOffset + t).
//
// Grid: blocks = seqLen * numHeads, threads = halfRope (rounded up to next
// pow2 / capped at 128 with strided loop).
extern "C" __global__ void __launch_bounds__(128) mla_rope_q_pe_f32(
    float* __restrict__ q,                  // [seqLen, numHeads, qkNope + qkRope]
    const float* __restrict__ cos_tab,      // [maxSeq, qkRope/2]
    const float* __restrict__ sin_tab,      // [maxSeq, qkRope/2]
    const int seq_len, const int num_heads,
    const int qk_nope_head_dim, const int qk_rope_head_dim,
    const int position_offset)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_len * num_heads) return;

    int t = block_id / num_heads;
    int h = block_id % num_heads;
    int half = qk_rope_head_dim / 2;
    int pos = position_offset + t;

    int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int q_stride = num_heads * qk_head_dim;
    float* q_pe = q + (size_t)t * q_stride + h * qk_head_dim + qk_nope_head_dim;
    const float* cos_row = cos_tab + (size_t)pos * half;
    const float* sin_row = sin_tab + (size_t)pos * half;

    for (int i = threadIdx.x; i < half; i += blockDim.x)
    {
        float a = q_pe[2 * i];
        float b = q_pe[2 * i + 1];
        float c = cos_row[i];
        float s = sin_row[i];
        q_pe[2 * i]     = a * c - b * s;
        q_pe[2 * i + 1] = b * c + a * s;
    }
}

// ── Apply RoPE to the MQA-shared K_pe (one rope vector per token, no head dim) ──
//
// Same Norm-pair convention as mla_rope_q_pe_f32 but operates on the
// shared k_pe[seqLen, qkRope] buffer (one row per token, broadcast across
// heads at attention time).
//
// Grid: blocks = seqLen, threads = halfRope (capped at 128 strided loop).
extern "C" __global__ void __launch_bounds__(128) mla_rope_k_pe_f32(
    float* __restrict__ k_pe,               // [seqLen, qkRope]
    const float* __restrict__ cos_tab,      // [maxSeq, qkRope/2]
    const float* __restrict__ sin_tab,      // [maxSeq, qkRope/2]
    const int seq_len, const int qk_rope_head_dim,
    const int position_offset)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;
    int half = qk_rope_head_dim / 2;
    int pos = position_offset + t;

    float* row = k_pe + (size_t)t * qk_rope_head_dim;
    const float* cos_row = cos_tab + (size_t)pos * half;
    const float* sin_row = sin_tab + (size_t)pos * half;

    for (int i = threadIdx.x; i < half; i += blockDim.x)
    {
        float a = row[2 * i];
        float b = row[2 * i + 1];
        float c = cos_row[i];
        float s = sin_row[i];
        row[2 * i]     = a * c - b * s;
        row[2 * i + 1] = b * c + a * s;
    }
}

// ── F32 RMSNorm with explicit (numRows, dim) layout ──────────────────────
//
// Used by the MLA pipeline for q_a_layernorm (numRows=seqLen, dim=qLora)
// and kv_a_layernorm (numRows=seqLen, dim=kvLora). The repo already ships
// rmsnorm_f32 with the same algorithmic shape — this kernel exists as a
// drop-in alias that takes (numRows, dim) at the launch level rather than
// caller-side gridding.
//
// One block per row, threads cooperate on reduction.
extern "C" __global__ void __launch_bounds__(128) mla_rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int dim, const float epsilon)
{
    int row = blockIdx.x;
    const float* in_row = input + (size_t)row * dim;
    float* out_row = output + (size_t)row * dim;

    // Sum of squares — block-strided.
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float x = in_row[i];
        sum_sq += x * x;
    }

    // Block reduction via shared memory + warp shfl.
    __shared__ float warp_sums[4];   // up to 128 threads = 4 warps
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();

    if (warp == 0)
    {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
        if (lane == 0) warp_sums[0] = sum_sq;
    }
    __syncthreads();

    float rms = sqrtf(warp_sums[0] / (float)dim + epsilon);
    float inv = 1.0f / rms;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] = in_row[i] * inv * weight[i];
}
