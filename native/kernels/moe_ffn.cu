// MoE (Mixture-of-Experts) helper kernels for the SwiGLU FFN forward path.
// All F32 to mirror the CPU oracle (DotLLM.Cpu.Kernels.MoeSwiGluMlp.Execute /
// ExecuteWithSharedExpert). Phase 1: a tiny set of orchestration kernels —
// the bulk of the math (gate / up / down GEMVs, swiglu, RMSNorm) reuses
// existing F32 launchers.
//
// Kernels in this file:
//   moe_softmax_topk_f32          — per-token softmax over [num_experts]
//                                    logits + lower-index-stable top-k pick
//                                    (matches MoeSwiGluMlp.SelectTopK).
//   moe_renorm_topk_f32           — per-token renormalise topk weights to sum
//                                    to 1.0 (Mixtral / Qwen3-MoE convention).
//   moe_zero_f32                  — zero a flat F32 buffer (used to clear the
//                                    output before per-expert accumulation).
//   moe_axpy_f32                  — y[i] += alpha * x[i] across a row, where
//                                    alpha is read from a host-supplied scalar.
//   moe_axpy_scaled_row_f32       — same but alpha is a fp32 read from a
//                                    device pointer at index `slot`. Used to
//                                    accumulate weight[slot] * down[hidden].
//   moe_sigmoid_logit_f32         — y = 1 / (1 + exp(-Σ x[k] * g[k])) for one
//                                    token (Qwen1.5-MoE shared_expert_gate).

#include <math.h>

// ── Helpers ──────────────────────────────────────────────────────────────

// Block-wide reduction (sum / max) using shared memory + warp shuffles.
// Block size assumed to be 128 threads (4 warps).
__device__ __forceinline__ float block_reduce_sum_128(float v, float* shared)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, off);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        v = (lane < 4) ? shared[lane] : 0.0f;
        for (int off = 2; off > 0; off >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max_128(float v, float* shared)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1)
    {
        float other = __shfl_down_sync(0xFFFFFFFF, v, off);
        if (other > v) v = other;
    }
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        v = (lane < 4) ? shared[lane] : -INFINITY;
        for (int off = 2; off > 0; off >>= 1)
        {
            float other = __shfl_down_sync(0xFFFFFFFF, v, off);
            if (other > v) v = other;
        }
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

// ── moe_softmax_topk_f32 ─────────────────────────────────────────────────
//
// Per-token: softmax over `num_experts` logits, then select the top-k largest
// entries (with ties broken in favour of the lower index — matches
// torch.topk's CPU forward-order behaviour and the CPU oracle's
// MoeSwiGluMlp.SelectTopK).
//
// Output (per token):
//   topk_idx[t * top_k + slot] : int32 — chosen expert index
//   topk_weight[t * top_k + slot] : float — softmax probability
//
// This kernel does NOT renormalise the top-k weights — the renormalise step
// runs as a separate kernel so the caller can disable it for
// `norm_topk_prob=false` configs (Qwen1.5-MoE).
//
// Layout: one block per token, 128 threads. We fan out over the experts in a
// strided loop. The top-k selection is a serialised single-thread max-scan
// (k iterations × num_experts comparisons). For the configurations we care
// about (E ≤ 256, k ≤ 8) this is ~2k cycles — negligible compared to the
// router GEMV / per-expert SwiGLU compute and easy to keep bit-near-identical
// to the CPU oracle.
extern "C" __global__ void __launch_bounds__(128) moe_softmax_topk_f32(
    const float* __restrict__ logits,      // [seq_len, num_experts]
    int*   __restrict__ topk_idx,          // [seq_len, top_k]
    float* __restrict__ topk_weight,       // [seq_len, top_k]
    const int seq_len, const int num_experts, const int top_k)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    extern __shared__ float s_softmax[];
    // Layout:
    //   s_softmax[0 .. num_experts)        : softmax probabilities
    //   s_softmax[num_experts .. n+4)      : warp-reduce scratch (4 floats)
    float* s_probs = s_softmax;
    float* s_scratch = s_softmax + num_experts;

    const float* row = logits + (size_t)t * num_experts;

    // Pass 1: row max.
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
    {
        float v = row[i];
        if (v > local_max) local_max = v;
    }
    float row_max = block_reduce_max_128(local_max, s_scratch);

    // Pass 2: exp(x - max), accumulate sum.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
    {
        float e = expf(row[i] - row_max);
        s_probs[i] = e;
        local_sum += e;
    }
    float row_sum = block_reduce_sum_128(local_sum, s_scratch);
    float inv_sum = 1.0f / row_sum;

    // Pass 3: divide.
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
        s_probs[i] = s_probs[i] * inv_sum;
    __syncthreads();

    // Pass 4: top-k selection. Single-threaded max-scan with index-stable ties
    // (matches MoeSwiGluMlp.SelectTopK exactly: strict `>` keeps the lower
    // index on ties).
    if (threadIdx.x == 0)
    {
        // Local stack-resident "claimed" array and outputs. top_k is small
        // (typically 2..8); cap at 64 to be safe — DeepSeek-V2 / V3 use 6.
        int   sel_idx[64];
        float sel_prob[64];
        for (int slot = 0; slot < top_k; slot++)
        {
            int   best_idx = -1;
            float best_val = -INFINITY;
            for (int i = 0; i < num_experts; i++)
            {
                bool claimed = false;
                for (int p = 0; p < slot; p++) { if (sel_idx[p] == i) { claimed = true; break; } }
                if (claimed) continue;
                float v = s_probs[i];
                if (v > best_val) { best_val = v; best_idx = i; }
            }
            sel_idx[slot] = best_idx;
            sel_prob[slot] = best_val;
        }
        for (int slot = 0; slot < top_k; slot++)
        {
            topk_idx[(size_t)t * top_k + slot] = sel_idx[slot];
            topk_weight[(size_t)t * top_k + slot] = sel_prob[slot];
        }
    }
}

// ── moe_renorm_topk_f32 ──────────────────────────────────────────────────
//
// In-place per-token renormalisation of the top-k weights so they sum to 1.0
// (Mixtral / Qwen3-MoE convention; the CPU oracle does this iff
// `normTopKProb` is true). Single-threaded per token — top_k is tiny.
extern "C" __global__ void __launch_bounds__(32) moe_renorm_topk_f32(
    float* __restrict__ topk_weight,       // [seq_len, top_k]
    const int seq_len, const int top_k)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;
    if (threadIdx.x != 0) return;

    float* row = topk_weight + (size_t)t * top_k;
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) sum += row[i];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < top_k; i++) row[i] *= inv;
}

// ── moe_zero_f32 ─────────────────────────────────────────────────────────
//
// Zero a flat F32 buffer. The CPU oracle clears the per-token accumulator at
// the start of FinalAccumulate; the GPU equivalent is to clear `output`
// before the per-expert weighted-add loop begins.
extern "C" __global__ void __launch_bounds__(256) moe_zero_f32(
    float* __restrict__ buf, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        buf[i] = 0.0f;
}

// ── moe_axpy_scaled_row_f32 ──────────────────────────────────────────────
//
// Per-token weighted accumulator:
//   out[t * hidden + i] += weight[t * top_k + slot] * down[t * hidden + i]
//
// Used after each expert's down projection lands its rows into
// `down_per_token` (one row per output-side selected token). Reading the
// scalar weight from device memory keeps the GPU pipeline pure — no extra
// HtoD per-expert.
//
// Grid: one block per token. Block size 256 with strided loop over `hidden`.
// `weight_offset_in_row` is the slot index inside each token's top_k stripe.
extern "C" __global__ void __launch_bounds__(256) moe_axpy_scaled_row_f32(
    float* __restrict__ output,            // [seq_len, hidden]
    const float* __restrict__ down,        // [seq_len, hidden] (one row per token in this token-set)
    const float* __restrict__ topk_weight, // [seq_len, top_k]
    const int* __restrict__ token_indices, // [batch] absolute token ids ⇒ which output row each batch row updates
    const int batch_size,
    const int hidden,
    const int top_k,
    const int slot_index)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int t = token_indices[b];
    float w = topk_weight[(size_t)t * top_k + slot_index];

    if (w == 0.0f) return;

    const float* down_row = down + (size_t)b * hidden;
    float* out_row = output + (size_t)t * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        float v = down_row[i] * w;
        // Atomic-free: each token row is written by exactly one block here
        // (batch is a permutation of distinct token ids per expert, and
        // different experts run in serialised launches on the same stream).
        out_row[i] = out_row[i] + v;
    }
}

// ── moe_axpy_unweighted_row_f32 ──────────────────────────────────────────
//
// Plain accumulator: `out[t * hidden + i] += down[t * hidden + i]`. Used by
// the shared-expert path (output is summed unconditionally; DeepSeek has no
// shared_expert_gate). Per-token over all `seq_len` tokens — no batch
// indirection.
extern "C" __global__ void __launch_bounds__(256) moe_axpy_unweighted_f32(
    float* __restrict__ output,            // [seq_len, hidden]
    const float* __restrict__ down,        // [seq_len, hidden]
    const int seq_len, const int hidden)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    const float* down_row = down + (size_t)t * hidden;
    float* out_row = output + (size_t)t * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out_row[i] = out_row[i] + down_row[i];
}

// ── moe_axpy_scaled_per_token_f32 ────────────────────────────────────────
//
// Per-token sigmoid-gated accumulator for Qwen1.5-MoE shared_expert_gate:
//   out[t, i] += scale[t] * down[t, i]
//
// Used when the shared-expert branch carries an explicit per-token sigmoid
// scale. DeepSeek-V2/V3 do NOT use this — they pass an empty gate and call
// the unweighted variant instead.
extern "C" __global__ void __launch_bounds__(256) moe_axpy_scaled_per_token_f32(
    float* __restrict__ output,            // [seq_len, hidden]
    const float* __restrict__ down,        // [seq_len, hidden]
    const float* __restrict__ scale,       // [seq_len]
    const int seq_len, const int hidden)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    float s = scale[t];
    if (s == 0.0f) return;

    const float* down_row = down + (size_t)t * hidden;
    float* out_row = output + (size_t)t * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out_row[i] = out_row[i] + s * down_row[i];
}

// ── moe_sigmoid_logit_f32 ────────────────────────────────────────────────
//
// Per-token sigmoid-gated dot product:
//   scale[t] = 1 / (1 + exp(-Σ x[t, k] * g[k]))
//
// Used by Qwen1.5-MoE shared_expert_gate (a [hidden] weight vector). One
// block per token, 128 threads. Trivial cost (single dot product per
// token); kept as a dedicated kernel rather than going through cuBLAS
// because the output is a scalar per token and we want to fold the
// sigmoid in the same launch.
extern "C" __global__ void __launch_bounds__(128) moe_sigmoid_logit_f32(
    const float* __restrict__ hidden,      // [seq_len, hidden_size]
    const float* __restrict__ g,           // [hidden_size]
    float* __restrict__ scale_out,         // [seq_len]
    const int seq_len, const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    extern __shared__ float s_scratch[];

    const float* x_row = hidden + (size_t)t * hidden_size;
    float local = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        local += x_row[i] * g[i];

    float total = block_reduce_sum_128(local, s_scratch);

    if (threadIdx.x == 0)
        scale_out[t] = 1.0f / (1.0f + expf(-total));
}

// ── moe_gather_token_rows_f32 ────────────────────────────────────────────
//
// Gathers `batch_size` rows from a [seq_len, hidden] source into a contiguous
// [batch_size, hidden] destination, indexed by `token_indices[b]` for b ∈
// [0, batch_size). Used by the per-expert grouped path: when expert e has
// been routed to by tokens {t0, t1, ...}, we copy those tokens' hidden rows
// into a contiguous batch buffer for the SwiGLU GEMMs.
extern "C" __global__ void __launch_bounds__(256) moe_gather_token_rows_f32(
    const float* __restrict__ src,         // [seq_len, hidden]
    float* __restrict__ dst,               // [batch_size, hidden]
    const int* __restrict__ token_indices, // [batch_size]
    const int batch_size, const int hidden)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;
    int t = token_indices[b];

    const float* src_row = src + (size_t)t * hidden;
    float* dst_row = dst + (size_t)b * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        dst_row[i] = src_row[i];
}
