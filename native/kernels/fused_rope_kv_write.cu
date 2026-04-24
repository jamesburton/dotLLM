// Fused decode-step RoPE + KV-cache write.
//
// Replaces three separate launches per layer on the eager decode path:
//   1. rope_f16     — rotate Q (in-place) and K (in-place) on the projection scratch
//   2. cuMemcpyDtoDAsync(K_scratch → K_cache_layer + pos*stride)
//   3. cuMemcpyDtoDAsync(V_scratch → V_cache_layer + pos*stride)
//
// Per-launch WDDM overhead is ~22 µs; on a 30-layer model that's 30*3*22 µs ≈ 2 ms
// of pure submission cost per token. The actual compute is trivial — for
// SmolLM-135M (num_heads=9, num_kv_heads=3, head_dim=64, rope_dim=64) it's
// 288 RoPE pairs + 96 K-rope-and-write + 192 V-copy = 576 elements/layer.
//
// Decode-only contract: seq_len == 1 hard-coded — we read positions[0] and write
// to one cache row per K-head and per V-head. Prefill keeps the original
// rope+memcpy path (different per-token destinations make the fused write awkward).
//
// Region layout in the flat thread grid:
//   [0,                   r0)   r0 = num_heads    * (rope_dim/2)   → Q rotation pair (in-place on q_src)
//   [r0,                  r1)   r1 = r0 + num_kv_heads * (rope_dim/2)  → K rotation pair (write to k_cache row)
//   [r1,                  r2)   r2 = r1 + num_kv_heads * (head_dim - rope_dim)  → K tail plain copy (only if rope_dim < head_dim)
//   [r2,                  r3)   r3 = r2 + num_kv_heads * head_dim    → V plain copy (write to v_cache row)
//
// Q is laid out [num_heads * head_dim] in `q_src`; K [num_kv_heads * head_dim] in
// `k_src`; V [num_kv_heads * head_dim] in `v_src`. The fused-QKV decode path
// passes slice offsets into the packed [Q|K|V] buffer — the kernel doesn't care
// whether they alias or are separate allocations.
//
// `cache_pos` is a host-side int (decode token's absolute KV row index). The
// graph-friendly variant `fused_rope_kv_write_f16_dyn` reads it from a device
// pointer instead so the address `cache_K + cache_pos * kv_stride` is computed
// device-side — preventing CUDA Graphs from baking in the row index.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) fused_rope_kv_write_f16(
    half* __restrict__ q_src,            // [num_heads * head_dim]    in/out (rotated in place)
    const half* __restrict__ k_src,      // [num_kv_heads * head_dim]    in (read-only)
    const half* __restrict__ v_src,      // [num_kv_heads * head_dim]    in (read-only)
    half* __restrict__ k_cache,          // [max_seq_len, kv_stride] base ptr for this layer
    half* __restrict__ v_cache,          // [max_seq_len, kv_stride] base ptr for this layer
    const int* __restrict__ positions,   // [1] device int (RoPE position)
    int cache_pos,                       // host-side absolute KV row index
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rope_dim,
    int kv_stride,                       // = num_kv_heads * head_dim
    float theta,
    int rope_type)                       // 0 = standard (interleaved pairs), 1 = neox (split halves)
{
    const int half_rope = rope_dim / 2;
    const int tail = head_dim - rope_dim;

    const int r0 = num_heads * half_rope;
    const int r1 = r0 + num_kv_heads * half_rope;
    const int r2 = r1 + num_kv_heads * tail;
    const int r3 = r2 + num_kv_heads * head_dim;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= r3) return;

    const int pos = positions[0];
    const size_t cache_row_offset = (size_t)cache_pos * (size_t)kv_stride;

    // ── Region 0: Q rotation pairs (in-place on q_src) ──
    if (tid < r0)
    {
        const int pair = tid % half_rope;
        const int head = tid / half_rope;

        const float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        const float angle = (float)pos * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);

        const int base_idx = head * head_dim;
        int i0, i1;
        if (rope_type == 1) // neox
        {
            i0 = base_idx + pair;
            i1 = base_idx + pair + half_rope;
        }
        else // standard
        {
            i0 = base_idx + 2 * pair;
            i1 = base_idx + 2 * pair + 1;
        }

        const float v0 = __half2float(q_src[i0]);
        const float v1 = __half2float(q_src[i1]);
        q_src[i0] = __float2half(v0 * c - v1 * s);
        q_src[i1] = __float2half(v0 * s + v1 * c);
        return;
    }

    // ── Region 1: K rotation pairs (read from k_src, write rotated to k_cache row) ──
    if (tid < r1)
    {
        const int local = tid - r0;
        const int pair = local % half_rope;
        const int head = local / half_rope;

        const float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        const float angle = (float)pos * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);

        const int base_idx = head * head_dim;
        int i0, i1;
        if (rope_type == 1) // neox
        {
            i0 = base_idx + pair;
            i1 = base_idx + pair + half_rope;
        }
        else // standard
        {
            i0 = base_idx + 2 * pair;
            i1 = base_idx + 2 * pair + 1;
        }

        const float v0 = __half2float(k_src[i0]);
        const float v1 = __half2float(k_src[i1]);

        half* k_dst = k_cache + cache_row_offset;
        k_dst[i0] = __float2half(v0 * c - v1 * s);
        k_dst[i1] = __float2half(v0 * s + v1 * c);
        return;
    }

    // ── Region 2: K tail copy (rope_dim < head_dim, no rotation) ──
    if (tid < r2)
    {
        const int local = tid - r1;
        const int t = local % tail;            // tail offset within head
        const int head = local / tail;
        const int idx = head * head_dim + rope_dim + t;
        half* k_dst = k_cache + cache_row_offset;
        k_dst[idx] = k_src[idx];
        return;
    }

    // ── Region 3: V plain copy (no rotation) ──
    {
        const int idx = tid - r2;
        half* v_dst = v_cache + cache_row_offset;
        v_dst[idx] = v_src[idx];
    }
}

// Graph-friendly variant: cache_pos is read from a device pointer.
// Used by the CUDA Graphs decode replay path so the row index can change between
// replays without re-instantiating the graph.
extern "C" __global__ void __launch_bounds__(256) fused_rope_kv_write_f16_dyn(
    half* __restrict__ q_src,
    const half* __restrict__ k_src,
    const half* __restrict__ v_src,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const int* __restrict__ positions,
    const int* __restrict__ cache_pos_ptr,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rope_dim,
    int kv_stride,
    float theta,
    int rope_type)
{
    const int half_rope = rope_dim / 2;
    const int tail = head_dim - rope_dim;

    const int r0 = num_heads * half_rope;
    const int r1 = r0 + num_kv_heads * half_rope;
    const int r2 = r1 + num_kv_heads * tail;
    const int r3 = r2 + num_kv_heads * head_dim;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= r3) return;

    const int pos = positions[0];
    const int cache_pos = cache_pos_ptr[0];
    const size_t cache_row_offset = (size_t)cache_pos * (size_t)kv_stride;

    if (tid < r0)
    {
        const int pair = tid % half_rope;
        const int head = tid / half_rope;
        const float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        const float angle = (float)pos * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const int base_idx = head * head_dim;
        int i0, i1;
        if (rope_type == 1) { i0 = base_idx + pair; i1 = base_idx + pair + half_rope; }
        else { i0 = base_idx + 2 * pair; i1 = base_idx + 2 * pair + 1; }
        const float v0 = __half2float(q_src[i0]);
        const float v1 = __half2float(q_src[i1]);
        q_src[i0] = __float2half(v0 * c - v1 * s);
        q_src[i1] = __float2half(v0 * s + v1 * c);
        return;
    }

    if (tid < r1)
    {
        const int local = tid - r0;
        const int pair = local % half_rope;
        const int head = local / half_rope;
        const float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        const float angle = (float)pos * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const int base_idx = head * head_dim;
        int i0, i1;
        if (rope_type == 1) { i0 = base_idx + pair; i1 = base_idx + pair + half_rope; }
        else { i0 = base_idx + 2 * pair; i1 = base_idx + 2 * pair + 1; }
        const float v0 = __half2float(k_src[i0]);
        const float v1 = __half2float(k_src[i1]);
        half* k_dst = k_cache + cache_row_offset;
        k_dst[i0] = __float2half(v0 * c - v1 * s);
        k_dst[i1] = __float2half(v0 * s + v1 * c);
        return;
    }

    if (tid < r2)
    {
        const int local = tid - r1;
        const int t = local % tail;
        const int head = local / tail;
        const int idx = head * head_dim + rope_dim + t;
        half* k_dst = k_cache + cache_row_offset;
        k_dst[idx] = k_src[idx];
        return;
    }

    {
        const int idx = tid - r2;
        half* v_dst = v_cache + cache_row_offset;
        v_dst[idx] = v_src[idx];
    }
}
