// attention_kvtq.cu -- issue #1892 (TurboQuant KV, step 2: read in place).
//
// attention_kvq (attention_kvq.cu) reading the dense in-place TurboQuant KV cache (CudaTurboQuantInPlaceKvCache) IN PLACE.
// Each K / V head vector is stored as `mse_bits`-bit Lloyd-Max codes, packed LSB-first (coordinate i at bit i*mse_bits of the
// vector), at a stride of `code_uints_per_vec` 32-bit words per (position, kv head), plus one norm per head vector in the
// cache's norm format. Nothing wider than the codes is ever stored for the history (CLAUDE.md rule 2): each element is
// decoded in registers as norm * centroid[code].
//
// Attention runs in CODE SPACE. The caller has already rotated every query head into K code space,
//     q~ = H(S_k . q) / sqrt(d)                           (tq_rotate_q_f16, before this kernel)
// so score_j = n_j * sum_i q~_i * centroid[code_ij], and this kernel writes the code-space output
//     a~ = sum_j p_j * n_j * centroid[code_j]             (F16, the attention output buffer)
// which the caller rotates back with the V signs, o = S_v . H(a~) / sqrt(d) (tq_rotate_o_f16, after this kernel). Both
// rotations are orthonormal, so this equals attention over the decoded history up to rounding.
//
// Everything except the K / V element reads is a copy of attention_kvq_body without the FP16 window ring (TurboQuant has no
// original-space window): same grid (one block per (query row, head)), same tile size, masking, online softmax, sinks and
// reduction order. The V norm is folded into the softmax weight AFTER the tile sum has taken the raw weight.
// The centroid table is read from global memory through the read-only cache (__ldg), never staged in shared memory.
//
// Precision: F32 accumulation in registers / shared memory, F16 output -- identical to attention_kvq.
// Compiled with the default flags (precise math, default fmad), like attention_kvq.cu.
//
// Entry: attention_kvtq -- scalar seq_kv / position_offset (eager decode, short prefill, sliding window, sinks, any
//        power-of-two head_dim <= 256, GQA group of any size).

#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

#define TILE_KV 256

// Norm storage, identical to turboquant.cu's tq_load_norm: 0 = F32 reference, 1 = F16, 2 = BF16.
__device__ __forceinline__ float kvtq_load_norm(const void* __restrict__ norms, const int fmt, const size_t i)
{
    if (fmt == 0) return __ldg(reinterpret_cast<const float*>(norms) + i);
    const unsigned short h = __ldg(reinterpret_cast<const unsigned short*>(norms) + i);
    if (fmt == 1) return __half2float(__ushort_as_half(h));
    return __uint_as_float(((unsigned)h) << 16);
}

// q~ . centroid[code] over one head vector: walks the vector's words once with a 64-bit bit buffer, reading only the words
// that hold its head_dim * bits code bits (never past the vector, so no guard word is needed). Same d order as attention_kvq.
__device__ __forceinline__ float kvtq_dot(const float* q_shared, const uint32_t* __restrict__ w,
                                          const float* __restrict__ centroids, const int head_dim, const int bits)
{
    const unsigned mask = (1u << bits) - 1u;
    unsigned long long buf = 0ull;
    int have = 0, wi = 0;
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++)
    {
        if (have < bits)
        {
            buf |= (unsigned long long)__ldg(&w[wi++]) << have;
            have += 32;
        }
        const unsigned code = (unsigned)buf & mask;
        buf >>= bits;
        have -= bits;
        score += q_shared[d] * __ldg(&centroids[code]);
    }
    return score;
}

extern "C" __global__ void __launch_bounds__(256) attention_kvtq(
    const half* __restrict__ q,                 // [seq_q, num_heads * head_dim], already rotated into K code space
    const uint32_t* __restrict__ k_codes,       // [pos, kv head, code_uints_per_vec]
    const uint32_t* __restrict__ v_codes,
    const void* __restrict__ k_norms,           // [pos, kv head] in norm_format
    const void* __restrict__ v_norms,
    const float* __restrict__ centroids,        // [1 << mse_bits], scaled by 1/sqrt(head_dim)
    half* __restrict__ output,                  // [seq_q, num_heads * head_dim], V code space
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window,
    const float* __restrict__ sinks,
    const int mse_bits,
    const int code_uints_per_vec,
    const int norm_format)
{
    const int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    const int tq = block_id / num_heads;
    const int hq = block_id % num_heads;
    const int group_size = num_heads / num_kv_heads;
    const int hkv = hq / group_size;

    const float scale = rsqrtf((float)head_dim);
    const int q_stride = num_heads * head_dim;
    const size_t vec_words = (size_t)code_uints_per_vec;

    extern __shared__ float smem[];
    float* q_shared     = smem;
    float* score_tile   = smem + head_dim;
    float* out_accum    = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = __half2float(q_vec[d]);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    const int pos_q = position_offset + tq;

    // This thread's V coordinate (d) bit position inside a head vector: fixed for the whole kernel.
    const unsigned vmask = (1u << mse_bits) - 1u;

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        const int tile_len = t_end - t_start;

        // Q~ . K~ scores for this tile (K codes read in place; one norm per key row).
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            const int tkv = t_start + t;
            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv >= sliding_window))
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }
            const size_t hv = (size_t)tkv * num_kv_heads + hkv;
            const float dot = kvtq_dot(q_shared, k_codes + hv * vec_words, centroids, head_dim, mse_bits);
            score_tile[t] = dot * kvtq_load_norm(k_norms, norm_format, hv) * scale;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();
        if (warp_id == 0)
        {
            const int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        const float new_max = fmaxf(running_max, tile_max);
        const float correction = (running_max > -FLT_MAX + 1.0f) ? expf(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        __syncthreads();

        // Softmax weights. The tile sum takes the raw weight; the stored weight carries the V norm, so the V loop below only
        // multiplies by the centroid (one norm load per key row instead of one per element).
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            const float w = (score_tile[t] > -FLT_MAX + 1.0f) ? expf(score_tile[t] - running_max) : 0.0f;
            tile_sum += w;
            score_tile[t] = w > 0.0f
                ? w * kvtq_load_norm(v_norms, norm_format, (size_t)(t_start + t) * num_kv_heads + hkv)
                : 0.0f;
        }
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();
        if (warp_id == 0)
        {
            const int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // Weighted V in code space: coordinate d of key row t is centroid[code at bit d*bits of that row's vector]. The word
        // and shift are fixed per thread; a code straddles two words only when shift + bits > 32 (bits not dividing 32).
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            const unsigned bitpos = (unsigned)d * (unsigned)mse_bits;
            const unsigned word = bitpos >> 5;
            const unsigned shift = bitpos & 31u;
            const bool straddle = shift + (unsigned)mse_bits > 32u;
            const uint32_t* vw = v_codes + ((size_t)t_start * num_kv_heads + hkv) * vec_words + word;
            const size_t row_words = (size_t)num_kv_heads * vec_words;
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++, vw += row_words)
            {
                const float wn = score_tile[t];
                if (wn > 0.0f)
                {
                    unsigned val = __ldg(vw) >> shift;
                    if (straddle) val |= __ldg(vw + 1) << (32u - shift);
                    v_acc += wn * __ldg(&centroids[val & vmask]);
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // gpt-oss attention sinks (#365) -- identical to attention_kvq.
    if (sinks != nullptr)
    {
        const float sink = sinks[hq];
        const float new_max = fmaxf(running_max, sink);
        const float correction = (running_max > -FLT_MAX + 1.0f) ? expf(running_max - new_max) : 0.0f;
        running_sum = running_sum * correction + expf(sink - new_max);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
    }

    const float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;
    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d] * sum_inv);
}
