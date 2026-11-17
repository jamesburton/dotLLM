// attention_kvq.cu -- issue #1850.
//
// attention_f16 (attention.cu) reading the dense quantized KV cache (CudaQuantizedKvCache) IN PLACE:
// K/V history rows are KV-format Q8_0 (half d + 32 int8, 34 B / 32 elems) or Q4_0 (half d + 16 bytes of
// ADJACENT-PAIR nibbles -- the KV convention, not the GGUF weight layout; 18 B / 32 elems) blocks, and the
// newest rows may sit in an optional FP16 ring (the mixed-precision window). Before #1850 the cache
// dequantized the whole live history into a [maxSeqLen, maxStride] FP16 scratch before every attention
// call (a CLAUDE.md rule-2 expansion); this kernel dequantizes each element in registers instead.
//
// Everything except the K/V element reads is a verbatim copy of attention_f16_body (attention.cu stays
// untouched so its validated SASS cannot move). Each element is produced with the eager dequant kernels'
// expression __float2half(d * (float)q) and then widened with __half2float, exactly what attention_f16
// read from the dequantized scratch, and the reduction order over d / keys is unchanged -- so the output
// is bit-identical to dequant-into-scratch + attention_f16 (CudaAttentionKvqTests checks it bit for bit).
// Compiled with the same flags as attention.cu (precise math, default fmad), which that identity needs.
//
// Row resolution: KV row tkv lives in the quantized buffer when tkv < quant_len, otherwise in the FP16
// ring at slot tkv % window (window == 0: no ring, every row is quantized). The eager entry takes
// quant_len from the host (a rollback can leave it above seq_kv - window); the graph entry derives it as
// max(0, seq_kv - window), the invariant the captured quantize-on-evict maintains.
//
// Format codes (k_fmt / v_fmt): 1 = Q8_0, 2 = Q4_0. The ring is always FP16.
//
// Precision: F32 accumulation in registers / shared memory, F16 output -- identical to attention_f16.
//
// Entries:
//   attention_kvq       -- scalar seq_kv / position_offset / quant_len (eager decode + prefill fallback).
//   attention_kvq_dyn   -- seq_kv / position_offset from device pointers (CUDA-Graphs decode replay).

#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

#define TILE_KV 256
#define KVQ_FMT_Q8_0 1
#define KVQ_FMT_Q4_0 2

// Block formats, specialised at compile time (the format codes are uniform per launch, so a runtime branch per element
// only cost issue slots in this latency-bound kernel: one block per query row and head, so decode runs num_heads blocks).
//
// Element value: the eager dequant kernels write __float2half(d * (float)q). d * q is exact in F32 (an 11-bit by at most
// 8-bit product), so that is the correctly rounded half product, which is what __hmul(d, q) computes. q itself is built
// exactly in half without a conversion instruction: bits 0x6600 + u are the half 1536 + u for 0 <= u < 512 (spacing 1 in
// [1024, 2048)), so __ushort_as_half(0x6600 + (q + bias)) - (1536 + bias) == q exactly. Same bits as before, fewer ops.
template <int FMT> struct KvqFmt;

template <> struct KvqFmt<KVQ_FMT_Q8_0>
{
    static constexpr int kBlockBytes = 34;
    // Element j (0..31) of the block at b, whose scale is d.
    __device__ __forceinline__ static __half elem(const uint8_t* b, const __half d, const int j)
    {
        const unsigned u = (unsigned)b[2 + j] ^ 0x80u;   // int8 q + 128, in [0, 255]
        const __half qh = __hsub(__ushort_as_half((unsigned short)(0x6600u + u)), __ushort_as_half((unsigned short)0x6680u)); // - 1664
        return __hmul(d, qh);
    }
};

template <> struct KvqFmt<KVQ_FMT_Q4_0>
{
    static constexpr int kBlockBytes = 18;
    // KV Q4_0 packs ADJACENT PAIRS (low nibble = even element), not the GGUF weight layout (issue #956).
    __device__ __forceinline__ static __half elem(const uint8_t* b, const __half d, const int j)
    {
        const unsigned byte = b[2 + (j >> 1)];
        const unsigned u = (j & 1) ? (byte >> 4) : (byte & 0x0Fu);   // q + 8, in [0, 15]
        const __half qh = __hsub(__ushort_as_half((unsigned short)(0x6600u + u)), __ushort_as_half((unsigned short)0x6608u)); // - 1544
        return __hmul(d, qh);
    }
};

__device__ __forceinline__ __half kvq_scale(const uint8_t* b) { return *reinterpret_cast<const __half*>(b); }

// Q . K for one quantized K head-row: same d order (0..head_dim-1) and the same fused multiply-add per element as
// attention_f16's `score += q_shared[d] * __half2float(k_vec[d])`.
template <int FMT>
__device__ __forceinline__ float kvq_dot_quant(const float* q_shared, const uint8_t* row, int elem0, int head_dim)
{
    constexpr int BB = KvqFmt<FMT>::kBlockBytes;
    float score = 0.0f;
    // A head that does not start and end on a 32-element block boundary (head_dim % 32 != 0, e.g. hd80 / hd112: the cache
    // only requires the whole KV row to be a multiple of 32) straddles blocks; the block loop below would read the wrong
    // block and run past q_shared. Fall back to one element at a time -- same d order, same per-element expression and FMA.
    if (((elem0 | head_dim) & 31) != 0)
    {
        for (int d = 0; d < head_dim; d++)
        {
            const int e = elem0 + d;
            const uint8_t* b = row + (size_t)(e >> 5) * BB;
            score += q_shared[d] * __half2float(KvqFmt<FMT>::elem(b, kvq_scale(b), e & 31));
        }
        return score;
    }
    for (int bb = 0; bb < head_dim; bb += 32)
    {
        const uint8_t* b = row + (size_t)((elem0 + bb) >> 5) * BB;
        const __half d = kvq_scale(b);
        #pragma unroll 8
        for (int j = 0; j < 32; j++)
            score += q_shared[bb + j] * __half2float(KvqFmt<FMT>::elem(b, d, j));
    }
    return score;
}

template <bool DeviceIndirect, int KF, int VF>
__device__ __forceinline__ void attention_kvq_body(
    const half* __restrict__ q,
    const uint8_t* __restrict__ kq,
    const uint8_t* __restrict__ vq,
    const half* __restrict__ k_ring,
    const half* __restrict__ v_ring,
    half* __restrict__ output,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset,
    int sliding_window,
    const int* __restrict__ seq_kv_ptr,
    const int* __restrict__ position_offset_ptr,
    const float* __restrict__ sinks,
    int quant_len,
    const int window)
{
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;
    const size_t k_row_bytes = (size_t)(kv_stride >> 5) * KvqFmt<KF>::kBlockBytes;
    const size_t v_row_bytes = (size_t)(kv_stride >> 5) * KvqFmt<VF>::kBlockBytes;
    const int elem0 = hkv * head_dim;   // this KV head's first element within a KV row

    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    int* bcast = (int*)(warp_scratch + 30);
    if (DeviceIndirect && threadIdx.x == 0)
    {
        bcast[0] = seq_kv_ptr[0];
        bcast[1] = position_offset_ptr[0];
    }

    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = __half2float(q_vec[d]);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    if (DeviceIndirect)
    {
        seq_kv = bcast[0];
        position_offset = bcast[1];
        quant_len = window > 0 ? max(0, seq_kv - window) : seq_kv;
    }

    int pos_q = position_offset + tq;

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // 2a. Q.K scores for this tile (K read in place).
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;

            if (tkv > pos_q)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            if (sliding_window > 0 && pos_q - tkv >= sliding_window)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            float score;
            if (tkv < quant_len)
            {
                score = kvq_dot_quant<KF>(q_shared, kq + (size_t)tkv * k_row_bytes, elem0, head_dim);
            }
            else
            {
                const half* k_vec = k_ring + (size_t)(tkv % window) * kv_stride + elem0;
                score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += q_shared[d] * __half2float(k_vec[d]);
            }

            score_tile[t] = score * scale;
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
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // 2e. Weighted V (V read in place).
        // Same t order and the same `v_acc += w * v` per weight as attention_f16; the tile is split at quant_len so neither
        // half carries a per-element region branch, and this thread's block offset / in-block index are hoisted.
        int tq = quant_len - t_start;
        tq = tq < 0 ? 0 : (tq > tile_len ? tile_len : tq);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            const int e = elem0 + d;
            const int j = e & 31;
            const uint8_t* vb = vq + (size_t)t_start * v_row_bytes + (size_t)(e >> 5) * KvqFmt<VF>::kBlockBytes;
            for (int t = 0; t < tq; t++, vb += v_row_bytes)
            {
                const float w = score_tile[t];
                if (w > 0.0f)
                    v_acc += w * __half2float(KvqFmt<VF>::elem(vb, kvq_scale(vb), j));
            }
            for (int t = tq; t < tile_len; t++)
            {
                const float w = score_tile[t];
                if (w > 0.0f)
                    v_acc += w * __half2float(v_ring[(size_t)((t_start + t) % window) * kv_stride + e]);
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // gpt-oss attention sinks (#365) -- identical to attention_f16.
    if (sinks != nullptr)
    {
        float sink = sinks[hq];
        float new_max = fmaxf(running_max, sink);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;

        running_sum = running_sum * correction + expf(sink - new_max);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
    }

    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d] * sum_inv);
}

extern "C" __global__ void __launch_bounds__(256) attention_kvq(
    const half* __restrict__ q,
    const uint8_t* __restrict__ kq,
    const uint8_t* __restrict__ vq,
    const half* __restrict__ k_ring,
    const half* __restrict__ v_ring,
    half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window,
    const float* __restrict__ sinks,
    const int k_fmt,
    const int v_fmt,
    const int quant_len,
    const int window)
{
#define KVQ_EAGER(KF, VF) attention_kvq_body<false, KF, VF>(q, kq, vq, k_ring, v_ring, output, seq_q, seq_kv, num_heads, \
        num_kv_heads, head_dim, position_offset, sliding_window, nullptr, nullptr, sinks, quant_len, window)
    if (k_fmt == KVQ_FMT_Q8_0) { if (v_fmt == KVQ_FMT_Q8_0) KVQ_EAGER(KVQ_FMT_Q8_0, KVQ_FMT_Q8_0); else KVQ_EAGER(KVQ_FMT_Q8_0, KVQ_FMT_Q4_0); }
    else                       { if (v_fmt == KVQ_FMT_Q8_0) KVQ_EAGER(KVQ_FMT_Q4_0, KVQ_FMT_Q8_0); else KVQ_EAGER(KVQ_FMT_Q4_0, KVQ_FMT_Q4_0); }
#undef KVQ_EAGER
}

extern "C" __global__ void __launch_bounds__(256) attention_kvq_dyn(
    const half* __restrict__ q,
    const uint8_t* __restrict__ kq,
    const uint8_t* __restrict__ vq,
    const half* __restrict__ k_ring,
    const half* __restrict__ v_ring,
    half* __restrict__ output,
    const int seq_q,
    const int* __restrict__ seq_kv_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int* __restrict__ position_offset_ptr,
    const int sliding_window,
    const float* __restrict__ sinks,
    const int k_fmt,
    const int v_fmt,
    const int window)
{
#define KVQ_DYN(KF, VF) attention_kvq_body<true, KF, VF>(q, kq, vq, k_ring, v_ring, output, seq_q, /*resolved post-barrier*/ 0, \
        num_heads, num_kv_heads, head_dim, /*resolved post-barrier*/ 0, sliding_window, seq_kv_ptr, position_offset_ptr, sinks, \
        /*derived*/ 0, window)
    if (k_fmt == KVQ_FMT_Q8_0) { if (v_fmt == KVQ_FMT_Q8_0) KVQ_DYN(KVQ_FMT_Q8_0, KVQ_FMT_Q8_0); else KVQ_DYN(KVQ_FMT_Q8_0, KVQ_FMT_Q4_0); }
    else                       { if (v_fmt == KVQ_FMT_Q8_0) KVQ_DYN(KVQ_FMT_Q4_0, KVQ_FMT_Q8_0); else KVQ_DYN(KVQ_FMT_Q4_0, KVQ_FMT_Q4_0); }
#undef KVQ_DYN
}
