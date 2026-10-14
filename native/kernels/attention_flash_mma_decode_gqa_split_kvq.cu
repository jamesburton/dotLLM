// attention_flash_mma_decode_gqa_split_kvq.cu -- issue #956.
//
// Quantized-KV variants of attention_flash_mma_decode_gqa_split_f16: the K/V history is read
// straight from KV-format Q8_0 (block_q8_0: half d + 32 int8) or Q4_0 (block_q4_0: half d + 16
// bytes, ADJACENT-PAIR nibbles -- the KV convention, not the GGUF weight layout) blocks inside the
// cooperative K/V tile load, instead of first dequantizing the whole history into an F16 scratch
// (which costs a full-history read + write per attention layer per decode step).
//
// Everything except the two tile-load expressions is a verbatim copy of
// attention_flash_mma_decode_gqa_split.cu (that file stays untouched so its validated numerics
// cannot move). Element values are produced with the SAME expression the eager dequant kernels use,
// __float2half(d * (float)q), so output is bit-identical to dequantize-into-F16 + the F16 kernel.
//
// Entries (same launch geometry / arguments as the F16 kernel; k and v are raw block bytes):
//   attention_flash_mma_decode_gqa_split_kvq_q8q8 / _q4q4 / _q8q4 (K Q8_0, V Q4_0) / _q4q8

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>

#define HEAD_DIM 256
#define Q_TILE 16
#define KV_TILE 16
#define MAX_GQA_GROUP 8          // MUST match CudaKernels.MaxGqaGroup / attention_f32.cu's MAX_GQA_GROUP.
#define NUM_WARPS 8               // blockDim = NUM_WARPS*32 = 256, matches this project's BlockSize convention.
#define PV_CHUNKS_PER_WARP (HEAD_DIM / 8 / NUM_WARPS)   // 256/8/8 = 4

// (#501 removed the Schraudolph `fast_exp_neg` helper that used to live here — see the
// Precision section of the file header. Every exponential in this kernel is now `expf`.)

// ── PTX helpers -- duplicated verbatim from attention_flash_mma.cu / v1's decode kernel ───
// (see attention_flash_mma.cu's header for the fragment-layout notes; unchanged here).

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr)
{
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], const void* smem_ptr)
{
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// Load / store a {f32 x4} C/D accumulator fragment from/to a row-major [Q_TILE x HEAD_DIM]
// f32 shared buffer, at the PTX C/D lane map -- identical to v1's decode kernel.
__device__ __forceinline__ void load_o_frag(
    float (&d)[4], const float* sO, int r0, int col_lo, int col_hi)
{
    d[0] = sO[r0 * HEAD_DIM + col_lo];
    d[1] = sO[r0 * HEAD_DIM + col_hi];
    d[2] = sO[(r0 + 8) * HEAD_DIM + col_lo];
    d[3] = sO[(r0 + 8) * HEAD_DIM + col_hi];
}

__device__ __forceinline__ void store_o_frag(
    float* sO, int r0, int col_lo, int col_hi, const float (&d)[4])
{
    sO[r0 * HEAD_DIM + col_lo] = d[0];
    sO[r0 * HEAD_DIM + col_hi] = d[1];
    sO[(r0 + 8) * HEAD_DIM + col_lo] = d[2];
    sO[(r0 + 8) * HEAD_DIM + col_hi] = d[3];
}


// KM/VM: 1 = Q8_0 (34 B / 32 elems), 2 = Q4_0 pair layout (18 B / 32 elems).
// `elem` indexes the logical [seqKv, numKvHeads*HEAD_DIM] element grid; rows are multiples of 32
// elements (HEAD_DIM = 256), so a block never straddles a row.
template <int MODE>
__device__ __forceinline__ half load_kv_elem(const uint8_t* __restrict__ base, size_t elem)
{
    const size_t blk = elem >> 5;
    const int j = (int)(elem & 31);
    if (MODE == 1)
    {
        const uint8_t* b = base + blk * 34;
        float d = __half2float(*reinterpret_cast<const half*>(b));
        int q = (int)reinterpret_cast<const int8_t*>(b + 2)[j];
        return __float2half(d * (float)q);
    }
    else
    {
        const uint8_t* b = base + blk * 18;
        float d = __half2float(*reinterpret_cast<const half*>(b));
        uint8_t p = (b + 2)[j >> 1];
        int q = (j & 1) ? ((int)(p >> 4) - 8) : ((int)(p & 0x0F) - 8);
        return __float2half(d * (float)q);
    }
}

template <int KM, int VM>
__device__ __forceinline__ void attn_decode_gqa_split_kvq_body(
    const half* __restrict__ q,        // [numHeads, HEAD_DIM] row-major (seqQ==1)
    const uint8_t* __restrict__ k,     // [seqKv, numKvHeads*HEAD_DIM] KV-format quantized blocks
    const uint8_t* __restrict__ v,
    float* __restrict__ output,        // [numHeads, HEAD_DIM] row-major, F32
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const float scale,
    const int kv_split,
    float* __restrict__ partial_max,   // [num_heads, kv_split]
    float* __restrict__ partial_sum,   // [num_heads, kv_split]
    float* __restrict__ partial_out)   // [num_heads, kv_split, HEAD_DIM]
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int hkv = blockIdx.x;
    const int s = blockIdx.y;
    const int group = num_heads / num_kv_heads;   // <= MAX_GQA_GROUP, caller-enforced (CanUse).
    const int kv_stride = num_kv_heads * HEAD_DIM;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    __shared__ half sQ[Q_TILE * HEAD_DIM];
    __shared__ half sK[KV_TILE * HEAD_DIM];
    __shared__ half sVt[HEAD_DIM * KV_TILE];       // V transposed: [d][key]
    __shared__ float sO[Q_TILE * HEAD_DIM];        // O accumulator, shared across the WHOLE group
                                                     // (rows 0..group-1 real -- see file header).
    __shared__ float sScore[Q_TILE * KV_TILE];
    __shared__ half sP[Q_TILE * KV_TILE];
    __shared__ float sM[Q_TILE];
    __shared__ float sL[Q_TILE];
    __shared__ float sCorr[Q_TILE];

    // Pack `group` distinct query heads (rows 0..group-1) sharing this KV head into ONE
    // Q_TILE=16 mma tile; rows group..15 zero-padded (never read again -- same "wasted lanes
    // are not a correctness issue" convention v1 used for its 15 padding rows, just now only
    // Q_TILE-group of them are wasted instead of Q_TILE-1).
    for (int i = threadIdx.x; i < Q_TILE * HEAD_DIM; i += blockDim.x)
    {
        int row = i / HEAD_DIM, d = i % HEAD_DIM;
        sQ[i] = (row < group) ? q[(size_t)(hkv * group + row) * HEAD_DIM + d] : __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < Q_TILE * HEAD_DIM; i += blockDim.x)
        sO[i] = 0.0f;
    // sP rows >= group are never written by the softmax phase below (it only ever touches
    // lane < group) -- zero them ONCE here rather than leaving them as uninitialized shared
    // memory. Not a correctness requirement for the REAL rows' mma output (mma.sync computes
    // each output row independently -- a garbage row >= group in the P operand cannot leak
    // into rows < group's accumulation), but avoids reading uninitialized memory at all
    // (matters for tools like compute-sanitizer --tool initcheck) at negligible one-time cost.
    for (int i = threadIdx.x; i < Q_TILE * KV_TILE; i += blockDim.x)
        sP[i] = __float2half(0.0f);
    if ((int)threadIdx.x < group) { sM[threadIdx.x] = -FLT_MAX; sL[threadIdx.x] = 0.0f; }
    __syncthreads();

    const int r0 = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int col_hi = col_lo + 1;

    // This split's contiguous KV sub-range [kv_lo, kv_hi) -- identical chunking to
    // attention_f32_gqa_split_kv (attention_f32.cu), so the two kernels' kvSplit heuristics
    // and MaxSafe* co-residency queries stay directly comparable.
    int chunk = (seq_kv + kv_split - 1) / kv_split;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    for (int k0 = kv_lo; k0 < kv_hi; k0 += KV_TILE)
    {
        int gk_limit = min(KV_TILE, kv_hi - k0);

        // Cooperative K/V tile load -- ALL threadIdx.x (256) help, not just one warp's 32
        // lanes as in v1 -- a real, independent speedup for this phase on top of the mma win.
        for (int i = threadIdx.x; i < KV_TILE * HEAD_DIM; i += blockDim.x)
        {
            int kr = i / HEAD_DIM, d = i % HEAD_DIM;
            int gk = k0 + kr;
            half kv_ = (kr < gk_limit) ? load_kv_elem<KM>(k, (size_t)gk * kv_stride + hkv * HEAD_DIM + d) : __float2half(0.0f);
            sK[i] = kv_;
            half vv = (kr < gk_limit) ? load_kv_elem<VM>(v, (size_t)gk * kv_stride + hkv * HEAD_DIM + d) : __float2half(0.0f);
            sVt[(size_t)d * KV_TILE + kr] = vv;
        }
        __syncthreads();

        // ---- QK: S[16 x 16] = Q_packed . K^T -- warps 0/1 split the 2 N-subtiles (keys
        // 0..7 / 8..15), each doing all HEAD_DIM/16=16 k-steps for its own subtile. Halves
        // v1's serial 32-mma QK chain to 16 (see file header). ----
        if (warp_id < 2)
        {
            int ns = warp_id;
            float d_frag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            #pragma unroll
            for (int ks = 0; ks < HEAD_DIM / 16; ks++)
            {
                unsigned a[4];
                const half* aptr = &sQ[(lane & 15) * HEAD_DIM + ks * 16 + (lane >> 4) * 8];
                ldmatrix_x4(a, aptr);

                unsigned b[2];
                const half* bptr = &sK[(ns * 8 + (lane & 7)) * HEAD_DIM + ks * 16 + ((lane >> 3) & 1) * 8];
                ldmatrix_x2(b, bptr);
                mma_m16n8k16(d_frag, a, b);
            }
            int kcol = ns * 8 + col_lo;
            sScore[r0 * KV_TILE + kcol] = d_frag[0] * scale;
            sScore[r0 * KV_TILE + ns * 8 + col_hi] = d_frag[1] * scale;
            sScore[(r0 + 8) * KV_TILE + kcol] = d_frag[2] * scale;
            sScore[(r0 + 8) * KV_TILE + ns * 8 + col_hi] = d_frag[3] * scale;
        }
        __syncthreads();

        // ---- Online softmax over this KV tile, warp 0 only, one lane per REAL query row
        // (lane < group, not lane < Q_TILE -- v1 iterated the full padded Q_TILE=16 since it
        // only ever had 1 real row; here up to MAX_GQA_GROUP=8 lanes do real work, cheaper
        // and correct). ----
        if (warp_id == 0 && lane < group)
        {
            float m_prev = sM[lane];
            float l_prev = sL[lane];

            float m_cur = m_prev;
            for (int j = 0; j < gk_limit; j++)
            {
                float sc = sScore[lane * KV_TILE + j];
                if (sc > m_cur) m_cur = sc;
            }

            // Precise expf, as everywhere in this kernel since #501. Historically only this
            // cross-tile correction was precise while the per-key P weights used a bit-trick
            // approximation; see the Precision section of the file header.
            float correction = (m_prev == -FLT_MAX) ? 1.0f : expf(m_prev - m_cur);
            float l_cur = l_prev * correction;

            for (int j = 0; j < KV_TILE; j++)
            {
                float p = 0.0f;
                if (j < gk_limit)
                {
                    p = expf(sScore[lane * KV_TILE + j] - m_cur);
                    l_cur += p;
                }
                sP[lane * KV_TILE + j] = __float2half(p);
            }

            sM[lane] = m_cur;
            sL[lane] = l_cur;
            sCorr[lane] = correction;
        }
        __syncthreads();

        // Rescale the shared O accumulator by this tile's per-row correction (real rows only
        // -- rows >= group have no valid sCorr entry and are never read downstream).
        for (int i = threadIdx.x; i < Q_TILE * HEAD_DIM; i += blockDim.x)
        {
            int row = i / HEAD_DIM;
            if (row < group) sO[i] *= sCorr[row];
        }
        __syncthreads();

        // ---- PV: O[16 x 256] += P[16 x 16] . V[16 x 256] -- HEAD_DIM/8=32 d-chunks split
        // across all NUM_WARPS=8 warps (PV_CHUNKS_PER_WARP=4 each), writing DISJOINT sO
        // columns per warp (no inter-warp sync needed within this loop). Real 8-way
        // within-block parallelism, unlike v1's single serialized warp and unlike
        // attention_f32_gqa_split_kv's sequential per-head reduction loop (see file header). ----
        unsigned pa[4];
        const half* pptr = &sP[(lane & 15) * KV_TILE + (lane >> 4) * 8];
        ldmatrix_x4(pa, pptr);

        #pragma unroll
        for (int c = 0; c < PV_CHUNKS_PER_WARP; c++)
        {
            int n = warp_id * PV_CHUNKS_PER_WARP + c;
            unsigned vb[2];
            const half* vptr = &sVt[(n * 8 + (lane & 7)) * KV_TILE + ((lane >> 3) & 1) * 8];
            ldmatrix_x2(vb, vptr);

            float o_frag[4];
            load_o_frag(o_frag, sO, r0, n * 8 + col_lo, n * 8 + col_hi);
            mma_m16n8k16(o_frag, pa, vb);
            store_o_frag(sO, r0, n * 8 + col_lo, n * 8 + col_hi, o_frag);
        }
        // Every warp must finish reading sK/sVt/sP before the next tile's cooperative load
        // overwrites them, and all PV writes to sO must land before the next tile's rescale.
        __syncthreads();
    }

    // Publish this split's partial (UNNORMALIZED) for EACH of this block's group heads, using
    // the EXACT SAME [numHeads, kv_split] / [numHeads, kv_split, headDim] layout
    // attention_f32_gqa_split_kv already defines (see file header) -- the combine phase below
    // and the C# scratch allocator depend on it being unchanged, just indexed by
    // hq = hkv*group + g and fed from THIS kernel's packed sM/sL/sO rows instead of that
    // kernel's per-head register-blocked accumulators.
    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;
        if (threadIdx.x == 0)
        {
            partial_max[(size_t)hq * kv_split + s] = sM[g];
            partial_sum[(size_t)hq * kv_split + s] = sL[g];
        }
        float* partial_out_vec = partial_out + ((size_t)hq * kv_split + s) * HEAD_DIM;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
            partial_out_vec[d] = sO[g * HEAD_DIM + d];
    }

    grid.sync();   // ALL blocks (every kv head, every split) must have published before combine.

    if (s != 0) return; // only the split=0 block per KV head performs the final combine.

    __shared__ float s_combined_max;
    __shared__ float s_combined_sum;

    // ── Combine: ported verbatim from attention_f32_gqa_split_kv (attention_f32.cu) -- see
    // this file's header for why the algebra is not re-derived here. ──
    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;

        if (kv_split == 1)
        {
            // Trivial one-way combine: nothing to reassociate, skip the expf(0) reweighting
            // so this path stays as close as possible to the un-split accumulation.
            if (threadIdx.x == 0)
            {
                s_combined_max = partial_max[(size_t)hq * kv_split + 0];
                s_combined_sum = partial_sum[(size_t)hq * kv_split + 0];
            }
            __syncthreads();
            float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;
            const float* partial_out_vec = partial_out + ((size_t)hq * kv_split + 0) * HEAD_DIM;
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
                output[(size_t)hq * HEAD_DIM + d] = partial_out_vec[d] * sum_inv;
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0)
        {
            float m = -FLT_MAX;
            for (int i = 0; i < kv_split; i++)
                m = fmaxf(m, partial_max[(size_t)hq * kv_split + i]);

            float l = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float li = partial_sum[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? expf(mi - m) : 0.0f;
                l += li * w;
            }
            s_combined_max = m;
            s_combined_sum = l;
        }
        __syncthreads();

        float m = s_combined_max;
        float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;

        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
        {
            float o = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? expf(mi - m) : 0.0f;
                float oi = partial_out[((size_t)hq * kv_split + i) * HEAD_DIM + d];
                o += oi * w;
            }
            output[(size_t)hq * HEAD_DIM + d] = o * sum_inv;
        }
        __syncthreads();
    }
}

#define KVQ_ENTRY(NAME, KM, VM)                                                              \
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32) NAME(                           \
    const half* __restrict__ q, const uint8_t* __restrict__ k, const uint8_t* __restrict__ v, \
    float* __restrict__ output, const int seq_kv, const int num_heads, const int num_kv_heads, \
    const float scale, const int kv_split, float* __restrict__ partial_max,                  \
    float* __restrict__ partial_sum, float* __restrict__ partial_out)                        \
{                                                                                            \
    attn_decode_gqa_split_kvq_body<KM, VM>(q, k, v, output, seq_kv, num_heads, num_kv_heads, \
        scale, kv_split, partial_max, partial_sum, partial_out);                             \
}

KVQ_ENTRY(attention_flash_mma_decode_gqa_split_kvq_q8q8, 1, 1)
KVQ_ENTRY(attention_flash_mma_decode_gqa_split_kvq_q4q4, 2, 2)
KVQ_ENTRY(attention_flash_mma_decode_gqa_split_kvq_q8q4, 1, 2)
KVQ_ENTRY(attention_flash_mma_decode_gqa_split_kvq_q4q8, 2, 1)
