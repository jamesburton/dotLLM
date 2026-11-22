// attention_flash_mma_prefill_kvq.cu -- issue #1850.
//
// Tensor-core (mma.sync.m16n8k16) flash-style causal PREFILL attention for the dense quantized KV cache
// (CudaQuantizedKvCache), head_dim 64 and 128, reading KV-format Q8_0 / Q4_0 blocks IN PLACE: each K/V
// tile is dequantized in registers as it is loaded into shared memory, so prefill needs no full-history
// FP16 scratch (CLAUDE.md rule 2). Covers the cold prefill (position_offset 0) and the #1847 prefix-reuse
// suffix prefill (position_offset > 0) with one kernel.
//
// ─── Relation to attention_flash_mma_prefill_hd256.cuh (#962 / #1094) ─────────────────────────────────
// Same FlashAttention-2 design, copied rather than shared so the validated hd256 PTX cannot move:
//   * Grid = (ceil(seq_q/16) q-tiles, num_kv_heads), heaviest causal tiles first. Block = `group` warps
//     (one per query head of the GQA group, group <= 8), all sharing the same 16 query rows, so each K/V
//     tile is fetched once and reused `group` times out of shared memory.
//   * Per warp: 16 x HD F32 O accumulator in registers (mma C layout), F32 running max / sum, online
//     softmax in the exp2 domain, P re-packed from the S C-fragment into the PV A-fragment (no smem trip).
//   * Query row i has absolute position position_offset + i and attends keys 0..position_offset+i;
//     seq_kv >= position_offset + seq_q (AttentionSpanInvariant). Keys past seq_kv in the tail tile are
//     clamped to a valid row for the load and masked.
// Differences: HD is a template parameter (64 / 128), the output is F16 (the dense path's attention
// output buffer; no F32 storage), and the quantized loaders are synchronous: a Q8_0 head-row is 136 B
// (hd128) / 68 B (hd64) and a Q4_0 one 72 B / 36 B, none a multiple of 16, so #1094's raw cp.async
// staging (which needs 16-byte rows) does not apply. Each thread decodes its 8-element chunk with the
// #988 block decoders (V2LoadQ8 / V2LoadQ4 in attention_mma_decode_v2.cuh, element =
// __float2half(d * (float)q), the eager dequant kernels' expression) and stores the 8 halves to the
// tile. The F16 entry (cp.async, used by the kernel-level oracle test) runs the same body, so the
// quantized entry is bit-identical to dequant-into-F16 + the F16 entry.
//
// Double buffering is kept: tile t+1 is loaded at the top of iteration t into the buffer last read in
// iteration t-1 (released by that iteration's closing barrier), and published by iteration t's closing
// barrier -- for the synchronous loaders the stores simply complete before it.
//
// ─── Precision ─────────────────────────────────────────────────────────────────────────────────────────
// F16 Q/K/V/P operands, F32 mma accumulation and softmax statistics, in registers. Output F16.
//
// Dynamic shared memory: group*16*(HD+8)*2 (Q) + NBUF*2*KV_TILE*(HD+8)*2 (K, V); hd128 group 8 = 52 KB,
// so the launcher opts the functions into > 48 KB.
//
// Entries: attention_flash_mma_prefill_kvq_hd{64,128}_{q8q8,q4q4,q8q4,q4q8,f16f16}, and the TurboQuant in-place
// attention_flash_mma_prefill_kvq_hd{64,128}_tqtq (#1892, at the end of the file).

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include "attention_mma_decode_v2.cuh"   // V2LoadQ8 / V2LoadQ4 / V2LoadF16 (issue #988)

#define PK_Q_TILE 16
#define PK_KV_TILE 16
#define PK_NBUF 2
#define PK_NT (PK_KV_TILE / 8)
#define PK_LOG2E 1.4426950408889634f
#define PK_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ void pk_ldmatrix_x4(unsigned (&r)[4], const void* p)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pk_ldmatrix_x4_trans(unsigned (&r)[4], const void* p)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pk_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void pk_cp_async16(void* smem_dst, const void* gmem_src)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void pk_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void pk_cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ unsigned pk_pack_half2(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// Tile loaders: chunk(dst, base, elem) fills 8 consecutive halves of the shared tile at dst (16-byte
// aligned) from element elem (multiple of 8) of the logical [seq_kv, num_kv_heads*HD] KV grid at base.
struct PkLoadF16
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        pk_cp_async16(dst, reinterpret_cast<const half*>(base) + elem);
    }
};

template <class L>
struct PkLoadQuant
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        *reinterpret_cast<uint4*>(dst) = L::load8(base, elem);
    }
};

template <int HD, class KL, class VL>
__device__ __forceinline__ void pk_prefill_body(
    const half* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int position_offset,
    const float scale)
{
    constexpr int SS = HD + 8;           // padded row stride (halves): ldmatrix rows in distinct banks
    constexpr int CPR = HD / 8;          // 16-byte chunks per row
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = num_heads / num_kv_heads;
    const int nq_tiles = (seq_q + PK_Q_TILE - 1) / PK_Q_TILE;
    const int qt = nq_tiles - 1 - (int)blockIdx.x;      // heaviest causal tiles first
    const int hkv = blockIdx.y;
    const int hq = hkv * group + warp;
    const int qbase = qt * PK_Q_TILE;
    const int q_stride = num_heads * HD;
    const int kv_stride = num_kv_heads * HD;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* sQ = reinterpret_cast<half*>(smem_raw);                  // [group][16][SS]
    half* sK = sQ + group * PK_Q_TILE * SS;                        // [NBUF][KV_TILE][SS]
    half* sV = sK + PK_NBUF * PK_KV_TILE * SS;                     // [NBUF][KV_TILE][SS]

    for (int i = threadIdx.x; i < group * PK_Q_TILE * CPR; i += blockDim.x)
    {
        int c = i % CPR;
        int r = (i / CPR) % PK_Q_TILE;
        int w = i / (PK_Q_TILE * CPR);
        int qrow = qbase + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (qrow < seq_q)
            val = *reinterpret_cast<const uint4*>(
                q + (size_t)qrow * q_stride + (size_t)(hkv * group + w) * HD + c * 8);
        *reinterpret_cast<uint4*>(sQ + (size_t)(w * PK_Q_TILE + r) * SS + c * 8) = val;
    }

    int kv_end = position_offset + qbase + PK_Q_TILE;
    if (kv_end > seq_kv) kv_end = seq_kv;
    const int num_tiles = (kv_end + PK_KV_TILE - 1) / PK_KV_TILE;

    auto load_tile = [&](int t, int buf)
    {
        const int k0 = t * PK_KV_TILE;
        half* dK = sK + (size_t)buf * PK_KV_TILE * SS;
        half* dV = sV + (size_t)buf * PK_KV_TILE * SS;
        for (int i = threadIdx.x; i < PK_KV_TILE * CPR; i += blockDim.x)
        {
            int c = i % CPR, r = i / CPR;
            int key = k0 + r;
            if (key > seq_kv - 1) key = seq_kv - 1;          // clamp: masked later, stays finite
            size_t g = (size_t)key * kv_stride + (size_t)hkv * HD + c * 8;
            KL::chunk(dK + (size_t)r * SS + c * 8, k, g);
            VL::chunk(dV + (size_t)r * SS + c * 8, v, g);
        }
    };

    float o[HD / 8][4];
    #pragma unroll
    for (int n = 0; n < HD / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m_lo = PK_NEG_INF, m_hi = PK_NEG_INF;
    float l_lo = 0.0f, l_hi = 0.0f;

    const float scale_log2 = scale * PK_LOG2E;
    const int r_lo = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int pos_lo = position_offset + qbase + r_lo;
    const int pos_hi = pos_lo + 8;
    const half* qw = sQ + (size_t)warp * PK_Q_TILE * SS;

    if (num_tiles > 0) { load_tile(0, 0); pk_cp_async_commit(); }

    for (int t = 0; t < num_tiles; t++)
    {
        const int buf = t & 1;
        if (t + 1 < num_tiles) load_tile(t + 1, (t + 1) & 1);
        pk_cp_async_commit();
        pk_cp_async_wait<1>();
        __syncthreads();

        if (warp < group)
        {
            const half* kb = sK + (size_t)buf * PK_KV_TILE * SS;
            const half* vb = sV + (size_t)buf * PK_KV_TILE * SS;
            const int k0 = t * PK_KV_TILE;

            // ---- S = Q . K^T ----
            float sa[PK_NT][4], sb[PK_NT][4];
            #pragma unroll
            for (int n = 0; n < PK_NT; n++)
            { sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f; sb[n][0] = sb[n][1] = sb[n][2] = sb[n][3] = 0.f; }

            #pragma unroll
            for (int ks = 0; ks < HD / 16; ks++)
            {
                unsigned a[4];
                pk_ldmatrix_x4(a, qw + (size_t)(lane & 15) * SS + ks * 16 + (lane >> 4) * 8);
                #pragma unroll
                for (int np = 0; np < PK_NT / 2; np++)
                {
                    unsigned b[4];
                    pk_ldmatrix_x4(b, kb + (size_t)(np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * SS
                                         + ks * 16 + ((lane >> 3) & 1) * 8);
                    if ((ks & 1) == 0)
                    {
                        pk_mma(sa[2 * np], a, b[0], b[1]);
                        pk_mma(sa[2 * np + 1], a, b[2], b[3]);
                    }
                    else
                    {
                        pk_mma(sb[2 * np], a, b[0], b[1]);
                        pk_mma(sb[2 * np + 1], a, b[2], b[3]);
                    }
                }
            }

            // ---- scale, causal/tail mask, tile row max ----
            float tmax_lo = PK_NEG_INF, tmax_hi = PK_NEG_INF;
            #pragma unroll
            for (int n = 0; n < PK_NT; n++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    int key = k0 + n * 8 + col_lo + (j & 1);
                    int pos = (j >> 1) ? pos_hi : pos_lo;
                    float sv = (sa[n][j] + sb[n][j]) * scale_log2;
                    sv = (key <= pos && key < seq_kv) ? sv : PK_NEG_INF;
                    sa[n][j] = sv;
                    if (j >> 1) tmax_hi = fmaxf(tmax_hi, sv); else tmax_lo = fmaxf(tmax_lo, sv);
                }
            }
            tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 1));
            tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 2));
            tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 1));
            tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 2));

            const float mn_lo = fmaxf(m_lo, tmax_lo);
            const float mn_hi = fmaxf(m_hi, tmax_hi);
            const float mu_lo = (mn_lo == PK_NEG_INF) ? 0.0f : mn_lo;
            const float mu_hi = (mn_hi == PK_NEG_INF) ? 0.0f : mn_hi;

            const bool changed = (mn_lo != m_lo) || (mn_hi != m_hi);
            if (__any_sync(0xffffffffu, changed))
            {
                const float c_lo = exp2f(m_lo - mu_lo);
                const float c_hi = exp2f(m_hi - mu_hi);
                l_lo *= c_lo; l_hi *= c_hi;
                #pragma unroll
                for (int n = 0; n < HD / 8; n++)
                { o[n][0] *= c_lo; o[n][1] *= c_lo; o[n][2] *= c_hi; o[n][3] *= c_hi; }
            }
            m_lo = mn_lo; m_hi = mn_hi;

            // ---- P = exp2(S - m), accumulate l, pack as A fragments ----
            unsigned pa[PK_NT / 2][4];
            #pragma unroll
            for (int n = 0; n < PK_NT; n++)
            {
                float p0 = exp2f(sa[n][0] - mu_lo);
                float p1 = exp2f(sa[n][1] - mu_lo);
                float p2 = exp2f(sa[n][2] - mu_hi);
                float p3 = exp2f(sa[n][3] - mu_hi);
                l_lo += p0 + p1; l_hi += p2 + p3;
                pa[n >> 1][(n & 1) * 2 + 0] = pk_pack_half2(p0, p1);
                pa[n >> 1][(n & 1) * 2 + 1] = pk_pack_half2(p2, p3);
            }

            // ---- O += P . V ----
            #pragma unroll
            for (int kk = 0; kk < PK_KV_TILE / 16; kk++)
            {
                #pragma unroll
                for (int np = 0; np < HD / 16; np++)
                {
                    unsigned b[4];
                    pk_ldmatrix_x4_trans(b, vb + (size_t)(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * SS
                                               + (np * 2 + (lane >> 4)) * 8);
                    pk_mma(o[2 * np], pa[kk], b[0], b[1]);
                    pk_mma(o[2 * np + 1], pa[kk], b[2], b[3]);
                }
            }
        }
        __syncthreads();   // all warps done with this buffer before it is refilled
    }

    // ---- epilogue: O / l -> F16 output ----
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 1);
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 2);
    l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 1);
    l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 2);
    const float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
    const float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;

    if (warp < group)
    {
        const int row_lo = qbase + r_lo;
        const int row_hi = row_lo + 8;
        half* out_lo = output + (size_t)row_lo * q_stride + (size_t)hq * HD + col_lo;
        half* out_hi = output + (size_t)row_hi * q_stride + (size_t)hq * HD + col_lo;
        #pragma unroll
        for (int n = 0; n < HD / 8; n++)
        {
            if (row_lo < seq_q)
                *reinterpret_cast<__half2*>(out_lo + n * 8) = __floats2half2_rn(o[n][0] * inv_lo, o[n][1] * inv_lo);
            if (row_hi < seq_q)
                *reinterpret_cast<__half2*>(out_hi + n * 8) = __floats2half2_rn(o[n][2] * inv_hi, o[n][3] * inv_hi);
        }
    }
}

#define PK_ENTRY(NAME, HD, KLOAD, VLOAD)                                                          \
extern "C" __global__ void __launch_bounds__(256) NAME(                                           \
    const half* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,           \
    half* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,            \
    const int num_kv_heads, const int position_offset, const float scale)                         \
{                                                                                                 \
    pk_prefill_body<HD, KLOAD, VLOAD>(q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads,    \
                                      position_offset, scale);                                    \
}

PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q8q8, 128, PkLoadQuant<V2LoadQ8>, PkLoadQuant<V2LoadQ8>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q4q4, 128, PkLoadQuant<V2LoadQ4>, PkLoadQuant<V2LoadQ4>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q8q4, 128, PkLoadQuant<V2LoadQ8>, PkLoadQuant<V2LoadQ4>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q4q8, 128, PkLoadQuant<V2LoadQ4>, PkLoadQuant<V2LoadQ8>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_f16f16, 128, PkLoadF16, PkLoadF16)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q8q8, 64, PkLoadQuant<V2LoadQ8>, PkLoadQuant<V2LoadQ8>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q4q4, 64, PkLoadQuant<V2LoadQ4>, PkLoadQuant<V2LoadQ4>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q8q4, 64, PkLoadQuant<V2LoadQ8>, PkLoadQuant<V2LoadQ4>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q4q8, 64, PkLoadQuant<V2LoadQ4>, PkLoadQuant<V2LoadQ8>)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_f16f16, 64, PkLoadF16, PkLoadF16)

// ─── TurboQuant in place (#1892 Task 6) ────────────────────────────────────────────────────────────────
// The dense in-place TurboQuant cache (CudaTurboQuantInPlaceKvCache) stores each K / V head vector as mse_bits-bit Lloyd-Max
// codes (LSB-first, coordinate i at bit i*mse_bits, code_uints_per_vec words per (position, kv head)) plus one norm per head
// vector. Attention runs in code space: the caller rotates Q into K code space before this kernel (tq_rotate_q_f16) and the
// F16 output back out of V code space after it (tq_rotate_o_f16), so a tile element is simply __float2half(norm * centroid[code])
// -- the same single F16 rounding of an mma operand the F16 / q8 / q4 loaders apply, in code space.
//
// The loader needs per-side context (codes, norms, centroids, bits, stride, norm format) that the stateless chunk(dst, base,
// elem) contract cannot carry, so the TQ entries pass a pointer to a per-side context struct as `base`. Everything is
// force-inlined into the entry, so the struct lives in registers (the PTX has no local depot). pk_prefill_body and the ten
// entries above are untouched, so their PTX cannot move. The centroid table is read with __ldg from global memory, never
// staged in shared memory; the loader is register-only, so the shared-memory size is the same as the q8 / q4 entries'.
// `elem` = key * num_kv_heads * HD + hkv * HD + c * 8, so the head vector is elem / HD and the first coordinate elem % HD.
struct PkTqSide
{
    const uint32_t* codes;
    const void* norms;
    const float* centroids;
    int bits;
    int code_uints_per_vec;
    int norm_format;   // 0 = F32 reference, 1 = F16, 2 = BF16 (turboquant.cu's tq_load_norm)
};

__device__ __forceinline__ float pk_tq_norm(const void* norms, const int fmt, const size_t i)
{
    if (fmt == 0) return __ldg(reinterpret_cast<const float*>(norms) + i);
    const unsigned short h = __ldg(reinterpret_cast<const unsigned short*>(norms) + i);
    if (fmt == 1) return __half2float(__ushort_as_half(h));
    return __uint_as_float(((unsigned)h) << 16);
}

template <int HD>
struct PkLoadTq
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        const PkTqSide& s = *reinterpret_cast<const PkTqSide*>(base);
        const size_t hv = elem / HD;
        const unsigned bit0 = (unsigned)(elem % HD) * (unsigned)s.bits;
        const uint32_t* w = s.codes + hv * (size_t)s.code_uints_per_vec + (bit0 >> 5);
        const unsigned shift = bit0 & 31u;
        const unsigned need = shift + 8u * (unsigned)s.bits;   // bits to read starting at w[0]; reads stay inside the vector
        unsigned long long win = __ldg(w);
        if (need > 32u) win |= (unsigned long long)__ldg(w + 1) << 32;
        win >>= shift;
        if (need > 64u) win |= (unsigned long long)__ldg(w + 2) << (64u - shift);
        const unsigned mask = (1u << s.bits) - 1u;
        const float n = pk_tq_norm(s.norms, s.norm_format, hv);
        unsigned packed[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const float lo = __ldg(&s.centroids[(unsigned)(win >> ((2 * j) * s.bits)) & mask]);
            const float hi = __ldg(&s.centroids[(unsigned)(win >> ((2 * j + 1) * s.bits)) & mask]);
            packed[j] = pk_pack_half2(n * lo, n * hi);
        }
        *reinterpret_cast<uint4*>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
};

#define PK_TQ_ENTRY(NAME, HD)                                                                     \
extern "C" __global__ void __launch_bounds__(256) NAME(                                           \
    const half* __restrict__ q, const uint32_t* __restrict__ k_codes,                             \
    const uint32_t* __restrict__ v_codes, const void* __restrict__ k_norms,                       \
    const void* __restrict__ v_norms, const float* __restrict__ centroids,                        \
    half* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,            \
    const int num_kv_heads, const int position_offset, const float scale,                         \
    const int mse_bits, const int code_uints_per_vec, const int norm_format)                      \
{                                                                                                 \
    const PkTqSide ks = { k_codes, k_norms, centroids, mse_bits, code_uints_per_vec, norm_format };\
    const PkTqSide vs = { v_codes, v_norms, centroids, mse_bits, code_uints_per_vec, norm_format };\
    pk_prefill_body<HD, PkLoadTq<HD>, PkLoadTq<HD>>(q, &ks, &vs, output, seq_q, seq_kv, num_heads, \
                                                    num_kv_heads, position_offset, scale);        \
}

PK_TQ_ENTRY(attention_flash_mma_prefill_kvq_hd128_tqtq, 128)
PK_TQ_ENTRY(attention_flash_mma_prefill_kvq_hd64_tqtq, 64)
