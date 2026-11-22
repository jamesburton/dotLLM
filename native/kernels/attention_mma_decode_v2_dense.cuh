// Tensor-core split-KV decode attention for the dense CudaTransformerModel (issue #1922, E1/E3).
//
// The #988 v2 body (attention_mma_decode_v2.cuh) generalised over the head dim (64 / 128 / 256) and reading K / V through
// loader OBJECTS rather than static policies, so a loader can carry context (TurboQuant codes + norms + a shared-memory
// centroid table). The fragment layout, warp-independent online-softmax streams and in-block warp merge are those of v2,
// line for line; see that file's header for the derivation. attention_mma_decode_v2.cuh itself is NOT changed, so the hybrid
// model's validated hd256 PTX cannot move.
//
// Differences from the #988 body, all deliberate:
//   * HD is a template parameter: KB = HD/32 K "betas" (one LDG.128 each), GQ = HD/64 V groups, NO = HD/16 O fragments.
//     Only warps < KB build Q fragments (every warp still reaches the barrier).
//   * Output is F16 (_state.AttnOutput of the plain model), written once by the combine. No F32 output buffer.
//   * Split partials (James's ruling, 2026-11-20): the partial O is stored NORMALISED (o_i / l_i) in F16,
//     with the running max m_i and sum l_i in F32. Normalising bounds every stored value by max|v|, so
//     F16 cannot overflow however long the split is (the unnormalised #988 form grows with l_i, i.e. with chunk length).
//     The combine is out = sum_i c_i * o^_i with c_i = l_i e^(m_i - M) / sum_j l_j e^(m_j - M), algebraically identical to
//     the #988 combine. An F32-storage control arm measured indistinguishable (Qwen3-4B teacher-forced KL, #1922) and was removed.
//
// Precision: Q/K/V operands F16, mma accumulation F32, online softmax F32 (expf), P packed to F16 for PV (as #988),
// in-block warp merge F32 in shared memory, F32 combine in registers. Stored partials: F16 o^, F32 m, l.

#pragma once

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include "attention_mma_decode_v2.cuh"   // v2_mma, v2_pack_half2, V2LoadF16 / V2LoadQ8 / V2LoadQ4, V2_WARPS / V2_TILE

#define DV2_MAX_SPLIT 64   // MUST match CudaKernels.AttentionMmaDecodeDenseMaxSplit

// ── Loader objects: load8(elem) -> 8 consecutive halves of the logical [seqKv, numKvHeads*HD] grid, elem % 8 == 0. ──
struct DLoadF16
{
    const void* base;
    __device__ __forceinline__ uint4 load8(size_t elem) const { return V2LoadF16::load8(base, elem); }
};
struct DLoadQ8
{
    const void* base;
    __device__ __forceinline__ uint4 load8(size_t elem) const { return V2LoadQ8::load8(base, elem); }
};
struct DLoadQ4
{
    const void* base;
    __device__ __forceinline__ uint4 load8(size_t elem) const { return V2LoadQ4::load8(base, elem); }
};

// TurboQuant in place (#1892 layout): head vector hv = pos * numKvHeads + kvHead holds HD `bits`-bit codes, LSB-first, in
// `cu` 32-bit words, plus one norm per vector. Element = half(norm * centroid[code]) with F32 norm and centroid: the same
// expression as attention_flash_mma_prefill_kvq's PkLoadTq, so decode and prefill decode the history identically. The
// centroid table (2^bits floats, already scaled by 1/sqrt(HD)) is staged once per block in shared memory by the entry.
__device__ __forceinline__ float dv2_tq_norm(const void* norms, const int fmt, const size_t i)
{
    if (fmt == 0) return __ldg(reinterpret_cast<const float*>(norms) + i);
    const unsigned short h = __ldg(reinterpret_cast<const unsigned short*>(norms) + i);
    if (fmt == 1) return __half2float(__ushort_as_half(h));
    return __uint_as_float(((unsigned)h) << 16);
}

template <int HD>
struct DLoadTq
{
    const uint32_t* codes;
    const void* norms;
    const float* cent;      // shared memory, 1 << bits entries
    int bits;
    int cu;                 // code words per head vector
    int norm_format;

    __device__ __forceinline__ uint4 load8(size_t elem) const
    {
        const size_t hv = elem / HD;
        const unsigned bit0 = (unsigned)(elem % HD) * (unsigned)bits;
        const uint32_t* w = codes + hv * (size_t)cu + (bit0 >> 5);
        const unsigned shift = bit0 & 31u;
        const unsigned need = shift + 8u * (unsigned)bits;   // reads stay inside the vector
        unsigned long long win = __ldg(w);
        if (need > 32u) win |= (unsigned long long)__ldg(w + 1) << 32;
        win >>= shift;
        if (need > 64u) win |= (unsigned long long)__ldg(w + 2) << (64u - shift);
        const unsigned mask = (1u << bits) - 1u;
        const float n = dv2_tq_norm(norms, norm_format, hv);
        unsigned packed[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const float lo = cent[(unsigned)(win >> ((2 * j) * bits)) & mask];
            const float hi = cent[(unsigned)(win >> ((2 * j + 1) * bits)) & mask];
            packed[j] = v2_pack_half2(n * lo, n * hi);
        }
        return make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
};

template <int HD, class KL, class VL>
__device__ __forceinline__ void attn_decode_v2_dense_body(
    const half* __restrict__ q,        // [numHeads, HD] (seqQ == 1)
    const KL kl,
    const VL vl,
    half* __restrict__ output,         // [numHeads, HD] F16
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const float scale,
    const int kv_split,                // <= DV2_MAX_SPLIT (host-enforced)
    float* __restrict__ partial_max,   // [num_heads, kv_split] F32
    float* __restrict__ partial_sum,   // [num_heads, kv_split] F32
    __half* __restrict__ partial_out)  // [num_heads, kv_split, HD], NORMALISED o / l
{
    constexpr int KB = HD / 32;   // K betas (32 dims each)
    constexpr int GQ = HD / 64;   // V groups (64 dims each)
    constexpr int NO = HD / 16;   // O^T fragments
    static_assert(HD == 64 || HD == 128 || HD == 256, "HD must be 64, 128 or 256");

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int hkv = blockIdx.x;
    const int s = blockIdx.y;
    const int group = num_heads / num_kv_heads;
    const size_t kv_row = (size_t)num_kv_heads * HD;   // elements per key row

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;

    __shared__ uint4 sQf[KB][32];
    __shared__ float sO[V2_MAX_GROUP * HD];
    __shared__ float sMw[V2_WARPS][8];
    __shared__ float sLw[V2_WARPS][8];
    __shared__ float sLb[V2_MAX_GROUP];
    __shared__ float sC[V2_MAX_GROUP][DV2_MAX_SPLIT];

    // Q A fragments: warp w < KB builds beta = w. Row g (query head hkv*group+g), d = 32*beta + 8t..+7.
    if (warp < KB)
    {
        uint4 qf = make_uint4(0, 0, 0, 0);
        if (g < group)
            qf = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(hkv * group + g) * HD + 32 * warp + 8 * t));
        sQf[warp][lane] = qf;
    }
    __syncthreads();   // also publishes anything the entry staged in shared memory (TQ centroid table)

    const int chunk = (seq_kv + kv_split - 1) / kv_split;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    float m_run = -FLT_MAX;
    float l_run = 0.0f;
    float o[NO][4];
    #pragma unroll
    for (int i = 0; i < NO; i++) { o[i][0] = 0.f; o[i][1] = 0.f; o[i][2] = 0.f; o[i][3] = 0.f; }

    const size_t head_off = (size_t)hkv * HD;

    for (int k0 = kv_lo + warp * V2_TILE; k0 < kv_hi; k0 += V2_TILE * V2_WARPS)
    {
        const int last = kv_hi - 1;

        uint4 kr[2][KB];
        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            const int key = min(k0 + 8 * j + g, last);
            const size_t e = (size_t)key * kv_row + head_off + 8 * t;
            #pragma unroll
            for (int b = 0; b < KB; b++) kr[j][b] = kl.load8(e + 32 * b);
        }

        uint4 vr[GQ][4];
        {
            const int ka = min(k0 + 2 * t, last), kb = min(k0 + 2 * t + 1, last);
            const int kc = min(k0 + 2 * t + 8, last), kd = min(k0 + 2 * t + 9, last);
            #pragma unroll
            for (int gq = 0; gq < GQ; gq++)
            {
                const size_t c = head_off + 64 * gq + 8 * g;
                vr[gq][0] = vl.load8((size_t)ka * kv_row + c);
                vr[gq][1] = vl.load8((size_t)kb * kv_row + c);
                vr[gq][2] = vl.load8((size_t)kc * kv_row + c);
                vr[gq][3] = vl.load8((size_t)kd * kv_row + c);
            }
        }

        float sacc[2][2][4];
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int c = 0; c < 2; c++)
                { sacc[j][c][0] = 0.f; sacc[j][c][1] = 0.f; sacc[j][c][2] = 0.f; sacc[j][c][3] = 0.f; }
        #pragma unroll
        for (int b = 0; b < KB; b++)
        {
            const uint4 qf = sQf[b][lane];
            #pragma unroll
            for (int j = 0; j < 2; j++)
            {
                v2_mma(sacc[j][0], qf.x, 0u, qf.y, 0u, kr[j][b].x, kr[j][b].y);
                v2_mma(sacc[j][1], qf.z, 0u, qf.w, 0u, kr[j][b].z, kr[j][b].w);
            }
        }

        float sv[4];
        float tmax = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int e = 0; e < 2; e++)
            {
                const int key = k0 + 8 * j + 2 * t + e;
                float sc = (sacc[j][0][e] + sacc[j][1][e]) * scale;
                sv[2 * j + e] = (key < kv_hi) ? sc : -FLT_MAX;
                tmax = fmaxf(tmax, sv[2 * j + e]);
            }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        const float m_new = fmaxf(m_run, tmax);
        const float corr = (m_run == -FLT_MAX) ? 1.0f : expf(m_run - m_new);
        float p[4];
        float psum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int e = 0; e < 2; e++)
            {
                const int key = k0 + 8 * j + 2 * t + e;
                float pv = (key < kv_hi) ? expf(sv[2 * j + e] - m_new) : 0.0f;
                p[2 * j + e] = pv;
                psum += pv;
            }
        l_run = l_run * corr + psum;
        m_run = m_new;

        const float ca = __shfl_sync(0xffffffffu, corr, 8 * t);
        const float cb = __shfl_sync(0xffffffffu, corr, 8 * t + 4);
        #pragma unroll
        for (int i = 0; i < NO; i++)
        {
            o[i][0] *= ca; o[i][2] *= ca;
            o[i][1] *= cb; o[i][3] *= cb;
        }

        const unsigned pb0 = v2_pack_half2(p[0], p[1]);
        const unsigned pb1 = v2_pack_half2(p[2], p[3]);
        #pragma unroll
        for (int gq = 0; gq < GQ; gq++)
        {
            const unsigned wa[4] = { vr[gq][0].x, vr[gq][0].y, vr[gq][0].z, vr[gq][0].w };
            const unsigned wb[4] = { vr[gq][1].x, vr[gq][1].y, vr[gq][1].z, vr[gq][1].w };
            const unsigned wc[4] = { vr[gq][2].x, vr[gq][2].y, vr[gq][2].z, vr[gq][2].w };
            const unsigned wd[4] = { vr[gq][3].x, vr[gq][3].y, vr[gq][3].z, vr[gq][3].w };
            #pragma unroll
            for (int tau = 0; tau < 4; tau++)
            {
                const unsigned a0 = __byte_perm(wa[tau], wb[tau], 0x5410);
                const unsigned a1 = __byte_perm(wa[tau], wb[tau], 0x7632);
                const unsigned a2 = __byte_perm(wc[tau], wd[tau], 0x5410);
                const unsigned a3 = __byte_perm(wc[tau], wd[tau], 0x7632);
                v2_mma(o[gq * 4 + tau], a0, a1, a2, a3, pb0, pb1);
            }
        }
    }

    // ---- per-warp stats ----
    l_run += __shfl_xor_sync(0xffffffffu, l_run, 1);
    l_run += __shfl_xor_sync(0xffffffffu, l_run, 2);
    if (t == 0) { sMw[warp][g] = m_run; sLw[warp][g] = l_run; }
    __syncthreads();

    float wgt[2];
    #pragma unroll
    for (int c = 0; c < 2; c++)
    {
        const int qc = 2 * t + c;
        float mx = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < V2_WARPS; w++) mx = fmaxf(mx, sMw[w][qc]);
        const float mw = sMw[warp][qc];
        wgt[c] = (mw > -FLT_MAX + 1.0f) ? expf(mw - mx) : 0.0f;
    }

    #pragma unroll
    for (int r = 0; r < V2_WARPS; r++)
    {
        if (warp == r)
        {
            #pragma unroll
            for (int i = 0; i < NO; i++)
            {
                const int gq = i >> 2, tau = i & 3;
                const int d0 = 64 * gq + 8 * g + 2 * tau;
                #pragma unroll
                for (int c = 0; c < 2; c++)
                {
                    const int qc = 2 * t + c;
                    const float v0 = o[i][c] * wgt[c];
                    const float v1 = o[i][2 + c] * wgt[c];
                    float* dst = &sO[qc * HD + d0];
                    if (r == 0) { dst[0] = v0; dst[1] = v1; }
                    else        { dst[0] += v0; dst[1] += v1; }
                }
            }
        }
        __syncthreads();
    }

    // ---- publish this block's split partial: m, l in F32; o / l in F16 ----
    if (threadIdx.x < group)
    {
        const int qc = threadIdx.x;
        float mx = -FLT_MAX;
        for (int w = 0; w < V2_WARPS; w++) mx = fmaxf(mx, sMw[w][qc]);
        float l = 0.0f;
        for (int w = 0; w < V2_WARPS; w++)
        {
            const float mw = sMw[w][qc];
            const float wg = (mw > -FLT_MAX + 1.0f) ? expf(mw - mx) : 0.0f;
            l += sLw[w][qc] * wg;
        }
        const int hq = hkv * group + qc;
        partial_max[(size_t)hq * kv_split + s] = mx;
        partial_sum[(size_t)hq * kv_split + s] = l;
        sLb[qc] = (l > 1e-10f) ? (1.0f / l) : 0.0f;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < group * HD; idx += blockDim.x)
    {
        const int qc = idx / HD, d = idx % HD;
        const int hq = hkv * group + qc;
        partial_out[((size_t)hq * kv_split + s) * HD + d] = __float2half(sO[idx] * sLb[qc]);
    }

    grid.sync();   // ALL blocks must have published before the combine.

    if (s != 0) return;

    // ---- combine: c_i = l_i e^(m_i - M) / sum_j l_j e^(m_j - M); out = sum_i c_i o^_i ----
    if (threadIdx.x < group)
    {
        const int gg = threadIdx.x;
        const int hq = hkv * group + gg;
        float m = -FLT_MAX;
        for (int i = 0; i < kv_split; i++) m = fmaxf(m, partial_max[(size_t)hq * kv_split + i]);
        float L = 0.0f;
        for (int i = 0; i < kv_split; i++)
        {
            const float mi = partial_max[(size_t)hq * kv_split + i];
            const float wi = (mi > -FLT_MAX + 1.0f) ? expf(mi - m) : 0.0f;
            const float ci = partial_sum[(size_t)hq * kv_split + i] * wi;
            sC[gg][i] = ci;
            L += ci;
        }
        const float inv = (L > 1e-10f) ? (1.0f / L) : 0.0f;
        for (int i = 0; i < kv_split; i++) sC[gg][i] *= inv;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < group * HD; idx += blockDim.x)
    {
        const int gg = idx / HD, d = idx % HD;
        const int hq = hkv * group + gg;
        float oo = 0.0f;
        for (int i = 0; i < kv_split; i++)
            oo += sC[gg][i] * __half2float(partial_out[((size_t)hq * kv_split + i) * HD + d]);
        output[(size_t)hq * HD + d] = __float2half(oo);
    }
}
