// Tensor-core decode attention v2 body (issue #988): head_dim 256, seqQ == 1, GQA group <= 8.
//
// Shared by attention_flash_mma_decode_gqa_split.cu (FP16 KV) and
// attention_flash_mma_decode_gqa_split_kvq.cu (fused Q8_0/Q4_0 KV): the body is templated on a
// "loader" policy that turns (key row, 8-element-aligned column offset) into 8 packed halves.
// Identical arithmetic in both variants, so fused == dequant-scratch bit-for-bit.
//
// ─── Why v2 ──────────────────────────────────────────────────────────────────────────────
// v1 (KV_TILE=16, 256 threads cooperating on ONE tile) was latency/sync bound: ~12 us per
// 16-key tile per block (5 __syncthreads, a 16-deep serial mma chain on 2 warps, a serial
// 6-lane softmax, shared-memory O read-modify-write, scalar 2-byte K/V loads + transposed
// V stores). v2 removes ALL block-level synchronisation from the key loop:
//
//   * Every warp is an independent online-softmax stream over its own interleaved 16-key tiles
//     (warp w takes tiles w, w+8, w+16, ...). Block-level merge happens ONCE at the end.
//   * K and V are read STRAIGHT FROM GLOBAL INTO MMA FRAGMENT REGISTERS with 16-byte loads.
//     No shared-memory staging, no transposes. This uses the fact that the mma K dimension is
//     a free permutation as long as A and B agree: a thread loads 8 consecutive d values of
//     one key (one LDG.128) and feeds them to two consecutive k16 steps; Q's A fragments are
//     built with the identical permutation. For V (A operand of the O^T = V^T . P^T product)
//     the thread loads 8 consecutive d of one key and the m (d) row assignment is permuted the
//     same way; two keys are interleaved with __byte_perm.
//   * QK: S[q16 x key8] = Q . K^T with M = packed query rows (rows >= group are zero, A regs
//     a1/a3 are literal zeros). PV is computed TRANSPOSED: O^T[d16 x q8] += V^T[d16 x key16] .
//     P^T[key16 x q8], so M = d uses all 16 rows and N = q (8 >= group) — no wasted rows, and
//     the QK C fragment (row q = lane/4, cols = keys 2t,2t+1) is exactly the PV B fragment
//     (k = key slot 2t,2t+1 / n = q = lane/4) with no shuffles (FA2-style register reuse).
//   * Online softmax is warp-parallel (quad max via 2 shuffles, per-thread partial l that is
//     only quad-reduced at the end).
//
// Output layout / scratch layout / cross-split combine are byte-for-byte those of v1.

#pragma once

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <cooperative_groups.h>

#define V2_HEAD_DIM 256
#define V2_WARPS 8
#define V2_TILE 16
#define V2_MAX_GROUP 8

__device__ __forceinline__ void v2_mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                       unsigned a3, unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned v2_pack_half2(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// ── Loader policies: load8(base, elem) -> 8 consecutive halves of the logical
//    [seqKv, numKvHeads*HEAD_DIM] element grid starting at element `elem` (multiple of 8). ──
struct V2LoadF16
{
    __device__ __forceinline__ static uint4 load8(const void* base, size_t elem)
    {
        return __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(base) + elem));
    }
};

// Element loaders for the block decoders below: global memory through the read-only path, or
// plain loads (used on a raw block staged in shared memory by the flash prefill kvq kernel,
// issue #1094). The decode arithmetic is the same code either way.
struct V2GlobalLd
{
    __device__ __forceinline__ static __half h(const __half* p) { return __ldg(p); }
    __device__ __forceinline__ static unsigned u16(const uint16_t* p) { return __ldg(p); }
};
struct V2PlainLd
{
    __device__ __forceinline__ static __half h(const __half* p) { return *p; }
    __device__ __forceinline__ static unsigned u16(const uint16_t* p) { return *p; }
};

// KV-format Q8_0: 34 B / 32 elems (half scale + 32 int8). Blocks are 2-byte aligned only.
struct V2LoadQ8
{
    static constexpr int kBlockBytes = 34;

    __device__ __forceinline__ static uint4 load8(const void* base, size_t elem)
    {
        return decode8<V2GlobalLd>(reinterpret_cast<const uint8_t*>(base) + (elem >> 5) * kBlockBytes,
                                   (int)(elem & 31));
    }

    // b: block start; j: element offset within the block (multiple of 8).
    template <class Ld>
    __device__ __forceinline__ static uint4 decode8(const uint8_t* b, const int j)
    {
        const float d = __half2float(Ld::h(reinterpret_cast<const __half*>(b)));
        const uint16_t* p = reinterpret_cast<const uint16_t*>(b + 2 + j);
        unsigned w[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) w[i] = Ld::u16(p + i);
        unsigned out[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float lo = d * (float)(int)(int8_t)(w[i] & 0xFF);
            float hi = d * (float)(int)(int8_t)(w[i] >> 8);
            __half2 h = __halves2half2(__float2half(lo), __float2half(hi));
            out[i] = *reinterpret_cast<unsigned*>(&h);
        }
        return make_uint4(out[0], out[1], out[2], out[3]);
    }
};

// KV-format Q4_0 pair layout: 18 B / 32 elems (half scale + 16 bytes; element 2i = low nibble
// of byte i, element 2i+1 = high nibble).
struct V2LoadQ4
{
    static constexpr int kBlockBytes = 18;

    __device__ __forceinline__ static uint4 load8(const void* base, size_t elem)
    {
        return decode8<V2GlobalLd>(reinterpret_cast<const uint8_t*>(base) + (elem >> 5) * kBlockBytes,
                                   (int)(elem & 31));
    }

    // b: block start; j: element offset within the block (multiple of 8).
    template <class Ld>
    __device__ __forceinline__ static uint4 decode8(const uint8_t* b, const int j)
    {
        const float d = __half2float(Ld::h(reinterpret_cast<const __half*>(b)));
        const uint16_t* p = reinterpret_cast<const uint16_t*>(b + 2 + (j >> 1));
        unsigned w0 = Ld::u16(p), w1 = Ld::u16(p + 1);
        unsigned bytes[4] = { w0 & 0xFF, w0 >> 8, w1 & 0xFF, w1 >> 8 };
        unsigned out[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float lo = d * (float)((int)(bytes[i] & 0x0F) - 8);
            float hi = d * (float)((int)(bytes[i] >> 4) - 8);
            __half2 h = __halves2half2(__float2half(lo), __float2half(hi));
            out[i] = *reinterpret_cast<unsigned*>(&h);
        }
        return make_uint4(out[0], out[1], out[2], out[3]);
    }
};

template <class KL, class VL>
__device__ __forceinline__ void attn_decode_v2_body(
    const half* __restrict__ q,        // [numHeads, 256] (seqQ == 1)
    const void* __restrict__ k,        // KL-format KV
    const void* __restrict__ v,        // VL-format KV
    float* __restrict__ output,        // [numHeads, 256] F32
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const float scale,
    const int kv_split,
    float* __restrict__ partial_max,   // [num_heads, kv_split]
    float* __restrict__ partial_sum,   // [num_heads, kv_split]
    float* __restrict__ partial_out)   // [num_heads, kv_split, 256]
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int hkv = blockIdx.x;
    const int s = blockIdx.y;
    const int group = num_heads / num_kv_heads;
    const size_t kv_row = (size_t)num_kv_heads * V2_HEAD_DIM;   // elements per key row

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;     // mma "group id": query row (QK) / key column (K load) / d row (V)
    const int t = lane & 3;      // thread-in-quad

    __shared__ uint4 sQf[8][32];                 // Q A-fragments, [beta][lane]
    __shared__ float sO[V2_MAX_GROUP * V2_HEAD_DIM];
    __shared__ float sMw[V2_WARPS][8];
    __shared__ float sLw[V2_WARPS][8];

    // Q A fragments: warp w builds beta = w. Row g (query head hkv*group+g), d = 32*beta + 8t..+7.
    {
        uint4 qf = make_uint4(0, 0, 0, 0);
        if (g < group)
            qf = __ldg(reinterpret_cast<const uint4*>(
                q + (size_t)(hkv * group + g) * V2_HEAD_DIM + 32 * warp + 8 * t));
        sQf[warp][lane] = qf;
    }
    __syncthreads();

    const int chunk = (seq_kv + kv_split - 1) / kv_split;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    float m_run = -FLT_MAX;
    float l_run = 0.0f;
    float o[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { o[i][0] = 0.f; o[i][1] = 0.f; o[i][2] = 0.f; o[i][3] = 0.f; }

    const size_t head_off = (size_t)hkv * V2_HEAD_DIM;

    for (int k0 = kv_lo + warp * V2_TILE; k0 < kv_hi; k0 += V2_TILE * V2_WARPS)
    {
        const int last = kv_hi - 1;

        // ---- K B-fragments: key (k0 + 8j + g), d = 32*beta + 8t.. (one LDG.128 per beta). ----
        uint4 kr[2][8];
        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            const int key = min(k0 + 8 * j + g, last);
            const size_t e = (size_t)key * kv_row + head_off + 8 * t;
            #pragma unroll
            for (int b = 0; b < 8; b++) kr[j][b] = KL::load8(k, e + 32 * b);
        }

        // ---- V rows for this thread: keys 2t, 2t+1, 2t+8, 2t+9 of the tile; d = 64*gq + 8g.. ----
        uint4 vr[4][4];
        {
            const int ka = min(k0 + 2 * t, last), kb = min(k0 + 2 * t + 1, last);
            const int kc = min(k0 + 2 * t + 8, last), kd = min(k0 + 2 * t + 9, last);
            #pragma unroll
            for (int gq = 0; gq < 4; gq++)
            {
                const size_t c = head_off + 64 * gq + 8 * g;
                vr[gq][0] = VL::load8(v, (size_t)ka * kv_row + c);
                vr[gq][1] = VL::load8(v, (size_t)kb * kv_row + c);
                vr[gq][2] = VL::load8(v, (size_t)kc * kv_row + c);
                vr[gq][3] = VL::load8(v, (size_t)kd * kv_row + c);
            }
        }

        // ---- S = Q . K^T for two 8-key tiles; two accumulator chains per tile for ILP. ----
        float sacc[2][2][4];
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int c = 0; c < 2; c++)
                { sacc[j][c][0] = 0.f; sacc[j][c][1] = 0.f; sacc[j][c][2] = 0.f; sacc[j][c][3] = 0.f; }
        #pragma unroll
        for (int b = 0; b < 8; b++)
        {
            const uint4 qf = sQf[b][lane];
            #pragma unroll
            for (int j = 0; j < 2; j++)
            {
                v2_mma(sacc[j][0], qf.x, 0u, qf.y, 0u, kr[j][b].x, kr[j][b].y);
                v2_mma(sacc[j][1], qf.z, 0u, qf.w, 0u, kr[j][b].z, kr[j][b].w);
            }
        }

        // ---- online softmax (this thread owns q = g, keys 8j + 2t + {0,1}) ----
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

        // ---- rescale O^T: this thread's columns are q = 2t, 2t+1; corr lives on lanes of g=2t, 2t+1 ----
        const float ca = __shfl_sync(0xffffffffu, corr, 8 * t);
        const float cb = __shfl_sync(0xffffffffu, corr, 8 * t + 4);
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            o[i][0] *= ca; o[i][2] *= ca;
            o[i][1] *= cb; o[i][3] *= cb;
        }

        // ---- O^T += V^T . P^T ----
        const unsigned pb0 = v2_pack_half2(p[0], p[1]);
        const unsigned pb1 = v2_pack_half2(p[2], p[3]);
        #pragma unroll
        for (int gq = 0; gq < 4; gq++)
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

    // ---- per-warp stats: quad-reduce l, publish (m, l) per query row ----
    l_run += __shfl_xor_sync(0xffffffffu, l_run, 1);
    l_run += __shfl_xor_sync(0xffffffffu, l_run, 2);
    if (t == 0) { sMw[warp][g] = m_run; sLw[warp][g] = l_run; }
    __syncthreads();

    // Block max / weights for the two query columns this thread's O fragments hold.
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

    // Deterministic warp-ordered accumulation into sO[q][d].
    #pragma unroll
    for (int r = 0; r < V2_WARPS; r++)
    {
        if (warp == r)
        {
            #pragma unroll
            for (int i = 0; i < 16; i++)
            {
                const int gq = i >> 2, tau = i & 3;
                const int d0 = 64 * gq + 8 * g + 2 * tau;
                #pragma unroll
                for (int c = 0; c < 2; c++)
                {
                    const int qc = 2 * t + c;
                    const float v0 = o[i][c] * wgt[c];        // (m = g,   n = qc) -> d0
                    const float v1 = o[i][2 + c] * wgt[c];    // (m = g+8, n = qc) -> d0 + 1
                    float* dst = &sO[qc * V2_HEAD_DIM + d0];
                    if (r == 0) { dst[0] = v0; dst[1] = v1; }
                    else        { dst[0] += v0; dst[1] += v1; }
                }
            }
        }
        __syncthreads();
    }

    // ---- publish this block's split partial, v1 layout ----
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
    }
    for (int idx = threadIdx.x; idx < group * V2_HEAD_DIM; idx += blockDim.x)
    {
        const int qc = idx / V2_HEAD_DIM, d = idx % V2_HEAD_DIM;
        const int hq = hkv * group + qc;
        partial_out[((size_t)hq * kv_split + s) * V2_HEAD_DIM + d] = sO[idx];
    }

    grid.sync();   // ALL blocks must have published before the combine.

    if (s != 0) return;

    __shared__ float s_combined_max;
    __shared__ float s_combined_sum;

    // ── Combine: verbatim from v1 / attention_f32_gqa_split_kv. ──
    for (int gg = 0; gg < group; gg++)
    {
        const int hq = hkv * group + gg;

        if (kv_split == 1)
        {
            if (threadIdx.x == 0)
            {
                s_combined_max = partial_max[(size_t)hq * kv_split + 0];
                s_combined_sum = partial_sum[(size_t)hq * kv_split + 0];
            }
            __syncthreads();
            float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;
            const float* pov = partial_out + ((size_t)hq * kv_split + 0) * V2_HEAD_DIM;
            for (int d = threadIdx.x; d < V2_HEAD_DIM; d += blockDim.x)
                output[(size_t)hq * V2_HEAD_DIM + d] = pov[d] * sum_inv;
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

        const float m = s_combined_max;
        const float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;
        for (int d = threadIdx.x; d < V2_HEAD_DIM; d += blockDim.x)
        {
            float oo = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? expf(mi - m) : 0.0f;
                oo += partial_out[((size_t)hq * kv_split + i) * V2_HEAD_DIM + d] * w;
            }
            output[(size_t)hq * V2_HEAD_DIM + d] = oo * sum_inv;
        }
        __syncthreads();
    }
}
