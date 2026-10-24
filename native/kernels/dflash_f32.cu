// DFlash / DFlash2 block-diffusion drafter kernels (issue #1175).
//
//   * dflash_conv_f32       — DFlash2 grouped dynamic causal convolution along the draft block:
//                             y[p][c] = sum_tap (base[side][tap][c] + dyn[p][side][tap][c / group]) * x[p - tap][c],
//                             zero before the block start. Out of place.
//   * dflash_attention_f32  — non-causal attention of the block's queries over the drafter's context K/V (an F16 ring
//                             indexed by position % ring_cap, keys within the sliding window of each query) followed by
//                             the block's own K/V (F32). One CTA per (query row, head); scores in shared memory.
//
// Reference: z-lab dflash/model.py (pin 07ebd93d) and llama.cpp src/models/dflash.cpp. The CPU twin is
// DotLLM.Models.Architectures.DFlash.DFlashDrafter (ConvInPlace / AttendHead).

#include <math.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256) dflash_conv_f32(
    const float* __restrict__ x, const float* __restrict__ dyn, const float* __restrict__ base,
    float* __restrict__ y, const int rows, const int hidden, const int kernel, const int group, const int side)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hidden) return;
    int p = idx / hidden, c = idx % hidden;
    int groups = hidden / group;
    const float* d = dyn + (size_t)p * 2 * kernel * groups + (size_t)side * kernel * groups;
    float s = 0.f;
    for (int tap = 0; tap < kernel && tap <= p; tap++)
        s += (base[(side * kernel + tap) * hidden + c] + d[tap * groups + c / group]) * x[(size_t)(p - tap) * hidden + c];
    y[idx] = s;
}

// Shared memory: q[head_dim] + scores[n_keys] + red[32].
extern "C" __global__ void __launch_bounds__(128) dflash_attention_f32(
    const float* __restrict__ q,          // [b, heads * hd]
    const float* __restrict__ kb,         // [b, kv_heads * hd]   block keys
    const float* __restrict__ vb,         // [b, kv_heads * hd]   block values
    const __half* __restrict__ ring_k,    // [ring_cap, kv_heads * hd]
    const __half* __restrict__ ring_v,
    float* __restrict__ out,              // [b, heads * hd]
    const int b, const int heads, const int kv_heads, const int hd,
    const int ring_cap, const int lo, const int start, const int window, const float scale)
{
    extern __shared__ float sm[];
    int row = blockIdx.x / heads, head = blockIdx.x % heads;
    int kvh = head / (heads / kv_heads);
    int n_ctx = start - lo, n_keys = n_ctx + b;
    int q_stride = heads * hd, kv_stride = kv_heads * hd;
    float* qs = sm;
    float* sc = sm + hd;
    float* red = sc + n_keys;

    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = q[(size_t)row * q_stride + head * hd + d];
    __syncthreads();

    // Scores.
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x)
    {
        float s = 0.f;
        if (j < n_ctx)
        {
            int pos = lo + j;
            if (start + row - pos >= window) { sc[j] = __int_as_float(0xff800000); continue; }
            const __half* kr = ring_k + (size_t)(pos % ring_cap) * kv_stride + kvh * hd;
            for (int d = 0; d < hd; d++) s += qs[d] * __half2float(kr[d]);
        }
        else
        {
            const float* kr = kb + (size_t)(j - n_ctx) * kv_stride + kvh * hd;
            for (int d = 0; d < hd; d++) s += qs[d] * kr[d];
        }
        sc[j] = s * scale;
    }
    __syncthreads();

    // Max.
    float m = __int_as_float(0xff800000);
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) m = fmaxf(m, sc[j]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float mm = red[0];
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();

    // Exp + sum.
    float sum = 0.f;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x)
    {
        float e = expf(sc[j] - m);
        sc[j] = e;
        sum += e;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float ss = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) ss += red[w];
        red[0] = ss;
    }
    __syncthreads();
    float inv = 1.f / red[0];

    // Weighted values: one output dim per thread.
    for (int d = threadIdx.x; d < hd; d += blockDim.x)
    {
        float acc = 0.f;
        for (int j = 0; j < n_ctx; j++)
            acc += sc[j] * __half2float(ring_v[(size_t)((lo + j) % ring_cap) * kv_stride + kvh * hd + d]);
        for (int j = 0; j < b; j++)
            acc += sc[n_ctx + j] * vb[(size_t)j * kv_stride + kvh * hd + d];
        out[(size_t)row * q_stride + head * hd + d] = acc * inv;
    }
}

// ── Multi-column K-quant GEMV, F32 in / F32 out ─────────────────────────────────────────────────
// y[c][row] = sum_k W[row][k] * x[c][k] for c < ncols (<= 8): the drafter's 2..8-row projections read each weight
// block once for every row of the block instead of once per row (and never widen the weights). One warp per output row,
// lane j owns byte j of every 32-byte quant run, so weight and activation loads are both coalesced.

#define DFLASH_MAX_COLS 8

__device__ __forceinline__ float dflash_warp_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

extern "C" __global__ void __launch_bounds__(128) dflash_gemv_q4_k_multi_f32(
    const uint8_t* __restrict__ weight, const float* __restrict__ x, float* __restrict__ y,
    const int n, const int k, const int ncols)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const int sbs = k / 256;
    const uint8_t* w_row = weight + (size_t)row * sbs * 144;
    float acc[DFLASH_MAX_COLS] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int sb = 0; sb < sbs; sb++)
    {
        const uint8_t* block = w_row + sb * 144;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const uint8_t* sr = block + 4;
        const uint8_t* qs = block + 16;
        #pragma unroll
        for (int pair = 0; pair < 4; pair++)
        {
            int e = pair * 2, o = e + 1;
            int sc0, m0, sc1, m1;
            if (e < 4)
            {
                sc0 = sr[e] & 0x3F; m0 = sr[e + 4] & 0x3F;
                sc1 = sr[o] & 0x3F; m1 = sr[o + 4] & 0x3F;
            }
            else
            {
                sc0 = (sr[e + 4] & 0x0F) | ((sr[e - 4] >> 6) << 4);
                m0 = (sr[e + 4] >> 4) | ((sr[e] >> 6) << 4);
                sc1 = (sr[o + 4] & 0x0F) | ((sr[o - 4] >> 6) << 4);
                m1 = (sr[o + 4] >> 4) | ((sr[o] >> 6) << 4);
            }
            uint8_t q = qs[pair * 32 + lane];
            float w0 = d * (float)sc0 * (float)(q & 0x0F) - dmin * (float)m0;
            float w1 = d * (float)sc1 * (float)(q >> 4) - dmin * (float)m1;
            int xi = sb * 256 + pair * 64 + lane;
            #pragma unroll
            for (int c = 0; c < DFLASH_MAX_COLS; c++)
                if (c < ncols) acc[c] += w0 * x[(size_t)c * k + xi] + w1 * x[(size_t)c * k + xi + 32];
        }
    }
    #pragma unroll
    for (int c = 0; c < DFLASH_MAX_COLS; c++)
    {
        if (c >= ncols) break;
        float s = dflash_warp_sum(acc[c]);
        if (lane == 0) y[(size_t)c * n + row] = s;
    }
}

extern "C" __global__ void __launch_bounds__(128) dflash_gemv_q6_k_multi_f32(
    const uint8_t* __restrict__ weight, const float* __restrict__ x, float* __restrict__ y,
    const int n, const int k, const int ncols)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int l = threadIdx.x & 31;
    if (row >= n) return;
    const int sbs = k / 256;
    const uint8_t* w_row = weight + (size_t)row * sbs * 210;
    float acc[DFLASH_MAX_COLS] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int sb = 0; sb < sbs; sb++)
    {
        const uint8_t* block = w_row + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
        #pragma unroll
        for (int half_ = 0; half_ < 2; half_++)
        {
            const uint8_t* qlh = ql + 64 * half_;
            uint8_t h = qh[32 * half_ + l];
            const int8_t* s = sc + 8 * half_;
            int is = l / 16;
            float w1 = d * s[is + 0] * (float)((int)((qlh[l] & 0xF) | (((h >> 0) & 3) << 4)) - 32);
            float w2 = d * s[is + 2] * (float)((int)((qlh[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            float w3 = d * s[is + 4] * (float)((int)((qlh[l] >> 4) | (((h >> 4) & 3) << 4)) - 32);
            float w4 = d * s[is + 6] * (float)((int)((qlh[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32);
            int xi = sb * 256 + 128 * half_ + l;
            #pragma unroll
            for (int c = 0; c < DFLASH_MAX_COLS; c++)
            {
                if (c >= ncols) break;
                const float* xc = x + (size_t)c * k + xi;
                acc[c] += w1 * xc[0] + w2 * xc[32] + w3 * xc[64] + w4 * xc[96];
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < DFLASH_MAX_COLS; c++)
    {
        if (c >= ncols) break;
        float s = dflash_warp_sum(acc[c]);
        if (l == 0) y[(size_t)c * n + row] = s;
    }
}
