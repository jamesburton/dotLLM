// Q4_1 matrix-vector / small-batch matrix kernels that read the GGUF blocks in place (issue #1172, Flint F6).
//
// block_q4_1 (ggml): { half d; half m; uint8_t qs[16]; } = 20 bytes per 32 values, x = d * q + m, where
// qs[j] low nibble is element j and high nibble is element j + 16.
//
// Mapping: one warp per output row. Four lanes share a block (lane & 3 = 4-byte qs chunk c, covering elements
// 4c..4c+3 from the low nibbles and 16+4c..16+4c+3 from the high nibbles), so one warp pass reads 8 consecutive
// blocks = 160 contiguous bytes of the row. TOK activation rows share every weight load: TOK = 1 is the decode GEMV,
// TOK = 8 the prefill / multi-token GEMM, which reads each weight byte once per 8 tokens. There is no dequant scratch
// and no wider copy of the weights anywhere (CLAUDE.md rule 2).
//
// Activations stay in their native float format (F16 for the dense FP16 loop, F32 for the MoE expert path), and the
// accumulation is F32: sum_i (d q_i + m) x_i = d * sum(q_i x_i) + m * sum(x_i). Unlike a Q8_1 activation path there
// is no quantized block sum stored in F16 that could overflow on large MoE inputs (Flint's routed SwiGLU output
// reaches ~1.3e4; llama.cpp needed an amax rescale for its F16 `ds`, dotLLM CPU a power-of-two rescale, #1158).
//
// Requirements: k % 32 == 0, weight rows contiguous (row stride = k / 32 * 20 bytes, 4-byte aligned).

#include <cuda_fp16.h>
#include <stdint.h>

#define Q4_1_BLOCK_BYTES 20
#define Q4_1_ROWS_PER_BLOCK 8     // 8 warps = 256 threads, one row each
#define Q4_1_TOK_TILE 8           // activation rows per weight load in the GEMM variants

__device__ __forceinline__ void q4_1_load4(const float* p, float* o)
{
    float4 v = *reinterpret_cast<const float4*>(p);
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}

__device__ __forceinline__ void q4_1_load4(const half* p, float* o)
{
    uint2 raw = *reinterpret_cast<const uint2*>(p);
    float2 a = __half22float2(*reinterpret_cast<half2*>(&raw.x));
    float2 b = __half22float2(*reinterpret_cast<half2*>(&raw.y));
    o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;
}

__device__ __forceinline__ void q4_1_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void q4_1_store(half* p, float v) { *p = __float2half(v); }

// y[t, row] = sum_k W[row, k] * x[t, k] for t in [tok0, min(tok0 + TOK, m)). x row stride k, y row stride ldy.
template <int TOK, typename TIn, typename TOut>
__device__ __forceinline__ void q4_1_row(
    const uint8_t* __restrict__ weight, const TIn* __restrict__ x, TOut* __restrict__ y,
    int row, int k, int tok0, int m, int ldy)
{
    const int lane = threadIdx.x & 31;
    const int c = lane & 3;
    const int bpr = k >> 5;
    const uint8_t* w_row = weight + (size_t)row * bpr * Q4_1_BLOCK_BYTES;
    const int tok_count = min(TOK, m - tok0);

    float acc[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; t++) acc[t] = 0.0f;

    for (int b = lane >> 2; b < bpr; b += 8)
    {
        const uint8_t* blk = w_row + (size_t)b * Q4_1_BLOCK_BYTES;
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(blk));
        const uint32_t q = *reinterpret_cast<const uint32_t*>(blk + 4 + 4 * c);
        float wl[4], wh[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            wl[i] = (float)((q >> (8 * i)) & 0xF);
            wh[i] = (float)((q >> (8 * i + 4)) & 0xF);
        }

        #pragma unroll
        for (int t = 0; t < TOK; t++)
        {
            if (t < tok_count)
            {
                const TIn* xb = x + (size_t)(tok0 + t) * k + b * 32;
                float xl[4], xh[4];
                q4_1_load4(xb + 4 * c, xl);
                q4_1_load4(xb + 16 + 4 * c, xh);
                float sqx = 0.0f, sx = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    sqx += wl[i] * xl[i] + wh[i] * xh[i];
                    sx += xl[i] + xh[i];
                }
                acc[t] += dm.x * sqx + dm.y * sx;
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < TOK; t++)
    {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, off);
        if (lane == 0 && t < tok_count)
            q4_1_store(y + (size_t)(tok0 + t) * ldy + row, v);
    }
}

// Decode GEMV, FP16 in/out: y[n] = W[n, k] @ x[k]. Grid: ceil(n / 8). Block: 256.
extern "C" __global__ void __launch_bounds__(256) quantized_gemv_q4_1(
    const uint8_t* __restrict__ weight, const half* __restrict__ x, half* __restrict__ y, const int n, const int k)
{
    const int row = blockIdx.x * Q4_1_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    q4_1_row<1>(weight, x, y, row, k, 0, 1, n);
}

// Multi-token GEMM, FP16 in/out: Y[m, n] = X[m, k] @ W[n, k]^T. Grid: (ceil(n / 8), ceil(m / 8)). Block: 256.
extern "C" __global__ void __launch_bounds__(256) quantized_gemm_q4_1_f16(
    const uint8_t* __restrict__ weight, const half* __restrict__ x, half* __restrict__ y,
    const int n, const int k, const int m)
{
    const int row = blockIdx.x * Q4_1_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    q4_1_row<Q4_1_TOK_TILE>(weight, x, y, row, k, blockIdx.y * Q4_1_TOK_TILE, m, n);
}

// Decode GEMV, FP32 in/out (MoE expert projections run F32 activations). Grid: ceil(n / 8). Block: 256.
extern "C" __global__ void __launch_bounds__(256) quantized_gemv_q4_1_f32(
    const uint8_t* __restrict__ weight, const float* __restrict__ x, float* __restrict__ y, const int n, const int k)
{
    const int row = blockIdx.x * Q4_1_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    q4_1_row<1>(weight, x, y, row, k, 0, 1, n);
}

// Multi-token GEMM, FP32 in/out. Grid: (ceil(n / 8), ceil(m / 8)). Block: 256.
extern "C" __global__ void __launch_bounds__(256) quantized_gemm_q4_1_f32(
    const uint8_t* __restrict__ weight, const float* __restrict__ x, float* __restrict__ y,
    const int n, const int k, const int m)
{
    const int row = blockIdx.x * Q4_1_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n) return;
    q4_1_row<Q4_1_TOK_TILE>(weight, x, y, row, k, blockIdx.y * Q4_1_TOK_TILE, m, n);
}

// MoE grouped decode GEMV (same contract as moe_grouped_gemv_*_f16): outputs[e][m] = W_e[m, K] @ x[K] for the
// K_active experts in one launch. Grid: (ceil(M / 8), K_active). Block: 256.
extern "C" __global__ void __launch_bounds__(256) moe_grouped_gemv_q4_1_f16(
    const half* __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    half* const* __restrict__ outputs,
    const int M, const int K, const int K_active)
{
    const int e = blockIdx.y;
    const int row = blockIdx.x * Q4_1_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (e >= K_active || row >= M) return;
    q4_1_row<1>(weights[e], x, outputs[e], row, K, 0, 1, M);
}

// MoE grouped decode GEMV with one F32 input row PER expert (the routed down projection): y[e * M + m] =
// W_e[m, K] @ x[e * K + :]. x and y are the contiguous [K_active, K] / [K_active, M] F32 expert batches.
// Grid: (ceil(M / 8), K_active). Block: 256.
extern "C" __global__ void __launch_bounds__(256) moe_grouped_gemv_q4_1_f32_xe(
    const float* __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    float* __restrict__ y,
    const int M, const int K, const int K_active)
{
    const int e = blockIdx.y;
    const int row = blockIdx.x * Q4_1_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (e >= K_active || row >= M) return;
    q4_1_row<1>(weights[e], x + (size_t)e * K, y + (size_t)e * M, row, K, 0, 1, M);
}
