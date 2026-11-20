// TurboQuant Q/O rotation kernels (#1892 step 2): attention runs in code space, so the query is rotated INTO K code space
// and the accumulated output is rotated back OUT of V code space.
//
//   query : q~ = H(S_k . q) / sqrt(d)        score  = n . sum_i q~_i * centroid[code_i]
//   output: o  = S_v . H(a~) / sqrt(d)       a~ = code-space accumulation
//
// Same operation order as TurboQuantCodec.Rotate / InverseRotate (sign flip, unnormalised WHT, then x invSqrtD), and the file
// is built with -fmad=false (native/build_ptx.bat and build.ps1 NO_FMA set, csproj FmadFlag) so rotate matches the CPU codec
// bit for bit. K signs use the codec seed, V signs `seed ^ 0xD1B54A32D192ED03` -- the caller passes the matching vector.
//
// Layout: one CUDA block per (row, head), blockDim.x = 256 (one thread per coordinate; threads with t >= headDim stay live but
// idle so every thread reaches each __syncthreads()). headDim must be a power of two <= 256. In place.
//
// _dyn twins read the live row count from a device pointer so they are CUDA-graph-capture safe: the grid is sized for the
// capture-time maximum and blocks past *rowsPtr * numHeads return before any barrier (the decision is block-uniform).

#include <stdint.h>
#include <cuda_fp16.h>

#define TQ_MAX_HEADDIM 256

__device__ __forceinline__ void tq_wht(float* s, const int t, const int d, const bool live)
{
    for (int len = 1; len < d; len <<= 1)
    {
        if (live && (t & len) == 0)
        {
            const float u = s[t];
            const float v = s[t + len];
            s[t]       = u + v;
            s[t + len] = u - v;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void tq_rotate_q_body(half* __restrict__ q, const float* __restrict__ signs,
                                                 const int headDim, const float invSqrtD, const int b)
{
    __shared__ float s[TQ_MAX_HEADDIM];
    const int t = threadIdx.x;
    const bool live = t < headDim;
    const size_t off = (size_t)b * headDim;
    if (live) s[t] = __half2float(q[off + t]) * signs[t];
    __syncthreads();
    tq_wht(s, t, headDim, live);
    if (live) q[off + t] = __float2half(s[t] * invSqrtD);
}

__device__ __forceinline__ void tq_rotate_o_body(float* __restrict__ o, const float* __restrict__ signs,
                                                 const int headDim, const float invSqrtD, const int b)
{
    __shared__ float s[TQ_MAX_HEADDIM];
    const int t = threadIdx.x;
    const bool live = t < headDim;
    const size_t off = (size_t)b * headDim;
    if (live) s[t] = o[off + t];
    __syncthreads();
    tq_wht(s, t, headDim, live);
    if (live) o[off + t] = s[t] * (invSqrtD * signs[t]);
}

// q: half [rows, numHeads, headDim], rotated in place. Grid = rows * numHeads.
extern "C" __global__ void __launch_bounds__(256) tq_rotate_q_f16(
    half* __restrict__ q, const float* __restrict__ signs,
    const int rows, const int numHeads, const int headDim, const float invSqrtD)
{
    const int b = blockIdx.x;
    if (b >= rows * numHeads) return;
    tq_rotate_q_body(q, signs, headDim, invSqrtD, b);
}

extern "C" __global__ void __launch_bounds__(256) tq_rotate_q_f16_dyn(
    half* __restrict__ q, const float* __restrict__ signs, const int* __restrict__ rowsPtr,
    const int numHeads, const int headDim, const float invSqrtD)
{
    const int b = blockIdx.x;
    if (b >= rowsPtr[0] * numHeads) return;
    tq_rotate_q_body(q, signs, headDim, invSqrtD, b);
}

// o: float [rows, numHeads, headDim], rotated in place (F32 attention accumulator). Grid = rows * numHeads.
extern "C" __global__ void __launch_bounds__(256) tq_rotate_o_f32(
    float* __restrict__ o, const float* __restrict__ signs,
    const int rows, const int numHeads, const int headDim, const float invSqrtD)
{
    const int b = blockIdx.x;
    if (b >= rows * numHeads) return;
    tq_rotate_o_body(o, signs, headDim, invSqrtD, b);
}

extern "C" __global__ void __launch_bounds__(256) tq_rotate_o_f32_dyn(
    float* __restrict__ o, const float* __restrict__ signs, const int* __restrict__ rowsPtr,
    const int numHeads, const int headDim, const float invSqrtD)
{
    const int b = blockIdx.x;
    if (b >= rowsPtr[0] * numHeads) return;
    tq_rotate_o_body(o, signs, headDim, invSqrtD, b);
}
