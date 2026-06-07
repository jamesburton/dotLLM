// Pre-Q8_1 input quantization kernel.
//
// Quantizes an FP16 activation vector x[k] into INT8 with one FP16 scale per
// 32-element chunk and per-half-chunk FP16 sums. Output layout matches the
// shared-memory scratch produced by the legacy MMQ kernels' Stage 1, but lives
// in device global memory so it can be shared across all GEMV calls in a fused
// projection (and across QKV/GateUp/Down/O/LmHead in a forward pass).
//
// Layout (scratch is one contiguous device allocation per CudaForwardState):
//   int8_t  xq [num_chunks * 32]        // bytes [0,                    k)
//   half    dx [num_chunks]             // bytes [k,             k + 2C)   C = num_chunks
//   half    sx2[num_chunks * 2]         // bytes [k + 2C,        k + 2C + 4C)
//
// where num_chunks = k / 32. sx2[c*2 + 0] = Σ xq[c*32 + 0..15] (lo-half),
// sx2[c*2 + 1] = Σ xq[c*32 + 16..31] (hi-half). Q4_K / Q5_K consume the
// full-chunk sum (sx2[c*2+0] + sx2[c*2+1]) lazily inside the GEMV kernel —
// one extra add per chunk. Q6_K uses both halves separately (its 16-element
// sub-blocks need per-half sums).
//
// Launch: blockDim = (32, 8, 1) = 256 threads. Each warp owns one chunk; one
// warp-stride pass covers num_chunks chunks. gridDim = ceil(num_chunks / 8).
//
// Numerics: bit-near-identical to the legacy in-kernel Stage 1 — same rounding
// (__float2int_rn), clamp to [-127, 127], inv_scale = 127 / max|x|, dx = max|x| / 127.

#include <cuda_fp16.h>
#include <stdint.h>

#define QX_THREADS_X 32
#define QX_WARPS_PER_BLOCK 8
#define QX_THREADS (QX_THREADS_X * QX_WARPS_PER_BLOCK)

extern "C" __global__ void __launch_bounds__(QX_THREADS) quantize_x_to_q8_1(
    const half* __restrict__ x,
    int8_t* __restrict__ xq,
    half*   __restrict__ dx,
    half*   __restrict__ sx2,
    const int k)
{
    const int num_chunks = k >> 5;          // k / 32
    const int warp_id = threadIdx.y;        // 0..QX_WARPS_PER_BLOCK-1
    const int lane    = threadIdx.x;        // 0..31
    const int chunk   = blockIdx.x * QX_WARPS_PER_BLOCK + warp_id;
    if (chunk >= num_chunks) return;

    const int idx = chunk * 32 + lane;
    float v = __half2float(x[idx]);
    float a = fabsf(v);

    // Full-warp max-abs reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
        a = fmaxf(a, other);
    }

    float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
    int qi = __float2int_rn(v * inv_scale);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    xq[idx] = (int8_t)qi;

    // Half-warp sum reduction (offset stops at 8 — lanes 0..15 stay isolated
    // from lanes 16..31). Each half-warp holds the sum of its 16 lanes' qi.
    int s = qi;
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

    if (lane == 0)
    {
        dx[chunk] = __float2half(a / 127.0f);
        sx2[chunk * 2 + 0] = __float2half((float)s);
    }
    if (lane == 16)
    {
        sx2[chunk * 2 + 1] = __float2half((float)s);
    }
}
