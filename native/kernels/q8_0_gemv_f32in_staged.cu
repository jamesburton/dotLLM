// Q8_0 GEMV with FP32 input — shared-memory-staged, BIT-IDENTICAL twin of
// quantized_gemv_q8_0_f32in (quantized_gemv_f32in.cu). Issue #482.
//
// y[n] = W_q8_0[n,k] @ x_f32[k], output FP32.
//
// Why: the original kernel is one 256-thread block per row, thread t owning Q8_0 blocks
// t, t+256, ... and reading them straight from global memory. Each warp-wide load of
// x[b*32 + j] touches 32 different 128-byte lines (lanes are 128 B apart), and each byte load
// of qs[j] spans ~9 lines (lanes 34 B apart). The L1 serves those one line per wavefront, so the
// kernel runs at a small fraction of DRAM bandwidth. Bonsai 2's MTP head is entirely Q8_0
// (~450 MB per draft step), so this kernel is most of the ~12 ms draft.
//
// What changes: both operands are staged through shared memory with coalesced global loads —
// the row's weight bytes for a 256-block chunk as 16-byte words, and x in two 16-element halves
// per chunk, padded (stride 17) so the per-thread reads are bank-conflict-free. Two rows share
// each x stage (512 threads = 2 groups of 256), halving x traffic from L2.
//
// What does NOT change — the arithmetic, operation for operation:
//   * thread t of a row's 256-thread group owns blocks t, t+256, ... in increasing order
//     (chunk c's block c*256 + t), exactly as the original's `b += blockDim.x` loop;
//   * per block: s = 0; s = fma((float)q_j, x_j, s) for j = 0..31 in order; acc = fma(d, s, acc)
//     — the original's PTX is exactly these fma.rn.f32 (nvcc default -fmad=true contracts its
//     `s += q*x` / `acc += d*s`); here they are explicit __fmaf_rn, so the result does not
//     depend on -fmad;
//   * the same warp shuffle tree and the same 8-warp second stage (lanes >= 8 add 0.0f).
// So y is bit-identical to quantized_gemv_q8_0_f32in for every input.
//
// Requirements: k % 32 == 0 (Q8_0). Weight rows need no particular alignment (byte fallback).
// Launch: grid = ceil(n / Q8S_ROWS), block = Q8S_ROWS * Q8S_GROUP, no dynamic shared memory.
//
// Build (compute_75 like the rest of the tree; the fmad flag is irrelevant to the result):
//   nvcc -ptx -arch=compute_75 -o native/ptx/q8_0_gemv_f32in_staged.ptx native/kernels/q8_0_gemv_f32in_staged.cu

#include <cuda_fp16.h>
#include <stdint.h>

#define Q8S_GROUP 256        // threads per row — MUST equal the original kernel's blockDim
#define Q8S_ROWS 2           // rows per thread block (share one x stage)
#define Q8S_BLOCK_BYTES 34   // Q8_0 block: f16 scale + 32 int8
#define Q8S_HALF 16          // x elements per block staged per pass
#define Q8S_XPAD 17          // odd stride -> conflict-free per-thread reads

__device__ __forceinline__ void q8s_stage_bytes(
    uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int bytes, int tid, int nthreads)
{
    if (((reinterpret_cast<size_t>(src) & 15) == 0) && ((bytes & 15) == 0))
    {
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (int i = tid; i < (bytes >> 4); i += nthreads) d4[i] = s4[i];
    }
    else if (((reinterpret_cast<size_t>(src) & 3) == 0) && ((bytes & 3) == 0))
    {
        const uint32_t* s1 = reinterpret_cast<const uint32_t*>(src);
        uint32_t* d1 = reinterpret_cast<uint32_t*>(dst);
        for (int i = tid; i < (bytes >> 2); i += nthreads) d1[i] = s1[i];
    }
    else
    {
        for (int i = tid; i < bytes; i += nthreads) dst[i] = src[i];
    }
}

extern "C" __global__ void __launch_bounds__(Q8S_GROUP * Q8S_ROWS) q8_0_gemv_f32in_staged(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n, const int k)
{
    __shared__ float xs[Q8S_GROUP * Q8S_XPAD];                                  // 17408 B
    __shared__ __align__(16) uint8_t wsm[Q8S_ROWS][Q8S_GROUP * Q8S_BLOCK_BYTES]; // 17408 B
    __shared__ float red[Q8S_ROWS][Q8S_GROUP / 32];

    const int g = threadIdx.x / Q8S_GROUP;   // row slot within this thread block
    const int t = threadIdx.x % Q8S_GROUP;   // == the original kernel's threadIdx.x
    const int row = blockIdx.x * Q8S_ROWS + g;
    const bool active = row < n;             // inactive groups still hit every __syncthreads
    const int bpr = k / 32;
    const uint8_t* w_row = weight + (size_t)(active ? row : 0) * bpr * Q8S_BLOCK_BYTES;
    uint8_t* my_w = wsm[g];

    float acc = 0.0f;
    for (int c0 = 0; c0 < bpr; c0 += Q8S_GROUP)
    {
        const int nb = min(Q8S_GROUP, bpr - c0);   // blocks in this chunk
        __syncthreads();                           // previous chunk's readers are done
        if (active)
            q8s_stage_bytes(my_w, w_row + (size_t)c0 * Q8S_BLOCK_BYTES, nb * Q8S_BLOCK_BYTES, t, Q8S_GROUP);

        const bool owns = active && t < nb;        // original: thread t has block c0 + t
        float s = 0.0f;
        #pragma unroll
        for (int h = 0; h < 32 / Q8S_HALF; h++)
        {
            if (h > 0) __syncthreads();            // readers of the previous half are done
            for (int i = threadIdx.x; i < nb * Q8S_HALF; i += blockDim.x)
            {
                const int bb = i / Q8S_HALF, jj = i % Q8S_HALF;
                xs[bb * Q8S_XPAD + jj] = x[(size_t)(c0 + bb) * 32 + h * Q8S_HALF + jj];
            }
            __syncthreads();
            if (owns)
            {
                const int8_t* qs = reinterpret_cast<const int8_t*>(my_w + t * Q8S_BLOCK_BYTES + 2 + h * Q8S_HALF);
                const float* xr = xs + t * Q8S_XPAD;
                #pragma unroll
                for (int jj = 0; jj < Q8S_HALF; jj++)
                    s = __fmaf_rn((float)qs[jj], xr[jj], s);
            }
        }
        if (owns)
        {
            const float d = __half2float(*reinterpret_cast<const half*>(my_w + t * Q8S_BLOCK_BYTES));
            acc = __fmaf_rn(d, s, acc);
        }
    }

    // Reduction — identical to quantized_gemv_q8_0_f32in with blockDim.x == 256.
    for (int off = 16; off > 0; off >>= 1)
        acc = __fadd_rn(acc, __shfl_down_sync(0xFFFFFFFF, acc, off));
    const int lane = t % 32, wid = t / 32;
    if (lane == 0) red[g][wid] = acc;
    __syncthreads();
    if (wid == 0)
    {
        acc = (lane < Q8S_GROUP / 32) ? red[g][lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            acc = __fadd_rn(acc, __shfl_down_sync(0xFFFFFFFF, acc, off));
    }
    if (t == 0 && active) y[row] = acc;
}
