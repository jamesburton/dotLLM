// Q8_0 GEMV with FP32 input — register-blocked, warp-independent, BIT-IDENTICAL twin of
// quantized_gemv_q8_0_f32in (quantized_gemv_f32in.cu) and of q8_0_gemv_f32in_staged. Issue #486.
//
//   y[c][n] = W_q8_0[n,k] @ x[c][k]   for c in [0, ncols), output FP32.
//
// Two entry points:
//   q8_0_gemv_f32in_rb        one input column   (the MTP draft step's projections)
//   q8_0_gemv_f32in_rb_multi  up to 8 columns    (the MTP batched absorb: eh_proj / K / V over S rows)
// Every column of the multi kernel is bit-identical to the single-column kernel, which is
// bit-identical to the original — the weights are simply read once for all columns.
//
// Why a third Q8_0 GEMV: the #482 staged kernel reaches 109-145 GB/s (RTX 3060, 360 peak). Per
// 256-block chunk it runs 4 __syncthreads phases (stage W, stage x half, compute, repeat), so a
// block never has loads in flight while it computes, and each x stage is shared by only 2 rows —
// x traffic from L2 is ~2x the weight traffic for k = 5120. Its 512 threads x 64 registers also
// cap it at 2 blocks/SM, and for k = 5120 (bpr = 160) 96 of every 256 threads own no block.
//
// What changes here — data flow only:
//   * Each thread block computes Q8R_ROWS = 4 rows; each thread holds its block's 32 int8 weights
//     for all 4 rows in registers, so one staged x block feeds 4 rows (x traffic / 4).
//   * Warps are independent: a warp stages its own 32-block segment (coalesced 16-byte loads of
//     all 4 rows at once -> ~4.4 KB in flight per warp) and its x slice through a warp-private
//     shared buffer with __syncwarp only. The single __syncthreads is in the final reduction.
//   * Only warps that own at least one block are launched: blockDim = 32 * min(8, ceil(bpr/32)).
//
// What does NOT change — the arithmetic, operation for operation (the proof of bit-identity):
//   * Logical thread t = 32 * warp + lane of a row's 256-thread group owns blocks t, t+256, ...
//     in increasing order — chunk c0's block c0 + t is exactly lane `lane` of warp `warp`'s
//     segment [c0 + 32*warp, c0 + 32*warp + 32).
//   * Per block: s = 0; s = fma((float)q_j, x_j, s) for j = 0..31 in order; acc = fma(d, s, acc).
//     Explicit __fmaf_rn (the original's PTX is exactly these fma.rn.f32 under nvcc's default
//     -fmad=true), so the result does not depend on -fmad.
//   * Reduction: the same 5-step __shfl_down tree per 32-lane warp (lanes owning no block
//     contribute +0.0, as in the original), then the same 8-slot second-stage tree. Warps the
//     original launches but that own no block would contribute a partial of exactly +0.0f; they
//     are not launched here and their slots read as +0.0f — the same bits.
//
// Requirements (checked by the host wrapper, which falls back otherwise): k % 32 == 0; x 16-byte
// aligned; ldx % 4 == 0 (so every column is 16-byte aligned). Weight rows need no alignment: the
// staging uses 16-byte loads when the base and the row stride (bpr * 34) are 16-byte aligned and
// 2-byte loads otherwise (a Q8_0 row always starts 2-byte aligned).
//
// Launch: grid = ceil(n / Q8R_ROWS), block = 32 * min(8, ceil((k/32) / 32)), no dynamic smem.
//
// Build (CUDA 12.8, default flags, compute_75 like the rest of the tree):
//   nvcc -ptx -arch=compute_75 -o native/ptx/q8_0_gemv_f32in_rb.ptx native/kernels/q8_0_gemv_f32in_rb.cu

#include <cuda_fp16.h>
#include <stdint.h>

#define Q8R_GROUP 256                                   // logical threads per row (original blockDim)
#define Q8R_WARPS (Q8R_GROUP / 32)                      // 8
#define Q8R_ROWS 4                                      // rows per thread block (register-blocked)
#define Q8R_BLOCK_BYTES 34                              // Q8_0 block: f16 scale + 32 int8
#define Q8R_SEG_BYTES (32 * Q8R_BLOCK_BYTES)            // one warp segment: 32 blocks = 1088 B
#define Q8R_SEG_STRIDE 1104                             // per-row slot: 1088 + slack for the 9-word read, 16-aligned
#define Q8R_XSTRIDE 36                                  // floats per staged x block: 32 + 4 pad -> conflict-free 16-B reads
#define Q8R_BUF_BYTES (32 * Q8R_XSTRIDE * 4)            // 4608 B per warp, >= Q8R_ROWS * Q8R_SEG_STRIDE (4416)
#define Q8R_MAX_COLS 8

static_assert(Q8R_ROWS * Q8R_SEG_STRIDE <= Q8R_BUF_BYTES, "weight slots must fit the warp buffer");
static_assert(Q8R_SEG_STRIDE % 16 == 0, "row slots must stay 16-byte aligned");

// Signed byte `b` of a little-endian word, as float (exact, like (float)(int8_t)q).
__device__ __forceinline__ float q8r_byte(uint32_t w, int b)
{
    return __int2float_rn(((int)(w << (24 - 8 * b))) >> 24);
}

template <int NCOLS>
__device__ __forceinline__ void q8r_body(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ x, const int ldx,
    float* __restrict__ y, const int ldy,
    const int n, const int k, const int ncols,
    uint8_t (* __restrict__ buf)[Q8R_BUF_BYTES],
    float (* __restrict__ red)[Q8R_WARPS])
{
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;           // == original t / 32
    const int nwarps = blockDim.x >> 5;         // launched (block-owning) logical warps
    const int bpr = k >> 5;
    const size_t row_bytes = (size_t)bpr * Q8R_BLOCK_BYTES;
    const int row0 = blockIdx.x * Q8R_ROWS;
    uint8_t* wb = buf[wid];
    float* xbuf = reinterpret_cast<float*>(wb);
    // Block-uniform staging mode: 16-byte loads need every segment start 16-byte aligned. A segment
    // starts at row * row_bytes + (a multiple of 1088 = 16 * 68), so base + row stride decide it.
    const bool vec16 = ((reinterpret_cast<size_t>(weight) & 15) == 0) && ((row_bytes & 15) == 0);

    float acc[Q8R_ROWS][NCOLS];
    #pragma unroll
    for (int r = 0; r < Q8R_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < NCOLS; c++)
            acc[r][c] = 0.0f;

    for (int c0 = 0; c0 < bpr; c0 += Q8R_GROUP)
    {
        const int seg0 = c0 + wid * 32;             // this warp's first block in this chunk
        if (seg0 >= bpr) break;                     // warp-uniform; later chunks start further on
        const int nblk = min(32, bpr - seg0);       // blocks in the segment (lanes that own one)
        const int seg_bytes = nblk * Q8R_BLOCK_BYTES;

        __syncwarp();                               // previous chunk's x readers of wb are done

        // ── 1. Stage the 4 rows' segments. Rows past n re-read row n-1 (valid memory) and are
        //       never stored. All loads are issued before any store, so ~4.4 KB is in flight.
        if (vec16)
        {
            const int nwords = seg_bytes >> 4;      // whole 16-byte words (68 for a full segment)
            const int tail = seg_bytes & 15;        // 0..14 bytes, always even
            uint4 v[Q8R_ROWS][3];
            unsigned short tv[Q8R_ROWS];
            #pragma unroll
            for (int r = 0; r < Q8R_ROWS; r++)
            {
                const int row = min(row0 + r, n - 1);
                const uint8_t* src = weight + (size_t)row * row_bytes + (size_t)seg0 * Q8R_BLOCK_BYTES;
                const uint4* s4 = reinterpret_cast<const uint4*>(src);
                #pragma unroll
                for (int q = 0; q < 3; q++)
                    if (lane + 32 * q < nwords) v[r][q] = __ldg(s4 + lane + 32 * q);
                if (lane < (tail >> 1))
                    tv[r] = __ldg(reinterpret_cast<const unsigned short*>(src + (nwords << 4)) + lane);
            }
            #pragma unroll
            for (int r = 0; r < Q8R_ROWS; r++)
            {
                uint8_t* dst = wb + r * Q8R_SEG_STRIDE;
                uint4* d4 = reinterpret_cast<uint4*>(dst);
                #pragma unroll
                for (int q = 0; q < 3; q++)
                    if (lane + 32 * q < nwords) d4[lane + 32 * q] = v[r][q];
                if (lane < (tail >> 1))
                    reinterpret_cast<unsigned short*>(dst + (nwords << 4))[lane] = tv[r];
            }
        }
        else
        {
            const int nhalf = seg_bytes >> 1;       // 2-byte words (544 for a full segment)
            #pragma unroll
            for (int r = 0; r < Q8R_ROWS; r++)
            {
                const int row = min(row0 + r, n - 1);
                const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
                    weight + (size_t)row * row_bytes + (size_t)seg0 * Q8R_BLOCK_BYTES);
                unsigned short* d2 = reinterpret_cast<unsigned short*>(wb + r * Q8R_SEG_STRIDE);
                for (int i = lane; i < nhalf; i += 32) d2[i] = __ldg(s2 + i);
            }
        }
        __syncwarp();

        // ── 2. Pull this lane's block (block seg0 + lane) of every row into registers. Block
        //       `lane` starts at byte 34*lane: word-aligned for an even lane (d = low half of
        //       word 0, qs start 2 bytes in) and 2 bytes past a word for an odd lane (d = high half
        //       of word 0, qs start exactly at word 1). Nine aligned words cover it either way, and
        //       a funnel shift by 16 (even) or 32 (odd, clamped = the high word) realigns qs.
        const bool owns = lane < nblk;
        uint32_t qw[Q8R_ROWS][8];
        float dsc[Q8R_ROWS];
        if (owns)
        {
            const int off = lane * Q8R_BLOCK_BYTES;
            const bool odd = (off & 2) != 0;
            const uint32_t sh = odd ? 32u : 16u;
            #pragma unroll
            for (int r = 0; r < Q8R_ROWS; r++)
            {
                const uint32_t* wp = reinterpret_cast<const uint32_t*>(wb + r * Q8R_SEG_STRIDE) + (off >> 2);
                uint32_t wv[9];
                #pragma unroll
                for (int i = 0; i < 9; i++) wv[i] = wp[i];
                const unsigned short dbits = (unsigned short)(odd ? (wv[0] >> 16) : (wv[0] & 0xFFFFu));
                dsc[r] = __half2float(__ushort_as_half(dbits));
                #pragma unroll
                for (int i = 0; i < 8; i++) qw[r][i] = __funnelshift_rc(wv[i], wv[i + 1], sh);
            }
        }
        __syncwarp();                               // wb is reused for x below

        // ── 3. Per column: stage x's slice for the segment (coalesced), accumulate all 4 rows. ──
        #pragma unroll
        for (int c = 0; c < NCOLS; c++)
        {
            if (c < ncols)                          // uniform
            {
                const float4* xs4 = reinterpret_cast<const float4*>(x + (size_t)c * ldx + (size_t)seg0 * 32);
                for (int i = lane; i < nblk * 8; i += 32)   // block i >> 3, quad i & 7
                    *reinterpret_cast<float4*>(xbuf + (i >> 3) * Q8R_XSTRIDE + ((i & 7) << 2)) = __ldg(xs4 + i);
                __syncwarp();
                if (owns)
                {
                    float s[Q8R_ROWS];
                    #pragma unroll
                    for (int r = 0; r < Q8R_ROWS; r++) s[r] = 0.0f;
                    const float* xr = xbuf + lane * Q8R_XSTRIDE;
                    #pragma unroll
                    for (int q = 0; q < 8; q++)     // elements 4q..4q+3, in order
                    {
                        const float4 xv = *reinterpret_cast<const float4*>(xr + 4 * q);
                        #pragma unroll
                        for (int r = 0; r < Q8R_ROWS; r++)
                        {
                            const uint32_t wq = qw[r][q];
                            s[r] = __fmaf_rn(q8r_byte(wq, 0), xv.x, s[r]);
                            s[r] = __fmaf_rn(q8r_byte(wq, 1), xv.y, s[r]);
                            s[r] = __fmaf_rn(q8r_byte(wq, 2), xv.z, s[r]);
                            s[r] = __fmaf_rn(q8r_byte(wq, 3), xv.w, s[r]);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < Q8R_ROWS; r++) acc[r][c] = __fmaf_rn(dsc[r], s[r], acc[r][c]);
                }
                __syncwarp();                       // readers done before the next column / chunk
            }
        }
    }

    // ── 4. Reduction: per (row, column), the original's warp tree ...
    #pragma unroll
    for (int r = 0; r < Q8R_ROWS; r++)
    {
        #pragma unroll
        for (int c = 0; c < NCOLS; c++)
        {
            if (c < ncols)
            {
                float v = acc[r][c];
                for (int o = 16; o > 0; o >>= 1)
                    v = __fadd_rn(v, __shfl_down_sync(0xFFFFFFFF, v, o));
                if (lane == 0) red[r * NCOLS + c][wid] = v;
            }
        }
    }
    __syncthreads();
    // ... then its 8-slot second stage, one (row, column) per warp at a time. Slots of logical
    // warps that were not launched (they own no block) are the original's +0.0f partials.
    for (int idx = wid; idx < Q8R_ROWS * NCOLS; idx += nwarps)
    {
        const int r = idx / NCOLS, c = idx % NCOLS;
        const int row = row0 + r;
        if (c >= ncols || row >= n) continue;       // warp-uniform
        float v = (lane < nwarps) ? red[idx][lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            v = __fadd_rn(v, __shfl_down_sync(0xFFFFFFFF, v, o));
        if (lane == 0) y[(size_t)c * ldy + row] = v;
    }
}

extern "C" __global__ void __launch_bounds__(Q8R_GROUP) q8_0_gemv_f32in_rb(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n, const int k)
{
    __shared__ __align__(16) uint8_t buf[Q8R_WARPS][Q8R_BUF_BYTES];     // 36864 B
    __shared__ float red[Q8R_ROWS * 1][Q8R_WARPS];
    q8r_body<1>(weight, x, k, y, n, n, k, 1, buf, red);
}

extern "C" __global__ void __launch_bounds__(Q8R_GROUP) q8_0_gemv_f32in_rb_multi(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ x, const int ldx,
    float* __restrict__ y, const int ldy,
    const int n, const int k, const int ncols)
{
    __shared__ __align__(16) uint8_t buf[Q8R_WARPS][Q8R_BUF_BYTES];     // 36864 B
    __shared__ float red[Q8R_ROWS * Q8R_MAX_COLS][Q8R_WARPS];           // 1024 B
    q8r_body<Q8R_MAX_COLS>(weight, x, ldx, y, ldy, n, k, ncols, buf, red);
}
