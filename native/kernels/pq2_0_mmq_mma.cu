// Issue #1222 (#1132) — PQ2_0 PREFILL GEMM on int8 tensor cores: y[s, row] = W[row, :] . x[s, :] for
// the S > 8 token columns of a prefill, the weights read PACKED from the split layout and decoded in
// registers straight into mma.sync A fragments. No F16 weight copy (rule 2, #495).
//
// ───────────────────────── Same contract and numerics as pq2_0_mmq_dp4a.cu ─────────────────────────
// Inputs are exactly #490's: activations from pq2_0_dp4a_quantize_x (int8 [S, k] PERMUTED within each
// 16-element chunk, int2 { bits(d), sum q } per 32), weights in the split PQ2_0 layout (fp16 scales,
// then 32 code bytes per 128-group). Read pq2_0_gemv_dp4a.cu's header for the layout tricks.
//
// The tensor-core mapping rests on one fact: a dot product does not care how K is ordered, as long as
// both operands use the same order. Take the 8 words of a 32-element block in the permuted order
// (word j: chunk j/4, bit plane j%4). Weight word j is (codeword_{j/4} >> 2*(j%4)) & 0x03030303 and
// activation word j is the j-th permuted int8 word — #485's pairing. Feeding word j as logical
// K bytes 4j..4j+3 of mma.m16n8k32's A (row-major) and B (col-major) fragments therefore gives the
// exact int32 sum of code*q for the block. As in the dp4a kernels, codes stay 0..3 and the "-1"
// becomes "- sum q" per (column, block).
//
// Per output element the float work is then IDENTICAL to pq2_0_mmq_dp4a's, in the same order:
// for each group g ascending, each 32-block b ascending,
//     acc = fmaf(i2f(isum - sum_q), d * ws, acc)
// with the same magic-number i2f. So this kernel is bit-identical to the dp4a MMQ (the existing
// #490 tests are its oracle), only faster: 32 MACs per mma lane-op instead of 4 per dp4a.
//
// ───────────────────────── Tiling ─────────────────────────
// Block = 256 threads = 8 warps arranged 4 (rows) x 2 (columns); warp tile 32 rows x 32 columns =
// 2 x 4 mma tiles of 16x8. Block tile BM = 128 rows x BN = 64 columns. One 128-element PQ2_0 group
// is staged per step (4 mma K-steps). The next group's global loads are issued into registers
// before the current group is computed (register double buffering), so one smem buffer suffices.
// Grid: x = column tile, y = row tile — sibling column tiles of one weight row tile run together and
// share the weight stream through L2 (see pq2_0_mmq_dp4a.cu, "Grid order").
//
// Shared layouts (bank-conflict free for the fragment reads):
//   wsm [BM][10] u32 : row r's 8 code words at r*10 .. r*10+7 (stride 10 spreads the 8 gids over
//                      distinct bank pairs; the 4 tig lanes of a gid read the same uint2 = broadcast).
//   xsm [BN][36] u32 : column c's 32 permuted words at c*36 (stride 36: gid*4 + tig is a permutation
//                      of the 32 banks).
//
// Tails: rows clamp to n-1 (never stored); dead columns stage zeros with d = 0, sum = 0.
// Contract: k % 128 == 0, xq / xmeta 16-byte aligned. mma.sync .s8 needs sm_80+: built at
// compute_80 (ARCH_86 list) and gated by the host on the module loading.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2T_GROUP    128
#define PQ2T_QBLOCK   32
#define PQ2T_BPG      (PQ2T_GROUP / PQ2T_QBLOCK)   // 4 activation blocks per group
#define PQ2T_BM       128
#define PQ2T_BN       64
#define PQ2T_THREADS  256
#define PQ2T_WSTRIDE  10                           // u32 per staged weight row
#define PQ2T_XSTRIDE  36                           // u32 per staged activation column

__device__ __forceinline__ size_t pq2t_codes_base_offset(long long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

// Exact for |v| < 2^22 (block dots are <= 16256); identical to pq2_0_mmq_dp4a.cu's pq2m_i2f.
__device__ __forceinline__ float pq2t_i2f(int v)
{
    return __int_as_float(0x4B400000 + v) - 12582912.0f;
}

__device__ __forceinline__ void pq2t_mma(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(PQ2T_THREADS, 2) pq2_0_mmq_mma_f32y(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq,
    const int2*    __restrict__ xmeta,
    float*         __restrict__ y,
    const int n,
    const int k,
    const int columns)
{
    __shared__ __align__(16) unsigned wsm[PQ2T_BM * PQ2T_WSTRIDE];
    __shared__ __align__(16) unsigned xsm[PQ2T_BN * PQ2T_XSTRIDE];
    __shared__ float wscale[PQ2T_BM];
    __shared__ float xd[PQ2T_BN][PQ2T_BPG];
    __shared__ int   xs[PQ2T_BN][PQ2T_BPG];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    const int wm   = warp & 3;    // warp's 32-row slice
    const int wn   = warp >> 2;   // warp's 32-column slice

    const int rowBase = blockIdx.y * PQ2T_BM;
    const int colBase = blockIdx.x * PQ2T_BN;

    const int gpr = k / PQ2T_GROUP;
    const int bpr = k / PQ2T_QBLOCK;
    const long long totalGroups = (long long)n * gpr;
    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2t_codes_base_offset(totalGroups);

    // Fixed per-thread staging assignments (one or two 16-byte loads per thread per group).
    const int wRow  = tid >> 1, wHalf = tid & 1;                      // weights: 128 rows x 2 halves
    const int wRowG = min(rowBase + wRow, n - 1);
    const int sRowG = min(rowBase + (tid & (PQ2T_BM - 1)), n - 1);    // scales: threads 0..127
    int  xCol[2], xWord[2];
    bool xLive[2];
    #pragma unroll
    for (int j = 0; j < 2; j++)
    {
        const int i = tid + j * PQ2T_THREADS;                          // activations: 64 cols x 8 uint4
        xCol[j] = i >> 3; xWord[j] = i & 7;
        xLive[j] = colBase + xCol[j] < columns;
    }
    const int mCol = tid >> 2, mBlk = tid & 3;                        // metadata: 64 cols x 4 blocks
    const bool mLive = colBase + mCol < columns;

    uint4 nw, nx[2];
    float ns = 0.0f;
    int2  nm;
    auto fetch = [&](int g)
    {
        nw = __ldg(reinterpret_cast<const uint4*>(codesBase + ((size_t)wRowG * gpr + g) * 32 + 16 * wHalf));
        if (tid < PQ2T_BM) ns = __half2float(scales[(size_t)sRowG * gpr + g]);
        #pragma unroll
        for (int j = 0; j < 2; j++)
            nx[j] = xLive[j]
                ? __ldg(reinterpret_cast<const uint4*>(
                      xq + (size_t)(colBase + xCol[j]) * k + (size_t)g * PQ2T_GROUP + 16 * xWord[j]))
                : make_uint4(0u, 0u, 0u, 0u);
        nm = mLive ? __ldg(xmeta + (size_t)(colBase + mCol) * bpr + (size_t)g * PQ2T_BPG + mBlk)
                   : make_int2(0, 0);
    };

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int e = 0; e < 4; e++)
                acc[mi][ni][e] = 0.0f;

    fetch(0);
    for (int g = 0; g < gpr; g++)
    {
        // ── publish the prefetched group ──
        *reinterpret_cast<uint2*>(&wsm[wRow * PQ2T_WSTRIDE + 4 * wHalf])     = make_uint2(nw.x, nw.y);
        *reinterpret_cast<uint2*>(&wsm[wRow * PQ2T_WSTRIDE + 4 * wHalf + 2]) = make_uint2(nw.z, nw.w);
        if (tid < PQ2T_BM) wscale[tid] = ns;
        #pragma unroll
        for (int j = 0; j < 2; j++)
            *reinterpret_cast<uint4*>(&xsm[xCol[j] * PQ2T_XSTRIDE + 4 * xWord[j]]) = nx[j];
        xd[mCol][mBlk] = __int_as_float(nm.x);
        xs[mCol][mBlk] = nm.y;
        __syncthreads();

        if (g + 1 < gpr) fetch(g + 1);   // in flight while this group computes

        float ws[2][2];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
        {
            ws[mi][0] = wscale[wm * 32 + mi * 16 + gid];
            ws[mi][1] = wscale[wm * 32 + mi * 16 + gid + 8];
        }

        #pragma unroll
        for (int b = 0; b < PQ2T_BPG; b++)
        {
            unsigned af[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
            {
                const int r0 = wm * 32 + mi * 16 + gid;
                const uint2 c0 = *reinterpret_cast<const uint2*>(&wsm[r0 * PQ2T_WSTRIDE + 2 * b]);
                const uint2 c1 = *reinterpret_cast<const uint2*>(&wsm[(r0 + 8) * PQ2T_WSTRIDE + 2 * b]);
                af[mi][0] = (c0.x >> (2 * tig)) & 0x03030303u;   // row gid,   word tig
                af[mi][1] = (c1.x >> (2 * tig)) & 0x03030303u;   // row gid+8, word tig
                af[mi][2] = (c0.y >> (2 * tig)) & 0x03030303u;   // row gid,   word 4+tig
                af[mi][3] = (c1.y >> (2 * tig)) & 0x03030303u;   // row gid+8, word 4+tig
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
            {
                const int cl = wn * 32 + ni * 8;
                const unsigned b0 = xsm[(cl + gid) * PQ2T_XSTRIDE + 8 * b + tig];
                const unsigned b1 = xsm[(cl + gid) * PQ2T_XSTRIDE + 8 * b + 4 + tig];
                const int   cA = cl + 2 * tig, cB = cA + 1;
                const float dA = xd[cA][b], dB = xd[cB][b];
                const int   sA = xs[cA][b], sB = xs[cB][b];

                #pragma unroll
                for (int mi = 0; mi < 2; mi++)
                {
                    int c[4] = {0, 0, 0, 0};
                    pq2t_mma(c, af[mi][0], af[mi][1], af[mi][2], af[mi][3], b0, b1);
                    acc[mi][ni][0] = fmaf(pq2t_i2f(c[0] - sA), dA * ws[mi][0], acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(pq2t_i2f(c[1] - sB), dB * ws[mi][0], acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(pq2t_i2f(c[2] - sA), dA * ws[mi][1], acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(pq2t_i2f(c[3] - sB), dB * ws[mi][1], acc[mi][ni][3]);
                }
            }
        }
        __syncthreads();   // everyone is done reading before the next group is published
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
    {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                const int row = rowBase + wm * 32 + mi * 16 + gid + (e >= 2 ? 8 : 0);
                const int col = colBase + wn * 32 + ni * 8 + 2 * tig + (e & 1);
                if (row < n && col < columns) y[(size_t)col * n + row] = acc[mi][ni][e];
            }
        }
    }
}
