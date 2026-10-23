// Issue #490 — PQ2_0 PREFILL GEMM (MMQ): y[s, row] = W[row, :] . x[s, :] for s < S, S > 8, with the
// weights read PACKED and decoded in registers. The tiled twin of #485's dp4a GEMV.
//
// ───────────────────────── What this replaces ─────────────────────────
// Every PQ2_0 projection wider than the GEMV's 8 columns used to dequantize the WHOLE weight matrix
// to F16 into scratch and hand it to cuBLAS HGEMM. On Bonsai 2 27B (RTX 3060) that is a fixed
// ~330 ms per forward — the reason prefill is 37.1 tok/s at p=16 but 150 at p=256 — plus a ceiling
// of ~150 tok/s once it amortises, because the F16 round trip moves 2 bytes per weight where the
// packed form moves 0.26.
//
// ───────────────────────── Shared contract with #485 ─────────────────────────
// This kernel consumes EXACTLY the activation layout pq2_0_dp4a_quantize_x produces, and decodes the
// weights exactly as pq2_0_gemv_dp4a.cu does. Read that file's header for the two load-bearing
// tricks; restated in one line each:
//   * activations: int8 [S, k] row-major, permuted inside every 16-element chunk (element 16c+4j+i at
//     byte 16c+4i+j), with an int2 { bits(half-rounded scale d), sum of the block's int8 } per 32;
//   * weights (SPLIT layout, pq2_0_repack.cu): all n*gpr fp16 group scales, then 32 code bytes per
//     128-element group; in a 32-bit code word covering 16 elements, (w >> 2i) & 0x03030303 is
//     exactly the permuted activation word i, and since codes 0..3 are valid signed int8,
//     sum (code-1)*q = dp4a-sum(code, q) - sum q, the "- sum q" seeded free into the dp4a chain.
// So the numerics are the CPU W2A8 tier's, identical to the decode path, and the activation
// quantizer is REUSED verbatim — no second quantizer to keep byte-exact.
//
// ───────────────────────── Tile choice, and why ─────────────────────────
// Let M = output rows (n), N = token columns (S), K the reduction. Per K-element a block tile reads
// BM/4 bytes of weight (2 bits/weight) and BN bytes of activation, so BM ~ 4*BN balances the two
// streams. The shipped tile is BM=128, BN=32, 256 threads, TM=TN=4 (16 outputs/thread):
//   * 32 B of weight and 32 B of activation per K-element — balanced, as above.
//   * 9.5 KB of shared memory, so occupancy is bounded by registers, not smem; __launch_bounds__
//     (256, 2) holds ptxas to <= 128 registers, i.e. 2 blocks/SM. (#486's q8_0_gemv_f32in_rb_multi
//     measured badly at 184 regs / 37.9 KB smem — this kernel deliberately leaves that headroom.)
//   * TN = 4 sets the decode overhead: a row's 8 code-plane words are decoded once (16 int ops) and
//     reused by TN columns, against TN*8 dp4a — 2/TN = 0.5 int ops of decode per dp4a. TN=2 would
//     double it; TN=8 needs 2*TN uint4 activation registers (64) and spills.
// A second instantiation, BM=256/BN=16/TM=8/TN=2, serves S <= 16, where a 32-wide tile would idle
// half its lanes; there the weight stream is read once either way, so the narrower tile only trades
// decode overhead for utilisation. Both are dispatched automatically (and can be forced from C#).
//
// ───────────────────────── Grid order (load-bearing) ─────────────────────────
// blockIdx.x is the COLUMN tile and blockIdx.y the ROW tile, so the ceil(S/BN) blocks that share a
// weight row tile are scheduled together and the weight stream is read from DRAM ~once and served to
// its siblings from L2. With the GEMV's row-major grid the weights would be re-streamed ceil(S/BN)
// times (~300 ms at S=512 on a 3060 — the very cost this kernel removes). For the same reason the
// weight loads use __ldg, NOT __ldcs: evict-first would defeat that L2 reuse.
//
// ───────────────────────── Per-thread inner loop ─────────────────────────
// Per (row, column, 32-element block): 8 dp4a, plus one integer->float conversion of the exact int32
// block dot, one FMUL by d*ws and one FFMA. The conversion uses the magic-number identity
//   __int_as_float(0x4B400000 + isum) - 12582912.0f   (exact for |isum| < 2^22; here |isum| <= 16256)
// because I2F is quarter rate on sm_86 while IADD/FADD are not; #define PQ2M_MAGIC_I2F 0 reverts to
// __int2float_rn for a one-line numerics A/B (the two agree bit for bit over this range).
//
// ───────────────────────── Tails ─────────────────────────
// n % BM and S % BN are both allowed. Row tails CLAMP their loads to row n-1 and mask the store;
// column tails zero-fill BOTH the staged int8 and the staged metadata (d = 0, sum = 0), so a dead
// column contributes exactly zero and only the store mask matters. There is no early `return`
// anywhere: every thread must reach every __syncthreads.
//
// Contract: k % 128 == 0; xq/xmeta 16-byte aligned (cuMemAlloc'd scratch is 256-byte aligned).
//
// Build (default flags — no --use_fast_math; nvcc -ptx stops before ptxas, so the register report is
// a second command):
//   nvcc -ptx -arch=compute_75 -o native/ptx/pq2_0_mmq_dp4a.ptx native/kernels/pq2_0_mmq_dp4a.cu
//   ptxas -v -arch=sm_86 native/ptx/pq2_0_mmq_dp4a.ptx -o NUL
// dp4a needs sm_61+; compute_75 is the repo floor.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2M_GROUP             128                                  // PQ2_0 group (one fp16 scale)
#define PQ2M_QBLOCK            32                                   // activation quant block
#define PQ2M_BLOCKS_PER_GROUP  (PQ2M_GROUP / PQ2M_QBLOCK)           // 4
#define PQ2M_THREADS           256
#define PQ2M_THREADS_M         32                                   // threads along the row axis
#define PQ2M_THREADS_N         (PQ2M_THREADS / PQ2M_THREADS_M)      // 8, along the column axis

// 1: magic-number int->float (IADD + FADD, full rate). 0: __int2float_rn (quarter rate on sm_86).
// Bit-identical over |isum| <= 2^22, which bounds every block dot here (3*127*32 + 127*32 = 16256).
#ifndef PQ2M_MAGIC_I2F
#define PQ2M_MAGIC_I2F 1
#endif

// Must match pq2_0_gemv_dp4a.cu / pq2_0_repack.cu / dequant_pq2_0.cu's helper of the same shape.
__device__ __forceinline__ size_t pq2m_codes_base_offset(long long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

__device__ __forceinline__ float pq2m_i2f(int v)
{
#if PQ2M_MAGIC_I2F
    return __int_as_float(0x4B400000 + v) - 12582912.0f;
#else
    return __int2float_rn(v);
#endif
}

// BN: token columns per block tile. TN: columns per thread. TM: rows per thread (BM = 32*TM).
template <int BN, int TN, int TM>
__device__ __forceinline__ void pq2_0_mmq_dp4a_body(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq,
    const int2*    __restrict__ xmeta,
    float*         __restrict__ y,
    const int n,
    const int k,
    const int columns)
{
    constexpr int BM = PQ2M_THREADS_M * TM;

    // wsm is TRANSPOSED to [block][row] so a warp's 32 lanes — which own 32 consecutive rows — read
    // 32 consecutive 8-byte elements: conflict-free. [row][block] would make every lane hit the same
    // bank. xsm is read warp-uniformly (a warp shares one column group), i.e. broadcast.
    __shared__ uint2 wsm[PQ2M_BLOCKS_PER_GROUP][BM];
    __shared__ float wscale[BM];
    __shared__ uint4 xsm[BN][PQ2M_BLOCKS_PER_GROUP * 2];
    __shared__ float xd[BN][PQ2M_BLOCKS_PER_GROUP];
    __shared__ int   xs[BN][PQ2M_BLOCKS_PER_GROUP];

    const int tid  = threadIdx.x;
    const int tIdM = tid & (PQ2M_THREADS_M - 1);
    const int tIdN = tid / PQ2M_THREADS_M;

    const int rowBase  = blockIdx.y * BM;
    const int colBase  = blockIdx.x * BN;
    const int colLocal = tIdN * TN;

    const int gpr = k / PQ2M_GROUP;    // groups per weight row
    const int bpr = k / PQ2M_QBLOCK;   // activation blocks per column
    const long long totalGroups = (long long)n * gpr;
    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2m_codes_base_offset(totalGroups);

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int c = 0; c < TN; c++)
            acc[r][c] = 0.0f;

    for (int g = 0; g < gpr; g++)
    {
        // ── stage one 128-element group ──
        // Weight codes: BM rows x 32 bytes, as 16-byte chunks (two 32-element blocks each).
        for (int i = tid; i < BM * 2; i += PQ2M_THREADS)
        {
            const int r = i >> 1, h = i & 1;
            const int rowG = min(rowBase + r, n - 1);            // clamped: tail rows are never stored
            const uint4 cw = __ldg(reinterpret_cast<const uint4*>(
                codesBase + ((size_t)rowG * gpr + g) * 32 + (size_t)16 * h));
            wsm[2 * h][r]     = make_uint2(cw.x, cw.y);
            wsm[2 * h + 1][r] = make_uint2(cw.z, cw.w);
        }
        for (int i = tid; i < BM; i += PQ2M_THREADS)
        {
            const int rowG = min(rowBase + i, n - 1);
            wscale[i] = __half2float(scales[(size_t)rowG * gpr + g]);
        }
        // Activations: BN columns x 128 int8, as 16-byte chunks. Dead columns stage zeros.
        for (int i = tid; i < BN * 8; i += PQ2M_THREADS)
        {
            const int c = i >> 3, w = i & 7;
            const int colG = colBase + c;
            xsm[c][w] = colG < columns
                ? __ldg(reinterpret_cast<const uint4*>(
                      xq + (size_t)colG * k + (size_t)g * PQ2M_GROUP + (size_t)16 * w))
                : make_uint4(0u, 0u, 0u, 0u);
        }
        // Metadata: BN columns x 4 blocks. d = 0 and sum = 0 make a dead column contribute exactly 0.
        for (int i = tid; i < BN * PQ2M_BLOCKS_PER_GROUP; i += PQ2M_THREADS)
        {
            const int c = i >> 2, b = i & (PQ2M_BLOCKS_PER_GROUP - 1);
            const int colG = colBase + c;
            const int2 md = colG < columns
                ? __ldg(xmeta + (size_t)colG * bpr + (size_t)g * PQ2M_BLOCKS_PER_GROUP + b)
                : make_int2(0, 0);
            xd[c][b] = __int_as_float(md.x);
            xs[c][b] = md.y;
        }
        __syncthreads();

        // ── compute the group ──
        float ws[TM];
        #pragma unroll
        for (int r = 0; r < TM; r++)
            ws[r] = wscale[tIdM + PQ2M_THREADS_M * r];

        #pragma unroll
        for (int b = 0; b < PQ2M_BLOCKS_PER_GROUP; b++)
        {
            uint4 a0[TN], a1[TN];
            float d[TN];
            int   xsum[TN];
            #pragma unroll
            for (int c = 0; c < TN; c++)
            {
                a0[c]   = xsm[colLocal + c][2 * b];
                a1[c]   = xsm[colLocal + c][2 * b + 1];
                d[c]    = xd[colLocal + c][b];
                xsum[c] = xs[colLocal + c][b];
            }

            #pragma unroll
            for (int r = 0; r < TM; r++)
            {
                // 8 code bytes = the 32 codes of this row's block; decoded once for all TN columns.
                const uint2 cw = wsm[b][tIdM + PQ2M_THREADS_M * r];
                int wp[8];
                #pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    wp[i]     = (int)((cw.x >> (2 * i)) & 0x03030303u);
                    wp[4 + i] = (int)((cw.y >> (2 * i)) & 0x03030303u);
                }

                #pragma unroll
                for (int c = 0; c < TN; c++)
                {
                    int isum = -xsum[c];   // sum (code-1)*q = sum code*q - sum q
                    isum = __dp4a(wp[0], (int)a0[c].x, isum);
                    isum = __dp4a(wp[1], (int)a0[c].y, isum);
                    isum = __dp4a(wp[2], (int)a0[c].z, isum);
                    isum = __dp4a(wp[3], (int)a0[c].w, isum);
                    isum = __dp4a(wp[4], (int)a1[c].x, isum);
                    isum = __dp4a(wp[5], (int)a1[c].y, isum);
                    isum = __dp4a(wp[6], (int)a1[c].z, isum);
                    isum = __dp4a(wp[7], (int)a1[c].w, isum);
                    acc[r][c] = fmaf(pq2m_i2f(isum), d[c] * ws[r], acc[r][c]);
                }
            }
        }
        __syncthreads();   // before the next iteration overwrites the staged group
    }

    #pragma unroll
    for (int r = 0; r < TM; r++)
    {
        const int row = rowBase + tIdM + PQ2M_THREADS_M * r;
        if (row >= n) continue;
        #pragma unroll
        for (int c = 0; c < TN; c++)
        {
            const int col = colBase + colLocal + c;
            if (col < columns) y[(size_t)col * n + row] = acc[r][c];
        }
    }
}

#define PQ2M_DEFINE_ENTRY(NAME_, BN_, TN_, TM_)                                        \
    extern "C" __global__ void __launch_bounds__(PQ2M_THREADS, 2) NAME_(               \
        const uint8_t* __restrict__ weight,                                            \
        const int8_t*  __restrict__ xq,                                                \
        const int2*    __restrict__ xmeta,                                             \
        float*         __restrict__ y,                                                 \
        const int n,                                                                   \
        const int k,                                                                   \
        const int columns)                                                             \
    {                                                                                  \
        pq2_0_mmq_dp4a_body<BN_, TN_, TM_>(weight, xq, xmeta, y, n, k, columns);        \
    }

// BM = 128, BN = 32 — the default tile (balanced weight/activation streams).
PQ2M_DEFINE_ENTRY(pq2_0_mmq_dp4a_f32y_bn32, 32, 4, 4)
// BM = 256, BN = 16 — for S <= 16, where a 32-wide tile idles half its lanes.
PQ2M_DEFINE_ENTRY(pq2_0_mmq_dp4a_f32y_bn16, 16, 2, 8)
