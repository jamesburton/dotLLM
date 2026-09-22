// Issue #485 — PQ2_0 GEMV v2: int8 activations + dp4a (W2A8), y[s, row] = W[row, :] . x[s, :] for
// s < S, S = 1..8.
//
// #482's multi-column kernel (pq2_0_gemv_multi.cu) reads each packed weight byte once for all S
// columns, but spends ~1 FP32 FMA per weight per column plus the half->float conversion, so on an
// RTX 3060 it turns ALU-bound from S ~= 3 (Bonsai 2 27B forward: S=1 57.7 ms, S=4 90.7, S=8 157.5)
// and even S=1 sits at ~35-40% of the 360 GB/s. This kernel moves the inner loop to integers:
// the activation rows are quantized to int8 ONCE per projection input (the quantizer kernel below)
// and each 4 weights x 4 activations become one __dp4a.
//
// ───────────────────────── Numerics: the CPU W2A8 tier, exactly ─────────────────────────
// The activation quantization is the CPU's Q8_0 activation quantizer (MatMul.QuantizeF32ToQ8_0,
// which the CPU W2A8 PQ2_0 GEMV — MatMul.GemvPQ2_0 on SSSE3/AVX2 hardware — applies to x):
//   per 32-element block:  amax  = max |x_i|
//                          scale = amax / 127                    (FP32, correctly rounded)
//                          q_i   = rne(x_i * (1 / scale))        (FP32 product, round-half-even)
//                                  clamped to [-127, 127]; all zero when scale == 0
//                          d     = float(half(scale))            (the dot uses the HALF-rounded scale:
//                                                                 the CPU stores (Half)scale and its
//                                                                 dot reads that back — ConvertQ8_0Scales)
// Group size 32 rather than PQ2_0's 128 because (a) it IS the CPU W2A8 block, so the CPU quantizer
// is a byte-exact oracle and the CPU W2A8 GEMV is a tight kernel oracle — and the Bonsai 2 greedy
// oracle is itself that W2A8 tier; (b) a per-32 max-abs bounds each block's rounding error by its
// own max, not by the max of 128 elements, so one activation outlier degrades 32 values, not 128;
// (c) it costs one extra 8-byte metadata load per 32 elements per column, and the per-block int32
// sum can never overflow (|sum c*q| <= 3*127*32 = 12192).
//
// ───────────────────────── Quantized activation layout (the contract) ─────────────────────────
//   xq   : int8, [S, k] row-major (column s at xq + s*k), but PERMUTED within every 16-element chunk:
//          chunk c's int32 word i (bytes 4i..4i+3) holds elements 16c+i, 16c+4+i, 16c+8+i, 16c+12+i,
//          i.e. element 16c + 4j + i sits at byte 16c + 4i + j (a 4x4 byte transpose — its own
//          inverse). This matches what a single mask pulls out of a PQ2_0 code word (see decode below),
//          so the weight decode needs no byte shuffles at all.
//   xmeta: int2-sized pairs, [S, k/32]: .x = bits of the float d (half-rounded scale), .y = int32
//          sum_i q_i over the block (needed by the offset identity below).
//
// ───────────────────────── Weight layout / decode ─────────────────────────
// The load-time SPLIT PQ2_0 layout (pq2_0_repack.cu): all n*gpr fp16 group scales, then (from
// pq2d_codes_base_offset) all code bytes, 32 per 128-element group. Byte b of a group holds elements
// 4b..4b+3 at bit offsets {0,2,4,6}; value = code - 1, code in {0,1,2,3} -> {-1,0,+1,+2} (code 3 is
// +2 — PQ2_0 is not strictly ternary; the CPU tests exercise it). So in a 32-bit code word covering
// 16 elements, element 4b+i sits at bit 8b+2i and
//     (word >> 2i) & 0x03030303   = bytes { code(i), code(4+i), code(8+i), code(12+i) }
// — exactly the permuted activation word i. Codes 0..3 are valid SIGNED int8, so
//     sum_e (code_e - 1) * q_e  =  dp4a-sum(code, q)  -  sum_e q_e
// is exact in int32, and the "- sum q" is free: the dp4a chain starts at -xsum. (Decoding to
// code-1 per byte instead would borrow across bytes on code 0, or cost ~3 extra ops per word.)
//
// ───────────────────────── Work split ─────────────────────────
// 256 threads = 8 warps; each warp owns PQ2D_ROWS_PER_WARP = 2 consecutive output rows (16 rows per
// block, same grid as #482). A LANE owns one whole 32-element activation block of one group:
//     lane = 4*gsub + qb  ->  group g = g0 + gsub (8 groups per warp step), block qb of it,
//                             code bytes 8qb..8qb+7 of the group = ONE 8-byte load per row,
// so per row a warp step reads 8 consecutive groups' codes = 256 contiguous bytes, and per column
// each lane reads its 32 int8 activations as two 16-byte loads (1 KB contiguous per warp) plus one
// 8-byte metadata word. Weight words are loaded with __ldcs (evict-first streaming), activations
// with __ldg (reused by the block's 8 warps through L1). No shared memory, no block barriers.
//
// Per (row, column, 32-element block): 8 dp4a + I2F + FMUL (d * weight scale) + FFMA. The weight
// decode (7 SHF/LOP3 per code word, 2 words per row) is shared by all S columns. If I2F (quarter
// rate on sm_86) shows up as the limiter, the exact magic-number conversion
//     __int_as_float(0x4B400000 + isum) - 12582912.0f      (valid for |isum| < 2^22; ours <= 16256)
// replaces it with IADD + FADD. Not done here first, to keep the kernel plainly correct.
//
// Float arithmetic: each lane accumulates acc += float(isum) * (d * ws) over its blocks in
// ascending g, then the warp reduces with a fixed shuffle tree. The integer part is exact, so the
// result differs from the CPU W2A8 GEMV only by FP32 summation order.
//
// Exact widths: one instantiation per S (1..8), as #470/#482 measured dead-column variants losing
// ~40%. Contract: k % 128 == 0; xq 16-byte aligned (cuMemAlloc'd scratch is 256-byte aligned).
//
// Build: default flags (no --use_fast_math — the quantizer must round exactly like the CPU):
//   nvcc -ptx -arch=compute_75 -o native/ptx/pq2_0_gemv_dp4a.ptx native/kernels/pq2_0_gemv_dp4a.cu
// dp4a needs sm_61+; compute_75 is the repo floor.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2D_GROUP_SIZE      128
#define PQ2D_QBLOCK          32                                            // activation quant block
#define PQ2D_ROWS_PER_WARP   2
#define PQ2D_WARPS_PER_BLOCK 8
#define PQ2D_ROWS_PER_BLOCK  (PQ2D_WARPS_PER_BLOCK * PQ2D_ROWS_PER_WARP)   // 16; launcher mirrors it
#define PQ2D_GROUPS_PER_STEP 8                                             // groups a warp covers per step

// Must match pq2_0_gemv.cu / pq2_0_repack.cu / dequant_pq2_0.cu's helper of the same shape.
__device__ __forceinline__ size_t pq2d_codes_base_offset(long long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

__device__ __forceinline__ float pq2d_warp_reduce(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

// ═════════════════════════ Activation quantizer ═════════════════════════
// One thread per 32-element block (blocks never straddle columns since k % 32 == 0).
// x: F32 [S*k]; xq: permuted int8 [S*k]; xmeta: int2 [S*k/32] = { bits(d), sum q }.
extern "C" __global__ void __launch_bounds__(256) pq2_0_dp4a_quantize_x(
    const float* __restrict__ x,
    int8_t*      __restrict__ xq,
    int2*        __restrict__ xmeta,
    const int totalBlocks)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= totalBlocks) return;

    const float4* src = reinterpret_cast<const float4*>(x + (size_t)b * PQ2D_QBLOCK);
    float v[PQ2D_QBLOCK];
    #pragma unroll
    for (int i = 0; i < PQ2D_QBLOCK / 4; i++)
    {
        const float4 f = __ldg(src + i);
        v[4 * i + 0] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
    }

    // Same comparison the CPU scalar reference uses (max is exact, so order does not matter).
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < PQ2D_QBLOCK; i++)
    {
        const float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }

    // Explicit round-to-nearest divisions, so exactness does not hinge on -prec-div.
    const float scale = __fdiv_rn(amax, 127.0f);
    const float d = __half2float(__float2half_rn(scale));

    int q[PQ2D_QBLOCK];
    if (scale == 0.0f)
    {
        #pragma unroll
        for (int i = 0; i < PQ2D_QBLOCK; i++) q[i] = 0;
    }
    else
    {
        // The CPU rounds x * (1/scale) with the FLOAT scale (not the half-rounded d).
        const float inv = __fdiv_rn(1.0f, scale);
        #pragma unroll
        for (int i = 0; i < PQ2D_QBLOCK; i++)
            q[i] = min(127, max(-127, __float2int_rn(__fmul_rn(v[i], inv))));
    }

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < PQ2D_QBLOCK; i++) sum += q[i];

    // Permute: chunk c (16 elements), output word i = bytes { q[16c+i], q[16c+4+i], q[16c+8+i], q[16c+12+i] }.
    unsigned int w[PQ2D_QBLOCK / 4];
    #pragma unroll
    for (int c = 0; c < 2; c++)
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            w[4 * c + i] =  ((unsigned int)q[16 * c + i]      & 0xFFu)
                         | (((unsigned int)q[16 * c + 4 + i]  & 0xFFu) << 8)
                         | (((unsigned int)q[16 * c + 8 + i]  & 0xFFu) << 16)
                         | (((unsigned int)q[16 * c + 12 + i] & 0xFFu) << 24);
        }
    }
    uint4* dst = reinterpret_cast<uint4*>(xq + (size_t)b * PQ2D_QBLOCK);
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    xmeta[b] = make_int2(__float_as_int(d), sum);
}

// ═════════════════════════ dp4a GEMV ═════════════════════════
template <int S>
__device__ __forceinline__ void pq2_0_gemv_dp4a_body(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq,
    const int2*    __restrict__ xmeta,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2D_ROWS_PER_BLOCK + wid * PQ2D_ROWS_PER_WARP;
    if (rowBase >= n) return;   // no block-level barriers anywhere below, so this is safe

    const int gpr = k / PQ2D_GROUP_SIZE;
    const int bpr = k / PQ2D_QBLOCK;          // activation blocks per column
    const long long totalGroups = (long long)n * gpr;
    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2d_codes_base_offset(totalGroups);

    const int gsub = lane >> 2;   // which of the step's 8 groups this lane works on
    const int qb   = lane & 3;    // 32-element block within the group: code bytes 8qb..8qb+7

    int rows[PQ2D_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < PQ2D_ROWS_PER_WARP; rr++)
        rows[rr] = min(rowBase + rr, n - 1);   // clamp the tail row; its result is never stored

    float acc[PQ2D_ROWS_PER_WARP][S];
    #pragma unroll
    for (int rr = 0; rr < PQ2D_ROWS_PER_WARP; rr++)
        #pragma unroll
        for (int s = 0; s < S; s++)
            acc[rr][s] = 0.0f;

    for (int g0 = 0; g0 < gpr; g0 += PQ2D_GROUPS_PER_STEP)
    {
        const int g = g0 + gsub;
        if (g >= gpr) continue;   // only diverges on a tail step when gpr % 8 != 0

        // Decode both rows' 32 codes once into 8 code-plane words each; reused for every column.
        // wp[rr][4*h + i] = (codeWord_h >> 2i) & 0x03030303, h = 0 (elements 0..15), 1 (16..31).
        int   wp[PQ2D_ROWS_PER_WARP][8];
        float ws[PQ2D_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < PQ2D_ROWS_PER_WARP; rr++)
        {
            const long long gFlat = (long long)rows[rr] * gpr + g;
            ws[rr] = __half2float(scales[gFlat]);   // 8 distinct addresses per warp
            // gFlat*32 + 8qb is 8-byte aligned (codesBase is 32-byte aligned).
            const uint2 cw = __ldcs(reinterpret_cast<const uint2*>(
                codesBase + (size_t)gFlat * 32 + 8 * qb));
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                wp[rr][i]     = (int)((cw.x >> (2 * i)) & 0x03030303u);
                wp[rr][4 + i] = (int)((cw.y >> (2 * i)) & 0x03030303u);
            }
        }

        const int blk = g * (PQ2D_GROUP_SIZE / PQ2D_QBLOCK) + qb;   // activation block index in a column
        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            // 32 permuted int8 activations of this block: 32-byte aligned (blk*32 + s*k, k % 128 == 0).
            const uint4* xp = reinterpret_cast<const uint4*>(xq + (size_t)s * k + (size_t)blk * PQ2D_QBLOCK);
            const uint4 a  = __ldg(xp);
            const uint4 b  = __ldg(xp + 1);
            const int2  md = __ldg(xmeta + (size_t)s * bpr + blk);
            const float d  = __int_as_float(md.x);

            #pragma unroll
            for (int rr = 0; rr < PQ2D_ROWS_PER_WARP; rr++)
            {
                int isum = -md.y;   // sum (code - 1) * q = sum code * q - sum q
                isum = __dp4a(wp[rr][0], (int)a.x, isum);
                isum = __dp4a(wp[rr][1], (int)a.y, isum);
                isum = __dp4a(wp[rr][2], (int)a.z, isum);
                isum = __dp4a(wp[rr][3], (int)a.w, isum);
                isum = __dp4a(wp[rr][4], (int)b.x, isum);
                isum = __dp4a(wp[rr][5], (int)b.y, isum);
                isum = __dp4a(wp[rr][6], (int)b.z, isum);
                isum = __dp4a(wp[rr][7], (int)b.w, isum);
                acc[rr][s] = fmaf(__int2float_rn(isum), d * ws[rr], acc[rr][s]);
            }
        }
    }

    // Reduce every (row, column) accumulator across the warp's 32 lanes (8 groups x 4 blocks each).
    #pragma unroll
    for (int rr = 0; rr < PQ2D_ROWS_PER_WARP; rr++)
    {
        const int row = rowBase + rr;
        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            const float v = pq2d_warp_reduce(acc[rr][s]);
            if (lane == 0 && row < n)
                y[(size_t)s * n + row] = v;
        }
    }
}

// Exact-width entry points (extern "C" so the driver API can look them up by name).
#define PQ2D_DEFINE_ENTRY(S_)                                                          \
    extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_dp4a_f32y_##S_(      \
        const uint8_t* __restrict__ weight,                                            \
        const int8_t*  __restrict__ xq,                                                \
        const int2*    __restrict__ xmeta,                                             \
        float*         __restrict__ y,                                                 \
        const int n,                                                                   \
        const int k)                                                                   \
    {                                                                                  \
        pq2_0_gemv_dp4a_body<S_>(weight, xq, xmeta, y, n, k);                          \
    }

PQ2D_DEFINE_ENTRY(1)
PQ2D_DEFINE_ENTRY(2)
PQ2D_DEFINE_ENTRY(3)
PQ2D_DEFINE_ENTRY(4)
PQ2D_DEFINE_ENTRY(5)
PQ2D_DEFINE_ENTRY(6)
PQ2D_DEFINE_ENTRY(7)
PQ2D_DEFINE_ENTRY(8)
