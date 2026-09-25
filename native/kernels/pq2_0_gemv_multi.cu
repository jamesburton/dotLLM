// Issue #482 — small-S multi-column PQ2_0 GEMV: y[s, row] = W[row, :] . x[s, :] for s < S, S = 1..8.
//
// The CUDA twin of Vulkan #470 (matmul_pq2_0_f32_gemv_multicol.glsl). An MTP / speculative verify
// forward runs S = K+1 = 2..8 tokens. Before this kernel, CudaQwen3HybridDenseTransformerModel.Gemm
// sent every seqLen > 1 PQ2_0 projection through dequant-to-F16 + cuBLAS HGEMM, which re-expands
// the WHOLE packed model (about 330 ms on Bonsai 2 27B, RTX 3060) on every verify step, so verify
// cost 6.4x a decode step. Here each packed weight byte is read and decoded ONCE and applied to all
// S activation rows, so weight traffic is flat in S and only the FMA count grows with it.
//
// ───────────────────────── Layout contract ─────────────────────────
//   * weight: the SPLIT PQ2_0 layout produced at load by pq2_0_repack.cu (all n*gpr fp16 group
//     scales, then — from pq2_0_codes_base_offset — all n*gpr*32 code bytes). Group gFlat =
//     row*gpr + g owns scales[gFlat] and codes[gFlat*32 .. +31]. Byte b of a group holds elements
//     4b..4b+3 at ascending bit offsets {0,2,4,6}, value = code - 1 (see pq2_0_gemv.cu's #269 note).
//   * x: HALF activations, [S, k] row-major (column s starts at x + s*k). The caller converts the
//     F32 activations with convert_f32_to_f16 first — the SAME round-to-nearest the single-column
//     production kernel (pq2_0_gemv_f32io) applies when it stages x into its half xs[] buffer, so
//     both kernels see bit-identical activation values.
//   * y: F32, [S, n] row-major (column s at y + s*n) — the layout cuBLAS LinearF16 produced here.
//   * k % 128 == 0 (PQ2_0 groups). x must be 16-byte aligned (a cuMemAlloc'd scratch buffer is).
//
// ───────────────────────── Work split ─────────────────────────
// 256 threads = 8 warps; each warp owns PQ2M_ROWS_PER_WARP = 2 consecutive output rows, so a block
// covers 16 rows (the same row tiling as pq2_0_gemv_f32io). Unlike that kernel, a LANE here owns 16
// consecutive elements of ONE group rather than 4 elements of every group:
//     lane = 8*gsub + q  ->  group g = g0 + gsub (4 groups per warp step), code bytes 4q..4q+3,
//                           elements 16q..16q+15 of that group.
// so per row a warp step reads 4 consecutive groups' codes = 128 contiguous bytes with ONE 32-bit
// load per lane (the S=1 kernel issues one 8-bit load per lane per group), and per column two
// 16-byte loads per lane (32 contiguous bytes; 1 KB per warp).
//
// ───────────────────────── Why no shared memory ─────────────────────────
// The S=1 kernel stages x into a half xs[] window of 17 KB. Multiplying that by S does not fit
// (S=8: 136 KB > the 48 KB static cap and the 100 KB/SM total), and shrinking the window instead
// multiplies the __syncthreads() count (34 per launch at k=17408, S=8) — the cost that sank the
// batched-staging experiment recorded in pq2_0_gemv.cu. So x is read straight from global with
// vectorised loads (the #470 lesson: one vec4 per code byte instead of four scalars removed a cliff
// on ffn_down). The 8 warps of a block read the same x bytes, which the L1 serves; the weight words
// are loaded with __ldcs (evict-first streaming) so the once-read weight stream does not push the
// reused activations out of L1. There are no block-level barriers at all, so a warp past the last
// row simply returns.
//
// ───────────────────────── Arithmetic (why it is not bit-identical to S=1) ─────────────────────────
// Per row and group the 16 codes are decoded once into t_j = code_j - 1 in {-1, 0, +1} with the
// exact magic-number trick  t = as_float(0x4B000000 | code) - (2^23 + 1)  (no I2F, which runs at
// quarter rate on sm_86), then per column
//     p = sum_j t_j * x_j   (16 FMAs, ascending j)       acc += p * scale
// The S=1 kernel instead computes (sum_j code_j * x_j - sum_j x_j) * scale over 4 elements per lane
// with a different lane-to-element mapping and reduction tree, so results differ by FP32
// reassociation — a few ULP of the row sum, far below the half rounding both kernels share. The
// parity tests hold the single-column kernel as a tight-tolerance oracle, like #470 does on Vulkan.
//
// ───────────────────────── Cost model (the reason for the choices above) ─────────────────────────
// Per element and row: 3 ALU ops of decode (SHF + LOP3 + FADD) amortised over all S columns, plus
// ~(1 + 1/16) FMA per column, plus 1/2 op per column for the half->float conversion (shared by the
// warp's 2 rows). At S >= 3 the FMA term dominates and the kernel becomes ALU-bound on an RTX 3060
// (FP32 lanes vs 360 GB/s), which is why the per-column work is kept at the one-FMA-per-weight
// floor. The next lever past that is dp4a on int8-quantised activations (llama.cpp's MMVQ route),
// which changes the numerics and is deliberately not done here.
//
// Exact widths: one explicit instantiation per S (1..8). #470 measured 2/4/8-only variants costing
// ~40% at S=5 from dead columns. S=1 is instantiated for A/B against pq2_0_gemv_f32io only; the
// production S=1 path stays on that kernel unless measurement says otherwise.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2M_GROUP_SIZE      128
#define PQ2M_ROWS_PER_WARP   2
#define PQ2M_WARPS_PER_BLOCK 8
#define PQ2M_ROWS_PER_BLOCK  (PQ2M_WARPS_PER_BLOCK * PQ2M_ROWS_PER_WARP)   // 16; launcher mirrors it
#define PQ2M_GROUPS_PER_STEP 4                                             // groups a warp covers per step

// Must match pq2_0_gemv.cu / pq2_0_repack.cu / dequant_pq2_0.cu's helper of the same shape.
__device__ __forceinline__ size_t pq2m_codes_base_offset(long long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

__device__ __forceinline__ float pq2m_warp_reduce(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

// code c in {0,1,2,3} at bit offset 2*j of `word` -> (float)(c - 1), exactly.
// as_float(0x4B000000 | c) == 2^23 + c exactly, so subtracting 2^23 + 1 is exact.
__device__ __forceinline__ float pq2m_code_minus_one(unsigned int word, int j)
{
    return __uint_as_float(0x4B000000u | ((word >> (2 * j)) & 3u)) - 8388609.0f;
}

template <int S>
__device__ __forceinline__ void pq2_0_gemv_multi_body(
    const uint8_t* __restrict__ weight,
    const half*    __restrict__ x,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int rowBase = blockIdx.x * PQ2M_ROWS_PER_BLOCK + wid * PQ2M_ROWS_PER_WARP;
    if (rowBase >= n) return;   // no block-level barriers anywhere below, so this is safe

    const int gpr = k / PQ2M_GROUP_SIZE;
    const long long totalGroups = (long long)n * gpr;
    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2m_codes_base_offset(totalGroups);

    const int gsub = lane >> 3;   // which of the step's 4 groups this lane works on
    const int q    = lane & 7;    // code-byte quad within the group: bytes 4q..4q+3, elements 16q..16q+15

    int rows[PQ2M_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < PQ2M_ROWS_PER_WARP; rr++)
        rows[rr] = min(rowBase + rr, n - 1);   // clamp the tail row; its result is never stored

    float acc[PQ2M_ROWS_PER_WARP][S];
    #pragma unroll
    for (int rr = 0; rr < PQ2M_ROWS_PER_WARP; rr++)
        #pragma unroll
        for (int s = 0; s < S; s++)
            acc[rr][s] = 0.0f;

    for (int g0 = 0; g0 < gpr; g0 += PQ2M_GROUPS_PER_STEP)
    {
        const int g = g0 + gsub;
        if (g >= gpr) continue;   // only diverges on a tail step when gpr % 4 != 0

        // Decode both rows' 16 codes once; reused for every column below.
        float t[PQ2M_ROWS_PER_WARP][16];
        float scale[PQ2M_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < PQ2M_ROWS_PER_WARP; rr++)
        {
            const long long gFlat = (long long)rows[rr] * gpr + g;
            scale[rr] = __half2float(scales[gFlat]);   // 4 distinct addresses per warp
            // gFlat*32 + 4q is 4-byte aligned (codesBase is 32-byte aligned). Byte b of the word
            // holds elements 16q + 4b + i at bit 8b + 2i, i.e. element 16q + j sits at bit 2j.
            const unsigned int word = __ldcs(reinterpret_cast<const unsigned int*>(
                codesBase + (size_t)gFlat * 32 + 4 * q));
            #pragma unroll
            for (int j = 0; j < 16; j++)
                t[rr][j] = pq2m_code_minus_one(word, j);
        }

        const size_t xElem = (size_t)g * PQ2M_GROUP_SIZE + 16 * q;   // multiple of 16 halfs -> 32-byte aligned
        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)s * k + xElem);
            const uint4 a = __ldg(xp);
            const uint4 b = __ldg(xp + 1);

            float xf[16];
            float2 f;
            f = __half22float2(*reinterpret_cast<const __half2*>(&a.x)); xf[0]  = f.x; xf[1]  = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&a.y)); xf[2]  = f.x; xf[3]  = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&a.z)); xf[4]  = f.x; xf[5]  = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&a.w)); xf[6]  = f.x; xf[7]  = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&b.x)); xf[8]  = f.x; xf[9]  = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&b.y)); xf[10] = f.x; xf[11] = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&b.z)); xf[12] = f.x; xf[13] = f.y;
            f = __half22float2(*reinterpret_cast<const __half2*>(&b.w)); xf[14] = f.x; xf[15] = f.y;

            #pragma unroll
            for (int rr = 0; rr < PQ2M_ROWS_PER_WARP; rr++)
            {
                float p = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    p = fmaf(t[rr][j], xf[j], p);
                acc[rr][s] = fmaf(p, scale[rr], acc[rr][s]);
            }
        }
    }

    // Reduce every (row, column) accumulator across the warp's 32 lanes (4 groups x 8 quads each).
    #pragma unroll
    for (int rr = 0; rr < PQ2M_ROWS_PER_WARP; rr++)
    {
        const int row = rowBase + rr;
        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            const float v = pq2m_warp_reduce(acc[rr][s]);
            if (lane == 0 && row < n)
                y[(size_t)s * n + row] = v;
        }
    }
}

// Exact-width entry points (extern "C" so the driver API can look them up by name; the template
// body above is shared, so every width runs the same instruction sequence per column).
#define PQ2M_DEFINE_ENTRY(S_)                                                             \
    extern "C" __global__ void __launch_bounds__(256) pq2_0_gemv_multi_f16x_f32y_##S_(   \
        const uint8_t* __restrict__ weight,                                               \
        const half*    __restrict__ x,                                                    \
        float*         __restrict__ y,                                                    \
        const int n,                                                                      \
        const int k)                                                                      \
    {                                                                                     \
        pq2_0_gemv_multi_body<S_>(weight, x, y, n, k);                                    \
    }

PQ2M_DEFINE_ENTRY(1)
PQ2M_DEFINE_ENTRY(2)
PQ2M_DEFINE_ENTRY(3)
PQ2M_DEFINE_ENTRY(4)
PQ2M_DEFINE_ENTRY(5)
PQ2M_DEFINE_ENTRY(6)
PQ2M_DEFINE_ENTRY(7)
PQ2M_DEFINE_ENTRY(8)
