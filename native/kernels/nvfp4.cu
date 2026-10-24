// NVFP4 (GGUF type 40) kernels for dotLLM (issue #1038).
//
// Block = 36 bytes per 64 elements (llama.cpp block_nvfp4):
//   uint8 d[4]   one unsigned UE4M3 scale per 16-element sub-block
//   uint8 qs[32] E2M1 nibbles; sub-block s owns qs[8s .. 8s+7], and byte j holds
//                element 16s+j in its low nibble and element 16s+8+j in its high nibble.
//   value = kvalues_mxfp4[nibble] * ue4m3_half(d[s])
// Semantics are a direct port of llama.cpp's dequantize_row_nvfp4 / ggml_ue4m3_to_fp32 and are
// cross-checked against DotLLM.Cpu.Kernels.Dequantize (itself pinned bit-exact to llama.cpp b11739).
//
// The per-tensor second-level scale (the sibling X.scale tensor) multiplies the matmul OUTPUT
// (llama.cpp build_lora_mm). nvfp4_gemv_f32 takes it directly; the dequant + cuBLAS prefill path
// applies it afterwards with scale_f32_inplace.

#include <cuda_fp16.h>
#include <stdint.h>

#define NVFP4_BLOCK_SIZE 64
#define NVFP4_BLOCK_BYTES 36

// E2M1 code points doubled (ggml kvalues_mxfp4); the 0.5 lives in ue4m3_half.
__device__ __constant__ int8_t kvalues_nvfp4_device[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Literal port of ggml_ue4m3_to_fp32: 0x00 and 0x7F (the E4M3 NaN) decode to 0, bit 7 is ignored.
__device__ __forceinline__ float ue4m3_half(uint32_t x)
{
    if (x == 0u || x == 0x7Fu) return 0.0f;
    int e = (int)((x >> 3) & 0xFu);
    float m = (float)(x & 7u);
    float raw = (e == 0) ? ldexpf(m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return raw * 0.5f;
}

// Element `e` (0..63) of the block at `block`.
__device__ __forceinline__ float nvfp4_element(const uint8_t* __restrict__ block, int e)
{
    int s = e >> 4;
    int w = e & 15;
    uint8_t packed = block[4 + 8 * s + (w & 7)];
    int q = (w < 8) ? (packed & 0x0F) : (packed >> 4);
    return (float)kvalues_nvfp4_device[q] * ue4m3_half(block[s]);
}

// 256 threads cover 4 NVFP4 blocks per CUDA block (64 elements each).
extern "C" __global__ void __launch_bounds__(256) dequant_nvfp4_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int sub = threadIdx.x >> 6;
    int e = threadIdx.x & 63;
    for (int tile = blockIdx.x; tile * 4 < total_blocks; tile += gridDim.x) {
        int b = tile * 4 + sub;
        if (b >= total_blocks) continue;
        dst[(size_t)b * NVFP4_BLOCK_SIZE + e] =
            __float2half(nvfp4_element(src + (size_t)b * NVFP4_BLOCK_BYTES, e));
    }
}

extern "C" __global__ void __launch_bounds__(256) dequant_nvfp4_f32(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const int total_blocks)
{
    int sub = threadIdx.x >> 6;
    int e = threadIdx.x & 63;
    for (int tile = blockIdx.x; tile * 4 < total_blocks; tile += gridDim.x) {
        int b = tile * 4 + sub;
        if (b >= total_blocks) continue;
        dst[(size_t)b * NVFP4_BLOCK_SIZE + e] = nvfp4_element(src + (size_t)b * NVFP4_BLOCK_BYTES, e);
    }
}

// Decode GEMV: y[m] = scale * sum_k W[m, k] * x[k]. One warp per output row (4 rows per 128-thread
// CUDA block); each lane walks whole 64-element blocks, reading the 36-byte block as nine aligned 32-bit
// words (blocks and rows are 4-byte aligned) and x as float4s, then a warp-shuffle reduce. FP32
// activations (no Q8_0 rounding).
extern "C" __global__ void __launch_bounds__(128) nvfp4_gemv_f32(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int m,
    const int blocks_per_row,
    const float scale)
{
    // Per-lane divergent indices into __constant__ memory serialize, so the 16 code points live in
    // shared memory (bank-conflict-free broadcast-or-distinct reads) for the inner loop.
    // The 256 UE4M3 scale decodes are tabled the same way (two entries per thread).
    __shared__ float kv[16];
    __shared__ float ue[256];
    if (threadIdx.x < 16) kv[threadIdx.x] = (float)kvalues_nvfp4_device[threadIdx.x];
    ue[threadIdx.x] = ue4m3_half(threadIdx.x);
    ue[threadIdx.x + 128] = ue4m3_half(threadIdx.x + 128);
    __syncthreads();

    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= m) return;
    const uint32_t* wrow = reinterpret_cast<const uint32_t*>(w + (size_t)row * blocks_per_row * NVFP4_BLOCK_BYTES);

    float acc = 0.0f;
    for (int b = lane; b < blocks_per_row; b += 32) {
        const uint32_t* blk = wrow + (size_t)b * (NVFP4_BLOCK_BYTES / 4);
        const float4* xb = reinterpret_cast<const float4*>(x + (size_t)b * NVFP4_BLOCK_SIZE);
        uint32_t scales = blk[0];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            float d = ue[(scales >> (8 * s)) & 0xFFu];
            float sub = 0.0f;
            #pragma unroll
            for (int h = 0; h < 2; h++) {                 // bytes 4h..4h+3 of this sub-block's qs
                uint32_t q = blk[1 + 2 * s + h];
                float4 lo = xb[4 * s + h];                // elements 16s + 4h .. +3      (low nibbles)
                float4 hi = xb[4 * s + 2 + h];            // elements 16s + 8 + 4h .. +3  (high nibbles)
                sub += lo.x * kv[(q      ) & 0xF] + hi.x * kv[(q >>  4) & 0xF]
                     + lo.y * kv[(q >>  8) & 0xF] + hi.y * kv[(q >> 12) & 0xF]
                     + lo.z * kv[(q >> 16) & 0xF] + hi.z * kv[(q >> 20) & 0xF]
                     + lo.w * kv[(q >> 24) & 0xF] + hi.w * kv[(q >> 28) & 0xF];
            }
            acc += d * sub;
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
        y[row] = acc * scale;
}

// ═════════════════════════ dp4a decode GEMV on int8 activations (#1061) ═════════════════════════
// y[m] = scale * sum_k W[m, k] * x[k] with x quantized to int8 ONCE per token by the house F32 -> int8
// activation quantizer pq2_0_dp4a_quantize_x (pq2_0_gemv_dp4a.cu). That quantizer is byte-exact with the
// CPU's MatMul.QuantizeF32ToQ8_0 (32-element blocks, q = rne(x * (1 / (amax / 127))), dot scale =
// float(half(amax / 127))), which is the activation precision of the CPU oracle (MatMul.GemmNvfp4, a
// transcription of llama.cpp's ggml_vec_dot_nvfp4_q8_0) and of llama.cpp CUDA's vec_dot_nvfp4_q8_1. So
// per 16-element sub-block the int32 sum is exact and only the FP32 summation order differs from the CPU.
//
// Activation layout (the quantizer's contract): xq is int8 PERMUTED within each 16-element chunk -- chunk
// c's word i holds elements { 16c+i, 16c+4+i, 16c+8+i, 16c+12+i }; xmeta[blk] = { bits(d), sum q } per
// 32-element block (the sum is unused: E2M1 codes are symmetric, no offset term).
//
// Work unit = one 32-element activation block = half an NVFP4 block (sub-blocks 2h, 2h+1 of block b,
// u = 2b + h). For one sub-block, qs words w0 = bytes 0..3 (low nibbles: elements 0..3, high: 8..11) and
// w1 = bytes 4..7 (low: 4..7, high: 12..15). Gathering byte i of w0 and of w1 side by side and looking the
// nibbles up in the E2M1 table gives elements { i, 8+i, 4+i, 12+i }; one byte swap makes that activation
// word i's order { i, 4+i, 8+i, 12+i }, so each 16 elements are 2 gathers + 2 table lookups + 4 swaps + 4 dp4a.
//
// Work split: each warp owns NV4D_ROWS_PER_WARP = 2 consecutive output rows (4 warps = 128 threads, 8 rows
// per CUDA block); lanes stride over the units, so a warp step reads 16 consecutive NVFP4 blocks (576
// contiguous bytes) per row and 1 KB of int8 activations, reused from registers for both rows. Then a
// shuffle reduce per row. Measured on an RTX 3060 (#1061), 2 rows per warp matched or beat 1 and 4 on every
// shape of the sweep that is not launch-bound (4 lost up to 11% at k = 18560 and 8192).
// Contract: k % 64 == 0; xq 16-byte aligned, xmeta 8-byte aligned (cuMemAlloc'd scratch is 256-aligned).

// kvalues_mxfp4 (E2M1 doubled) as int8, four per word, for __byte_perm lookups (codes 0..3, 4..7, 8..11, 12..15).
#define NV4_TAB0 0x03020100u   //  0,  1,  2,   3
#define NV4_TAB1 0x0C080604u   //  4,  6,  8,  12
#define NV4_TAB2 0xFDFEFF00u   //  0, -1, -2,  -3
#define NV4_TAB3 0xF4F8FAFCu   // -4, -6, -8, -12

#define NV4D_WARPS_PER_BLOCK 4
#define NV4D_ROWS_PER_WARP   2   // the launcher mirrors both

// Table lookup of the 8 nibbles of q (nibble n = bits 4n..4n+3): byte j of lo = kvalue[nibble j],
// byte j of hi = kvalue[nibble 4+j]. __byte_perm uses the low 3 bits of each selector nibble, so each
// lookup selects within the 8 positive or the 8 negative codes, and bit 3 picks between the two
// (llama.cpp get_int_from_table_16's CUDA path, without its final even/odd split).
__device__ __forceinline__ void nvfp4_lut8(const uint32_t q, uint32_t& lo, uint32_t& hi)
{
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    lo = __byte_perm(__byte_perm(NV4_TAB0, NV4_TAB1, q), __byte_perm(NV4_TAB2, NV4_TAB3, q), sel);
    const uint32_t q16 = q >> 16;
    hi = __byte_perm(__byte_perm(NV4_TAB0, NV4_TAB1, q16), __byte_perm(NV4_TAB2, NV4_TAB3, q16), sel >> 16);
}

// ue4m3_half without ldexpf or a table: bit 7 ignored, 0x7F -> 0, exponent 0 -> man * 2^-10, otherwise
// (1 + man/8) * 2^(exp-8) built directly as float bits. Exact for all 256 bytes (same values as ue4m3_half).
__device__ __forceinline__ float ue4m3_half_bits(const uint32_t x)
{
    const uint32_t v = x & 0x7Fu;
    const uint32_t e = v >> 3, m = v & 7u;
    const float r = (e == 0u) ? (float)m * 0.0009765625f : __uint_as_float(((e + 119u) << 23) | (m << 20));
    return (x == 0x7Fu) ? 0.0f : r;
}

// Exact int32 dot of one 16-element NVFP4 sub-block (qs words w0, w1) with its permuted activation chunk xa.
__device__ __forceinline__ int nvfp4_dot16(const uint32_t w0, const uint32_t w1, const int4 xa)
{
    uint32_t lo, hi;
    nvfp4_lut8(__byte_perm(w0, w1, 0x5140u), lo, hi);    // bytes 0,1 of w0/w1 -> elements {0,8,4,12}, {1,9,5,13}
    int s = __dp4a((int)__byte_perm(lo, 0u, 0x3120u), xa.x, 0);
    s = __dp4a((int)__byte_perm(hi, 0u, 0x3120u), xa.y, s);
    nvfp4_lut8(__byte_perm(w0, w1, 0x7362u), lo, hi);    // bytes 2,3 -> elements {2,10,6,14}, {3,11,7,15}
    s = __dp4a((int)__byte_perm(lo, 0u, 0x3120u), xa.z, s);
    s = __dp4a((int)__byte_perm(hi, 0u, 0x3120u), xa.w, s);
    return s;
}

extern "C" __global__ void __launch_bounds__(32 * NV4D_WARPS_PER_BLOCK) nvfp4_gemv_q8(
    const uint8_t* __restrict__ w,
    const int8_t*  __restrict__ xq,
    const int2*    __restrict__ xmeta,
    float* __restrict__ y,
    const int m,
    const int blocks_per_row,
    const float scale)
{
    const int row0 = (blockIdx.x * NV4D_WARPS_PER_BLOCK + (threadIdx.x >> 5)) * NV4D_ROWS_PER_WARP;
    const int lane = threadIdx.x & 31;
    if (row0 >= m) return;
    const int nrows = min(NV4D_ROWS_PER_WARP, m - row0);   // warp-uniform
    const size_t row_words = (size_t)blocks_per_row * (NVFP4_BLOCK_BYTES / 4);
    const uint32_t* w0row = reinterpret_cast<const uint32_t*>(w) + (size_t)row0 * row_words;
    const int4* xq4 = reinterpret_cast<const int4*>(xq);

    float acc[NV4D_ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < NV4D_ROWS_PER_WARP; r++) acc[r] = 0.0f;

    const int units = 2 * blocks_per_row;
    for (int u = lane; u < units; u += 32) {
        const int b = u >> 1, h = u & 1;
        const int4 xa = __ldg(xq4 + 2 * u);                  // sub-block 2h
        const int4 xb = __ldg(xq4 + 2 * u + 1);              // sub-block 2h+1
        const float dx = __int_as_float(__ldg(&xmeta[u].x));
        #pragma unroll
        for (int r = 0; r < NV4D_ROWS_PER_WARP; r++) {
            if (r < nrows) {
                const uint32_t* blk = w0row + (size_t)r * row_words + (size_t)b * (NVFP4_BLOCK_BYTES / 4);
                const uint32_t sc = __ldg(blk) >> (16 * h);  // d[2h] | d[2h+1] << 8
                const uint32_t* qs = blk + 1 + 4 * h;
                const int s0 = nvfp4_dot16(__ldg(qs + 0), __ldg(qs + 1), xa);
                const int s1 = nvfp4_dot16(__ldg(qs + 2), __ldg(qs + 3), xb);
                acc[r] += dx * (ue4m3_half_bits(sc & 0xFFu) * (float)s0
                              + ue4m3_half_bits((sc >> 8) & 0xFFu) * (float)s1);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < NV4D_ROWS_PER_WARP; r++) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0 && r < nrows)
            y[row0 + r] = v * scale;                          // the per-tensor X.scale, once
    }
}

// y[i] *= scale — the per-tensor X.scale after the dequant + cuBLAS prefill GEMM.
extern "C" __global__ void __launch_bounds__(256) scale_f32_inplace(
    float* __restrict__ y,
    const int n,
    const float scale)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] *= scale;
}
