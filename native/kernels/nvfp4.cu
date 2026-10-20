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
