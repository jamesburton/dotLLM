// BF16 and MXFP4 dequantization kernels for dotLLM (issue #258).
//
// Both formats were previously CPU/Vulkan-only; CUDA rejected them with
// "GPU dequantization not supported". Semantics below are a direct port of
// llama.cpp's reference row-dequantizers, cross-checked element-for-element
// against DotLLM.Cpu.Kernels.Dequantize:
//
//   BF16   ( 2 B / elem): the top 16 bits of the F32 bit pattern. Expansion is
//                         bit-exact — shift left 16 and reinterpret.
//   MXFP4  (17 B / 32 elem): uint8 e (E8M0 exponent); uint8 qs[16].
//                         Element j     = kvalues_mxfp4[qs[j] & 0x0F] * d
//                         Element j+16  = kvalues_mxfp4[qs[j] >> 4]   * d
//                         where d = ggml_e8m0_to_fp32_half(e).
//
// The low/high nibbles map to the two *halves* of the block, not to adjacent
// element pairs — the same convention that Q4_0/Q4_1 got wrong in #254.

#include <cuda_fp16.h>
#include <stdint.h>

// ──────────────────────────── BF16 ────────────────────────────

extern "C" __global__ void __launch_bounds__(256) dequant_bf16_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_elements)
{
    const uint16_t* bf = reinterpret_cast<const uint16_t*>(src);
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        uint32_t bits = (uint32_t)bf[i] << 16;
        dst[i] = __float2half(__uint_as_float(bits));
    }
}

extern "C" __global__ void __launch_bounds__(256) dequant_bf16_f32(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const int total_elements)
{
    const uint16_t* bf = reinterpret_cast<const uint16_t*>(src);
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        uint32_t bits = (uint32_t)bf[i] << 16;
        dst[i] = __uint_as_float(bits);
    }
}

// ──────────────────────────── MXFP4 ────────────────────────────

#define MXFP4_BLOCK_SIZE 32
#define MXFP4_BLOCK_BYTES 17

// llama.cpp kvalues_mxfp4 — the E2M1 code points *doubled*, paired with the
// halved block scale from ggml_e8m0_to_fp32_half.
__device__ __constant__ int8_t kvalues_mxfp4_device[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ggml_e8m0_to_fp32_half: 0.5 * 2^(e-127) = 2^(e-128). For e < 2 the result is
// denormal, so the bit pattern is built by shifting 2^-128's mantissa bit.
__device__ __forceinline__ float e8m0_to_fp32_half(uint8_t e)
{
    uint32_t bits = (e < 2)
        ? (0x00200000u << e)          // 2^-128, 2^-127 (denormal patterns)
        : ((uint32_t)(e - 1) << 23);  // 2^(e-128)
    return __uint_as_float(bits);
}

// 256 threads cover 8 MXFP4 blocks per CUDA block (32 elements each).
extern "C" __global__ void __launch_bounds__(256) dequant_mxfp4_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int sub = threadIdx.x >> 5;   // which of the 8 blocks in this tile
    int elem = threadIdx.x & 31;  // element within the block
    int j = elem & 15;

    for (int tile = blockIdx.x; tile * 8 < total_blocks; tile += gridDim.x) {
        int block_idx = tile * 8 + sub;
        if (block_idx >= total_blocks) {
            continue;
        }
        const uint8_t* block = src + (size_t)block_idx * MXFP4_BLOCK_BYTES;
        float d = e8m0_to_fp32_half(block[0]);
        uint8_t packed = block[1 + j];
        int q = (elem < 16) ? (packed & 0x0F) : (packed >> 4);
        dst[(size_t)block_idx * MXFP4_BLOCK_SIZE + elem] =
            __float2half((float)kvalues_mxfp4_device[q] * d);
    }
}

extern "C" __global__ void __launch_bounds__(256) dequant_mxfp4_f32(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const int total_blocks)
{
    int sub = threadIdx.x >> 5;
    int elem = threadIdx.x & 31;
    int j = elem & 15;

    for (int tile = blockIdx.x; tile * 8 < total_blocks; tile += gridDim.x) {
        int block_idx = tile * 8 + sub;
        if (block_idx >= total_blocks) {
            continue;
        }
        const uint8_t* block = src + (size_t)block_idx * MXFP4_BLOCK_BYTES;
        float d = e8m0_to_fp32_half(block[0]);
        uint8_t packed = block[1 + j];
        int q = (elem < 16) ? (packed & 0x0F) : (packed >> 4);
        dst[(size_t)block_idx * MXFP4_BLOCK_SIZE + elem] =
            (float)kvalues_mxfp4_device[q] * d;
    }
}
