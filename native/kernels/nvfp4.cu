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

// Decode GEMV: y[m] = scale * sum_k W[m, k] * x[k]. One 128-thread CUDA block per output row; each
// thread walks whole 64-element blocks (one sub-block scale decode per 16 elements), then a
// warp-shuffle + shared-memory reduce. FP32 activations (no Q8_0 rounding).
extern "C" __global__ void __launch_bounds__(128) nvfp4_gemv_f32(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int m,
    const int blocks_per_row,
    const float scale)
{
    int row = blockIdx.x;
    if (row >= m) return;
    const uint8_t* wrow = w + (size_t)row * blocks_per_row * NVFP4_BLOCK_BYTES;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* blk = wrow + (size_t)b * NVFP4_BLOCK_BYTES;
        const float* xb = x + (size_t)b * NVFP4_BLOCK_SIZE;
        for (int s = 0; s < 4; s++) {
            float d = ue4m3_half(blk[s]);
            const uint8_t* qs = blk + 4 + 8 * s;
            const float* xs = xb + 16 * s;
            float sub = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t p = qs[j];
                sub += xs[j] * (float)kvalues_nvfp4_device[p & 0x0F]
                     + xs[j + 8] * (float)kvalues_nvfp4_device[p >> 4];
            }
            acc += d * sub;
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float warp_sums[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        y[row] = (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * scale;
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
