// IQ-family dequantization kernels for dotLLM.
// Layouts mirror ggml-common.h block_iq4_nl and block_iq4_xs.

#include <cuda_fp16.h>
#include <stdint.h>

__device__ __constant__ int8_t kvalues_iq4nl_device[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

#define IQ4_NL_BLOCK_SIZE 32
#define IQ4_NL_BLOCK_BYTES 18

// IQ4_NL blocks are independent 32-element/18-byte units — unlike IQ4_XS/IQ3_S
// there is no 256-element super-block grouping them on disk. A launch_bounds(256)
// CTA mapping 1 thread per element of ONE block therefore only keeps 32 of 256
// threads active (1/8 occupancy, issue #265). To match the IQ4_XS/IQ3_S pattern
// of "one thread per element across a 256-wide unit of work", each CTA instead
// covers a virtual group of 8 consecutive IQ4_NL blocks (8 * 32 = 256 elements):
// thread t handles local block (t >> 5) within the group, element (t & 31)
// within that block. This changes only the thread-to-work mapping and grid-loop
// granularity — the on-disk block layout and per-element math are unchanged.
#define IQ4_NL_BLOCKS_PER_GROUP 8

extern "C" __global__ void __launch_bounds__(256) dequant_iq4_nl_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int t = threadIdx.x; // 0..255
    int local_block = t >> 5;   // 0..7
    int elem = t & 31;          // 0..31

    int total_groups = (total_blocks + IQ4_NL_BLOCKS_PER_GROUP - 1) / IQ4_NL_BLOCKS_PER_GROUP;

    for (int group = blockIdx.x; group < total_groups; group += gridDim.x) {
        int block_idx = group * IQ4_NL_BLOCKS_PER_GROUP + local_block;
        if (block_idx >= total_blocks) {
            continue;
        }

        const uint8_t* block = src + (size_t)block_idx * IQ4_NL_BLOCK_BYTES;
        const uint8_t* qs = block + 2;
        float d = __half2float(*reinterpret_cast<const half*>(block));

        int j = elem & 15;
        uint8_t packed = qs[j];
        int q = elem < 16 ? (packed & 0x0F) : (packed >> 4);
        dst[(size_t)block_idx * IQ4_NL_BLOCK_SIZE + elem] =
            __float2half(d * (float)kvalues_iq4nl_device[q]);
    }
}

extern "C" __global__ void __launch_bounds__(256) dequant_iq4_nl_f32(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const int total_blocks)
{
    int t = threadIdx.x; // 0..255
    int local_block = t >> 5;   // 0..7
    int elem = t & 31;          // 0..31

    int total_groups = (total_blocks + IQ4_NL_BLOCKS_PER_GROUP - 1) / IQ4_NL_BLOCKS_PER_GROUP;

    for (int group = blockIdx.x; group < total_groups; group += gridDim.x) {
        int block_idx = group * IQ4_NL_BLOCKS_PER_GROUP + local_block;
        if (block_idx >= total_blocks) {
            continue;
        }

        const uint8_t* block = src + (size_t)block_idx * IQ4_NL_BLOCK_BYTES;
        const uint8_t* qs = block + 2;
        float d = __half2float(*reinterpret_cast<const half*>(block));

        int j = elem & 15;
        uint8_t packed = qs[j];
        int q = elem < 16 ? (packed & 0x0F) : (packed >> 4);
        dst[(size_t)block_idx * IQ4_NL_BLOCK_SIZE + elem] =
            d * (float)kvalues_iq4nl_device[q];
    }
}

#define IQ4_XS_SUPER_BLOCK_SIZE 256
#define IQ4_XS_BLOCK_BYTES 136

extern "C" __global__ void __launch_bounds__(256) dequant_iq4_xs_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x) {
        const uint8_t* block = src + (size_t)sb_idx * IQ4_XS_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        uint16_t scales_h = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;

        int ib = t >> 5;
        int elem = t & 31;
        int q_index = elem & 15;

        int low = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        int high = (scales_h >> (2 * ib)) & 0x03;
        int ls = low | (high << 4);
        float dl = d * (float)(ls - 32);

        uint8_t packed = qs[ib * 16 + q_index];
        int q = elem < 16 ? (packed & 0x0F) : (packed >> 4);
        dst[(size_t)sb_idx * IQ4_XS_SUPER_BLOCK_SIZE + t] =
            __float2half(dl * (float)kvalues_iq4nl_device[q]);
    }
}

extern "C" __global__ void __launch_bounds__(256) dequant_iq4_xs_f32(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x) {
        const uint8_t* block = src + (size_t)sb_idx * IQ4_XS_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        uint16_t scales_h = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;

        int ib = t >> 5;
        int elem = t & 31;
        int q_index = elem & 15;

        int low = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        int high = (scales_h >> (2 * ib)) & 0x03;
        int ls = low | (high << 4);
        float dl = d * (float)(ls - 32);

        uint8_t packed = qs[ib * 16 + q_index];
        int q = elem < 16 ? (packed & 0x0F) : (packed >> 4);
        dst[(size_t)sb_idx * IQ4_XS_SUPER_BLOCK_SIZE + t] =
            dl * (float)kvalues_iq4nl_device[q];
    }
}
