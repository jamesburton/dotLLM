// Decode-step KV cache write. Replaces a host-side cuMemcpyDtoDAsync where the
// destination address is `dst_base + posPtr[0] * row_bytes` — fine in eager
// mode but fatal under CUDA Graphs because the host-computed address gets baked
// into the graph at instantiate time and the next decode step would clobber the
// same row.
//
// Single decode token, FP16. Launch with grid=(num_blocks_for_row), block=256.
// Writes one row of `kv_stride` FP16 elements from `src` to
// `dst_base + posPtr[0] * row_stride_fp16`.
//
// We pass row_stride_fp16 (in element units, not bytes) to keep arithmetic in
// the kernel cheap and unambiguous.

#include <cuda_fp16.h>

extern "C" __global__ void kv_write_one_f16(
    const half* __restrict__ src,        // [kv_stride] FP16 (one row of new K or V)
    half* __restrict__ dst_base,         // [maxSeqLen, kv_stride] FP16 cache
    const int kv_stride,                 // = num_kv_heads * head_dim
    const int* __restrict__ pos_ptr)     // device-resident write index
{
    int pos = pos_ptr[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kv_stride) return;

    half* dst = dst_base + (size_t)pos * kv_stride;
    dst[tid] = src[tid];
}

// ───────────────────────────────────────────────────────────────────
// Quantized-KV graph-friendly helpers.
//
// kv_write_one_f16_ring: writes the new FP16 row into the per-layer
//   ring buffer at slot `pos % window_size`. Same role as
//   kv_write_one_f16, just modulo-indexed.
// ───────────────────────────────────────────────────────────────────

#include <stdint.h>

#define KV_Q8_0_BLOCK_SIZE 32
#define KV_Q8_0_BLOCK_BYTES 34
#define KV_Q4_0_BLOCK_SIZE 32
#define KV_Q4_0_BLOCK_BYTES 18

extern "C" __global__ void kv_write_one_f16_ring(
    const half* __restrict__ src,        // [kv_stride] FP16
    half* __restrict__ ring_base,        // [window_size, kv_stride] FP16
    const int kv_stride,
    const int window_size,
    const int* __restrict__ pos_ptr)
{
    int pos = pos_ptr[0];
    int slot = pos % window_size;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kv_stride) return;
    half* dst = ring_base + (size_t)slot * kv_stride;
    dst[tid] = src[tid];
}

// Eager (non-graph) dequant of a contiguous run of KV-format Q4_0 blocks.
//
// The KV Q4_0 convention is ADJACENT PAIRS (low nibble = even element, high nibble = odd
// element) -- what quant_f16_to_q4_0 / the CPU KvQuantize write. It is NOT the GGUF/ggml weight layout (low nibbles = elements 0..15, high = 16..31)
// that the generic dequant_q4_0_f16 in dequant.cu unpacks. Using the generic kernel on KV
// data permutes elements inside every 32-block (issue #956).
extern "C" __global__ void kv_dequant_q4_0(
    const uint8_t* __restrict__ quant_base,
    half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % KV_Q4_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / KV_Q4_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / KV_Q4_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / KV_Q4_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = quant_base + (size_t)block_idx * KV_Q4_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        uint8_t packed = (block + 2)[lane / 2];
        int val = (lane & 1) ? ((int)(packed >> 4) - 8) : ((int)(packed & 0x0F) - 8);
        dst[(size_t)block_idx * KV_Q4_0_BLOCK_SIZE + lane] = __float2half(d * (float)val);
    }
}
