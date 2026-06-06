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
//
// kv_dequant_q{8,4}_0_dyn: dequantizes the [0, quant_len) prefix of
//   the per-layer Q-cache into the FP16 attention scratch, where
//   quant_len = max(0, pos+1 - window_size) is read device-side.
//   Grid-stride loop, predicated.
//
// kv_window_to_scratch_dyn: copies the live FP16 window into the
//   contiguous scratch starting at row `quant_len`. Each block handles
//   one ring slot; predicated on whether that slot is currently live.
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

extern "C" __global__ void kv_dequant_q8_0_dyn(
    const uint8_t* __restrict__ quant_base,    // [maxSeqLen, kv_stride/32 * 34]
    half* __restrict__ scratch_base,           // [maxSeqLen, kv_stride]
    const int kv_stride,
    const int window_size,
    const int* __restrict__ pos_ptr)
{
    int pos = pos_ptr[0];
    int quant_len = pos + 1 - window_size;
    if (quant_len <= 0) return;

    int blocks_per_row = kv_stride / KV_Q8_0_BLOCK_SIZE;
    int total_blocks = quant_len * blocks_per_row;

    int lane = threadIdx.x % KV_Q8_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / KV_Q8_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / KV_Q8_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / KV_Q8_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = quant_base + (size_t)block_idx * KV_Q8_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
        scratch_base[(size_t)block_idx * KV_Q8_0_BLOCK_SIZE + lane] = __float2half(d * (float)q);
    }
}

extern "C" __global__ void kv_dequant_q4_0_dyn(
    const uint8_t* __restrict__ quant_base,
    half* __restrict__ scratch_base,
    const int kv_stride,
    const int window_size,
    const int* __restrict__ pos_ptr)
{
    int pos = pos_ptr[0];
    int quant_len = pos + 1 - window_size;
    if (quant_len <= 0) return;

    int blocks_per_row = kv_stride / KV_Q4_0_BLOCK_SIZE;
    int total_blocks = quant_len * blocks_per_row;

    int lane = threadIdx.x % KV_Q4_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / KV_Q4_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / KV_Q4_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / KV_Q4_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = quant_base + (size_t)block_idx * KV_Q4_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;
        int byte_idx = lane / 2;
        uint8_t packed = qs[byte_idx];
        int val = (lane & 1) ? ((int)(packed >> 4) - 8) : ((int)(packed & 0x0F) - 8);
        scratch_base[(size_t)block_idx * KV_Q4_0_BLOCK_SIZE + lane] = __float2half(d * (float)val);
    }
}

// Scatters the live FP16 window from the per-layer ring buffer into the
// contiguous attention scratch starting at scratch row `quant_len`. One
// CUDA block per ring slot — each block decides whether its slot is live
// at the current decode position and, if so, what absolute scratch row to
// land at.
//
// Grid: (window_size, 1, 1).  Block: (kv_stride threads), capped — caller
// uses ceil(kv_stride / 256) along grid.x_inner if kv_stride exceeds the
// block dim. We keep it simple: each block has `min(kv_stride, 1024)`
// threads and uses a per-thread stride loop over kv_stride.
extern "C" __global__ void kv_window_to_scratch_dyn(
    const half* __restrict__ ring_base,        // [window_size, kv_stride]
    half* __restrict__ scratch_base,           // [maxSeqLen, kv_stride]
    const int kv_stride,
    const int window_size,
    const int* __restrict__ pos_ptr)
{
    int pos = pos_ptr[0];
    int current_len = pos + 1;
    int quant_len = current_len - window_size;
    if (quant_len < 0) quant_len = 0;
    int window_len = current_len - quant_len; // == min(current_len, window_size)

    int slot = blockIdx.x;                    // which ring slot this block owns
    if (slot >= window_size) return;
    if (slot >= window_len) return;           // ring slot not yet populated

    // Determine absolute position whose row sits in `slot` right now.
    // Ring is filled by `pos % window_size`. The set of absolute positions
    // ever stored at this slot is {slot, slot + W, slot + 2W, ...}. The
    // most recent one ≤ pos is the current occupant.
    //
    // Special case pos < window_size: ring is partially filled — slot s
    // holds position s itself.
    int abs_pos;
    if (pos < window_size)
    {
        abs_pos = slot;
    }
    else
    {
        // Two candidates: slot + (quant_len / window_size) * window_size
        //                  and the same plus window_size. Pick whichever is
        // ≤ pos AND ≥ quant_len.
        int base = (quant_len / window_size) * window_size;
        int cand_a = base + slot;
        if (cand_a >= quant_len && cand_a <= pos)
        {
            abs_pos = cand_a;
        }
        else
        {
            abs_pos = cand_a + window_size;       // wraps once
        }
    }

    const half* src_row = ring_base + (size_t)slot * kv_stride;
    half* dst_row = scratch_base + (size_t)abs_pos * kv_stride;

    for (int i = threadIdx.x; i < kv_stride; i += blockDim.x)
        dst_row[i] = src_row[i];
}
