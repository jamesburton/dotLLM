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
