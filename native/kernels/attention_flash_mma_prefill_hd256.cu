// Tensor-core (mma.sync.m16n8k16) FP16 flash-style PREFILL attention for head_dim 256 that reads
// the (F16) KV cache directly -- issue #962.
//
// The kernel body (design, precision and shared-memory notes) lives in
// attention_flash_mma_prefill_hd256.cuh, shared with the quantized-KV entries of
// attention_flash_mma_prefill_hd256_kvq.cu (issue #1094). This entry instantiates it with the
// cp.async F16 tile loader, which is exactly the load the kernel used before the body was factored
// out, so its numerics are unchanged.

#include "attention_flash_mma_prefill_hd256.cuh"

// q       : [seq_q, num_heads*HEAD_DIM] F16 row-major
// k, v    : [seq_kv, num_kv_heads*HEAD_DIM] F16 row-major (KV cache layout)
// output  : [seq_q, num_heads*HEAD_DIM] F32 row-major
// scale   : softmax scale applied to Q.K (typically rsqrt(HEAD_DIM))
extern "C" __global__ void __launch_bounds__(MAX_GROUP * 32, 1) attention_flash_mma_prefill_hd256_f16(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int position_offset,
    const float scale)
{
    attn_flash_prefill_hd256_body<PrefillTileLoadF16, PrefillTileLoadF16>(
        q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads, position_offset, scale);
}
