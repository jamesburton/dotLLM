#include <cuda_fp16.h>
#include <float.h>

// Causal softmax for the G3 prefill-attention path when the queries start part way into the
// cache (issue #1594): a prefix-cache hit, or any multi-token prefill after a cached prefix.
//
//   QK^T -> cublasGemmStridedBatchedEx -> scores (col-major [sq x skv] per query head, ldc = sq)
//   THIS -> causal softmax with the query rows offset by `offset` positions
//   P*V  -> cublasGemmStridedBatchedEx (k = skv)
//
// Query row tq sits at absolute position offset + tq, so it may attend keys 0..offset+tq.
// Same math as attention_softmax_causal_coalesced_f32in_f16out (FP32 scores in, FP32
// max/exp/sum, normalized probs written to a separate FP16 buffer, masked tail zeroed) with
// the square [s x s] plane generalised to [sq x skv]; offset 0 with sq == skv is that kernel.
// One thread per softmax row: consecutive threads take consecutive tq, so every tk step is
// a coalesced read across the warp.
extern "C" __global__ void __launch_bounds__(256) attention_softmax_causal_offset_f32in_f16out(
    const float* __restrict__ scores,
    half* __restrict__ probs,
    const int sq,
    const int skv,
    const int offset,
    const int num_heads)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= num_heads * sq) return;

    int hq = row_id / sq;
    int tq = row_id % sq;

    size_t plane = (size_t)hq * sq * skv;
    const float* srow = scores + plane + tq;  // element (tq, tk) at srow[tk * sq]
    half* prow = probs + plane + tq;
    int causal_len = min(offset + tq + 1, skv);

    float max_val = -FLT_MAX;
    for (int tk = 0; tk < causal_len; tk++)
    {
        float v = srow[(size_t)tk * sq];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int tk = 0; tk < causal_len; tk++)
        sum_exp += expf(srow[(size_t)tk * sq] - max_val);

    float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (int tk = 0; tk < causal_len; tk++)
        prow[(size_t)tk * sq] = __float2half(expf(srow[(size_t)tk * sq] - max_val) * sum_inv);

    for (int tk = causal_len; tk < skv; tk++)
        prow[(size_t)tk * sq] = __float2half(0.0f);
}
