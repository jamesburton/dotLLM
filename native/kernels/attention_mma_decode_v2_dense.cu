// attention_mma_decode_v2_dense.cu -- issue #1922 (CUDA decode at depth, E1 / E3).
//
// Split-KV, GQA-packed, tensor-core decode attention for the dense CudaTransformerModel: the #988 v2 design generalised to
// head dim 64 / 128 (and 256 for TurboQuant), F16 output, normalised F16 split partials. Body: attention_mma_decode_v2_dense.cuh.
// Cooperative launch, grid (numKvHeads, kvSplit), 256 threads, static shared memory only.
//
// Entries (q: F16 [numHeads, HD]; output: F16 [numHeads, HD]; partials: F32 m / l, F16 normalised o):
//   attention_mma_decode_v2_hd{64,128}_{f16,q8q8,q4q4,q8q4,q4q8}
//   attention_mma_decode_v2_hd{64,128,256}_tq       TurboQuant in place (#1892 layout), code space: the caller rotates Q in and
//                                                   O out exactly as for attention_kvtq
// k / v are raw cache bytes in the loader's format: F16 [seqKv, numKvHeads*HD]; KV-format Q8_0 / Q4_0 blocks over the same
// logical grid (bit-identical to dequantizing into F16 first: same element expression as the eager dequant kernels).
// The F32-partial control and 2-blocks-per-SM variants used for the #1922 measurements were removed after the bench
// (James 2026-11-21, bench 2026-11-22).
// Build: compute_86 (mma.sync), -std=c++17 (cooperative_groups), default fmad, no fast math.

#include "attention_mma_decode_v2_dense.cuh"

#define DV2_ENTRY(NAME, HD, KLT, VLT)                                                                 \
extern "C" __global__ void __launch_bounds__(V2_WARPS * 32, 1) NAME(                                   \
    const half* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,               \
    half* __restrict__ output, const int seq_kv, const int num_heads, const int num_kv_heads,         \
    const float scale, const int kv_split, float* __restrict__ partial_max,                           \
    float* __restrict__ partial_sum, __half* __restrict__ partial_out)                                \
{                                                                                                     \
    const KLT kl = { k };                                                                             \
    const VLT vl = { v };                                                                             \
    attn_decode_v2_dense_body<HD, KLT, VLT>(q, kl, vl, output, seq_kv, num_heads, num_kv_heads,       \
        scale, kv_split, partial_max, partial_sum, partial_out);                                      \
}

#define DV2_ENTRIES(HD)                                                                               \
DV2_ENTRY(attention_mma_decode_v2_hd##HD##_f16,  HD, DLoadF16, DLoadF16)                              \
DV2_ENTRY(attention_mma_decode_v2_hd##HD##_q8q8, HD, DLoadQ8,  DLoadQ8)                               \
DV2_ENTRY(attention_mma_decode_v2_hd##HD##_q4q4, HD, DLoadQ4,  DLoadQ4)                               \
DV2_ENTRY(attention_mma_decode_v2_hd##HD##_q8q4, HD, DLoadQ8,  DLoadQ4)                               \
DV2_ENTRY(attention_mma_decode_v2_hd##HD##_q4q8, HD, DLoadQ4,  DLoadQ8)

DV2_ENTRIES(64)
DV2_ENTRIES(128)

// ── TurboQuant in place (E3). The 2^bits centroid table (F32, scaled by 1/sqrt(HD) by the codec) is staged once per block;
// the body's first barrier publishes it. bits in 1..8. ──
#define DV2_TQ_ENTRY(NAME, HD)                                                                        \
extern "C" __global__ void __launch_bounds__(V2_WARPS * 32, 1) NAME(                                   \
    const half* __restrict__ q, const uint32_t* __restrict__ k_codes,                                 \
    const uint32_t* __restrict__ v_codes, const void* __restrict__ k_norms,                           \
    const void* __restrict__ v_norms, const float* __restrict__ centroids,                            \
    half* __restrict__ output, const int seq_kv, const int num_heads, const int num_kv_heads,         \
    const float scale, const int kv_split, float* __restrict__ partial_max,                           \
    float* __restrict__ partial_sum, __half* __restrict__ partial_out,                                \
    const int mse_bits, const int code_uints_per_vec, const int norm_format)                          \
{                                                                                                     \
    __shared__ float s_cent[256];                                                                     \
    for (int i = threadIdx.x; i < (1 << mse_bits); i += blockDim.x) s_cent[i] = centroids[i];         \
    const DLoadTq<HD> kl = { k_codes, k_norms, s_cent, mse_bits, code_uints_per_vec, norm_format };   \
    const DLoadTq<HD> vl = { v_codes, v_norms, s_cent, mse_bits, code_uints_per_vec, norm_format };   \
    attn_decode_v2_dense_body<HD, DLoadTq<HD>, DLoadTq<HD>>(q, kl, vl, output, seq_kv, num_heads,     \
        num_kv_heads, scale, kv_split, partial_max, partial_sum, partial_out);                        \
}

DV2_TQ_ENTRY(attention_mma_decode_v2_hd64_tq,  64)
DV2_TQ_ENTRY(attention_mma_decode_v2_hd128_tq, 128)
DV2_TQ_ENTRY(attention_mma_decode_v2_hd256_tq, 256)
