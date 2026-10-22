// Issue #1109 precision experiment: the head_dim-256 flash PREFILL kernel with F16 tensor-core
// accumulation in place of F32, for the QK^T scores and/or the PV output. EXPERIMENT ONLY -- loaded
// lazily by C# when DOTLLM_CUDA_FLASH_ACC selects a variant (default unset = never loaded, the
// shipped F32-accumulate entries in attention_flash_mma_prefill_hd256(_kvq).cu run).
//
// Why: GeForce Ampere runs mma.m16n8k16 with an F32 accumulator at half the F16-accumulator rate.
// The question is how much accuracy each accumulator costs over a long key range (4k..100k keys on
// Bonsai-2 27B), so CLAUDE.md's precision rules can say, with numbers, which ones stay F32.
//
// Variants (template arguments of attn_flash_prefill_hd256_body, see the .cuh):
//   _qk16      F16 S accumulation over head_dim (Q pre-scaled by 1/16, exact), F32 PV
//   _pv16      F32 S, F16 O accumulated over the whole key range (rescaled in F16)
//   _pv16t     F32 S, F16 PV within one 16-key tile, promoted into the F32 O every tile
//   _qk16pv16  both F16
// for the F16 KV cache and for Q4_0 K + Q4_0 V (the 128k configuration).
//
// Build: compute_86 (mma.sync), C++17, like the shipped hd256 entries; PTX .version 8.7 (CUDA 12.8).

#define HD256_KVQ_NO_ENTRIES
#include "attention_flash_mma_prefill_hd256_kvq.cu"   // body + PrefillTileLoadStaged, no shipped entries

#define XACC_F16KV_ENTRY(NAME, QKA, PVA)                                                          \
extern "C" __global__ void __launch_bounds__(MAX_GROUP * 32, 1) NAME(                             \
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,           \
    float* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,           \
    const int num_kv_heads, const int position_offset, const float scale)                         \
{                                                                                                 \
    attn_flash_prefill_hd256_body<PrefillTileLoadF16, PrefillTileLoadF16, QKA, PVA>(              \
        q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads, position_offset, scale);         \
}

#define XACC_Q4Q4_ENTRY(NAME, QKA, PVA)                                                           \
extern "C" __global__ void __launch_bounds__(MAX_GROUP * 32, 1) NAME(                             \
    const half* __restrict__ q, const uint8_t* __restrict__ k, const uint8_t* __restrict__ v,     \
    float* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,           \
    const int num_kv_heads, const int position_offset, const float scale)                         \
{                                                                                                 \
    attn_flash_prefill_hd256_body<PrefillTileLoadStaged<V2LoadQ4>, PrefillTileLoadStaged<V2LoadQ4>, \
                                  QKA, PVA>(                                                      \
        q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads, position_offset, scale);         \
}

XACC_F16KV_ENTRY(attention_flash_mma_prefill_hd256_f16_qk16, ACC_F16, ACC_F32)
XACC_F16KV_ENTRY(attention_flash_mma_prefill_hd256_f16_pv16, ACC_F32, ACC_F16)
XACC_F16KV_ENTRY(attention_flash_mma_prefill_hd256_f16_pv16t, ACC_F32, ACC_F16_TILE)
XACC_F16KV_ENTRY(attention_flash_mma_prefill_hd256_f16_qk16pv16, ACC_F16, ACC_F16)

XACC_Q4Q4_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q4_qk16, ACC_F16, ACC_F32)
XACC_Q4Q4_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q4_pv16, ACC_F32, ACC_F16)
XACC_Q4Q4_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q4_pv16t, ACC_F32, ACC_F16_TILE)
XACC_Q4Q4_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q4_qk16pv16, ACC_F16, ACC_F16)
