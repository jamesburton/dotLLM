// attention_flash_mma_prefill_hd256_kvq.cu -- issue #1094.
//
// Quantized-KV variants of attention_flash_mma_prefill_hd256_f16: the K/V history is read straight
// from KV-format Q8_0 (half d + 32 int8, 34 B / 32 elems) or Q4_0 (half d + 16 bytes of
// ADJACENT-PAIR nibbles -- the KV convention, not the GGUF weight layout; 18 B / 32 elems) blocks
// inside the K/V tile load, instead of first dequantizing the whole history [0, seq_kv) into an
// F16 scratch per attention layer per prefill tile.
//
// ─── How ────────────────────────────────────────────────────────────────────────────────
// The body is attention_flash_mma_prefill_hd256.cuh, shared verbatim with the F16 entry. Only the
// tile loader differs: each 16-byte chunk of the padded F16 shared-memory K/V tile is produced by
// the block decoders of the KVQ decode kernel (V2LoadQ8 / V2LoadQ4 in attention_mma_decode_v2.cuh,
// reused, not duplicated) and written with one 16-byte st.shared. Those decoders produce each
// element with the same expression as the eager dequant kernels, __float2half(d * (float)q), so the
// shared-memory tile holds exactly the halves the F16 kernel would read from a dequantized scratch
// and the output is bit-identical to dequantize-into-F16 + attention_flash_mma_prefill_hd256_f16.
//
// The dequantized values exist only in the (already allocated) per-block shared-memory K/V tile:
// no global scratch and no extra shared memory (same layout, size and 99 KB cap as the F16
// kernel). The loads are synchronous (the bytes must pass through registers to be decoded); the
// double-buffer ordering is unchanged -- tile t+1 is written into the other buffer before tile t
// is consumed, the body's cp.async commit / wait are no-ops here and its __syncthreads publish it.
//
// ─── Cost (RTX 3060, seqQ 512, issue #1094 bench) ───────────────────────────────────────
// ~0.72x the speed of dequant-into-scratch + the F16 kernel at seqKv 64k (~0.78x at 4k). The cost
// is the five 2-byte-aligned global loads per 8-element chunk (Q8_0 blocks are 34 B, so wider loads
// are misaligned) competing with the MMA pipeline -- not latency, decode ALU or address math:
// register prefetch of tile t+1, packed-half2 decode and hoisted row addressing were each measured
// and made no difference, while one aligned 16-byte load per chunk would recover most of the gap.
//
// Entries (same launch geometry / arguments as the F16 kernel; k and v are raw block bytes):
//   attention_flash_mma_prefill_hd256_kvq_q8q8 / _q4q4 / _q8q4 (K Q8_0, V Q4_0) / _q4q8

#include "attention_flash_mma_prefill_hd256.cuh"
#include "attention_mma_decode_v2.cuh"   // V2LoadQ8 / V2LoadQ4 block decoders (issue #988)

// Adapts a decode-v2 loader (load8: 8 consecutive halves at an 8-aligned element) to the prefill
// body's tile-chunk policy: decode in registers, one 16-byte shared-memory store.
template <class L>
struct PrefillTileLoadDequant
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        *reinterpret_cast<uint4*>(dst) = L::load8(base, elem);
    }
};

#define KVQ_PREFILL_ENTRY(NAME, KLOAD, VLOAD)                                                     \
extern "C" __global__ void __launch_bounds__(MAX_GROUP * 32, 1) NAME(                             \
    const half* __restrict__ q, const uint8_t* __restrict__ k, const uint8_t* __restrict__ v,     \
    float* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,           \
    const int num_kv_heads, const int position_offset, const float scale)                         \
{                                                                                                 \
    attn_flash_prefill_hd256_body<PrefillTileLoadDequant<KLOAD>, PrefillTileLoadDequant<VLOAD>>(  \
        q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads, position_offset, scale);         \
}

KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q8q8, V2LoadQ8, V2LoadQ8)
KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q4, V2LoadQ4, V2LoadQ4)
KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q8q4, V2LoadQ8, V2LoadQ4)
KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q8, V2LoadQ4, V2LoadQ8)
