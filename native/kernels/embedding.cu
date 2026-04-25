// Embedding lookup kernel for dotLLM.
// output[t] = embedTable[tokenIds[t]]
// Supports F32, F16, Q8_0, Q4_K, Q5_K, Q6_K source embedding tables → FP16 output.
//
// K-quant variants (Q4_K/Q5_K/Q6_K) dequantize the requested rows on the fly,
// avoiding the 1.16 GiB FP16 expansion of a 151k-vocab × 4096-hidden table at
// load. Math is bit-identical to the bulk dequant_q{4,5,6}_k_f16 kernels — same
// per-superblock arithmetic, just gated by token-id row selection.
//
// Required precondition: hidden_size must be a multiple of 256 (the K-quant
// super-block size). Caller (CudaWeights) must verify this before routing
// through these kernels and fall back to bulk-dequant otherwise.

#include <cuda_fp16.h>
#include <stdint.h>

// Q8_0 block: 2 bytes (half scale) + 32 bytes (int8 values) = 34 bytes, 32 elements
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

// K-quant super-block sizes (256 elements per super-block).
#define K_QUANT_SUPER_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144
#define Q5_K_BLOCK_BYTES 176
#define Q6_K_BLOCK_BYTES 210

extern "C" __global__ void __launch_bounds__(256) embedding_lookup_f32(
    const float* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    const float* row = embed_table + (size_t)token_id * hidden_size;
    half* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = __float2half(row[i]);
}

extern "C" __global__ void __launch_bounds__(256) embedding_lookup_f16(
    const half* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    const half* row = embed_table + (size_t)token_id * hidden_size;
    half* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = row[i];
}

extern "C" __global__ void __launch_bounds__(256) embedding_lookup_q8_0(
    const uint8_t* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    int blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    const uint8_t* row = embed_table + (size_t)token_id * blocks_per_row * Q8_0_BLOCK_BYTES;
    half* out_row = output + (size_t)t * hidden_size;

    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x)
    {
        const uint8_t* block = row + b * Q8_0_BLOCK_BYTES;
        // First 2 bytes: half scale
        half scale = *reinterpret_cast<const half*>(block);
        float d = __half2float(scale);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            out_row[b * Q8_0_BLOCK_SIZE + j] = __float2half(d * (float)qs[j]);
    }
}

// ── Per-row Q4_K embedding lookup ─────────────────────────────────────────
// Dequant math mirrors dequant_q4_k_f16 in dequant.cu (bit-identical).
// One CUDA block per token; 256 threads, each thread emits one element of
// each super-block in the row (grid-stride over super-blocks via b).
extern "C" __global__ void __launch_bounds__(256) embedding_lookup_q4_k(
    const uint8_t* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    int superblocks_per_row = hidden_size / K_QUANT_SUPER_BLOCK_SIZE;
    const uint8_t* row = embed_table +
        (size_t)token_id * superblocks_per_row * Q4_K_BLOCK_BYTES;
    half* out_row = output + (size_t)t * hidden_size;

    int tid = threadIdx.x; // 0..255 — one thread per element of a super-block

    // Pre-compute thread's intra-superblock indices once (loop-invariant)
    int pair = tid / 64;
    int pos_in_pair = tid % 64;
    int is_odd = pos_in_pair / 32;
    int j = pos_in_pair % 32;
    int sb_even = pair * 2;
    int sb_odd = pair * 2 + 1;
    int sb_cur = is_odd ? sb_odd : sb_even;

    for (int b = 0; b < superblocks_per_row; b++)
    {
        const uint8_t* block = row + b * Q4_K_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        int sc, m;
        if (sb_cur < 4)
        {
            sc = scales_raw[sb_cur] & 0x3F;
            m = scales_raw[sb_cur + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sb_cur + 4] & 0x0F) | ((scales_raw[sb_cur - 4] >> 6) << 4);
            m = (scales_raw[sb_cur + 4] >> 4) | ((scales_raw[sb_cur] >> 6) << 4);
        }

        uint8_t byte_val = qs[pair * 32 + j];
        int nibble = is_odd ? (byte_val >> 4) : (byte_val & 0x0F);

        float result = d * (float)sc * (float)nibble - dmin * (float)m;
        out_row[b * K_QUANT_SUPER_BLOCK_SIZE + tid] = __float2half(result);
    }
}

// ── Per-row Q5_K embedding lookup ─────────────────────────────────────────
// Dequant math mirrors dequant_q5_k_f16 in dequant.cu (bit-identical).
extern "C" __global__ void __launch_bounds__(256) embedding_lookup_q5_k(
    const uint8_t* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    int superblocks_per_row = hidden_size / K_QUANT_SUPER_BLOCK_SIZE;
    const uint8_t* row = embed_table +
        (size_t)token_id * superblocks_per_row * Q5_K_BLOCK_BYTES;
    half* out_row = output + (size_t)t * hidden_size;

    int tid = threadIdx.x; // 0..255

    int sub = tid / 32;
    int pos = tid % 32;
    int j = pos / 2;

    for (int b = 0; b < superblocks_per_row; b++)
    {
        const uint8_t* block = row + b * Q5_K_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        int sc, m;
        if (sub < 4)
        {
            sc = scales_raw[sub] & 0x3F;
            m = scales_raw[sub + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
            m = (scales_raw[sub + 4] >> 4) | ((scales_raw[sub] >> 6) << 4);
        }

        float scale = d * (float)sc;
        float min_val = dmin * (float)m;

        const uint8_t* sub_qs = qs + sub * 16;
        const uint8_t* sub_qh = qh + sub * 4;

        uint8_t packed = sub_qs[j];
        int nibble = (pos & 1) ? (packed >> 4) : (packed & 0x0F);
        int bit = (sub_qh[j / 4] >> ((j % 4) * 2 + (pos & 1))) & 1;
        int val = nibble | (bit << 4);

        out_row[b * K_QUANT_SUPER_BLOCK_SIZE + tid] = __float2half(scale * (float)val - min_val);
    }
}

// ── Per-row Q6_K embedding lookup ─────────────────────────────────────────
// Dequant math mirrors dequant_q6_k_f16 in dequant.cu (bit-identical).
extern "C" __global__ void __launch_bounds__(256) embedding_lookup_q6_k(
    const uint8_t* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    int superblocks_per_row = hidden_size / K_QUANT_SUPER_BLOCK_SIZE;
    const uint8_t* row = embed_table +
        (size_t)token_id * superblocks_per_row * Q6_K_BLOCK_BYTES;
    half* out_row = output + (size_t)t * hidden_size;

    int tid = threadIdx.x; // 0..255

    int half_idx = tid / 128;
    int pos_in_half = tid % 128;
    int group = pos_in_half / 32;
    int l = pos_in_half % 32;
    int isc = l / 16;

    for (int b = 0; b < superblocks_per_row; b++)
    {
        const uint8_t* block = row + b * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh_base = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh_base + half_idx * 32;
        const int8_t* sc_half = scales + half_idx * 8;

        int q_val;
        switch (group)
        {
            case 0:
                q_val = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
                break;
            case 1:
                q_val = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
                break;
            case 2:
                q_val = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32;
                break;
            default: // case 3
                q_val = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32;
                break;
        }

        float sc = d * (float)sc_half[isc + group * 2];
        out_row[b * K_QUANT_SUPER_BLOCK_SIZE + tid] = __float2half(sc * (float)q_val);
    }
}
