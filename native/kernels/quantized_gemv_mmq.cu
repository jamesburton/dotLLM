// MMQ-style fused dequant+matmul GEMV kernels.
// Input activation x is quantized on-the-fly to INT8 with a per-32-element FP16 scale,
// matching the Q4_K sub-block layout. Dot products use __dp4a (4× INT8×INT8 multiply-add
// per cycle on sm_61+) instead of the FP fmuladd path used by quantized_gemv.cu.
//
// Q4_K math:
//   w[i]   = d * sc_s * q[i] - dmin * m_s     for sub-block s, q[i] ∈ [0,15]
//   xq[i]  = round(x[i] / dx_c * 127)         for input chunk c (32 elements per chunk)
//   x[i]   ≈ dx_c * xq[i]
//   dot_s  = Σ_i w[i]*x[j_i]
//          ≈ dx_c * (d*sc_s * Σ_i q[i]*xq[j_i]   -   dmin*m_s * Σ_i xq[j_i])
//
// The Σ q[i]*xq[j_i] sum is the dp4a accumulator (8 dp4a calls per 32-element sub-block).
// Σ xq[j_i] is precomputed once per chunk during the input-quantization pass.
//
// Tile parallelism: one CUDA block processes MMQ_ROWS_PER_BLOCK output rows. All rows in
// the block share the input-quantization pass, so for small models like SmolLM-135M
// (k=576, 2 superblocks per row) the dp4a phase across MMQ_ROWS_PER_BLOCK*2 superblocks
// fans out across BlockSize=256 threads instead of leaving most idle.

#include <cuda_fp16.h>
#include <stdint.h>

// Per-chunk Stage 1 scratch lives in dynamic shared memory, sized at launch from `k`.
// Layout (one extern __shared__ block, host passes `sharedMemBytes` to cuLaunchKernel):
//   int8_t  s_xq [num_chunks * 32]                   // INT8-quantized x bytes
//   half    s_dx [num_chunks]                        // per-chunk FP16 scale
//   half    s_sx [num_chunks]   (Q4_K, Q5_K)         // Σ xq per full chunk
//   half    s_sx2[num_chunks * 2] (Q6_K)             // Σ xq per half-chunk (lo, hi)
// The static __shared__ regions (s_acc, s_warp_partials) sit alongside; the dynamic
// region starts after them. With 32-element chunks, num_chunks*32 is always multiple
// of 32 so the half pointers downstream stay naturally aligned.
//
// Limits: dynamic shmem caps at the device's MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
// (≥ 100 KB on sm_86 / RTX 3060). The host side calls cuFuncSetAttribute once
// per kernel to opt in, so any k that fits in that budget works at runtime.
//   - SmolLM-135M k=576    → 18 chunks  → ~660 bytes
//   - Qwen3-8B   k=12288   → 384 chunks → ~13.5 KB (Q4_K/Q5_K), ~14.2 KB (Q6_K)
//   - Llama-70B  k=14336   → 448 chunks → ~15.7 KB
//   - Llama-405B k=53248   → 1664 chunks → ~58.4 KB (still under 100 KB optin)

// Output rows per CUDA block. 4 rows × 2 superblocks/row (for SmolLM-135M) = 8 superblocks
// distributed across 256 threads → 32 superblocks per warp before grouping. Larger models
// with k≥1024 (≥4 superblocks/row) become work-saturated regardless.
#define MMQ_ROWS_PER_BLOCK 4

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q4_k_mmq(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;            // 8 chunks per super-block

    // Dynamic shmem: [s_xq | s_dx | s_sx]. Sized by the host as
    // num_chunks * (32 + 2 + 2) bytes. Static s_acc lives separately.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx = s_dx + num_chunks;

    // Per-row partial sums staged in shared memory between Stage 2 and Stage 3.
    // Each row's dp4a contributions are scattered across all threads (whichever
    // thread happens to own that row+superblock pair); we sum them with a single
    // cross-block reduction at the end.
    __shared__ float s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // ── Stage 1: quantize x to INT8 per 32-element chunk ────────────────
    for (int c = warp_id; c < num_chunks; c += num_warps)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        int s = qi;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx[c] = __float2half((float)s);
        }
    }

    // Zero the per-row accumulator scratch in parallel with the input-quant
    // tail (whichever finishes last, the syncthreads below pairs them).
    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    // ── Stage 2: dp4a accumulation across (rows × superblocks) ──────────
    // Map each thread to a (row_in_block, superblock) work unit. Total work
    // = rows_in_block * superblocks_per_row. Distribute round-robin across
    // blockDim threads via stride loop.
    const int total_units = rows_in_block * superblocks_per_row;

    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;
        const uint8_t* block = w_row + sb * 144;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        float row_acc = 0.0f;

        #pragma unroll
        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd  = pair * 2 + 1;

            int sc0, m0, sc1, m1;
            if (sb_even < 4)
            {
                sc0 = scales_raw[sb_even]     & 0x3F;
                m0  = scales_raw[sb_even + 4] & 0x3F;
                sc1 = scales_raw[sb_odd]      & 0x3F;
                m1  = scales_raw[sb_odd + 4]  & 0x3F;
            }
            else
            {
                sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
                m0  = (scales_raw[sb_even + 4] >> 4)   | ((scales_raw[sb_even]     >> 6) << 4);
                sc1 = (scales_raw[sb_odd + 4]  & 0x0F) | ((scales_raw[sb_odd - 4]  >> 6) << 4);
                m1  = (scales_raw[sb_odd + 4]  >> 4)   | ((scales_raw[sb_odd]      >> 6) << 4);
            }

            const uint8_t* pair_qs = qs + pair * 32;
            int chunk_even = sb * 8 + sb_even;
            int chunk_odd  = sb * 8 + sb_odd;

            const int8_t* xq_even = s_xq + chunk_even * 32;
            const int8_t* xq_odd  = s_xq + chunk_odd  * 32;

            int dot0 = 0;
            int dot1 = 0;

            #pragma unroll
            for (int g = 0; g < 8; g++)
            {
                uint32_t qpacked = *reinterpret_cast<const uint32_t*>(pair_qs + g * 4);
                int lo = (int)(qpacked & 0x0F0F0F0F);
                int hi = (int)((qpacked >> 4) & 0x0F0F0F0F);

                int xq_e_packed = *reinterpret_cast<const int*>(xq_even + g * 4);
                int xq_o_packed = *reinterpret_cast<const int*>(xq_odd  + g * 4);

                dot0 = __dp4a(lo, xq_e_packed, dot0);
                dot1 = __dp4a(hi, xq_o_packed, dot1);
            }

            float dx_e = __half2float(s_dx[chunk_even]);
            float dx_o = __half2float(s_dx[chunk_odd]);
            float sx_e = __half2float(s_sx[chunk_even]);
            float sx_o = __half2float(s_sx[chunk_odd]);

            row_acc += dx_e * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * sx_e);
            row_acc += dx_o * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * sx_o);
        }

        // Stage to shared memory at this thread's per-row slot. Threads working on
        // the same row but different superblocks land at distinct tid offsets, so
        // no atomics needed.
        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    // ── Stage 3: per-row reduction. One warp per row; each lane sums 256/32 = 8
    // of the row's slots, then a warp-shfl sums the lanes. With MMQ_ROWS_PER_BLOCK=4
    // we use 4 warps (of 8 total); the remaining warps are idle in this stage.
    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        // Each lane reads 8 of the 256 slots: lane 0 reads slots 0,32,64,...,224.
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Q2_K MMQ — 84 bytes per 256 values
//
// Q2_K math (per sub-block s, 16 elements each — 16 sub-blocks per super-block):
//   w[i]   = d * sc_s * q[i] - dmin * dm_s     for q[i] ∈ [0, 3]  (2-bit)
//   xq[i]  = round(x[i] / dx_c * 127)          for input chunk c (32 elements)
//   x[i]   ≈ dx_c * xq[i]
//   dot_s  ≈ dx_c * (d*sc_s * Σ q[i]*xq[j_i]   -   dmin*dm_s * Σ xq[j_i])
//
// Block layout:
//   scales[16] at offset 0  — one byte per sub-block: low nibble = sc, high nibble = dm
//   qs[64]     at offset 16 — 2 bits/element, 4 elements/byte; sub_block s spans
//                             qs[s*4 + 0..3] (16 elements as 4 bytes)
//   d          at offset 80 — half super-block delta
//   dmin       at offset 82 — half super-block min delta
//
// Tile geometry: each chunk (32 input elements) covers TWO Q2_K sub-blocks
// (16 + 16 elements). Same chunk-vs-sub-block ratio as Q6_K, so we reuse Q6_K's
// half-chunk-sum scheme: s_sx2[c*2 + 0] = Σ xq[c*32..c*32+15],
//                       s_sx2[c*2 + 1] = Σ xq[c*32+16..c*32+31].
//
// Per super-block: 8 chunks (q_quad ∈ [0, 8)) × 2 sub-blocks/chunk = 16 sub-blocks.
// dp4a inner loop: 4 calls per sub-block (16 elements / 4 elements per dp4a).
// Weight bytes are pre-multiplied by sc into INT8 dp4a-friendly form? No — we
// keep q ∈ [0,3] in the dp4a accumulator and apply sc as an FP multiplier post-dot
// (matches Q5_K MMQ; q small enough that even 16-element dot fits int32 trivially).

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q2_k_mmq(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;            // 8 chunks per super-block

    // Dynamic shmem: [s_xq | s_dx | s_sx2]. s_sx2 is 2 halves per chunk (one per
    // half-chunk = one per Q2_K sub-block), so dynamic budget is num_chunks*(32+2+4) bytes.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq  = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx  = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx2 = s_dx + num_chunks;        // [a, b] per chunk

    __shared__ float s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // ── Stage 1: input quantization with per-half-chunk sums (mirrors Q6_K MMQ) ──
    // dx is per-32-element chunk. sx is split: sx_a = Σ xq[0..15], sx_b = Σ xq[16..31].
    // Half-warp reductions: stop at offset=8 so lanes 0..15 and 16..31 stay isolated.
    for (int c = warp_id; c < num_chunks; c += num_warps)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        int s = qi;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx2[c * 2 + 0] = __float2half((float)s);
        }
        if (lane == 16)
        {
            s_sx2[c * 2 + 1] = __float2half((float)s);
        }
    }

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    // ── Stage 2: dp4a accumulation (one work unit per row × superblock) ────
    const int total_units = rows_in_block * superblocks_per_row;

    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 84;
        const uint8_t* block = w_row + sb * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        float d    = __half2float(*reinterpret_cast<const half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 82));

        float row_acc = 0.0f;

        // 8 chunks per super-block, each chunk covers 2 sub-blocks (isc=0,1).
        #pragma unroll
        for (int q_quad = 0; q_quad < 8; q_quad++)
        {
            int chunk_idx = sb * 8 + q_quad;
            const uint8_t* chunk_qs = qs + q_quad * 8;   // 8 bytes = 2 sub-blocks of 4 bytes

            #pragma unroll
            for (int isc = 0; isc < 2; isc++)
            {
                int sub = q_quad * 2 + isc;              // sub-block index 0..15
                int sc  = scales[sub] & 0x0F;            // 4-bit scale
                int dm  = (scales[sub] >> 4) & 0x0F;     // 4-bit dmin coef

                const uint8_t* sub_qs = chunk_qs + isc * 4;   // 4 bytes for this sub-block
                int l_base = isc * 16;
                const int8_t* xq = s_xq + chunk_idx * 32 + l_base;

                int dot = 0;
                // 16 elements / 4 per dp4a = 4 dp4a calls. Each byte qs[g] holds 4
                // consecutive 2-bit values at bit offsets 0,2,4,6.
                #pragma unroll
                for (int g = 0; g < 4; g++)
                {
                    uint32_t qbyte = sub_qs[g];
                    int q0 = (int)((qbyte >> 0) & 0x3);
                    int q1 = (int)((qbyte >> 2) & 0x3);
                    int q2v = (int)((qbyte >> 4) & 0x3);
                    int q3 = (int)((qbyte >> 6) & 0x3);
                    int wpack = (q0 & 0xFF)
                              | ((q1 & 0xFF) << 8)
                              | ((q2v & 0xFF) << 16)
                              | ((q3 & 0xFF) << 24);
                    int xpack = *reinterpret_cast<const int*>(xq + 4 * g);
                    dot = __dp4a(wpack, xpack, dot);
                }

                float dx = __half2float(s_dx[chunk_idx]);
                float sx = __half2float(s_sx2[chunk_idx * 2 + isc]);

                // Per-element identity:
                //   Σ (d*sc*q - dmin*dm) * x  ≈  dx * (d*sc * Σ q*xq  -  dmin*dm * Σ xq)
                row_acc += dx * (d * (float)sc * (float)dot - dmin * (float)dm * sx);
            }
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    // ── Stage 3: per-row reduction (same shape as Q4_K/Q5_K/Q6_K MMQ) ────
    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Q5_K MMQ — 176 bytes per 256 values
//
// Q5_K math (per sub-block s, 32 elements each):
//   w[i]   = d * sc_s * q[i] - dmin * m_s     for q[i] ∈ [0, 31]  (5-bit)
//   xq[i]  = round(x[i] / dx_c * 127)         for input chunk c (32 elements)
//   x[i]   ≈ dx_c * xq[i]
//   dot_s  ≈ dx_c * (d*sc_s * Σ q[i]*xq[j_i]   -   dmin*m_s * Σ xq[j_i])
//
// Element layout within a sub-block (matches dequant order):
//   out[2j]   = (qs[j] & 0x0F) | (qh_bit_lo[j] << 4)   for j ∈ [0, 16)
//   out[2j+1] = (qs[j] >> 4)   | (qh_bit_hi[j] << 4)
// That is, sub_out[0..31] = [lo_0, hi_0, lo_1, hi_1, ..., lo_15, hi_15].
//
// dp4a chunk packing: emit weight bytes in input memory order so they line up
// with s_xq[chunk_idx*32 + 4g .. 4g+3] = x bytes 4g..4g+3 within the chunk.
// For chunk g (0..7), we cover j = 2g and j = 2g+1; pack [lo_e, hi_e, lo_o, hi_o].
//
// Sub-block s ∈ [0, 8) maps 1:1 to chunk c = sb*8 + s, same as Q4_K MMQ — so
// the input-quantization pass shared with Q4_K MMQ produces the right s_dx/s_sx.

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q5_k_mmq(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;

    // Dynamic shmem: [s_xq | s_dx | s_sx]. See Q4_K MMQ above for layout.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx = s_dx + num_chunks;

    __shared__ float   s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // ── Stage 1: input quantization (identical to Q4_K MMQ) ────────────────
    for (int c = warp_id; c < num_chunks; c += num_warps)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        int s = qi;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx[c] = __float2half((float)s);
        }
    }

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    // ── Stage 2: dp4a accumulation (one work unit per row × superblock) ────
    const int total_units = rows_in_block * superblocks_per_row;

    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 176;
        const uint8_t* block = w_row + sb * 176;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;   // 32 bytes
        const uint8_t* qs = block + 48;   // 128 bytes

        float row_acc = 0.0f;

        // 8 sub-blocks of 32 elements each
        #pragma unroll
        for (int sub = 0; sub < 8; sub++)
        {
            int sc, m;
            if (sub < 4)
            {
                sc = scales_raw[sub] & 0x3F;
                m  = scales_raw[sub + 4] & 0x3F;
            }
            else
            {
                sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
                m  = (scales_raw[sub + 4] >> 4)   | ((scales_raw[sub]     >> 6) << 4);
            }

            const uint8_t* sub_qs = qs + sub * 16;
            const uint8_t* sub_qh = qh + sub * 4;
            int chunk_idx = sb * 8 + sub;
            const int8_t* xq = s_xq + chunk_idx * 32;

            int dot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
            {
                // Cover j=2g (even) and j=2g+1 (odd); pack [lo_e, hi_e, lo_o, hi_o].
                int j_e = 2 * g;
                int j_o = 2 * g + 1;
                int p_e = sub_qs[j_e];
                int p_o = sub_qs[j_o];
                int qhb_e = sub_qh[j_e >> 2];   // = sub_qh[g >> 1]
                int qhb_o = sub_qh[j_o >> 2];   // = sub_qh[g >> 1]

                int blo_e = (qhb_e >> ((j_e & 3) * 2)) & 1;
                int bhi_e = (qhb_e >> ((j_e & 3) * 2 + 1)) & 1;
                int blo_o = (qhb_o >> ((j_o & 3) * 2)) & 1;
                int bhi_o = (qhb_o >> ((j_o & 3) * 2 + 1)) & 1;

                int lo_e = (p_e & 0x0F) | (blo_e << 4);
                int hi_e = (p_e >> 4)   | (bhi_e << 4);
                int lo_o = (p_o & 0x0F) | (blo_o << 4);
                int hi_o = (p_o >> 4)   | (bhi_o << 4);

                // Pack 4 INT8 weight values; values ∈ [0, 31] fit in signed INT8.
                int wpack = (lo_e & 0xFF) | ((hi_e & 0xFF) << 8)
                          | ((lo_o & 0xFF) << 16) | ((hi_o & 0xFF) << 24);
                int xpack = *reinterpret_cast<const int*>(xq + 4 * g);

                dot = __dp4a(wpack, xpack, dot);
            }

            float dx = __half2float(s_dx[chunk_idx]);
            float sx = __half2float(s_sx[chunk_idx]);
            row_acc += dx * (d * (float)sc * (float)dot - dmin * (float)m * sx);
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    // ── Stage 3: per-row reduction (same shape as Q4_K MMQ) ────────────────
    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Q6_K MMQ — 210 bytes per 256 values
//
// Q6_K math (per sub-block s, 16 elements each, 16 sub-blocks per superblock):
//   w[i]   = d * sc_s * (q[i] - 32)            for q[i] ∈ [0, 63]  (6-bit)
//   x[i]   ≈ dx_c * xq[i]
//   dot_s  ≈ dx_c * (d*sc_s * Σ q[i]*xq[j_i]   -   32 * d*sc_s * Σ xq[j_i])
//
// Q6_K has NO dmin term — much simpler than Q4_K/Q5_K.
//
// Sub-block size is 16 (vs 32 for Q4_K/Q5_K), so each 32-element input chunk
// holds TWO Q6_K sub-blocks. Their scales differ, so we need separate sums of
// x over the two halves of the chunk: sx_a (lanes 0..15) and sx_b (lanes 16..31).
// We store both in s_sx2[c*2 + 0..1].
//
// Element layout within a half-superblock (128 elements covered by ql_half[64]
// + qh_half[32] + sc_half[8]):
//   q1[l] = (ql_half[l]      & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)   → x[+0..31]
//   q2[l] = (ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)   → x[+32..63]
//   q3[l] = (ql_half[l]      >> 4)   | (((qh_half[l] >> 4) & 3) << 4)   → x[+64..95]
//   q4[l] = (ql_half[l + 32] >> 4)   | (((qh_half[l] >> 6) & 3) << 4)   → x[+96..127]
// Each q_quad ∈ {0..3} → ql_offset = (qq & 1) * 32, ql_shift = (qq & 2) << 1,
// qh_shift = qq << 1. Sub-block scale = sc_half[q_quad*2 + (l < 16 ? 0 : 1)].

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q6_k_mmq(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;

    // Dynamic shmem: [s_xq | s_dx | s_sx2]. s_sx2 is 2 halves per chunk
    // (one per half-chunk) instead of 1, so the dynamic budget here is
    // num_chunks * (32 + 2 + 4) bytes.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq  = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx  = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx2 = s_dx + num_chunks;            // [a, b] per chunk

    __shared__ float   s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // ── Stage 1: input quantization with per-half-chunk sums ───────────────
    // dx is per-32-element chunk. sx is split: sx_a = Σ xq[0..15], sx_b = Σ xq[16..31].
    // Half-warp reductions: omit the offset=16 step so each lane in 0..15 holds
    // the sum of x[chunk_base + 0..15] and each lane in 16..31 holds the sum
    // of x[chunk_base + 16..31] independently.
    for (int c = warp_id; c < num_chunks; c += num_warps)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        // Full warp max for dx (chunk-wide scale).
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        // Half-warp sum: stop at offset=8 so lanes 0..15 and 16..31 stay isolated.
        int s = qi;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx2[c * 2 + 0] = __float2half((float)s);
        }
        if (lane == 16)
        {
            s_sx2[c * 2 + 1] = __float2half((float)s);
        }
    }

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    // ── Stage 2: dp4a accumulation ─────────────────────────────────────────
    // One work unit per (row, superblock); each unit walks 16 sub-blocks
    // (2 halves × 4 q_quads × 2 sub-blocks-per-quad).
    const int total_units = rows_in_block * superblocks_per_row;

    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 210;
        const uint8_t* block = w_row + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        float row_acc = 0.0f;

        #pragma unroll
        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            const uint8_t* ql_half = ql + half_idx * 64;
            const uint8_t* qh_half = qh + half_idx * 32;
            const int8_t*  sc_half = scales + half_idx * 8;

            #pragma unroll
            for (int q_quad = 0; q_quad < 4; q_quad++)
            {
                int ql_offset = (q_quad & 1) * 32;
                int ql_shift  = (q_quad & 2) << 1;   // 0 or 4
                int qh_shift  = q_quad << 1;         // 0,2,4,6

                int chunk_idx = sb * 8 + half_idx * 4 + q_quad;

                // Two sub-blocks per (half, q_quad) — l ∈ [0,16) and l ∈ [16,32).
                #pragma unroll
                for (int isc = 0; isc < 2; isc++)
                {
                    int sc = sc_half[q_quad * 2 + isc];

                    int dot = 0;
                    int l_base = isc * 16;
                    const int8_t* xq = s_xq + chunk_idx * 32 + l_base;

                    #pragma unroll
                    for (int g = 0; g < 4; g++)
                    {
                        int wpack = 0;
                        #pragma unroll
                        for (int b = 0; b < 4; b++)
                        {
                            int l = l_base + g * 4 + b;
                            int q = ((ql_half[l + ql_offset] >> ql_shift) & 0x0F)
                                  | (((qh_half[l] >> qh_shift) & 3) << 4);
                            wpack |= (q & 0xFF) << (b * 8);
                        }
                        int xpack = *reinterpret_cast<const int*>(xq + 4 * g);
                        dot = __dp4a(wpack, xpack, dot);
                    }

                    float dx = __half2float(s_dx[chunk_idx]);
                    float sx = __half2float(s_sx2[chunk_idx * 2 + isc]);

                    // d * sc * (Σ q*xq - 32 * Σ xq) scaled by chunk dx.
                    row_acc += dx * d * (float)sc * ((float)dot - 32.0f * sx);
                }
            }
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    // ── Stage 3: per-row reduction ─────────────────────────────────────────
    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MMVQ-large variants — 1 row per CUDA block, 128 threads (4 warps).
//
// Tuned for large-k regime (k >= 1024, ≥4 super-blocks per row). Modeled on
// llama.cpp's mul_mat_vec_q<Q4_K, 1, false> structure: each block computes
// exactly one output row's dot product, all 128 threads cooperate on the same
// accumulator (kept in registers, no per-row shmem), final reduction is a
// per-warp __shfl_xor_sync sum followed by a small 4-warp fan-in via shmem.
//
// The 4-rows-per-block kernels above remain the right choice for small k
// (≤3 super-blocks/row, e.g. SmolLM-135M k=576) where the input-quantization
// amortization across rows is the dominant win and there's not enough work
// per row to saturate 128 threads.
//
// Stage 1 (input quant) is structurally identical to the 4-row variant: 4
// warps walk all num_chunks chunks via warp-stride, producing s_xq/s_dx/s_sx
// (or s_sx2 for Q6_K) in shared memory.
//
// Stage 2 distributes one row's dot-product work across all 128 threads using
// a "g-cell" decomposition: each (super-block, sub-block-or-pair, g) triple is
// a single dp4a-pair work unit; each thread accumulates partial dp4a results
// into a register float `acc`. The mins/origin term (which is constant per
// (sb, sub) and would over-count if added per-cell) is added by exactly one
// designated cell (g==0) per (sb, sub) — the rest contribute only the dot
// product.
//
// Stage 3: warp-shuffle reduction within each warp, 4 partials staged into
// __shared__ float[4], one final reduction in warp 0, write y[row] from
// thread (0, 0).

// 4 warps × 32 = 128 threads/block, mirroring llama.cpp's mul_mat_vec_q for ncols_dst=1.
// Smaller blocks give more concurrent blocks per SM (occupancy-vs-per-block-work tradeoff
// favors small blocks for GEMV where each row is shallow on its own).
#define MMVQ_LARGE_THREADS 128
#define MMVQ_LARGE_NWARPS  4

extern "C" __global__ void __launch_bounds__(MMVQ_LARGE_THREADS) quantized_gemv_q4_k_mmvq_large(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;

    // Dynamic shmem: [s_xq | s_dx | s_sx]. See Q4_K MMQ above for layout.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx = s_dx + num_chunks;

    __shared__ float   s_warp_partials[MMVQ_LARGE_NWARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // ── Stage 1: quantize x to INT8 per 32-element chunk ────────────────
    for (int c = warp_id; c < num_chunks; c += MMVQ_LARGE_NWARPS)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        int s = qi;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx[c] = __float2half((float)s);
        }
    }

    __syncthreads();

    // ── Stage 2: dp4a accumulation across (sb, pair) units ───────────────
    // Each (super-block, pair) is one work unit covering 8 g-cells (16 dp4a)
    // with constant per-pair scales sc0/m0/sc1/m1. Total units per row =
    // superblocks_per_row * 4. For k=4096 → 64 units; k=12288 (Qwen3-8B
    // MlpDown) → 192 units. Stride loop over 128 threads.
    const int total_units = superblocks_per_row * 4;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;
    float acc = 0.0f;

    for (int unit = tid; unit < total_units; unit += MMVQ_LARGE_THREADS)
    {
        int sb = unit >> 2;            // unit / 4
        int pair = unit & 3;           // 0..3

        const uint8_t* block = w_row + sb * 144;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        int sb_even = pair * 2;
        int sb_odd  = pair * 2 + 1;
        int sc0, m0, sc1, m1;
        if (sb_even < 4)
        {
            sc0 = scales_raw[sb_even]     & 0x3F;
            m0  = scales_raw[sb_even + 4] & 0x3F;
            sc1 = scales_raw[sb_odd]      & 0x3F;
            m1  = scales_raw[sb_odd + 4]  & 0x3F;
        }
        else
        {
            sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
            m0  = (scales_raw[sb_even + 4] >> 4)   | ((scales_raw[sb_even]     >> 6) << 4);
            sc1 = (scales_raw[sb_odd + 4]  & 0x0F) | ((scales_raw[sb_odd - 4]  >> 6) << 4);
            m1  = (scales_raw[sb_odd + 4]  >> 4)   | ((scales_raw[sb_odd]      >> 6) << 4);
        }

        const uint8_t* pair_qs = qs + pair * 32;
        int chunk_even = sb * 8 + sb_even;
        int chunk_odd  = sb * 8 + sb_odd;

        const int8_t* xq_even = s_xq + chunk_even * 32;
        const int8_t* xq_odd  = s_xq + chunk_odd  * 32;

        int dot0 = 0;
        int dot1 = 0;

        #pragma unroll
        for (int g = 0; g < 8; g++)
        {
            uint32_t qpacked = *reinterpret_cast<const uint32_t*>(pair_qs + g * 4);
            int lo = (int)(qpacked & 0x0F0F0F0F);
            int hi = (int)((qpacked >> 4) & 0x0F0F0F0F);

            int xq_e_packed = *reinterpret_cast<const int*>(xq_even + g * 4);
            int xq_o_packed = *reinterpret_cast<const int*>(xq_odd  + g * 4);

            dot0 = __dp4a(lo, xq_e_packed, dot0);
            dot1 = __dp4a(hi, xq_o_packed, dot1);
        }

        float dx_e = __half2float(s_dx[chunk_even]);
        float dx_o = __half2float(s_dx[chunk_odd]);
        float sx_e = __half2float(s_sx[chunk_even]);
        float sx_o = __half2float(s_sx[chunk_odd]);

        acc += dx_e * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * sx_e);
        acc += dx_o * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * sx_o);
    }

    // ── Stage 3: warp-shfl reduction → 4-warp shmem fan-in → final sum ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        s_warp_partials[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0)
    {
        float v = (lane < MMVQ_LARGE_NWARPS) ? s_warp_partials[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            y[row] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Q5_K MMVQ-large — same 1-row-per-block structure, Q5_K weight decode.
//
// Cell layout per super-block: 8 sub-blocks × 8 g = 64 cells/sb.
// Each sub-block = one 32-element chunk with its own scale and 5-bit quants.
// For k=4096 → 16 sb × 64 = 1024 cells / 128 threads = 8 cells per thread.

extern "C" __global__ void __launch_bounds__(MMVQ_LARGE_THREADS) quantized_gemv_q5_k_mmvq_large(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;

    // Dynamic shmem: [s_xq | s_dx | s_sx]. See Q4_K MMQ above for layout.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx = s_dx + num_chunks;

    __shared__ float   s_warp_partials[MMVQ_LARGE_NWARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // ── Stage 1: input quantization (identical to Q4_K MMVQ-large) ─────
    for (int c = warp_id; c < num_chunks; c += MMVQ_LARGE_NWARPS)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        int s = qi;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx[c] = __float2half((float)s);
        }
    }

    __syncthreads();

    // ── Stage 2: dp4a accumulation across (sb, sub) units ──────────────
    // Each (super-block, sub-block) is one work unit (8 g-cells, 8 dp4a, with
    // constant per-sub scales sc/m). Total units per row = superblocks_per_row * 8.
    // For k=4096 → 128 units exactly = 1 unit per thread; for k=12288 → 384 units = 3 each.
    const int total_units = superblocks_per_row * 8;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 176;
    float acc = 0.0f;

    for (int unit = tid; unit < total_units; unit += MMVQ_LARGE_THREADS)
    {
        int sb = unit >> 3;            // unit / 8
        int sub = unit & 7;            // 0..7

        const uint8_t* block = w_row + sb * 176;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        int sc, m;
        if (sub < 4)
        {
            sc = scales_raw[sub] & 0x3F;
            m  = scales_raw[sub + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
            m  = (scales_raw[sub + 4] >> 4)   | ((scales_raw[sub]     >> 6) << 4);
        }

        const uint8_t* sub_qs = qs + sub * 16;
        const uint8_t* sub_qh = qh + sub * 4;
        int chunk_idx = sb * 8 + sub;
        const int8_t* xq = s_xq + chunk_idx * 32;

        int dot = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++)
        {
            int j_e = 2 * g;
            int j_o = 2 * g + 1;
            int p_e = sub_qs[j_e];
            int p_o = sub_qs[j_o];
            int qhb_e = sub_qh[j_e >> 2];
            int qhb_o = sub_qh[j_o >> 2];

            int blo_e = (qhb_e >> ((j_e & 3) * 2)) & 1;
            int bhi_e = (qhb_e >> ((j_e & 3) * 2 + 1)) & 1;
            int blo_o = (qhb_o >> ((j_o & 3) * 2)) & 1;
            int bhi_o = (qhb_o >> ((j_o & 3) * 2 + 1)) & 1;

            int lo_e = (p_e & 0x0F) | (blo_e << 4);
            int hi_e = (p_e >> 4)   | (bhi_e << 4);
            int lo_o = (p_o & 0x0F) | (blo_o << 4);
            int hi_o = (p_o >> 4)   | (bhi_o << 4);

            int wpack = (lo_e & 0xFF) | ((hi_e & 0xFF) << 8)
                      | ((lo_o & 0xFF) << 16) | ((hi_o & 0xFF) << 24);
            int xpack = *reinterpret_cast<const int*>(xq + 4 * g);

            dot = __dp4a(wpack, xpack, dot);
        }

        float dx = __half2float(s_dx[chunk_idx]);
        float sx = __half2float(s_sx[chunk_idx]);
        acc += dx * (d * (float)sc * (float)dot - dmin * (float)m * sx);
    }

    // ── Stage 3: warp-shfl reduction → 4-warp shmem fan-in → final sum ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        s_warp_partials[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0)
    {
        float v = (lane < MMVQ_LARGE_NWARPS) ? s_warp_partials[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            y[row] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Q6_K MMVQ-large — same 1-row-per-block structure, Q6_K weight decode.
//
// Q6_K has 16 sub-blocks per super-block (16 elements each, vs 32 for Q4/5_K),
// no dmin term, and per-half-chunk x sums. The cell decomposition unrolls the
// 4-row kernel's nested (half, q_quad, isc) loop into a flat (sb, slot, g)
// indexing where slot ∈ [0, 16) covers the 16 sub-blocks (2 halves × 4 q_quads
// × 2 isc) and g ∈ [0, 4) covers the 4 dp4a calls per sub-block.
//
// Cell layout per super-block: 16 sub-blocks × 4 g = 64 cells/sb.
// For k=4096 → 16 sb × 64 = 1024 cells / 128 threads = 8 cells per thread.

extern "C" __global__ void __launch_bounds__(MMVQ_LARGE_THREADS) quantized_gemv_q6_k_mmvq_large(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k / 32;

    // Dynamic shmem: [s_xq | s_dx | s_sx2]. See Q6_K MMQ above for layout.
    extern __shared__ uint8_t s_dyn[];
    int8_t* s_xq  = reinterpret_cast<int8_t*>(s_dyn);
    half*   s_dx  = reinterpret_cast<half*>(s_xq + num_chunks * 32);
    half*   s_sx2 = s_dx + num_chunks;            // [a, b] per chunk

    __shared__ float   s_warp_partials[MMVQ_LARGE_NWARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // ── Stage 1: input quantization with per-half-chunk sums ───────────
    for (int c = warp_id; c < num_chunks; c += MMVQ_LARGE_NWARPS)
    {
        float v = __half2float(x[c * 32 + lane]);
        float a = fabsf(v);

        // Full warp max for dx (chunk-wide scale).
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            a = fmaxf(a, other);
        }

        float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
        int qi = __float2int_rn(v * inv_scale);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        s_xq[c * 32 + lane] = (int8_t)qi;

        // Half-warp sum: stop at offset=8 so lanes 0..15 and 16..31 stay isolated.
        int s = qi;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

        if (lane == 0)
        {
            s_dx[c] = __float2half(a / 127.0f);
            s_sx2[c * 2 + 0] = __float2half((float)s);
        }
        if (lane == 16)
        {
            s_sx2[c * 2 + 1] = __float2half((float)s);
        }
    }

    __syncthreads();

    // ── Stage 2: dp4a accumulation across (sb, slot) units ────────────
    // Each (super-block, slot) is one Q6_K 16-element sub-block (4 dp4a).
    // slot ∈ [0, 16) decodes as: half_idx = slot >> 3, q_quad = (slot >> 1) & 3,
    // isc = slot & 1. Total units per row = superblocks_per_row * 16.
    // For k=4096 → 256 units = 2 per thread; for k=12288 → 768 units = 6 each.
    const int total_units = superblocks_per_row * 16;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 210;
    float acc = 0.0f;

    for (int unit = tid; unit < total_units; unit += MMVQ_LARGE_THREADS)
    {
        int sb = unit >> 4;            // unit / 16
        int slot = unit & 15;          // 0..15

        int half_idx = slot >> 3;      // 0..1
        int q_quad   = (slot >> 1) & 3; // 0..3
        int isc      = slot & 1;        // 0..1

        const uint8_t* block = w_row + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh + half_idx * 32;
        const int8_t*  sc_half = scales + half_idx * 8;

        int ql_offset = (q_quad & 1) * 32;
        int ql_shift  = (q_quad & 2) << 1;   // 0 or 4
        int qh_shift  = q_quad << 1;         // 0,2,4,6
        int chunk_idx = sb * 8 + half_idx * 4 + q_quad;
        int sc = sc_half[q_quad * 2 + isc];

        int l_base = isc * 16;
        const int8_t* xq = s_xq + chunk_idx * 32 + l_base;

        int dot = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++)
        {
            int wpack = 0;
            #pragma unroll
            for (int b = 0; b < 4; b++)
            {
                int l = l_base + g * 4 + b;
                int q = ((ql_half[l + ql_offset] >> ql_shift) & 0x0F)
                      | (((qh_half[l] >> qh_shift) & 3) << 4);
                wpack |= (q & 0xFF) << (b * 8);
            }
            int xpack = *reinterpret_cast<const int*>(xq + 4 * g);
            dot = __dp4a(wpack, xpack, dot);
        }

        float dx = __half2float(s_dx[chunk_idx]);
        float sx = __half2float(s_sx2[chunk_idx * 2 + isc]);

        acc += dx * d * (float)sc * ((float)dot - 32.0f * sx);
    }

    // ── Stage 3: warp-shfl reduction → 4-warp shmem fan-in → final sum ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        s_warp_partials[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0)
    {
        float v = (lane < MMVQ_LARGE_NWARPS) ? s_warp_partials[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            y[row] = __float2half(v);
    }
}

// ============================================================================
// PRE-QUANTIZED VARIANTS (`_preq`)
//
// These kernels skip Stage 1 entirely. The host launches `quantize_x_to_q8_1`
// once per input vector to populate device-resident scratch buffers, then
// passes those pointers into the GEMV. This eliminates the redundant Stage 1
// work that the on-the-fly kernels run once per CUDA block (n times for
// MMVQ-large, n/4 times for MMQ-4-rows).
//
// Scratch layout (matches the in-kernel s_xq/s_dx/s_sx[2] for Q6_K):
//   const int8_t* xq   [num_chunks * 32]
//   const half*   dx   [num_chunks]
//   const half*   sx2  [num_chunks * 2]   per-half-chunk sums (lo, hi)
//
// Q4_K / Q5_K consume the full-chunk sum lazily as sx2[c*2+0] + sx2[c*2+1]
// (one extra integer-as-FP add per chunk-use), so a single scratch layout
// works for all three quant types and the pre-quant kernel writes per-half
// sums identically to Q6_K's existing in-kernel Stage 1.
// ============================================================================

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q2_k_mmq_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;

    __shared__ float s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    // Stage 2: dp4a accumulation. Same shape as quantized_gemv_q2_k_mmq's Stage 2,
    // except the per-chunk INT8 packs / dx / per-half-chunk sums come from device
    // scratch populated upstream by quantize_x_to_q8_1 (matching Q6_K's layout).
    const int total_units = rows_in_block * superblocks_per_row;

    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 84;
        const uint8_t* block = w_row + sb * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        float d    = __half2float(*reinterpret_cast<const half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 82));

        float row_acc = 0.0f;

        // 8 chunks per super-block, each chunk covers 2 sub-blocks (isc=0,1).
        #pragma unroll
        for (int q_quad = 0; q_quad < 8; q_quad++)
        {
            int chunk_idx = sb * 8 + q_quad;
            const uint8_t* chunk_qs = qs + q_quad * 8;   // 8 bytes = 2 sub-blocks of 4 bytes

            #pragma unroll
            for (int isc = 0; isc < 2; isc++)
            {
                int sub = q_quad * 2 + isc;              // sub-block index 0..15
                int sc  = scales[sub] & 0x0F;            // 4-bit scale
                int dm  = (scales[sub] >> 4) & 0x0F;     // 4-bit dmin coef

                const uint8_t* sub_qs = chunk_qs + isc * 4;   // 4 bytes for this sub-block
                int l_base = isc * 16;
                const int8_t* xq = xq_in + chunk_idx * 32 + l_base;

                int dot = 0;
                // 16 elements / 4 per dp4a = 4 dp4a calls. Each byte qs[g] holds 4
                // consecutive 2-bit values at bit offsets 0,2,4,6.
                #pragma unroll
                for (int g = 0; g < 4; g++)
                {
                    uint32_t qbyte = sub_qs[g];
                    int q0 = (int)((qbyte >> 0) & 0x3);
                    int q1 = (int)((qbyte >> 2) & 0x3);
                    int q2v = (int)((qbyte >> 4) & 0x3);
                    int q3 = (int)((qbyte >> 6) & 0x3);
                    int wpack = (q0 & 0xFF)
                              | ((q1 & 0xFF) << 8)
                              | ((q2v & 0xFF) << 16)
                              | ((q3 & 0xFF) << 24);
                    int xpack = *reinterpret_cast<const int*>(xq + 4 * g);
                    dot = __dp4a(wpack, xpack, dot);
                }

                float dx = __half2float(dx_in[chunk_idx]);
                float sx = __half2float(sx2_in[chunk_idx * 2 + isc]);

                // Per-element identity:
                //   Σ (d*sc*q - dmin*dm) * x  ≈  dx * (d*sc * Σ q*xq  -  dmin*dm * Σ xq)
                row_acc += dx * (d * (float)sc * (float)dot - dmin * (float)dm * sx);
            }
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    // Stage 3: per-row reduction (same shape as Q4_K/Q5_K/Q6_K _preq).
    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q4_k_mmq_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;

    __shared__ float s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    const int total_units = rows_in_block * superblocks_per_row;
    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;
        const uint8_t* block = w_row + sb * 144;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        float row_acc = 0.0f;

        #pragma unroll
        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd  = pair * 2 + 1;

            int sc0, m0, sc1, m1;
            if (sb_even < 4)
            {
                sc0 = scales_raw[sb_even]     & 0x3F;
                m0  = scales_raw[sb_even + 4] & 0x3F;
                sc1 = scales_raw[sb_odd]      & 0x3F;
                m1  = scales_raw[sb_odd + 4]  & 0x3F;
            }
            else
            {
                sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
                m0  = (scales_raw[sb_even + 4] >> 4)   | ((scales_raw[sb_even]     >> 6) << 4);
                sc1 = (scales_raw[sb_odd + 4]  & 0x0F) | ((scales_raw[sb_odd - 4]  >> 6) << 4);
                m1  = (scales_raw[sb_odd + 4]  >> 4)   | ((scales_raw[sb_odd]      >> 6) << 4);
            }

            const uint8_t* pair_qs = qs + pair * 32;
            int chunk_even = sb * 8 + sb_even;
            int chunk_odd  = sb * 8 + sb_odd;

            const int8_t* xq_even = xq_in + chunk_even * 32;
            const int8_t* xq_odd  = xq_in + chunk_odd  * 32;

            int dot0 = 0;
            int dot1 = 0;

            #pragma unroll
            for (int g = 0; g < 8; g++)
            {
                uint32_t qpacked = *reinterpret_cast<const uint32_t*>(pair_qs + g * 4);
                int lo = (int)(qpacked & 0x0F0F0F0F);
                int hi = (int)((qpacked >> 4) & 0x0F0F0F0F);

                int xq_e_packed = *reinterpret_cast<const int*>(xq_even + g * 4);
                int xq_o_packed = *reinterpret_cast<const int*>(xq_odd  + g * 4);

                dot0 = __dp4a(lo, xq_e_packed, dot0);
                dot1 = __dp4a(hi, xq_o_packed, dot1);
            }

            float dx_e = __half2float(dx_in[chunk_even]);
            float dx_o = __half2float(dx_in[chunk_odd]);
            // Full-chunk sum = lo-half + hi-half (Q6_K stores both; Q4_K only needs the sum).
            float sx_e = __half2float(sx2_in[chunk_even * 2 + 0]) + __half2float(sx2_in[chunk_even * 2 + 1]);
            float sx_o = __half2float(sx2_in[chunk_odd  * 2 + 0]) + __half2float(sx2_in[chunk_odd  * 2 + 1]);

            row_acc += dx_e * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * sx_e);
            row_acc += dx_o * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * sx_o);
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q5_k_mmq_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;

    __shared__ float s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    const int total_units = rows_in_block * superblocks_per_row;
    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 176;
        const uint8_t* block = w_row + sb * 176;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        float row_acc = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++)
        {
            int sc, m;
            if (sub < 4)
            {
                sc = scales_raw[sub] & 0x3F;
                m  = scales_raw[sub + 4] & 0x3F;
            }
            else
            {
                sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
                m  = (scales_raw[sub + 4] >> 4)   | ((scales_raw[sub]     >> 6) << 4);
            }

            const uint8_t* sub_qs = qs + sub * 16;
            const uint8_t* sub_qh = qh + sub * 4;
            int chunk_idx = sb * 8 + sub;
            const int8_t* xq = xq_in + chunk_idx * 32;

            int dot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
            {
                int j_e = 2 * g;
                int j_o = 2 * g + 1;
                int p_e = sub_qs[j_e];
                int p_o = sub_qs[j_o];
                int qhb_e = sub_qh[j_e >> 2];
                int qhb_o = sub_qh[j_o >> 2];

                int blo_e = (qhb_e >> ((j_e & 3) * 2)) & 1;
                int bhi_e = (qhb_e >> ((j_e & 3) * 2 + 1)) & 1;
                int blo_o = (qhb_o >> ((j_o & 3) * 2)) & 1;
                int bhi_o = (qhb_o >> ((j_o & 3) * 2 + 1)) & 1;

                int lo_e = (p_e & 0x0F) | (blo_e << 4);
                int hi_e = (p_e >> 4)   | (bhi_e << 4);
                int lo_o = (p_o & 0x0F) | (blo_o << 4);
                int hi_o = (p_o >> 4)   | (bhi_o << 4);

                int wpack = (lo_e & 0xFF) | ((hi_e & 0xFF) << 8)
                          | ((lo_o & 0xFF) << 16) | ((hi_o & 0xFF) << 24);
                int xpack = *reinterpret_cast<const int*>(xq + 4 * g);

                dot = __dp4a(wpack, xpack, dot);
            }

            float dx = __half2float(dx_in[chunk_idx]);
            float sx = __half2float(sx2_in[chunk_idx * 2 + 0]) + __half2float(sx2_in[chunk_idx * 2 + 1]);
            row_acc += dx * (d * (float)sc * (float)dot - dmin * (float)m * sx);
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q6_k_mmq_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    if (row_base >= n) return;
    const int rows_in_block = (n - row_base) < MMQ_ROWS_PER_BLOCK
                                  ? (n - row_base)
                                  : MMQ_ROWS_PER_BLOCK;

    const int superblocks_per_row = k / 256;

    __shared__ float s_acc[MMQ_ROWS_PER_BLOCK * 256];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int r = 0; r < MMQ_ROWS_PER_BLOCK; r++)
        s_acc[r * 256 + tid] = 0.0f;

    __syncthreads();

    const int total_units = rows_in_block * superblocks_per_row;
    for (int unit = tid; unit < total_units; unit += blockDim.x)
    {
        int r = unit / superblocks_per_row;
        int sb = unit % superblocks_per_row;
        int row = row_base + r;

        const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 210;
        const uint8_t* block = w_row + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        float row_acc = 0.0f;

        #pragma unroll
        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            const uint8_t* ql_half = ql + half_idx * 64;
            const uint8_t* qh_half = qh + half_idx * 32;
            const int8_t*  sc_half = scales + half_idx * 8;

            #pragma unroll
            for (int q_quad = 0; q_quad < 4; q_quad++)
            {
                int ql_offset = (q_quad & 1) * 32;
                int ql_shift  = (q_quad & 2) << 1;
                int qh_shift  = q_quad << 1;
                int chunk_idx = sb * 8 + half_idx * 4 + q_quad;

                #pragma unroll
                for (int isc = 0; isc < 2; isc++)
                {
                    int sc = sc_half[q_quad * 2 + isc];

                    int dot = 0;
                    int l_base = isc * 16;
                    const int8_t* xq = xq_in + chunk_idx * 32 + l_base;

                    #pragma unroll
                    for (int g = 0; g < 4; g++)
                    {
                        int wpack = 0;
                        #pragma unroll
                        for (int b = 0; b < 4; b++)
                        {
                            int l = l_base + g * 4 + b;
                            int q = ((ql_half[l + ql_offset] >> ql_shift) & 0x0F)
                                  | (((qh_half[l] >> qh_shift) & 3) << 4);
                            wpack |= (q & 0xFF) << (b * 8);
                        }
                        int xpack = *reinterpret_cast<const int*>(xq + 4 * g);
                        dot = __dp4a(wpack, xpack, dot);
                    }

                    float dx = __half2float(dx_in[chunk_idx]);
                    float sx = __half2float(sx2_in[chunk_idx * 2 + isc]);
                    row_acc += dx * d * (float)sc * ((float)dot - 32.0f * sx);
                }
            }
        }

        s_acc[r * 256 + tid] += row_acc;
    }
    __syncthreads();

    if (warp_id < rows_in_block)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v += s_acc[warp_id * 256 + i * 32 + lane];

        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);

        if (lane == 0)
            y[row_base + warp_id] = __float2half(v);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MMVQ-large `_preq` variants — 1 row per CUDA block, 128 threads, no Stage 1.
// Stage 2/3 identical to the on-the-fly variants except scratch comes from the
// passed device pointers instead of __shared__ buffers populated in-kernel.
// ────────────────────────────────────────────────────────────────────────────

extern "C" __global__ void __launch_bounds__(MMVQ_LARGE_THREADS) quantized_gemv_q4_k_mmvq_large_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;

    __shared__ float s_warp_partials[MMVQ_LARGE_NWARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int total_units = superblocks_per_row * 4;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;
    float acc = 0.0f;

    for (int unit = tid; unit < total_units; unit += MMVQ_LARGE_THREADS)
    {
        int sb = unit >> 2;
        int pair = unit & 3;

        const uint8_t* block = w_row + sb * 144;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        int sb_even = pair * 2;
        int sb_odd  = pair * 2 + 1;
        int sc0, m0, sc1, m1;
        if (sb_even < 4)
        {
            sc0 = scales_raw[sb_even]     & 0x3F;
            m0  = scales_raw[sb_even + 4] & 0x3F;
            sc1 = scales_raw[sb_odd]      & 0x3F;
            m1  = scales_raw[sb_odd + 4]  & 0x3F;
        }
        else
        {
            sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
            m0  = (scales_raw[sb_even + 4] >> 4)   | ((scales_raw[sb_even]     >> 6) << 4);
            sc1 = (scales_raw[sb_odd + 4]  & 0x0F) | ((scales_raw[sb_odd - 4]  >> 6) << 4);
            m1  = (scales_raw[sb_odd + 4]  >> 4)   | ((scales_raw[sb_odd]      >> 6) << 4);
        }

        const uint8_t* pair_qs = qs + pair * 32;
        int chunk_even = sb * 8 + sb_even;
        int chunk_odd  = sb * 8 + sb_odd;

        const int8_t* xq_even = xq_in + chunk_even * 32;
        const int8_t* xq_odd  = xq_in + chunk_odd  * 32;

        int dot0 = 0;
        int dot1 = 0;

        #pragma unroll
        for (int g = 0; g < 8; g++)
        {
            uint32_t qpacked = *reinterpret_cast<const uint32_t*>(pair_qs + g * 4);
            int lo = (int)(qpacked & 0x0F0F0F0F);
            int hi = (int)((qpacked >> 4) & 0x0F0F0F0F);

            int xq_e_packed = *reinterpret_cast<const int*>(xq_even + g * 4);
            int xq_o_packed = *reinterpret_cast<const int*>(xq_odd  + g * 4);

            dot0 = __dp4a(lo, xq_e_packed, dot0);
            dot1 = __dp4a(hi, xq_o_packed, dot1);
        }

        float dx_e = __half2float(dx_in[chunk_even]);
        float dx_o = __half2float(dx_in[chunk_odd]);
        float sx_e = __half2float(sx2_in[chunk_even * 2 + 0]) + __half2float(sx2_in[chunk_even * 2 + 1]);
        float sx_o = __half2float(sx2_in[chunk_odd  * 2 + 0]) + __half2float(sx2_in[chunk_odd  * 2 + 1]);

        acc += dx_e * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * sx_e);
        acc += dx_o * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * sx_o);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        s_warp_partials[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0)
    {
        float v = (lane < MMVQ_LARGE_NWARPS) ? s_warp_partials[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            y[row] = __float2half(v);
    }
}

extern "C" __global__ void __launch_bounds__(MMVQ_LARGE_THREADS) quantized_gemv_q5_k_mmvq_large_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;

    __shared__ float s_warp_partials[MMVQ_LARGE_NWARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int total_units = superblocks_per_row * 8;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 176;
    float acc = 0.0f;

    for (int unit = tid; unit < total_units; unit += MMVQ_LARGE_THREADS)
    {
        int sb = unit >> 3;
        int sub = unit & 7;

        const uint8_t* block = w_row + sb * 176;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        int sc, m;
        if (sub < 4)
        {
            sc = scales_raw[sub] & 0x3F;
            m  = scales_raw[sub + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
            m  = (scales_raw[sub + 4] >> 4)   | ((scales_raw[sub]     >> 6) << 4);
        }

        const uint8_t* sub_qs = qs + sub * 16;
        const uint8_t* sub_qh = qh + sub * 4;
        int chunk_idx = sb * 8 + sub;
        const int8_t* xq = xq_in + chunk_idx * 32;

        int dot = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++)
        {
            int j_e = 2 * g;
            int j_o = 2 * g + 1;
            int p_e = sub_qs[j_e];
            int p_o = sub_qs[j_o];
            int qhb_e = sub_qh[j_e >> 2];
            int qhb_o = sub_qh[j_o >> 2];

            int blo_e = (qhb_e >> ((j_e & 3) * 2)) & 1;
            int bhi_e = (qhb_e >> ((j_e & 3) * 2 + 1)) & 1;
            int blo_o = (qhb_o >> ((j_o & 3) * 2)) & 1;
            int bhi_o = (qhb_o >> ((j_o & 3) * 2 + 1)) & 1;

            int lo_e = (p_e & 0x0F) | (blo_e << 4);
            int hi_e = (p_e >> 4)   | (bhi_e << 4);
            int lo_o = (p_o & 0x0F) | (blo_o << 4);
            int hi_o = (p_o >> 4)   | (bhi_o << 4);

            int wpack = (lo_e & 0xFF) | ((hi_e & 0xFF) << 8)
                      | ((lo_o & 0xFF) << 16) | ((hi_o & 0xFF) << 24);
            int xpack = *reinterpret_cast<const int*>(xq + 4 * g);

            dot = __dp4a(wpack, xpack, dot);
        }

        float dx = __half2float(dx_in[chunk_idx]);
        float sx = __half2float(sx2_in[chunk_idx * 2 + 0]) + __half2float(sx2_in[chunk_idx * 2 + 1]);
        acc += dx * (d * (float)sc * (float)dot - dmin * (float)m * sx);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        s_warp_partials[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0)
    {
        float v = (lane < MMVQ_LARGE_NWARPS) ? s_warp_partials[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            y[row] = __float2half(v);
    }
}

extern "C" __global__ void __launch_bounds__(MMVQ_LARGE_THREADS) quantized_gemv_q6_k_mmvq_large_preq(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,
    const half*    __restrict__ dx_in,
    const half*    __restrict__ sx2_in,
    half* __restrict__ y,
    const int n,
    const int k)
{
    const int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;

    __shared__ float s_warp_partials[MMVQ_LARGE_NWARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int total_units = superblocks_per_row * 16;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 210;
    float acc = 0.0f;

    for (int unit = tid; unit < total_units; unit += MMVQ_LARGE_THREADS)
    {
        int sb = unit >> 4;
        int slot = unit & 15;

        int half_idx = slot >> 3;
        int q_quad   = (slot >> 1) & 3;
        int isc      = slot & 1;

        const uint8_t* block = w_row + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh + half_idx * 32;
        const int8_t*  sc_half = scales + half_idx * 8;

        int ql_offset = (q_quad & 1) * 32;
        int ql_shift  = (q_quad & 2) << 1;
        int qh_shift  = q_quad << 1;
        int chunk_idx = sb * 8 + half_idx * 4 + q_quad;
        int sc = sc_half[q_quad * 2 + isc];

        int l_base = isc * 16;
        const int8_t* xq = xq_in + chunk_idx * 32 + l_base;

        int dot = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++)
        {
            int wpack = 0;
            #pragma unroll
            for (int b = 0; b < 4; b++)
            {
                int l = l_base + g * 4 + b;
                int q = ((ql_half[l + ql_offset] >> ql_shift) & 0x0F)
                      | (((qh_half[l] >> qh_shift) & 3) << 4);
                wpack |= (q & 0xFF) << (b * 8);
            }
            int xpack = *reinterpret_cast<const int*>(xq + 4 * g);
            dot = __dp4a(wpack, xpack, dot);
        }

        float dx = __half2float(dx_in[chunk_idx]);
        float sx = __half2float(sx2_in[chunk_idx * 2 + isc]);
        acc += dx * d * (float)sc * ((float)dot - 32.0f * sx);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        s_warp_partials[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0)
    {
        float v = (lane < MMVQ_LARGE_NWARPS) ? s_warp_partials[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            y[row] = __float2half(v);
    }
}
