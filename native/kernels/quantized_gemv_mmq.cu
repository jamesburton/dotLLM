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

// MMQ_MAX_CHUNKS: per-block shared-memory budget for input quantization scratch.
// Each chunk = 32 int8 + 1 half scale + 1 half sum = 36 bytes; 256 chunks fits in 9216 B.
// Supports k up to 8192. SmolLM-135M k=576 → 18 chunks.
#define MMQ_MAX_CHUNKS 256

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

    __shared__ int8_t  s_xq[MMQ_MAX_CHUNKS * 32];
    __shared__ half    s_dx[MMQ_MAX_CHUNKS];
    __shared__ half    s_sx[MMQ_MAX_CHUNKS];

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

    __shared__ int8_t  s_xq[MMQ_MAX_CHUNKS * 32];
    __shared__ half    s_dx[MMQ_MAX_CHUNKS];
    __shared__ half    s_sx[MMQ_MAX_CHUNKS];
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

    __shared__ int8_t  s_xq[MMQ_MAX_CHUNKS * 32];
    __shared__ half    s_dx[MMQ_MAX_CHUNKS];
    __shared__ half    s_sx2[MMQ_MAX_CHUNKS * 2];   // [a, b] per chunk
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
