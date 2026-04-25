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
