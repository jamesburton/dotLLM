// MoE grouped quantized GEMV: walks K_active experts in a single launch.
//
// Each expert e in [0, K_active) computes:
//   outputs[e][m] = W_e[m, k] @ x[k]    for m in [0, M)
//
// where x is a single [K] FP16 input row shared across all experts (decode
// batch=1 — all K_active routed experts feed off the same hidden vector for
// that one token), and each W_e is a [M, K] quantized matrix with raw blocks
// pointed at by weights[e].
//
// Grid: (M, K_active, 1). Each block computes one (expert, output_row) pair.
// Block: 256 threads. Body matches `quantized_gemv_q4_k` — FP32 accumulation,
// warp-then-block reduction, single half store at the end.
//
// Shared input x is read by all K_active experts; we rely on L1 / SM scope
// to coalesce reads across blocks issued in the same SM (typical K=2048
// ⇒ 4 KB ⇒ fits in L1). Per-block we don't __shared__-cache it because the
// inner loop already streams via half-unrolled reads with good locality.

#include <cuda_fp16.h>
#include <stdint.h>

// ── Q4_K: 144 bytes per 256 values ──────────────────────────────────

extern "C" __global__ void __launch_bounds__(256, 2) moe_grouped_gemv_q4_k_f16(
    const half*       __restrict__ x,         // [K]
    const uint8_t* const* __restrict__ weights, // [K_active] per-expert ptrs
    half*       const* __restrict__ outputs,  // [K_active] per-expert ptrs
    const int M,
    const int K,
    const int K_active)
{
    int expert_idx = blockIdx.y;
    int row = blockIdx.x;
    if (expert_idx >= K_active || row >= M) return;

    const uint8_t* weight = weights[expert_idx];
    half*          y      = outputs[expert_idx];

    const int superblocks_per_row = K / 256;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;

    float acc = 0.0f;

    for (int sb = threadIdx.x; sb < superblocks_per_row; sb += blockDim.x)
    {
        const uint8_t* block = w_row + sb * 144;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        // Q4_K: 4 pairs of sub-blocks, each pair shares 32 qs bytes.
        // Lower nibbles → even sub-block, upper nibbles → odd sub-block.
        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd = pair * 2 + 1;

            int sc0, m0, sc1, m1;
            if (sb_even < 4)
            {
                sc0 = scales_raw[sb_even] & 0x3F;
                m0 = scales_raw[sb_even + 4] & 0x3F;
                sc1 = scales_raw[sb_odd] & 0x3F;
                m1 = scales_raw[sb_odd + 4] & 0x3F;
            }
            else
            {
                sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
                m0 = (scales_raw[sb_even + 4] >> 4) | ((scales_raw[sb_even] >> 6) << 4);
                sc1 = (scales_raw[sb_odd + 4] & 0x0F) | ((scales_raw[sb_odd - 4] >> 6) << 4);
                m1 = (scales_raw[sb_odd + 4] >> 4) | ((scales_raw[sb_odd] >> 6) << 4);
            }

            float scale0 = d * (float)sc0;
            float min0 = dmin * (float)m0;
            float scale1 = d * (float)sc1;
            float min1 = dmin * (float)m1;

            const uint8_t* pair_qs = qs + pair * 32;
            int base_x_idx = sb * 256 + pair * 64;

            for (int j = 0; j < 32; j++)
            {
                uint8_t byte_val = pair_qs[j];
                float x_even = __half2float(x[base_x_idx + j]);
                float x_odd = __half2float(x[base_x_idx + j + 32]);

                acc += (scale0 * (float)(byte_val & 0x0F) - min0) * x_even;
                acc += (scale1 * (float)(byte_val >> 4) - min1) * x_odd;
            }
        }
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (threadIdx.x == 0)
        y[row] = __float2half(acc);
}
