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

__device__ __constant__ int8_t kvalues_iq4nl_grouped[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// ── Q2_K: 84 bytes per 256 values ───────────────────────────────────
// struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; half d; half dmin; };
//   scales: 16 × (4-bit scale | 4-bit dmin coef), one byte per sub-block of 16 elements
//   qs:     2 bits per element (256 elements × 2 bits = 64 bytes)
//   d:      FP16 super-block delta
//   dmin:   FP16 super-block min delta
// Body mirrors quantized_gemv_q2_k — per-expert weight pointer indirection at
// block.y plus per-expert output writes at block.y. FP32 accumulation, single
// FP16 store at y[row].

extern "C" __global__ void __launch_bounds__(256, 2) moe_grouped_gemv_q2_k_f16(
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
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 84;

    float acc = 0.0f;

    for (int sb = threadIdx.x; sb < superblocks_per_row; sb += blockDim.x)
    {
        const uint8_t* block = w_row + sb * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        float d = __half2float(*reinterpret_cast<const half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 82));

        // 16 sub-blocks of 16 elements
        for (int sub = 0; sub < 16; sub++) {
            int sc = scales[sub] & 0xF;
            int dm = (scales[sub] >> 4) & 0xF;

            float sub_acc = 0.0f;
            float xsum_sub = 0.0f;
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                int t = sub * 16 + j;
                int byte_idx = t >> 2;
                int bit_off = (t & 0x3) << 1;
                int q2 = (qs[byte_idx] >> bit_off) & 0x3;
                float xv = __half2float(x[sb * 256 + t]);
                sub_acc += (float)q2 * xv;
                xsum_sub += xv;
            }
            acc += d * (float)sc * sub_acc - dmin * (float)dm * xsum_sub;
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

    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (threadIdx.x == 0) y[row] = __float2half(acc);
}

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

// ── Q5_K: 176 bytes per 256 values ──────────────────────────────────
// struct block_q5_K { half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };

extern "C" __global__ void __launch_bounds__(256, 2) moe_grouped_gemv_q5_k_f16(
    const half*       __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    half*       const* __restrict__ outputs,
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
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 176;

    float acc = 0.0f;

    for (int sb = threadIdx.x; sb < superblocks_per_row; sb += blockDim.x)
    {
        const uint8_t* block = w_row + sb * 176;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;   // 32 bytes
        const uint8_t* qs = block + 48;   // 128 bytes
        int base_x = sb * 256;

        for (int sub = 0; sub < 8; sub++)
        {
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
            int x_off = base_x + sub * 32;

            for (int j = 0; j < 16; j++)
            {
                uint8_t packed = sub_qs[j];
                int bit_lo = (sub_qh[j / 4] >> ((j % 4) * 2)) & 1;
                int bit_hi = (sub_qh[j / 4] >> ((j % 4) * 2 + 1)) & 1;
                int lo = (packed & 0x0F) | (bit_lo << 4);
                int hi = (packed >> 4) | (bit_hi << 4);

                acc += (scale * (float)lo - min_val) * __half2float(x[x_off + 2 * j]);
                acc += (scale * (float)hi - min_val) * __half2float(x[x_off + 2 * j + 1]);
            }
        }
    }

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

// ── Q6_K: 210 bytes per 256 values ──────────────────────────────────

extern "C" __global__ void __launch_bounds__(256, 2) moe_grouped_gemv_q6_k_f16(
    const half*       __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    half*       const* __restrict__ outputs,
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
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 210;

    float acc = 0.0f;

    for (int sb = threadIdx.x; sb < superblocks_per_row; sb += blockDim.x)
    {
        const uint8_t* block = w_row + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        int base_x = sb * 256;

        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            const uint8_t* ql_half = ql + half_idx * 64;
            const uint8_t* qh_half = qh + half_idx * 32;
            const int8_t* sc_half = scales + half_idx * 8;
            int x_off = base_x + half_idx * 128;

            for (int l = 0; l < 32; l++)
            {
                int isc = l / 16;

                int q1 = ((ql_half[l]      & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql_half[l]      >> 4)    | (((qh_half[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql_half[l + 32] >> 4)    | (((qh_half[l] >> 6) & 3) << 4)) - 32;

                float s0 = d * (float)sc_half[isc];
                float s1 = d * (float)sc_half[isc + 2];
                float s2 = d * (float)sc_half[isc + 4];
                float s3 = d * (float)sc_half[isc + 6];

                acc += s0 * (float)q1 * __half2float(x[x_off + l]);
                acc += s1 * (float)q2 * __half2float(x[x_off + l + 32]);
                acc += s2 * (float)q3 * __half2float(x[x_off + l + 64]);
                acc += s3 * (float)q4 * __half2float(x[x_off + l + 96]);
            }
        }
    }

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

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
// Note: K must be a multiple of 32 (the Q8_0 block size). Q5_K/Q6_K above
// require K%256==0, but Q8_0 has finer granularity — same K%256==0 gate at
// the dispatch layer keeps the API uniform.

extern "C" __global__ void __launch_bounds__(256) moe_grouped_gemv_iq4_nl_f16(
    const half*       __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    half*       const* __restrict__ outputs,
    const int M,
    const int K,
    const int K_active)
{
    int expert_idx = blockIdx.y;
    int row = blockIdx.x;
    if (expert_idx >= K_active || row >= M) return;

    const uint8_t* weight = weights[expert_idx];
    half*          y      = outputs[expert_idx];

    const int blocks_per_row = K / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 18;
    float acc = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x)
    {
        const uint8_t* block = w_row + b * 18;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;

        #pragma unroll 16
        for (int j = 0; j < 16; j++)
        {
            uint8_t packed = qs[j];
            acc += d * (float)kvalues_iq4nl_grouped[packed & 0x0F] * __half2float(x[b * 32 + j]);
            acc += d * (float)kvalues_iq4nl_grouped[packed >> 4] * __half2float(x[b * 32 + j + 16]);
        }
    }

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

extern "C" __global__ void __launch_bounds__(256, 2) moe_grouped_gemv_iq4_xs_f16(
    const half*       __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    half*       const* __restrict__ outputs,
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
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 136;
    float acc = 0.0f;

    for (int sb = threadIdx.x; sb < superblocks_per_row; sb += blockDim.x)
    {
        const uint8_t* block = w_row + sb * 136;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        uint16_t scales_h = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;
        int base_x = sb * 256;

        for (int ib = 0; ib < 8; ib++)
        {
            int low = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
            int high = (scales_h >> (2 * ib)) & 0x03;
            int ls = low | (high << 4);
            float dl = d * (float)(ls - 32);
            const uint8_t* sub_qs = qs + ib * 16;
            int x_off = base_x + ib * 32;

            #pragma unroll 16
            for (int j = 0; j < 16; j++)
            {
                uint8_t packed = sub_qs[j];
                acc += dl * (float)kvalues_iq4nl_grouped[packed & 0x0F] * __half2float(x[x_off + j]);
                acc += dl * (float)kvalues_iq4nl_grouped[packed >> 4] * __half2float(x[x_off + j + 16]);
            }
        }
    }

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

extern "C" __global__ void __launch_bounds__(256) moe_grouped_gemv_q8_0_f16(
    const half*       __restrict__ x,
    const uint8_t* const* __restrict__ weights,
    half*       const* __restrict__ outputs,
    const int M,
    const int K,
    const int K_active)
{
    int expert_idx = blockIdx.y;
    int row = blockIdx.x;
    if (expert_idx >= K_active || row >= M) return;

    const uint8_t* weight = weights[expert_idx];
    half*          y      = outputs[expert_idx];

    const int blocks_per_row = K / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 34;

    float acc = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x)
    {
        const uint8_t* block = w_row + b * 34;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

        float block_sum = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < 32; j++)
            block_sum += (float)qs[j] * __half2float(x[b * 32 + j]);

        acc += d * block_sum;
    }

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
