// I2_S (BitNet b1.58 ternary) dequantization to FP16 for dotLLM's prefill path.
// dst[row*k + col] = (code(W[row,col]) - 1) * scale
//
// dotLLM I2_S on-disk layout (must match i2_s_gemv.cu / MatMul.I2S.cs / Dequantize.DequantizeI2_S):
//   * Row-major W[n,k], k a multiple of 128. Row stride = k/4 bytes (4 codes/byte, 2 bits each).
//   * 128-element block = 32 bytes. Byte gp in [0,31] holds elements {gp, gp+32, gp+64, gp+96}
//     at bit offsets {6,4,2,0}.
//   * Code mapping value = code - 1  (0->-1, 1->0, 2->+1).
//   * ONE per-tensor float32 scale at the tensor tail, byte offset (size_t)n*(k/4).
//
// Output is dense row-major FP16 [n, k] for a cuBLAS HGEMM prefill, exactly like the Q4_K/Q6_K
// dequant kernels in dequant.cu. The tail scale offset is derived from (n,k) — the generic
// element-count dequant API cannot locate it, so this kernel takes n and k explicitly.
//
// ── v2: warp-cooperative block decode (2026-07-21) ──────────────────────────────────────────
// v1 assigned ONE THREAD per 128-element block, each decoding its 32 packed bytes in a private
// scalar loop. Adjacent threads (same warp) then processed ADJACENT blocks — bp addresses 32
// bytes apart — so at any instruction the warp's 32 lanes were reading/writing 32
// bytes/halfs scattered far apart: the same uncoalesced pattern found (and fixed) in
// dequant_pq2_0.cu's sibling kernel and pq2_0_gemv.cu's decode GEMV — see those files' v2/v3
// comments for the full analysis. Applying the identical fix here: one WARP per block, lane L
// reads packed byte L (`bp[lane]`, coalesced) and writes to `out_base[lane]`/`[lane+32]`/
// `[lane+64]`/`[lane+96]` (each a coalesced 32-lane write). Byte L's decode target is elements
// {L, L+32, L+64, L+96} of the block, so `lane` substitutes directly for the old per-thread
// `gp` loop variable.

#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256) dequant_i2_s_f16(
    const uint8_t* __restrict__ weight,   // packed codes [n × k/4 bytes] + trailing f32 scale
    half*          __restrict__ dst,      // [n × k] dense FP16
    const int n,
    const int k)
{
    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int blocks_per_row = k / 128;
    const long total_blocks  = (long)n * blocks_per_row;
    const int  row_bytes     = k / 4;

    const int  lane        = threadIdx.x & 31;
    const long warpsInGrid = ((long)gridDim.x * blockDim.x) >> 5;
    const long warpId0     = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    for (long blk = warpId0; blk < total_blocks; blk += warpsInGrid)
    {
        int row     = (int)(blk / blocks_per_row);
        int blk_in  = (int)(blk % blocks_per_row);

        const uint8_t* bp = weight + (size_t)row * row_bytes + (size_t)blk_in * 32;
        half* out_base = dst + (size_t)row * k + (size_t)blk_in * 128;

        uint8_t p = bp[lane];   // coalesced: 32 lanes read 32 consecutive bytes
        int c0 = ((p >> 6) & 0x3) - 1;
        int c1 = ((p >> 4) & 0x3) - 1;
        int c2 = ((p >> 2) & 0x3) - 1;
        int c3 = ( p       & 0x3) - 1;
        out_base[lane]      = __float2half((float)c0 * scale);
        out_base[lane + 32] = __float2half((float)c1 * scale);
        out_base[lane + 64] = __float2half((float)c2 * scale);
        out_base[lane + 96] = __float2half((float)c3 * scale);
    }
}
