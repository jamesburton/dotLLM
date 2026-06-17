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
// Grid-stride over 128-element blocks: total_blocks = n * (k / 128). Each thread decodes one
// 128-element block (32 packed bytes) and writes its 128 FP16 outputs.

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

    for (long blk = (long)blockIdx.x * blockDim.x + threadIdx.x;
         blk < total_blocks;
         blk += (long)gridDim.x * blockDim.x)
    {
        int row     = (int)(blk / blocks_per_row);
        int blk_in  = (int)(blk % blocks_per_row);

        const uint8_t* bp = weight + (size_t)row * row_bytes + (size_t)blk_in * 32;
        half* out_base = dst + (size_t)row * k + (size_t)blk_in * 128;

        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = bp[gp];
            int c0 = ((p >> 6) & 0x3) - 1;
            int c1 = ((p >> 4) & 0x3) - 1;
            int c2 = ((p >> 2) & 0x3) - 1;
            int c3 = ( p       & 0x3) - 1;
            out_base[gp]      = __float2half((float)c0 * scale);
            out_base[gp + 32] = __float2half((float)c1 * scale);
            out_base[gp + 64] = __float2half((float)c2 * scale);
            out_base[gp + 96] = __float2half((float)c3 * scale);
        }
    }
}
