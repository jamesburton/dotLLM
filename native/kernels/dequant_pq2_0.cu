// PQ2_0 (PrismML Bonsai ternary) dequantization to FP16 for dotLLM's prefill path.
// dst[row*k + col] = (code(W[row,col]) - 1) * group_scale
//
// dotLLM PQ2_0 on-disk layout (must match pq2_0_gemv.cu / MatMul.PQ2S.cs / Dequantize.DequantizePQ2_0):
//   * Row-major W[n,k], k a multiple of 128. Row stride = (k/128)*34 bytes.
//   * 128-element GROUP = 34 bytes: scale(Half, 2 bytes) THEN codes[32] (4 codes/byte, 2 bits
//     each). Unlike I2_S (one per-tensor tail scale), PQ2_0's scale is PER-GROUP and comes
//     BEFORE its codes — empirically verified against real Ternary-Bonsai-27B-Q2_0.gguf bytes.
//   * Byte gp in [0,31] within a group's codes holds elements {gp, gp+32, gp+64, gp+96} at bit
//     offsets {6,4,2,0}.
//   * Code mapping value = code - 1 (0->-1, 1->0, 2->+1).
//
// Output is dense row-major FP16 [n, k] for a cuBLAS HGEMM prefill, mirroring dequant_i2_s.cu.
// Grid-stride over 128-element groups: total_groups = n * (k / 128). Each thread decodes one
// group (2-byte scale + 32 packed bytes) and writes its 128 FP16 outputs.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2_0_GROUP_SIZE  128
#define PQ2_0_GROUP_BYTES 34

extern "C" __global__ void __launch_bounds__(256) dequant_pq2_0_f16(
    const uint8_t* __restrict__ weight,   // [n x rowBytes] rowBytes = (k/128)*34
    half*          __restrict__ dst,      // [n x k] dense FP16
    const int n,
    const int k)
{
    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;
    const int  row_bytes      = groups_per_row * PQ2_0_GROUP_BYTES;

    for (long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
         g < total_groups;
         g += (long)gridDim.x * blockDim.x)
    {
        int row = (int)(g / groups_per_row);
        int gi  = (int)(g % groups_per_row);

        const uint8_t* group_base = weight + (size_t)row * row_bytes + (size_t)gi * PQ2_0_GROUP_BYTES;
        const float scale = __half2float(*reinterpret_cast<const half*>(group_base));
        const uint8_t* codes = group_base + 2;
        half* out_base = dst + (size_t)row * k + (size_t)gi * PQ2_0_GROUP_SIZE;

        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = codes[gp];
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
