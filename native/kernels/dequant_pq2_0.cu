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
//
// ── v2: warp-cooperative group decode (2026-07-21) ─────────────────────────────────────────
// v1 assigned ONE THREAD per 128-element group, each decoding its 32 code bytes in a private
// scalar loop. Adjacent threads (same warp) then processed ADJACENT groups — group_base
// addresses PQ2_0_GROUP_BYTES (34) apart — so at any given instruction the warp's 32 lanes
// were reading 32 bytes scattered ~1088 bytes apart: the same uncoalesced "N-byte stride
// between lanes" problem pq2_0_gemv.cu's v1/v2 GEMV kernel had (see that file's v3 comment for
// the full analysis) — this dequant kernel just never got the same fix, since prefill wasn't
// this session's initial focus.
//
// v2 assigns ONE WARP per group instead: the grid-stride loop now advances by warp (not
// thread), and within a group lane L reads code byte L (`group_base[2 + lane]`) — 32 lanes
// reading 32 consecutive bytes, one coalesced transaction — and writes to `out_base[lane]`,
// `[lane+32]`, `[lane+64]`, `[lane+96]`, each also a coalesced 32-lane write. Byte L's decode
// target is elements {L, L+32, L+64, L+96} of the group per the layout above, so `lane`
// substitutes directly for the old `gp` loop variable; the scale read is redundant across
// lanes but that's a hardware broadcast (single address, not a coalescing concern).

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

    const int  lane        = threadIdx.x & 31;
    const long warpsInGrid = ((long)gridDim.x * blockDim.x) >> 5;
    const long warpId0     = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    for (long g = warpId0; g < total_groups; g += warpsInGrid)
    {
        int row = (int)(g / groups_per_row);
        int gi  = (int)(g % groups_per_row);

        const uint8_t* group_base = weight + (size_t)row * row_bytes + (size_t)gi * PQ2_0_GROUP_BYTES;
        float scale = __half2float(*reinterpret_cast<const half*>(group_base));
        uint8_t p = group_base[2 + lane];   // coalesced: 32 lanes read 32 consecutive bytes

        int c0 = ((p >> 6) & 0x3) - 1;
        int c1 = ((p >> 4) & 0x3) - 1;
        int c2 = ((p >> 2) & 0x3) - 1;
        int c3 = ( p       & 0x3) - 1;

        half* out_base = dst + (size_t)row * k + (size_t)gi * PQ2_0_GROUP_SIZE;
        out_base[lane]      = __float2half((float)c0 * scale);
        out_base[lane + 32] = __float2half((float)c1 * scale);
        out_base[lane + 64] = __float2half((float)c2 * scale);
        out_base[lane + 96] = __float2half((float)c3 * scale);
    }
}
