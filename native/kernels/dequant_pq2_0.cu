// PQ2_0 (PrismML Bonsai ternary) dequantization to FP16 for dotLLM's prefill path.
// dst[row*k + col] = (code(W[row,col]) - 1) * group_scale
//
// dotLLM PQ2_0 on-disk layout (must match pq2_0_gemv.cu / MatMul.PQ2S.cs / Dequantize.DequantizePQ2_0):
//   * Row-major W[n,k], k a multiple of 128. Row stride = (k/128)*34 bytes.
//   * 128-element GROUP = 34 bytes: scale(Half, 2 bytes) THEN codes[32] (4 codes/byte, 2 bits
//     each). Unlike I2_S (one per-tensor tail scale), PQ2_0's scale is PER-GROUP and comes
//     BEFORE its codes — empirically verified against real Ternary-Bonsai-27B-Q2_0.gguf bytes.
//   * Byte b in [0,31] within a group's codes holds the 4 CONSECUTIVE elements
//     {4b, 4b+1, 4b+2, 4b+3} at ASCENDING bit offsets {0,2,4,6} — verified byte-for-byte against
//     PrismML's own reference `dequantize_row_q2_0` in their PrismML-Eng/llama.cpp fork
//     (ggml-quants.c: byte_index = j/4; bit_offset = (j%4)*2). This is NOT I2_S's strided
//     {gp,+32,+64,+96}/descending-bits layout — an earlier version of this kernel wrongly copied
//     that convention (issue #269 follow-up, 2026-08-05), which silently scrambled every weight's
//     position within its 128-element group (same value SET, wrong positions) while leaving
//     per-tensor activation statistics looking numerically unremarkable throughout the whole
//     forward pass — the root cause of Bonsai-27B's garbled generation. See
//     DotLLM.Cpu/Kernels/Dequantize.cs's DequantizePQ2_0 doc comment for the full writeup.
//   * Code mapping value = code - 1 (0->-1, 1->0, 2->+1, 3->+2).
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
//
// ── v3: split-layout addressing (weight repack follow-up) ─────────────────────────────────
// `weight` is now the SPLIT layout produced by `pq2_0_repack.cu`'s `pq2_0_repack_split_f16`
// (see that file's header, and pq2_0_gemv.cu's "Split-layout addressing" note, for the full
// rationale) rather than the interleaved on-disk layout: all `total_groups` scales first, then
// all `total_groups * 32` code bytes. A flat group index `g` addresses `scales[g]` and
// `codesBase[g*32 + lane]` directly — since `g*32` is trivially a multiple of 32, every group's
// coalesced 32-lane code read is now unconditionally 32-byte-aligned, same benefit as the GEMV
// kernel's v3->split-layout change. `row`/`gi` (still derived from `g` for the OUTPUT address,
// which stays dense row-major regardless of the input's layout change) are unaffected.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2_0_GROUP_SIZE  128
#define PQ2_0_GROUP_BYTES 34

// Must match the identical helper in pq2_0_gemv.cu and pq2_0_repack.cu.
__device__ __forceinline__ size_t pq2_0_codes_base_offset(long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

extern "C" __global__ void __launch_bounds__(256) dequant_pq2_0_f16(
    const uint8_t* __restrict__ weight,   // split layout — see file header's v3 note
    half*          __restrict__ dst,      // [n x k] dense FP16
    const int n,
    const int k)
{
    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;

    const half*    scales    = reinterpret_cast<const half*>(weight);
    const uint8_t* codesBase = weight + pq2_0_codes_base_offset(total_groups);

    const int  lane        = threadIdx.x & 31;
    const long warpsInGrid = ((long)gridDim.x * blockDim.x) >> 5;
    const long warpId0     = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    for (long g = warpId0; g < total_groups; g += warpsInGrid)
    {
        int row = (int)(g / groups_per_row);
        int gi  = (int)(g % groups_per_row);

        float scale = __half2float(scales[g]);
        uint8_t p = codesBase[(size_t)g * 32 + lane];   // unconditionally aligned+coalesced — see file header

        // Ascending bit offsets {0,2,4,6} → consecutive elements {4*lane, 4*lane+1, 4*lane+2, 4*lane+3}.
        int c0 = ( p       & 0x3) - 1;
        int c1 = ((p >> 2) & 0x3) - 1;
        int c2 = ((p >> 4) & 0x3) - 1;
        int c3 = ((p >> 6) & 0x3) - 1;

        half* out_base = dst + (size_t)row * k + (size_t)gi * PQ2_0_GROUP_SIZE;
        int outIdx = 4 * lane;
        out_base[outIdx]     = __float2half((float)c0 * scale);
        out_base[outIdx + 1] = __float2half((float)c1 * scale);
        out_base[outIdx + 2] = __float2half((float)c2 * scale);
        out_base[outIdx + 3] = __float2half((float)c3 * scale);
    }
}
