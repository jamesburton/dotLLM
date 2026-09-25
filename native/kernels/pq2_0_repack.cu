// PQ2_0 (PrismML Bonsai ternary) weight repack: interleaved -> split layout.
//
// dotLLM's on-disk PQ2_0 layout interleaves each 128-element group's 2-byte Half scale directly
// before its 32 code bytes (see pq2_0_gemv.cu / dequant_pq2_0.cu file headers for the full
// on-disk format). PQ2_0_GROUP_BYTES (34) never 32-byte-aligns a group's code span, so every
// group's coalesced 32-lane code read (`group_base[2 + lane]`, see pq2_0_gemv.cu's v3 comment)
// spans 2 physical 32-byte sectors instead of 1 — ncu measured ~45% sector efficiency from this.
// An in-kernel batched/staged fix was explored (see pq2_0_gemv.cu's git history) but the added
// __syncwarp() barriers between staged batches cost more than the coalescing gained.
//
// This kernel performs a ONE-TIME, load-time repack instead: physically reorder each tensor's
// bytes (same total group count, same total scale+code byte content) into a SPLIT layout —
// all scales first, then all codes — so that every group's 32 code bytes land at a genuinely,
// unconditionally 32-byte-aligned offset with no per-group variation and NO added kernel-side
// synchronization. pq2_0_gemv.cu / dequant_pq2_0.cu read this split layout directly.
//
// Split layout (must match pq2_0_gemv.cu's and dequant_pq2_0.cu's split-addressing):
//   * Scales region (offset 0): `n * groups_per_row` contiguous Half values. Flat group index
//     `g = row * groups_per_row + gi` maps directly to `scales[g]`.
//   * Codes region (offset `codesBaseOffset`, see below): `n * groups_per_row * 32` contiguous
//     bytes. Group `g`'s 32 code bytes start at `codes[g * 32]` — always 32-byte aligned relative
//     to `codesBaseOffset` since `g * 32` is trivially a multiple of 32.
//   * `codesBaseOffset = roundUp32(totalGroups * sizeof(Half))`. The round-up (rather than the
//     unpadded `totalGroups * 2`) is deliberate: it guarantees `codesBaseOffset` itself is a
//     multiple of 32 (and hence every group's code span is 32-byte aligned relative to a
//     32-byte-aligned tensor base, which `cuMemAlloc` always returns) REGARDLESS of whether
//     `totalGroups` happens to be even. Without the round-up, `codesBaseOffset` would only be a
//     multiple of 32 when `totalGroups` is a multiple of 16 — true for both of Bonsai-27B's
//     dominant FFN shapes (gate/up: n=17408, groups_per_row=40; down: n=5120,
//     groups_per_row=136 — both have even n) but not proven true for every tensor this kernel
//     might ever see (e.g. a hypothetical small odd-`n` projection). The padding costs at most 31
//     wasted bytes per tensor — negligible next to multi-MB+ tensors — in exchange for making the
//     "unconditionally aligned" property actually unconditional. Numerically inert either way:
//     repack (writer) and the GEMV/dequant kernels (readers) compute this offset identically, so
//     a mismatch here would be a compile-time-obvious bug, not a runtime hazard.
//
// One warp per group (mirrors dequant_pq2_0.cu's v2 warp-per-group grid-stride loop exactly —
// this is pure data movement, no math): lane L reads/writes code byte L (coalesced both
// directions); lane 0 additionally copies the group's 2-byte scale.

#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2_0_GROUP_SIZE  128
#define PQ2_0_GROUP_BYTES 34

// Must match the identical helper in pq2_0_gemv.cu and dequant_pq2_0.cu.
__device__ __forceinline__ size_t pq2_0_codes_base_offset(long totalGroups)
{
    size_t scalesBytes = (size_t)totalGroups * sizeof(half);
    return (scalesBytes + 31) & ~(size_t)31;
}

extern "C" __global__ void __launch_bounds__(256) pq2_0_repack_split_f16(
    const uint8_t* __restrict__ interleaved,   // [n x rowBytes] rowBytes = (k/128)*34, dotLLM's on-disk layout
    uint8_t*       __restrict__ split,         // same total (padded) byte count, split layout — see file header
    const int n,
    const int k)
{
    const int  groups_per_row = k / PQ2_0_GROUP_SIZE;
    const long total_groups   = (long)n * groups_per_row;
    const int  row_bytes      = groups_per_row * PQ2_0_GROUP_BYTES;

    half*    scales = reinterpret_cast<half*>(split);
    uint8_t* codes  = split + pq2_0_codes_base_offset(total_groups);

    const int  lane        = threadIdx.x & 31;
    const long warpsInGrid = ((long)gridDim.x * blockDim.x) >> 5;
    const long warpId0     = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    for (long g = warpId0; g < total_groups; g += warpsInGrid)
    {
        int row = (int)(g / groups_per_row);
        int gi  = (int)(g % groups_per_row);

        const uint8_t* group_base = interleaved + (size_t)row * row_bytes + (size_t)gi * PQ2_0_GROUP_BYTES;

        if (lane == 0)
            scales[g] = *reinterpret_cast<const half*>(group_base);

        codes[(size_t)g * 32 + lane] = group_base[2 + lane];   // coalesced: 32 lanes, 32 consecutive bytes, both directions
    }
}
