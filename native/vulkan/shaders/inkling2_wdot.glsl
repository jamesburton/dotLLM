// Row dot products against the Inkling 2 routed-expert banks, read IN PLACE (never expanded): NVFP4 (WT_NVFP4) or Q8_0 (WT_Q8_0, via olala_wdot.glsl).
// Contract of the includer (as olala_wdot.glsl): WG_SIZE = 64, `const uint NCOL = 1u`, `readonly buffer { uint weight[]; }` at binding 0,
//   float xAt(uint col, uint k)    and (X_LOAD8) xLoad8(col, k0, out vec4 a, out vec4 b) = x[k0 .. k0 + 7] (k0 % 8 == 0).
// NVFP4 block = 36 bytes = 9 words (nvfp4_decode.glsl): word 0 = four UE4M3 scale bytes (one per 16 elements), words 1..8 = 32 bytes of nibbles; sub-block s owns
// bytes 8s .. 8s+7, byte j holding element 16s + j in its low nibble and element 16s + 8 + j in its high nibble. value = e2m1Doubled(nibble) * ue4m3_half(scale).
// Four lanes per block (one 16-element sub-block each), 16 blocks per 64-lane window; F32 accumulation. The per-expert tensor scale (X.scale) is applied
// by the caller to the finished dot.
#if defined(WT_NVFP4)

float gAcc[NCOL];
#include "nvfp4_decode.glsl"

void rowDot(uint row, uint K, uint lane, uint colBase, uint colLimit) {
    uint bpr = K >> 6u;
    uint rowWord = row * bpr * NVFP4_BLOCK_UINTS;
    uint blockInWindow = lane >> 2u;
    uint sub = lane & 3u;
    for (uint w0 = 0u; w0 < bpr; w0 += WG_SIZE / 4u) {
        uint blk = w0 + blockInWindow;
        if (blk >= bpr) continue;
        uint b = rowWord + blk * NVFP4_BLOCK_UINTS;
        float sc = ue4m3_half((weight[b] >> (8u * sub)) & 0xFFu);
        uvec2 qs = uvec2(weight[b + 1u + 2u * sub], weight[b + 2u + 2u * sub]);
        uint k0 = blk * 64u + sub * 16u;
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
            uint col = min(colBase + cc, colLimit);
            vec4 xla, xlb, xha, xhb;
            xLoad8(col, k0, xla, xlb);
            xLoad8(col, k0 + 8u, xha, xhb);
            float xl[8] = float[8](xla.x, xla.y, xla.z, xla.w, xlb.x, xlb.y, xlb.z, xlb.w);
            float xh[8] = float[8](xha.x, xha.y, xha.z, xha.w, xhb.x, xhb.y, xhb.z, xhb.w);
            float dq = 0.0;
            [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                uint by = ((j < 4u ? qs.x : qs.y) >> (8u * (j & 3u))) & 0xFFu;
                dq = fma(e2m1Doubled(by & 0xFu), xl[j], dq);
                dq = fma(e2m1Doubled(by >> 4u), xh[j], dq);
            }
            gAcc[cc] = fma(sc, dq, gAcc[cc]);
        }
    }
}

// Workgroup sum of one value per lane (subgroup add + shared pass); the result is valid in lane 0 only.
shared float wdotPartials[NCOL][WG_SIZE];
float gTot[NCOL];
void reduceAcc(uint lane) {
    float sums[NCOL];
    [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) sums[cc] = subgroupAdd(gAcc[cc]);
    if (subgroupElect())
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) wdotPartials[cc][gl_SubgroupID] = sums[cc];
    barrier();
    [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
        float tot = 0.0;
        if (lane == 0u)
            for (uint i = 0u; i < gl_NumSubgroups; i++) tot += wdotPartials[cc][i];
        gTot[cc] = tot;
    }
    barrier();
}

#else
#include "olala_wdot.glsl"
#endif
