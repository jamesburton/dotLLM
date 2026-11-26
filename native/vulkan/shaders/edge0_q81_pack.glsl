// Edge0 latent MoE (#1981): quantises ONE Q8_1 block (32 values, one per lane of a 32-thread workgroup) exactly like quantize_q8_1_rows.comp:
//   d = amax / 127 (IEEE, q81DivBy127), id = 1 / d (q81Recip), q = roundEven(v * id), s = d * sum(q).
// Output layout (the same one the Q8_1 consumers read): xq[qWord0 + 0..7] = 4 int8 per uint, xds[dsIdx] = (d, s).
// Every lane of the workgroup must call it (it contains barriers). Needs `xq` (uint[]) and `xds` (vec2[]) buffers declared before the include.
#extension GL_GOOGLE_include_directive : require
#include "q8_1_exact_div.glsl"

shared float edge0SA[32];
shared int   edge0SQ[32];

void edge0PackQ81(float v, uint lane, uint qWord0, uint dsIdx) {
    edge0SA[lane] = abs(v);
    barrier();
    for (uint s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) edge0SA[lane] = max(edge0SA[lane], edge0SA[lane + s]);
        barrier();
    }
    float amax = edge0SA[0];
    float d = q81DivBy127(amax);
    float id = (d != 0.0) ? q81Recip(d) : 0.0;
    edge0SQ[lane] = int(roundEven(v * id));
    barrier();
    if (lane < 8u) {
        uint b = lane * 4u;
        xq[qWord0 + lane] = (uint(edge0SQ[b]) & 0xFFu) | ((uint(edge0SQ[b + 1u]) & 0xFFu) << 8u)
                          | ((uint(edge0SQ[b + 2u]) & 0xFFu) << 16u) | ((uint(edge0SQ[b + 3u]) & 0xFFu) << 24u);
    }
    if (lane == 0u) {
        int sumI = 0;
        for (uint i = 0u; i < 32u; i++) sumI += edge0SQ[i];
        xds[dsIdx] = vec2(d, d * float(sumI));
    }
    barrier();
}
