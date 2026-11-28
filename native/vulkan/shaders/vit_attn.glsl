// Inkling 2 vision tower (issue #2062, part B): full (non-causal) multi-head attention, 16 heads x 64, scale 1/8, flash-style: the keys are streamed in tiles of 32
// through shared memory with an online softmax, so no score matrix (F32 or otherwise) is ever stored: scores, the running max / sum and the output accumulator live in
// registers and shared memory only. Q, K, V, O are BF16 [npos][1024] (head h at columns h*64). One workgroup = 16 queries of one head; qStart / the dispatch size let the
// host split a long sequence into several submits (a single dispatch over 16k queries would outlive the GPU watchdog).
#extension GL_GOOGLE_include_directive : require
#include "olala_common.glsl"

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer BufQ { ACT_T qin[]; };
layout(set = 0, binding = 1, std430) readonly buffer BufK { ACT_T kin[]; };
layout(set = 0, binding = 2, std430) readonly buffer BufV { ACT_T vin[]; };
layout(set = 0, binding = 3, std430) writeonly buffer BufO { ACT_T oout[]; };

layout(push_constant) uniform PushConstants { uint npos; uint qStart; } pc;

shared float sQ[16][64];
shared float sK[32][65];
shared float sV[32][64];
shared float sS[16][32];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint h = gl_WorkGroupID.y;
    uint q0 = pc.qStart + gl_WorkGroupID.x * 16u;
    uint qi = tid >> 3u, jb = tid & 7u;

    for (uint e = 0u; e < 8u; ++e) {
        uint idx = tid + e * 128u;
        uint r = idx >> 6u, d = idx & 63u;
        sQ[r][d] = (q0 + r < pc.npos) ? actToF(qin[(q0 + r) * 1024u + h * 64u + d]) : 0.0;
    }
    float m = -1e30, l = 0.0;
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = 0.0;

    for (uint k0 = 0u; k0 < pc.npos; k0 += 32u) {
        barrier();
        for (uint e = 0u; e < 16u; ++e) {
            uint idx = tid + e * 128u;
            uint r = idx >> 6u, d = idx & 63u;
            bool ok = k0 + r < pc.npos;
            sK[r][d] = ok ? actToF(kin[(k0 + r) * 1024u + h * 64u + d]) : 0.0;
            sV[r][d] = ok ? actToF(vin[(k0 + r) * 1024u + h * 64u + d]) : 0.0;
        }
        barrier();
        for (uint jj = 0u; jj < 4u; ++jj) {
            uint j = jb + jj * 8u;
            float s = 0.0;
            for (uint d = 0u; d < 64u; ++d) s = fma(sQ[qi][d], sK[j][d], s);
            sS[qi][j] = (k0 + j < pc.npos) ? s * 0.125 : -1e30;
        }
        barrier();
        float mt = -1e30;
        for (uint j = 0u; j < 32u; ++j) mt = max(mt, sS[qi][j]);
        float mn = max(m, mt);
        float alpha = exp(m - mn);
        float p[32];
        float ps = 0.0;
        for (uint j = 0u; j < 32u; ++j) { p[j] = exp(sS[qi][j] - mn); ps += p[j]; }
        l = l * alpha + ps;
        for (uint d = 0u; d < 8u; ++d) {
            float a = acc[d] * alpha;
            for (uint j = 0u; j < 32u; ++j) a = fma(p[j], sV[j][jb * 8u + d], a);
            acc[d] = a;
        }
        m = mn;
    }
    if (q0 + qi < pc.npos) {
        float inv = 1.0 / l;
        for (uint d = 0u; d < 8u; ++d) oout[(q0 + qi) * 1024u + h * 64u + jb * 8u + d] = fToAct(acc[d] * inv);
    }
}
