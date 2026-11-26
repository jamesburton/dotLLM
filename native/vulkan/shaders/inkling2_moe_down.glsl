// Inkling 2 routed-expert down projection with the routing-weighted sum over the topk slots (issue #2034; ExecuteExperts):
//   out[t, m] = sum_s w[t, s] * ( scale[e_s] * (Wdown[e_s, m, :] . h[t * topk + s]) )         m < N (hidden), h: [T * topk, K = ff]
// The bank [E, N, K] is read in place (NVFP4 with the per-expert F32 scale = HASSCALE 1, or Q8_0); the per-slot dot is scaled, then weighted, as the CPU does.
// One 64-lane workgroup per (row, token).
#include "inkling2_common.glsl"

#define WG_SIZE 64u
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint NCOL = 1u;
layout(constant_id = 0) const uint HASSCALE = 0u;

layout(set = 0, binding = 0, std430) readonly  buffer BufW     { uint weight[]; };
#ifdef ACT_F32
layout(set = 0, binding = 1, std430) readonly  buffer BufX     { vec4 xin4[]; };
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufX     { uvec4 xin4[]; };
#endif
layout(set = 0, binding = 2, std430) writeonly buffer BufY     { ACT_T yout[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufSel   { uint sel[]; };
layout(set = 0, binding = 4, std430) readonly  buffer BufSelW  { ACT_T selw[]; };
layout(set = 0, binding = 5, std430) readonly  buffer BufScale { float escale[]; };

layout(push_constant) uniform PushConstants {
    uint N;       // output rows per expert (hidden)
    uint K;       // expert hidden width (ff)
    uint topk;
} pc;

uint gXRow;
#ifdef ACT_F32
float xAt(uint col, uint k) { uint i = gXRow * pc.K + k; return xin4[i >> 2u][i & 3u]; }
void xLoad8(uint col, uint k0, out vec4 a, out vec4 b) { uint i = (gXRow * pc.K + k0) >> 2u; a = xin4[i]; b = xin4[i + 1u]; }
#else
float xAt(uint col, uint k) {
    uint i = gXRow * pc.K + k;
    uint w = xin4[i >> 3u][(i >> 1u) & 3u];
    return ((i & 1u) == 0u) ? bf16Lo(w) : bf16Hi(w);
}
void xLoad8(uint col, uint k0, out vec4 a, out vec4 b) {
    uvec4 u = xin4[(gXRow * pc.K + k0) >> 3u];
    a = vec4(bf16Lo(u.x), bf16Hi(u.x), bf16Lo(u.y), bf16Hi(u.y));
    b = vec4(bf16Lo(u.z), bf16Hi(u.z), bf16Lo(u.w), bf16Hi(u.w));
}
#endif
#define X_LOAD8 1

#include "inkling2_wdot.glsl"

void main() {
    uint m = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    if (m >= pc.N) return;
    uint lane = gl_LocalInvocationID.x;
    float total = 0.0;
    for (uint s = 0u; s < pc.topk; ++s) {
        uint p = t * pc.topk + s;
        uint e = sel[p];
        gXRow = p;
        gAcc[0] = 0.0;
        rowDot(e * pc.N + m, pc.K, lane, 0u, 0u);
        reduceAcc(lane);
        float d = gTot[0];   // lane 0 only (gTot is zero elsewhere)
        if (HASSCALE == 1u) d *= escale[e];
        total = fma(actToF(selw[p]), d, total);
    }
    if (lane == 0u) yout[t * pc.N + m] = fToAct(total);
}
