// Olala routed-expert down projection with the routing-weighted sum over the topk slots (#1983):
//   out[t, m] = sum_s w[t, s] * sum_k Wdown[e_s, m, k] * h[t * topk + s, k]       m < N (latent width), h: [T * topk, K = ff]
// The expert bank [E, N, K] is read in place (Q6_K in safe2: the ReLU^2 inputs reach ~1e9, so no quantized activation path is used).
// Per-slot dots are reduced separately and weighted afterwards, as the CPU does. One 64-lane workgroup per (row, token).
#include "olala_common.glsl"

#define WG_SIZE 64u
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint NCOL = 1u;

layout(set = 0, binding = 0, std430) readonly  buffer BufW    { uint weight[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufX    { ACT_T xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY    { ACT_T yout[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufSel  { uint sel[]; };
layout(set = 0, binding = 4, std430) readonly  buffer BufSelW { ACT_T selw[]; };

layout(push_constant) uniform PushConstants {
    uint N;       // output rows per expert (latent width)
    uint K;       // expert hidden width (ff)
    uint topk;
} pc;

uint gXRow;
float xAt(uint col, uint k) { return actToF(xin[gXRow * pc.K + k]); }

#include "olala_wdot.glsl"

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
        total = fma(actToF(selw[p]), gTot[0], total);   // lane 0 only (gTot is zero elsewhere)
    }
    if (lane == 0u) yout[t * pc.N + m] = fToAct(total);
}
