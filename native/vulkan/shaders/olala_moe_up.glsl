// Olala routed-expert up projection (#1983): for every (token, slot) pair p = t * topk + s, expert e = sel[p]:
//   h[p, m] = relu( sum_k Wup[e, m, k] * x0[t, k] )^2        m < ff, x0 = latent_down(x), [T, K]
// The expert bank [E, ff, K] is read in place (Q8_0 / Q6_K / BF16). One 64-lane workgroup per (row, pair).
#include "olala_common.glsl"

#define WG_SIZE 64u
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint NCOL = 1u;

layout(set = 0, binding = 0, std430) readonly  buffer BufW   { uint weight[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufX   { ACT_T xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY   { ACT_T yout[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufSel { uint sel[]; };

layout(push_constant) uniform PushConstants {
    uint ff;      // rows per expert
    uint K;       // latent width
    uint topk;
} pc;

uint gXRow;
float xAt(uint col, uint k) { return actToF(xin[gXRow * pc.K + k]); }

#include "olala_wdot.glsl"

void main() {
    uint m = gl_WorkGroupID.x;
    uint p = gl_WorkGroupID.y;
    if (m >= pc.ff) return;
    uint lane = gl_LocalInvocationID.x;
    uint e = sel[p];
    gXRow = p / pc.topk;
    gAcc[0] = 0.0;
    rowDot(e * pc.ff + m, pc.K, lane, 0u, 0u);
    reduceAcc(lane);
    if (lane == 0u) {
        float r = max(gTot[0], 0.0);
        yout[p * pc.ff + m] = fToAct(r * r);
    }
}
