// Inkling 2 routed-expert gate/up projection + SITU (issue #2034; Inkling2TransformerModel.ExecuteExperts). For every (token, slot) pair p = t * topk + s with
// expert e = sel[p]:
//   g = scale[e] * (Wgu[e, j, :] . x[t]),   u = scale[e] * (Wgu[e, ff + j, :] . x[t])       (merged bank: gate rows first, then up rows)
//   h[p, j] = beta tanh(g / beta) * sigmoid(g) * lb tanh(u / lb)                              (SITU)
// The bank [E, 2ff, K] is read in place (NVFP4 with the per-expert F32 `.scale` = HASSCALE 1, or Q8_0 without); activations are BF16, accumulation F32, the
// gate / up values never leave registers. One 64-lane workgroup per (row j, pair): x = ff rows, y = T * topk pairs.
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
layout(set = 0, binding = 4, std430) readonly  buffer BufScale { float escale[]; };

layout(push_constant) uniform PushConstants {
    uint ff;      // rows of the gate half (= rows of the up half)
    uint K;       // hidden width
    uint topk;
    float beta;
    float lb;
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
    uint j = gl_WorkGroupID.x;
    uint p = gl_WorkGroupID.y;
    if (j >= pc.ff) return;
    uint lane = gl_LocalInvocationID.x;
    uint e = sel[p];
    gXRow = p / pc.topk;
    gAcc[0] = 0.0;
    rowDot(e * 2u * pc.ff + j, pc.K, lane, 0u, 0u);
    reduceAcc(lane);
    float gate = gTot[0];
    gAcc[0] = 0.0;
    rowDot(e * 2u * pc.ff + pc.ff + j, pc.K, lane, 0u, 0u);
    reduceAcc(lane);
    float up = gTot[0];
    if (lane == 0u) {
        if (HASSCALE == 1u) { float s = escale[e]; gate *= s; up *= s; }
        float act = tanhF(gate * (1.0 / pc.beta)) * pc.beta;
        act *= sigmoidF(gate);
        if (pc.lb > 0.0) up = tanhF(up * (1.0 / pc.lb)) * pc.lb;
        yout[p * pc.ff + j] = fToAct(act * up);
    }
}
