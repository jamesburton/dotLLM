// Olala MoE router + top-k + renormalisation, fully on the GPU (#1983). One 256-thread workgroup per token:
//   logits = x Wr^T (F32 router, read in place), s = sigmoid(logits), top-k of s + expert_bias (ties -> lower index),
//   w = s[sel] / (sum s[sel] + 1e-20) * scale.
// The 256 router scores live in shared memory only (no F32 activation storage); the outputs are the integer expert ids and the routing
// weights in the activation type (BF16).
#include "olala_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufRw   { float rw[]; };      // [E, H] F32
layout(set = 0, binding = 1, std430) readonly  buffer BufX    { ACT_T xin[]; };     // [T, H]
layout(set = 0, binding = 2, std430) readonly  buffer BufBias { float ebias[]; };   // [E]
layout(set = 0, binding = 3, std430) writeonly buffer BufSel  { uint sel[]; };      // [T, topk]
layout(set = 0, binding = 4, std430) writeonly buffer BufSelW { ACT_T selw[]; };    // [T, topk]

layout(push_constant) uniform PushConstants {
    uint H;
    uint E;       // <= 1024
    uint topk;    // <= 16
    float scale;
} pc;

shared float sx[4096];            // the token row (H <= 4096)
shared float sig[1024];           // sigmoid scores
shared float scored[1024];        // sig + bias, selected entries set to -inf
shared float redv[32];
shared uint  redi[32];
shared uint  chosen[16];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    for (uint k = tid; k < pc.H; k += 256u) sx[k] = actToF(xin[t * pc.H + k]);
    barrier();

    // Scores: one subgroup per expert at a time, coalesced over k.
    uint sgSize = gl_SubgroupSize, nsg = gl_NumSubgroups, sg = gl_SubgroupID, lane = gl_SubgroupInvocationID;
    for (uint e = sg; e < pc.E; e += nsg) {
        float a = 0.0;
        for (uint k = lane; k < pc.H; k += sgSize) a = fma(rw[e * pc.H + k], sx[k], a);
        a = subgroupAdd(a);
        if (subgroupElect()) {
            float s = sigmoidF(a);
            sig[e] = s;
            scored[e] = s + ebias[e];
        }
    }
    barrier();

    // Top-k by repeated arg-max (lowest index wins a tie, like the CPU's insertion order).
    for (uint r = 0u; r < pc.topk; ++r) {
        float best = -3.4e38;
        for (uint e = tid; e < pc.E; e += 256u) best = max(best, scored[e]);
        best = subgroupMax(best);
        if (subgroupElect()) redv[sg] = best;
        barrier();
        float gm = redv[0];
        for (uint i = 1u; i < nsg; ++i) gm = max(gm, redv[i]);
        uint cand = 0xFFFFFFFFu;
        for (uint e = tid; e < pc.E; e += 256u) if (scored[e] == gm) { cand = min(cand, e); }
        cand = subgroupMin(cand);
        if (subgroupElect()) redi[sg] = cand;
        barrier();
        uint gi = redi[0];
        for (uint i = 1u; i < nsg; ++i) gi = min(gi, redi[i]);
        if (tid == 0u) { chosen[r] = gi; scored[gi] = -3.4e38; }
        barrier();
    }
    if (tid == 0u) {
        float sum = 0.0;
        for (uint i = 0u; i < pc.topk; ++i) sum += sig[chosen[i]];
        for (uint i = 0u; i < pc.topk; ++i) {
            sel[t * pc.topk + i] = chosen[i];
            selw[t * pc.topk + i] = fToAct(sig[chosen[i]] / (sum + 1e-20) * pc.scale);
        }
    }
}
