// Olala MoE router + top-k + renormalisation, fully on the GPU (#1983, #2000). One 256-thread workgroup per token:
//   logits = x Wr^T (F32 router, read in place), s = sigmoid(logits), top-k of s + expert_bias (ties -> lower index),
//   w = s[sel] / (sum s[sel] + 1e-20) * scale.
// The 256 router scores live in shared memory only (no F32 activation storage); the outputs are the integer expert ids and the routing
// weights in the activation type (BF16).
// #2000: the router row scan was latency-bound (decode: one workgroup, 430 us for 1.5 MB = 3.6 GB/s, 51% of a decode step). The F32 router is now read
// in place as vec4 (H % 4 == 0, checked by the loader) and each subgroup keeps four expert rows in flight at once. The scores still never leave
// shared memory (no score tensor is stored).
#include "olala_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufRw   { vec4 rw4[]; };     // [E, H] F32 (read as vec4)
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

shared vec4  sx4[1024];           // the token row (H <= 4096, H % 4 == 0)
shared float sig[1024];           // sigmoid scores
shared float scored[1024];        // sig + bias, selected entries set to -inf
shared float redv[32];
shared uint  redi[32];
shared uint  chosen[16];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    uint H4 = pc.H >> 2;
    for (uint k = tid; k < H4; k += 256u) {
        uint b = t * pc.H + 4u * k;
        sx4[k] = vec4(actToF(xin[b]), actToF(xin[b + 1u]), actToF(xin[b + 2u]), actToF(xin[b + 3u]));
    }
    barrier();

    // Scores: each subgroup scores four experts at a time (four independent row streams in flight), coalesced vec4 over k.
    uint sgSize = gl_SubgroupSize, nsg = gl_NumSubgroups, sg = gl_SubgroupID, lane = gl_SubgroupInvocationID;
    for (uint e0 = sg * 4u; e0 < pc.E; e0 += nsg * 4u) {
        uint r0 = min(e0, pc.E - 1u) * H4, r1 = min(e0 + 1u, pc.E - 1u) * H4, r2 = min(e0 + 2u, pc.E - 1u) * H4, r3 = min(e0 + 3u, pc.E - 1u) * H4;
        float a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        for (uint k = lane; k < H4; k += sgSize) {
            vec4 x = sx4[k];
            vec4 w0 = rw4[r0 + k], w1 = rw4[r1 + k], w2 = rw4[r2 + k], w3 = rw4[r3 + k];
            a0 = fma(w0.x, x.x, a0); a0 = fma(w0.y, x.y, a0); a0 = fma(w0.z, x.z, a0); a0 = fma(w0.w, x.w, a0);
            a1 = fma(w1.x, x.x, a1); a1 = fma(w1.y, x.y, a1); a1 = fma(w1.z, x.z, a1); a1 = fma(w1.w, x.w, a1);
            a2 = fma(w2.x, x.x, a2); a2 = fma(w2.y, x.y, a2); a2 = fma(w2.z, x.z, a2); a2 = fma(w2.w, x.w, a2);
            a3 = fma(w3.x, x.x, a3); a3 = fma(w3.y, x.y, a3); a3 = fma(w3.z, x.z, a3); a3 = fma(w3.w, x.w, a3);
        }
        a0 = subgroupAdd(a0); a1 = subgroupAdd(a1); a2 = subgroupAdd(a2); a3 = subgroupAdd(a3);
        if (subgroupElect()) {
            float s0 = sigmoidF(a0); sig[e0] = s0; scored[e0] = s0 + ebias[e0];
            if (e0 + 1u < pc.E) { float s1 = sigmoidF(a1); sig[e0 + 1u] = s1; scored[e0 + 1u] = s1 + ebias[e0 + 1u]; }
            if (e0 + 2u < pc.E) { float s2 = sigmoidF(a2); sig[e0 + 2u] = s2; scored[e0 + 2u] = s2 + ebias[e0 + 2u]; }
            if (e0 + 3u < pc.E) { float s3 = sigmoidF(a3); sig[e0 + 3u] = s3; scored[e0 + 3u] = s3 + ebias[e0 + 3u]; }
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
