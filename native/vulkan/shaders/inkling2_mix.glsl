// Inkling 2 depth ("attention-residual") mix (issue #2034; Inkling2Ops.DepthMixToken, llama.cpp graph::mix). Per token, over the stored RMS-normalised
// block inputs plus the RMS-normalised current state (gamma-free, eps 1e-6):
//   w_q = softmax_i( state_i . query_q / sqrt(H) ),   out[q, t, :] = sum_i w_q[i] * state_i
// The mix is over the NORMALISED states. Blocks live in one stack buffer [slot][T][H] (slot ids come in the push constants, so the stack needs no
// per-block binding); queries are F32 (attn_res_query, ffn_res_query, output_res_query) or BF16 (loop_msg_query, QBF16 = 1). One 256-thread workgroup
// per token; the dot products are reduced in F32 shared memory and the weights never leave it. Output layout [nq][T][H] so a query's rows are a
// contiguous [T, H] activation (the next GEMV / norm reads it in place).
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint QBF16 = 0u;

layout(set = 0, binding = 0, std430) readonly  buffer BufBlocks { ACT_T blocks[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufCur    { ACT_T cur[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufQ      { uint qw[]; };
layout(set = 0, binding = 3, std430) writeonly buffer BufOut    { ACT_T outp[]; };

layout(push_constant) uniform PushConstants {
    uint H;
    uint T;
    uint nq;          // <= 4
    uint nb;          // stored blocks, nb + 1 <= 6 states
    uint slotStride;  // elements between block slots
    uint curOff;      // element offset of row 0 of cur
    float eps;
    uint slots[6];
} pc;

const uint MAXS = 6u, MAXQ = 4u;
shared float i2Dot[MAXS * MAXQ];
shared float i2DotPart[MAXS * MAXQ][32];
shared float i2Wt[MAXQ][MAXS];

float qAt(uint q, uint k) {
    uint i = q * pc.H + k;
    if (QBF16 == 1u) {
        uint w = qw[i >> 1u];
        return ((i & 1u) == 0u) ? bf16Lo(w) : bf16Hi(w);
    }
    return uintBitsToFloat(qw[i]);
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    uint H = pc.H, ns = pc.nb + 1u, nq = pc.nq;
    uint cb = pc.curOff + t * H;

    float ss = 0.0;
    for (uint k = tid; k < H; k += 256u) { float v = actToF(cur[cb + k]); ss = fma(v, v, ss); }
    ss = wgSum(ss);
    float inv = 1.0 / sqrt(ss / float(H) + pc.eps);

    float acc[MAXS * MAXQ];
    [[unroll]] for (uint i = 0u; i < MAXS * MAXQ; ++i) acc[i] = 0.0;
    for (uint k = tid; k < H; k += 256u) {
        float qv[MAXQ];
        [[unroll]] for (uint q = 0u; q < MAXQ; ++q) qv[q] = q < nq ? qAt(q, k) : 0.0;
        for (uint i = 0u; i < ns; ++i) {
            float s = (i < pc.nb) ? actToF(blocks[pc.slots[i] * pc.slotStride + t * H + k]) : actToF(cur[cb + k]) * inv;
            [[unroll]] for (uint q = 0u; q < MAXQ; ++q) acc[i * MAXQ + q] = fma(s, qv[q], acc[i * MAXQ + q]);
        }
    }
    [[unroll]] for (uint j = 0u; j < MAXS * MAXQ; ++j) {
        float v = subgroupAdd(acc[j]);
        if (subgroupElect()) i2DotPart[j][gl_SubgroupID] = v;
    }
    barrier();
    if (tid < MAXS * MAXQ) {
        float tot = 0.0;
        for (uint s = 0u; s < gl_NumSubgroups; ++s) tot += i2DotPart[tid][s];
        i2Dot[tid] = tot;
    }
    barrier();
    if (tid < nq) {
        float sc = 1.0 / sqrt(float(H));
        float mx = -3.4e38;
        for (uint i = 0u; i < ns; ++i) mx = max(mx, i2Dot[i * MAXQ + tid] * sc);
        float sum = 0.0;
        for (uint i = 0u; i < ns; ++i) { float e = exp(i2Dot[i * MAXQ + tid] * sc - mx); i2Wt[tid][i] = e; sum += e; }
        float rs = 1.0 / sum;
        for (uint i = 0u; i < ns; ++i) i2Wt[tid][i] *= rs;
    }
    barrier();
    for (uint k = tid; k < H; k += 256u) {
        float o[MAXQ];
        [[unroll]] for (uint q = 0u; q < MAXQ; ++q) o[q] = 0.0;
        for (uint i = 0u; i < ns; ++i) {
            float s = (i < pc.nb) ? actToF(blocks[pc.slots[i] * pc.slotStride + t * H + k]) : actToF(cur[cb + k]) * inv;
            [[unroll]] for (uint q = 0u; q < MAXQ; ++q) o[q] = fma(i2Wt[q][i], s, o[q]);
        }
        for (uint q = 0u; q < nq; ++q) outp[(q * pc.T + t) * H + k] = fToAct(o[q]);
    }
}
