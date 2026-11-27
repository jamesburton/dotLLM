// Inkling 2 attention (issue #2034; Inkling2TransformerModel.Attend, docs/INKLING2.md section 4). One 256-thread workgroup per (kv head g, token t), GQA 3:1
// (heads 3g .. 3g+2). The score of key j for head h is
//     l = q.k / sqrt(128) + bias[h][d] - sink_h,   d = pos_q - pos_j,   bias[h][d] = relBias[d] . R[h]  (d < extent, else 0),
// and the softmax runs over the keys PLUS one extra zero logit (the sink contributes no value): p = exp(l - M) / (exp(-M) + sum exp(l - M)).
// out[h] = sigmoid(gate) * sum_j p_j v_j.
//
// Lane layout: 16 lanes per key (lane l owns head dims 8l .. 8l+7 and relative-bias dim l); the 256 threads are 16 key groups that stride the key range
// by 16. The score is reduced across the 16 lanes with xor shuffles (the q.k partial and the relBias[d][l] * R[l] partial add before the reduce);
// each key group keeps an online softmax (m, s, acc[8]) in registers; the 16 group states are merged once, in F32 shared memory, together with the sink
// (this is the combine step of a split reduction: in registers / shared memory only, nothing is stored). No split-KV, no stored partials.
//
// K / V cache: [position][kv head][128], F16 (KVQ8 = 0: 16 B per lane) or q8_0 (KVQ8 = 1: 4 blocks of [fp16 scale, 32 int8]); local layers are a ring of
// ringCap positions (key j lives at j % ringCap), global layers are linear. Local layers see keys with pos_q - pos_j < window.
#extension GL_KHR_shader_subgroup_shuffle : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint KVQ8 = 0u;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ    { ACT_T q[]; };        // [T, nh*128]
layout(set = 0, binding = 1, std430) readonly  buffer BufRel  { ACT_T rel[]; };      // [T, nh*16]
layout(set = 0, binding = 2, std430) readonly  buffer BufSink { ACT_T sinkv[]; };    // [T, nh]
layout(set = 0, binding = 3, std430) readonly  buffer BufGate { ACT_T gate[]; };     // [T, nh*128]
layout(set = 0, binding = 4, std430) readonly  buffer BufK    { uvec4 kv4[]; };
layout(set = 0, binding = 5, std430) readonly  buffer BufV    { uvec4 vv4[]; };
layout(set = 0, binding = 6, std430) readonly  buffer BufRb   { float relBias[]; };  // [extent][16]
layout(set = 0, binding = 7, std430) writeonly buffer BufOut  { ACT_T outp[]; };     // [T, nh*128]

layout(push_constant) uniform PushConstants {
    uint T;
    uint pos0;
    uint nh;
    uint nkv;
    uint isLocal;
    uint window;
    uint ringCap;     // 0 = linear
    uint extent;      // rows of relBias
} pc;

shared float shM[16 * 3];
shared float shS[16 * 3];
shared float shAcc[16 * 3 * 128];

uint u16K(uint i) { return (kv4[i >> 3u][(i >> 1u) & 3u] >> ((i & 1u) * 16u)) & 0xFFFFu; }
uint u16V(uint i) { return (vv4[i >> 3u][(i >> 1u) & 3u] >> ((i & 1u) * 16u)) & 0xFFFFu; }

void load8K(uint row, uint l, out float v[8]) {
    if (KVQ8 == 0u) {
        uvec4 u = kv4[row * 16u + l];
        vec2 a = unpackHalf2x16(u.x), b = unpackHalf2x16(u.y), c = unpackHalf2x16(u.z), d = unpackHalf2x16(u.w);
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y; v[4] = c.x; v[5] = c.y; v[6] = d.x; v[7] = d.y;
    } else {
        uint b16 = (row * 4u + (l >> 2u)) * 17u;
        float sc = unpackHalf2x16(u16K(b16)).x;
        uint qb = b16 + 1u + (l & 3u) * 4u;
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            uint w = u16K(qb + i);
            v[2u * i] = sc * float(int((w & 0xFFu) ^ 0x80u) - 128);
            v[2u * i + 1u] = sc * float(int(((w >> 8u) & 0xFFu) ^ 0x80u) - 128);
        }
    }
}

void load8V(uint row, uint l, out float v[8]) {
    if (KVQ8 == 0u) {
        uvec4 u = vv4[row * 16u + l];
        vec2 a = unpackHalf2x16(u.x), b = unpackHalf2x16(u.y), c = unpackHalf2x16(u.z), d = unpackHalf2x16(u.w);
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y; v[4] = c.x; v[5] = c.y; v[6] = d.x; v[7] = d.y;
    } else {
        uint b16 = (row * 4u + (l >> 2u)) * 17u;
        float sc = unpackHalf2x16(u16V(b16)).x;
        uint qb = b16 + 1u + (l & 3u) * 4u;
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            uint w = u16V(qb + i);
            v[2u * i] = sc * float(int((w & 0xFFu) ^ 0x80u) - 128);
            v[2u * i + 1u] = sc * float(int(((w >> 8u) & 0xFFu) ^ 0x80u) - 128);
        }
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint g = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    uint kg = tid >> 4u, l = tid & 15u;
    uint nh = pc.nh, nkv = pc.nkv;
    uint p = pc.pos0 + t;
    uint lo = pc.isLocal != 0u ? (p + 1u > pc.window ? p + 1u - pc.window : 0u) : 0u;
    const float scale = 0.08838834764831845;   // 1 / sqrt(128)

    float qv[3][8], rv[3], sk[3];
    [[unroll]] for (uint s = 0u; s < 3u; ++s) {
        uint h = g * 3u + s;
        [[unroll]] for (uint j = 0u; j < 8u; ++j) qv[s][j] = actToF(q[t * nh * 128u + h * 128u + l * 8u + j]);
        rv[s] = actToF(rel[t * nh * 16u + h * 16u + l]);
        sk[s] = actToF(sinkv[t * nh + h]);
    }

    float m[3], sum[3], acc[3][8];
    [[unroll]] for (uint s = 0u; s < 3u; ++s) {
        m[s] = -3.4e38; sum[s] = 0.0;
        [[unroll]] for (uint j = 0u; j < 8u; ++j) acc[s][j] = 0.0;
    }

    uint iters = (p - lo + 16u) / 16u;   // ceil((p - lo + 1) / 16)
    for (uint it = 0u; it < iters; ++it) {
        uint j = lo + kg + 16u * it;
        bool valid = j <= p;
        uint jj = valid ? j : p;
        uint slot = pc.ringCap != 0u ? jj % pc.ringCap : jj;
        uint row = slot * nkv + g;
        float k8[8], v8[8];
        load8K(row, l, k8);
        load8V(row, l, v8);
        uint d = p - jj;
        float rb = d < pc.extent ? relBias[d * 16u + l] : 0.0;
        [[unroll]] for (uint s = 0u; s < 3u; ++s) {
            float part = 0.0;
            [[unroll]] for (uint e = 0u; e < 8u; ++e) part = fma(qv[s][e], k8[e], part);
            part = part * scale + rb * rv[s];
            part += subgroupShuffleXor(part, 8u);
            part += subgroupShuffleXor(part, 4u);
            part += subgroupShuffleXor(part, 2u);
            part += subgroupShuffleXor(part, 1u);
            float score = part - sk[s];
            if (valid) {
                float nm = max(m[s], score);
                float corr = exp(m[s] - nm);
                float pe = exp(score - nm);
                sum[s] = sum[s] * corr + pe;
                [[unroll]] for (uint e = 0u; e < 8u; ++e) acc[s][e] = fma(pe, v8[e], acc[s][e] * corr);
                m[s] = nm;
            }
        }
    }

    [[unroll]] for (uint s = 0u; s < 3u; ++s) {
        if (l == 0u) { shM[kg * 3u + s] = m[s]; shS[kg * 3u + s] = sum[s]; }
        [[unroll]] for (uint e = 0u; e < 8u; ++e) shAcc[(kg * 3u + s) * 128u + l * 8u + e] = acc[s][e];
    }
    barrier();
    for (uint idx = tid; idx < 3u * 128u; idx += 256u) {
        uint s = idx >> 7u, dd = idx & 127u;
        float M = 0.0;                              // the zero sink logit joins the max
        for (uint k = 0u; k < 16u; ++k) M = max(M, shM[k * 3u + s]);
        float S = exp(-M), O = 0.0;
        for (uint k = 0u; k < 16u; ++k) {
            float w = exp(shM[k * 3u + s] - M);
            S = fma(shS[k * 3u + s], w, S);
            O = fma(shAcc[(k * 3u + s) * 128u + dd], w, O);
        }
        uint o = t * nh * 128u + (g * 3u + s) * 128u + dd;
        outp[o] = fToAct(O / S * sigmoidF(actToF(gate[o])));
    }
}
