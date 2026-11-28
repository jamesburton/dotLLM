// DSpark block attention (issue #2046; CPU Inkling2DSparkDrafter.Attend, llama.cpp block_uniform mask): the b block rows attend NON-causally over the same keys:
// the context keys at positions [lo, pos0) (F16 cache, a ring of ringCap slots or linear) and the block's own b keys (F16 block scratch). Scale 1/sqrt(128).
// One 256-thread workgroup per KV head; its 3 query heads x b rows (<= 24 query vectors) share every key/value read. Online softmax over tiles of 64 keys:
// the running max / sum, the score tile and the output accumulators are F32 in shared memory / registers; nothing F32 is stored. q is BF16, K/V are F16,
// the output is BF16 [b][heads * 128].
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ  { ACT_T q[]; };          // [b][heads * 128] activation type
layout(set = 0, binding = 1, std430) readonly  buffer BufKc { uint kctx[]; };         // [slots][kv * 128] F16 pairs
layout(set = 0, binding = 2, std430) readonly  buffer BufVc { uint vctx[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufKb { uint kblk[]; };         // [b][kv * 128] F16 pairs
layout(set = 0, binding = 4, std430) readonly  buffer BufVb { uint vblk[]; };
layout(set = 0, binding = 5, std430) writeonly buffer BufO  { ACT_T outp[]; };       // [b][heads * 128] activation type

layout(push_constant) uniform PushConstants {
    uint b;
    uint lo;
    uint pos0;
    uint ringCap;
    uint heads;
    uint kv;
    float scale;
} pc;

shared float qs[24][128];
shared float sc[24][64];
shared float mS[24], lS[24], aS[24];

// Word index of element (key j, kv head kvh, dim 0) in the F16-pair arrays.
uint keyBase(uint j, uint kvh, uint nCtx, out bool blk) {
    blk = j >= nCtx;
    uint row = blk ? j - nCtx : (pc.ringCap == 0u ? pc.lo + j : (pc.lo + j) % pc.ringCap);
    return (row * pc.kv + kvh) * 64u;
}

void main() {
    uint t = gl_LocalInvocationID.x, kvh = gl_WorkGroupID.x;
    uint nq = pc.b * 3u;
    uint nCtx = pc.pos0 - pc.lo, nKeys = nCtx + pc.b;
    for (uint i = t; i < nq * 128u; i += 256u) {
        uint u = i / 128u, dd = i % 128u, row = u / 3u, g = u % 3u;
        qs[u][dd] = actToF(q[(row * pc.heads + kvh * 3u + g) * 128u + dd]);
    }
    if (t < 24u) { mS[t] = -3.0e38; lS[t] = 0.0; }
    barrier();

    uint d = t & 127u, half_ = t >> 7;
    uint ub = half_ * 12u;
    float o[12];
    for (uint i = 0u; i < 12u; ++i) o[i] = 0.0;

    for (uint base = 0u; base < nKeys; base += 64u) {
        // scores: thread -> key (t & 63), query vectors [ (t >> 6) * 6, +6 )
        uint kk = t & 63u, qg = t >> 6;
        uint j = base + kk;
        float acc[6];
        for (uint i = 0u; i < 6u; ++i) acc[i] = 0.0;
        if (j < nKeys) {
            bool blk;
            uint kb = keyBase(j, kvh, nCtx, blk);
            for (uint w = 0u; w < 64u; ++w) {
                uint word = blk ? kblk[kb + w] : kctx[kb + w];
                vec2 kvv = unpackHalf2x16(word);
                for (uint i = 0u; i < 6u; ++i) {
                    uint u = qg * 6u + i;
                    acc[i] = fma(qs[u][2u * w], kvv.x, fma(qs[u][2u * w + 1u], kvv.y, acc[i]));
                }
            }
        }
        for (uint i = 0u; i < 6u; ++i) sc[qg * 6u + i][kk] = j < nKeys ? acc[i] * pc.scale : -3.0e38;
        barrier();
        // online softmax: one leader per query vector
        if (t < nq) {
            float mt = -3.0e38;
            for (uint k = 0u; k < 64u; ++k) mt = max(mt, sc[t][k]);
            float mn = max(mS[t], mt);
            float al = exp(mS[t] - mn);
            float ls = 0.0;
            for (uint k = 0u; k < 64u; ++k) { float p = base + k < nKeys ? exp(sc[t][k] - mn) : 0.0; sc[t][k] = p; ls += p; }
            lS[t] = lS[t] * al + ls; mS[t] = mn; aS[t] = al;
        }
        barrier();
        // accumulate: thread -> dim d, query vectors [ub, ub + 12)
        for (uint i = 0u; i < 12u; ++i) o[i] *= (ub + i < nq) ? aS[ub + i] : 1.0;
        for (uint k = 0u; k < 64u; ++k) {
            uint jk = base + k;
            if (jk >= nKeys) break;
            bool blk;
            uint vb = keyBase(jk, kvh, nCtx, blk);
            uint word = blk ? vblk[vb + (d >> 1)] : vctx[vb + (d >> 1)];
            vec2 vv = unpackHalf2x16(word);
            float v = (d & 1u) == 0u ? vv.x : vv.y;
            for (uint i = 0u; i < 12u; ++i) if (ub + i < nq) o[i] = fma(sc[ub + i][k], v, o[i]);
        }
        barrier();
    }
    for (uint i = 0u; i < 12u; ++i) {
        uint u = ub + i;
        if (u >= nq) break;
        uint row = u / 3u, g = u % 3u;
        outp[(row * pc.heads + kvh * 3u + g) * 128u + d] = fToAct(o[i] / lS[u]);
    }
}
