// Inkling 2 attention input preparation + KV-cache store (issue #2034; Inkling2Ops.ClipL2 / RmsNorm / LogScale, llama.cpp graph::clip).
// One 128-thread workgroup per (row, token); gl_WorkGroupID.x selects the row:
//   0 .. nh-1            Q head:   x *= q_clip[h, :];  x = x * t / max(|x|, t)  (t = qk_clip); global layers: x *= tau
//   nh .. nh+nkv-1       K head:   x = x * t / max(|x|, t)                       -> KV cache (K)
//   nh+nkv .. nh+2nkv-1  V head:   x = x / sqrt(mean(x^2) + eps)  (no gain)     -> KV cache (V)
//   nh+2nkv .. +nh-1     R head:   x *= rel_clip[h, :];  clip(rel_clip_t); global layers: x *= tau
// tau = 1 + alpha * ln(max(1, (pos + 1) / floor)). q and rel are updated in place. K / V are stored as F16 (KVQ8 = 0) or q8_0 (KVQ8 = 1: 4 blocks of
// [fp16 scale, 32 int8] per 128-element head row, quantised like ggml quantize_row_q8_0_ref) at cache position `pos` (local layers: pos % ringCap).
// Cache layout [position][kv head][head_dim] (position-major: growth appends).
#include "inkling2_common.glsl"

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint KVQ8 = 0u;

layout(set = 0, binding = 0, std430)          buffer BufQ    { ACT_T q[]; };         // [T, nh*128]
layout(set = 0, binding = 1, std430)          buffer BufRel  { ACT_T rel[]; };       // [T, nh*relLen]
layout(set = 0, binding = 2, std430) readonly buffer BufK    { ACT_T kin[]; };       // [T, nkv*128]
layout(set = 0, binding = 3, std430) readonly buffer BufV    { ACT_T vin[]; };
layout(set = 0, binding = 4, std430) readonly buffer BufQc   { float qclip[]; };     // [nh][128]
layout(set = 0, binding = 5, std430) readonly buffer BufRc   { float relclip[]; };   // [nh][relLen]
layout(set = 0, binding = 6, std430) writeonly buffer BufKc  { uint16_t kc[]; };
layout(set = 0, binding = 7, std430) writeonly buffer BufVc  { uint16_t vc[]; };

layout(push_constant) uniform PushConstants {
    uint T;
    uint pos0;
    uint ringCap;     // 0 = linear (global) cache
    uint isGlobal;
    float qkClip;
    float relClip;
    float eps;
    float logFloor;
    float logAlpha;
    uint nh;
    uint nkv;
    uint relLen;
} pc;

shared float i2Abs[128];
shared int i2Q[128];

uint f16bits(float x) { return packHalf2x16(vec2(x, 0.0)) & 0xFFFFu; }

// Stores one 128-element head row (thread d owns element d) into `cache` at row index `row` (= slot * nkv + head).
void storeRow(float x, uint d, uint row, bool isK) {
    if (KVQ8 == 0u) {
        if (isK) kc[row * 128u + d] = uint16_t(f16bits(x)); else vc[row * 128u + d] = uint16_t(f16bits(x));
    } else {
        i2Abs[d] = abs(x);
        barrier();
        uint blk = d >> 5u;
        float amax = 0.0;
        for (uint i = 0u; i < 32u; ++i) amax = max(amax, i2Abs[blk * 32u + i]);
        float dsc = amax / 127.0;
        float id = dsc != 0.0 ? 1.0 / dsc : 0.0;
        float v = x * id;
        i2Q[d] = int(sign(v) * floor(abs(v) + 0.5));
        barrier();
        uint b16 = (row * 4u + blk) * 17u;
        if ((d & 31u) == 0u) { uint16_t sc = uint16_t(f16bits(dsc)); if (isK) kc[b16] = sc; else vc[b16] = sc; }
        if ((d & 1u) == 0u) {
            uint w = (uint(i2Q[d]) & 0xFFu) | ((uint(i2Q[d + 1u]) & 0xFFu) << 8u);
            uint wi = b16 + 1u + ((d & 31u) >> 1u);
            if (isK) kc[wi] = uint16_t(w); else vc[wi] = uint16_t(w);
        }
        barrier();
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint id = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    uint nh = pc.nh, nkv = pc.nkv;
    uint pos = pc.pos0 + t;
    float tau = 1.0;
    if (pc.isGlobal != 0u) tau = pc.logAlpha * log(max(1.0, (float(pos) + 1.0) * (1.0 / pc.logFloor))) + 1.0;

    if (id < nh) {
        uint i = t * nh * 128u + id * 128u + tid;
        float x = actToF(q[i]) * qclip[id * 128u + tid];
        float ss = wgSum(x * x);
        float s = 1.0 / max(sqrt(ss), pc.qkClip);
        x = x * s * pc.qkClip;
        if (pc.isGlobal != 0u) x *= tau;
        q[i] = fToAct(x);
    } else if (id < nh + nkv) {
        uint g = id - nh;
        float x = actToF(kin[t * nkv * 128u + g * 128u + tid]);
        float ss = wgSum(x * x);
        float s = 1.0 / max(sqrt(ss), pc.qkClip);
        x = x * s * pc.qkClip;
        uint slot = pc.ringCap != 0u ? pos % pc.ringCap : pos;
        storeRow(x, tid, slot * nkv + g, true);
    } else if (id < nh + 2u * nkv) {
        uint g = id - nh - nkv;
        float x = actToF(vin[t * nkv * 128u + g * 128u + tid]);
        float ss = wgSum(x * x);
        float inv = 1.0 / sqrt(ss / 128.0 + pc.eps);
        uint slot = pc.ringCap != 0u ? pos % pc.ringCap : pos;
        storeRow(x * inv, tid, slot * nkv + g, false);
    } else {
        uint h = id - nh - 2u * nkv;
        float x = 0.0;
        uint i = t * nh * pc.relLen + h * pc.relLen + tid;
        if (tid < pc.relLen) x = actToF(rel[i]) * relclip[h * pc.relLen + tid];
        float ss = wgSum(x * x);
        float s = 1.0 / max(sqrt(ss), pc.relClip);
        x = x * s * pc.relClip;
        if (pc.isGlobal != 0u) x *= tau;
        if (tid < pc.relLen) rel[i] = fToAct(x);
    }
}
