// DSpark Markov head, vocabulary pass of one step (issue #2046): for every vocab row v
//   col[v] = softcap( base[v] + markov_w2[v] . c2 + markov_w2.bias[v] ),  softcap(x) = cap * tanh(x / cap)   (AFTER the bias; llama.cpp softcap_after_bias)
// written back over `base` (the row's softcapped logits, F32 like the target's logits) and reduced to the argmax (lowest index on ties): each workgroup handles 32 vocab rows
// (8 rows at a time, 32 lanes per row reading a contiguous 512 B of markov_w2 = [vocab][R] BF16 in place) and writes one (value, index) partial.
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#ifdef LOGITS_F16
layout(set = 0, binding = 0, std430)          buffer BufBase { uint16_t base16[]; };    // the drafter logits, [rows][vocab], F16 bits (#2046)
#else
layout(set = 0, binding = 0, std430)          buffer BufBase { float base[]; };         // the drafter logits, [rows][vocab]
#endif
layout(set = 0, binding = 1, std430) readonly  buffer BufW2   { uint w2w[]; };           // [vocab][R] BF16 pairs
layout(set = 0, binding = 2, std430) readonly  buffer BufB2   { float b2[]; };           // [vocab]
layout(set = 0, binding = 3, std430) readonly  buffer BufC    { uint16_t c2[]; };        // [R] BF16
layout(set = 0, binding = 4, std430) writeonly buffer BufPart { vec2 part[]; };          // [wgs] (value, bit-cast index)

layout(push_constant) uniform PushConstants {
    uint vocab;
    uint baseOff;
    uint hasBias;
    float cap;
} pc;

shared float red[256];
shared float bestV[8];
shared uint  bestI[8];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u, rg = t >> 5;
    float cv[8];
    for (uint i = 0u; i < 8u; ++i) cv[i] = bf16ToF(c2[lane * 8u + i]);
    float wgBestV = -3.0e38; uint wgBestI = 0xFFFFFFFFu;
    for (uint it = 0u; it < 4u; ++it) {
        uint v = gl_WorkGroupID.x * 32u + it * 8u + rg;
        float acc = 0.0;
        if (v < pc.vocab) {
            uint wb = v * 128u + lane * 4u;         // R / 2 = 128 words per row, 4 words (8 BF16) per lane
            for (uint w = 0u; w < 4u; ++w) {
                uint word = w2w[wb + w];
                acc = fma(bf16Lo(word), cv[2u * w], fma(bf16Hi(word), cv[2u * w + 1u], acc));
            }
        }
        red[t] = acc;
        barrier();
        for (uint s = 16u; s >= 1u; s >>= 1u) {
            if (lane < s) red[t] += red[t + s];
            barrier();
        }
        if (lane == 0u && v < pc.vocab) {
#ifdef LOGITS_F16
            float x = unpackHalf2x16(uint(base16[pc.baseOff + v])).x + red[t] + (pc.hasBias != 0u ? b2[v] : 0.0);
#else
            float x = base[pc.baseOff + v] + red[t] + (pc.hasBias != 0u ? b2[v] : 0.0);
#endif
            if (pc.cap > 0.0) x = pc.cap * tanhF(x / pc.cap);
#ifdef LOGITS_F16
            base16[pc.baseOff + v] = uint16_t(packHalf2x16(vec2(x, 0.0)) & 0xFFFFu);
#else
            base[pc.baseOff + v] = x;
#endif
            if (x > wgBestV || (x == wgBestV && v < wgBestI)) { wgBestV = x; wgBestI = v; }
        }
        barrier();
    }
    if (lane == 0u) { bestV[rg] = wgBestV; bestI[rg] = wgBestI; }
    barrier();
    if (t == 0u) {
        float bv = bestV[0]; uint bi = bestI[0];
        for (uint i = 1u; i < 8u; ++i) if (bestV[i] > bv || (bestV[i] == bv && bestI[i] < bi)) { bv = bestV[i]; bi = bestI[i]; }
        part[gl_WorkGroupID.x] = vec2(bv, uintBitsToFloat(bi));
    }
}
