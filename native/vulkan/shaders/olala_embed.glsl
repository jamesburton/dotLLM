// Olala token embedding gather + L2 normalisation (normalize_embeddings_ngpt, #1983): res[t, :] = unit(E[ids[t], :]).
// The table is read in place (Q8_0 or BF16), one 256-thread workgroup per token; the sum of squares is F32.
#include "olala_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufE   { uint table[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufIds { uint ids[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY   { ACT_T yout[]; };

layout(push_constant) uniform PushConstants {
    uint H;
    uint vocab;
} pc;

uint readU16(uint idx16) {
    uint w = table[idx16 >> 1u];
    return ((idx16 & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

float elem(uint id, uint k) {
#if defined(EMB_Q8_0)
    uint blk = k >> 5u, i = k & 31u;
    uint b16 = (id * (pc.H >> 5u) + blk) * 17u;
    float d = unpackHalf2x16(readU16(b16)).x;
    uint q = readU16(b16 + 1u + (i >> 1u));
    int v = int(((i & 1u) == 0u ? (q & 0xFFu) : (q >> 8u)) ^ 0x80u) - 128;
    return d * float(v);
#else
    uint w = table[(id * pc.H + k) >> 1u];
    return ((k & 1u) == 0u) ? bf16Lo(w) : bf16Hi(w);
#endif
}

shared float red[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    uint id = min(ids[t], pc.vocab - 1u);
    float ss = 0.0;
    for (uint k = tid; k < pc.H; k += 256u) { float v = elem(id, k); ss = fma(v, v, ss); }
    red[tid] = ss;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    float inv = 1.0 / max(sqrt(red[0]), 1e-12);
    for (uint k = tid; k < pc.H; k += 256u) yout[t * pc.H + k] = fToAct(elem(id, k) * inv);
}
