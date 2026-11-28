// DSpark Markov head, one walk step (issue #2046). Computes tanh(c2) of step `step` (1-based) for markov_w2:
//   walk j = first..step (first = 1; a test mutant starts at `step`, i.e. resets the state each step):  s = pre[j-1][.] + J[.][0:R] . state + J[.][R:2R] . w1[prev_j]            (prev_1 = anchor, prev_j = draft j-1; token ids in `tok`)
//     gate = sigmoid(s[0:R]); cand = tanh(s[R:2R]); state = cand + gate * (state - cand); c2 = tanh(s[2R:3R])
//   out[0:R] = BF16(c2 of the LAST walked step)
// The F32 state lives ONLY in shared memory: each step re-derives it by replaying steps 1..step from the stored BF16 pre-activations and the stored token ids
// (K <= 7, so at most 7 x 3 x 512 MACs per thread). Nothing F32 crosses a dispatch boundary (user rule: the Markov accumulator is F32 in registers / shared only).
// One 256-thread workgroup, thread q owns state element q. J is BF16 [768][2R + 2H] read in place; w1 is BF16 [vocab][R].
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufJ  { uint16_t J[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufW1 { uint16_t w1[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufP  { ACT_T pre[]; };          // [rows][768] activation type
layout(set = 0, binding = 3, std430) readonly  buffer BufT  { uint tok[]; };           // [0] anchor, [i] draft i
layout(set = 0, binding = 4, std430) writeonly buffer BufO  { ACT_T outv[]; };         // [R] activation type

layout(push_constant) uniform PushConstants {
    uint H;
    uint R;
    uint step;
    uint vocab;
    uint first;
} pc;

shared float st[256];

void main() {
    uint q = gl_LocalInvocationID.x;
    uint stride = 2u * pc.R + 2u * pc.H;
    st[q] = 0.0;
    barrier();
    for (uint j = pc.first; j <= pc.step; ++j) {
        uint prev = min(tok[j - 1u], pc.vocab - 1u);
        uint wb = prev * pc.R;
        float s0 = actToF(pre[(j - 1u) * 768u + q]);
        float s1 = actToF(pre[(j - 1u) * 768u + 256u + q]);
        float s2 = actToF(pre[(j - 1u) * 768u + 512u + q]);
        uint r0 = q * stride, r1 = (256u + q) * stride, r2 = (512u + q) * stride;
        for (uint k = 0u; k < 256u; ++k) {
            float sv = st[k];
            float wv = bf16ToF(w1[wb + k]);
            s0 = fma(bf16ToF(J[r0 + k]), sv, fma(bf16ToF(J[r0 + 256u + k]), wv, s0));
            s1 = fma(bf16ToF(J[r1 + k]), sv, fma(bf16ToF(J[r1 + 256u + k]), wv, s1));
            s2 = fma(bf16ToF(J[r2 + k]), sv, fma(bf16ToF(J[r2 + 256u + k]), wv, s2));
        }
        float gate = sigmoidF(s0), cand = tanhF(s1);
        float old = st[q];
        barrier();
        st[q] = cand + gate * (old - cand);
        if (j == pc.step) outv[q] = fToAct(tanhF(s2));
        barrier();
    }
}
