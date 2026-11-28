// DSpark Markov head, the state-independent part of joint_proj (issue #2046; llama.cpp build_dspark_markov_head, is_rnn): for every drafted row r (hidden_r) and output j
//   pre[r][j] = joint_bias[j] + sum_k J[j][2R + k] * hidden_r[k] + sum_k J[j][2R + H + k] * frnn[k]     (J = joint_proj [768][2R + 2H] BF16, R = rank 256)
// i.e. everything except the columns that multiply the recurrent state and markov_w1[prev], which depend on the walk. One workgroup per output j; BF16 weights are read in place,
// F32 accumulation in registers; the stored result is BF16 [rows][768]. f_rnn is the F16 ring row of position pos0 - 1 (frnnSlot).
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufJ  { uint16_t J[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufH  { ACT_T hid[]; };         // [rows][H] activation type
layout(set = 0, binding = 2, std430) readonly  buffer BufF  { uint16_t frnn[]; };     // [16][H] F16 ring
layout(set = 0, binding = 3, std430) readonly  buffer BufB  { float jbias[]; };       // [768]
layout(set = 0, binding = 4, std430) writeonly buffer BufP  { ACT_T pre[]; };         // [rows][768] activation type

layout(push_constant) uniform PushConstants {
    uint H;
    uint R;
    uint rows;
    uint frnnSlot;
} pc;

float halfBitsToF(uint16_t v) { return unpackHalf2x16(uint(v)).x; }

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint j = gl_WorkGroupID.x;
    uint stride = 2u * pc.R + 2u * pc.H;
    uint hb = j * stride + 2u * pc.R, fb = hb + pc.H;
    float fa = 0.0;
    float acc[8];
    for (uint r = 0u; r < 8u; ++r) acc[r] = 0.0;
    for (uint k = tid; k < pc.H; k += 256u) {
        float wh = bf16ToF(J[hb + k]);
        for (uint r = 0u; r < 8u; ++r) if (r < pc.rows) acc[r] = fma(wh, actToF(hid[r * pc.H + k]), acc[r]);
        fa = fma(bf16ToF(J[fb + k]), halfBitsToF(frnn[pc.frnnSlot * pc.H + k]), fa);
    }
    fa = wgSum(fa);
    for (uint r = 0u; r < 8u; ++r) {
        float s = wgSum(r < pc.rows ? acc[r] : 0.0);
        if (tid == 0u && r < pc.rows) pre[r * 768u + j] = fToAct(s + fa + jbias[j]);
    }
}
