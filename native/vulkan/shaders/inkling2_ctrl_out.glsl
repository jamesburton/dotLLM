// Inkling 2 loop-controller combine + the norm that always follows it (issue #2034; Inkling2Ops.ControlCombine, graph::control):
//   o = (cur - tanh(forgetPre) * cur) + tanh(gainPre) * message;   y = o / sqrt(mean(o^2) + eps) * gamma
// The combined row stays in F32 shared memory between the combine and the norm (no extra rounding, no stored F32). H <= 4096.
// One 256-thread workgroup per token.
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufCur    { ACT_T cur[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufMsg    { ACT_T msg[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufGain   { ACT_T gainPre[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufForget { ACT_T forgetPre[]; };
layout(set = 0, binding = 4, std430) readonly  buffer BufGamma  { float gamma[]; };
layout(set = 0, binding = 5, std430) writeonly buffer BufOut    { ACT_T outp[]; };

layout(push_constant) uniform PushConstants {
    uint H;
    uint curOff;
    uint msgOff;
    float eps;
} pc;

shared float i2Row[4096];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    uint H = pc.H;
    uint cb = pc.curOff + t * H, mb = pc.msgOff + t * H, gb = t * H;
    float ss = 0.0;
    for (uint k = tid; k < H; k += 256u) {
        float c = actToF(cur[cb + k]);
        float f = tanhF(actToF(forgetPre[gb + k])), g = tanhF(actToF(gainPre[gb + k]));
        float o = (c - f * c) + g * actToF(msg[mb + k]);
        i2Row[k] = o;
        ss = fma(o, o, ss);
    }
    ss = wgSum(ss);
    float inv = 1.0 / sqrt(ss / float(H) + pc.eps);
    for (uint k = tid; k < H; k += 256u) outp[gb + k] = fToAct(i2Row[k] * inv * gamma[k]);
}
