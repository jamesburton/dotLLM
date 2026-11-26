// Inkling 2 loop-controller gate input (issue #2034; llama.cpp graph::control): out[t, :] = [ rms(cur[t]) ; rms(msg[t]) ] (gamma-free, eps 1e-6),
// the [2H] input of the rank-64 `ctrl_down` projection. One 256-thread workgroup per token. msgOff selects the controller's message rows
// inside the [4][T][H] message stack.
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufCur { ACT_T cur[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufMsg { ACT_T msg[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufOut { ACT_T outp[]; };

layout(push_constant) uniform PushConstants {
    uint H;
    uint curOff;
    uint msgOff;
    float eps;
} pc;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    uint H = pc.H;
    uint cb = pc.curOff + t * H, mb = pc.msgOff + t * H, ob = t * 2u * H;
    float sc = 0.0, sm = 0.0;
    for (uint k = tid; k < H; k += 256u) {
        float c = actToF(cur[cb + k]), m = actToF(msg[mb + k]);
        sc = fma(c, c, sc); sm = fma(m, m, sm);
    }
    sc = wgSum(sc);
    sm = wgSum(sm);
    float ic = 1.0 / sqrt(sc / float(H) + pc.eps), im = 1.0 / sqrt(sm / float(H) + pc.eps);
    for (uint k = tid; k < H; k += 256u) {
        outp[ob + k] = fToAct(actToF(cur[cb + k]) * ic);
        outp[ob + H + k] = fToAct(actToF(msg[mb + k]) * im);
    }
}
