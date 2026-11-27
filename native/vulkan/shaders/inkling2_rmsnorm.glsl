// Inkling 2 row RMS norm (issue #2034): y[t, :] = x[t, :] / sqrt(mean(x^2) + eps) * gamma (gamma optional). One 256-thread workgroup per row.
// inOff / outOff are element offsets of row 0 (the depth-mix outputs and the block stack are slices of larger buffers).
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufX { ACT_T xin[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufG { float gamma[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY { ACT_T yout[]; };

layout(push_constant) uniform PushConstants {
    uint H;
    uint inOff;
    uint outOff;
    uint hasGamma;
    float eps;
} pc;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t = gl_WorkGroupID.x;
    uint ib = pc.inOff + t * pc.H, ob = pc.outOff + t * pc.H;
    float ss = 0.0;
    for (uint k = tid; k < pc.H; k += 256u) { float v = actToF(xin[ib + k]); ss = fma(v, v, ss); }
    ss = wgSum(ss);
    float inv = 1.0 / sqrt(ss / float(pc.H) + pc.eps);
    for (uint k = tid; k < pc.H; k += 256u) {
        float y = actToF(xin[ib + k]) * inv;
        if (pc.hasGamma != 0u) y *= gamma[k];
        yout[ob + k] = fToAct(y);
    }
}
