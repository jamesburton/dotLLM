// Inkling 2 vision tower (issue #2062, part B): y = (x - mean) / sqrt(var + eps) * gamma + beta per 1024-wide row (ggml_norm + gain + bias), BF16 in / out, F32 statistics
// (two-pass: the variance is taken about the F32 mean), one 256-thread workgroup per row.
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer BufX { ACT_T xin[]; };
layout(set = 0, binding = 1, std430) readonly buffer BufG { float gamma[]; };
layout(set = 0, binding = 2, std430) readonly buffer BufB { float beta[]; };
layout(set = 0, binding = 3, std430) writeonly buffer BufY { ACT_T yout[]; };

layout(push_constant) uniform PushConstants { uint rows; float eps; } pc;

void main() {
    uint r = gl_WorkGroupID.x, tid = gl_LocalInvocationID.x;
    if (r >= pc.rows) return;
    float v[4];
    float s = 0.0;
    for (uint i = 0u; i < 4u; ++i) { v[i] = actToF(xin[r * 1024u + tid + i * 256u]); s += v[i]; }
    float mean = wgSum(s) * (1.0 / 1024.0);
    float q = 0.0;
    for (uint i = 0u; i < 4u; ++i) { float d = v[i] - mean; q = fma(d, d, q); }
    float inv = 1.0 / sqrt(wgSum(q) * (1.0 / 1024.0) + pc.eps);
    for (uint i = 0u; i < 4u; ++i) {
        uint c = tid + i * 256u;
        yout[r * 1024u + c] = fToAct((v[i] - mean) * inv * gamma[c] + beta[c]);
    }
}
