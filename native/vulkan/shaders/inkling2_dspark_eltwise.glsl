// DSpark elementwise (issue #2046), BF16 in / BF16 out, F32 arithmetic, one rounding on the store.
//   MODE 0: a = a + b (residual add)        MODE 1: a = silu(a) * b (SwiGLU)
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint MODE = 0u;

layout(set = 0, binding = 0, std430) buffer BufA { ACT_T a[]; };
layout(set = 0, binding = 1, std430) readonly buffer BufB { ACT_T b[]; };

layout(push_constant) uniform PushConstants { uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    float x = actToF(a[i]), y = actToF(b[i]);
    a[i] = fToAct(MODE == 0u ? x + y : siluF(x) * y);
}
