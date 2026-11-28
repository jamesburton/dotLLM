// Body of the rounding probe (see inkling2_actround_f16act.comp).
#include "olala_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer BufIn { uint inBits[]; };
layout(set = 0, binding = 1, std430) writeonly buffer BufOut { ACT_T outp[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufRt { float rt[]; };
layout(push_constant) uniform PushConstants { uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    ACT_T h = fToAct(uintBitsToFloat(inBits[i]));
    outp[i] = h;
    rt[i] = actToF(h);
}
