// DSpark context mix (issue #2046; llama.cpp build_inject_ctx_mix, CPU Inkling2DSparkDrafter.Inject): per row
//   m = sum_k mix[k] * tap[r][k]          (mix: the converter's already-softmaxed F32 vector; taps BF16 [rows][4][H])
//   y = m / sqrt(mean(m^2) + eps) * gamma
// MODE 0: y stored as BF16 [rows][H] (the input of attn_k_ctx / attn_v_ctx).  MODE 1: y stored as F16 into the f_rnn ring, slot (posBase + r) % 16.
// Accumulation and the sum of squares are F32 in registers; the only stored value is the single rounded output.
#extension GL_GOOGLE_include_directive : require
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint MODE = 0u;

layout(set = 0, binding = 0, std430) readonly  buffer BufTap { ACT_T tap[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufMix { float mixw[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufG   { float gamma[]; };
layout(set = 0, binding = 3, std430) writeonly buffer BufY   { uint16_t yout[]; };

layout(push_constant) uniform PushConstants {
    uint H;
    uint posBase;
    float eps;
} pc;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint r = gl_WorkGroupID.x;
    // MODE 1: rows r and r + 16 share a ring slot; only the LAST row of each slot is stored (same result as writing in order, but race-free).
    if (MODE == 1u && r + 16u < gl_NumWorkGroups.x) return;
    uint tb = r * 4u * pc.H;
    float m0 = mixw[0], m1 = mixw[1], m2 = mixw[2], m3 = mixw[3];
    float ss = 0.0;
    for (uint k = tid; k < pc.H; k += 256u) {
        float v = m0 * actToF(tap[tb + k]) + m1 * actToF(tap[tb + pc.H + k]) + m2 * actToF(tap[tb + 2u * pc.H + k]) + m3 * actToF(tap[tb + 3u * pc.H + k]);
        ss = fma(v, v, ss);
    }
    ss = wgSum(ss);
    float inv = 1.0 / sqrt(ss / float(pc.H) + pc.eps);
    uint ob = MODE == 0u ? r * pc.H : ((pc.posBase + r) % 16u) * pc.H;
    for (uint k = tid; k < pc.H; k += 256u) {
        float v = m0 * actToF(tap[tb + k]) + m1 * actToF(tap[tb + pc.H + k]) + m2 * actToF(tap[tb + 2u * pc.H + k]) + m3 * actToF(tap[tb + 3u * pc.H + k]);
        float y = v * inv * gamma[k];
        yout[ob + k] = MODE == 0u ? fToAct(y) : uint16_t(packHalf2x16(vec2(y, 0.0)) & 0xFFFFu);
    }
}
