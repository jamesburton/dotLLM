// Inkling 2 short causal convolution with carried state (issue #2034; Inkling2ConvState.Run, llama.cpp graph::convolve + ggml_ssm_conv), K = 4 taps.
// One thread per channel walks the chunk's tokens with a sliding window of the last 3 inputs (BF16 state, oldest first, [channel][3]):
//   acc = sum_i win[i] * w[c][i]            (tap 0 = the OLDEST input; win[3] = the current input)
//   DYN: acc += sum_lag win[3 - lag] * dyn[t][head(c)][lag]   (lag 0 = the current input; head(c) = c / (C / heads))
//   y = (x_t + acc) / sqrt((DYN ? 2 : 1) * 4 + 1)  [+ add[t, c] when ADD: the residual of the layer, added before the single BF16 rounding]
// The window state is read at entry and written at exit by the same thread, so the dispatch has no cross-thread hazard; the input may not alias the output.
#include "inkling2_common.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint DYN = 0u;
layout(constant_id = 1) const uint ADD = 0u;
layout(constant_id = 2) const uint SNAP = 0u;   // 1: also write the verify window [old state (3 rows); the T inputs] as BF16 [3 + T][C] at snapOff (row snapshots, #2046)

layout(set = 0, binding = 0, std430) readonly  buffer BufX   { ACT_T xin[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufW   { float w[]; };        // [C][4]
layout(set = 0, binding = 2, std430) readonly  buffer BufDyn { ACT_T dyn[]; };      // [T][heads][4]
layout(set = 0, binding = 3, std430)           buffer BufSt  { uint16_t st[]; };    // [C][3] BF16
layout(set = 0, binding = 4, std430) readonly  buffer BufAdd { ACT_T addv[]; };     // [T][C]
layout(set = 0, binding = 5, std430) writeonly buffer BufY   { ACT_T yout[]; };
layout(set = 0, binding = 6, std430) writeonly buffer BufSnap { uint16_t snap[]; };   // [3 + T][C] BF16, only written when SNAP == 1

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
    uint heads;
    uint xOff;
    uint addOff;
    uint snapOff;
} pc;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= pc.C) return;
    float s0 = bf16ToF(st[c * 3u]), s1 = bf16ToF(st[c * 3u + 1u]), s2 = bf16ToF(st[c * 3u + 2u]);
    float w0 = w[c * 4u], w1 = w[c * 4u + 1u], w2 = w[c * 4u + 2u], w3 = w[c * 4u + 3u];
    uint perHead = DYN == 1u ? pc.C / pc.heads : 1u;
    float scale = 1.0 / sqrt(float((DYN == 1u ? 2 : 1) * 4 + 1));
    if (SNAP == 1u) { snap[pc.snapOff + c] = st[c * 3u]; snap[pc.snapOff + pc.C + c] = st[c * 3u + 1u]; snap[pc.snapOff + 2u * pc.C + c] = st[c * 3u + 2u]; }
    for (uint t = 0u; t < pc.T; ++t) {
        float x = actToF(xin[pc.xOff + t * pc.C + c]);
        if (SNAP == 1u) snap[pc.snapOff + (3u + t) * pc.C + c] = fToBf16(x);
        float acc = 0.0;
        acc += s0 * w0; acc += s1 * w1; acc += s2 * w2; acc += x * w3;
        if (DYN == 1u) {
            uint d = t * pc.heads * 4u + (c / perHead) * 4u;
            acc += x * actToF(dyn[d]);
            acc += s2 * actToF(dyn[d + 1u]);
            acc += s1 * actToF(dyn[d + 2u]);
            acc += s0 * actToF(dyn[d + 3u]);
        }
        float y = (x + acc) * scale;
        if (ADD == 1u) y += actToF(addv[pc.addOff + t * pc.C + c]);
        yout[t * pc.C + c] = fToAct(y);
        s0 = s1; s1 = s2; s2 = x;
    }
    st[c * 3u] = fToBf16(s0); st[c * 3u + 1u] = fToBf16(s1); st[c * 3u + 2u] = fToBf16(s2);
}
