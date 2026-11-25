// Olala GEMV / small-batch GEMM with weights read in place and BF16 activations (#1983):
//   y[col, m] = epilogue( sum_k W[m, k] * x[col, k] ),   x: [T, K] BF16 (or F32 in the reference arm), y: [T, M].
// W is a raw Q8_0 / Q6_K / BF16 blob (selected by the including .comp). F32 accumulation; the activation is NEVER quantized (no FP16-scaled
// Q8_1 path: the unbounded-input tensors of safe2 would overflow), so this kernel is valid for every Olala projection.
// Dispatch: one 64-lane workgroup per output row (x + 32768 y, folded for M > 65535), z = column group of NCOL columns.
// Specialization: 0 NCOL, 1 EPI (0 none, 1 relu^2, 2 add aux[col, m]), 2 OUTF32 (store F32 to yf instead of the activation type: logits).
#include "olala_common.glsl"

#define WG_SIZE 64u
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint NCOL = 1u;
layout(constant_id = 1) const uint EPI = 0u;
layout(constant_id = 2) const uint OUTF32 = 0u;

layout(set = 0, binding = 0, std430) readonly  buffer BufW   { uint weight[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufX   { ACT_T xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY   { ACT_T yout[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufAux { ACT_T aux[]; };
layout(set = 0, binding = 4, std430) writeonly buffer BufYf  { float yf[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint T;   // columns (tokens)
    uint xOff; // element offset of column 0 inside x (the LM head reads the last row of the residual)
} pc;

float xAt(uint col, uint k) { return actToF(xin[pc.xOff + col * pc.K + k]); }

#include "olala_wdot.glsl"

void main() {
    uint m = gl_WorkGroupID.y * 32768u + gl_WorkGroupID.x;
    if (m >= pc.M) return;
    uint lane = gl_LocalInvocationID.x;
    uint c0 = gl_WorkGroupID.z * NCOL;
    [[unroll]] for (uint i = 0u; i < NCOL; ++i) gAcc[i] = 0.0;
    rowDot(m, pc.K, lane, c0, pc.T - 1u);
    reduceAcc(lane);
    if (lane == 0u) {
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
            uint col = c0 + cc;
            if (col < pc.T) {
                float v = gTot[cc];
                if (EPI == 1u) { float r = max(v, 0.0); v = r * r; }
                else if (EPI == 2u) v += actToF(aux[col * pc.M + m]);
                if (OUTF32 == 1u) yf[col * pc.M + m] = v;
                else yout[col * pc.M + m] = fToAct(v);
            }
        }
    }
}
