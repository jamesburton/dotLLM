// Olala GEMV / small-batch GEMM with weights read in place and BF16 activations (#1983):
//   y[col, m] = epilogue( sum_k W[m, k] * x[col, k] ),   x: [T, K] BF16 (or F32 in the reference arm), y: [T, M].
// W is a raw Q8_0 / Q6_K / BF16 blob (selected by the including .comp). F32 accumulation; the activation is NEVER quantized (no FP16-scaled
// Q8_1 path: the unbounded-input tensors of safe2 would overflow), so this kernel is valid for every Olala projection.
// Dispatch: one 64-lane workgroup per ROWS output rows (ROWS = 1 except Q8_0; x + 32768 y, folded for > 32768 groups), z = column group of NCOL columns.
// Specialization: 0 NCOL, 1 EPI (0 none, 1 relu^2, 2 add aux[col, m]), 2 OUTF32 (store F32 to yf instead of the activation type: logits),
//                 3 ROWS (Q8_0 only, #2006).
#include "olala_common.glsl"

#define WG_SIZE 64u
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint NCOL = 1u;
layout(constant_id = 1) const uint EPI = 0u;
layout(constant_id = 2) const uint OUTF32 = 0u;
#if defined(WT_Q8_0)
layout(constant_id = 3) const uint ROWS = 1u;   // output rows per workgroup (Q8_0 only; the dispatch is ceil(M / ROWS) workgroups)
#define WDOT_MULTIROW 1
#endif

layout(set = 0, binding = 0, std430) readonly  buffer BufW   { uint weight[]; };
// The activation is read through 16-byte vectors (#2006): 8 BF16 (or 8 F32 as two vec4) per load instead of eight 16-bit loads.
// Requires xOff % 8 == 0 (the host checks it). Same elements, same order, same F32 FMAs: bit-identical to the scalar loads.
#ifdef ACT_F32
layout(set = 0, binding = 1, std430) readonly  buffer BufX   { vec4 xin4[]; };
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufX   { uvec4 xin4[]; };
#endif
layout(set = 0, binding = 2, std430) writeonly buffer BufY   { ACT_T yout[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufAux { ACT_T aux[]; };
layout(set = 0, binding = 4, std430) writeonly buffer BufYf  { float yf[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint T;   // columns (tokens)
    uint xOff; // element offset of column 0 inside x (the LM head reads the last row of the residual)
} pc;

#ifdef ACT_F32
float xAt(uint col, uint k) { uint i = pc.xOff + col * pc.K + k; return xin4[i >> 2u][i & 3u]; }
// x[col, k0 .. k0 + 7] (k0 % 8 == 0 and xOff % 8 == 0) as two vec4.
void xLoad8(uint col, uint k0, out vec4 a, out vec4 b) {
    uint i = (pc.xOff + col * pc.K + k0) >> 2u;
    a = xin4[i]; b = xin4[i + 1u];
}
#else
float xAt(uint col, uint k) {
    uint i = pc.xOff + col * pc.K + k;
    uint w = xin4[i >> 3u][(i >> 1u) & 3u];
    return ((i & 1u) == 0u) ? actWordLo(w) : actWordHi(w);
}
void xLoad8(uint col, uint k0, out vec4 a, out vec4 b) {
    uvec4 u = xin4[(pc.xOff + col * pc.K + k0) >> 3u];
    a = vec4(actWordLo(u.x), actWordHi(u.x), actWordLo(u.y), actWordHi(u.y));
    b = vec4(actWordLo(u.z), actWordHi(u.z), actWordLo(u.w), actWordHi(u.w));
}
#endif
#define X_LOAD8 1

#include "olala_wdot.glsl"

void main() {
#ifdef WDOT_MULTIROW
    uint m0 = (gl_WorkGroupID.y * 32768u + gl_WorkGroupID.x) * ROWS;
    if (m0 >= pc.M) return;
#else
    uint m0 = gl_WorkGroupID.y * 32768u + gl_WorkGroupID.x;
    if (m0 >= pc.M) return;
#endif
    uint lane = gl_LocalInvocationID.x;
    uint c0 = gl_WorkGroupID.z * NCOL;
    [[unroll]] for (uint i = 0u; i < NACC; ++i) gAcc[i] = 0.0;
#ifdef WDOT_MULTIROW
    rowDotRows(m0, pc.M, pc.K, lane, c0, pc.T - 1u);
#else
    rowDot(m0, pc.K, lane, c0, pc.T - 1u);
#endif
    reduceAcc(lane);
    if (lane == 0u) {
#ifdef WDOT_MULTIROW
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
            uint m = m0 + r;
            if (m >= pc.M) break;
#else
        {
            const uint r = 0u; uint m = m0;
#endif
            [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
                uint col = c0 + cc;
                if (col < pc.T) {
                    float v = gTot[r * NCOL + cc];
                    if (EPI == 1u) { float rl = max(v, 0.0); v = rl * rl; }
                    else if (EPI == 2u) v += actToF(aux[col * pc.M + m]);
                    if (OUTF32 == 1u) yf[col * pc.M + m] = v;
                    else yout[col * pc.M + m] = fToAct(v);
                }
            }
        }
    }
}
