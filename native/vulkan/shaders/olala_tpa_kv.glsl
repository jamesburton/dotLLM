// Olala differential-TPA K/V construction (#1983): token shift, RMS-normed K, BF16 KV-cache store. One workgroup per (kv head, token).
//   rawK[h, d] = sum_r ak[t, h, r] * bk[t, r, d] / rank        (same for V with av / bv)
//   alpha = pos == 0 ? 0 : sigmoid(shift[t, h]);  K~ = alpha * prevRawK + (1 - alpha) * rawK;  V~ likewise
//   K cache <- RMSNorm_d(K~) * gamma_k,  V cache <- V~                (BF16 always, even in the F32-activation reference arm)
// prevRaw is the previous token's RAW (pre-shift, pre-norm) K/V: recomputed from the projections of token t-1 inside the batch, or read from the
// F32 carry state for the first token of the batch. mode 1 only stores the batch's last raw K/V into that F32 carry (dispatched after mode 0 so
// no workgroup of mode 0 can read it mid-update).
#include "olala_common.glsl"

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufAk   { ACT_T ak[]; };    // [T, nk*rank]
layout(set = 0, binding = 1, std430) readonly  buffer BufAv   { ACT_T av[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufBk   { ACT_T bk[]; };    // [T, rank*hd]
layout(set = 0, binding = 3, std430) readonly  buffer BufBv   { ACT_T bv[]; };
layout(set = 0, binding = 4, std430) readonly  buffer BufSk   { ACT_T sk[]; };    // [T, nk]
layout(set = 0, binding = 5, std430) readonly  buffer BufSv   { ACT_T sv[]; };
layout(set = 0, binding = 6, std430) readonly  buffer BufGk   { float gammaK[]; };// [hd]
layout(set = 0, binding = 7, std430)           buffer BufLast { float lastKV[]; };// [2, nk*hd] F32 carry (K then V)
layout(set = 0, binding = 8, std430) writeonly buffer BufKc   { uint16_t kc[]; }; // [pos, nk, hd] BF16
layout(set = 0, binding = 9, std430) writeonly buffer BufVc   { uint16_t vc[]; };

layout(push_constant) uniform PushConstants {
    uint T;
    uint base;    // position of token 0 of this batch
    uint nk;
    uint hd;
    uint rank;
    uint mode;
    float eps;
} pc;

shared float red[128];

float rawK(uint t, uint h, uint d) {
    float a = 0.0;
    for (uint r = 0u; r < pc.rank; ++r) a += actToF(ak[t * pc.nk * pc.rank + h * pc.rank + r]) * actToF(bk[t * pc.rank * pc.hd + r * pc.hd + d]);
    return a * (1.0 / float(pc.rank));
}
float rawV(uint t, uint h, uint d) {
    float a = 0.0;
    for (uint r = 0u; r < pc.rank; ++r) a += actToF(av[t * pc.nk * pc.rank + h * pc.rank + r]) * actToF(bv[t * pc.rank * pc.hd + r * pc.hd + d]);
    return a * (1.0 / float(pc.rank));
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint h = gl_WorkGroupID.x;
    uint t = (pc.mode == 1u) ? pc.T - 1u : gl_WorkGroupID.y;
    uint pos = pc.base + t;
    uint hd = pc.hd, nk = pc.nk;

    if (pc.mode == 1u) {
        for (uint d = tid; d < hd; d += 128u) {
            lastKV[h * hd + d] = rawK(t, h, d);
            lastKV[nk * hd + h * hd + d] = rawV(t, h, d);
        }
        return;
    }

    float alK = (pos == 0u) ? 0.0 : sigmoidF(actToF(sk[t * nk + h]));
    float alV = (pos == 0u) ? 0.0 : sigmoidF(actToF(sv[t * nk + h]));
    float ssq = 0.0;
    float kk[4];   // hd <= 512
    for (uint i = 0u; i < 4u; ++i) {
        uint d = tid + i * 128u;
        kk[i] = 0.0;
        if (d >= hd) continue;
        float rk = rawK(t, h, d), rv = rawV(t, h, d);
        float pk = 0.0, pv = 0.0;
        if (pos != 0u) {
            if (t > 0u) { pk = rawK(t - 1u, h, d); pv = rawV(t - 1u, h, d); }
            else { pk = lastKV[h * hd + d]; pv = lastKV[nk * hd + h * hd + d]; }
        }
        float k2 = alK * pk + (1.0 - alK) * rk;
        float v2 = alV * pv + (1.0 - alV) * rv;
        kk[i] = k2;
        ssq = fma(k2, k2, ssq);
        vc[(pos * nk + h) * hd + d] = fToBf16(v2);
    }
    red[tid] = ssq;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    float inv = 1.0 / sqrt(red[0] / float(hd) + pc.eps);
    for (uint i = 0u; i < 4u; ++i) {
        uint d = tid + i * 128u;
        if (d < hd) kc[(pos * nk + h) * hd + d] = fToBf16(kk[i] * inv * gammaK[d]);
    }
}
