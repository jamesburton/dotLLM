// Olala differential tensor-product attention core (#1983): one 128-thread workgroup per (kv group, token).
//   q~[h] = RMSNorm_d(q[h]) * gamma_q * scaler[h] * log(min(pos + 1, window))         (scalable softmax; no RoPE)
//   s_j = softcap * tanh( q~ . K[j] / hd / softcap ),  j in [max(0, pos - window), pos];  softmax over the window, out[h] = sum_j p_j V[j]
//   y[g*(group-1) + s] = (out[g*group + s] - sigmoid(lambda[g]) * out[g*group + group-1]) * silu(gate + gateBias)
// The KV cache is BF16 (read in place); scores/softmax run in F32 registers/shared memory with an online (flash-style) softmax over blocks of
// 128 keys, so no score tensor is ever stored in global memory. group = 4 (3 signal heads + 1 noise head per kv head).
#include "olala_common.glsl"

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ    { ACT_T q[]; };         // [T, nq*hd]
layout(set = 0, binding = 1, std430) readonly  buffer BufKc   { uvec4 kc4[]; };       // [pos, nk, hd] BF16, 8 per uvec4
layout(set = 0, binding = 2, std430) readonly  buffer BufVc   { uint16_t vc[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufLam  { ACT_T lam[]; };       // [T, nk]
layout(set = 0, binding = 4, std430) readonly  buffer BufGate { ACT_T gate[]; };      // [T, nk*(group-1)*hd]
layout(set = 0, binding = 5, std430) readonly  buffer BufGq   { float gammaQ[]; };    // [hd]
layout(set = 0, binding = 6, std430) readonly  buffer BufSc   { float scaler[]; };    // [nq]
layout(set = 0, binding = 7, std430) writeonly buffer BufY    { ACT_T yout[]; };      // [T, nk*(group-1)*hd]

layout(push_constant) uniform PushConstants {
    uint T;
    uint base;
    uint nk;
    uint hd;
    uint group;     // 4
    uint window;
    float softcap;  // 0 = off
    float gateBias;
    float eps;
} pc;

shared float sq[4][512];     // normalised, scaled query per head of the group (hd <= 512)
shared float sc[4][128];     // block scores / probabilities
shared float red[4][128];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint g = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    uint pos = pc.base + t;
    uint hd = pc.hd, nk = pc.nk, group = pc.group;
    uint nq = nk * group;
    uint qW = nq * hd;

    float logPos = log(min(float(pos + 1u), float(pc.window)));
    // Query RMSNorm per head of the group.
    for (uint hh = 0u; hh < group; ++hh) {
        float ss = 0.0;
        for (uint d = tid; d < hd; d += 128u) { float v = actToF(q[t * qW + (g * group + hh) * hd + d]); ss = fma(v, v, ss); }
        red[hh][tid] = ss;
    }
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (tid < s) for (uint hh = 0u; hh < group; ++hh) red[hh][tid] += red[hh][tid + s];
        barrier();
    }
    for (uint hh = 0u; hh < group; ++hh) {
        float inv = 1.0 / sqrt(red[hh][0] / float(hd) + pc.eps);
        float mult = scaler[g * group + hh] * logPos;
        for (uint d = tid; d < hd; d += 128u)
            sq[hh][d] = actToF(q[t * qW + (g * group + hh) * hd + d]) * inv * gammaQ[d] * mult;
    }
    barrier();

    uint lo = (pos > pc.window) ? pos - pc.window : 0u;
    uint nKeys = pos - lo + 1u;
    float m[4] = float[4](-3.4e38, -3.4e38, -3.4e38, -3.4e38);
    float l[4] = float[4](0.0, 0.0, 0.0, 0.0);
    float acc[4] = float[4](0.0, 0.0, 0.0, 0.0);
    float invHd = 1.0 / float(hd);

    for (uint b0 = 0u; b0 < nKeys; b0 += 128u) {
        uint j = lo + b0 + tid;
        bool valid = (b0 + tid) < nKeys;
        float s[4] = float[4](0.0, 0.0, 0.0, 0.0);
        if (valid) {
            uint row4 = ((j * nk + g) * hd) >> 3u;
            for (uint c = 0u; c < (hd >> 3u); ++c) {
                uvec4 w = kc4[row4 + c];
                float kv[8] = float[8](bf16Lo(w.x), bf16Hi(w.x), bf16Lo(w.y), bf16Hi(w.y), bf16Lo(w.z), bf16Hi(w.z), bf16Lo(w.w), bf16Hi(w.w));
                for (uint hh = 0u; hh < group; ++hh)
                    for (uint e = 0u; e < 8u; ++e) s[hh] = fma(kv[e], sq[hh][c * 8u + e], s[hh]);
            }
        }
        for (uint hh = 0u; hh < group; ++hh) {
            float v = s[hh] * invHd;
            if (pc.softcap > 0.0) v = tanh(v / pc.softcap) * pc.softcap;
            sc[hh][tid] = valid ? v : -3.4e38;
        }
        barrier();
        // Every thread folds the block redundantly (cheap, and avoids a reduction tree): block max, rescale, accumulate p * V.
        uint nb = min(128u, nKeys - b0);
        for (uint hh = 0u; hh < group; ++hh) {
            float bm = -3.4e38;
            for (uint i = 0u; i < nb; ++i) bm = max(bm, sc[hh][i]);
            float mNew = max(m[hh], bm);
            float corr = exp(m[hh] - mNew);
            l[hh] *= corr; acc[hh] *= corr;
            m[hh] = mNew;
        }
        for (uint i = 0u; i < nb; ++i) {
            uint jj = lo + b0 + i;
            for (uint hh = 0u; hh < group; ++hh) {
                float p = exp(sc[hh][i] - m[hh]);
                l[hh] += p;
                if (tid < hd) acc[hh] = fma(p, bf16ToF(vc[(jj * nk + g) * hd + tid]), acc[hh]);
            }
        }
        barrier();
    }

    // Differential combine + silu gate.
    uint snr = group - 1u;
    if (tid < hd) {
        float outv[4];
        for (uint hh = 0u; hh < group; ++hh) outv[hh] = acc[hh] / l[hh];
        float lambda = sigmoidF(actToF(lam[t * nk + g]));
        uint gateW = nk * snr * hd;
        for (uint s2 = 0u; s2 < snr; ++s2) {
            uint oi = (g * snr + s2) * hd + tid;
            float combined = outv[s2] - lambda * outv[snr];
            yout[t * gateW + oi] = fToAct(combined * siluF(actToF(gate[t * gateW + oi]) + pc.gateBias));
        }
    }
}
