// Olala geodesic residual update (OlalaOps.GeodesicUpdate, #1983), one 256-thread workgroup per token, in place on x:
//   g_perp = g - (x.g / |x|^2) x;  R = |x|;  t = min(|g_perp| / R, pi/4);  theta = min((t * scale + bias) / (layer + 1), pi/4)
//   x <- x cos(theta) + (g_perp / |g_perp|) R sin(theta)
// The CPU reduces in double because the mixer outputs carry outliers up to ~1e14; the GPU uses F32 shared-memory tree reductions (squares of
// 1e14 are 1e28, inside F32 range) and recomputes g - proj x from the stored values for the tangent norm, exactly as the CPU does.
#include "olala_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430)          buffer BufX { ACT_T x[]; };   // [T, H] residual, updated in place
layout(set = 0, binding = 1, std430) readonly buffer BufG { ACT_T g[]; };   // [T, H] mixer / MLP output

layout(push_constant) uniform PushConstants {
    uint H;
    uint layerPlus1;
    float scale;
    float bias;
} pc;

shared float red[256];

float wgSum(float v, uint tid) {
    red[tid] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint base = gl_WorkGroupID.x * pc.H;
    float xx = 0.0, xg = 0.0;
    for (uint k = tid; k < pc.H; k += 256u) {
        float xv = actToF(x[base + k]), gv = actToF(g[base + k]);
        xx = fma(xv, xv, xx);
        xg = fma(xv, gv, xg);
    }
    xx = wgSum(xx, tid);
    xg = wgSum(xg, tid);
    float xNormSq = max(xx, 1e-12);
    float proj = xg / xNormSq;
    float tt = 0.0;
    for (uint k = tid; k < pc.H; k += 256u) {
        float d = actToF(g[base + k]) - proj * actToF(x[base + k]);
        tt = fma(d, d, tt);
    }
    tt = wgSum(tt, tid);
    float safeTangent = max(sqrt(tt), 1e-8);
    float safeR = max(sqrt(xx), 1e-6);
    const float clampA = 0.7853981633974483;
    float theta = min(safeTangent / safeR, clampA);
    theta = min((theta * pc.scale + pc.bias) / float(pc.layerPlus1), clampA);
    float c = cos(theta), s = sin(theta);
    float tangentCoeff = safeR * s / safeTangent;
    for (uint k = tid; k < pc.H; k += 256u) {
        float xv = actToF(x[base + k]);
        float d = actToF(g[base + k]) - proj * xv;
        x[base + k] = fToAct(xv * c + d * tangentCoeff);
    }
}
