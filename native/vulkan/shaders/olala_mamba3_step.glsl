// Olala Mamba-3 MIMO mixer (OlalaOps.Mamba3MimoStep, #1983): T tokens in order, one 64-thread workgroup per head, F32 recurrent state in place.
//
//   alpha = exp(A dt), beta = (1 - trap) dt alpha, gamma = trap dt   (A = min(-softplus(a), -1e-4), dt = softplus(dt_raw + dt_bias))
//   h_t = alpha h_{t-1} + beta u_{t-1} + gamma u_t,   u_t = sum_r k_{t,r} (x_t * MimoX_r)^T,   o_r = h_t q_r + D (x_t * MimoX_r)
//   y = sum_r MimoO_r * o_r * silu(z * MimoZ_r)
// with q = rot(RMSNorm(C) + C_bias), k = rot(RMSNorm(B) + B_bias), rotated by the cumulative data-dependent angle of the head.
//
// Inputs are the ssm_in (per head interleaved [z(P), x(P), dt, A, trap]) and ssm_in_dyn ([B(R*N), C(R*N), raw angles]) projections in the
// activation type. The recurrent state is ALWAYS F32 (OlalaStatePolicy): per layer one F32 buffer
//   [ ssm: H*N*P (state-major so lanes over p are coalesced) | angle: H*A | kState: R*H*N | vState: H*P ].
#include "olala_common.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufZ   { ACT_T zxdt[]; };   // [T, H*(2P+3)]
layout(set = 0, binding = 1, std430) readonly  buffer BufDyn { ACT_T dyn[]; };    // [T, 2*R*N + A]
layout(set = 0, binding = 2, std430) readonly  buffer BufPrm { float prm[]; };    // learned constants (layout below)
layout(set = 0, binding = 3, std430)           buffer BufSt  { float st[]; };     // recurrent state, F32
layout(set = 0, binding = 4, std430) writeonly buffer BufY   { ACT_T yout[]; };   // [T, H*P]

layout(push_constant) uniform PushConstants {
    uint T;
    uint H;
    uint P;
    uint N;
    uint R;
    uint A;
    float eps;
} pc;

// prm layout: bnorm[N] cnorm[N] bbias[H*R*N] cbias[H*R*N] dtbias[H] D[H] mimoX[H*R*P] mimoZ[H*R*P] mimoO[H*R*P]

const uint MAXRN = 1024u;   // R * N upper bound (R = 4, N = 128)
shared float sk[MAXRN], sq[MAXRN], skp[MAXRN];   // rotated K, rotated Q, previous token's rotated K
shared float rn[16 * 64];                         // partial sums of squares: 2R rows x 64 lanes
shared float rinv[16];
shared float scos[64], ssin[64];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint h = gl_WorkGroupID.x;
    uint H = pc.H, P = pc.P, N = pc.N, R = pc.R, A = pc.A;
    uint ph = 2u * P + 3u;
    uint dynW = 2u * R * N + A;
    uint offBn = 0u, offCn = N, offBb = 2u * N, offCb = offBb + H * R * N, offDt = offCb + H * R * N, offD = offDt + H,
         offMx = offD + H, offMz = offMx + H * R * P, offMo = offMz + H * R * P;
    uint stSsm = 0u, stAng = H * N * P, stK = stAng + H * A, stV = stK + R * H * N;
    uint half_ = N / 2u;

    for (uint t = 0u; t < pc.T; ++t) {
        uint zb = t * H * ph + h * ph;
        uint db = t * dynW;
        float dtRaw = actToF(zxdt[zb + 2u * P]), aRaw = actToF(zxdt[zb + 2u * P + 1u]), trapRaw = actToF(zxdt[zb + 2u * P + 2u]);
        float decayRate = min(-softplusF(aRaw), -1e-4);
        float dt = softplusF(dtRaw + prm[offDt + h]);
        float trap = sigmoidF(trapRaw);
        float alpha = exp(decayRate * dt);
        float beta = (1.0 - trap) * dt * alpha;
        float gamma = trap * dt;

        // B / C RMSNorm statistics (2R rows of N), shared by every head.
        for (uint row = 0u; row < 2u * R; ++row) {
            float ss = 0.0;
            for (uint n = tid; n < N; n += 64u) { float v = actToF(dyn[db + row * N + n]); ss = fma(v, v, ss); }
            rn[row * 64u + tid] = ss;
        }
        barrier();
        for (uint s = 32u; s > 0u; s >>= 1u) {
            if (tid < s)
                for (uint row = 0u; row < 2u * R; ++row) rn[row * 64u + tid] += rn[row * 64u + tid + s];
            barrier();
        }
        if (tid < 2u * R) rinv[tid] = 1.0 / sqrt(rn[tid * 64u] / float(N) + pc.eps);
        // Angle state: cumulative phase wrapped to [0, 2pi).
        if (tid < A) {
            float raw = actToF(dyn[db + 2u * R * N + tid]);
            float v = st[stAng + h * A + tid] + tanh(raw) * dt * 3.14159265358979;
            const float TWO_PI = 6.28318530717959;
            v -= floor(v / TWO_PI) * TWO_PI;
            st[stAng + h * A + tid] = v;
            scos[tid] = cos(v);
            ssin[tid] = sin(v);
        }
        barrier();
        // q = C_norm + C_bias, k = B_norm + B_bias (per rank), previous K copied out of the state.
        for (uint i = tid; i < R * N; i += 64u) {
            uint r = i / N, n = i - r * N;
            sq[i] = actToF(dyn[db + R * N + i]) * rinv[R + r] * prm[offCn + n] + prm[offCb + (h * R + r) * N + n];
            sk[i] = actToF(dyn[db + i]) * rinv[r] * prm[offBn + n] + prm[offBb + (h * R + r) * N + n];
            skp[i] = st[stK + (r * H + h) * N + n];
        }
        barrier();
        // Rotate pairs (i, i + N/2) for i < A.
        for (uint i = tid; i < R * A; i += 64u) {
            uint r = i / A, a = i - r * A;
            uint i0 = r * N + a, i1 = r * N + half_ + a;
            float c = scos[a], s = ssin[a];
            float q0 = sq[i0], q1 = sq[i1], k0 = sk[i0], k1 = sk[i1];
            sq[i0] = q0 * c - q1 * s; sq[i1] = q0 * s + q1 * c;
            sk[i0] = k0 * c - k1 * s; sk[i1] = k0 * s + k1 * c;
        }
        barrier();

        float dSkip = prm[offD + h];
        for (uint p = tid; p < P; p += 64u) {
            float xp = actToF(zxdt[zb + P + p]), zp = actToF(zxdt[zb + p]);
            float vPrev = st[stV + h * P + p];
            float xv[8], xs[8], o[8];   // R <= 8
            for (uint r = 0u; r < R; ++r) {
                float mx = prm[offMx + (h * R + r) * P + p];
                xv[r] = xp * mx;
                xs[r] = vPrev * mx;
                o[r] = 0.0;
            }
            for (uint n = 0u; n < N; ++n) {
                uint si = stSsm + (h * N + n) * P + p;
                float s = st[si] * alpha;
                for (uint r = 0u; r < R; ++r)
                    s += xv[r] * gamma * sk[r * N + n] + xs[r] * beta * skp[r * N + n];
                st[si] = s;
                for (uint r = 0u; r < R; ++r) o[r] += s * sq[r * N + n];
            }
            float acc = 0.0;
            for (uint r = 0u; r < R; ++r) {
                float mz = prm[offMz + (h * R + r) * P + p];
                float mo = prm[offMo + (h * R + r) * P + p];
                acc += (o[r] + xv[r] * dSkip) * siluF(zp * mz) * mo;
            }
            yout[t * H * P + h * P + p] = fToAct(acc);
            st[stV + h * P + p] = xp;   // this token's raw x feeds the next token's beta-term
        }
        // This token's rotated K becomes the next token's "previous" (every lane finished reading skp above).
        barrier();
        for (uint i = tid; i < R * N; i += 64u) {
            uint r = i / N, n = i - r * N;
            st[stK + (r * H + h) * N + n] = sk[i];
        }
        barrier();
    }
}
