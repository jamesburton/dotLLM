// Olala Mamba-3 MIMO mixer (OlalaOps.Mamba3MimoStep, #1983): T tokens in order, one 128-thread workgroup per head, F32 recurrent state in place.
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

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
const uint NT = 128u;   // threads: lane p = tid & 63 (P <= 64), state half = tid >> 6

// #2001: the F32 state column of lane p lives in REGISTERS for the whole chunk (loaded once, stored once) instead of being read-modify-written in
// global memory on every token. N and R are specialization constants so the n / r loops unroll (no dynamically indexed private arrays).
// Requires P <= 64 (one lane per p; checked by the loader). The source arithmetic and its order are unchanged, but the result is NOT bit-identical
// to the per-token global form (measured, #2002/#2003 review: F32-act control vs the old shader KL <= 6e-6 on the real capital/code cases, quality
// vs HF unchanged; most likely different FMA contraction once the loops unroll, not attributed). At N = 128, R = 4 the driver uses all 256 VGPRs
// and spills 128-144 B per lane to scratch (VK_AMD_shader_info, gfx1151), so part of the column still round-trips memory; N = 16 does not spill.
// #2005: to stop the spill each state column is split across TWO lanes (128-thread workgroup): lane p of half 0 owns n in [0, N/2), half 1 owns
// [N/2, N); each accumulates its own partial o[r] and half 0 adds half 1's (through shared memory). N must be even.
layout(constant_id = 0) const uint N_C = 128u;
layout(constant_id = 1) const uint R_C = 4u;

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
shared float rn[16 * NT];                         // partial sums of squares: 2R rows x NT threads
shared float so[8 * 64];                          // half 1's partial o[r] per lane (R <= 8)
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
    uint halfN = N / 2u;

    const uint NH = N_C / 2u;
    uint lane = tid & 63u, hf = tid >> 6u;
    uint n0 = hf * NH;
    float sreg[NH];
    // Previous token's raw x (the beta-term input) is carried in a register by BOTH halves: a per-token global read of st[stV] by half 1 would
    // race half 0's write (barrier() orders workgroup memory only, not buffer memory). Loaded once, written back once.
    float vPrev = 0.0;
    if (lane < P) {
        [[unroll]] for (uint n = 0u; n < NH; ++n) sreg[n] = st[stSsm + (h * N + n0 + n) * P + lane];
        vPrev = st[stV + h * P + lane];
    }

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
            for (uint n = tid; n < N; n += NT) { float v = actToF(dyn[db + row * N + n]); ss = fma(v, v, ss); }
            rn[row * NT + tid] = ss;
        }
        barrier();
        for (uint s = NT / 2u; s > 0u; s >>= 1u) {
            if (tid < s)
                for (uint row = 0u; row < 2u * R; ++row) rn[row * NT + tid] += rn[row * NT + tid + s];
            barrier();
        }
        if (tid < 2u * R) rinv[tid] = 1.0 / sqrt(rn[tid * NT] / float(N) + pc.eps);
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
        for (uint i = tid; i < R * N; i += NT) {
            uint r = i / N, n = i - r * N;
            sq[i] = actToF(dyn[db + R * N + i]) * rinv[R + r] * prm[offCn + n] + prm[offCb + (h * R + r) * N + n];
            sk[i] = actToF(dyn[db + i]) * rinv[r] * prm[offBn + n] + prm[offBb + (h * R + r) * N + n];
            skp[i] = st[stK + (r * H + h) * N + n];
        }
        barrier();
        // Rotate pairs (i, i + N/2) for i < A.
        for (uint i = tid; i < R * A; i += NT) {
            uint r = i / A, a = i - r * A;
            uint i0 = r * N + a, i1 = r * N + halfN + a;
            float c = scos[a], s = ssin[a];
            float q0 = sq[i0], q1 = sq[i1], k0 = sk[i0], k1 = sk[i1];
            sq[i0] = q0 * c - q1 * s; sq[i1] = q0 * s + q1 * c;
            sk[i0] = k0 * c - k1 * s; sk[i1] = k0 * s + k1 * c;
        }
        barrier();

        float dSkip = prm[offD + h];
        float xp = 0.0, zp = 0.0;
        float xv[R_C], xs[R_C], o[R_C];
        uint p = lane;
        if (p < P) {
            xp = actToF(zxdt[zb + P + p]); zp = actToF(zxdt[zb + p]);
            [[unroll]] for (uint r = 0u; r < R_C; ++r) {
                float mx = prm[offMx + (h * R + r) * P + p];
                xv[r] = xp * mx;
                xs[r] = vPrev * mx;
                o[r] = 0.0;
            }
            [[unroll]] for (uint n = 0u; n < NH; ++n) {
                float s = sreg[n] * alpha;
                [[unroll]] for (uint r = 0u; r < R_C; ++r)
                    s += xv[r] * gamma * sk[r * N_C + n0 + n] + xs[r] * beta * skp[r * N_C + n0 + n];
                sreg[n] = s;
                [[unroll]] for (uint r = 0u; r < R_C; ++r) o[r] += s * sq[r * N_C + n0 + n];
            }
            if (hf == 1u) {
                [[unroll]] for (uint r = 0u; r < R_C; ++r) so[r * 64u + p] = o[r];
            }
        }
        barrier();
        if (p < P && hf == 0u) {
            float acc = 0.0;
            for (uint r = 0u; r < R_C; ++r) {
                float mz = prm[offMz + (h * R + r) * P + p];
                float mo = prm[offMo + (h * R + r) * P + p];
                acc += ((o[r] + so[r * 64u + p]) + xv[r] * dSkip) * siluF(zp * mz) * mo;
            }
            yout[t * H * P + h * P + p] = fToAct(acc);
        }
        vPrev = xp;   // this token's raw x feeds the next token's beta-term (both halves; 0 on idle lanes, never stored)
        // This token's rotated K becomes the next token's "previous" (every lane finished reading skp above).
        barrier();
        for (uint i = tid; i < R * N; i += NT) {
            uint r = i / N, n = i - r * N;
            st[stK + (r * H + h) * N + n] = sk[i];
        }
        barrier();
    }
    if (lane < P) {
        [[unroll]] for (uint n = 0u; n < NH; ++n) st[stSsm + (h * N + n0 + n) * P + lane] = sreg[n];
        if (hf == 0u) st[stV + h * P + lane] = vPrev;
    }
}
